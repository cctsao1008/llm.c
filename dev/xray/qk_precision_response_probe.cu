#pragma push_macro("main")
#undef main
#define main xray_qk_bilinearity_embedded_main
#include "qk_bilinearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>

static void xray_attention_prefix_math(float* qkvr, float* preatt, float* att,
                                       float* blended_qkv,
                                       int B, int T, int C, int NH,
                                       cublasMath_t math_mode) {
    const int block_size = 256;
    const int softmax_block_size = 256;
    const int HS = C / NH;

    float* q = qkvr + 0 * (size_t)B * T * C;
    float* k = qkvr + 1 * (size_t)B * T * C;
    float* v = qkvr + 2 * (size_t)B * T * C;
    (void)v;

    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, blended_qkv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    cublasCheck(cublasSetMathMode(cublas_handle, math_mode));
    const float one = 1.0f;
    const float zero = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, HS,
        &one,
        k, HS, T * HS,
        q, HS, T * HS,
        &zero,
        preatt, T, T * T,
        B * NH));

    float scale = 1.0f / sqrtf((float)HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());
}

static const char* math_name(cublasMath_t mode) {
    return mode == CUBLAS_PEDANTIC_MATH ? "PEDANTIC" : "TF32";
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    const int layer = 2;
    if (B <= 0 || T <= 0) return 2;

    cudaCheck(cudaSetDevice(0));
    cublasCheck(cublasCreate(&cublas_handle));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    // Keep the natural source perturbation identical to prior probes.
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward(&model, loader.inputs, loader.targets, B, T);
    cudaCheck(cudaDeviceSynchronize());

    int C = model.config.channels;
    int NH = model.config.num_heads;
    size_t BTC = (size_t)B * T * C;
    size_t QKVN = 3 * BTC;
    size_t BHTT = (size_t)B * NH * T * T;

    ParameterTensors p = model.params;
    ActivationTensors a = model.acts;
    float* residual = a.residual3 + (size_t)(layer - 1) * BTC;
    float* ln1 = a.ln1 + (size_t)layer * BTC;
    float* ln1_mean = a.ln1_mean + (size_t)layer * B * T;
    float* ln1_rstd = a.ln1_rstd + (size_t)layer * B * T;
    float* ln1w = p.ln1w + layer * C;
    float* ln1b = p.ln1b + layer * C;
    float* qkvw = p.qkvw + (size_t)layer * 3 * C * C;
    float* qkvb = p.qkvb + (size_t)layer * 3 * C;

    layernorm_forward(ln1, ln1_mean, ln1_rstd, residual, ln1w, ln1b, B, T, C);

    float *d_custom=nullptr, *d_tf32=nullptr, *d_blend=nullptr;
    float *d_qkvr=nullptr, *d_preatt=nullptr, *d_att=nullptr;
    cudaCheck(cudaMalloc((void**)&d_custom, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_tf32, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_blend, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_qkvr, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_preatt, BHTT * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_att, BHTT * sizeof(float)));

    matmul_forward(d_custom, ln1, qkvw, qkvb, B, T, C, 3 * C);
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    xray_matmul_cublas(d_tf32, ln1, qkvw, qkvb, B, T, C, 3 * C);
    cudaCheck(cudaDeviceSynchronize());

    const int block = 256;
    const int grid = (int)((QKVN + block - 1) / block);
    const int modes[] = {XRAY_Q_ONLY, XRAY_K_ONLY};
    const float alphas[] = {0.125f, 0.25f, 0.5f, 1.0f};
    const cublasMath_t math_modes[] = {CUBLAS_TF32_TENSOR_OP_MATH, CUBLAS_PEDANTIC_MATH};

    printf("[xray][qk-precision] L02 natural QKV delta; Q-only/K-only are affine in exact arithmetic\n");
    printf("[xray][qk-precision] compare internal Q*K^T cuBLAS math mode: TF32 vs PEDANTIC; softmax unchanged\n");

    for (cublasMath_t mm : math_modes) {
        QKStage preatt{"preatt-QK", {}, {}, {}};
        QKStage att{"att-softmax", {}, {}, {}};

        cudaCheck(cudaMemcpy(d_blend, d_custom, QKVN * sizeof(float), cudaMemcpyDeviceToDevice));
        xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_blend, B, T, C, NH, mm);
        cudaCheck(cudaDeviceSynchronize());
        xray_copy(preatt.baseline, d_preatt, BHTT);
        xray_copy(att.baseline, d_att, BHTT);

        for (int mode : modes) {
            for (float alpha : alphas) {
                xray_selective_qkv_blend<<<grid, block>>>(d_blend, d_custom, d_tf32, QKVN, C, +alpha, mode);
                cudaCheck(cudaGetLastError());
                xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_blend, B, T, C, NH, mm);
                cudaCheck(cudaDeviceSynchronize());
                xray_copy(preatt.plus, d_preatt, BHTT);
                xray_copy(att.plus, d_att, BHTT);

                xray_selective_qkv_blend<<<grid, block>>>(d_blend, d_custom, d_tf32, QKVN, C, -alpha, mode);
                cudaCheck(cudaGetLastError());
                xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_blend, B, T, C, NH, mm);
                cudaCheck(cudaDeviceSynchronize());
                xray_copy(preatt.minus, d_preatt, BHTT);
                xray_copy(att.minus, d_att, BHTT);

                LocalSymmetryStats ps = local_symmetry_stats(preatt.baseline, preatt.plus, preatt.minus, alpha, nullptr);
                LocalSymmetryStats as = local_symmetry_stats(att.baseline, att.plus, att.minus, alpha, nullptr);
                printf("[xray][qk-precision] math=%-8s mode=%-6s alpha=%5.3f stage=preatt-QK   odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                       math_name(mm), mode_name(mode), alpha, ps.odd_rel_l2, ps.even_rel_l2, ps.curvature_ratio);
                printf("[xray][qk-precision] math=%-8s mode=%-6s alpha=%5.3f stage=att-softmax odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                       math_name(mm), mode_name(mode), alpha, as.odd_rel_l2, as.even_rel_l2, as.curvature_ratio);
            }
        }
    }

    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_blend));
    cudaCheck(cudaFree(d_tf32));
    cudaCheck(cudaFree(d_custom));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
