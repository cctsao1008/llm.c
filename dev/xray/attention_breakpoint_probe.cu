#pragma push_macro("main")
#undef main
#define main xray_local_response_embedded_main
#include "local_response_symmetry_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>

static void xray_copy(std::vector<float>& dst, const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void xray_attention_prefix(float* qkvr, float* preatt, float* att,
                                  float* blended_qkv,
                                  int B, int T, int C, int NH) {
    const int block_size = 256;
    const int softmax_block_size = 256;
    const int HS = C / NH;

    float* q = qkvr + 0 * (size_t)B * T * C;
    float* k = qkvr + 1 * (size_t)B * T * C;
    float* v = qkvr + 2 * (size_t)B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, blended_qkv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

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

struct BreakpointStage {
    const char* name;
    std::vector<float> baseline;
    std::vector<float> plus;
    std::vector<float> minus;
};

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    const int layer = 2;
    if (B <= 0 || T <= 0) return 2;

    cudaCheck(cudaSetDevice(0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    // Run the untouched graph once. The L1 residual is the exact fixed input to L2.
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

    const int blend_block = 256;
    const int blend_grid = (int)((QKVN + blend_block - 1) / blend_block);

    std::vector<BreakpointStage> stages = {
        {"qkvr", {}, {}, {}},
        {"preatt-QK", {}, {}, {}},
        {"att-softmax", {}, {}, {}}
    };

    auto run_and_capture = [&](float alpha, int side) {
        xray_blend_kernel<<<blend_grid, blend_block>>>(d_blend, d_custom, d_tf32, QKVN, alpha);
        cudaCheck(cudaGetLastError());
        xray_attention_prefix(d_qkvr, d_preatt, d_att, d_blend, B, T, C, NH);
        cudaCheck(cudaDeviceSynchronize());

        std::vector<float>* qdst;
        std::vector<float>* pdst;
        std::vector<float>* adst;
        if (side == 0) {
            qdst=&stages[0].baseline; pdst=&stages[1].baseline; adst=&stages[2].baseline;
        } else if (side > 0) {
            qdst=&stages[0].plus; pdst=&stages[1].plus; adst=&stages[2].plus;
        } else {
            qdst=&stages[0].minus; pdst=&stages[1].minus; adst=&stages[2].minus;
        }
        xray_copy(*qdst, d_qkvr, QKVN);
        xray_copy(*pdst, d_preatt, BHTT);
        xray_copy(*adst, d_att, BHTT);
    };

    run_and_capture(0.0f, 0);

    const float alphas[] = {0.125f, 0.25f, 0.5f, 1.0f};
    printf("[xray][attention-breakpoint] source=qkv L02 natural TF32-custom direction; split QK score GEMM from softmax\n");
    printf("[xray][attention-breakpoint] qkvr=post-permute; preatt-QK=raw Q*K^T scores before scale/softmax; att-softmax=post scale+causal softmax\n");

    for (float alpha : alphas) {
        run_and_capture(+alpha, +1);
        run_and_capture(-alpha, -1);
        for (const auto& stage : stages) {
            LocalSymmetryStats s = local_symmetry_stats(stage.baseline, stage.plus, stage.minus, alpha, nullptr);
            printf("[xray][attention-breakpoint] alpha=%5.3f stage=%-11s plus_rel=%10.6e minus_rel=%10.6e odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   alpha, stage.name, s.plus_rel_l2, s.minus_rel_l2,
                   s.odd_rel_l2, s.even_rel_l2, s.curvature_ratio);
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
