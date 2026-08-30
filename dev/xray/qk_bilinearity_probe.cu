#pragma push_macro("main")
#undef main
#define main xray_attention_breakpoint_embedded_main
#include "attention_breakpoint_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>

enum XrayQKMode {
    XRAY_Q_ONLY = 1,
    XRAY_K_ONLY = 2,
    XRAY_QK_BOTH = 3,
    XRAY_V_ONLY = 4,
};

__global__ void xray_selective_qkv_blend(float* out,
                                         const float* custom,
                                         const float* tf32,
                                         size_t n,
                                         int C,
                                         float alpha,
                                         int mode) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c3 = (int)(i % (size_t)(3 * C));
    int part = c3 / C; // 0=Q, 1=K, 2=V
    bool selected = (part == 0 && (mode & XRAY_Q_ONLY)) ||
                    (part == 1 && (mode & XRAY_K_ONLY)) ||
                    (part == 2 && (mode & XRAY_V_ONLY));

    float a = custom[i];
    out[i] = selected ? a + alpha * (tf32[i] - a) : a;
}

struct QKStage {
    const char* name;
    std::vector<float> baseline;
    std::vector<float> plus;
    std::vector<float> minus;
};

static const char* mode_name(int mode) {
    switch (mode) {
        case XRAY_Q_ONLY: return "Q-only";
        case XRAY_K_ONLY: return "K-only";
        case XRAY_QK_BOTH: return "Q+K";
        case XRAY_V_ONLY: return "V-only";
        default: return "unknown";
    }
}

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

    // Establish the exact fixed L2 input used by the prior probes.
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

    QKStage preatt{"preatt-QK", {}, {}, {}};
    QKStage att{"att-softmax", {}, {}, {}};

    // Baseline is custom Q,K,V and is shared across all selective modes.
    cudaCheck(cudaMemcpy(d_blend, d_custom, QKVN * sizeof(float), cudaMemcpyDeviceToDevice));
    xray_attention_prefix(d_qkvr, d_preatt, d_att, d_blend, B, T, C, NH);
    cudaCheck(cudaDeviceSynchronize());
    xray_copy(preatt.baseline, d_preatt, BHTT);
    xray_copy(att.baseline, d_att, BHTT);

    auto run_side = [&](int mode, float alpha, bool plus_side) {
        xray_selective_qkv_blend<<<grid, block>>>(d_blend, d_custom, d_tf32, QKVN, C, alpha, mode);
        cudaCheck(cudaGetLastError());
        xray_attention_prefix(d_qkvr, d_preatt, d_att, d_blend, B, T, C, NH);
        cudaCheck(cudaDeviceSynchronize());
        xray_copy(plus_side ? preatt.plus : preatt.minus, d_preatt, BHTT);
        xray_copy(plus_side ? att.plus : att.minus, d_att, BHTT);
    };

    printf("[xray][qk-bilinearity] source=L02 natural TF32-custom QKV delta; isolate Q-only, K-only, Q+K, V-only\n");
    printf("[xray][qk-bilinearity] real arithmetic prediction: Q-only and K-only make raw QK scores affine in alpha; Q+K adds alpha^2*dQ*dK^T; V-only cannot change scores\n");

    const int modes[] = {XRAY_Q_ONLY, XRAY_K_ONLY, XRAY_QK_BOTH, XRAY_V_ONLY};
    const float alphas[] = {0.125f, 0.25f, 0.5f, 1.0f};

    for (int mode : modes) {
        for (float alpha : alphas) {
            run_side(mode, +alpha, true);
            run_side(mode, -alpha, false);

            LocalSymmetryStats ps = local_symmetry_stats(preatt.baseline, preatt.plus, preatt.minus, alpha, nullptr);
            LocalSymmetryStats as = local_symmetry_stats(att.baseline, att.plus, att.minus, alpha, nullptr);

            printf("[xray][qk-bilinearity] mode=%-6s alpha=%5.3f stage=preatt-QK   odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   mode_name(mode), alpha, ps.odd_rel_l2, ps.even_rel_l2, ps.curvature_ratio);
            printf("[xray][qk-bilinearity] mode=%-6s alpha=%5.3f stage=att-softmax odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   mode_name(mode), alpha, as.odd_rel_l2, as.even_rel_l2, as.curvature_ratio);
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
