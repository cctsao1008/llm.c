#pragma push_macro("main")
#undef main
#define main xray_qk_precision_embedded_main
#include "qk_precision_response_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>
#include <cstdint>

// Software emulation of the documented TF32 input conversion for finite normal
// FP32 values: keep sign/exponent and the top 10 fraction bits, rounding the
// discarded 13 fraction bits to nearest-even. NaN/Inf are passed through.
__device__ __forceinline__ float xray_round_fp32_to_tf32_rne(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t exp = u & 0x7f800000u;
    if (exp == 0x7f800000u) return x;

    // Round-to-nearest-even before clearing the low 13 fraction bits.
    uint32_t lsb = (u >> 13) & 1u;
    u += 0x00000fffu + lsb;
    u &= 0xffffe000u;
    return __uint_as_float(u);
}

__global__ void xray_tf32_round_kernel(float* out, const float* in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = xray_round_fp32_to_tf32_rne(in[i]);
}

struct RasterStats {
    size_t changed_plus = 0;
    size_t changed_minus = 0;
    size_t changed_union = 0;
    size_t asymmetric_cells = 0;
    double odd_rel = 0.0;
    double even_rel = 0.0;
    double curvature_ratio = 0.0;
};

static RasterStats raster_stats(const std::vector<float>& base,
                                const std::vector<float>& plus,
                                const std::vector<float>& minus) {
    RasterStats r;
    long double base2 = 0.0L, odd2 = 0.0L, even2 = 0.0L;
    for (size_t i = 0; i < base.size(); ++i) {
        float b = base[i], p = plus[i], m = minus[i];
        bool cp = (p != b);
        bool cm = (m != b);
        r.changed_plus += cp;
        r.changed_minus += cm;
        r.changed_union += (cp || cm);

        // In a perfectly odd local response around the same cell center,
        // p-b == -(m-b). Count coordinates where that fails exactly.
        float dp = p - b;
        float dm = m - b;
        r.asymmetric_cells += (dp != -dm);

        long double odd = 0.5L * ((long double)p - (long double)m);
        long double even = 0.5L * ((long double)p + (long double)m) - (long double)b;
        base2 += (long double)b * (long double)b;
        odd2 += odd * odd;
        even2 += even * even;
    }
    long double denom = sqrtl(base2);
    long double on = sqrtl(odd2);
    long double en = sqrtl(even2);
    r.odd_rel = denom > 0 ? (double)(on / denom) : 0.0;
    r.even_rel = denom > 0 ? (double)(en / denom) : 0.0;
    r.curvature_ratio = on > 0 ? (double)(en / on) : 0.0;
    return r;
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

    float *d_custom=nullptr, *d_tf32=nullptr, *d_blend=nullptr, *d_rounded=nullptr;
    float *d_qkvr=nullptr, *d_preatt=nullptr, *d_att=nullptr;
    cudaCheck(cudaMalloc((void**)&d_custom, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_tf32, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_blend, QKVN * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_rounded, QKVN * sizeof(float)));
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

    // Baseline: software-TF32-rounded custom QKV, then PEDANTIC QK GEMM.
    cudaCheck(cudaMemcpy(d_blend, d_custom, QKVN * sizeof(float), cudaMemcpyDeviceToDevice));
    xray_tf32_round_kernel<<<grid, block>>>(d_rounded, d_blend, QKVN);
    cudaCheck(cudaGetLastError());
    xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_rounded, B, T, C, NH, CUBLAS_PEDANTIC_MATH);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> rounded_base(QKVN);
    std::vector<float> preatt_base(BHTT), att_base(BHTT);
    cudaCheck(cudaMemcpy(rounded_base.data(), d_rounded, QKVN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(preatt_base.data(), d_preatt, BHTT * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(att_base.data(), d_att, BHTT * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[xray][tf32-raster] L02 natural QKV delta; test whether TF32 input rounding alone reproduces QK symmetry breaking\n");
    printf("[xray][tf32-raster] path=software TF32 RNE input rasterization -> PEDANTIC Q*K^T -> unchanged softmax\n");

    for (int mode : modes) {
        for (float alpha : alphas) {
            std::vector<float> rounded_plus(QKVN), rounded_minus(QKVN);
            std::vector<float> preatt_plus(BHTT), preatt_minus(BHTT);
            std::vector<float> att_plus(BHTT), att_minus(BHTT);

            xray_selective_qkv_blend<<<grid, block>>>(d_blend, d_custom, d_tf32, QKVN, C, +alpha, mode);
            xray_tf32_round_kernel<<<grid, block>>>(d_rounded, d_blend, QKVN);
            xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_rounded, B, T, C, NH, CUBLAS_PEDANTIC_MATH);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaMemcpy(rounded_plus.data(), d_rounded, QKVN * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(preatt_plus.data(), d_preatt, BHTT * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(att_plus.data(), d_att, BHTT * sizeof(float), cudaMemcpyDeviceToHost));

            xray_selective_qkv_blend<<<grid, block>>>(d_blend, d_custom, d_tf32, QKVN, C, -alpha, mode);
            xray_tf32_round_kernel<<<grid, block>>>(d_rounded, d_blend, QKVN);
            xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_rounded, B, T, C, NH, CUBLAS_PEDANTIC_MATH);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaMemcpy(rounded_minus.data(), d_rounded, QKVN * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(preatt_minus.data(), d_preatt, BHTT * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(att_minus.data(), d_att, BHTT * sizeof(float), cudaMemcpyDeviceToHost));

            RasterStats rs = raster_stats(rounded_base, rounded_plus, rounded_minus);
            LocalSymmetryStats ps = local_symmetry_stats(preatt_base, preatt_plus, preatt_minus, alpha, nullptr);
            LocalSymmetryStats as = local_symmetry_stats(att_base, att_plus, att_minus, alpha, nullptr);

            printf("[xray][tf32-raster] mode=%-6s alpha=%5.3f stage=input-cells changed_plus=%zu changed_minus=%zu changed_union=%zu asymmetric=%zu odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   mode_name(mode), alpha, rs.changed_plus, rs.changed_minus, rs.changed_union, rs.asymmetric_cells,
                   rs.odd_rel, rs.even_rel, rs.curvature_ratio);
            printf("[xray][tf32-raster] mode=%-6s alpha=%5.3f stage=preatt-QK   odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   mode_name(mode), alpha, ps.odd_rel_l2, ps.even_rel_l2, ps.curvature_ratio);
            printf("[xray][tf32-raster] mode=%-6s alpha=%5.3f stage=att-softmax odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   mode_name(mode), alpha, as.odd_rel_l2, as.even_rel_l2, as.curvature_ratio);
        }
    }

    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_rounded));
    cudaCheck(cudaFree(d_blend));
    cudaCheck(cudaFree(d_tf32));
    cudaCheck(cudaFree(d_custom));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
