#pragma push_macro("main")
#undef main
#define main xray_tf32_raster_embedded_main
#include "tf32_rasterization_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

struct TransformStats {
    size_t valid = 0;
    double identity_rmse_base = 0.0;
    double identity_rmse_plus = 0.0;
    double identity_rmse_minus = 0.0;
    double identity_max = 0.0;
    LocalSymmetryStats score_sym{};
    LocalSymmetryStats logodds_sym{};
};

static TransformStats measure_transform(const std::vector<float>& pre0,
                                        const std::vector<float>& prep,
                                        const std::vector<float>& prem,
                                        const std::vector<float>& att0,
                                        const std::vector<float>& attp,
                                        const std::vector<float>& attm,
                                        int B, int T, int NH, int HS,
                                        float alpha) {
    std::vector<float> score0, scorep, scorem;
    std::vector<float> log0, logp, logm;
    long double e0 = 0.0L, ep = 0.0L, em = 0.0L;
    double maxe = 0.0;
    const double scale = 1.0 / std::sqrt((double)HS);

    score0.reserve((size_t)B * NH * T);
    scorep.reserve((size_t)B * NH * T);
    scorem.reserve((size_t)B * NH * T);
    log0.reserve((size_t)B * NH * T);
    logp.reserve((size_t)B * NH * T);
    logm.reserve((size_t)B * NH * T);

    for (int bh = 0; bh < B * NH; ++bh) {
        size_t base = (size_t)bh * T * T;
        for (int t = 1; t < T; ++t) { // t=0 has only one causal key
            int j1 = 0, j2 = 1;
            if (att0[base + (size_t)t * T + j2] > att0[base + (size_t)t * T + j1]) std::swap(j1, j2);
            for (int j = 2; j <= t; ++j) {
                float p = att0[base + (size_t)t * T + j];
                if (p > att0[base + (size_t)t * T + j1]) {
                    j2 = j1; j1 = j;
                } else if (p > att0[base + (size_t)t * T + j2]) {
                    j2 = j;
                }
            }

            size_t i1 = base + (size_t)t * T + j1;
            size_t i2 = base + (size_t)t * T + j2;
            double p0a = att0[i1], p0b = att0[i2];
            double ppa = attp[i1], ppb = attp[i2];
            double pma = attm[i1], pmb = attm[i2];
            if (!(p0a > 0.0 && p0b > 0.0 && ppa > 0.0 && ppb > 0.0 && pma > 0.0 && pmb > 0.0)) continue;

            double s0 = scale * ((double)pre0[i1] - (double)pre0[i2]);
            double sp = scale * ((double)prep[i1] - (double)prep[i2]);
            double sm = scale * ((double)prem[i1] - (double)prem[i2]);
            double l0 = std::log(p0a / p0b);
            double lp = std::log(ppa / ppb);
            double lm = std::log(pma / pmb);

            score0.push_back((float)s0); scorep.push_back((float)sp); scorem.push_back((float)sm);
            log0.push_back((float)l0); logp.push_back((float)lp); logm.push_back((float)lm);

            double d0 = l0 - s0, dp = lp - sp, dm = lm - sm;
            e0 += (long double)d0 * d0;
            ep += (long double)dp * dp;
            em += (long double)dm * dm;
            maxe = std::max(maxe, std::max(std::fabs(d0), std::max(std::fabs(dp), std::fabs(dm))));
        }
    }

    TransformStats r;
    r.valid = score0.size();
    if (r.valid) {
        r.identity_rmse_base = std::sqrt((double)(e0 / r.valid));
        r.identity_rmse_plus = std::sqrt((double)(ep / r.valid));
        r.identity_rmse_minus = std::sqrt((double)(em / r.valid));
        r.identity_max = maxe;
        r.score_sym = local_symmetry_stats(score0, scorep, scorem, alpha, nullptr);
        r.logodds_sym = local_symmetry_stats(log0, logp, logm, alpha, nullptr);
    }
    return r;
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    const int layer = 2;
    if (B <= 0 || T <= 1) return 2;

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
    int HS = C / NH;
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

    auto capture = [&](int mode, float alpha, std::vector<float>& pre, std::vector<float>& att) {
        xray_selective_qkv_blend<<<grid, block>>>(d_blend, d_custom, d_tf32, QKVN, C, alpha, mode);
        cudaCheck(cudaGetLastError());
        xray_tf32_round_kernel<<<grid, block>>>(d_rounded, d_blend, QKVN);
        cudaCheck(cudaGetLastError());
        xray_attention_prefix_math(d_qkvr, d_preatt, d_att, d_rounded, B, T, C, NH, CUBLAS_PEDANTIC_MATH);
        cudaCheck(cudaDeviceSynchronize());
        pre.resize(BHTT); att.resize(BHTT);
        cudaCheck(cudaMemcpy(pre.data(), d_preatt, BHTT * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(att.data(), d_att, BHTT * sizeof(float), cudaMemcpyDeviceToHost));
    };

    std::vector<float> pre0, att0;
    capture(XRAY_Q_ONLY, 0.0f, pre0, att0);

    printf("[xray][logodds-transform] L02 software-TF32 raster -> PEDANTIC QK -> softmax\n");
    printf("[xray][logodds-transform] transform test: log(p_i/p_j) should equal scaled score difference for causal top-2 baseline keys\n");

    const int modes[] = {XRAY_Q_ONLY, XRAY_K_ONLY};
    const float alphas[] = {0.125f, 1.0f};
    for (int mode : modes) {
        for (float alpha : alphas) {
            std::vector<float> prep, attp, prem, attm;
            capture(mode, +alpha, prep, attp);
            capture(mode, -alpha, prem, attm);
            TransformStats s = measure_transform(pre0, prep, prem, att0, attp, attm, B, T, NH, HS, alpha);
            printf("[xray][logodds-transform] mode=%-6s alpha=%5.3f valid=%zu identity_rmse(base,+,-)=(%.3e,%.3e,%.3e) identity_max=%.3e\n",
                   mode_name(mode), alpha, s.valid, s.identity_rmse_base, s.identity_rmse_plus, s.identity_rmse_minus, s.identity_max);
            printf("[xray][logodds-transform] mode=%-6s alpha=%5.3f score_diff odd_rel=%10.6e even_rel=%10.6e curvature=%10.6e\n",
                   mode_name(mode), alpha, s.score_sym.odd_rel_l2, s.score_sym.even_rel_l2, s.score_sym.curvature_ratio);
            printf("[xray][logodds-transform] mode=%-6s alpha=%5.3f log_odds   odd_rel=%10.6e even_rel=%10.6e curvature=%10.6e\n",
                   mode_name(mode), alpha, s.logodds_sym.odd_rel_l2, s.logodds_sym.even_rel_l2, s.logodds_sym.curvature_ratio);
        }
    }

    cudaCheck(cudaFree(d_att)); cudaCheck(cudaFree(d_preatt)); cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_rounded)); cudaCheck(cudaFree(d_blend)); cudaCheck(cudaFree(d_tf32)); cudaCheck(cudaFree(d_custom));
    dataloader_free(&loader); gpt2_free(&model); cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
