#pragma push_macro("main")
#undef main
#define main xray_downstream_decision_linearity_embedded_main
#include "downstream_decision_linearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

struct XrayDecisionTrace {
    std::vector<float> final_residual; // target row before final LayerNorm, [C]
    std::vector<float> lnf;            // target row after GPU final LayerNorm, [C]
    std::vector<float> gpu_logits;     // target GPU logits, [V]
};

struct XrayReadoutStats {
    int winner;
    int runner;
    double top1;
    double top2;
    double top2_margin;
    double pair_margin;
    double gpu_error_rms;
    double gpu_error_max;
};

static int xray_sign64(double x) {
    return (x > 0.0) - (x < 0.0);
}

static void xray_copy_host(std::vector<float>& dst, const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static XrayReadoutStats xray_stats_from_gpu_logits(const std::vector<float>& logits,
                                                     int pair_a, int pair_b) {
    int winner = -1, runner = -1;
    double top1 = -std::numeric_limits<double>::infinity();
    double top2 = -std::numeric_limits<double>::infinity();
    for (int v = 0; v < (int)logits.size(); ++v) {
        const double z = (double)logits[v];
        if (z > top1) {
            top2 = top1;
            runner = winner;
            top1 = z;
            winner = v;
        } else if (z > top2) {
            top2 = z;
            runner = v;
        }
    }
    return XrayReadoutStats{
        winner, runner, top1, top2, top1 - top2,
        (double)logits[pair_a] - (double)logits[pair_b],
        0.0, 0.0
    };
}

static std::vector<double> xray_cpu64_layernorm(const std::vector<float>& residual,
                                                 const std::vector<float>& weight,
                                                 const std::vector<float>& bias) {
    const int C = (int)residual.size();
    double sum = 0.0;
    for (int c = 0; c < C; ++c) sum += (double)residual[c];
    const double mean = sum / (double)C;

    double var_sum = 0.0;
    for (int c = 0; c < C; ++c) {
        const double d = (double)residual[c] - mean;
        var_sum += d * d;
    }
    const double rstd = 1.0 / std::sqrt(var_sum / (double)C + 1.0e-5);

    std::vector<double> out(C);
    for (int c = 0; c < C; ++c) {
        const double n = ((double)residual[c] - mean) * rstd;
        out[c] = n * (double)weight[c] + (double)bias[c];
    }
    return out;
}

static XrayReadoutStats xray_cpu64_classifier(const std::vector<double>& hidden,
                                                const std::vector<float>& wte,
                                                const std::vector<float>& gpu_logits,
                                                int V, int C,
                                                int pair_a, int pair_b) {
    int winner = -1, runner = -1;
    double top1 = -std::numeric_limits<double>::infinity();
    double top2 = -std::numeric_limits<double>::infinity();
    double pair_a_score = 0.0;
    double pair_b_score = 0.0;
    long double err2 = 0.0L;
    double err_max = 0.0;

    for (int v = 0; v < V; ++v) {
        const float* w = wte.data() + (size_t)v * C;
        double z = 0.0;
        for (int c = 0; c < C; ++c) {
            z += hidden[c] * (double)w[c];
        }

        const double err = z - (double)gpu_logits[v];
        err2 += (long double)err * err;
        err_max = std::max(err_max, std::fabs(err));

        if (v == pair_a) pair_a_score = z;
        if (v == pair_b) pair_b_score = z;

        if (z > top1) {
            top2 = top1;
            runner = winner;
            top1 = z;
            winner = v;
        } else if (z > top2) {
            top2 = z;
            runner = v;
        }
    }

    return XrayReadoutStats{
        winner, runner, top1, top2, top1 - top2,
        pair_a_score - pair_b_score,
        std::sqrt((double)(err2 / (long double)V)), err_max
    };
}

static double xray_rel_l2_cpu64_vs_gpu(const std::vector<double>& cpu,
                                        const std::vector<float>& gpu,
                                        double* max_abs) {
    long double ref2 = 0.0L;
    long double diff2 = 0.0L;
    double mx = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const long double a = (long double)cpu[i];
        const long double d = (long double)gpu[i] - a;
        ref2 += a * a;
        diff2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
    }
    *max_abs = mx;
    return ref2 > 0.0L ? std::sqrt((double)(diff2 / ref2)) : std::sqrt((double)diff2);
}

static void xray_capture_decision_trace(XrayDecisionTrace* tr,
                                        GPT2* model, float* l00_state,
                                        int B, int T, int target) {
    const int L = model->config.num_layers;
    const int C = model->config.channels;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;
    const size_t BTC = (size_t)B * T * C;

    xray_forward_from_residual3(model, l00_state, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());

    xray_copy_host(tr->final_residual,
                   model->acts.residual3 + (size_t)(L - 1) * BTC + (size_t)target * C,
                   C);
    xray_copy_host(tr->lnf,
                   model->acts.lnf + (size_t)target * C,
                   C);
    xray_copy_host(tr->gpu_logits,
                   model->acts.output + (size_t)target * Vp,
                   V);
}

static double xray_abs_over_error(double signal, double error) {
    const double e = std::fabs(error);
    return e > 0.0 ? std::fabs(signal) / e : std::numeric_limits<double>::infinity();
}

static void xray_audit_state(const char* label,
                             const XrayDecisionTrace& tr,
                             const std::vector<float>& wte,
                             const std::vector<float>& lnfw,
                             const std::vector<float>& lnfb,
                             int V, int C,
                             int pair_a, int pair_b,
                             int* all_readout_top1,
                             int* all_suffix_top1,
                             int* all_readout_pair_sign,
                             int* all_suffix_pair_sign,
                             double* min_readout_safety,
                             double* min_suffix_safety) {
    const XrayReadoutStats gpu = xray_stats_from_gpu_logits(tr.gpu_logits, pair_a, pair_b);

    std::vector<double> gpu_lnf64(C);
    for (int c = 0; c < C; ++c) gpu_lnf64[c] = (double)tr.lnf[c];
    const XrayReadoutStats cpu_readout =
        xray_cpu64_classifier(gpu_lnf64, wte, tr.gpu_logits, V, C, pair_a, pair_b);

    const std::vector<double> cpu_lnf64 =
        xray_cpu64_layernorm(tr.final_residual, lnfw, lnfb);
    const XrayReadoutStats cpu_suffix =
        xray_cpu64_classifier(cpu_lnf64, wte, tr.gpu_logits, V, C, pair_a, pair_b);

    double lnf_max_abs = 0.0;
    const double lnf_rel_l2 = xray_rel_l2_cpu64_vs_gpu(cpu_lnf64, tr.lnf, &lnf_max_abs);

    const double readout_pair_err = cpu_readout.pair_margin - gpu.pair_margin;
    const double suffix_pair_err = cpu_suffix.pair_margin - gpu.pair_margin;
    const double suffix_minus_readout = cpu_suffix.pair_margin - cpu_readout.pair_margin;
    const int readout_top1_agree = cpu_readout.winner == gpu.winner;
    const int suffix_top1_agree = cpu_suffix.winner == gpu.winner;
    const int readout_pair_sign_agree = xray_sign64(cpu_readout.pair_margin) == xray_sign64(gpu.pair_margin);
    const int suffix_pair_sign_agree = xray_sign64(cpu_suffix.pair_margin) == xray_sign64(gpu.pair_margin);
    const double readout_safety = xray_abs_over_error(cpu_readout.pair_margin, readout_pair_err);
    const double suffix_safety = xray_abs_over_error(cpu_suffix.pair_margin, suffix_pair_err);

    *all_readout_top1 &= readout_top1_agree;
    *all_suffix_top1 &= suffix_top1_agree;
    *all_readout_pair_sign &= readout_pair_sign_agree;
    *all_suffix_pair_sign &= suffix_pair_sign_agree;
    *min_readout_safety = std::min(*min_readout_safety, readout_safety);
    *min_suffix_safety = std::min(*min_suffix_safety, suffix_safety);

    printf("[xray][decision-cpu64-state] state=%s gpu_top1=%d gpu_runner=%d gpu_top2_margin=%+.9e pair_gpu=%+.9e cpu64_readout_top1=%d cpu64_readout_runner=%d cpu64_readout_top2_margin=%+.9e pair_readout=%+.9e cpu64_suffix_top1=%d cpu64_suffix_runner=%d cpu64_suffix_top2_margin=%+.9e pair_suffix=%+.9e\n",
           label,
           gpu.winner, gpu.runner, gpu.top2_margin, gpu.pair_margin,
           cpu_readout.winner, cpu_readout.runner, cpu_readout.top2_margin, cpu_readout.pair_margin,
           cpu_suffix.winner, cpu_suffix.runner, cpu_suffix.top2_margin, cpu_suffix.pair_margin);

    printf("[xray][decision-cpu64-error] state=%s readout_pair_err=%+.9e suffix_pair_err=%+.9e suffix_minus_readout=%+.9e lnf_rel_l2=%.9e lnf_max_abs=%.9e readout_logit_rms_err=%.9e readout_logit_max_err=%.9e suffix_logit_rms_err=%.9e suffix_logit_max_err=%.9e\n",
           label,
           readout_pair_err, suffix_pair_err, suffix_minus_readout,
           lnf_rel_l2, lnf_max_abs,
           cpu_readout.gpu_error_rms, cpu_readout.gpu_error_max,
           cpu_suffix.gpu_error_rms, cpu_suffix.gpu_error_max);

    printf("[xray][decision-cpu64-stability] state=%s top1_readout_agree=%d top1_suffix_agree=%d pair_sign_readout_agree=%d pair_sign_suffix_agree=%d abs_pair_over_readout_err=%.9e abs_pair_over_suffix_err=%.9e\n",
           label,
           readout_top1_agree, suffix_top1_agree,
           readout_pair_sign_agree, suffix_pair_sign_agree,
           readout_safety, suffix_safety);
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int target = argc > 3 ? atoi(argv[3]) : 1186;
    if (B <= 0 || T <= 0 || target < 0 || target >= B * T) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [token=1186]\n", argv[0]);
        return 2;
    }

    cudaCheck(cudaSetDevice(0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
                    B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    const int C = model.config.channels;
    const int V = model.config.vocab_size;
    const int b = target / T;
    const int t = target % T;
    if (t < 3) {
        fprintf(stderr, "target local t=%d is too small for [t-3,t-2] pair\n", t);
        return 3;
    }

    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    // Natural reference execution and the known alternate L00 fcproj execution.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_top2_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_top2_margin);

    float *ref_l00 = nullptr, *low_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_l00, state_bytes));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3,
                         state_bytes, cudaMemcpyDeviceToDevice));

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_top2_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_top2_margin);
    cudaCheck(cudaMemcpy(low_l00, model.acts.residual3,
                         state_bytes, cudaMemcpyDeviceToDevice));

    if (ref_winner == low_winner) {
        printf("[xray][decision-cpu64] target=%d has no execution disagreement ref=%d low=%d\n",
               target, ref_winner, low_winner);
        return 0;
    }

    // Host copies of the exact FP32 parameters used by the GPU final LayerNorm
    // and tied vocabulary projection. CPU arithmetic below is IEEE double.
    std::vector<float> wte((size_t)V * C);
    std::vector<float> lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         C * sizeof(float), cudaMemcpyDeviceToHost));

    const int row_a = b * T + (t - 3);
    const int row_b = b * T + (t - 2);
    float *s00 = nullptr, *s10 = nullptr, *s01 = nullptr, *s11 = nullptr;
    cudaCheck(cudaMalloc((void**)&s00, state_bytes));
    cudaCheck(cudaMalloc((void**)&s10, state_bytes));
    cudaCheck(cudaMalloc((void**)&s01, state_bytes));
    cudaCheck(cudaMalloc((void**)&s11, state_bytes));
    cudaCheck(cudaMemcpy(s00, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s10, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s01, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11, low_l00, state_bytes, cudaMemcpyDeviceToDevice));

    cudaCheck(cudaMemcpy(s10 + (size_t)row_a * C, ref_l00 + (size_t)row_a * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s01 + (size_t)row_b * C, ref_l00 + (size_t)row_b * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11 + (size_t)row_a * C, ref_l00 + (size_t)row_a * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11 + (size_t)row_b * C, ref_l00 + (size_t)row_b * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));

    XrayDecisionTrace z00{}, z10{}, z01{}, z11{};
    xray_capture_decision_trace(&z00, &model, s00, B, T, target);
    xray_capture_decision_trace(&z10, &model, s10, B, T, target);
    xray_capture_decision_trace(&z01, &model, s01, B, T, target);
    xray_capture_decision_trace(&z11, &model, s11, B, T, target);

    printf("[xray][decision-cpu64] exact A=t%d B=t%d target=t%d ref=%d low=%d\n",
           t - 3, t - 2, t, ref_winner, low_winner);
    printf("[xray][decision-cpu64] readout: hold GPU lnf fixed and recompute tied classifier on CPU double; suffix: hold GPU final residual fixed and recompute final LayerNorm + classifier on CPU double\n");
    printf("[xray][decision-cpu64] this audits the local final decision readout only; it does not claim a CPU-double reference for transformer layers 1..11\n");

    int all_readout_top1 = 1;
    int all_suffix_top1 = 1;
    int all_readout_pair_sign = 1;
    int all_suffix_pair_sign = 1;
    double min_readout_safety = std::numeric_limits<double>::infinity();
    double min_suffix_safety = std::numeric_limits<double>::infinity();

    xray_audit_state("00", z00, wte, lnfw, lnfb, V, C, ref_winner, low_winner,
                     &all_readout_top1, &all_suffix_top1,
                     &all_readout_pair_sign, &all_suffix_pair_sign,
                     &min_readout_safety, &min_suffix_safety);
    xray_audit_state("10", z10, wte, lnfw, lnfb, V, C, ref_winner, low_winner,
                     &all_readout_top1, &all_suffix_top1,
                     &all_readout_pair_sign, &all_suffix_pair_sign,
                     &min_readout_safety, &min_suffix_safety);
    xray_audit_state("01", z01, wte, lnfw, lnfb, V, C, ref_winner, low_winner,
                     &all_readout_top1, &all_suffix_top1,
                     &all_readout_pair_sign, &all_suffix_pair_sign,
                     &min_readout_safety, &min_suffix_safety);
    xray_audit_state("11", z11, wte, lnfw, lnfb, V, C, ref_winner, low_winner,
                     &all_readout_top1, &all_suffix_top1,
                     &all_readout_pair_sign, &all_suffix_pair_sign,
                     &min_readout_safety, &min_suffix_safety);

    printf("[xray][decision-cpu64-summary] all_top1_readout_agree=%d all_top1_suffix_agree=%d all_pair_sign_readout_agree=%d all_pair_sign_suffix_agree=%d min_abs_pair_over_readout_err=%.9e min_abs_pair_over_suffix_err=%.9e\n",
           all_readout_top1, all_suffix_top1,
           all_readout_pair_sign, all_suffix_pair_sign,
           min_readout_safety, min_suffix_safety);

    cudaCheck(cudaFree(s11));
    cudaCheck(cudaFree(s01));
    cudaCheck(cudaFree(s10));
    cudaCheck(cudaFree(s00));
    cudaCheck(cudaFree(low_l00));
    cudaCheck(cudaFree(ref_l00));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
