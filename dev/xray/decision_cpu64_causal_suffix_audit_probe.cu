#pragma push_macro("main")
#undef main
#define main xray_decision_cpu64_readout_embedded_main
#include "decision_cpu64_readout_audit_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct XrayCpu64LayerWeights {
    std::vector<float> ln1w, ln1b;
    std::vector<float> qkvw, qkvb;
    std::vector<float> attprojw, attprojb;
    std::vector<float> ln2w, ln2b;
    std::vector<float> fcw, fcb;
    std::vector<float> fcprojw, fcprojb;
};

struct XrayGpuCausalTrace {
    std::vector<float> target_residual3; // layers 1..L-1, flattened [(L-1),C]
    std::vector<float> final_residual;   // target row, [C]
    std::vector<float> gpu_logits;       // target row, [V]
};

struct XrayCpu64CausalResult {
    std::vector<double> target_residual3; // layers 1..L-1, flattened [(L-1),C]
    std::vector<double> final_residual;   // target row, [C]
    std::vector<double> lnf;              // target row, [C]
};

static void xray_cpu64_copy_param(std::vector<float>& dst,
                                  const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void xray_cpu64_load_suffix_weights(std::vector<XrayCpu64LayerWeights>& hw,
                                           GPT2* model) {
    const int L = model->config.num_layers;
    const int C = model->config.channels;
    hw.resize(L);
    for (int l = 1; l < L; ++l) {
        auto& w = hw[l];
        xray_cpu64_copy_param(w.ln1w, model->params.ln1w + (size_t)l * C, C);
        xray_cpu64_copy_param(w.ln1b, model->params.ln1b + (size_t)l * C, C);
        xray_cpu64_copy_param(w.qkvw, model->params.qkvw + (size_t)l * 3 * C * C,
                              (size_t)3 * C * C);
        xray_cpu64_copy_param(w.qkvb, model->params.qkvb + (size_t)l * 3 * C, 3 * C);
        xray_cpu64_copy_param(w.attprojw, model->params.attprojw + (size_t)l * C * C,
                              (size_t)C * C);
        xray_cpu64_copy_param(w.attprojb, model->params.attprojb + (size_t)l * C, C);
        xray_cpu64_copy_param(w.ln2w, model->params.ln2w + (size_t)l * C, C);
        xray_cpu64_copy_param(w.ln2b, model->params.ln2b + (size_t)l * C, C);
        xray_cpu64_copy_param(w.fcw, model->params.fcw + (size_t)l * 4 * C * C,
                              (size_t)4 * C * C);
        xray_cpu64_copy_param(w.fcb, model->params.fcb + (size_t)l * 4 * C, 4 * C);
        xray_cpu64_copy_param(w.fcprojw, model->params.fcprojw + (size_t)l * C * 4 * C,
                              (size_t)C * 4 * C);
        xray_cpu64_copy_param(w.fcprojb, model->params.fcprojb + (size_t)l * C, C);
    }
}

static void xray_cpu64_layernorm_rows(std::vector<double>& out,
                                      const std::vector<double>& inp,
                                      const std::vector<float>& weight,
                                      const std::vector<float>& bias,
                                      int N, int C) {
    out.resize((size_t)N * C);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const double* x = inp.data() + (size_t)n * C;
        double* y = out.data() + (size_t)n * C;
        double sum = 0.0;
        for (int c = 0; c < C; ++c) sum += x[c];
        const double mean = sum / (double)C;
        double var_sum = 0.0;
        for (int c = 0; c < C; ++c) {
            const double d = x[c] - mean;
            var_sum += d * d;
        }
        const double rstd = 1.0 / std::sqrt(var_sum / (double)C + 1.0e-5);
        for (int c = 0; c < C; ++c) {
            const double norm = (x[c] - mean) * rstd;
            y[c] = norm * (double)weight[c] + (double)bias[c];
        }
    }
}

static std::vector<double> xray_cpu64_layernorm_row_double(
    const std::vector<double>& inp,
    const std::vector<float>& weight,
    const std::vector<float>& bias) {
    const int C = (int)inp.size();
    double sum = 0.0;
    for (double x : inp) sum += x;
    const double mean = sum / (double)C;
    double var_sum = 0.0;
    for (double x : inp) {
        const double d = x - mean;
        var_sum += d * d;
    }
    const double rstd = 1.0 / std::sqrt(var_sum / (double)C + 1.0e-5);
    std::vector<double> out(C);
    for (int c = 0; c < C; ++c) {
        const double norm = (inp[c] - mean) * rstd;
        out[c] = norm * (double)weight[c] + (double)bias[c];
    }
    return out;
}

static void xray_cpu64_matmul(std::vector<double>& out,
                              const std::vector<double>& inp,
                              const std::vector<float>& weight,
                              const std::vector<float>* bias,
                              int N, int C, int OC) {
    out.resize((size_t)N * OC);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < OC; ++o) {
        const float* w = weight.data() + (size_t)o * C;
        const double b = bias ? (double)(*bias)[o] : 0.0;
        for (int n = 0; n < N; ++n) {
            const double* x = inp.data() + (size_t)n * C;
            double acc = b;
            for (int c = 0; c < C; ++c) {
                acc += x[c] * (double)w[c];
            }
            out[(size_t)n * OC + o] = acc;
        }
    }
}

static void xray_cpu64_attention(std::vector<double>& out,
                                 const std::vector<double>& qkv,
                                 int P, int C, int NH) {
    const int HS = C / NH;
    const int C3 = 3 * C;
    const double scale = 1.0 / std::sqrt((double)HS);
    out.assign((size_t)P * C, 0.0);
    // One probability row for each (query position, head). This is modest for
    // the causal prefix used here and avoids recomputing exp() for every HS.
    std::vector<double> prob((size_t)P * NH * P, 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < P; ++t) {
        for (int h = 0; h < NH; ++h) {
            const double* q = qkv.data() + (size_t)t * C3 + (size_t)h * HS;
            double* p = prob.data() + ((size_t)t * NH + h) * P;
            double max_score = -std::numeric_limits<double>::infinity();

            for (int t2 = 0; t2 <= t; ++t2) {
                const double* k = qkv.data() + (size_t)t2 * C3 + C + (size_t)h * HS;
                double dot = 0.0;
                for (int c = 0; c < HS; ++c) dot += q[c] * k[c];
                const double score = dot * scale;
                p[t2] = score;
                max_score = std::max(max_score, score);
            }

            double denom = 0.0;
            for (int t2 = 0; t2 <= t; ++t2) {
                const double e = std::exp(p[t2] - max_score);
                p[t2] = e;
                denom += e;
            }
            const double inv_denom = 1.0 / denom;
            for (int t2 = 0; t2 <= t; ++t2) p[t2] *= inv_denom;

            double* y = out.data() + (size_t)t * C + (size_t)h * HS;
            for (int c = 0; c < HS; ++c) {
                double acc = 0.0;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const double* v = qkv.data() + (size_t)t2 * C3 + 2 * C + (size_t)h * HS;
                    acc += p[t2] * v[c];
                }
                y[c] = acc;
            }
        }
    }
}

static void xray_cpu64_add(std::vector<double>& out,
                           const std::vector<double>& a,
                           const std::vector<double>& b) {
    out.resize(a.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)a.size(); ++i) out[(size_t)i] = a[(size_t)i] + b[(size_t)i];
}

static void xray_cpu64_gelu(std::vector<double>& out,
                            const std::vector<double>& inp) {
    // Same approximate GELU formula as llmc/gelu.cuh, evaluated in double.
    constexpr double gelu_scale = 0.79788456080286535587989211986876; // sqrt(2/pi)
    constexpr double gelu_cube = 0.044715;
    out.resize(inp.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)inp.size(); ++i) {
        const double x = inp[(size_t)i];
        const double cube = gelu_cube * x * x * x;
        out[(size_t)i] = 0.5 * x * (1.0 + std::tanh(gelu_scale * (x + cube)));
    }
}

static double xray_cpu64_rel_l2_row(const double* cpu, const float* gpu,
                                    int C, double* max_abs) {
    long double ref2 = 0.0L;
    long double diff2 = 0.0L;
    double mx = 0.0;
    for (int c = 0; c < C; ++c) {
        const long double a = (long double)cpu[c];
        const long double d = (long double)gpu[c] - a;
        ref2 += a * a;
        diff2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
    }
    *max_abs = mx;
    return ref2 > 0.0L ? std::sqrt((double)(diff2 / ref2)) : std::sqrt((double)diff2);
}

static XrayCpu64CausalResult xray_cpu64_run_causal_suffix(
    const std::vector<float>& l00_prefix,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw,
    const std::vector<float>& lnfb,
    int L, int P, int C, int NH) {

    std::vector<double> residual(l00_prefix.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)l00_prefix.size(); ++i) {
        residual[(size_t)i] = (double)l00_prefix[(size_t)i];
    }

    XrayCpu64CausalResult result;
    result.target_residual3.resize((size_t)(L - 1) * C);

    std::vector<double> ln1, qkv, atty, attproj, residual2;
    std::vector<double> ln2, fch, gelu, fcproj, residual3;

    for (int l = 1; l < L; ++l) {
        const auto& w = hw[l];
        xray_cpu64_layernorm_rows(ln1, residual, w.ln1w, w.ln1b, P, C);
        xray_cpu64_matmul(qkv, ln1, w.qkvw, &w.qkvb, P, C, 3 * C);
        xray_cpu64_attention(atty, qkv, P, C, NH);
        xray_cpu64_matmul(attproj, atty, w.attprojw, &w.attprojb, P, C, C);
        xray_cpu64_add(residual2, residual, attproj);
        xray_cpu64_layernorm_rows(ln2, residual2, w.ln2w, w.ln2b, P, C);
        xray_cpu64_matmul(fch, ln2, w.fcw, &w.fcb, P, C, 4 * C);
        xray_cpu64_gelu(gelu, fch);
        xray_cpu64_matmul(fcproj, gelu, w.fcprojw, &w.fcprojb, P, 4 * C, C);
        xray_cpu64_add(residual3, residual2, fcproj);

        const double* target_row = residual3.data() + (size_t)(P - 1) * C;
        std::copy(target_row, target_row + C,
                  result.target_residual3.begin() + (size_t)(l - 1) * C);
        residual.swap(residual3);
    }

    result.final_residual.assign(residual.end() - C, residual.end());
    result.lnf = xray_cpu64_layernorm_row_double(result.final_residual, lnfw, lnfb);
    return result;
}

static void xray_capture_gpu_causal_trace(XrayGpuCausalTrace* tr,
                                          GPT2* model, float* l00_state,
                                          int B, int T, int target) {
    const int L = model->config.num_layers;
    const int C = model->config.channels;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;
    const size_t BTC = (size_t)B * T * C;

    xray_forward_from_residual3(model, l00_state, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());

    tr->target_residual3.resize((size_t)(L - 1) * C);
    for (int l = 1; l < L; ++l) {
        cudaCheck(cudaMemcpy(tr->target_residual3.data() + (size_t)(l - 1) * C,
                             model->acts.residual3 + (size_t)l * BTC + (size_t)target * C,
                             C * sizeof(float), cudaMemcpyDeviceToHost));
    }
    xray_copy_host(tr->final_residual,
                   model->acts.residual3 + (size_t)(L - 1) * BTC + (size_t)target * C,
                   C);
    xray_copy_host(tr->gpu_logits,
                   model->acts.output + (size_t)target * Vp,
                   V);
}

static std::vector<float> xray_copy_state_prefix(float* state,
                                                  int b, int T, int P, int C) {
    std::vector<float> out((size_t)P * C);
    const float* src = state + (size_t)b * T * C;
    cudaCheck(cudaMemcpy(out.data(), src, (size_t)P * C * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

static void xray_print_cpu64_causal_state(
    const char* label,
    const XrayGpuCausalTrace& gpu_trace,
    const XrayCpu64CausalResult& cpu,
    const std::vector<float>& wte,
    int V, int C, int L,
    int pair_a, int pair_b,
    int* all_top1_agree,
    int* all_pair_sign_agree,
    double* min_pair_over_delta) {

    const XrayReadoutStats gpu = xray_stats_from_gpu_logits(gpu_trace.gpu_logits, pair_a, pair_b);
    const XrayReadoutStats ref = xray_cpu64_classifier(cpu.lnf, wte, gpu_trace.gpu_logits,
                                                       V, C, pair_a, pair_b);
    const double pair_delta = ref.pair_margin - gpu.pair_margin;
    const int top1_agree = ref.winner == gpu.winner;
    const int pair_sign_agree = xray_sign64(ref.pair_margin) == xray_sign64(gpu.pair_margin);
    const double ratio = xray_abs_over_error(ref.pair_margin, pair_delta);

    *all_top1_agree &= top1_agree;
    *all_pair_sign_agree &= pair_sign_agree;
    *min_pair_over_delta = std::min(*min_pair_over_delta, ratio);

    printf("[xray][decision-cpu64-causal-state] state=%s gpu_top1=%d gpu_runner=%d gpu_top2_margin=%+.9e pair_gpu=%+.9e cpu64_top1=%d cpu64_runner=%d cpu64_top2_margin=%+.9e pair_cpu64=%+.9e pair_delta=%+.9e top1_agree=%d pair_sign_agree=%d abs_pair_over_gpu_cpu64_delta=%.9e logit_rms_delta=%.9e logit_max_delta=%.9e\n",
           label,
           gpu.winner, gpu.runner, gpu.top2_margin, gpu.pair_margin,
           ref.winner, ref.runner, ref.top2_margin, ref.pair_margin,
           pair_delta, top1_agree, pair_sign_agree, ratio,
           ref.gpu_error_rms, ref.gpu_error_max);

    double max_rel = 0.0;
    int max_rel_layer = -1;
    for (int l = 1; l < L; ++l) {
        const double* cr = cpu.target_residual3.data() + (size_t)(l - 1) * C;
        const float* gr = gpu_trace.target_residual3.data() + (size_t)(l - 1) * C;
        double max_abs = 0.0;
        const double rel = xray_cpu64_rel_l2_row(cr, gr, C, &max_abs);
        if (rel > max_rel) {
            max_rel = rel;
            max_rel_layer = l;
        }
        printf("[xray][decision-cpu64-causal-layer] state=%s L=%02d target_residual_rel_l2=%.9e max_abs=%.9e\n",
               label, l, rel, max_abs);
    }
    printf("[xray][decision-cpu64-causal-path] state=%s max_target_residual_rel_l2=%.9e max_layer=%02d final_target_residual_rel_l2=%.9e\n",
           label, max_rel, max_rel_layer,
           [&](){ double mx=0.0; return xray_cpu64_rel_l2_row(cpu.final_residual.data(), gpu_trace.final_residual.data(), C, &mx); }());
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

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const int NH = model.config.num_heads;
    const int V = model.config.vocab_size;
    const int b = target / T;
    const int t = target % T;
    const int P = t + 1; // causal prefix is sufficient for target t
    if (t < 3) {
        fprintf(stderr, "target local t=%d is too small for [t-3,t-2] pair\n", t);
        return 3;
    }

    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    // Build the same exact L00 reference/alternate states used by the prior audits.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);

    float *ref_l00 = nullptr, *low_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_l00, state_bytes));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_margin);
    cudaCheck(cudaMemcpy(low_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    if (ref_winner == low_winner) {
        printf("[xray][decision-cpu64-causal] target=%d has no execution disagreement ref=%d low=%d\n",
               target, ref_winner, low_winner);
        return 0;
    }

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

    XrayGpuCausalTrace g00{}, g10{}, g01{}, g11{};
    xray_capture_gpu_causal_trace(&g00, &model, s00, B, T, target);
    xray_capture_gpu_causal_trace(&g10, &model, s10, B, T, target);
    xray_capture_gpu_causal_trace(&g01, &model, s01, B, T, target);
    xray_capture_gpu_causal_trace(&g11, &model, s11, B, T, target);

    // Host parameters remain the exact FP32 checkpoint values; every downstream
    // arithmetic operation in the causal reference path is evaluated in double.
    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte, (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw, C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb, C * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<float> p00 = xray_copy_state_prefix(s00, b, T, P, C);
    const std::vector<float> p10 = xray_copy_state_prefix(s10, b, T, P, C);
    const std::vector<float> p01 = xray_copy_state_prefix(s01, b, T, P, C);
    const std::vector<float> p11 = xray_copy_state_prefix(s11, b, T, P, C);

#ifdef _OPENMP
    const int host_threads = omp_get_max_threads();
#else
    const int host_threads = 1;
#endif
    printf("[xray][decision-cpu64-causal] exact L00 hybrid states; CPU64 recomputes layers 1..%d on only the mathematically sufficient causal prefix [0..%d]\n",
           L - 1, t);
    printf("[xray][decision-cpu64-causal] target=%d b=%d t=%d prefix=%d ref=%d low=%d host_threads=%d\n",
           target, b, t, P, ref_winner, low_winner, host_threads);
    printf("[xray][decision-cpu64-causal] checkpoint weights stay exact FP32 values; LayerNorm, GEMMs, attention/softmax, GELU, residual adds and final readout are evaluated in CPU double\n");

    const auto t0 = std::chrono::steady_clock::now();
    const XrayCpu64CausalResult c00 = xray_cpu64_run_causal_suffix(p00, hw, lnfw, lnfb, L, P, C, NH);
    const XrayCpu64CausalResult c10 = xray_cpu64_run_causal_suffix(p10, hw, lnfw, lnfb, L, P, C, NH);
    const XrayCpu64CausalResult c01 = xray_cpu64_run_causal_suffix(p01, hw, lnfw, lnfb, L, P, C, NH);
    const XrayCpu64CausalResult c11 = xray_cpu64_run_causal_suffix(p11, hw, lnfw, lnfb, L, P, C, NH);
    const auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int all_top1_agree = 1;
    int all_pair_sign_agree = 1;
    double min_pair_over_delta = std::numeric_limits<double>::infinity();
    xray_print_cpu64_causal_state("00", g00, c00, wte, V, C, L, ref_winner, low_winner,
                                  &all_top1_agree, &all_pair_sign_agree, &min_pair_over_delta);
    xray_print_cpu64_causal_state("10", g10, c10, wte, V, C, L, ref_winner, low_winner,
                                  &all_top1_agree, &all_pair_sign_agree, &min_pair_over_delta);
    xray_print_cpu64_causal_state("01", g01, c01, wte, V, C, L, ref_winner, low_winner,
                                  &all_top1_agree, &all_pair_sign_agree, &min_pair_over_delta);
    xray_print_cpu64_causal_state("11", g11, c11, wte, V, C, L, ref_winner, low_winner,
                                  &all_top1_agree, &all_pair_sign_agree, &min_pair_over_delta);

    printf("[xray][decision-cpu64-causal-summary] all_top1_agree=%d all_pair_sign_agree=%d min_abs_pair_over_gpu_cpu64_delta=%.9e cpu64_four_states_ms=%.3f\n",
           all_top1_agree, all_pair_sign_agree, min_pair_over_delta, cpu_ms);

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
