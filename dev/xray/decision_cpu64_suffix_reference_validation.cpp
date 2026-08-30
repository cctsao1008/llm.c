#define TESTING
#include "../../train_gpt2.c"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct XrayCpu32SuffixResult {
    std::vector<float> target_residual3; // layers 1..L-1, flattened [(L-1),C]
    std::vector<float> lnf;              // target row [C]
    std::vector<float> logits;           // target row [V]
};

struct XrayCpu64SuffixResult {
    std::vector<double> target_residual3; // layers 1..L-1, flattened [(L-1),C]
    std::vector<double> lnf;              // target row [C]
    std::vector<double> logits;           // target row [V]
};

struct XrayTop2F {
    int winner;
    int runner;
    double margin;
};

static XrayTop2F xray_top2_float(const std::vector<float>& row) {
    int a = -1, b = -1;
    double av = -std::numeric_limits<double>::infinity();
    double bv = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)row.size(); ++i) {
        const double z = (double)row[i];
        if (z > av) {
            bv = av; b = a; av = z; a = i;
        } else if (z > bv) {
            bv = z; b = i;
        }
    }
    return XrayTop2F{a, b, av - bv};
}

static XrayTop2F xray_top2_double(const std::vector<double>& row) {
    int a = -1, b = -1;
    double av = -std::numeric_limits<double>::infinity();
    double bv = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)row.size(); ++i) {
        const double z = row[i];
        if (z > av) {
            bv = av; b = a; av = z; a = i;
        } else if (z > bv) {
            bv = z; b = i;
        }
    }
    return XrayTop2F{a, b, av - bv};
}

static void xray_target_classifier_cpu32(std::vector<float>& logits,
                                         const float* hidden,
                                         const float* wte,
                                         int V, int C) {
    logits.resize(V);
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < V; ++v) {
        const float* w = wte + (size_t)v * C;
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) acc += hidden[c] * w[c];
        logits[v] = acc;
    }
}

static void xray_target_classifier_cpu64(std::vector<double>& logits,
                                         const double* hidden,
                                         const float* wte,
                                         int V, int C) {
    logits.resize(V);
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < V; ++v) {
        const float* w = wte + (size_t)v * C;
        double acc = 0.0;
        for (int c = 0; c < C; ++c) acc += hidden[c] * (double)w[c];
        logits[v] = acc;
    }
}

// Replay from a layer-0 residual3 checkpoint using the existing train_gpt2.c
// CPU operators verbatim. This is the structural oracle for suffix wiring.
static XrayCpu32SuffixResult xray_cpu32_reference_suffix(
    const std::vector<float>& l00,
    const ParameterTensors& params,
    int L, int P, int C, int NH, int V) {

    std::vector<float> residual = l00;
    XrayCpu32SuffixResult result;
    result.target_residual3.resize((size_t)(L - 1) * C);

    std::vector<float> ln1((size_t)P * C), ln1_mean(P), ln1_rstd(P);
    std::vector<float> qkv((size_t)P * 3 * C);
    std::vector<float> atty((size_t)P * C);
    std::vector<float> preatt((size_t)NH * P * P), att((size_t)NH * P * P);
    std::vector<float> attproj((size_t)P * C), residual2((size_t)P * C);
    std::vector<float> ln2((size_t)P * C), ln2_mean(P), ln2_rstd(P);
    std::vector<float> fch((size_t)P * 4 * C), fch_gelu((size_t)P * 4 * C);
    std::vector<float> fcproj((size_t)P * C), residual3((size_t)P * C);

    for (int l = 1; l < L; ++l) {
        float* l_ln1w = params.ln1w + (size_t)l * C;
        float* l_ln1b = params.ln1b + (size_t)l * C;
        float* l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float* l_qkvb = params.qkvb + (size_t)l * 3 * C;
        float* l_attprojw = params.attprojw + (size_t)l * C * C;
        float* l_attprojb = params.attprojb + (size_t)l * C;
        float* l_ln2w = params.ln2w + (size_t)l * C;
        float* l_ln2b = params.ln2b + (size_t)l * C;
        float* l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float* l_fcb = params.fcb + (size_t)l * 4 * C;
        float* l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float* l_fcprojb = params.fcprojb + (size_t)l * C;

        layernorm_forward(ln1.data(), ln1_mean.data(), ln1_rstd.data(),
                          residual.data(), l_ln1w, l_ln1b, 1, P, C);
        matmul_forward(qkv.data(), ln1.data(), l_qkvw, l_qkvb, 1, P, C, 3 * C);
        attention_forward(atty.data(), preatt.data(), att.data(), qkv.data(), 1, P, C, NH);
        matmul_forward(attproj.data(), atty.data(), l_attprojw, l_attprojb, 1, P, C, C);
        residual_forward(residual2.data(), residual.data(), attproj.data(), P * C);
        layernorm_forward(ln2.data(), ln2_mean.data(), ln2_rstd.data(),
                          residual2.data(), l_ln2w, l_ln2b, 1, P, C);
        matmul_forward(fch.data(), ln2.data(), l_fcw, l_fcb, 1, P, C, 4 * C);
        gelu_forward(fch_gelu.data(), fch.data(), P * 4 * C);
        matmul_forward(fcproj.data(), fch_gelu.data(), l_fcprojw, l_fcprojb,
                       1, P, 4 * C, C);
        residual_forward(residual3.data(), residual2.data(), fcproj.data(), P * C);

        const float* target_row = residual3.data() + (size_t)(P - 1) * C;
        std::copy(target_row, target_row + C,
                  result.target_residual3.begin() + (size_t)(l - 1) * C);
        residual.swap(residual3);
    }

    std::vector<float> lnf_all((size_t)P * C), mean(P), rstd(P);
    layernorm_forward(lnf_all.data(), mean.data(), rstd.data(), residual.data(),
                      params.lnfw, params.lnfb, 1, P, C);
    const float* target_lnf = lnf_all.data() + (size_t)(P - 1) * C;
    result.lnf.assign(target_lnf, target_lnf + C);
    xray_target_classifier_cpu32(result.logits, result.lnf.data(), params.wte, V, C);
    return result;
}

static void xray_cpu64_layernorm_rows(std::vector<double>& out,
                                      const std::vector<double>& inp,
                                      const float* weight, const float* bias,
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

static void xray_cpu64_matmul(std::vector<double>& out,
                              const std::vector<double>& inp,
                              const float* weight, const float* bias,
                              int N, int C, int OC) {
    out.resize((size_t)N * OC);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < OC; ++o) {
        const float* w = weight + (size_t)o * C;
        const double b = bias ? (double)bias[o] : 0.0;
        for (int n = 0; n < N; ++n) {
            const double* x = inp.data() + (size_t)n * C;
            double acc = b;
            for (int c = 0; c < C; ++c) acc += x[c] * (double)w[c];
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
            const double inv = 1.0 / denom;
            for (int t2 = 0; t2 <= t; ++t2) p[t2] *= inv;

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
    for (long long i = 0; i < (long long)a.size(); ++i) {
        out[(size_t)i] = a[(size_t)i] + b[(size_t)i];
    }
}

static void xray_cpu64_gelu(std::vector<double>& out,
                            const std::vector<double>& inp) {
    constexpr double scale = 0.79788456080286535587989211986876;
    constexpr double cube_k = 0.044715;
    out.resize(inp.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)inp.size(); ++i) {
        const double x = inp[(size_t)i];
        const double cube = cube_k * x * x * x;
        out[(size_t)i] = 0.5 * x * (1.0 + std::tanh(scale * (x + cube)));
    }
}

static XrayCpu64SuffixResult xray_cpu64_suffix(
    const std::vector<float>& l00,
    const ParameterTensors& params,
    int L, int P, int C, int NH, int V) {

    std::vector<double> residual(l00.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)l00.size(); ++i) residual[(size_t)i] = (double)l00[(size_t)i];

    XrayCpu64SuffixResult result;
    result.target_residual3.resize((size_t)(L - 1) * C);
    std::vector<double> ln1, qkv, atty, attproj, residual2;
    std::vector<double> ln2, fch, gelu, fcproj, residual3;

    for (int l = 1; l < L; ++l) {
        xray_cpu64_layernorm_rows(ln1, residual,
                                  params.ln1w + (size_t)l * C,
                                  params.ln1b + (size_t)l * C, P, C);
        xray_cpu64_matmul(qkv, ln1,
                          params.qkvw + (size_t)l * 3 * C * C,
                          params.qkvb + (size_t)l * 3 * C, P, C, 3 * C);
        xray_cpu64_attention(atty, qkv, P, C, NH);
        xray_cpu64_matmul(attproj, atty,
                          params.attprojw + (size_t)l * C * C,
                          params.attprojb + (size_t)l * C, P, C, C);
        xray_cpu64_add(residual2, residual, attproj);
        xray_cpu64_layernorm_rows(ln2, residual2,
                                  params.ln2w + (size_t)l * C,
                                  params.ln2b + (size_t)l * C, P, C);
        xray_cpu64_matmul(fch, ln2,
                          params.fcw + (size_t)l * 4 * C * C,
                          params.fcb + (size_t)l * 4 * C, P, C, 4 * C);
        xray_cpu64_gelu(gelu, fch);
        xray_cpu64_matmul(fcproj, gelu,
                          params.fcprojw + (size_t)l * C * 4 * C,
                          params.fcprojb + (size_t)l * C, P, 4 * C, C);
        xray_cpu64_add(residual3, residual2, fcproj);

        const double* target_row = residual3.data() + (size_t)(P - 1) * C;
        std::copy(target_row, target_row + C,
                  result.target_residual3.begin() + (size_t)(l - 1) * C);
        residual.swap(residual3);
    }

    std::vector<double> lnf_all;
    xray_cpu64_layernorm_rows(lnf_all, residual, params.lnfw, params.lnfb, P, C);
    const double* target_lnf = lnf_all.data() + (size_t)(P - 1) * C;
    result.lnf.assign(target_lnf, target_lnf + C);
    xray_target_classifier_cpu64(result.logits, result.lnf.data(), params.wte, V, C);
    return result;
}

static double xray_rel_l2_ff(const float* a, const float* b, int n, double* max_abs, int* unequal) {
    long double a2 = 0.0L, d2 = 0.0L;
    double mx = 0.0;
    int neq = 0;
    for (int i = 0; i < n; ++i) {
        const long double av = (long double)a[i];
        const long double d = (long double)b[i] - av;
        a2 += av * av;
        d2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
        neq += a[i] != b[i];
    }
    *max_abs = mx;
    *unequal = neq;
    return a2 > 0.0L ? std::sqrt((double)(d2 / a2)) : std::sqrt((double)d2);
}

static double xray_rel_l2_df(const double* a, const float* b, int n, double* max_abs) {
    long double a2 = 0.0L, d2 = 0.0L;
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        const long double av = (long double)a[i];
        const long double d = (long double)b[i] - av;
        a2 += av * av;
        d2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
    }
    *max_abs = mx;
    return a2 > 0.0L ? std::sqrt((double)(d2 / a2)) : std::sqrt((double)d2);
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int target = argc > 3 ? atoi(argv[3]) : 1186;
    const int pair_a = argc > 4 ? atoi(argv[4]) : 11906;
    const int pair_b = argc > 5 ? atoi(argv[5]) : 262;
    if (B <= 0 || T <= 0 || target < 0 || target >= B * T) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [token=1186] [pair_a=11906] [pair_b=262]\n", argv[0]);
        return 2;
    }

    const int b = target / T;
    const int t = target % T;
    const int P = t + 1;

    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);
    std::vector<int> prefix(P);
    for (int i = 0; i < P; ++i) prefix[i] = loader.inputs[b * T + i];
    dataloader_free(&loader);

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const int NH = model.config.num_heads;
    const int V = model.config.vocab_size;
    const int Vp = model.config.padded_vocab_size;
    if (pair_a < 0 || pair_a >= V || pair_b < 0 || pair_b >= V) {
        fprintf(stderr, "pair ids out of range V=%d\n", V);
        return 3;
    }

#ifdef _OPENMP
    const int host_threads = omp_get_max_threads();
#else
    const int host_threads = 1;
#endif
    printf("[xray][cpu64-validator] independent oracle=train_gpt2.c CPU reference; source batch B=%d T=%d target=%d -> isolated causal prefix P=%d host_threads=%d\n",
           B, T, target, P, host_threads);
    printf("[xray][cpu64-validator] first validate CPU32 suffix replay against full CPU32 forward, then compare the CPU64 suffix arithmetic to that validated path\n");

    gpt2_forward(&model, prefix.data(), NULL, 1, P);

    const size_t PC = (size_t)P * C;
    std::vector<float> l00(PC);
    std::copy(model.acts.residual3, model.acts.residual3 + PC, l00.begin());

    std::vector<float> full_target_res((size_t)(L - 1) * C);
    for (int l = 1; l < L; ++l) {
        const float* src = model.acts.residual3 + (size_t)l * PC + (size_t)(P - 1) * C;
        std::copy(src, src + C, full_target_res.begin() + (size_t)(l - 1) * C);
    }
    std::vector<float> full_lnf(C);
    std::copy(model.acts.lnf + (size_t)(P - 1) * C,
              model.acts.lnf + (size_t)P * C, full_lnf.begin());
    std::vector<float> full_logits(V);
    std::copy(model.acts.logits + (size_t)(P - 1) * Vp,
              model.acts.logits + (size_t)(P - 1) * Vp + V, full_logits.begin());

    const XrayCpu32SuffixResult replay =
        xray_cpu32_reference_suffix(l00, model.params, L, P, C, NH, V);

    int replay_exact = 1;
    double max_replay_rel = 0.0;
    int max_replay_layer = -1;
    for (int l = 1; l < L; ++l) {
        double mx = 0.0;
        int neq = 0;
        const double rel = xray_rel_l2_ff(
            full_target_res.data() + (size_t)(l - 1) * C,
            replay.target_residual3.data() + (size_t)(l - 1) * C,
            C, &mx, &neq);
        replay_exact &= (neq == 0);
        if (rel > max_replay_rel) { max_replay_rel = rel; max_replay_layer = l; }
        printf("[xray][cpu64-validator-cpu32-replay-layer] L=%02d rel_l2=%.9e max_abs=%.9e unequal=%d/%d\n",
               l, rel, mx, neq, C);
    }
    double replay_lnf_max = 0.0;
    int replay_lnf_neq = 0;
    const double replay_lnf_rel = xray_rel_l2_ff(full_lnf.data(), replay.lnf.data(), C,
                                                  &replay_lnf_max, &replay_lnf_neq);
    double replay_logit_max = 0.0;
    int replay_logit_neq = 0;
    const double replay_logit_rel = xray_rel_l2_ff(full_logits.data(), replay.logits.data(), V,
                                                    &replay_logit_max, &replay_logit_neq);
    replay_exact &= (replay_lnf_neq == 0 && replay_logit_neq == 0);
    printf("[xray][cpu64-validator-cpu32-replay-summary] exact=%d max_layer_rel_l2=%.9e max_layer=%02d lnf_rel_l2=%.9e lnf_unequal=%d/%d logits_rel_l2=%.9e logits_max_abs=%.9e logits_unequal=%d/%d\n",
           replay_exact, max_replay_rel, max_replay_layer,
           replay_lnf_rel, replay_lnf_neq, C,
           replay_logit_rel, replay_logit_max, replay_logit_neq, V);

    const XrayCpu64SuffixResult cpu64 =
        xray_cpu64_suffix(l00, model.params, L, P, C, NH, V);

    double max_cpu64_rel = 0.0;
    int max_cpu64_layer = -1;
    for (int l = 1; l < L; ++l) {
        double mx = 0.0;
        const double rel = xray_rel_l2_df(
            cpu64.target_residual3.data() + (size_t)(l - 1) * C,
            full_target_res.data() + (size_t)(l - 1) * C,
            C, &mx);
        if (rel > max_cpu64_rel) { max_cpu64_rel = rel; max_cpu64_layer = l; }
        printf("[xray][cpu64-validator-cpu64-layer] L=%02d rel_l2=%.9e max_abs=%.9e\n",
               l, rel, mx);
    }
    double cpu64_lnf_max = 0.0;
    const double cpu64_lnf_rel = xray_rel_l2_df(cpu64.lnf.data(), full_lnf.data(), C, &cpu64_lnf_max);
    double cpu64_logit_max = 0.0;
    const double cpu64_logit_rel = xray_rel_l2_df(cpu64.logits.data(), full_logits.data(), V, &cpu64_logit_max);

    const XrayTop2F full_t2 = xray_top2_float(full_logits);
    const XrayTop2F replay_t2 = xray_top2_float(replay.logits);
    const XrayTop2F cpu64_t2 = xray_top2_double(cpu64.logits);
    const double pair_full = (double)full_logits[pair_a] - (double)full_logits[pair_b];
    const double pair_replay = (double)replay.logits[pair_a] - (double)replay.logits[pair_b];
    const double pair_cpu64 = cpu64.logits[pair_a] - cpu64.logits[pair_b];

    printf("[xray][cpu64-validator-decision] cpu32_full_top1=%d runner=%d top2_margin=%+.9e pair=%+.9e cpu32_replay_top1=%d runner=%d top2_margin=%+.9e pair=%+.9e cpu64_top1=%d runner=%d top2_margin=%+.9e pair=%+.9e\n",
           full_t2.winner, full_t2.runner, full_t2.margin, pair_full,
           replay_t2.winner, replay_t2.runner, replay_t2.margin, pair_replay,
           cpu64_t2.winner, cpu64_t2.runner, cpu64_t2.margin, pair_cpu64);
    printf("[xray][cpu64-validator-cpu64-summary] max_layer_rel_l2=%.9e max_layer=%02d lnf_rel_l2=%.9e lnf_max_abs=%.9e logits_rel_l2=%.9e logits_max_abs=%.9e pair_delta_cpu64_minus_cpu32=%+.9e top1_agree=%d pair_sign_agree=%d\n",
           max_cpu64_rel, max_cpu64_layer,
           cpu64_lnf_rel, cpu64_lnf_max,
           cpu64_logit_rel, cpu64_logit_max,
           pair_cpu64 - pair_full,
           cpu64_t2.winner == full_t2.winner,
           ((pair_cpu64 > 0.0) - (pair_cpu64 < 0.0)) == ((pair_full > 0.0) - (pair_full < 0.0)));
    printf("[xray][cpu64-validator-summary] cpu32_replay_exact=%d; interpret CPU64/GPU results only after this structural oracle passes\n",
           replay_exact);

    gpt2_free(&model);
    return replay_exact ? 0 : 4;
}
