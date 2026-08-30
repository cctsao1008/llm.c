#pragma push_macro("main")
#undef main
#define main xray_l02_stage_embedded_main
#include "decision_numerical_fate_l02_stage_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct XrayF32Diff {
    int unequal;
    double max_abs;
};

static XrayF32Diff xray_f32_diff(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    int neq = 0;
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++neq;
        mx = std::max(mx, std::fabs((double)a[i] - (double)b[i]));
    }
    return XrayF32Diff{neq, mx};
}

static std::vector<float> xray_copy_att_batch_prefix(const float* base,
                                                     int b, int NH, int T, int P) {
    std::vector<float> full((size_t)NH * T * T);
    cudaCheck(cudaMemcpy(full.data(), base + (size_t)b * NH * T * T,
                         full.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> out((size_t)NH * P * P, 0.0f);
    for (int h = 0; h < NH; ++h) {
        for (int t = 0; t < P; ++t) {
            const float* src = full.data() + ((size_t)h * T + t) * T;
            float* dst = out.data() + ((size_t)h * P + t) * P;
            std::copy(src, src + P, dst);
        }
    }
    return out;
}

struct XrayGpuAttentionReplay {
    std::vector<float> preatt;   // [NH,P,P], raw QK scores before scale/softmax
    std::vector<float> att;      // [NH,P,P], post causal softmax
    std::vector<float> atty;     // [P,C], post PV + unpermute
    int att_replay_unequal;
    double att_replay_max_abs;
    int atty_replay_unequal;
    double atty_replay_max_abs;
};

static XrayGpuAttentionReplay xray_replay_l02_attention_exact(GPT2* model,
                                                               int b, int T, int P) {
    const int B = model->batch_size;
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    const int HS = C / NH;
    const int l = 2;
    const size_t BHTT = (size_t)B * NH * T * T;
    const size_t BTC = (size_t)B * T * C;
    const size_t layer_qkvr = (size_t)B * T * 3 * C;

    float* qkvr = model->acts.qkvr + (size_t)l * layer_qkvr;
    float* q = qkvr + 0 * BTC;
    float* k = qkvr + 1 * BTC;
    float* v = qkvr + 2 * BTC;

    float *d_preatt = nullptr, *d_att = nullptr, *d_vaccum = nullptr, *d_atty = nullptr;
    cudaCheck(cudaMalloc((void**)&d_preatt, BHTT * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_att, BHTT * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_vaccum, BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_atty, BTC * sizeof(float)));
    cudaCheck(cudaMemset(d_att, 0, BHTT * sizeof(float)));

    // Exact same subgraph and shapes as llmc/attention.cuh.
    matmul_cublaslt(d_preatt, k, q, nullptr,
                    T, T, HS, main_stream,
                    true, false, B * NH,
                    T * HS, T * HS, T * T);

    const int block_size = 256;
    const float scale = 1.0f / sqrtf((float)HS);
    const int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    softmax_forward_kernel5<<<grid_size, block_size, 0, main_stream>>>(
        d_att, scale, d_preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    matmul_cublaslt(d_vaccum, v, d_att, nullptr,
                    HS, T, T, main_stream,
                    false, false, B * NH,
                    T * HS, T * T, T * HS);
    const int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, main_stream>>>(
        d_vaccum, d_atty, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    XrayGpuAttentionReplay r;
    r.preatt = xray_copy_att_batch_prefix(d_preatt, b, NH, T, P);
    r.att = xray_copy_att_batch_prefix(d_att, b, NH, T, P);
    r.atty.resize((size_t)P * C);
    cudaCheck(cudaMemcpy(r.atty.data(), d_atty + (size_t)b * T * C,
                         r.atty.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const float* original_att = model->acts.att + (size_t)l * BHTT;
    const std::vector<float> original_att_prefix =
        xray_copy_att_batch_prefix(original_att, b, NH, T, P);
    std::vector<float> original_atty((size_t)P * C);
    cudaCheck(cudaMemcpy(original_atty.data(),
                         model->acts.atty + (size_t)l * BTC + (size_t)b * T * C,
                         original_atty.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const XrayF32Diff ad = xray_f32_diff(r.att, original_att_prefix);
    const XrayF32Diff yd = xray_f32_diff(r.atty, original_atty);
    r.att_replay_unequal = ad.unequal;
    r.att_replay_max_abs = ad.max_abs;
    r.atty_replay_unequal = yd.unequal;
    r.atty_replay_max_abs = yd.max_abs;

    cudaCheck(cudaFree(d_atty));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_preatt));
    return r;
}

static void xray_cpu64_raw_qk(std::vector<double>& raw,
                              const std::vector<double>& qkv,
                              int P, int C, int NH) {
    const int HS = C / NH;
    const int C3 = 3 * C;
    raw.assign((size_t)NH * P * P, 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < NH; ++h) {
        for (int t = 0; t < P; ++t) {
            const double* q = qkv.data() + (size_t)t * C3 + (size_t)h * HS;
            double* row = raw.data() + ((size_t)h * P + t) * P;
            for (int t2 = 0; t2 < P; ++t2) {
                const double* k = qkv.data() + (size_t)t2 * C3 + C + (size_t)h * HS;
                double acc = 0.0;
                for (int d = 0; d < HS; ++d) acc += q[d] * k[d];
                row[t2] = acc;
            }
        }
    }
}

static void xray_cpu64_softmax_from_raw_qk(std::vector<double>& prob,
                                            const std::vector<double>& raw,
                                            int P, int NH, int HS) {
    const double scale = 1.0 / std::sqrt((double)HS);
    prob.assign((size_t)NH * P * P, 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < NH; ++h) {
        for (int t = 0; t < P; ++t) {
            const double* s = raw.data() + ((size_t)h * P + t) * P;
            double* p = prob.data() + ((size_t)h * P + t) * P;
            double max_score = -std::numeric_limits<double>::infinity();
            for (int t2 = 0; t2 <= t; ++t2) {
                max_score = std::max(max_score, s[t2] * scale);
            }
            double denom = 0.0;
            for (int t2 = 0; t2 <= t; ++t2) {
                p[t2] = std::exp(s[t2] * scale - max_score);
                denom += p[t2];
            }
            const double inv = 1.0 / denom;
            for (int t2 = 0; t2 <= t; ++t2) p[t2] *= inv;
        }
    }
}

static void xray_cpu64_pv(std::vector<double>& atty,
                          const std::vector<double>& prob,
                          const std::vector<double>& qkv,
                          int P, int C, int NH) {
    const int HS = C / NH;
    const int C3 = 3 * C;
    atty.assign((size_t)P * C, 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < P; ++t) {
        for (int h = 0; h < NH; ++h) {
            const double* p = prob.data() + ((size_t)h * P + t) * P;
            double* out = atty.data() + (size_t)t * C + (size_t)h * HS;
            for (int d = 0; d < HS; ++d) {
                double acc = 0.0;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const double v = qkv[(size_t)t2 * C3 + 2 * C + (size_t)h * HS + d];
                    acc += p[t2] * v;
                }
                out[d] = acc;
            }
        }
    }
}

static std::vector<double> xray_cpu64_finish_l02_from_atty64(
    const std::vector<double>& atty,
    const XrayL02GpuPrefix& g,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw,
    const std::vector<float>& lnfb,
    int L, int P, int C, int NH) {

    const auto& w = hw[2];
    std::vector<double> input = xray_to_double(g.input);
    std::vector<double> attproj, residual2, ln2, fch, gelu, fcproj, residual3;
    xray_cpu64_matmul(attproj, atty, w.attprojw, &w.attprojb, P, C, C);
    xray_cpu64_add(residual2, input, attproj);
    xray_cpu64_layernorm_rows(ln2, residual2, w.ln2w, w.ln2b, P, C);
    xray_cpu64_matmul(fch, ln2, w.fcw, &w.fcb, P, C, 4 * C);
    xray_cpu64_gelu(gelu, fch);
    xray_cpu64_matmul(fcproj, gelu, w.fcprojw, &w.fcprojb, P, 4 * C, C);
    xray_cpu64_add(residual3, residual2, fcproj);

    std::vector<double> residual = std::move(residual3);
    std::vector<double> a, q, y, ap, r2, n2, f, ge, fp, r3;
    for (int l = 3; l < L; ++l) {
        const auto& wl = hw[l];
        xray_cpu64_layernorm_rows(a, residual, wl.ln1w, wl.ln1b, P, C);
        xray_cpu64_matmul(q, a, wl.qkvw, &wl.qkvb, P, C, 3 * C);
        xray_cpu64_attention(y, q, P, C, NH);
        xray_cpu64_matmul(ap, y, wl.attprojw, &wl.attprojb, P, C, C);
        xray_cpu64_add(r2, residual, ap);
        xray_cpu64_layernorm_rows(n2, r2, wl.ln2w, wl.ln2b, P, C);
        xray_cpu64_matmul(f, n2, wl.fcw, &wl.fcb, P, C, 4 * C);
        xray_cpu64_gelu(ge, f);
        xray_cpu64_matmul(fp, ge, wl.fcprojw, &wl.fcprojb, P, 4 * C, C);
        xray_cpu64_add(r3, r2, fp);
        residual.swap(r3);
    }

    std::vector<double> target(residual.end() - C, residual.end());
    return xray_cpu64_layernorm_row_double(target, lnfw, lnfb);
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int target = argc > 3 ? atoi(argv[3]) : 1186;
    if (B <= 0 || T <= 0 || target < 0 || target >= B * T) return 2;

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
    const int HS = C / NH;
    const int V = model.config.vocab_size;
    const int b = target / T;
    const int t = target % T;
    const int P = t + 1;

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_margin);
    if (ref_winner == low_winner) {
        printf("[xray][decision-l02-attn] target has no natural execution disagreement\n");
        return 0;
    }

    XrayL02GpuPrefix g;
    xray_capture_l02_gpu_prefix(&g, &model, b, T, P);
    const XrayGpuAttentionReplay replay = xray_replay_l02_attention_exact(&model, b, T, P);

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C), gpu_logits(V);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(gpu_logits.data(),
                         model.acts.output + (size_t)target * model.config.padded_vocab_size,
                         V * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<double> qkv64 = xray_to_double(g.qkv);
    std::vector<double> raw64_from_qkv;
    xray_cpu64_raw_qk(raw64_from_qkv, qkv64, P, C, NH);

    const XrayRelDiff64F32 qk_mismatch =
        xray_rel_diff64_f32(raw64_from_qkv, replay.preatt);

    const std::vector<double> gpu_preatt64 = xray_to_double(replay.preatt);
    std::vector<double> prob64_from_gpu_qk;
    xray_cpu64_softmax_from_raw_qk(prob64_from_gpu_qk, gpu_preatt64, P, NH, HS);
    const XrayRelDiff64F32 softmax_mismatch =
        xray_rel_diff64_f32(prob64_from_gpu_qk, replay.att);

    const std::vector<double> gpu_att64 = xray_to_double(replay.att);
    std::vector<double> atty64_from_gpu_att;
    xray_cpu64_pv(atty64_from_gpu_att, gpu_att64, qkv64, P, C, NH);
    const XrayRelDiff64F32 pv_mismatch =
        xray_rel_diff64_f32(atty64_from_gpu_att, replay.atty);

    std::vector<double> prob64_from_cpu_qk;
    xray_cpu64_softmax_from_raw_qk(prob64_from_cpu_qk, raw64_from_qkv, P, NH, HS);
    std::vector<double> atty64_decomposed;
    xray_cpu64_pv(atty64_decomposed, prob64_from_cpu_qk, qkv64, P, C, NH);
    std::vector<double> atty64_monolithic;
    xray_cpu64_attention(atty64_monolithic, qkv64, P, C, NH);
    std::vector<float> atty64_monolithic_f(atty64_monolithic.size());
    for (size_t i = 0; i < atty64_monolithic.size(); ++i) {
        atty64_monolithic_f[i] = (float)atty64_monolithic[i];
    }
    const XrayRelDiff64F32 decomp_mismatch =
        xray_rel_diff64_f32(atty64_decomposed, atty64_monolithic_f);

    // Nested numerical-realization switch points inside L02 attention.
    const std::vector<double> lnf_qkv =
        xray_cpu64_finish_from_l02_stage(ST_QKV, g, hw, lnfw, lnfb, L, P, C, NH);
    const std::vector<double> atty64_from_gpu_qk = [&]() {
        std::vector<double> y;
        xray_cpu64_pv(y, prob64_from_gpu_qk, qkv64, P, C, NH);
        return y;
    }();
    const std::vector<double> lnf_qk =
        xray_cpu64_finish_l02_from_atty64(atty64_from_gpu_qk, g, hw, lnfw, lnfb,
                                          L, P, C, NH);
    const std::vector<double> lnf_softmax =
        xray_cpu64_finish_l02_from_atty64(atty64_from_gpu_att, g, hw, lnfw, lnfb,
                                          L, P, C, NH);
    const std::vector<double> lnf_pv =
        xray_cpu64_finish_from_l02_stage(ST_ATTY, g, hw, lnfw, lnfb, L, P, C, NH);

    const XrayReadoutStats s_qkv =
        xray_cpu64_classifier(lnf_qkv, wte, gpu_logits, V, C, ref_winner, low_winner);
    const XrayReadoutStats s_qk =
        xray_cpu64_classifier(lnf_qk, wte, gpu_logits, V, C, ref_winner, low_winner);
    const XrayReadoutStats s_softmax =
        xray_cpu64_classifier(lnf_softmax, wte, gpu_logits, V, C, ref_winner, low_winner);
    const XrayReadoutStats s_pv =
        xray_cpu64_classifier(lnf_pv, wte, gpu_logits, V, C, ref_winner, low_winner);

    printf("[xray][decision-l02-attn] natural GPU L02 attention path; switch points qkv -> qk -> softmax -> pv, then validated CPU64 remainder\n");
    printf("[xray][decision-l02-attn] target=%d b=%d t=%d ref=%d low=%d prefix=%d\n",
           target, b, t, ref_winner, low_winner, P);
    printf("[xray][decision-l02-attn-gpu-replay] att_unequal=%d/%zu att_max_abs=%.9e atty_unequal=%d/%zu atty_max_abs=%.9e exact=%d\n",
           replay.att_replay_unequal, replay.att.size(), replay.att_replay_max_abs,
           replay.atty_replay_unequal, replay.atty.size(), replay.atty_replay_max_abs,
           replay.att_replay_unequal == 0 && replay.atty_replay_unequal == 0);
    printf("[xray][decision-l02-attn-local-mismatch] qk_cpu64_vs_gpu_rel_l2=%.9e qk_max_abs=%.9e softmax_cpu64_from_gpu_qk_vs_gpu_rel_l2=%.9e softmax_max_abs=%.9e pv_cpu64_from_gpu_p_vs_gpu_rel_l2=%.9e pv_max_abs=%.9e\n",
           qk_mismatch.rel_l2, qk_mismatch.max_abs,
           softmax_mismatch.rel_l2, softmax_mismatch.max_abs,
           pv_mismatch.rel_l2, pv_mismatch.max_abs);
    printf("[xray][decision-l02-attn-cpu64-decomposition] decomposed_vs_monolithic_atty_rel_l2=%.9e max_abs=%.9e\n",
           decomp_mismatch.rel_l2, decomp_mismatch.max_abs);

    const char* names[] = {"qkv", "qk", "softmax", "pv"};
    const XrayReadoutStats stats[] = {s_qkv, s_qk, s_softmax, s_pv};
    int sign_changes = 0;
    int first_change = -1;
    for (int i = 0; i < 4; ++i) {
        const int sign = xray_sign64(stats[i].pair_margin);
        const double step = i ? stats[i].pair_margin - stats[i - 1].pair_margin : 0.0;
        if (i && sign != xray_sign64(stats[i - 1].pair_margin)) {
            ++sign_changes;
            if (first_change < 0) first_change = i;
        }
        printf("[xray][decision-l02-attn-point] stage=%-7s cpu64_top1=%d cpu64_runner=%d cpu64_pair=%+.9e sign=%+d switch_delta=%+.9e\n",
               names[i], stats[i].winner, stats[i].runner,
               stats[i].pair_margin, sign, step);
    }

    const int replay_exact = replay.att_replay_unequal == 0 && replay.atty_replay_unequal == 0;
    printf("[xray][decision-l02-attn-summary] gpu_attention_replay_exact=%d sign_changes=%d first_sign_change_stage=%s qkv_pair=%+.9e pv_pair=%+.9e\n",
           replay_exact, sign_changes,
           first_change >= 0 ? names[first_change] : "none",
           s_qkv.pair_margin, s_pv.pair_margin);
    printf("[xray][decision-l02-attn-summary] interpretation gate: local mismatch norms and nested switch deltas localize numerical realization dependence; they are not standalone causal contributions\n");

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
