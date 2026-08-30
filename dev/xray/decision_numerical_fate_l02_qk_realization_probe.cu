#pragma push_macro("main")
#undef main
#define main xray_l02_attn_substage_embedded_main
#include "decision_numerical_fate_l02_attention_substage_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct XrayRelDiff64 {
    double rel_l2;
    double max_abs;
};

static XrayRelDiff64 xray_rel_diff64(const std::vector<double>& a,
                                     const std::vector<double>& b) {
    long double a2 = 0.0L, d2 = 0.0L;
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const long double av = (long double)a[i];
        const long double d = (long double)b[i] - av;
        a2 += av * av;
        d2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
    }
    return XrayRelDiff64{
        a2 > 0.0L ? std::sqrt((double)(d2 / a2)) : std::sqrt((double)d2),
        mx
    };
}

// IEEE FP32 -> TF32-style mantissa rasterization, round-to-nearest-even.
// NVIDIA documents TF32 input conversion as RNE. Inf/NaN are preserved.
static float xray_tf32_rne(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u) return x;
    const uint32_t lsb = (u >> 13) & 1u;
    u += 0x00000fffu + lsb;
    u &= 0xffffe000u;
    float y = 0.0f;
    std::memcpy(&y, &u, sizeof(y));
    return y;
}

static std::vector<double> xray_qkv_variant(const std::vector<float>& qkv,
                                             int P, int C,
                                             bool round_q, bool round_k) {
    std::vector<double> out(qkv.size());
    for (int t = 0; t < P; ++t) {
        const size_t row = (size_t)t * 3 * C;
        for (int c = 0; c < C; ++c) {
            const float q = qkv[row + c];
            const float k = qkv[row + C + c];
            out[row + c] = (double)(round_q ? xray_tf32_rne(q) : q);
            out[row + C + c] = (double)(round_k ? xray_tf32_rne(k) : k);
            out[row + 2 * C + c] = (double)qkv[row + 2 * C + c];
        }
    }
    return out;
}

static std::vector<float> xray_gpu_qk_with_math_mode(GPT2* model,
                                                      int b, int T, int P,
                                                      cublasMath_t mode) {
    const int B = model->batch_size;
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    const int HS = C / NH;
    const int l = 2;
    const size_t BTC = (size_t)B * T * C;
    const size_t BHTT = (size_t)B * NH * T * T;
    const size_t layer_qkvr = (size_t)B * T * 3 * C;

    float* qkvr = model->acts.qkvr + (size_t)l * layer_qkvr;
    float* q = qkvr + 0 * BTC;
    float* k = qkvr + 1 * BTC;

    float* d_preatt = nullptr;
    cudaCheck(cudaMalloc((void**)&d_preatt, BHTT * sizeof(float)));

    cublasCheck(cublasSetMathMode(cublas_handle, mode));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, HS,
        &alpha,
        k, HS, T * HS,
        q, HS, T * HS,
        &beta,
        d_preatt, T, T * T,
        B * NH));
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> out = xray_copy_att_batch_prefix(d_preatt, b, NH, T, P);
    cudaCheck(cudaFree(d_preatt));
    return out;
}

struct XrayQkVariantResult {
    const char* name;
    std::vector<double> raw;
    std::vector<double> prob;
    XrayReadoutStats stats;
    XrayRelDiff64 raw_vs_exact;
    XrayRelDiff64 prob_vs_exact;
    XrayRelDiff64F32 raw_vs_gpu;
};

static XrayQkVariantResult xray_eval_qk_variant(
    const char* name,
    const std::vector<double>& raw,
    const std::vector<double>& raw_exact,
    const std::vector<double>& prob_exact,
    const std::vector<float>& raw_gpu,
    const std::vector<double>& qkv_exact,
    const XrayL02GpuPrefix& g,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw,
    const std::vector<float>& lnfb,
    const std::vector<float>& wte,
    const std::vector<float>& gpu_logits,
    int L, int P, int C, int NH, int V,
    int ref_winner, int low_winner) {

    const int HS = C / NH;
    std::vector<double> prob;
    xray_cpu64_softmax_from_raw_qk(prob, raw, P, NH, HS);
    std::vector<double> atty;
    xray_cpu64_pv(atty, prob, qkv_exact, P, C, NH);
    const std::vector<double> lnf =
        xray_cpu64_finish_l02_from_atty64(atty, g, hw, lnfw, lnfb, L, P, C, NH);
    const XrayReadoutStats stats =
        xray_cpu64_classifier(lnf, wte, gpu_logits, V, C, ref_winner, low_winner);

    XrayQkVariantResult r;
    r.name = name;
    r.raw = raw;
    r.prob = std::move(prob);
    r.stats = stats;
    r.raw_vs_exact = xray_rel_diff64(raw_exact, r.raw);
    r.prob_vs_exact = xray_rel_diff64(prob_exact, r.prob);
    r.raw_vs_gpu = xray_rel_diff64_f32(r.raw, raw_gpu);
    return r;
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
        printf("[xray][decision-l02-qk-realization] target has no natural execution disagreement\n");
        return 0;
    }

    XrayL02GpuPrefix g;
    xray_capture_l02_gpu_prefix(&g, &model, b, T, P);
    const XrayGpuAttentionReplay replay = xray_replay_l02_attention_exact(&model, b, T, P);

    // Same full-T GPU Q/K state, but reevaluate QK with pedantic math.
    const std::vector<float> qk_pedantic_f =
        xray_gpu_qk_with_math_mode(&model, b, T, P, CUBLAS_PEDANTIC_MATH);
    // Restore the natural xray setting before any later GPU work/free paths.
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

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

    const std::vector<double> qkv_exact = xray_qkv_variant(g.qkv, P, C, false, false);
    const std::vector<double> qkv_q_tf32 = xray_qkv_variant(g.qkv, P, C, true, false);
    const std::vector<double> qkv_k_tf32 = xray_qkv_variant(g.qkv, P, C, false, true);
    const std::vector<double> qkv_qk_tf32 = xray_qkv_variant(g.qkv, P, C, true, true);

    std::vector<double> raw_exact, raw_q_tf32, raw_k_tf32, raw_qk_tf32;
    xray_cpu64_raw_qk(raw_exact, qkv_exact, P, C, NH);
    xray_cpu64_raw_qk(raw_q_tf32, qkv_q_tf32, P, C, NH);
    xray_cpu64_raw_qk(raw_k_tf32, qkv_k_tf32, P, C, NH);
    xray_cpu64_raw_qk(raw_qk_tf32, qkv_qk_tf32, P, C, NH);

    const std::vector<double> raw_gpu = xray_to_double(replay.preatt);
    const std::vector<double> raw_pedantic = xray_to_double(qk_pedantic_f);

    std::vector<double> prob_exact;
    xray_cpu64_softmax_from_raw_qk(prob_exact, raw_exact, P, NH, C / NH);

    std::vector<XrayQkVariantResult> results;
    results.push_back(xray_eval_qk_variant(
        "cpu64", raw_exact, raw_exact, prob_exact, replay.preatt, qkv_exact, g, hw,
        lnfw, lnfb, wte, gpu_logits, L, P, C, NH, V, ref_winner, low_winner));
    results.push_back(xray_eval_qk_variant(
        "q-tf32", raw_q_tf32, raw_exact, prob_exact, replay.preatt, qkv_exact, g, hw,
        lnfw, lnfb, wte, gpu_logits, L, P, C, NH, V, ref_winner, low_winner));
    results.push_back(xray_eval_qk_variant(
        "k-tf32", raw_k_tf32, raw_exact, prob_exact, replay.preatt, qkv_exact, g, hw,
        lnfw, lnfb, wte, gpu_logits, L, P, C, NH, V, ref_winner, low_winner));
    results.push_back(xray_eval_qk_variant(
        "qk-tf32", raw_qk_tf32, raw_exact, prob_exact, replay.preatt, qkv_exact, g, hw,
        lnfw, lnfb, wte, gpu_logits, L, P, C, NH, V, ref_winner, low_winner));
    results.push_back(xray_eval_qk_variant(
        "pedantic", raw_pedantic, raw_exact, prob_exact, replay.preatt, qkv_exact, g, hw,
        lnfw, lnfb, wte, gpu_logits, L, P, C, NH, V, ref_winner, low_winner));
    results.push_back(xray_eval_qk_variant(
        "gpu-tf32", raw_gpu, raw_exact, prob_exact, replay.preatt, qkv_exact, g, hw,
        lnfw, lnfb, wte, gpu_logits, L, P, C, NH, V, ref_winner, low_winner));

    const XrayRelDiff64F32 pedantic_vs_gpu =
        xray_rel_diff64_f32(raw_pedantic, replay.preatt);
    const XrayRelDiff64F32 tf32soft_vs_gpu =
        xray_rel_diff64_f32(raw_qk_tf32, replay.preatt);
    const XrayRelDiff64 pedantic_vs_exact = xray_rel_diff64(raw_exact, raw_pedantic);
    const XrayRelDiff64 tf32soft_vs_exact = xray_rel_diff64(raw_exact, raw_qk_tf32);

    printf("[xray][decision-l02-qk-realization] exact same natural L02 Q/K; compare CPU64, software TF32-RNE rasterization, cuBLAS pedantic, and natural GPU TF32 QK; all downstream softmax/PV/suffix are CPU64\n");
    printf("[xray][decision-l02-qk-realization] target=%d b=%d t=%d ref=%d low=%d prefix=%d gpu_attention_replay_exact=%d\n",
           target, b, t, ref_winner, low_winner, P,
           replay.att_replay_unequal == 0 && replay.atty_replay_unequal == 0);
    printf("[xray][decision-l02-qk-realization-control] pedantic_vs_cpu64_rel_l2=%.9e max_abs=%.9e pedantic_vs_gpu_rel_l2=%.9e max_abs=%.9e software_tf32_vs_cpu64_rel_l2=%.9e max_abs=%.9e software_tf32_vs_gpu_rel_l2=%.9e max_abs=%.9e\n",
           pedantic_vs_exact.rel_l2, pedantic_vs_exact.max_abs,
           pedantic_vs_gpu.rel_l2, pedantic_vs_gpu.max_abs,
           tf32soft_vs_exact.rel_l2, tf32soft_vs_exact.max_abs,
           tf32soft_vs_gpu.rel_l2, tf32soft_vs_gpu.max_abs);

    for (const auto& r : results) {
        printf("[xray][decision-l02-qk-realization-point] mode=%-9s top1=%d runner=%d pair=%+.9e sign=%+d qk_rel_vs_cpu64=%.9e qk_max_vs_cpu64=%.9e qk_rel_vs_gpu=%.9e qk_max_vs_gpu=%.9e prob_rel_vs_cpu64=%.9e prob_max_vs_cpu64=%.9e\n",
               r.name, r.stats.winner, r.stats.runner, r.stats.pair_margin,
               xray_sign64(r.stats.pair_margin),
               r.raw_vs_exact.rel_l2, r.raw_vs_exact.max_abs,
               r.raw_vs_gpu.rel_l2, r.raw_vs_gpu.max_abs,
               r.prob_vs_exact.rel_l2, r.prob_vs_exact.max_abs);
    }

    const int cpu64_sign = xray_sign64(results.front().stats.pair_margin);
    const int tf32_sign = xray_sign64(results[3].stats.pair_margin);
    const int pedantic_sign = xray_sign64(results[4].stats.pair_margin);
    const int gpu_sign = xray_sign64(results.back().stats.pair_margin);
    printf("[xray][decision-l02-qk-realization-summary] cpu64_sign=%+d software_tf32_sign=%+d pedantic_sign=%+d gpu_tf32_sign=%+d software_tf32_matches_gpu_sign=%d pedantic_matches_cpu64_sign=%d\n",
           cpu64_sign, tf32_sign, pedantic_sign, gpu_sign,
           tf32_sign == gpu_sign, pedantic_sign == cpu64_sign);
    printf("[xray][decision-l02-qk-realization-summary] interpretation gate: a matching sign is only decision-level sufficiency under this fixed checkpoint; use QK distance controls to distinguish TF32 input rasterization from accumulation/algorithm effects\n");

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
