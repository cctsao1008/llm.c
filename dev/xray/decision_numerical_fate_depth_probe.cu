#pragma push_macro("main")
#undef main
#define main xray_cpu64_causal_suffix_embedded_main
#include "decision_cpu64_causal_suffix_audit_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

struct XrayCpu64DepthResult {
    std::vector<double> lnf;
};

static XrayCpu64DepthResult xray_cpu64_run_from_checkpoint(
    const std::vector<float>& checkpoint_prefix,
    int checkpoint_layer,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw,
    const std::vector<float>& lnfb,
    int L, int P, int C, int NH) {

    std::vector<double> residual(checkpoint_prefix.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)checkpoint_prefix.size(); ++i) {
        residual[(size_t)i] = (double)checkpoint_prefix[(size_t)i];
    }

    std::vector<double> ln1, qkv, atty, attproj, residual2;
    std::vector<double> ln2, fch, gelu, fcproj, residual3;

    for (int l = checkpoint_layer + 1; l < L; ++l) {
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
        residual.swap(residual3);
    }

    std::vector<double> final_row(C);
    const double* src = residual.data() + (size_t)(P - 1) * C;
    std::copy(src, src + C, final_row.begin());

    XrayCpu64DepthResult out;
    out.lnf = xray_cpu64_layernorm_row_double(final_row, lnfw, lnfb);
    return out;
}

static void xray_compare_gpu_logits(const std::vector<float>& ref,
                                    const std::vector<float>& cur,
                                    int* unequal, double* max_abs) {
    int neq = 0;
    double mx = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != cur[i]) ++neq;
        mx = std::max(mx, std::fabs((double)cur[i] - (double)ref[i]));
    }
    *unequal = neq;
    *max_abs = mx;
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
    const int Vp = model.config.padded_vocab_size;
    const int b = target / T;
    const int t = target % T;
    const int P = t + 1;
    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    // Reference execution identifies the competing token pair only.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_top2_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_top2_margin);

    // Natural alternate execution: exactly L00 fcproj switches to TF32.  All
    // checkpoints below are states actually realized on this execution path;
    // no hybrid patching is used in this probe.
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_top2_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_top2_margin);
    if (ref_winner == low_winner) {
        printf("[xray][decision-depth] target=%d has no natural execution disagreement ref=%d low=%d\n",
               target, ref_winner, low_winner);
        return 0;
    }

    std::vector<float> low_logits(V);
    cudaCheck(cudaMemcpy(low_logits.data(),
                         model.acts.output + (size_t)target * Vp,
                         (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    const double low_pair = (double)low_logits[ref_winner] - (double)low_logits[low_winner];

    // Preserve every exact natural GPU checkpoint before any replay overwrites activations.
    std::vector<float*> checkpoints(L, nullptr);
    for (int k = 0; k < L; ++k) {
        cudaCheck(cudaMalloc((void**)&checkpoints[k], state_bytes));
        cudaCheck(cudaMemcpy(checkpoints[k],
                             model.acts.residual3 + (size_t)k * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
    }

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         C * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[xray][decision-depth] natural alternate trajectory only: L00 fcproj TF32, then original GPU suffix\n");
    printf("[xray][decision-depth] target=%d b=%d t=%d ref=%d low=%d low_pair=%+.9e prefix=%d\n",
           target, b, t, ref_winner, low_winner, low_pair, P);
    printf("[xray][decision-depth] for each exact GPU residual3 checkpoint hk, compare the original GPU suffix and validated CPU64 suffix from the same hk; no multi-state subtraction is used\n");

    int all_gpu_replay_exact = 1;
    int cpu64_sign_changes = 0;
    int first_cpu64_match_gpu_layer = -1;
    int first_cpu64_sign_change_layer = -1;
    int prev_cpu64_sign = 0;
    double prev_cpu64_pair = 0.0;
    const int gpu_sign = xray_sign64(low_pair);

    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < L; ++k) {
        // Control: replay the exact checkpoint through the original GPU suffix.
        xray_forward_from_residual3(&model, checkpoints[k], k, B, T);
        cudaCheck(cudaDeviceSynchronize());

        std::vector<float> replay_logits(V);
        cudaCheck(cudaMemcpy(replay_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int replay_unequal = 0;
        double replay_max_abs = 0.0;
        xray_compare_gpu_logits(low_logits, replay_logits, &replay_unequal, &replay_max_abs);
        all_gpu_replay_exact &= (replay_unequal == 0);

        const XrayReadoutStats gpu_stats =
            xray_stats_from_gpu_logits(replay_logits, ref_winner, low_winner);

        // Same exact checkpoint, but the remaining suffix is reevaluated in CPU double.
        const std::vector<float> prefix = xray_copy_state_prefix(checkpoints[k], b, T, P, C);
        const XrayCpu64DepthResult cpu =
            xray_cpu64_run_from_checkpoint(prefix, k, hw, lnfw, lnfb, L, P, C, NH);
        const XrayReadoutStats cpu_stats =
            xray_cpu64_classifier(cpu.lnf, wte, replay_logits,
                                  V, C, ref_winner, low_winner);

        const double delta = cpu_stats.pair_margin - gpu_stats.pair_margin;
        const int cpu_sign = xray_sign64(cpu_stats.pair_margin);
        const int sign_agree = cpu_sign == xray_sign64(gpu_stats.pair_margin);
        if (first_cpu64_match_gpu_layer < 0 && cpu_sign == gpu_sign) {
            first_cpu64_match_gpu_layer = k;
        }
        if (k > 0 && cpu_sign != prev_cpu64_sign) {
            ++cpu64_sign_changes;
            if (first_cpu64_sign_change_layer < 0) first_cpu64_sign_change_layer = k;
        }

        printf("[xray][decision-depth-checkpoint] L=%02d gpu_top1=%d gpu_runner=%d gpu_pair=%+.9e cpu64_top1=%d cpu64_runner=%d cpu64_pair=%+.9e cpu64_minus_gpu=%+.9e sign_agree=%d gpu_replay_unequal=%d/%d gpu_replay_max_abs=%.9e",
               k,
               gpu_stats.winner, gpu_stats.runner, gpu_stats.pair_margin,
               cpu_stats.winner, cpu_stats.runner, cpu_stats.pair_margin,
               delta, sign_agree, replay_unequal, V, replay_max_abs);
        if (k > 0) {
            printf(" cpu64_pair_step=%+.9e", cpu_stats.pair_margin - prev_cpu64_pair);
        }
        printf("\n");

        prev_cpu64_sign = cpu_sign;
        prev_cpu64_pair = cpu_stats.pair_margin;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("[xray][decision-depth-summary] all_gpu_replay_exact=%d gpu_pair_sign=%d first_cpu64_match_gpu_layer=%d cpu64_sign_changes=%d first_cpu64_sign_change_layer=%d elapsed_ms=%.3f\n",
           all_gpu_replay_exact, gpu_sign, first_cpu64_match_gpu_layer,
           cpu64_sign_changes, first_cpu64_sign_change_layer, elapsed_ms);
    printf("[xray][decision-depth-summary] interpretation gate: this probe reports direct decision fate from identical checkpoints; a sign change locates depth dependence but does not by itself identify an operator-level mechanism\n");

    for (float* p : checkpoints) {
        if (p) cudaCheck(cudaFree(p));
    }
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
