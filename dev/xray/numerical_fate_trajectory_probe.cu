#pragma push_macro("main")
#undef main
#define main xray_decision_numerical_fate_depth_embedded_main
#include "decision_numerical_fate_depth_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

struct XrayTrajectoryFamily {
    const char* name;
    XrayComponent component;
};

static const XrayTrajectoryFamily xray_trajectory_families[] = {
    {"qkv", XRAY_QKV},
    {"attproj", XRAY_ATTPROJ},
    {"fc", XRAY_FC},
    {"fcproj", XRAY_FCPROJ},
};

static const XrayTrajectoryFamily* xray_find_trajectory_family(const char* name) {
    for (const auto& f : xray_trajectory_families) {
        if (std::strcmp(name, f.name) == 0) return &f;
    }
    return nullptr;
}

static char xray_pair_symbol(double x) {
    if (x > 0.0) return '+';
    if (x < 0.0) return '-';
    return '0';
}

static char xray_winner_symbol(int winner, int ref_winner, int competitor) {
    if (winner == ref_winner) return 'R';
    if (winner == competitor) return 'C';
    return 'O';
}

static int xray_stable_winner_from(const std::vector<int>& winners, int wanted) {
    if (winners.empty() || winners.back() != wanted) return -1;
    int k = (int)winners.size() - 1;
    while (k > 0 && winners[(size_t)k - 1] == wanted) --k;
    return k;
}

static int xray_stable_sign_from(const std::vector<int>& signs, int wanted_sign) {
    if (signs.empty() || signs.back() != wanted_sign) return -1;
    int k = (int)signs.size() - 1;
    while (k > 0 && signs[(size_t)k - 1] == wanted_sign) --k;
    return k;
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 5;
    const int target = argc > 4 ? atoi(argv[4]) : 134;
    const char* family_name = argc > 5 ? argv[5] : "fc";

    if (B <= 0 || T <= 0 || batch_index < 0 || target < 0 || target >= B * T) {
        fprintf(stderr,
                "usage: %s [B=4] [T=512] [batch_index=5] [batch_local_token=134] [family=qkv|attproj|fc|fcproj]\n",
                argv[0]);
        return 2;
    }
    const XrayTrajectoryFamily* family = xray_find_trajectory_family(family_name);
    if (!family) {
        fprintf(stderr, "unknown family '%s'; expected qkv|attproj|fc|fcproj\n", family_name);
        return 2;
    }

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
                    B, T, 0, 1, 1);
    for (int i = 0; i <= batch_index; ++i) dataloader_next_batch(&loader);

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const int NH = model.config.num_heads;
    const int V = model.config.vocab_size;
    const int Vp = model.config.padded_vocab_size;
    const int b = target / T;
    const int t = target % T;
    const int P = t + 1;
    const long long scan_index = (long long)batch_index * B * T + target;
    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    // Reference execution defines the baseline winner/runner pair. Repeat it at
    // the exact target vocabulary row so every atlas case carries its own
    // determinism gate rather than relying only on the earlier population scan.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_top2_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_top2_margin);
    std::vector<float> ref_logits(V), ref_repeat_logits(V);
    cudaCheck(cudaMemcpy(ref_logits.data(),
                         model.acts.output + (size_t)target * Vp,
                         (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(ref_repeat_logits.data(),
                         model.acts.output + (size_t)target * Vp,
                         (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    int ref_repeat_unequal = 0;
    double ref_repeat_max_abs = 0.0;
    xray_compare_gpu_logits(ref_logits, ref_repeat_logits,
                            &ref_repeat_unequal, &ref_repeat_max_abs);

    // Natural execution perturbation: exactly one L00 GEMM family switches to
    // TF32 while the rest of the original GPU forward remains unchanged.
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL,
                                    B, T, 0, family->component);
    cudaCheck(cudaDeviceSynchronize());
    int alt_winner = -1, alt_runner = -1;
    float alt_top2_margin = 0.0f;
    xray_top2_at(&model, target, &alt_winner, &alt_runner, &alt_top2_margin);
    std::vector<float> alt_logits(V), alt_repeat_logits(V);
    cudaCheck(cudaMemcpy(alt_logits.data(),
                         model.acts.output + (size_t)target * Vp,
                         (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL,
                                    B, T, 0, family->component);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(alt_repeat_logits.data(),
                         model.acts.output + (size_t)target * Vp,
                         (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    int alt_repeat_unequal = 0;
    double alt_repeat_max_abs = 0.0;
    xray_compare_gpu_logits(alt_logits, alt_repeat_logits,
                            &alt_repeat_unequal, &alt_repeat_max_abs);

    // For flips, use the actual alternate winner. For non-flip controls, use
    // the reference runner. This is the same pair-selection rule as the
    // population fate-landscape survey and makes flip/control trajectories
    // directly comparable without inventing an artificial alternate token.
    const int final_flip = alt_winner != ref_winner;
    const int competitor = final_flip ? alt_winner : ref_runner;
    const char* competitor_source = final_flip ? "alt_winner" : "ref_runner";
    const double ref_pair =
        (double)ref_logits[ref_winner] - (double)ref_logits[competitor];
    const double alt_pair =
        (double)alt_logits[ref_winner] - (double)alt_logits[competitor];

    // The current activations are the repeated natural alternate trajectory.
    // Preserve all residual3 checkpoints before replay mutates them.
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

    printf("[xray][fate-trajectory] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d family=%s input_token=%d\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, family->name, loader.inputs[target]);
    printf("[xray][fate-trajectory] ref_top1=%d ref_runner=%d ref_top2_margin=%+.9e alt_top1=%d alt_runner=%d alt_top2_margin=%+.9e final_flip=%d competitor=%d competitor_source=%s ref_pair=%+.9e alt_pair=%+.9e\n",
           ref_winner, ref_runner, (double)ref_top2_margin,
           alt_winner, alt_runner, (double)alt_top2_margin,
           final_flip, competitor, competitor_source, ref_pair, alt_pair);
    printf("[xray][fate-trajectory-repeat] ref_target_repeat_exact=%d ref_repeat_unequal=%d/%d ref_repeat_max_abs=%.9e alt_target_repeat_exact=%d alt_repeat_unequal=%d/%d alt_repeat_max_abs=%.9e\n",
           ref_repeat_unequal == 0 ? 1 : 0, ref_repeat_unequal, V, ref_repeat_max_abs,
           alt_repeat_unequal == 0 ? 1 : 0, alt_repeat_unequal, V, alt_repeat_max_abs);
    printf("[xray][fate-trajectory] every point is the exact natural GPU residual3 checkpoint reevaluated through the validated CPU64 suffix; no finite-difference or additive attribution is used\n");

    int replay_exact = 1;
    std::vector<int> winners;
    std::vector<int> signs;
    std::vector<double> pairs;
    winners.reserve(L);
    signs.reserve(L);
    pairs.reserve(L);
    std::string pair_topology;
    std::string winner_topology;
    pair_topology.reserve(L);
    winner_topology.reserve(L);

    int winner_changes = 0;
    int pair_sign_changes = 0;
    int last_winner_change_layer = -1;
    int last_pair_sign_change_layer = -1;
    int min_abs_pair_layer = -1;
    double min_abs_pair = std::numeric_limits<double>::infinity();
    int max_abs_pair_step_layer = -1;
    double max_abs_pair_step = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < L; ++k) {
        xray_forward_from_residual3(&model, checkpoints[k], k, B, T);
        cudaCheck(cudaDeviceSynchronize());

        std::vector<float> replay_logits(V);
        cudaCheck(cudaMemcpy(replay_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int replay_unequal = 0;
        double replay_max_abs = 0.0;
        xray_compare_gpu_logits(alt_repeat_logits, replay_logits,
                                &replay_unequal, &replay_max_abs);
        replay_exact &= (replay_unequal == 0);

        const XrayReadoutStats gpu_stats =
            xray_stats_from_gpu_logits(replay_logits, ref_winner, competitor);

        const std::vector<float> prefix =
            xray_copy_state_prefix(checkpoints[k], b, T, P, C);
        const XrayCpu64DepthResult cpu =
            xray_cpu64_run_from_checkpoint(prefix, k, hw, lnfw, lnfb,
                                           L, P, C, NH);
        const XrayReadoutStats cpu_stats =
            xray_cpu64_classifier(cpu.lnf, wte, replay_logits,
                                  V, C, ref_winner, competitor);

        const int sign = xray_sign64(cpu_stats.pair_margin);
        winners.push_back(cpu_stats.winner);
        signs.push_back(sign);
        pairs.push_back(cpu_stats.pair_margin);
        pair_topology.push_back(xray_pair_symbol(cpu_stats.pair_margin));
        winner_topology.push_back(
            xray_winner_symbol(cpu_stats.winner, ref_winner, competitor));

        const double abs_pair = std::fabs(cpu_stats.pair_margin);
        if (abs_pair < min_abs_pair) {
            min_abs_pair = abs_pair;
            min_abs_pair_layer = k;
        }
        if (k > 0) {
            if (winners[(size_t)k] != winners[(size_t)k - 1]) {
                ++winner_changes;
                last_winner_change_layer = k;
            }
            if (signs[(size_t)k] != signs[(size_t)k - 1]) {
                ++pair_sign_changes;
                last_pair_sign_change_layer = k;
            }
            const double step = pairs[(size_t)k] - pairs[(size_t)k - 1];
            if (std::fabs(step) > max_abs_pair_step) {
                max_abs_pair_step = std::fabs(step);
                max_abs_pair_step_layer = k;
            }
        }

        printf("[xray][fate-trajectory-point] family=%-7s L=%02d gpu_top1=%d gpu_runner=%d gpu_pair=%+.9e cpu64_top1=%d cpu64_runner=%d cpu64_pair=%+.9e pair_sign=%+d winner_class=%c gpu_replay_unequal=%d/%d gpu_replay_max_abs=%.9e",
               family->name, k,
               gpu_stats.winner, gpu_stats.runner, gpu_stats.pair_margin,
               cpu_stats.winner, cpu_stats.runner, cpu_stats.pair_margin,
               sign,
               xray_winner_symbol(cpu_stats.winner, ref_winner, competitor),
               replay_unequal, V, replay_max_abs);
        if (k > 0) {
            printf(" cpu64_pair_step=%+.9e",
                   pairs[(size_t)k] - pairs[(size_t)k - 1]);
        }
        printf("\n");
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int stable_ref_from = xray_stable_winner_from(winners, ref_winner);
    const int stable_competitor_from = xray_stable_winner_from(winners, competitor);
    const int stable_pair_positive_from = xray_stable_sign_from(signs, +1);
    const int stable_pair_zero_from = xray_stable_sign_from(signs, 0);
    const int stable_pair_negative_from = xray_stable_sign_from(signs, -1);

    const int case_valid =
        (ref_repeat_unequal == 0) && (alt_repeat_unequal == 0) && replay_exact;

    printf("[xray][fate-trajectory-summary] family=%s scan_index=%lld final_flip=%d ref=%d competitor=%d competitor_source=%s pair_topology=%s winner_topology=%s winner_changes=%d pair_sign_changes=%d last_winner_change_layer=%d last_pair_sign_change_layer=%d stable_ref_from_layer=%d stable_competitor_from_layer=%d stable_pair_positive_from_layer=%d stable_pair_zero_from_layer=%d stable_pair_negative_from_layer=%d min_abs_pair=%.9e min_abs_pair_layer=%d max_abs_pair_step=%.9e max_abs_pair_step_layer=%d final_cpu64_top1=%d final_cpu64_pair=%+.9e repeat_valid=%d replay_exact=%d case_valid=%d elapsed_ms=%.3f\n",
           family->name, scan_index, final_flip, ref_winner, competitor,
           competitor_source, pair_topology.c_str(), winner_topology.c_str(),
           winner_changes, pair_sign_changes,
           last_winner_change_layer, last_pair_sign_change_layer,
           stable_ref_from, stable_competitor_from,
           stable_pair_positive_from, stable_pair_zero_from,
           stable_pair_negative_from,
           min_abs_pair, min_abs_pair_layer,
           max_abs_pair_step, max_abs_pair_step_layer,
           winners.empty() ? -1 : winners.back(),
           pairs.empty() ? std::numeric_limits<double>::quiet_NaN() : pairs.back(),
           (ref_repeat_unequal == 0 && alt_repeat_unequal == 0) ? 1 : 0,
           replay_exact, case_valid, elapsed_ms);
    printf("[xray][fate-trajectory-summary] interpretation gate: topology is a direct high-precision suffix fate observable. Adjacent pair steps describe transport across depth but are not operator attributions; controls use the reference runner as competitor and must not be described as hidden flips.\n");

    for (float* p : checkpoints) {
        if (p) cudaCheck(cudaFree(p));
    }
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
