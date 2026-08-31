#pragma push_macro("main")
#undef main
#define main xray_decision_numerical_fate_depth_embedded_main
#include "decision_numerical_fate_depth_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstring>

struct XrayCaseFamily {
    const char* name;
    XrayComponent component;
};

static const XrayCaseFamily xray_case_families[] = {
    {"qkv", XRAY_QKV},
    {"attproj", XRAY_ATTPROJ},
    {"fc", XRAY_FC},
    {"fcproj", XRAY_FCPROJ},
};

static bool xray_family_selected(const char* requested, const char* name) {
    return std::strcmp(requested, "all") == 0 || std::strcmp(requested, name) == 0;
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 5;
    const int target = argc > 4 ? atoi(argv[4]) : 134;
    const char* requested_family = argc > 5 ? argv[5] : "all";

    if (B <= 0 || T <= 0 || batch_index < 0 || target < 0 || target >= B * T) {
        fprintf(stderr,
                "usage: %s [B=4] [T=512] [batch_index=5] [batch_local_token=134] [family=all|qkv|attproj|fc|fcproj]\n",
                argv[0]);
        return 2;
    }

    bool known_family = std::strcmp(requested_family, "all") == 0;
    for (const auto& f : xray_case_families) {
        known_family |= std::strcmp(requested_family, f.name) == 0;
    }
    if (!known_family) {
        fprintf(stderr, "unknown family '%s'; expected all|qkv|attproj|fc|fcproj\n",
                requested_family);
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
    for (int i = 0; i <= batch_index; ++i) {
        dataloader_next_batch(&loader);
    }

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

    // Reference execution defines the baseline decision.  This is the same
    // batch indexing convention used by numerical_fate_landscape_probe.cu.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_top2_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_top2_margin);

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         C * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[xray][decision-case-depth] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d requested_family=%s\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, requested_family);
    printf("[xray][decision-case-depth] ref_top1=%d ref_runner=%d ref_top2_margin=%+.9e\n",
           ref_winner, ref_runner, (double)ref_top2_margin);
    printf("[xray][decision-case-depth] each family is a natural L00 single-GEMM TF32 trajectory; every residual3 checkpoint is replayed exactly on GPU and independently evaluated through the validated CPU64 suffix\n");

    int processed = 0;
    int disagreements = 0;
    int global_replay_exact = 1;
    int first_alt_winner = -1;
    int common_alt_winner = 1;
    int all_alt_are_ref_runner = 1;

    const auto global_t0 = std::chrono::steady_clock::now();

    for (const auto& family : xray_case_families) {
        if (!xray_family_selected(requested_family, family.name)) continue;
        ++processed;

        cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        gpt2_forward_one_component_tf32(&model, loader.inputs, NULL,
                                        B, T, 0, family.component);
        cudaCheck(cudaDeviceSynchronize());

        int alt_winner = -1, alt_runner = -1;
        float alt_top2_margin = 0.0f;
        xray_top2_at(&model, target, &alt_winner, &alt_runner, &alt_top2_margin);

        if (first_alt_winner < 0) first_alt_winner = alt_winner;
        else common_alt_winner &= (alt_winner == first_alt_winner);
        all_alt_are_ref_runner &= (alt_winner == ref_runner);

        std::vector<float> alt_logits(V);
        cudaCheck(cudaMemcpy(alt_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        const double alt_pair =
            (double)alt_logits[ref_winner] - (double)alt_logits[alt_winner];

        printf("[xray][decision-case-depth-family] family=%-7s ref=%d ref_runner=%d alt=%d alt_runner=%d alt_top2_margin=%+.9e ref_minus_alt=%+.9e alt_is_ref_runner=%d disagreement=%d\n",
               family.name, ref_winner, ref_runner, alt_winner, alt_runner,
               (double)alt_top2_margin, alt_pair,
               alt_winner == ref_runner ? 1 : 0,
               alt_winner != ref_winner ? 1 : 0);

        if (alt_winner == ref_winner) {
            printf("[xray][decision-case-depth-family-summary] family=%s skipped=1 reason=no_final_decision_disagreement\n",
                   family.name);
            continue;
        }
        ++disagreements;

        // Preserve the exact natural residual3 trajectory before replay mutates activations.
        std::vector<float*> checkpoints(L, nullptr);
        for (int k = 0; k < L; ++k) {
            cudaCheck(cudaMalloc((void**)&checkpoints[k], state_bytes));
            cudaCheck(cudaMemcpy(checkpoints[k],
                                 model.acts.residual3 + (size_t)k * state_n,
                                 state_bytes, cudaMemcpyDeviceToDevice));
        }

        int family_replay_exact = 1;
        int first_cpu64_alt_decision_layer = -1;
        int first_cpu64_not_ref_layer = -1;
        int first_cpu64_pair_nonpositive_layer = -1;
        int cpu64_winner_changes = 0;
        int cpu64_pair_sign_changes = 0;
        int prev_cpu64_winner = -1;
        int prev_cpu64_pair_sign = 0;
        int last_cpu64_winner = -1;
        double last_cpu64_pair = std::numeric_limits<double>::quiet_NaN();

        const auto family_t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < L; ++k) {
            // Validity control: exact checkpoint + original GPU suffix must
            // reconstruct the natural alternate target logits bit-for-bit.
            xray_forward_from_residual3(&model, checkpoints[k], k, B, T);
            cudaCheck(cudaDeviceSynchronize());

            std::vector<float> replay_logits(V);
            cudaCheck(cudaMemcpy(replay_logits.data(),
                                 model.acts.output + (size_t)target * Vp,
                                 (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
            int replay_unequal = 0;
            double replay_max_abs = 0.0;
            xray_compare_gpu_logits(alt_logits, replay_logits,
                                    &replay_unequal, &replay_max_abs);
            family_replay_exact &= (replay_unequal == 0);

            const XrayReadoutStats gpu_stats =
                xray_stats_from_gpu_logits(replay_logits, ref_winner, alt_winner);

            // Same exact checkpoint, but with the remaining model suffix
            // reevaluated in CPU double.  This is a direct fate evaluation,
            // not a finite-difference attribution.
            const std::vector<float> prefix =
                xray_copy_state_prefix(checkpoints[k], b, T, P, C);
            const XrayCpu64DepthResult cpu =
                xray_cpu64_run_from_checkpoint(prefix, k, hw, lnfw, lnfb,
                                               L, P, C, NH);
            const XrayReadoutStats cpu_stats =
                xray_cpu64_classifier(cpu.lnf, wte, replay_logits,
                                      V, C, ref_winner, alt_winner);

            const int cpu_pair_sign = xray_sign64(cpu_stats.pair_margin);
            const int cpu_matches_alt = cpu_stats.winner == alt_winner;
            const int cpu_is_ref = cpu_stats.winner == ref_winner;
            if (first_cpu64_alt_decision_layer < 0 && cpu_matches_alt) {
                first_cpu64_alt_decision_layer = k;
            }
            if (first_cpu64_not_ref_layer < 0 && !cpu_is_ref) {
                first_cpu64_not_ref_layer = k;
            }
            if (first_cpu64_pair_nonpositive_layer < 0 && cpu_stats.pair_margin <= 0.0) {
                first_cpu64_pair_nonpositive_layer = k;
            }
            if (k > 0 && cpu_stats.winner != prev_cpu64_winner) {
                ++cpu64_winner_changes;
            }
            if (k > 0 && cpu_pair_sign != prev_cpu64_pair_sign) {
                ++cpu64_pair_sign_changes;
            }

            printf("[xray][decision-case-depth-checkpoint] family=%-7s L=%02d gpu_top1=%d gpu_runner=%d gpu_pair=%+.9e cpu64_top1=%d cpu64_runner=%d cpu64_pair=%+.9e cpu64_matches_alt=%d cpu64_is_ref=%d pair_sign=%+d gpu_replay_unequal=%d/%d gpu_replay_max_abs=%.9e\n",
                   family.name, k,
                   gpu_stats.winner, gpu_stats.runner, gpu_stats.pair_margin,
                   cpu_stats.winner, cpu_stats.runner, cpu_stats.pair_margin,
                   cpu_matches_alt ? 1 : 0, cpu_is_ref ? 1 : 0,
                   cpu_pair_sign, replay_unequal, V, replay_max_abs);

            prev_cpu64_winner = cpu_stats.winner;
            prev_cpu64_pair_sign = cpu_pair_sign;
            last_cpu64_winner = cpu_stats.winner;
            last_cpu64_pair = cpu_stats.pair_margin;
        }
        const auto family_t1 = std::chrono::steady_clock::now();
        const double family_ms =
            std::chrono::duration<double, std::milli>(family_t1 - family_t0).count();

        global_replay_exact &= family_replay_exact;
        printf("[xray][decision-case-depth-family-summary] family=%-7s replay_exact=%d alt=%d alt_is_ref_runner=%d first_cpu64_alt_decision_layer=%d first_cpu64_not_ref_layer=%d first_cpu64_pair_nonpositive_layer=%d cpu64_winner_changes=%d cpu64_pair_sign_changes=%d final_cpu64_top1=%d final_cpu64_pair=%+.9e elapsed_ms=%.3f\n",
               family.name, family_replay_exact, alt_winner,
               alt_winner == ref_runner ? 1 : 0,
               first_cpu64_alt_decision_layer,
               first_cpu64_not_ref_layer,
               first_cpu64_pair_nonpositive_layer,
               cpu64_winner_changes, cpu64_pair_sign_changes,
               last_cpu64_winner, last_cpu64_pair, family_ms);

        for (float* p : checkpoints) {
            if (p) cudaCheck(cudaFree(p));
        }
    }

    const auto global_t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(global_t1 - global_t0).count();

    printf("[xray][decision-case-depth-summary] processed=%d disagreements=%d global_replay_exact=%d common_alt_winner=%d alt_winner=%d all_alt_are_ref_runner=%d elapsed_ms=%.3f\n",
           processed, disagreements, global_replay_exact,
           common_alt_winner, first_alt_winner, all_alt_are_ref_runner,
           elapsed_ms);
    printf("[xray][decision-case-depth-summary] interpretation gate: compare direct CPU64 decision trajectories across perturbation families only after replay_exact=1; convergence/divergence of fate depth is descriptive topology and does not by itself identify an operator-level mechanism\n");

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
