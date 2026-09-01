#pragma push_macro("main")
#undef main
#define main xray_numerical_fate_trajectory_embedded_main
#include "numerical_fate_trajectory_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static char xray_exec_agree_symbol(int ref_agree, int alt_agree) {
    if (ref_agree && alt_agree) return '.';
    if (ref_agree && !alt_agree) return 'C'; // disagreement created on alternate state
    if (!ref_agree && alt_agree) return 'R'; // disagreement removed on alternate state
    return 'B';                              // both states disagree across executions
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 0;
    const int target = argc > 4 ? atoi(argv[4]) : 42;
    const char* family_name = argc > 5 ? argv[5] : "attproj";

    if (B <= 0 || T <= 0 || batch_index < 0 || target < 0 || target >= B * T) {
        fprintf(stderr,
                "usage: %s [B=4] [T=512] [batch_index=0] [batch_local_token=42] [family=qkv|attproj|fc|fcproj]\n",
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

    // Reference natural execution, repeated exactly. Capture every exact residual3
    // checkpoint from the repeated run before any alternate execution mutates activations.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);
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

    std::vector<float*> ref_checkpoints(L, nullptr);
    for (int k = 0; k < L; ++k) {
        cudaCheck(cudaMalloc((void**)&ref_checkpoints[k], state_bytes));
        cudaCheck(cudaMemcpy(ref_checkpoints[k],
                             model.acts.residual3 + (size_t)k * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
    }

    // Alternate natural execution: exactly one L00 GEMM family switches to TF32.
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL,
                                    B, T, 0, family->component);
    cudaCheck(cudaDeviceSynchronize());
    int alt_winner = -1, alt_runner = -1;
    float alt_margin = 0.0f;
    xray_top2_at(&model, target, &alt_winner, &alt_runner, &alt_margin);
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

    if (alt_winner == ref_winner) {
        fprintf(stderr,
                "[xray][crossed-state-exec] expected a natural final disagreement but ref=%d alt=%d scan_index=%lld family=%s\n",
                ref_winner, alt_winner, scan_index, family->name);
        for (float* p : ref_checkpoints) if (p) cudaCheck(cudaFree(p));
        dataloader_free(&loader);
        gpt2_free(&model);
        cublasCheck(cublasDestroy(cublas_handle));
        return 4;
    }

    const int competitor = alt_winner;

    std::vector<float*> alt_checkpoints(L, nullptr);
    for (int k = 0; k < L; ++k) {
        cudaCheck(cudaMalloc((void**)&alt_checkpoints[k], state_bytes));
        cudaCheck(cudaMemcpy(alt_checkpoints[k],
                             model.acts.residual3 + (size_t)k * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
    }

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[xray][crossed-state-exec] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d family=%s\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, family->name);
    printf("[xray][crossed-state-exec] ref=%d competitor=%d alt=%d ref_top2_margin=%+.9e alt_top2_margin=%+.9e\n",
           ref_winner, competitor, alt_winner,
           (double)ref_margin, (double)alt_margin);
    printf("[xray][crossed-state-exec] each layer evaluates the full 2x2 square: exact reference/alternate residual3 state x original GPU/validated CPU64 complete suffix. Raw cell margins and winners are primary observables; no second-difference interaction scalar is formed.\n");
    printf("[xray][crossed-state-exec] cross pattern legend: .=both states execution-agree, C=execution disagreement created on alternate state, R=removed on alternate state, B=both states execution-disagree.\n");

    int all_ref_gpu_replay_exact = 1;
    int all_alt_gpu_replay_exact = 1;
    int created_winner_layers = 0;
    int removed_winner_layers = 0;
    int both_disagree_winner_layers = 0;

    std::string ref_gpu_pair_topology, ref_cpu64_pair_topology;
    std::string alt_gpu_pair_topology, alt_cpu64_pair_topology;
    std::string ref_gpu_winner_topology, ref_cpu64_winner_topology;
    std::string alt_gpu_winner_topology, alt_cpu64_winner_topology;
    std::string cross_winner_pattern, cross_pair_pattern;
    ref_gpu_pair_topology.reserve(L);
    ref_cpu64_pair_topology.reserve(L);
    alt_gpu_pair_topology.reserve(L);
    alt_cpu64_pair_topology.reserve(L);
    ref_gpu_winner_topology.reserve(L);
    ref_cpu64_winner_topology.reserve(L);
    alt_gpu_winner_topology.reserve(L);
    alt_cpu64_winner_topology.reserve(L);
    cross_winner_pattern.reserve(L);
    cross_pair_pattern.reserve(L);

    for (int k = 0; k < L; ++k) {
        // Cell (h_ref, E_gpu): exact reference checkpoint through original GPU suffix.
        xray_forward_from_residual3(&model, ref_checkpoints[k], k, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<float> ref_gpu_logits(V);
        cudaCheck(cudaMemcpy(ref_gpu_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int ref_replay_unequal = 0;
        double ref_replay_max_abs = 0.0;
        xray_compare_gpu_logits(ref_repeat_logits, ref_gpu_logits,
                                &ref_replay_unequal, &ref_replay_max_abs);
        all_ref_gpu_replay_exact &= (ref_replay_unequal == 0);
        const XrayReadoutStats ref_gpu =
            xray_stats_from_gpu_logits(ref_gpu_logits, ref_winner, competitor);

        // Cell (h_alt, E_gpu): exact alternate checkpoint through original GPU suffix.
        xray_forward_from_residual3(&model, alt_checkpoints[k], k, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<float> alt_gpu_logits(V);
        cudaCheck(cudaMemcpy(alt_gpu_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int alt_replay_unequal = 0;
        double alt_replay_max_abs = 0.0;
        xray_compare_gpu_logits(alt_repeat_logits, alt_gpu_logits,
                                &alt_replay_unequal, &alt_replay_max_abs);
        all_alt_gpu_replay_exact &= (alt_replay_unequal == 0);
        const XrayReadoutStats alt_gpu =
            xray_stats_from_gpu_logits(alt_gpu_logits, ref_winner, competitor);

        // Cell (h_ref, E_cpu64): same exact reference state through validated CPU64 suffix.
        const std::vector<float> ref_prefix =
            xray_copy_state_prefix(ref_checkpoints[k], b, T, P, C);
        const XrayCpu64DepthResult ref_cpu_state =
            xray_cpu64_run_from_checkpoint(ref_prefix, k, hw, lnfw, lnfb,
                                           L, P, C, NH);
        const XrayReadoutStats ref_cpu64 =
            xray_cpu64_classifier(ref_cpu_state.lnf, wte, ref_gpu_logits,
                                  V, C, ref_winner, competitor);

        // Cell (h_alt, E_cpu64): same exact alternate state through validated CPU64 suffix.
        const std::vector<float> alt_prefix =
            xray_copy_state_prefix(alt_checkpoints[k], b, T, P, C);
        const XrayCpu64DepthResult alt_cpu_state =
            xray_cpu64_run_from_checkpoint(alt_prefix, k, hw, lnfw, lnfb,
                                           L, P, C, NH);
        const XrayReadoutStats alt_cpu64 =
            xray_cpu64_classifier(alt_cpu_state.lnf, wte, alt_gpu_logits,
                                  V, C, ref_winner, competitor);

        const int ref_winner_agree = ref_gpu.winner == ref_cpu64.winner;
        const int alt_winner_agree = alt_gpu.winner == alt_cpu64.winner;
        const int ref_pair_agree =
            xray_sign64(ref_gpu.pair_margin) == xray_sign64(ref_cpu64.pair_margin);
        const int alt_pair_agree =
            xray_sign64(alt_gpu.pair_margin) == xray_sign64(alt_cpu64.pair_margin);

        const char cross_winner = xray_exec_agree_symbol(ref_winner_agree, alt_winner_agree);
        const char cross_pair = xray_exec_agree_symbol(ref_pair_agree, alt_pair_agree);
        cross_winner_pattern.push_back(cross_winner);
        cross_pair_pattern.push_back(cross_pair);
        if (cross_winner == 'C') ++created_winner_layers;
        else if (cross_winner == 'R') ++removed_winner_layers;
        else if (cross_winner == 'B') ++both_disagree_winner_layers;

        ref_gpu_pair_topology.push_back(xray_pair_symbol(ref_gpu.pair_margin));
        ref_cpu64_pair_topology.push_back(xray_pair_symbol(ref_cpu64.pair_margin));
        alt_gpu_pair_topology.push_back(xray_pair_symbol(alt_gpu.pair_margin));
        alt_cpu64_pair_topology.push_back(xray_pair_symbol(alt_cpu64.pair_margin));
        ref_gpu_winner_topology.push_back(xray_winner_symbol(ref_gpu.winner, ref_winner, competitor));
        ref_cpu64_winner_topology.push_back(xray_winner_symbol(ref_cpu64.winner, ref_winner, competitor));
        alt_gpu_winner_topology.push_back(xray_winner_symbol(alt_gpu.winner, ref_winner, competitor));
        alt_cpu64_winner_topology.push_back(xray_winner_symbol(alt_cpu64.winner, ref_winner, competitor));

        printf("[xray][crossed-state-exec-point] family=%-7s L=%02d "
               "ref_gpu_pair=%+.9e ref_gpu_winner=%d "
               "ref_cpu64_pair=%+.9e ref_cpu64_winner=%d "
               "alt_gpu_pair=%+.9e alt_gpu_winner=%d "
               "alt_cpu64_pair=%+.9e alt_cpu64_winner=%d "
               "ref_exec_winner_agree=%d alt_exec_winner_agree=%d "
               "ref_exec_pair_agree=%d alt_exec_pair_agree=%d "
               "cross_winner=%c cross_pair=%c "
               "ref_gpu_replay_unequal=%d/%d ref_gpu_replay_max_abs=%.9e "
               "alt_gpu_replay_unequal=%d/%d alt_gpu_replay_max_abs=%.9e\n",
               family->name, k,
               ref_gpu.pair_margin, ref_gpu.winner,
               ref_cpu64.pair_margin, ref_cpu64.winner,
               alt_gpu.pair_margin, alt_gpu.winner,
               alt_cpu64.pair_margin, alt_cpu64.winner,
               ref_winner_agree, alt_winner_agree,
               ref_pair_agree, alt_pair_agree,
               cross_winner, cross_pair,
               ref_replay_unequal, V, ref_replay_max_abs,
               alt_replay_unequal, V, alt_replay_max_abs);
    }

    const int repeat_valid =
        (ref_repeat_unequal == 0) && (alt_repeat_unequal == 0);
    const int case_valid =
        repeat_valid && all_ref_gpu_replay_exact && all_alt_gpu_replay_exact;

    printf("[xray][crossed-state-exec-validity] ref_repeat_exact=%d alt_repeat_exact=%d all_ref_gpu_replay_exact=%d all_alt_gpu_replay_exact=%d case_valid=%d\n",
           ref_repeat_unequal == 0, alt_repeat_unequal == 0,
           all_ref_gpu_replay_exact, all_alt_gpu_replay_exact, case_valid);
    printf("[xray][crossed-state-exec-summary] scan_index=%lld family=%s ref=%d competitor=%d "
           "ref_gpu_pair_topology=%s ref_cpu64_pair_topology=%s "
           "alt_gpu_pair_topology=%s alt_cpu64_pair_topology=%s "
           "ref_gpu_winner_topology=%s ref_cpu64_winner_topology=%s "
           "alt_gpu_winner_topology=%s alt_cpu64_winner_topology=%s "
           "cross_winner_pattern=%s cross_pair_pattern=%s "
           "created_winner_layers=%d removed_winner_layers=%d both_disagree_winner_layers=%d case_valid=%d\n",
           scan_index, family->name, ref_winner, competitor,
           ref_gpu_pair_topology.c_str(), ref_cpu64_pair_topology.c_str(),
           alt_gpu_pair_topology.c_str(), alt_cpu64_pair_topology.c_str(),
           ref_gpu_winner_topology.c_str(), ref_cpu64_winner_topology.c_str(),
           alt_gpu_winner_topology.c_str(), alt_cpu64_winner_topology.c_str(),
           cross_winner_pattern.c_str(), cross_pair_pattern.c_str(),
           created_winner_layers, removed_winner_layers,
           both_disagree_winner_layers, case_valid);
    printf("[xray][crossed-state-exec-summary] interpretation gate: C is the pre-registered positive signal for state-conditioned execution susceptibility (reference state execution-agrees while alternate state execution-disagrees). It is descriptive until matched margin/conditioning controls are checked. Raw 2x2 cells remain the primary evidence.\n");

    for (float* p : ref_checkpoints) if (p) cudaCheck(cudaFree(p));
    for (float* p : alt_checkpoints) if (p) cudaCheck(cudaFree(p));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return case_valid ? 0 : 5;
}
