#pragma push_macro("main")
#undef main
#define main xray_numerical_fate_trajectory_embedded_main
#include "numerical_fate_trajectory_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <cstring>
#include <vector>

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
    const int V = model.config.vocab_size;
    const int Vp = model.config.padded_vocab_size;
    const int b = target / T;
    const int t = target % T;
    const long long scan_index = (long long)batch_index * B * T + target;
    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    // Reference decision + target-row repeatability gate.
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

    // Natural alternate: exactly one L00 GEMM family switches to TF32.
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
                "[xray][terminal-readout] expected a natural final disagreement but ref=%d alt=%d for scan_index=%lld family=%s\n",
                ref_winner, alt_winner, scan_index, family->name);
        dataloader_free(&loader);
        gpt2_free(&model);
        cublasCheck(cublasDestroy(cublas_handle));
        return 4;
    }

    // Preserve the exact natural alternate terminal states before replay mutates
    // activations: full L-1 residual3 for exact GPU replay, target residual row,
    // and the exact GPU final-LayerNorm row.
    float* d_l11 = nullptr;
    cudaCheck(cudaMalloc((void**)&d_l11, state_bytes));
    cudaCheck(cudaMemcpy(d_l11,
                         model.acts.residual3 + (size_t)(L - 1) * state_n,
                         state_bytes, cudaMemcpyDeviceToDevice));

    std::vector<float> final_residual(C), gpu_lnf(C);
    cudaCheck(cudaMemcpy(final_residual.data(),
                         d_l11 + (size_t)target * C,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(gpu_lnf.data(),
                         model.acts.lnf + (size_t)target * C,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));

    // Exact terminal checkpoint + original GPU terminal suffix must reconstruct
    // the natural alternate target logits bit-for-bit.
    xray_forward_from_residual3(&model, d_l11, L - 1, B, T);
    cudaCheck(cudaDeviceSynchronize());
    std::vector<float> replay_logits(V);
    cudaCheck(cudaMemcpy(replay_logits.data(),
                         model.acts.output + (size_t)target * Vp,
                         (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    int replay_unequal = 0;
    double replay_max_abs = 0.0;
    xray_compare_gpu_logits(alt_repeat_logits, replay_logits,
                            &replay_unequal, &replay_max_abs);

    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));

    const XrayReadoutStats gpu =
        xray_stats_from_gpu_logits(replay_logits, ref_winner, alt_winner);

    // Switch only the tied vocabulary projection arithmetic to CPU double while
    // holding the exact GPU ln_f activation fixed.
    std::vector<double> gpu_lnf64(C);
    for (int c = 0; c < C; ++c) gpu_lnf64[c] = (double)gpu_lnf[c];
    const XrayReadoutStats cpu64_classifier =
        xray_cpu64_classifier(gpu_lnf64, wte, replay_logits,
                              V, C, ref_winner, alt_winner);

    // Now switch the whole terminal suffix from the exact L11 residual state:
    // final LayerNorm + tied vocabulary projection, both evaluated in double.
    const std::vector<double> cpu64_lnf =
        xray_cpu64_layernorm(final_residual, lnfw, lnfb);
    const XrayReadoutStats cpu64_terminal =
        xray_cpu64_classifier(cpu64_lnf, wte, replay_logits,
                              V, C, ref_winner, alt_winner);

    double lnf_max_abs = 0.0;
    const double lnf_rel_l2 =
        xray_rel_l2_cpu64_vs_gpu(cpu64_lnf, gpu_lnf, &lnf_max_abs);

    const int repeat_valid =
        (ref_repeat_unequal == 0) && (alt_repeat_unequal == 0);
    const int replay_exact = replay_unequal == 0;
    const int case_valid = repeat_valid && replay_exact;
    const int classifier_switch_restores_ref = cpu64_classifier.winner == ref_winner;
    const int terminal_cpu64_restores_ref = cpu64_terminal.winner == ref_winner;
    const int ln_added_changes_winner = cpu64_terminal.winner != cpu64_classifier.winner;

    printf("[xray][terminal-readout] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d family=%s\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, family->name);
    printf("[xray][terminal-readout] ref_top1=%d ref_runner=%d ref_margin=%+.9e alt_top1=%d alt_runner=%d alt_margin=%+.9e\n",
           ref_winner, ref_runner, (double)ref_margin,
           alt_winner, alt_runner, (double)alt_margin);
    printf("[xray][terminal-readout-validity] ref_repeat_exact=%d ref_repeat_unequal=%d/%d ref_repeat_max_abs=%.9e alt_repeat_exact=%d alt_repeat_unequal=%d/%d alt_repeat_max_abs=%.9e l11_gpu_replay_exact=%d replay_unequal=%d/%d replay_max_abs=%.9e case_valid=%d\n",
           ref_repeat_unequal == 0, ref_repeat_unequal, V, ref_repeat_max_abs,
           alt_repeat_unequal == 0, alt_repeat_unequal, V, alt_repeat_max_abs,
           replay_exact, replay_unequal, V, replay_max_abs, case_valid);

    printf("[xray][terminal-readout-stage] stage=gpu_terminal top1=%d runner=%d pair=%+.9e top2_margin=%+.9e\n",
           gpu.winner, gpu.runner, gpu.pair_margin, gpu.top2_margin);
    printf("[xray][terminal-readout-stage] stage=cpu64_classifier_on_gpu_lnf top1=%d runner=%d pair=%+.9e top2_margin=%+.9e gpu_to_cpu64_pair_delta=%+.9e logit_rms_vs_gpu=%.9e logit_max_vs_gpu=%.9e\n",
           cpu64_classifier.winner, cpu64_classifier.runner,
           cpu64_classifier.pair_margin, cpu64_classifier.top2_margin,
           cpu64_classifier.pair_margin - gpu.pair_margin,
           cpu64_classifier.gpu_error_rms, cpu64_classifier.gpu_error_max);
    printf("[xray][terminal-readout-stage] stage=cpu64_lnf_plus_classifier top1=%d runner=%d pair=%+.9e top2_margin=%+.9e lnf_added_pair_delta=%+.9e lnf_rel_l2=%.9e lnf_max_abs=%.9e logit_rms_vs_gpu=%.9e logit_max_vs_gpu=%.9e\n",
           cpu64_terminal.winner, cpu64_terminal.runner,
           cpu64_terminal.pair_margin, cpu64_terminal.top2_margin,
           cpu64_terminal.pair_margin - cpu64_classifier.pair_margin,
           lnf_rel_l2, lnf_max_abs,
           cpu64_terminal.gpu_error_rms, cpu64_terminal.gpu_error_max);

    printf("[xray][terminal-readout-summary] scan_index=%lld family=%s ref=%d alt=%d gpu_pair=%+.9e cpu64_classifier_pair=%+.9e cpu64_terminal_pair=%+.9e classifier_switch_restores_ref=%d terminal_cpu64_restores_ref=%d lnf_added_changes_winner=%d repeat_valid=%d replay_exact=%d case_valid=%d\n",
           scan_index, family->name, ref_winner, alt_winner,
           gpu.pair_margin, cpu64_classifier.pair_margin, cpu64_terminal.pair_margin,
           classifier_switch_restores_ref, terminal_cpu64_restores_ref,
           ln_added_changes_winner, repeat_valid, replay_exact, case_valid);
    printf("[xray][terminal-readout-summary] interpretation gate: the first mixed stage holds the exact GPU ln_f state fixed and changes only classifier arithmetic; the second additionally changes final LayerNorm arithmetic. These nested switches establish sufficiency under the tested execution semantics, not additive or exclusive operator attribution.\n");

    cudaCheck(cudaFree(d_l11));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return case_valid ? 0 : 5;
}
