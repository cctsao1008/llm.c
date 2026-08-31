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

struct XrayCpu64TerminalStateResult {
    std::vector<double> final_residual;
    std::vector<double> lnf;
};

static XrayCpu64TerminalStateResult xray_cpu64_run_from_checkpoint_terminal_state(
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

    XrayCpu64TerminalStateResult out;
    const double* final_row = residual.data() + (size_t)(P - 1) * C;
    out.final_residual.assign(final_row, final_row + C);
    out.lnf = xray_cpu64_layernorm_row_double(out.final_residual, lnfw, lnfb);
    return out;
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 5;
    const int target = argc > 4 ? atoi(argv[4]) : 134;
    const char* family_name = argc > 5 ? argv[5] : "qkv";

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

    // Reference decision and repeatability gate.
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

    // Natural alternate trajectory: exactly one L00 GEMM family switches to TF32.
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
                "[xray][dual-readout] expected final disagreement but ref=%d alt=%d scan_index=%lld family=%s\n",
                ref_winner, alt_winner, scan_index, family->name);
        return 4;
    }

    // Preserve every exact natural GPU residual3 checkpoint and the natural
    // alternate final residual tensor. The latter is used as a full-shape
    // template so the GPU terminal path keeps the original B*T GEMM shape.
    std::vector<float*> checkpoints(L, nullptr);
    for (int k = 0; k < L; ++k) {
        cudaCheck(cudaMalloc((void**)&checkpoints[k], state_bytes));
        cudaCheck(cudaMemcpy(checkpoints[k],
                             model.acts.residual3 + (size_t)k * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
    }
    float *d_alt_final = nullptr, *d_patched_final = nullptr;
    cudaCheck(cudaMalloc((void**)&d_alt_final, state_bytes));
    cudaCheck(cudaMalloc((void**)&d_patched_final, state_bytes));
    cudaCheck(cudaMemcpy(d_alt_final,
                         model.acts.residual3 + (size_t)(L - 1) * state_n,
                         state_bytes, cudaMemcpyDeviceToDevice));

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[xray][dual-readout] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d family=%s\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, family->name);
    printf("[xray][dual-readout] ref=%d alt=%d ref_margin=%+.9e alt_margin=%+.9e\n",
           ref_winner, alt_winner, (double)ref_margin, (double)alt_margin);
    printf("[xray][dual-readout] cpu64-all evaluates transformer suffix + final LN + classifier in CPU64; gpu-terminal evaluates the same CPU64-generated final residual after one explicit float cast through the original full-shape GPU final LN + classifier. cpu64-on-f32 evaluates that same float-cast residual with CPU64 terminal arithmetic to expose cast-only displacement.\n");

    std::string cpu64_topology, cpu64_f32_topology, gpu_terminal_topology;
    cpu64_topology.reserve(L);
    cpu64_f32_topology.reserve(L);
    gpu_terminal_topology.reserve(L);
    int all_gpu_replay_exact = 1;
    int l11_gpu_terminal_exact = 0;

    for (int k = 0; k < L; ++k) {
        // First validate that the exact natural checkpoint still reconstructs
        // the natural alternate logits through the original GPU suffix.
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
        all_gpu_replay_exact &= (replay_unequal == 0);

        const std::vector<float> prefix =
            xray_copy_state_prefix(checkpoints[k], b, T, P, C);
        const XrayCpu64TerminalStateResult cpu =
            xray_cpu64_run_from_checkpoint_terminal_state(prefix, k, hw, lnfw, lnfb,
                                                          L, P, C, NH);
        const XrayReadoutStats cpu64_all =
            xray_cpu64_classifier(cpu.lnf, wte, replay_logits,
                                  V, C, ref_winner, alt_winner);

        // Explicitly round the CPU64-generated final target residual to float.
        std::vector<float> final_f32(C);
        for (int c = 0; c < C; ++c) final_f32[c] = (float)cpu.final_residual[c];

        // Evaluate the same float residual under CPU64 terminal arithmetic.
        const std::vector<double> cpu64_f32_lnf =
            xray_cpu64_layernorm(final_f32, lnfw, lnfb);
        const XrayReadoutStats cpu64_on_f32 =
            xray_cpu64_classifier(cpu64_f32_lnf, wte, replay_logits,
                                  V, C, ref_winner, alt_winner);

        // Evaluate exactly that float residual under the original full-shape GPU
        // terminal path. Other rows remain the natural alternate final residual;
        // LayerNorm is row-local and the classifier keeps its original B*T shape.
        cudaCheck(cudaMemcpy(d_patched_final, d_alt_final,
                             state_bytes, cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_patched_final + (size_t)target * C,
                             final_f32.data(), (size_t)C * sizeof(float),
                             cudaMemcpyHostToDevice));
        layernorm_forward(model.acts.lnf, model.acts.lnf_mean, model.acts.lnf_rstd,
                          d_patched_final, model.params.lnfw, model.params.lnfb,
                          B, T, C);
        matmul_forward(model.acts.output, model.acts.lnf, model.params.wte, NULL,
                       B, T, C, Vp);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<float> gpu_terminal_logits(V);
        cudaCheck(cudaMemcpy(gpu_terminal_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        const XrayReadoutStats gpu_terminal =
            xray_stats_from_gpu_logits(gpu_terminal_logits, ref_winner, alt_winner);

        int l11_unequal = -1;
        double l11_max_abs = 0.0;
        if (k == L - 1) {
            xray_compare_gpu_logits(alt_repeat_logits, gpu_terminal_logits,
                                    &l11_unequal, &l11_max_abs);
            l11_gpu_terminal_exact = (l11_unequal == 0);
        }

        cpu64_topology.push_back(xray_pair_symbol(cpu64_all.pair_margin));
        cpu64_f32_topology.push_back(xray_pair_symbol(cpu64_on_f32.pair_margin));
        gpu_terminal_topology.push_back(xray_pair_symbol(gpu_terminal.pair_margin));

        printf("[xray][dual-readout-point] family=%-7s L=%02d cpu64_all_pair=%+.9e cpu64_on_f32_pair=%+.9e state_cast_delta=%+.9e gpu_terminal_pair=%+.9e terminal_semantics_delta=%+.9e gpu_replay_unequal=%d/%d gpu_replay_max_abs=%.9e",
               family->name, k,
               cpu64_all.pair_margin,
               cpu64_on_f32.pair_margin,
               cpu64_on_f32.pair_margin - cpu64_all.pair_margin,
               gpu_terminal.pair_margin,
               gpu_terminal.pair_margin - cpu64_on_f32.pair_margin,
               replay_unequal, V, replay_max_abs);
        if (k == L - 1) {
            printf(" l11_gpu_terminal_unequal=%d/%d l11_gpu_terminal_max_abs=%.9e",
                   l11_unequal, V, l11_max_abs);
        }
        printf("\n");
    }

    const int repeat_valid =
        (ref_repeat_unequal == 0) && (alt_repeat_unequal == 0);
    const int case_valid = repeat_valid && all_gpu_replay_exact && l11_gpu_terminal_exact;
    const int cpu64_vs_gpu_terminal_same_topology =
        cpu64_topology == gpu_terminal_topology;
    const int cpu64_vs_cpu64_f32_same_topology =
        cpu64_topology == cpu64_f32_topology;

    printf("[xray][dual-readout-validity] ref_repeat_exact=%d alt_repeat_exact=%d all_gpu_replay_exact=%d l11_gpu_terminal_exact=%d case_valid=%d\n",
           ref_repeat_unequal == 0, alt_repeat_unequal == 0,
           all_gpu_replay_exact, l11_gpu_terminal_exact, case_valid);
    printf("[xray][dual-readout-summary] scan_index=%lld family=%s ref=%d alt=%d cpu64_topology=%s cpu64_f32_topology=%s gpu_terminal_topology=%s cpu64_vs_cpu64_f32_same_topology=%d cpu64_vs_gpu_terminal_same_topology=%d case_valid=%d\n",
           scan_index, family->name, ref_winner, alt_winner,
           cpu64_topology.c_str(), cpu64_f32_topology.c_str(),
           gpu_terminal_topology.c_str(),
           cpu64_vs_cpu64_f32_same_topology,
           cpu64_vs_gpu_terminal_same_topology,
           case_valid);
    printf("[xray][dual-readout-summary] interpretation gate: topology disagreement between cpu64-all and gpu-terminal shows dependence on terminal execution semantics for the same CPU64-generated final residual (after the explicit float state boundary). Agreement supports, but does not by itself prove, a transport-level interpretation. No adjacent delta is treated as additive operator attribution.\n");

    for (float* p : checkpoints) {
        if (p) cudaCheck(cudaFree(p));
    }
    cudaCheck(cudaFree(d_alt_final));
    cudaCheck(cudaFree(d_patched_final));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return case_valid ? 0 : 5;
}
