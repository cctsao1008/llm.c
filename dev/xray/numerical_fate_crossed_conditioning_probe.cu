#pragma push_macro("main")
#undef main
#define main xray_numerical_fate_trajectory_embedded_main
#include "numerical_fate_trajectory_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

struct XrayDotCondition {
    double pair_dot;
    double abs_term_sum;
    double kappa;
};

static XrayDotCondition xray_pair_dot_condition_double(
    const std::vector<double>& lnf,
    const std::vector<float>& wte,
    int C,
    int ref_token,
    int competitor) {

    long double dot = 0.0L;
    long double abs_sum = 0.0L;
    const float* wr = wte.data() + (size_t)ref_token * C;
    const float* wc = wte.data() + (size_t)competitor * C;
    for (int c = 0; c < C; ++c) {
        const long double dw = (long double)wr[c] - (long double)wc[c];
        const long double term = (long double)lnf[c] * dw;
        dot += term;
        abs_sum += std::fabs(term);
    }

    XrayDotCondition out;
    out.pair_dot = (double)dot;
    out.abs_term_sum = (double)abs_sum;
    out.kappa = dot != 0.0L
        ? (double)(abs_sum / std::fabs(dot))
        : std::numeric_limits<double>::infinity();
    return out;
}

static std::vector<double> xray_float_row_to_double(const std::vector<float>& x) {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = (double)x[i];
    return out;
}

static std::vector<float> xray_copy_target_lnf(GPT2* model, int target, int C) {
    std::vector<float> out(C);
    cudaCheck(cudaMemcpy(out.data(),
                         model->acts.lnf + (size_t)target * C,
                         (size_t)C * sizeof(float),
                         cudaMemcpyDeviceToHost));
    return out;
}

static const char* xray_state_path_class(
    int gpu_winner,
    int cpu64_classifier_on_gpu_lnf_winner,
    int cpu64_suffix_winner) {

    if (gpu_winner == cpu64_suffix_winner) return "exec-agree";
    if (cpu64_classifier_on_gpu_lnf_winner == cpu64_suffix_winner) {
        return "terminal-classifier-sufficient";
    }
    if (cpu64_classifier_on_gpu_lnf_winner == gpu_winner) {
        return "suffix-state-change-required";
    }
    return "mixed-third-winner";
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 0;
    const int target = argc > 4 ? atoi(argv[4]) : 42;
    const char* family_name = argc > 5 ? argv[5] : "attproj";
    const int layer = argc > 6 ? atoi(argv[6]) : 11;

    if (B <= 0 || T <= 0 || batch_index < 0 || target < 0 || target >= B * T) {
        fprintf(stderr,
                "usage: %s [B=4] [T=512] [batch_index=0] [batch_local_token=42] [family=qkv|attproj|fc|fcproj] [layer=11]\n",
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

    if (layer < 0 || layer >= L) {
        fprintf(stderr, "layer=%d out of range [0,%d)\n", layer, L);
        return 2;
    }

    // Reference natural execution and exact checkpoint.
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

    float* d_ref_checkpoint = nullptr;
    cudaCheck(cudaMalloc((void**)&d_ref_checkpoint, state_bytes));
    cudaCheck(cudaMemcpy(d_ref_checkpoint,
                         model.acts.residual3 + (size_t)layer * state_n,
                         state_bytes, cudaMemcpyDeviceToDevice));

    // Alternate natural execution and exact checkpoint.
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
                "[xray][crossed-conditioning] expected natural final disagreement but ref=%d alt=%d scan_index=%lld family=%s\n",
                ref_winner, alt_winner, scan_index, family->name);
        cudaCheck(cudaFree(d_ref_checkpoint));
        dataloader_free(&loader);
        gpt2_free(&model);
        cublasCheck(cublasDestroy(cublas_handle));
        return 4;
    }
    const int competitor = alt_winner;

    float* d_alt_checkpoint = nullptr;
    cudaCheck(cudaMalloc((void**)&d_alt_checkpoint, state_bytes));
    cudaCheck(cudaMemcpy(d_alt_checkpoint,
                         model.acts.residual3 + (size_t)layer * state_n,
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

    printf("[xray][crossed-conditioning] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d family=%s layer=%d\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, family->name, layer);
    printf("[xray][crossed-conditioning] ref=%d competitor=%d alt=%d ref_top2_margin=%+.9e alt_top2_margin=%+.9e\n",
           ref_winner, competitor, alt_winner,
           (double)ref_margin, (double)alt_margin);
    printf("[xray][crossed-conditioning] nested path: exact checkpoint -> original GPU complete suffix -> exact GPU ln_f; then hold that exact ln_f fixed and switch only the tied classifier to CPU64. Separately, the same checkpoint is evaluated through the complete CPU64 suffix. kappa_dot is only the terminal pair-dot conditioning of the reported ln_f state, not a condition number for the complete suffix.\n");

    auto audit_state = [&](const char* role,
                           float* d_checkpoint,
                           const std::vector<float>& expected_gpu_logits) {
        // Original GPU suffix from the exact checkpoint.
        xray_forward_from_residual3(&model, d_checkpoint, layer, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<float> gpu_logits(V);
        cudaCheck(cudaMemcpy(gpu_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int replay_unequal = 0;
        double replay_max_abs = 0.0;
        xray_compare_gpu_logits(expected_gpu_logits, gpu_logits,
                                &replay_unequal, &replay_max_abs);
        const XrayReadoutStats gpu =
            xray_stats_from_gpu_logits(gpu_logits, ref_winner, competitor);

        // Exact GPU final-normalized target row; switch only classifier arithmetic.
        const std::vector<float> gpu_lnf_f32 = xray_copy_target_lnf(&model, target, C);
        const std::vector<double> gpu_lnf = xray_float_row_to_double(gpu_lnf_f32);
        const XrayReadoutStats cpu64_classifier_on_gpu_lnf =
            xray_cpu64_classifier(gpu_lnf, wte, gpu_logits,
                                  V, C, ref_winner, competitor);
        const XrayDotCondition gpu_lnf_cond =
            xray_pair_dot_condition_double(gpu_lnf, wte, C,
                                           ref_winner, competitor);

        // Same exact checkpoint through the complete validated CPU64 suffix.
        const std::vector<float> prefix =
            xray_copy_state_prefix(d_checkpoint, b, T, P, C);
        const XrayCpu64DepthResult cpu_state =
            xray_cpu64_run_from_checkpoint(prefix, layer, hw, lnfw, lnfb,
                                           L, P, C, NH);
        const XrayReadoutStats cpu64_suffix =
            xray_cpu64_classifier(cpu_state.lnf, wte, gpu_logits,
                                  V, C, ref_winner, competitor);
        const XrayDotCondition cpu64_lnf_cond =
            xray_pair_dot_condition_double(cpu_state.lnf, wte, C,
                                           ref_winner, competitor);

        const char* path_class = xray_state_path_class(
            gpu.winner,
            cpu64_classifier_on_gpu_lnf.winner,
            cpu64_suffix.winner);

        printf("[xray][crossed-conditioning-state] role=%s "
               "gpu_suffix_pair=%+.9e gpu_suffix_winner=%d "
               "cpu64_classifier_on_gpu_lnf_pair=%+.9e cpu64_classifier_on_gpu_lnf_winner=%d "
               "cpu64_suffix_pair=%+.9e cpu64_suffix_winner=%d "
               "gpu_lnf_pair_dot64=%+.9e gpu_lnf_abs_term_sum=%.9e gpu_lnf_kappa_dot=%.9e "
               "cpu64_lnf_pair_dot64=%+.9e cpu64_lnf_abs_term_sum=%.9e cpu64_lnf_kappa_dot=%.9e "
               "path_class=%s replay_unequal=%d/%d replay_max_abs=%.9e\n",
               role,
               gpu.pair_margin, gpu.winner,
               cpu64_classifier_on_gpu_lnf.pair_margin,
               cpu64_classifier_on_gpu_lnf.winner,
               cpu64_suffix.pair_margin, cpu64_suffix.winner,
               gpu_lnf_cond.pair_dot, gpu_lnf_cond.abs_term_sum, gpu_lnf_cond.kappa,
               cpu64_lnf_cond.pair_dot, cpu64_lnf_cond.abs_term_sum, cpu64_lnf_cond.kappa,
               path_class, replay_unequal, V, replay_max_abs);

        return replay_unequal == 0 ? 1 : 0;
    };

    const int ref_replay_exact = audit_state("reference", d_ref_checkpoint, ref_repeat_logits);
    const int alt_replay_exact = audit_state("alternate", d_alt_checkpoint, alt_repeat_logits);

    const int repeat_valid =
        (ref_repeat_unequal == 0) && (alt_repeat_unequal == 0);
    const int case_valid = repeat_valid && ref_replay_exact && alt_replay_exact;

    printf("[xray][crossed-conditioning-validity] ref_repeat_exact=%d alt_repeat_exact=%d ref_gpu_replay_exact=%d alt_gpu_replay_exact=%d case_valid=%d\n",
           ref_repeat_unequal == 0, alt_repeat_unequal == 0,
           ref_replay_exact, alt_replay_exact, case_valid);
    printf("[xray][crossed-conditioning-summary] scan_index=%lld family=%s layer=%d ref=%d competitor=%d case_valid=%d\n",
           scan_index, family->name, layer, ref_winner, competitor, case_valid);
    printf("[xray][crossed-conditioning-summary] interpretation gate: terminal-classifier-sufficient means changing only classifier arithmetic on the exact GPU ln_f is sufficient to reach the complete-CPU64-suffix winner. suffix-state-change-required means it is not: the full CPU64 suffix must change the downstream state before the winner changes. kappa_dot diagnoses only terminal pair-dot cancellation and cannot explain complete-suffix conditioning by itself.\n");

    cudaCheck(cudaFree(d_ref_checkpoint));
    cudaCheck(cudaFree(d_alt_checkpoint));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return case_valid ? 0 : 5;
}
