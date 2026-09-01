#pragma push_macro("main")
#undef main
#define main xray_crossed_conditioning_embedded_main
#include "numerical_fate_crossed_conditioning_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

__global__ static void xray_lerp_state_kernel(float* out,
                                               const float* a,
                                               const float* b,
                                               size_t n,
                                               float alpha) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + alpha * (b[i] - a[i]);
}

struct XrayDeltaStats {
    double ref_l2;
    double delta_l2;
    double rel_l2;
    double max_abs;
};

static XrayDeltaStats xray_delta_stats(const std::vector<float>& ref,
                                       const std::vector<float>& alt) {
    long double ref2 = 0.0L;
    long double d2 = 0.0L;
    double mx = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const long double r = (long double)ref[i];
        const long double d = (long double)alt[i] - r;
        ref2 += r * r;
        d2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
    }
    XrayDeltaStats s;
    s.ref_l2 = std::sqrt((double)ref2);
    s.delta_l2 = std::sqrt((double)d2);
    s.rel_l2 = ref2 > 0.0L ? std::sqrt((double)(d2 / ref2)) : s.delta_l2;
    s.max_abs = mx;
    return s;
}

static char xray_path_class_symbol(const char* cls) {
    if (std::strcmp(cls, "exec-agree") == 0) return '.';
    if (std::strcmp(cls, "terminal-classifier-sufficient") == 0) return 'T';
    if (std::strcmp(cls, "suffix-state-change-required") == 0) return 'S';
    return 'M';
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 0;
    const int target = argc > 4 ? atoi(argv[4]) : 1186;
    const char* family_name = argc > 5 ? argv[5] : "fcproj";
    const int layer = argc > 6 ? atoi(argv[6]) : 0;

    if (B <= 0 || T <= 0 || batch_index < 0 || target < 0 || target >= B * T) {
        fprintf(stderr,
                "usage: %s [B=4] [T=512] [batch_index=0] [batch_local_token=1186] [family=qkv|attproj|fc|fcproj] [layer=0]\n",
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

    // Reference natural execution, repeated exactly, then preserve the requested checkpoint.
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

    float* d_ref = nullptr;
    cudaCheck(cudaMalloc((void**)&d_ref, state_bytes));
    cudaCheck(cudaMemcpy(d_ref,
                         model.acts.residual3 + (size_t)layer * state_n,
                         state_bytes, cudaMemcpyDeviceToDevice));

    // Alternate natural execution, repeated exactly, then preserve the same checkpoint.
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
                "[xray][exec-boundary-path] expected natural final disagreement but ref=%d alt=%d scan_index=%lld family=%s\n",
                ref_winner, alt_winner, scan_index, family->name);
        cudaCheck(cudaFree(d_ref));
        dataloader_free(&loader);
        gpt2_free(&model);
        cublasCheck(cublasDestroy(cublas_handle));
        return 4;
    }
    const int competitor = alt_winner;

    float* d_alt = nullptr;
    float* d_interp = nullptr;
    cudaCheck(cudaMalloc((void**)&d_alt, state_bytes));
    cudaCheck(cudaMalloc((void**)&d_interp, state_bytes));
    cudaCheck(cudaMemcpy(d_alt,
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

    const std::vector<float> ref_prefix = xray_copy_state_prefix(d_ref, b, T, P, C);
    const std::vector<float> alt_prefix = xray_copy_state_prefix(d_alt, b, T, P, C);
    const XrayDeltaStats prefix_delta = xray_delta_stats(ref_prefix, alt_prefix);

    std::vector<float> ref_target(C), alt_target(C);
    cudaCheck(cudaMemcpy(ref_target.data(), d_ref + (size_t)target * C,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(alt_target.data(), d_alt + (size_t)target * C,
                         (size_t)C * sizeof(float), cudaMemcpyDeviceToHost));
    const XrayDeltaStats target_delta = xray_delta_stats(ref_target, alt_target);

    printf("[xray][exec-boundary-path] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d family=%s layer=%d\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, family->name, layer);
    printf("[xray][exec-boundary-path] ref=%d competitor=%d alt=%d ref_top2_margin=%+.9e alt_top2_margin=%+.9e\n",
           ref_winner, competitor, alt_winner,
           (double)ref_margin, (double)alt_margin);
    printf("[xray][exec-boundary-path-state] causal_prefix_ref_l2=%.9e causal_prefix_delta_l2=%.9e causal_prefix_rel_l2=%.9e causal_prefix_delta_max_abs=%.9e target_row_ref_l2=%.9e target_row_delta_l2=%.9e target_row_rel_l2=%.9e target_row_delta_max_abs=%.9e\n",
           prefix_delta.ref_l2, prefix_delta.delta_l2, prefix_delta.rel_l2, prefix_delta.max_abs,
           target_delta.ref_l2, target_delta.delta_l2, target_delta.rel_l2, target_delta.max_abs);
    printf("[xray][exec-boundary-path] alpha traces the straight FP32 state segment h(alpha)=h_ref+alpha*(h_alt-h_ref) at one exact residual3 checkpoint. alpha=0 and alpha=1 are copied bit-exact from the natural reference/alternate states; interior points are counterfactual interpolated states, not naturally executed checkpoints. No monotonicity or single-boundary assumption is made.\n");
    printf("[xray][exec-boundary-path] path_class legend: .=GPU and CPU64 suffix agree; T=terminal classifier switch on exact GPU ln_f is sufficient; S=complete CPU64 suffix state change is required; M=third-winner/mixed case.\n");

    static const double alphas[] = {
        0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
    };
    const int n_alpha = (int)(sizeof(alphas) / sizeof(alphas[0]));

    std::string gpu_winner_path, mixed_winner_path, cpu64_winner_path, class_path;
    std::string gpu_pair_path, mixed_pair_path, cpu64_pair_path;
    gpu_winner_path.reserve(n_alpha);
    mixed_winner_path.reserve(n_alpha);
    cpu64_winner_path.reserve(n_alpha);
    class_path.reserve(n_alpha);
    gpu_pair_path.reserve(n_alpha);
    mixed_pair_path.reserve(n_alpha);
    cpu64_pair_path.reserve(n_alpha);

    int endpoint_ref_exact = 0;
    int endpoint_alt_exact = 0;
    int disagreement_points = 0;
    int terminal_sufficient_points = 0;
    int suffix_required_points = 0;

    for (int ai = 0; ai < n_alpha; ++ai) {
        const double alpha = alphas[ai];
        if (ai == 0) {
            cudaCheck(cudaMemcpy(d_interp, d_ref, state_bytes, cudaMemcpyDeviceToDevice));
        } else if (ai == n_alpha - 1) {
            cudaCheck(cudaMemcpy(d_interp, d_alt, state_bytes, cudaMemcpyDeviceToDevice));
        } else {
            const int threads = 256;
            const int blocks = (int)((state_n + threads - 1) / threads);
            xray_lerp_state_kernel<<<blocks, threads>>>(d_interp, d_ref, d_alt, state_n, (float)alpha);
            cudaCheck(cudaGetLastError());
        }

        // Original GPU complete suffix.
        xray_forward_from_residual3(&model, d_interp, layer, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<float> gpu_logits(V);
        cudaCheck(cudaMemcpy(gpu_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        const XrayReadoutStats gpu =
            xray_stats_from_gpu_logits(gpu_logits, ref_winner, competitor);

        if (ai == 0) {
            int neq = 0;
            double mx = 0.0;
            xray_compare_gpu_logits(ref_repeat_logits, gpu_logits, &neq, &mx);
            endpoint_ref_exact = (neq == 0);
        }
        if (ai == n_alpha - 1) {
            int neq = 0;
            double mx = 0.0;
            xray_compare_gpu_logits(alt_repeat_logits, gpu_logits, &neq, &mx);
            endpoint_alt_exact = (neq == 0);
        }

        // Same exact GPU ln_f state, CPU64 classifier only.
        const std::vector<float> gpu_lnf_f32 = xray_copy_target_lnf(&model, target, C);
        const std::vector<double> gpu_lnf = xray_float_row_to_double(gpu_lnf_f32);
        const XrayReadoutStats mixed =
            xray_cpu64_classifier(gpu_lnf, wte, gpu_logits,
                                  V, C, ref_winner, competitor);
        const XrayDotCondition gpu_cond =
            xray_pair_dot_condition_double(gpu_lnf, wte, C,
                                           ref_winner, competitor);

        // Same interpolated checkpoint through the complete CPU64 suffix.
        const std::vector<float> prefix =
            xray_copy_state_prefix(d_interp, b, T, P, C);
        const XrayCpu64DepthResult cpu_state =
            xray_cpu64_run_from_checkpoint(prefix, layer, hw, lnfw, lnfb,
                                           L, P, C, NH);
        const XrayReadoutStats cpu64 =
            xray_cpu64_classifier(cpu_state.lnf, wte, gpu_logits,
                                  V, C, ref_winner, competitor);
        const XrayDotCondition cpu_cond =
            xray_pair_dot_condition_double(cpu_state.lnf, wte, C,
                                           ref_winner, competitor);

        const char* path_class = xray_state_path_class(
            gpu.winner, mixed.winner, cpu64.winner);
        const char class_symbol = xray_path_class_symbol(path_class);
        class_path.push_back(class_symbol);
        if (gpu.winner != cpu64.winner) ++disagreement_points;
        if (class_symbol == 'T') ++terminal_sufficient_points;
        if (class_symbol == 'S') ++suffix_required_points;

        gpu_winner_path.push_back(xray_winner_symbol(gpu.winner, ref_winner, competitor));
        mixed_winner_path.push_back(xray_winner_symbol(mixed.winner, ref_winner, competitor));
        cpu64_winner_path.push_back(xray_winner_symbol(cpu64.winner, ref_winner, competitor));
        gpu_pair_path.push_back(xray_pair_symbol(gpu.pair_margin));
        mixed_pair_path.push_back(xray_pair_symbol(mixed.pair_margin));
        cpu64_pair_path.push_back(xray_pair_symbol(cpu64.pair_margin));

        printf("[xray][exec-boundary-path-point] alpha=%.3f gpu_pair=%+.9e gpu_winner=%d mixed_pair=%+.9e mixed_winner=%d cpu64_suffix_pair=%+.9e cpu64_suffix_winner=%d gpu_lnf_kappa_dot=%.9e cpu64_lnf_kappa_dot=%.9e path_class=%s\n",
               alpha,
               gpu.pair_margin, gpu.winner,
               mixed.pair_margin, mixed.winner,
               cpu64.pair_margin, cpu64.winner,
               gpu_cond.kappa, cpu_cond.kappa,
               path_class);
    }

    const int repeat_valid =
        (ref_repeat_unequal == 0) && (alt_repeat_unequal == 0);
    const int case_valid = repeat_valid && endpoint_ref_exact && endpoint_alt_exact;

    printf("[xray][exec-boundary-path-validity] ref_repeat_exact=%d alt_repeat_exact=%d alpha0_gpu_replay_exact=%d alpha1_gpu_replay_exact=%d case_valid=%d\n",
           ref_repeat_unequal == 0, alt_repeat_unequal == 0,
           endpoint_ref_exact, endpoint_alt_exact, case_valid);
    printf("[xray][exec-boundary-path-summary] scan_index=%lld family=%s layer=%d ref=%d competitor=%d gpu_pair_path=%s mixed_pair_path=%s cpu64_pair_path=%s gpu_winner_path=%s mixed_winner_path=%s cpu64_winner_path=%s class_path=%s disagreement_points=%d terminal_sufficient_points=%d suffix_required_points=%d case_valid=%d\n",
           scan_index, family->name, layer, ref_winner, competitor,
           gpu_pair_path.c_str(), mixed_pair_path.c_str(), cpu64_pair_path.c_str(),
           gpu_winner_path.c_str(), mixed_winner_path.c_str(), cpu64_winner_path.c_str(),
           class_path.c_str(), disagreement_points,
           terminal_sufficient_points, suffix_required_points, case_valid);
    printf("[xray][exec-boundary-path-summary] interpretation gate: this is a one-dimensional counterfactual state path between two exact natural checkpoints. A suffix-required interval means terminal classifier arithmetic is insufficient there, but it does not by itself identify which downstream operator or prove novelty. Endpoint states remain the only naturally executed states.\n");

    cudaCheck(cudaFree(d_ref));
    cudaCheck(cudaFree(d_alt));
    cudaCheck(cudaFree(d_interp));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return case_valid ? 0 : 5;
}
