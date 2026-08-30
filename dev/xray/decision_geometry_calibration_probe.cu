#pragma push_macro("main")
#undef main
#define main xray_hardware_directional_risk_embedded_main
#include "hardware_directional_risk_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>
#include <cmath>

struct CalibrationSummary {
    int flips = 0;
    int flip_tokens[16]{};
    int nstored = 0;
    double max_identity_err_double = 0.0;
    double max_baseline_margin_recon_err = 0.0;
    double max_current_margin_recon_err = 0.0;
};

static CalibrationSummary capture_case(const char* label,
                                       GPT2* model,
                                       DataLoader* loader,
                                       int B, int T,
                                       int layer,
                                       XrayComponent component,
                                       const float* ref_logits,
                                       const float* ref_lnf,
                                       bool targets_first,
                                       MarginStats* d_stats,
                                       std::vector<float>* cur_lnf_host,
                                       const std::vector<float>* ref_lnf_host,
                                       const std::vector<float>* wte_host) {
    const int N = B * T;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;
    const int C = model->config.channels;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    if (targets_first) {
        gpt2_forward_one_component_tf32(model, loader->inputs, loader->targets,
                                        B, T, layer, component);
        cudaCheck(cudaDeviceSynchronize());
    }
    gpt2_forward_one_component_tf32(model, loader->inputs, NULL,
                                    B, T, layer, component);
    cudaCheck(cudaDeviceSynchronize());

    const int block = 128;
    decision_margin_kernel<<<CEIL_DIV(N, block), block>>>(
        ref_logits, model->acts.output, d_stats, N, V, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    std::vector<MarginStats> stats(N);
    cudaCheck(cudaMemcpy(stats.data(), d_stats, N * sizeof(MarginStats), cudaMemcpyDeviceToHost));

    CalibrationSummary out;
    if (cur_lnf_host) {
        cur_lnf_host->resize((size_t)N * C);
        cudaCheck(cudaMemcpy(cur_lnf_host->data(), model->acts.lnf,
                             (size_t)N * C * sizeof(float), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < N; ++i) {
        const MarginStats& s = stats[i];
        if (s.ref1 != s.cur1) {
            ++out.flips;
            if (out.nstored < 16) out.flip_tokens[out.nstored++] = i;
        }

        if (cur_lnf_host && ref_lnf_host && wte_host && s.ref1 >= 0 && s.ref2 >= 0) {
            const float* hr = ref_lnf_host->data() + (size_t)i * C;
            const float* hc = cur_lnf_host->data() + (size_t)i * C;
            const float* wa = wte_host->data() + (size_t)s.ref1 * C;
            const float* wb = wte_host->data() + (size_t)s.ref2 * C;

            long double pred_dm = 0.0L;
            long double ref_margin_cpu = 0.0L;
            long double cur_margin_cpu = 0.0L;
            for (int c = 0; c < C; ++c) {
                long double g = (long double)wa[c] - (long double)wb[c];
                pred_dm += g * ((long double)hc[c] - (long double)hr[c]);
                ref_margin_cpu += g * (long double)hr[c];
                cur_margin_cpu += g * (long double)hc[c];
            }

            long double exact_dm = (long double)s.refpair_cur_margin - (long double)s.ref_margin;
            out.max_identity_err_double = fmax(out.max_identity_err_double,
                                                (double)fabsl(pred_dm - exact_dm));
            out.max_baseline_margin_recon_err = fmax(out.max_baseline_margin_recon_err,
                                                (double)fabsl(ref_margin_cpu - (long double)s.ref_margin));
            out.max_current_margin_recon_err = fmax(out.max_current_margin_recon_err,
                                                (double)fabsl(cur_margin_cpu - (long double)s.refpair_cur_margin));
        }
    }

    printf("[xray][geometry-cal] %-20s component=%-7s L=%02d targets_first=%d flips=%d/%d tokens=",
           label, component_name(component), layer, targets_first ? 1 : 0, out.flips, N);
    if (out.nstored == 0) printf("none");
    for (int k = 0; k < out.nstored; ++k) printf("%s%d", k ? "," : "", out.flip_tokens[k]);
    printf("\n");

    if (cur_lnf_host && ref_lnf_host && wte_host) {
        printf("[xray][geometry-cal-identity] %-20s max|double_dot_delta - gpu_margin_delta|=%.9e max|cpu_ref_margin-gpu_ref_margin|=%.9e max|cpu_cur_margin-gpu_cur_margin|=%.9e\n",
               label, out.max_identity_err_double,
               out.max_baseline_margin_recon_err,
               out.max_current_margin_recon_err);
    }
    return out;
}

static int same_flip_set(const CalibrationSummary& a, const CalibrationSummary& b) {
    if (a.flips != b.flips || a.nstored != b.nstored) return 0;
    for (int i = 0; i < a.nstored; ++i) if (a.flip_tokens[i] != b.flip_tokens[i]) return 0;
    return 1;
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    if (B <= 0 || T <= 0) return 2;

    cudaCheck(cudaSetDevice(0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    const int N = B * T;
    const int C = model.config.channels;
    const int Vp = model.config.padded_vocab_size;
    const size_t logits_elems = (size_t)N * Vp;
    const size_t lnf_elems = (size_t)N * C;

    // Match the decision-geometry baseline exactly: forward with targets == NULL.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float* ref_logits = nullptr;
    float* ref_lnf = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&ref_lnf, lnf_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output, logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(ref_lnf, model.acts.lnf, lnf_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    std::vector<float> ref_lnf_host(lnf_elems);
    cudaCheck(cudaMemcpy(ref_lnf_host.data(), ref_lnf, lnf_elems * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> wte_host((size_t)model.config.vocab_size * C);
    cudaCheck(cudaMemcpy(wte_host.data(), model.params.wte,
                         wte_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

    MarginStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(MarginStats)));

    printf("[xray][geometry-cal] calibrate two details before pushing decision geometry into earlier layers\n");
    printf("[xray][geometry-cal] (1) repeatability / targets-first call-sequence effect on rare flips\n");
    printf("[xray][geometry-cal] (2) split the final linear-identity residual into GPU-classifier arithmetic vs geometry error\n");

    std::vector<float> cur_lnf;
    CalibrationSummary q0 = capture_case("qkv-null-1", &model, &loader, B, T, 2, XRAY_QKV,
                                         ref_logits, ref_lnf, false, d_stats,
                                         &cur_lnf, &ref_lnf_host, &wte_host);
    CalibrationSummary q1 = capture_case("qkv-null-2", &model, &loader, B, T, 2, XRAY_QKV,
                                         ref_logits, ref_lnf, false, d_stats,
                                         nullptr, nullptr, nullptr);
    CalibrationSummary qt = capture_case("qkv-targets-null", &model, &loader, B, T, 2, XRAY_QKV,
                                         ref_logits, ref_lnf, true, d_stats,
                                         nullptr, nullptr, nullptr);

    CalibrationSummary f0 = capture_case("fcproj-null-1", &model, &loader, B, T, 0, XRAY_FCPROJ,
                                         ref_logits, ref_lnf, false, d_stats,
                                         &cur_lnf, &ref_lnf_host, &wte_host);
    CalibrationSummary f1 = capture_case("fcproj-null-2", &model, &loader, B, T, 0, XRAY_FCPROJ,
                                         ref_logits, ref_lnf, false, d_stats,
                                         nullptr, nullptr, nullptr);
    CalibrationSummary ft = capture_case("fcproj-targets-null", &model, &loader, B, T, 0, XRAY_FCPROJ,
                                         ref_logits, ref_lnf, true, d_stats,
                                         nullptr, nullptr, nullptr);

    // Re-run the ordinary baseline after all intervention calls and compare top-1 against original baseline.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    decision_margin_kernel<<<CEIL_DIV(N, 128), 128>>>(
        ref_logits, model.acts.output, d_stats, N, model.config.vocab_size, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    std::vector<MarginStats> baseline_repeat(N);
    cudaCheck(cudaMemcpy(baseline_repeat.data(), d_stats, N * sizeof(MarginStats), cudaMemcpyDeviceToHost));
    int baseline_top1_changes = 0;
    double baseline_max_refpair_shift = 0.0;
    for (const auto& s : baseline_repeat) {
        baseline_top1_changes += (s.ref1 != s.cur1);
        baseline_max_refpair_shift = fmax(baseline_max_refpair_shift,
                                          fabs((double)s.refpair_cur_margin - (double)s.ref_margin));
    }

    printf("[xray][geometry-cal-repeat] qkv null-repeat_same=%d targets_sequence_same=%d fcproj null-repeat_same=%d targets_sequence_same=%d\n",
           same_flip_set(q0, q1), same_flip_set(q0, qt),
           same_flip_set(f0, f1), same_flip_set(f0, ft));
    printf("[xray][geometry-cal-baseline] repeated_baseline_top1_changes=%d/%d max_refpair_margin_shift=%.9e\n",
           baseline_top1_changes, N, baseline_max_refpair_shift);

    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(ref_lnf));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
