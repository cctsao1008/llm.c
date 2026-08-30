#pragma push_macro("main")
#undef main
#define main xray_downstream_breakpoint_embedded_main
#include "downstream_alpha_breakpoint_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>
#include <vector>

struct OddDirectionMetric {
    double norm;
    double cosine_to_ref;
    double rel_change_to_ref;
};

static void directional_response(std::vector<double>& out,
                                 const std::vector<float>& plus,
                                 const std::vector<float>& minus,
                                 double alpha) {
    out.resize(plus.size());
    const double scale = 1.0 / (2.0 * alpha);
    for (size_t i = 0; i < plus.size(); ++i)
        out[i] = ((double)plus[i] - minus[i]) * scale;
}

static OddDirectionMetric compare_direction(const std::vector<double>& ref,
                                            const std::vector<double>& cur) {
    long double rr = 0.0L, cc = 0.0L, rc = 0.0L, dd = 0.0L;
    for (size_t i = 0; i < ref.size(); ++i) {
        long double r = ref[i];
        long double c = cur[i];
        long double d = c - r;
        rr += r * r;
        cc += c * c;
        rc += r * c;
        dd += d * d;
    }
    const double rn = sqrt((double)rr);
    const double cn = sqrt((double)cc);
    const double cosine = (rn > 0.0 && cn > 0.0) ? (double)(rc / sqrt(rr * cc)) : 0.0;
    const double rel = rn > 0.0 ? sqrt((double)dd) / rn : 0.0;
    return OddDirectionMetric{cn, cosine, rel};
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int token = argc > 3 ? atoi(argv[3]) : 1186;
    if (B <= 0 || T <= 0 || token < 0 || token >= B * T) {
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

    const int C = model.config.channels;
    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int winner = -1, runner = -1;
    float margin0 = 0.0f;
    xray_top2_at(&model, token, &winner, &runner, &margin0);

    float* ref_l00 = nullptr;
    float* cur_l00 = nullptr;
    float* blend = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&cur_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&blend, state_bytes));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(cur_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));
    const float margin1 = xray_margin_at(&model, token, winner, runner);

    xray_forward_from_residual3(&model, ref_l00, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());
    auto stages = layer1_stages(&model, B, T);

    printf("[xray][odd-direction] test whether the L00 alpha breakpoint is a rotation of the decision-relevant odd response rather than a norm jump\n");
    printf("[xray][odd-direction] token=%d winner=%d runner=%d margin0=%+.9e margin1=%+.9e actual_dm=%+.9e\n",
           token, winner, runner, margin0, margin1, (double)margin1 - margin0);
    printf("[xray][odd-direction] derivative-like stage vector d_a=[x(+a)-x(-a)]/(2a); alpha=0.25 is the reference direction\n");

    const double alphas[] = {0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
    const int block = 256;
    const int grid = (int)((state_n + block - 1) / block);
    std::vector<std::vector<double>> ref_dir(stages.size());
    double ref_margin_dir = 0.0;

    for (size_t ai = 0; ai < sizeof(alphas) / sizeof(alphas[0]); ++ai) {
        const double alpha = alphas[ai];
        std::vector<std::vector<float>> plus_stage(stages.size()), minus_stage(stages.size());

        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, (float)+alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        const float mp = xray_margin_at(&model, token, winner, runner);
        stages = layer1_stages(&model, B, T);
        for (size_t s = 0; s < stages.size(); ++s)
            copy_token_slice(plus_stage[s], stages[s].ptr, token, stages[s].width);

        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, (float)-alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        const float mm = xray_margin_at(&model, token, winner, runner);
        stages = layer1_stages(&model, B, T);
        for (size_t s = 0; s < stages.size(); ++s)
            copy_token_slice(minus_stage[s], stages[s].ptr, token, stages[s].width);

        const double margin_dir = ((double)mp - mm) / (2.0 * alpha);
        if (ai == 0) ref_margin_dir = margin_dir;
        printf("[xray][odd-direction-margin] a=%.2f dm_dir=%+.9e ratio_to_ref=%+.6f sign_changed=%d\n",
               alpha, margin_dir,
               ref_margin_dir != 0.0 ? margin_dir / ref_margin_dir : 0.0,
               (margin_dir > 0.0) != (ref_margin_dir > 0.0));

        for (size_t s = 0; s < stages.size(); ++s) {
            std::vector<double> dir;
            directional_response(dir, plus_stage[s], minus_stage[s], alpha);
            if (ai == 0) ref_dir[s] = dir;
            OddDirectionMetric m = compare_direction(ref_dir[s], dir);
            printf("[xray][odd-direction-stage] a=%.2f stage=%-9s dir_norm=%.9e cos_ref=%.9f rel_change=%.9e\n",
                   alpha, stages[s].name, m.norm, m.cosine_to_ref, m.rel_change_to_ref);
        }
    }

    cudaCheck(cudaFree(blend));
    cudaCheck(cudaFree(cur_l00));
    cudaCheck(cudaFree(ref_l00));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
