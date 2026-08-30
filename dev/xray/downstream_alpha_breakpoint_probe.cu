#pragma push_macro("main")
#undef main
#define main xray_downstream_linearity_embedded_main
#include "downstream_decision_linearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct StageSpec {
    const char* name;
    float* ptr;
    size_t width;
};

struct SymmetryMetric {
    double odd_l2;
    double even_l2;
    double even_over_odd;
};

static void copy_token_slice(std::vector<float>& dst, const float* base,
                             int token, size_t width) {
    dst.resize(width);
    cudaCheck(cudaMemcpy(dst.data(), base + (size_t)token * width,
                         width * sizeof(float), cudaMemcpyDeviceToHost));
}

static SymmetryMetric symmetry_metric(const std::vector<float>& base,
                                      const std::vector<float>& plus,
                                      const std::vector<float>& minus) {
    long double odd2 = 0.0L;
    long double even2 = 0.0L;
    for (size_t i = 0; i < base.size(); ++i) {
        long double o = 0.5L * ((long double)plus[i] - minus[i]);
        long double e = 0.5L * ((long double)plus[i] + minus[i]) - base[i];
        odd2 += o * o;
        even2 += e * e;
    }
    double odd = sqrt((double)odd2);
    double even = sqrt((double)even2);
    return SymmetryMetric{odd, even, odd > 0.0 ? even / odd : 0.0};
}

static std::vector<StageSpec> layer1_stages(GPT2* model, int B, int T) {
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    (void)NH;
    const size_t BTC = (size_t)B * T * C;
    const int l = 1;
    ActivationTensors a = model->acts;
    return {
        {"ln1",       a.ln1       + (size_t)l * BTC,             (size_t)C},
        {"qkvr",      a.qkvr      + (size_t)l * B * T * 3 * C,  (size_t)3 * C},
        {"atty",      a.atty      + (size_t)l * BTC,             (size_t)C},
        {"attproj",   a.attproj   + (size_t)l * BTC,             (size_t)C},
        {"residual2", a.residual2 + (size_t)l * BTC,             (size_t)C},
        {"ln2",       a.ln2       + (size_t)l * BTC,             (size_t)C},
        {"fch",       a.fch       + (size_t)l * B * T * 4 * C,  (size_t)4 * C},
        {"gelu",      a.fch_gelu  + (size_t)l * B * T * 4 * C,  (size_t)4 * C},
        {"fcproj",    a.fcproj    + (size_t)l * BTC,             (size_t)C},
        {"residual3", a.residual3 + (size_t)l * BTC,             (size_t)C},
    };
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

    // Baseline full forward and L00 residual checkpoint.
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

    // Actual hardware perturbation: only fcproj L00 switches to cuBLAS TF32.
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(cur_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));
    float margin1 = xray_margin_at(&model, token, winner, runner);

    // Replay baseline from L00 and capture layer-1 stage anchor slices.
    xray_forward_from_residual3(&model, ref_l00, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());
    auto stages = layer1_stages(&model, B, T);
    std::vector<std::vector<float>> base_stage(stages.size());
    for (size_t s = 0; s < stages.size(); ++s)
        copy_token_slice(base_stage[s], stages[s].ptr, token, stages[s].width);

    printf("[xray][alpha-breakpoint] localize the L00 trajectory-space breakpoint for the actual fcproj-L00 TF32 perturbation\n");
    printf("[xray][alpha-breakpoint] token=%d b=%d t=%d winner=%d runner=%d margin0=%+.9e margin1=%+.9e actual_dm=%+.9e\n",
           token, token / T, token % T, winner, runner, margin0, margin1, (double)margin1 - margin0);
    printf("[xray][alpha-breakpoint] bracket alpha in [0.25,0.50]; stage metrics are token-local layer-1 odd/even L2 around the baseline replay\n");

    const float alphas[] = {0.25f, 0.30f, 0.35f, 0.40f, 0.45f, 0.50f};
    const int block = 256;
    const int grid = (int)((state_n + block - 1) / block);

    for (float alpha : alphas) {
        std::vector<std::vector<float>> plus_stage(stages.size()), minus_stage(stages.size());

        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, +alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        float mp = xray_margin_at(&model, token, winner, runner);
        stages = layer1_stages(&model, B, T);
        for (size_t s = 0; s < stages.size(); ++s)
            copy_token_slice(plus_stage[s], stages[s].ptr, token, stages[s].width);

        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, -alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        float mm = xray_margin_at(&model, token, winner, runner);
        stages = layer1_stages(&model, B, T);
        for (size_t s = 0; s < stages.size(); ++s)
            copy_token_slice(minus_stage[s], stages[s].ptr, token, stages[s].width);

        const double dm_lin = ((double)mp - mm) / (2.0 * alpha);
        const double even_margin = 0.5 * ((double)mp + mm) - margin0;
        printf("[xray][alpha-breakpoint-margin] a=%.2f mp=%+.9e mm=%+.9e dm_lin=%+.9e even=%+.9e risk_lin=%+.6f\n",
               alpha, mp, mm, dm_lin, even_margin,
               margin0 > 0.0f ? -dm_lin / margin0 : 0.0);

        for (size_t s = 0; s < stages.size(); ++s) {
            SymmetryMetric m = symmetry_metric(base_stage[s], plus_stage[s], minus_stage[s]);
            printf("[xray][alpha-breakpoint-stage] a=%.2f stage=%-9s odd_l2=%.9e even_l2=%.9e even/odd=%.6e\n",
                   alpha, stages[s].name, m.odd_l2, m.even_l2, m.even_over_odd);
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
