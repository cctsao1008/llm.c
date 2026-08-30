#pragma push_macro("main")
#undef main
#define main xray_decision_projection_embedded_main
#include "downstream_decision_projection_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>

__global__ void xray_add_state(float* out, const float* a, const float* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

static float xray_margin_from_residual2(GPT2* model, float* residual2,
                                        int B, int T, int token,
                                        int winner, int runner) {
    Layer1Snapshot s{};
    s.residual2 = residual2;
    return xray_replay_layer1_snapshot(model, &s, XRAY_STAGE_RESIDUAL2,
                                       B, T, token, winner, runner);
}

static float xray_margin_from_residual3_l1(GPT2* model, float* residual3,
                                           int B, int T, int token,
                                           int winner, int runner) {
    xray_forward_from_residual3(model, residual3, 1, B, T);
    cudaCheck(cudaDeviceSynchronize());
    return xray_margin_at(model, token, winner, runner);
}

static double xray_secant(float mp, float mm, float alpha) {
    return ((double)mp - (double)mm) / (2.0 * (double)alpha);
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
    const int block = 256;
    const int grid = (int)((state_n + block - 1) / block);

    // Baseline and realized L00 hardware trajectory.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int winner = -1, runner = -1;
    float margin0 = 0.0f;
    xray_top2_at(&model, token, &winner, &runner, &margin0);

    float* ref_l00 = nullptr;
    float* cur_l00 = nullptr;
    float* blend = nullptr;
    float* tmp_p = nullptr;
    float* tmp_m = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&cur_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&blend, state_bytes));
    cudaCheck(cudaMalloc((void**)&tmp_p, state_bytes));
    cudaCheck(cudaMalloc((void**)&tmp_m, state_bytes));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(cur_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));
    const float margin1 = xray_margin_at(&model, token, winner, runner);

    // Capture the baseline L1 state tuple once. This is the context anchor used
    // by branch-isolated patches below; unlike the prior path-preserving replay,
    // it intentionally does NOT move every paired state together.
    Layer1Snapshot base{}, plus{}, minus{};
    xray_alloc_snapshot(&base, B, T, C);
    xray_alloc_snapshot(&plus, B, T, C);
    xray_alloc_snapshot(&minus, B, T, C);

    xray_forward_from_residual3(&model, ref_l00, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());
    xray_capture_layer1_snapshot(&base, &model, ref_l00, B, T);

    printf("[xray][branch-decomp] baseline-anchored decomposition around the L1 residual additions\n");
    printf("[xray][branch-decomp] token=%d winner=%d runner=%d margin0=%+.9e margin1=%+.9e actual_dm=%+.9e\n",
           token, winner, runner, margin0, margin1, (double)margin1 - margin0);
    printf("[xray][branch-decomp] attention: residual2=input+attproj; MLP: residual3=residual2+fcproj\n");
    printf("[xray][branch-decomp] isolated branch secants keep the other summand at its baseline value; interaction=combined-skip-branch\n");

    const float alphas[] = {0.35f, 0.36f, 0.37f, 0.38f, 0.39f, 0.40f};

    for (float alpha : alphas) {
        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, +alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        xray_capture_layer1_snapshot(&plus, &model, blend, B, T);
        const float full_mp = xray_margin_at(&model, token, winner, runner);

        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, -alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        xray_capture_layer1_snapshot(&minus, &model, blend, B, T);
        const float full_mm = xray_margin_at(&model, token, winner, runner);

        const double full = xray_secant(full_mp, full_mm, alpha);
        printf("[xray][branch-decomp-margin] a=%.2f full=%+.9e sign=%+d\n",
               alpha, full, (full > 0.0) - (full < 0.0));

        // Attention residual addition: r2 = input + attproj.
        // skip-only: move input, freeze attproj at baseline.
        xray_add_state<<<grid, block>>>(tmp_p, plus.input, base.attproj, state_n);
        xray_add_state<<<grid, block>>>(tmp_m, minus.input, base.attproj, state_n);
        cudaCheck(cudaGetLastError());
        float att_skip_mp = xray_margin_from_residual2(&model, tmp_p, B, T, token, winner, runner);
        float att_skip_mm = xray_margin_from_residual2(&model, tmp_m, B, T, token, winner, runner);
        const double att_skip = xray_secant(att_skip_mp, att_skip_mm, alpha);

        // branch-only: freeze input at baseline, move attention projection.
        xray_add_state<<<grid, block>>>(tmp_p, base.input, plus.attproj, state_n);
        xray_add_state<<<grid, block>>>(tmp_m, base.input, minus.attproj, state_n);
        cudaCheck(cudaGetLastError());
        float att_branch_mp = xray_margin_from_residual2(&model, tmp_p, B, T, token, winner, runner);
        float att_branch_mm = xray_margin_from_residual2(&model, tmp_m, B, T, token, winner, runner);
        const double att_branch = xray_secant(att_branch_mp, att_branch_mm, alpha);

        // combined state at residual2 should reconstruct the full realized path.
        float att_comb_mp = xray_margin_from_residual2(&model, plus.residual2, B, T, token, winner, runner);
        float att_comb_mm = xray_margin_from_residual2(&model, minus.residual2, B, T, token, winner, runner);
        const double att_comb = xray_secant(att_comb_mp, att_comb_mm, alpha);
        const double att_inter = att_comb - att_skip - att_branch;

        printf("[xray][branch-decomp-attn] a=%.2f skip=%+.9e branch=%+.9e combined=%+.9e interaction=%+.9e recon_err=%+.3e\n",
               alpha, att_skip, att_branch, att_comb, att_inter, att_comb - full);

        // MLP residual addition: r3 = residual2 + fcproj.
        // skip-only: move residual2, freeze fcproj at baseline.
        xray_add_state<<<grid, block>>>(tmp_p, plus.residual2, base.fcproj, state_n);
        xray_add_state<<<grid, block>>>(tmp_m, minus.residual2, base.fcproj, state_n);
        cudaCheck(cudaGetLastError());
        float mlp_skip_mp = xray_margin_from_residual3_l1(&model, tmp_p, B, T, token, winner, runner);
        float mlp_skip_mm = xray_margin_from_residual3_l1(&model, tmp_m, B, T, token, winner, runner);
        const double mlp_skip = xray_secant(mlp_skip_mp, mlp_skip_mm, alpha);

        // branch-only: freeze residual2 at baseline, move fcproj.
        xray_add_state<<<grid, block>>>(tmp_p, base.residual2, plus.fcproj, state_n);
        xray_add_state<<<grid, block>>>(tmp_m, base.residual2, minus.fcproj, state_n);
        cudaCheck(cudaGetLastError());
        float mlp_branch_mp = xray_margin_from_residual3_l1(&model, tmp_p, B, T, token, winner, runner);
        float mlp_branch_mm = xray_margin_from_residual3_l1(&model, tmp_m, B, T, token, winner, runner);
        const double mlp_branch = xray_secant(mlp_branch_mp, mlp_branch_mm, alpha);

        float mlp_comb_mp = xray_margin_from_residual3_l1(&model, plus.residual3, B, T, token, winner, runner);
        float mlp_comb_mm = xray_margin_from_residual3_l1(&model, minus.residual3, B, T, token, winner, runner);
        const double mlp_comb = xray_secant(mlp_comb_mp, mlp_comb_mm, alpha);
        const double mlp_inter = mlp_comb - mlp_skip - mlp_branch;

        printf("[xray][branch-decomp-mlp] a=%.2f skip=%+.9e branch=%+.9e combined=%+.9e interaction=%+.9e recon_err=%+.3e\n",
               alpha, mlp_skip, mlp_branch, mlp_comb, mlp_inter, mlp_comb - full);
    }

    xray_free_snapshot(&minus);
    xray_free_snapshot(&plus);
    xray_free_snapshot(&base);
    cudaCheck(cudaFree(tmp_m));
    cudaCheck(cudaFree(tmp_p));
    cudaCheck(cudaFree(blend));
    cudaCheck(cudaFree(cur_l00));
    cudaCheck(cudaFree(ref_l00));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
