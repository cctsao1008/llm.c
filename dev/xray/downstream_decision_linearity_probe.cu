#define XRAY_COMPONENT_SENSITIVITY_EMBEDDED
#pragma push_macro("main")
#undef main
#define main xray_component_sensitivity_embedded_main
#include "component_sensitivity_probe.cu"
#undef main
#pragma pop_macro("main")
#undef XRAY_COMPONENT_SENSITIVITY_EMBEDDED

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void xray_blend_residual(float* out, const float* ref, const float* cur,
                                    float alpha, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ref[i] + alpha * (cur[i] - ref[i]);
}

static void xray_forward_from_residual3(GPT2* model, const float* checkpoint,
                                        int checkpoint_layer, int B, int T) {
    const int Vp = model->config.padded_vocab_size;
    const int L = model->config.num_layers;
    const int NH = model->config.num_heads;
    const int C = model->config.channels;
    const size_t BTC = (size_t)B * T * C;

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    for (int l = checkpoint_layer + 1; l < L; ++l) {
        const float* residual = (l == checkpoint_layer + 1)
            ? checkpoint
            : acts.residual3 + (size_t)(l - 1) * BTC;

        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float* l_qkvb = params.qkvb + (size_t)l * 3 * C;
        float* l_attprojw = params.attprojw + (size_t)l * C * C;
        float* l_attprojb = params.attprojb + (size_t)l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float* l_fcb = params.fcb + (size_t)l * 4 * C;
        float* l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float* l_fcprojb = params.fcprojb + (size_t)l * C;

        float* l_ln1 = acts.ln1 + (size_t)l * BTC;
        float* l_ln1_mean = acts.ln1_mean + (size_t)l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + (size_t)l * B * T;
        float* l_qkvr = acts.qkvr + (size_t)l * B * T * 3 * C;
        float* l_atty = acts.atty + (size_t)l * BTC;
        float* l_att = acts.att + (size_t)l * B * NH * T * T;
        float* l_attproj = acts.attproj + (size_t)l * BTC;
        float* l_residual2 = acts.residual2 + (size_t)l * BTC;
        float* l_ln2 = acts.ln2 + (size_t)l * BTC;
        float* l_ln2_mean = acts.ln2_mean + (size_t)l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + (size_t)l * B * T;
        float* l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float* l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float* l_fcproj = acts.fcproj + (size_t)l * BTC;
        float* l_residual3 = acts.residual3 + (size_t)l * BTC;
        float* scratch = acts.output;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual,
                          l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2,
                          l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }

    const float* final_residual = checkpoint_layer == L - 1
        ? checkpoint
        : acts.residual3 + (size_t)(L - 1) * BTC;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                      final_residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);
}

static float xray_margin_at(const GPT2* model, int token, int winner, int competitor) {
    const int Vp = model->config.padded_vocab_size;
    float a = 0.0f, b = 0.0f;
    cudaCheck(cudaMemcpy(&a, model->acts.output + (size_t)token * Vp + winner,
                         sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(&b, model->acts.output + (size_t)token * Vp + competitor,
                         sizeof(float), cudaMemcpyDeviceToHost));
    return a - b;
}

static void xray_top2_at(const GPT2* model, int token, int* winner, int* runner,
                         float* margin) {
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;
    std::vector<float> row(V);
    cudaCheck(cudaMemcpy(row.data(), model->acts.output + (size_t)token * Vp,
                         V * sizeof(float), cudaMemcpyDeviceToHost));
    int a = -1, b = -1;
    float av = -INFINITY, bv = -INFINITY;
    for (int i = 0; i < V; ++i) {
        if (row[i] > av) {
            bv = av; b = a; av = row[i]; a = i;
        } else if (row[i] > bv) {
            bv = row[i]; b = i;
        }
    }
    *winner = a;
    *runner = b;
    *margin = av - bv;
}

static double xray_residual_delta_norm(const float* ref, const float* cur, size_t n) {
    std::vector<float> a(n), b(n);
    cudaCheck(cudaMemcpy(a.data(), ref, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(b.data(), cur, n * sizeof(float), cudaMemcpyDeviceToHost));
    long double d2 = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        long double d = (long double)b[i] - a[i];
        d2 += d * d;
    }
    return sqrt((double)d2);
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

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);
    const int checkpoints[] = {0, 2, 5, 8, 11};
    constexpr int NCHECK = sizeof(checkpoints) / sizeof(checkpoints[0]);

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    int winner = -1, runner = -1;
    float baseline_margin = 0.0f;
    xray_top2_at(&model, token, &winner, &runner, &baseline_margin);

    std::vector<float*> ref_state(NCHECK, nullptr), cur_state(NCHECK, nullptr);
    for (int q = 0; q < NCHECK; ++q) {
        if (checkpoints[q] >= L) continue;
        cudaCheck(cudaMalloc((void**)&ref_state[q], state_bytes));
        cudaCheck(cudaMalloc((void**)&cur_state[q], state_bytes));
        cudaCheck(cudaMemcpy(ref_state[q],
                             model.acts.residual3 + (size_t)checkpoints[q] * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
    }

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T,
                                    0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    const float actual_margin = xray_margin_at(&model, token, winner, runner);
    const float actual_dm = actual_margin - baseline_margin;

    for (int q = 0; q < NCHECK; ++q) {
        if (checkpoints[q] >= L) continue;
        cudaCheck(cudaMemcpy(cur_state[q],
                             model.acts.residual3 + (size_t)checkpoints[q] * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
    }

    printf("[xray][downstream-linearity] actual hardware path = fcproj L00 custom-FP32 -> cuBLAS-TF32\n");
    printf("[xray][downstream-linearity] token=%d b=%d t=%d winner=%d runner=%d margin0=%+.9e margin1=%+.9e actual_dm=%+.9e flip=%d\n",
           token, token / T, token % T, winner, runner,
           baseline_margin, actual_margin, actual_dm, actual_margin < 0.0f);
    printf("[xray][downstream-linearity] at each residual3 checkpoint, replay the exact realized hardware delta through the remaining original/custom downstream graph\n");
    printf("[xray][downstream-linearity] central directional predictor: dm_lin(alpha)=[m(h+a*d)-m(h-a*d)]/(2a), reported in full-delta units\n");

    float* blend = nullptr;
    cudaCheck(cudaMalloc((void**)&blend, state_bytes));
    const float alphas[] = {0.25f, 0.5f, 1.0f};

    for (int q = 0; q < NCHECK; ++q) {
        const int l = checkpoints[q];
        if (l >= L) continue;
        const double dn = xray_residual_delta_norm(ref_state[q], cur_state[q], state_n);

        xray_forward_from_residual3(&model, ref_state[q], l, B, T);
        cudaCheck(cudaDeviceSynchronize());
        const float replay_m0 = xray_margin_at(&model, token, winner, runner);

        xray_forward_from_residual3(&model, cur_state[q], l, B, T);
        cudaCheck(cudaDeviceSynchronize());
        const float replay_m1 = xray_margin_at(&model, token, winner, runner);

        printf("[xray][downstream-linearity-checkpoint] L=%02d delta_norm=%.9e replay_m0=%+.9e replay_m1=%+.9e replay0_err=%+.3e replay1_err=%+.3e\n",
               l, dn, replay_m0, replay_m1,
               (double)replay_m0 - baseline_margin,
               (double)replay_m1 - actual_margin);

        for (float a : alphas) {
            const int block = 256;
            const int grid = (int)((state_n + block - 1) / block);

            xray_blend_residual<<<grid, block>>>(blend, ref_state[q], cur_state[q], +a, state_n);
            cudaCheck(cudaGetLastError());
            xray_forward_from_residual3(&model, blend, l, B, T);
            cudaCheck(cudaDeviceSynchronize());
            const float mp = xray_margin_at(&model, token, winner, runner);

            xray_blend_residual<<<grid, block>>>(blend, ref_state[q], cur_state[q], -a, state_n);
            cudaCheck(cudaGetLastError());
            xray_forward_from_residual3(&model, blend, l, B, T);
            cudaCheck(cudaDeviceSynchronize());
            const float mm = xray_margin_at(&model, token, winner, runner);

            const double dm_lin = ((double)mp - mm) / (2.0 * a);
            const double even = 0.5 * ((double)mp + mm) - replay_m0;
            const double residual = (double)actual_dm - dm_lin;
            const double rel = fabs(actual_dm) > 0.0 ? fabs(residual / actual_dm) : 0.0;
            const double risk_lin = baseline_margin > 0.0f ? -dm_lin / baseline_margin : 0.0;
            const double risk_actual = baseline_margin > 0.0f ? -(double)actual_dm / baseline_margin : 0.0;

            printf("[xray][downstream-linearity-alpha] L=%02d a=%.2f mp=%+.9e mm=%+.9e dm_lin=%+.9e actual_dm=%+.9e residual=%+.9e rel_residual=%.6e even=%+.9e risk_lin=%+.6f risk_actual=%+.6f\n",
                   l, a, mp, mm, dm_lin, (double)actual_dm,
                   residual, rel, even, risk_lin, risk_actual);
        }
    }

    cudaCheck(cudaFree(blend));
    for (int q = 0; q < NCHECK; ++q) {
        if (ref_state[q]) cudaCheck(cudaFree(ref_state[q]));
        if (cur_state[q]) cudaCheck(cudaFree(cur_state[q]));
    }
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
