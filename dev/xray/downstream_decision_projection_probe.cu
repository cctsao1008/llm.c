#pragma push_macro("main")
#undef main
#define main xray_downstream_linearity_embedded_main
#include "downstream_decision_linearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>

enum XrayReplayStage {
    XRAY_STAGE_INPUT = 0,
    XRAY_STAGE_QKVR,
    XRAY_STAGE_ATTY,
    XRAY_STAGE_ATTPROJ,
    XRAY_STAGE_RESIDUAL2,
    XRAY_STAGE_FCH,
    XRAY_STAGE_GELU,
    XRAY_STAGE_FCPROJ,
    XRAY_STAGE_RESIDUAL3,
};

static const char* xray_replay_stage_name(XrayReplayStage s) {
    switch (s) {
        case XRAY_STAGE_INPUT: return "input";
        case XRAY_STAGE_QKVR: return "qkvr";
        case XRAY_STAGE_ATTY: return "atty";
        case XRAY_STAGE_ATTPROJ: return "attproj";
        case XRAY_STAGE_RESIDUAL2: return "residual2";
        case XRAY_STAGE_FCH: return "fch";
        case XRAY_STAGE_GELU: return "gelu";
        case XRAY_STAGE_FCPROJ: return "fcproj";
        case XRAY_STAGE_RESIDUAL3: return "residual3";
    }
    return "?";
}

struct Layer1Snapshot {
    float* input;
    float* qkvr;
    float* atty;
    float* attproj;
    float* residual2;
    float* fch;
    float* gelu;
    float* fcproj;
    float* residual3;
};

static void xray_alloc_snapshot(Layer1Snapshot* s, int B, int T, int C) {
    const size_t BTC = (size_t)B * T * C;
    cudaCheck(cudaMalloc((void**)&s->input,     BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->qkvr,  3 * BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->atty,      BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->attproj,   BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->residual2, BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->fch,    4 * BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->gelu,   4 * BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->fcproj,    BTC * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->residual3, BTC * sizeof(float)));
}

static void xray_free_snapshot(Layer1Snapshot* s) {
    cudaCheck(cudaFree(s->residual3));
    cudaCheck(cudaFree(s->fcproj));
    cudaCheck(cudaFree(s->gelu));
    cudaCheck(cudaFree(s->fch));
    cudaCheck(cudaFree(s->residual2));
    cudaCheck(cudaFree(s->attproj));
    cudaCheck(cudaFree(s->atty));
    cudaCheck(cudaFree(s->qkvr));
    cudaCheck(cudaFree(s->input));
}

static void xray_capture_layer1_snapshot(Layer1Snapshot* s, GPT2* model,
                                         float* input, int B, int T) {
    const int C = model->config.channels;
    const size_t BTC = (size_t)B * T * C;
    ActivationTensors a = model->acts;
    const int l = 1;
    cudaCheck(cudaMemcpy(s->input, input, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->qkvr, a.qkvr + (size_t)l * 3 * BTC, 3 * BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->atty, a.atty + (size_t)l * BTC, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->attproj, a.attproj + (size_t)l * BTC, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->residual2, a.residual2 + (size_t)l * BTC, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->fch, a.fch + (size_t)l * 4 * BTC, 4 * BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->gelu, a.fch_gelu + (size_t)l * 4 * BTC, 4 * BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->fcproj, a.fcproj + (size_t)l * BTC, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s->residual3, a.residual3 + (size_t)l * BTC, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
}

static float xray_replay_layer1_snapshot(GPT2* model, const Layer1Snapshot* s,
                                         XrayReplayStage stage,
                                         int B, int T, int token,
                                         int winner, int runner) {
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    const size_t BTC = (size_t)B * T * C;
    const int l = 1;
    ParameterTensors p = model->params;
    ActivationTensors a = model->acts;

    if (stage == XRAY_STAGE_INPUT) {
        xray_forward_from_residual3(model, s->input, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        return xray_margin_at(model, token, winner, runner);
    }

    float* residual = s->input;
    float* l_qkvr = a.qkvr + (size_t)l * 3 * BTC;
    float* l_atty = a.atty + (size_t)l * BTC;
    float* l_att = a.att + (size_t)l * B * NH * T * T;
    float* l_attproj = a.attproj + (size_t)l * BTC;
    float* l_residual2 = a.residual2 + (size_t)l * BTC;
    float* l_ln2 = a.ln2 + (size_t)l * BTC;
    float* l_ln2_mean = a.ln2_mean + (size_t)l * B * T;
    float* l_ln2_rstd = a.ln2_rstd + (size_t)l * B * T;
    float* l_fch = a.fch + (size_t)l * 4 * BTC;
    float* l_gelu = a.fch_gelu + (size_t)l * 4 * BTC;
    float* l_fcproj = a.fcproj + (size_t)l * BTC;
    float* l_residual3 = a.residual3 + (size_t)l * BTC;

    float* l_attprojw = p.attprojw + (size_t)l * C * C;
    float* l_attprojb = p.attprojb + (size_t)l * C;
    float* l_ln2w = p.ln2w + l * C;
    float* l_ln2b = p.ln2b + l * C;
    float* l_fcw = p.fcw + (size_t)l * 4 * C * C;
    float* l_fcb = p.fcb + (size_t)l * 4 * C;
    float* l_fcprojw = p.fcprojw + (size_t)l * C * 4 * C;
    float* l_fcprojb = p.fcprojb + (size_t)l * C;

    if (stage == XRAY_STAGE_QKVR) {
        attention_forward(l_atty, l_qkvr, l_att, s->qkvr, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
    } else if (stage == XRAY_STAGE_ATTY) {
        matmul_forward(l_attproj, s->atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
    } else if (stage == XRAY_STAGE_ATTPROJ) {
        residual_forward(l_residual2, residual, s->attproj, B * T * C);
    } else if (stage >= XRAY_STAGE_RESIDUAL2) {
        cudaCheck(cudaMemcpy(l_residual2, s->residual2, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    if (stage <= XRAY_STAGE_RESIDUAL2) {
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2,
                          l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_gelu, l_fch, B * T * 4 * C);
        matmul_forward(l_fcproj, l_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    } else if (stage == XRAY_STAGE_FCH) {
        gelu_forward(l_gelu, s->fch, B * T * 4 * C);
        matmul_forward(l_fcproj, l_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    } else if (stage == XRAY_STAGE_GELU) {
        matmul_forward(l_fcproj, s->gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    } else if (stage == XRAY_STAGE_FCPROJ) {
        residual_forward(l_residual3, l_residual2, s->fcproj, B * T * C);
    } else if (stage == XRAY_STAGE_RESIDUAL3) {
        cudaCheck(cudaMemcpy(l_residual3, s->residual3, BTC * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    xray_forward_from_residual3(model, l_residual3, 1, B, T);
    cudaCheck(cudaDeviceSynchronize());
    return xray_margin_at(model, token, winner, runner);
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
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
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

    Layer1Snapshot plus{}, minus{};
    xray_alloc_snapshot(&plus, B, T, C);
    xray_alloc_snapshot(&minus, B, T, C);

    printf("[xray][decision-projection] localize the decision-projection zero crossing along the realized fcproj-L00 TF32 trajectory\n");
    printf("[xray][decision-projection] token=%d winner=%d runner=%d margin0=%+.9e margin1=%+.9e actual_dm=%+.9e\n",
           token, winner, runner, margin0, margin1, (double)margin1 - margin0);
    printf("[xray][decision-projection] stage_secant=[m(replay x_s(+a))-m(replay x_s(-a))]/(2a); paired skip state is preserved where required\n");

    const float alphas[] = {0.35f, 0.36f, 0.37f, 0.38f, 0.39f, 0.40f};
    const XrayReplayStage stages[] = {
        XRAY_STAGE_INPUT, XRAY_STAGE_QKVR, XRAY_STAGE_ATTY, XRAY_STAGE_ATTPROJ,
        XRAY_STAGE_RESIDUAL2, XRAY_STAGE_FCH, XRAY_STAGE_GELU,
        XRAY_STAGE_FCPROJ, XRAY_STAGE_RESIDUAL3
    };
    const int block = 256;
    const int grid = (int)((state_n + block - 1) / block);

    for (float alpha : alphas) {
        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, +alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        xray_capture_layer1_snapshot(&plus, &model, blend, B, T);

        xray_blend_residual<<<grid, block>>>(blend, ref_l00, cur_l00, -alpha, state_n);
        cudaCheck(cudaGetLastError());
        xray_forward_from_residual3(&model, blend, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        xray_capture_layer1_snapshot(&minus, &model, blend, B, T);

        float input_plus = xray_replay_layer1_snapshot(&model, &plus, XRAY_STAGE_INPUT, B, T, token, winner, runner);
        float input_minus = xray_replay_layer1_snapshot(&model, &minus, XRAY_STAGE_INPUT, B, T, token, winner, runner);
        const double full_secant = ((double)input_plus - input_minus) / (2.0 * alpha);
        printf("[xray][decision-projection-margin] a=%.2f full_secant=%+.9e sign=%+d\n",
               alpha, full_secant, (full_secant > 0.0) - (full_secant < 0.0));

        for (XrayReplayStage stage : stages) {
            float mp = xray_replay_layer1_snapshot(&model, &plus, stage, B, T, token, winner, runner);
            float mm = xray_replay_layer1_snapshot(&model, &minus, stage, B, T, token, winner, runner);
            const double sec = ((double)mp - mm) / (2.0 * alpha);
            const double ratio = fabs(full_secant) > 0.0 ? sec / full_secant : 0.0;
            printf("[xray][decision-projection-stage] a=%.2f stage=%-9s mp=%+.9e mm=%+.9e secant=%+.9e sign=%+d ratio_to_input=%+.6f\n",
                   alpha, xray_replay_stage_name(stage), mp, mm, sec,
                   (sec > 0.0) - (sec < 0.0), ratio);
        }
    }

    xray_free_snapshot(&minus);
    xray_free_snapshot(&plus);
    cudaCheck(cudaFree(blend));
    cudaCheck(cudaFree(cur_l00));
    cudaCheck(cudaFree(ref_l00));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
