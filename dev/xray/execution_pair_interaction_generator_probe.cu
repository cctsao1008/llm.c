#pragma push_macro("main")
#undef main
#define main xray_downstream_linearity_embedded_main
#include "downstream_decision_linearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>
#include <vector>

struct InteractionTrace {
    std::vector<float> input;
    std::vector<float> ln1;
    std::vector<float> att;
    std::vector<float> atty;
    std::vector<float> attproj;
    std::vector<float> residual2;
    std::vector<float> ln2;
    std::vector<float> fch;
    std::vector<float> gelu;
    std::vector<float> fcproj;
    std::vector<float> residual3;
    float pair_margin;
};

static void xray_copy_slice(std::vector<float>& dst, const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void xray_copy_att_query(std::vector<float>& dst, const GPT2* model,
                                int layer, int b, int t, int T) {
    const int NH = model->config.num_heads;
    dst.resize((size_t)NH * T);
    const float* base = model->acts.att + (size_t)layer * model->batch_size * NH * T * T;
    for (int h = 0; h < NH; ++h) {
        const float* row = base + ((size_t)b * NH + h) * T * T + (size_t)t * T;
        cudaCheck(cudaMemcpy(dst.data() + (size_t)h * T, row,
                             T * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

static void xray_capture_trace(InteractionTrace* tr, GPT2* model,
                               float* l00_state, int B, int T,
                               int target, int ref_winner, int low_winner) {
    const int C = model->config.channels;
    const int l = 1;
    const int b = target / T;
    const int t = target % T;
    const size_t BTC = (size_t)B * T * C;
    ActivationTensors a = model->acts;

    xray_forward_from_residual3(model, l00_state, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());

    xray_copy_slice(tr->input, l00_state + (size_t)target * C, C);
    xray_copy_slice(tr->ln1, a.ln1 + (size_t)l * BTC + (size_t)target * C, C);
    xray_copy_att_query(tr->att, model, l, b, t, T);
    xray_copy_slice(tr->atty, a.atty + (size_t)l * BTC + (size_t)target * C, C);
    xray_copy_slice(tr->attproj, a.attproj + (size_t)l * BTC + (size_t)target * C, C);
    xray_copy_slice(tr->residual2, a.residual2 + (size_t)l * BTC + (size_t)target * C, C);
    xray_copy_slice(tr->ln2, a.ln2 + (size_t)l * BTC + (size_t)target * C, C);
    xray_copy_slice(tr->fch, a.fch + (size_t)l * 4 * BTC + (size_t)target * 4 * C, 4 * C);
    xray_copy_slice(tr->gelu, a.fch_gelu + (size_t)l * 4 * BTC + (size_t)target * 4 * C, 4 * C);
    xray_copy_slice(tr->fcproj, a.fcproj + (size_t)l * BTC + (size_t)target * C, C);
    xray_copy_slice(tr->residual3, a.residual3 + (size_t)l * BTC + (size_t)target * C, C);
    tr->pair_margin = xray_margin_at(model, target, ref_winner, low_winner);
}

struct InteractionMetric {
    double a_norm;
    double b_norm;
    double joint_norm;
    double interaction_norm;
    double interaction_over_main;
    double interaction_over_joint;
    double max_abs;
};

static InteractionMetric xray_interaction_metric(const std::vector<float>& x00,
                                                  const std::vector<float>& x10,
                                                  const std::vector<float>& x01,
                                                  const std::vector<float>& x11) {
    long double a2 = 0.0L, b2 = 0.0L, j2 = 0.0L, i2 = 0.0L;
    double mx = 0.0;
    for (size_t k = 0; k < x00.size(); ++k) {
        const long double da = (long double)x10[k] - x00[k];
        const long double db = (long double)x01[k] - x00[k];
        const long double dj = (long double)x11[k] - x00[k];
        const long double di = (long double)x11[k] - x10[k] - x01[k] + x00[k];
        a2 += da * da;
        b2 += db * db;
        j2 += dj * dj;
        i2 += di * di;
        mx = fmax(mx, fabs((double)di));
    }
    const double an = sqrt((double)a2);
    const double bn = sqrt((double)b2);
    const double jn = sqrt((double)j2);
    const double in = sqrt((double)i2);
    return InteractionMetric{
        an, bn, jn, in,
        (an + bn) > 0.0 ? in / (an + bn) : 0.0,
        jn > 0.0 ? in / jn : 0.0,
        mx
    };
}

static void xray_print_stage(const char* name,
                             const std::vector<float>& x00,
                             const std::vector<float>& x10,
                             const std::vector<float>& x01,
                             const std::vector<float>& x11) {
    InteractionMetric m = xray_interaction_metric(x00, x10, x01, x11);
    printf("[xray][interaction-generator-stage] stage=%-10s a_l2=%.9e b_l2=%.9e joint_l2=%.9e interaction_l2=%.9e interaction/(a+b)=%.6e interaction/joint=%.6e max_abs=%.9e\n",
           name, m.a_norm, m.b_norm, m.joint_norm, m.interaction_norm,
           m.interaction_over_main, m.interaction_over_joint, m.max_abs);
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int target = argc > 3 ? atoi(argv[3]) : 1186;
    if (B <= 0 || T <= 0 || target < 0 || target >= B * T) {
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
    const int b = target / T;
    const int t = target % T;
    if (t < 3) {
        fprintf(stderr, "target local t=%d is too small for the known [t-3,t] interaction window\n", t);
        return 3;
    }

    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    // Full-custom execution: provides the exact rows used for state substitution.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);

    float* ref_l00 = nullptr;
    float* low_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_l00, state_bytes));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    // Alternate execution: only L00 fcproj switches to TF32.
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_top2_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_top2_margin);
    cudaCheck(cudaMemcpy(low_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    if (ref_winner == low_winner) {
        printf("[xray][interaction-generator] target=%d is not an execution disagreement in this run ref=%d low=%d\n",
               target, ref_winner, low_winner);
        return 0;
    }

    const int row_a = b * T + (t - 3); // local 159 for the current case
    const int row_b = b * T + (t - 2); // local 160 for the current case

    float *s00 = nullptr, *s10 = nullptr, *s01 = nullptr, *s11 = nullptr;
    cudaCheck(cudaMalloc((void**)&s00, state_bytes));
    cudaCheck(cudaMalloc((void**)&s10, state_bytes));
    cudaCheck(cudaMalloc((void**)&s01, state_bytes));
    cudaCheck(cudaMalloc((void**)&s11, state_bytes));
    cudaCheck(cudaMemcpy(s00, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s10, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s01, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11, low_l00, state_bytes, cudaMemcpyDeviceToDevice));

    cudaCheck(cudaMemcpy(s10 + (size_t)row_a * C, ref_l00 + (size_t)row_a * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s01 + (size_t)row_b * C, ref_l00 + (size_t)row_b * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11 + (size_t)row_a * C, ref_l00 + (size_t)row_a * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11 + (size_t)row_b * C, ref_l00 + (size_t)row_b * C,
                         C * sizeof(float), cudaMemcpyDeviceToDevice));

    InteractionTrace tr00{}, tr10{}, tr01{}, tr11{};
    xray_capture_trace(&tr00, &model, s00, B, T, target, ref_winner, low_winner);
    xray_capture_trace(&tr10, &model, s10, B, T, target, ref_winner, low_winner);
    xray_capture_trace(&tr01, &model, s01, B, T, target, ref_winner, low_winner);
    xray_capture_trace(&tr11, &model, s11, B, T, target, ref_winner, low_winner);

    const double margin_interaction =
        (double)tr11.pair_margin - tr10.pair_margin - tr01.pair_margin + tr00.pair_margin;

    printf("[xray][interaction-generator] exact pair intervention at L00 residual3; A=local_t%d B=local_t%d target_t=%d\n",
           t - 3, t - 2, t);
    printf("[xray][interaction-generator] target=%d b=%d t=%d ref_winner=%d low_winner=%d\n",
           target, b, t, ref_winner, low_winner);
    printf("[xray][interaction-generator] target-local traces are used deliberately: A and B are earlier causal positions, so target input/ln1 should remain invariant; the first nonzero interaction identifies where their information first mixes at the target\n");
    printf("[xray][interaction-generator-margin] m00=%+.9e m10=%+.9e m01=%+.9e m11=%+.9e interaction=%+.9e\n",
           tr00.pair_margin, tr10.pair_margin, tr01.pair_margin, tr11.pair_margin,
           margin_interaction);

    xray_print_stage("input",     tr00.input,     tr10.input,     tr01.input,     tr11.input);
    xray_print_stage("ln1",       tr00.ln1,       tr10.ln1,       tr01.ln1,       tr11.ln1);
    xray_print_stage("att",       tr00.att,       tr10.att,       tr01.att,       tr11.att);
    xray_print_stage("atty",      tr00.atty,      tr10.atty,      tr01.atty,      tr11.atty);
    xray_print_stage("attproj",   tr00.attproj,   tr10.attproj,   tr01.attproj,   tr11.attproj);
    xray_print_stage("residual2", tr00.residual2, tr10.residual2, tr01.residual2, tr11.residual2);
    xray_print_stage("ln2",       tr00.ln2,       tr10.ln2,       tr01.ln2,       tr11.ln2);
    xray_print_stage("fch",       tr00.fch,       tr10.fch,       tr01.fch,       tr11.fch);
    xray_print_stage("gelu",      tr00.gelu,      tr10.gelu,      tr01.gelu,      tr11.gelu);
    xray_print_stage("fcproj",    tr00.fcproj,    tr10.fcproj,    tr01.fcproj,    tr11.fcproj);
    xray_print_stage("residual3", tr00.residual3, tr10.residual3, tr01.residual3, tr11.residual3);

    cudaCheck(cudaFree(s11));
    cudaCheck(cudaFree(s01));
    cudaCheck(cudaFree(s10));
    cudaCheck(cudaFree(s00));
    cudaCheck(cudaFree(low_l00));
    cudaCheck(cudaFree(ref_l00));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
