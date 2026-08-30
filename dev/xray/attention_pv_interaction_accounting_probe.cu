#pragma push_macro("main")
#undef main
#define main xray_interaction_generator_embedded_main
#include "execution_pair_interaction_generator_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>
#include <vector>

struct PVTrace {
    std::vector<float> p;      // target query attention weights, [NH,T]
    std::vector<float> v;      // value bank for target batch element, [NH,T,HS]
    std::vector<float> atty;   // target attention output, [C]
};

static void xray_capture_pv(PVTrace* tr, GPT2* model, float* l00_state,
                            int B, int T, int target) {
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    const int HS = C / NH;
    const int l = 1;
    const int b = target / T;
    const int t = target % T;
    const size_t BTC = (size_t)B * T * C;

    xray_forward_from_residual3(model, l00_state, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());

    xray_copy_att_query(tr->p, model, l, b, t, T);

    tr->v.resize((size_t)NH * T * HS);
    const float* vbase = model->acts.qkvr + (size_t)l * B * T * 3 * C + 2 * BTC;
    const float* vb = vbase + (size_t)b * T * C;
    cudaCheck(cudaMemcpy(tr->v.data(), vb,
                         (size_t)T * C * sizeof(float), cudaMemcpyDeviceToHost));

    tr->atty.resize(C);
    cudaCheck(cudaMemcpy(tr->atty.data(),
                         model->acts.atty + (size_t)l * BTC + (size_t)target * C,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
}

static std::vector<double> xray_pv_apply(const std::vector<double>& p,
                                         const std::vector<double>& v,
                                         int NH, int T, int HS) {
    std::vector<double> out((size_t)NH * HS, 0.0);
    for (int h = 0; h < NH; ++h) {
        for (int j = 0; j < T; ++j) {
            const double w = p[(size_t)h * T + j];
            const double* vv = v.data() + ((size_t)h * T + j) * HS;
            double* oo = out.data() + (size_t)h * HS;
            for (int c = 0; c < HS; ++c) oo[c] += w * vv[c];
        }
    }
    return out;
}

static std::vector<double> xray_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = (double)x[i];
    return y;
}

static std::vector<double> xray_add(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    std::vector<double> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] + b[i];
    return y;
}

static std::vector<double> xray_sub(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    std::vector<double> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] - b[i];
    return y;
}

static std::vector<double> xray_scale(const std::vector<double>& a, double s) {
    std::vector<double> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = s * a[i];
    return y;
}

static double xray_l2(const std::vector<double>& a) {
    long double s = 0.0L;
    for (double x : a) s += (long double)x * x;
    return sqrt((double)s);
}

static double xray_max_abs(const std::vector<double>& a) {
    double m = 0.0;
    for (double x : a) m = fmax(m, fabs(x));
    return m;
}

static double xray_rel_l2(const std::vector<double>& a,
                          const std::vector<double>& b) {
    // ||a-b|| / ||b||
    long double d2 = 0.0L, b2 = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        long double d = (long double)a[i] - b[i];
        d2 += d * d;
        b2 += (long double)b[i] * b[i];
    }
    return b2 > 0.0L ? sqrt((double)(d2 / b2)) : sqrt((double)d2);
}

static void xray_print_term(const char* name,
                            const std::vector<double>& term,
                            double observed_norm) {
    const double n = xray_l2(term);
    printf("[xray][pv-accounting-term] term=%-24s l2=%.9e fraction_of_observed=%.9e max_abs=%.9e\n",
           name, n, observed_norm > 0.0 ? n / observed_norm : 0.0,
           xray_max_abs(term));
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
    const int NH = model.config.num_heads;
    const int HS = C / NH;
    const int b = target / T;
    const int t = target % T;
    if (t < 3) {
        fprintf(stderr, "target local t=%d is too small for [t-3,t-2] pair\n", t);
        return 3;
    }

    const size_t state_n = (size_t)B * T * C;
    const size_t state_bytes = state_n * sizeof(float);

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);

    float *ref_l00 = nullptr, *low_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_l00, state_bytes));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3,
                         state_bytes, cudaMemcpyDeviceToDevice));

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_margin);
    cudaCheck(cudaMemcpy(low_l00, model.acts.residual3,
                         state_bytes, cudaMemcpyDeviceToDevice));

    if (ref_winner == low_winner) {
        printf("[xray][pv-accounting] target=%d has no execution disagreement ref=%d low=%d\n",
               target, ref_winner, low_winner);
        return 0;
    }

    const int row_a = b * T + (t - 3);
    const int row_b = b * T + (t - 2);
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

    PVTrace z00{}, z10{}, z01{}, z11{};
    xray_capture_pv(&z00, &model, s00, B, T, target);
    xray_capture_pv(&z10, &model, s10, B, T, target);
    xray_capture_pv(&z01, &model, s01, B, T, target);
    xray_capture_pv(&z11, &model, s11, B, T, target);

    const auto P0 = xray_to_double(z00.p);
    const auto P10 = xray_to_double(z10.p);
    const auto P01 = xray_to_double(z01.p);
    const auto P11 = xray_to_double(z11.p);
    const auto V0 = xray_to_double(z00.v);
    const auto V10 = xray_to_double(z10.v);
    const auto V01 = xray_to_double(z01.v);
    const auto V11 = xray_to_double(z11.v);

    const auto dPA = xray_sub(P10, P0);
    const auto dPB = xray_sub(P01, P0);
    const auto IP = xray_add(xray_sub(P11, P10), xray_sub(P0, P01));
    const auto dVA = xray_sub(V10, V0);
    const auto dVB = xray_sub(V01, V0);
    const auto IV = xray_add(xray_sub(V11, V10), xray_sub(V0, V01));

    // CPU reconstruction of the four attention outputs.
    const auto O00 = xray_pv_apply(P0, V0, NH, T, HS);
    const auto O10 = xray_pv_apply(P10, V10, NH, T, HS);
    const auto O01 = xray_pv_apply(P01, V01, NH, T, HS);
    const auto O11 = xray_pv_apply(P11, V11, NH, T, HS);
    const auto Iobs = xray_add(xray_sub(O11, O10), xray_sub(O00, O01));

    // Exact algebraic decomposition of I_O for P11=P0+dPA+dPB+IP and
    // V11=V0+dVA+dVB+IV.
    const auto intrinsic_p = xray_pv_apply(IP, V0, NH, T, HS);
    const auto intrinsic_v = xray_pv_apply(P0, IV, NH, T, HS);
    const auto cross_ab = xray_add(
        xray_pv_apply(dPA, dVB, NH, T, HS),
        xray_pv_apply(dPB, dVA, NH, T, HS));
    const auto cross_p_iv = xray_add(
        xray_pv_apply(dPA, IV, NH, T, HS),
        xray_pv_apply(dPB, IV, NH, T, HS));
    const auto cross_ip_v = xray_add(
        xray_pv_apply(IP, dVA, NH, T, HS),
        xray_pv_apply(IP, dVB, NH, T, HS));
    const auto cross_ip_iv = xray_pv_apply(IP, IV, NH, T, HS);

    auto recon = intrinsic_p;
    recon = xray_add(recon, intrinsic_v);
    recon = xray_add(recon, cross_ab);
    recon = xray_add(recon, cross_p_iv);
    recon = xray_add(recon, cross_ip_v);
    recon = xray_add(recon, cross_ip_iv);

    const auto atty00 = xray_to_double(z00.atty);
    const auto atty10 = xray_to_double(z10.atty);
    const auto atty01 = xray_to_double(z01.atty);
    const auto atty11 = xray_to_double(z11.atty);
    const auto Igpu = xray_add(xray_sub(atty11, atty10), xray_sub(atty00, atty01));

    printf("[xray][pv-accounting] exact A=t%d B=t%d target=t%d; decompose target attention O=P*V interaction\n",
           t - 3, t - 2, t);
    printf("[xray][pv-accounting] ref_winner=%d low_winner=%d NH=%d HS=%d\n",
           ref_winner, low_winner, NH, HS);
    printf("[xray][pv-accounting-control] cpu_vs_gpu_rel_l2 00=%.9e 10=%.9e 01=%.9e 11=%.9e\n",
           xray_rel_l2(O00, atty00), xray_rel_l2(O10, atty10),
           xray_rel_l2(O01, atty01), xray_rel_l2(O11, atty11));
    printf("[xray][pv-accounting-state] p_interaction_l2=%.9e v_interaction_l2=%.9e observed_cpu_l2=%.9e observed_gpu_l2=%.9e cpu_gpu_interaction_rel_l2=%.9e\n",
           xray_l2(IP), xray_l2(IV), xray_l2(Iobs), xray_l2(Igpu),
           xray_rel_l2(Iobs, Igpu));

    const double observed_norm = xray_l2(Iobs);
    xray_print_term("intrinsic-routing IP*V0", intrinsic_p, observed_norm);
    xray_print_term("intrinsic-value P0*IV", intrinsic_v, observed_norm);
    xray_print_term("cross dPA*dVB+dPB*dVA", cross_ab, observed_norm);
    xray_print_term("cross dP*IV", cross_p_iv, observed_norm);
    xray_print_term("cross IP*dV", cross_ip_v, observed_norm);
    xray_print_term("cross IP*IV", cross_ip_iv, observed_norm);

    printf("[xray][pv-accounting-reconstruction] observed_l2=%.9e recon_l2=%.9e residual_l2=%.9e rel_residual=%.9e max_abs_residual=%.9e\n",
           observed_norm, xray_l2(recon), xray_l2(xray_sub(Iobs, recon)),
           xray_rel_l2(recon, Iobs), xray_max_abs(xray_sub(Iobs, recon)));

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
