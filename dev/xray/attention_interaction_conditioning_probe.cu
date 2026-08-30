#pragma push_macro("main")
#undef main
#define main xray_pv_accounting_embedded_main
#include "attention_pv_interaction_accounting_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>

static double xray_sum_norms4(const std::vector<double>& a,
                              const std::vector<double>& b,
                              const std::vector<double>& c,
                              const std::vector<double>& d) {
    return xray_l2(a) + xray_l2(b) + xray_l2(c) + xray_l2(d);
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
    if (t < 3) return 3;

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
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner = -1, low_runner = -1;
    float low_margin = 0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_margin);
    cudaCheck(cudaMemcpy(low_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    if (ref_winner == low_winner) {
        printf("[xray][interaction-conditioning] target=%d has no execution disagreement ref=%d low=%d\n",
               target, ref_winner, low_winner);
        return 0;
    }

    const int row_a = b * T + (t - 3);
    const int row_b = b * T + (t - 2);
    float *s00=nullptr, *s10=nullptr, *s01=nullptr, *s11=nullptr;
    cudaCheck(cudaMalloc((void**)&s00, state_bytes));
    cudaCheck(cudaMalloc((void**)&s10, state_bytes));
    cudaCheck(cudaMalloc((void**)&s01, state_bytes));
    cudaCheck(cudaMalloc((void**)&s11, state_bytes));
    cudaCheck(cudaMemcpy(s00, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s10, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s01, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s10 + (size_t)row_a*C, ref_l00 + (size_t)row_a*C, C*sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s01 + (size_t)row_b*C, ref_l00 + (size_t)row_b*C, C*sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11 + (size_t)row_a*C, ref_l00 + (size_t)row_a*C, C*sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(s11 + (size_t)row_b*C, ref_l00 + (size_t)row_b*C, C*sizeof(float), cudaMemcpyDeviceToDevice));

    PVTrace z00{}, z10{}, z01{}, z11{};
    xray_capture_pv(&z00, &model, s00, B, T, target);
    xray_capture_pv(&z10, &model, s10, B, T, target);
    xray_capture_pv(&z01, &model, s01, B, T, target);
    xray_capture_pv(&z11, &model, s11, B, T, target);

    const auto P00=xray_to_double(z00.p), P10=xray_to_double(z10.p), P01=xray_to_double(z01.p), P11=xray_to_double(z11.p);
    const auto V00=xray_to_double(z00.v), V10=xray_to_double(z10.v), V01=xray_to_double(z01.v), V11=xray_to_double(z11.v);
    const auto O00=xray_pv_apply(P00,V00,NH,T,HS), O10=xray_pv_apply(P10,V10,NH,T,HS);
    const auto O01=xray_pv_apply(P01,V01,NH,T,HS), O11=xray_pv_apply(P11,V11,NH,T,HS);
    const auto G00=xray_to_double(z00.atty), G10=xray_to_double(z10.atty);
    const auto G01=xray_to_double(z01.atty), G11=xray_to_double(z11.atty);

    const auto I_model = xray_add(xray_sub(O11,O10), xray_sub(O00,O01));
    const auto I_observed = xray_add(xray_sub(G11,G10), xray_sub(G00,G01));
    const auto e00=xray_sub(G00,O00), e10=xray_sub(G10,O10), e01=xray_sub(G01,O01), e11=xray_sub(G11,O11);
    const auto I_eval = xray_add(xray_sub(e11,e10), xray_sub(e00,e01));
    const auto recon = xray_add(I_model, I_eval);
    const auto residual = xray_sub(I_observed, recon);

    const double model_n=xray_l2(I_model), eval_n=xray_l2(I_eval), obs_n=xray_l2(I_observed);
    const double kappa = model_n > 0.0 ? xray_sum_norms4(O00,O10,O01,O11) / model_n : INFINITY;

    printf("[xray][interaction-conditioning] exact A=t%d B=t%d target=t%d; separate mathematical interaction from evaluation-error interaction\n", t-3,t-2,t);
    printf("[xray][interaction-conditioning-error] e00_l2=%.9e e10_l2=%.9e e01_l2=%.9e e11_l2=%.9e\n",
           xray_l2(e00),xray_l2(e10),xray_l2(e01),xray_l2(e11));
    printf("[xray][interaction-conditioning-state] model_l2=%.9e eval_l2=%.9e observed_l2=%.9e eval/observed=%.9e model/observed=%.9e\n",
           model_n,eval_n,obs_n,obs_n>0?eval_n/obs_n:0.0,obs_n>0?model_n/obs_n:0.0);
    printf("[xray][interaction-conditioning-kappa] sum_base_norms=%.9e model_interaction_l2=%.9e kappa_I=%.9e\n",
           xray_sum_norms4(O00,O10,O01,O11), model_n, kappa);
    printf("[xray][interaction-conditioning-reconstruction] observed_l2=%.9e recon_l2=%.9e residual_l2=%.9e rel_residual=%.9e max_abs_residual=%.9e\n",
           obs_n,xray_l2(recon),xray_l2(residual),xray_rel_l2(recon,I_observed),xray_max_abs(residual));

    cudaCheck(cudaFree(s11)); cudaCheck(cudaFree(s01)); cudaCheck(cudaFree(s10)); cudaCheck(cudaFree(s00));
    cudaCheck(cudaFree(low_l00)); cudaCheck(cudaFree(ref_l00));
    dataloader_free(&loader); gpt2_free(&model); cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
