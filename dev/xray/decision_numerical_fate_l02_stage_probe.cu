#pragma push_macro("main")
#undef main
#define main xray_cpu64_causal_embedded_main
#include "decision_cpu64_causal_suffix_audit_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

enum XrayL02Stage {
    ST_INPUT = 0,
    ST_LN1,
    ST_QKV,
    ST_ATTY,
    ST_ATTPROJ,
    ST_RESIDUAL2,
    ST_LN2,
    ST_FCH,
    ST_GELU,
    ST_FCPROJ,
    ST_RESIDUAL3,
    ST_COUNT
};

static const char* xray_stage_name(XrayL02Stage s) {
    switch (s) {
        case ST_INPUT: return "input";
        case ST_LN1: return "ln1";
        case ST_QKV: return "qkv";
        case ST_ATTY: return "atty";
        case ST_ATTPROJ: return "attproj";
        case ST_RESIDUAL2: return "residual2";
        case ST_LN2: return "ln2";
        case ST_FCH: return "fch";
        case ST_GELU: return "gelu";
        case ST_FCPROJ: return "fcproj";
        case ST_RESIDUAL3: return "residual3";
        default: return "?";
    }
}

struct XrayL02GpuPrefix {
    std::vector<float> input, ln1, qkv, atty, attproj, residual2;
    std::vector<float> ln2, fch, gelu, fcproj, residual3;
};

struct XrayRelDiff64F32 {
    double rel_l2;
    double max_abs;
};

static std::vector<double> xray_to_double(const std::vector<float>& src) {
    std::vector<double> dst(src.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)src.size(); ++i) dst[(size_t)i] = (double)src[(size_t)i];
    return dst;
}

static XrayRelDiff64F32 xray_rel_diff64_f32(const std::vector<double>& a,
                                             const std::vector<float>& b) {
    long double a2 = 0.0L, d2 = 0.0L;
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const long double av = (long double)a[i];
        const long double d = (long double)b[i] - av;
        a2 += av * av;
        d2 += d * d;
        mx = std::max(mx, std::fabs((double)d));
    }
    return XrayRelDiff64F32{
        a2 > 0.0L ? std::sqrt((double)(d2 / a2)) : std::sqrt((double)d2),
        mx
    };
}

static void xray_copy_gpu_prefix(std::vector<float>& dst, const float* base,
                                 int b, int T, int P, int width) {
    dst.resize((size_t)P * width);
    cudaCheck(cudaMemcpy(dst.data(), base + (size_t)b * T * width,
                         (size_t)P * width * sizeof(float), cudaMemcpyDeviceToHost));
}

// attention_forward() does not retain its raw [B,T,3C] QKV input. It permutes
// the matmul result into qkvr as three separate [B,NH,T,HS] planes. Reconstruct
// the canonical [P,3C] layout expected by xray_cpu64_attention().
static void xray_copy_qkv_canonical(std::vector<float>& dst,
                                    const float* qkvr_layer,
                                    int B, int b, int T, int P, int C, int NH) {
    const int HS = C / NH;
    const size_t plane = (size_t)B * T * C;
    std::vector<float> q((size_t)T * C), k((size_t)T * C), v((size_t)T * C);
    const size_t batch_off = (size_t)b * T * C;
    cudaCheck(cudaMemcpy(q.data(), qkvr_layer + 0 * plane + batch_off,
                         (size_t)T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(k.data(), qkvr_layer + 1 * plane + batch_off,
                         (size_t)T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(v.data(), qkvr_layer + 2 * plane + batch_off,
                         (size_t)T * C * sizeof(float), cudaMemcpyDeviceToHost));

    dst.resize((size_t)P * 3 * C);
    for (int t = 0; t < P; ++t) {
        for (int h = 0; h < NH; ++h) {
            for (int d = 0; d < HS; ++d) {
                const size_t src = ((size_t)h * T + t) * HS + d;
                const size_t c = (size_t)h * HS + d;
                dst[(size_t)t * 3 * C + 0 * C + c] = q[src];
                dst[(size_t)t * 3 * C + 1 * C + c] = k[src];
                dst[(size_t)t * 3 * C + 2 * C + c] = v[src];
            }
        }
    }
}

static void xray_capture_l02_gpu_prefix(XrayL02GpuPrefix* p, GPT2* model,
                                        int b, int T, int P) {
    const int B = model->batch_size;
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    const int l = 2;
    const size_t BTC = (size_t)B * T * C;
    xray_copy_gpu_prefix(p->input, model->acts.residual3 + (size_t)(l - 1) * BTC, b, T, P, C);
    xray_copy_gpu_prefix(p->ln1, model->acts.ln1 + (size_t)l * BTC, b, T, P, C);
    const float* qkvr_layer = model->acts.qkvr + (size_t)l * B * T * 3 * C;
    xray_copy_qkv_canonical(p->qkv, qkvr_layer, B, b, T, P, C, NH);
    xray_copy_gpu_prefix(p->atty, model->acts.atty + (size_t)l * BTC, b, T, P, C);
    xray_copy_gpu_prefix(p->attproj, model->acts.attproj + (size_t)l * BTC, b, T, P, C);
    xray_copy_gpu_prefix(p->residual2, model->acts.residual2 + (size_t)l * BTC, b, T, P, C);
    xray_copy_gpu_prefix(p->ln2, model->acts.ln2 + (size_t)l * BTC, b, T, P, C);
    xray_copy_gpu_prefix(p->fch, model->acts.fch + (size_t)l * B * T * 4 * C, b, T, P, 4 * C);
    xray_copy_gpu_prefix(p->gelu, model->acts.fch_gelu + (size_t)l * B * T * 4 * C, b, T, P, 4 * C);
    xray_copy_gpu_prefix(p->fcproj, model->acts.fcproj + (size_t)l * BTC, b, T, P, C);
    xray_copy_gpu_prefix(p->residual3, model->acts.residual3 + (size_t)l * BTC, b, T, P, C);
}

static std::vector<double> xray_cpu64_continue_from_residual_checkpoint(
    const std::vector<float>& checkpoint,
    int checkpoint_layer,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw, const std::vector<float>& lnfb,
    int L, int P, int C, int NH) {

    std::vector<double> residual = xray_to_double(checkpoint);
    std::vector<double> a, q, y, ap, r2, n2, f, ge, fp, r3;
    for (int l = checkpoint_layer + 1; l < L; ++l) {
        const auto& wl = hw[l];
        xray_cpu64_layernorm_rows(a, residual, wl.ln1w, wl.ln1b, P, C);
        xray_cpu64_matmul(q, a, wl.qkvw, &wl.qkvb, P, C, 3 * C);
        xray_cpu64_attention(y, q, P, C, NH);
        xray_cpu64_matmul(ap, y, wl.attprojw, &wl.attprojb, P, C, C);
        xray_cpu64_add(r2, residual, ap);
        xray_cpu64_layernorm_rows(n2, r2, wl.ln2w, wl.ln2b, P, C);
        xray_cpu64_matmul(f, n2, wl.fcw, &wl.fcb, P, C, 4 * C);
        xray_cpu64_gelu(ge, f);
        xray_cpu64_matmul(fp, ge, wl.fcprojw, &wl.fcprojb, P, 4 * C, C);
        xray_cpu64_add(r3, r2, fp);
        residual.swap(r3);
    }
    std::vector<double> target_row(residual.end() - C, residual.end());
    return xray_cpu64_layernorm_row_double(target_row, lnfw, lnfb);
}

static std::vector<double> xray_cpu64_finish_from_l02_stage(
    XrayL02Stage stage, const XrayL02GpuPrefix& g,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw, const std::vector<float>& lnfb,
    int L, int P, int C, int NH) {

    const auto& w = hw[2];
    std::vector<double> input = xray_to_double(g.input);
    std::vector<double> ln1, qkv, atty, attproj, residual2, ln2, fch, gelu, fcproj, residual3;

    if (stage >= ST_LN1) ln1 = xray_to_double(g.ln1);
    else xray_cpu64_layernorm_rows(ln1, input, w.ln1w, w.ln1b, P, C);

    if (stage >= ST_QKV) qkv = xray_to_double(g.qkv);
    else xray_cpu64_matmul(qkv, ln1, w.qkvw, &w.qkvb, P, C, 3 * C);

    if (stage >= ST_ATTY) atty = xray_to_double(g.atty);
    else xray_cpu64_attention(atty, qkv, P, C, NH);

    if (stage >= ST_ATTPROJ) attproj = xray_to_double(g.attproj);
    else xray_cpu64_matmul(attproj, atty, w.attprojw, &w.attprojb, P, C, C);

    if (stage >= ST_RESIDUAL2) residual2 = xray_to_double(g.residual2);
    else xray_cpu64_add(residual2, input, attproj);

    if (stage >= ST_LN2) ln2 = xray_to_double(g.ln2);
    else xray_cpu64_layernorm_rows(ln2, residual2, w.ln2w, w.ln2b, P, C);

    if (stage >= ST_FCH) fch = xray_to_double(g.fch);
    else xray_cpu64_matmul(fch, ln2, w.fcw, &w.fcb, P, C, 4 * C);

    if (stage >= ST_GELU) gelu = xray_to_double(g.gelu);
    else xray_cpu64_gelu(gelu, fch);

    if (stage >= ST_FCPROJ) fcproj = xray_to_double(g.fcproj);
    else xray_cpu64_matmul(fcproj, gelu, w.fcprojw, &w.fcprojb, P, 4 * C, C);

    if (stage >= ST_RESIDUAL3) residual3 = xray_to_double(g.residual3);
    else xray_cpu64_add(residual3, residual2, fcproj);

    std::vector<double> residual = std::move(residual3);
    std::vector<double> a, q, y, ap, r2, n2, f, ge, fp, r3;
    for (int l = 3; l < L; ++l) {
        const auto& wl = hw[l];
        xray_cpu64_layernorm_rows(a, residual, wl.ln1w, wl.ln1b, P, C);
        xray_cpu64_matmul(q, a, wl.qkvw, &wl.qkvb, P, C, 3 * C);
        xray_cpu64_attention(y, q, P, C, NH);
        xray_cpu64_matmul(ap, y, wl.attprojw, &wl.attprojb, P, C, C);
        xray_cpu64_add(r2, residual, ap);
        xray_cpu64_layernorm_rows(n2, r2, wl.ln2w, wl.ln2b, P, C);
        xray_cpu64_matmul(f, n2, wl.fcw, &wl.fcb, P, C, 4 * C);
        xray_cpu64_gelu(ge, f);
        xray_cpu64_matmul(fp, ge, wl.fcprojw, &wl.fcprojb, P, 4 * C, C);
        xray_cpu64_add(r3, r2, fp);
        residual.swap(r3);
    }

    std::vector<double> target(residual.end() - C, residual.end());
    return xray_cpu64_layernorm_row_double(target, lnfw, lnfb);
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int target = argc > 3 ? atoi(argv[3]) : 1186;
    if (B <= 0 || T <= 0 || target < 0 || target >= B * T) return 2;

    cudaCheck(cudaSetDevice(0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const int NH = model.config.num_heads;
    const int V = model.config.vocab_size;
    const int b = target / T;
    const int t = target % T;
    const int P = t + 1;

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner=-1, ref_runner=-1; float ref_margin=0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);

    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    int low_winner=-1, low_runner=-1; float low_margin=0.0f;
    xray_top2_at(&model, target, &low_winner, &low_runner, &low_margin);
    const float low_pair = xray_margin_at(&model, target, ref_winner, low_winner);
    if (ref_winner == low_winner) {
        printf("[xray][decision-l02-stage] target has no natural execution disagreement\n");
        return 0;
    }

    XrayL02GpuPrefix g;
    xray_capture_l02_gpu_prefix(&g, &model, b, T, P);

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C), gpu_logits(V);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte, (size_t)V*C*sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw, C*sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb, C*sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(gpu_logits.data(), model.acts.output + (size_t)target * model.config.padded_vocab_size,
                         V*sizeof(float), cudaMemcpyDeviceToHost));

    // Internal numerical sanity checks for the QKV boundary. They are not used
    // as mechanism evidence; they only guard against a catastrophic layout error.
    const auto& w2 = hw[2];
    std::vector<double> ln1_gpu64 = xray_to_double(g.ln1);
    std::vector<double> qkv_from_ln1_cpu64;
    xray_cpu64_matmul(qkv_from_ln1_cpu64, ln1_gpu64, w2.qkvw, &w2.qkvb, P, C, 3*C);
    const XrayRelDiff64F32 qkv_sanity = xray_rel_diff64_f32(qkv_from_ln1_cpu64, g.qkv);
    std::vector<double> atty_from_gpu_qkv_cpu64;
    xray_cpu64_attention(atty_from_gpu_qkv_cpu64, xray_to_double(g.qkv), P, C, NH);
    const XrayRelDiff64F32 atty_sanity = xray_rel_diff64_f32(atty_from_gpu_qkv_cpu64, g.atty);

    // Endpoint references use a generic continuation from the exact same residual3
    // checkpoint. g.input is L01 residual3 (the input of L02), so its continuation
    // begins at layer 2; g.residual3 is L02 residual3, so it begins at layer 3.
    const std::vector<double> input_ref_lnf =
        xray_cpu64_continue_from_residual_checkpoint(g.input, 1, hw, lnfw, lnfb, L, P, C, NH);
    const XrayReadoutStats input_stats =
        xray_cpu64_classifier(input_ref_lnf, wte, gpu_logits, V, C, ref_winner, low_winner);
    const std::vector<double> residual3_ref_lnf =
        xray_cpu64_continue_from_residual_checkpoint(g.residual3, 2, hw, lnfw, lnfb, L, P, C, NH);
    const XrayReadoutStats residual3_stats =
        xray_cpu64_classifier(residual3_ref_lnf, wte, gpu_logits, V, C, ref_winner, low_winner);

    printf("[xray][decision-l02-stage] natural GPU path; nested switch points inside L02: GPU prefix through stage, then CPU64 remainder of L02 and layers 3..11\n");
    printf("[xray][decision-l02-stage] target=%d b=%d t=%d ref=%d low=%d gpu_low_pair=%+.9e\n",
           target,b,t,ref_winner,low_winner,low_pair);
    printf("[xray][decision-l02-stage-qkv-sanity] reconstructed_qkv_vs_cpu64_matmul_rel_l2=%.9e max_abs=%.9e cpu64_attention_from_gpu_qkv_vs_gpu_atty_rel_l2=%.9e max_abs=%.9e\n",
           qkv_sanity.rel_l2, qkv_sanity.max_abs, atty_sanity.rel_l2, atty_sanity.max_abs);

    double prev_pair=0.0;
    int prev_sign=0;
    int first_sign_change_stage=-1;
    int sign_changes=0;
    double input_endpoint_err=0.0, residual3_endpoint_err=0.0;

    for (int si=0; si<ST_COUNT; ++si) {
        XrayL02Stage s=(XrayL02Stage)si;
        std::vector<double> lnf=xray_cpu64_finish_from_l02_stage(s,g,hw,lnfw,lnfb,L,P,C,NH);
        XrayReadoutStats st=xray_cpu64_classifier(lnf,wte,gpu_logits,V,C,ref_winner,low_winner);
        const int sg=xray_sign64(st.pair_margin);
        const double step = si ? st.pair_margin-prev_pair : 0.0;
        if (si && sg != prev_sign) {
            ++sign_changes;
            if (first_sign_change_stage < 0) first_sign_change_stage=si;
        }
        if (s==ST_INPUT) input_endpoint_err=st.pair_margin-input_stats.pair_margin;
        if (s==ST_RESIDUAL3) residual3_endpoint_err=st.pair_margin-residual3_stats.pair_margin;
        printf("[xray][decision-l02-stage-point] stage=%-10s cpu64_top1=%d cpu64_runner=%d cpu64_pair=%+.9e sign=%+d switch_delta=%+.9e\n",
               xray_stage_name(s),st.winner,st.runner,st.pair_margin,sg,step);
        prev_pair=st.pair_margin;
        prev_sign=sg;
    }

    printf("[xray][decision-l02-stage-validation] input_ref_pair=%+.9e input_endpoint_err=%+.9e residual3_ref_pair=%+.9e residual3_endpoint_err=%+.9e\n",
           input_stats.pair_margin,input_endpoint_err,residual3_stats.pair_margin,residual3_endpoint_err);
    printf("[xray][decision-l02-stage-summary] sign_changes=%d first_sign_change_stage=%s endpoint_valid=%d\n",
           sign_changes,
           first_sign_change_stage>=0?xray_stage_name((XrayL02Stage)first_sign_change_stage):"none",
           (std::fabs(input_endpoint_err)<1e-12 && std::fabs(residual3_endpoint_err)<1e-12));
    printf("[xray][decision-l02-stage-summary] interpretation gate: adjacent switch_delta is a nested execution-prefix effect, not a standalone operator contribution or mechanism claim\n");

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}