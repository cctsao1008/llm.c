#define XRAY_DECISION_MARGIN_EMBEDDED
#pragma push_macro("main")
#undef main
#define main xray_decision_margin_embedded_main
#include "decision_margin_probe.cu"
#undef main
#pragma pop_macro("main")
#undef XRAY_DECISION_MARGIN_EMBEDDED

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

static constexpr int XRAY_TOPK = 8;

struct BaselineTopK {
    int idx[XRAY_TOPK];
    float val[XRAY_TOPK];
};

struct DecisionGeometryStats {
    float delta_norm;
    float risk_max;
    float alignment;
    float boundary_distance;
    float step_over_distance;
    float predicted_dmargin;
    float exact_dmargin;
    float baseline_margin;
    float current_margin;
    int risk_candidate;
    int actual_flip;
    int predicted_flip;
    int current_winner_in_topk;
};

__global__ void baseline_topk_kernel(const float* logits, BaselineTopK* out,
                                     int N, int V, int Vp) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= N) return;

    float vals[XRAY_TOPK];
    int idx[XRAY_TOPK];
    #pragma unroll
    for (int k = 0; k < XRAY_TOPK; ++k) {
        vals[k] = -FLT_MAX;
        idx[k] = -1;
    }

    const float* row = logits + (size_t)token * Vp;
    for (int i = 0; i < V; ++i) {
        float v = row[i];
        if (v <= vals[XRAY_TOPK - 1]) continue;
        int pos = XRAY_TOPK - 1;
        while (pos > 0 && v > vals[pos - 1]) {
            vals[pos] = vals[pos - 1];
            idx[pos] = idx[pos - 1];
            --pos;
        }
        vals[pos] = v;
        idx[pos] = i;
    }

    BaselineTopK r;
    #pragma unroll
    for (int k = 0; k < XRAY_TOPK; ++k) {
        r.idx[k] = idx[k];
        r.val[k] = vals[k];
    }
    out[token] = r;
}

__global__ void decision_geometry_kernel(const float* ref_lnf,
                                         const float* cur_lnf,
                                         const float* ref_logits,
                                         const float* cur_logits,
                                         const float* wte,
                                         const BaselineTopK* topk,
                                         const MarginStats* margins,
                                         DecisionGeometryStats* out,
                                         int N, int C, int Vp) {
    int token = blockIdx.x;
    if (token >= N) return;

    constexpr int NC = XRAY_TOPK - 1;
    extern __shared__ float smem[];
    float* sd2 = smem;
    float* sdot = sd2 + blockDim.x;
    float* sg2 = sdot + NC * blockDim.x;

    const BaselineTopK tk = topk[token];
    const int winner = tk.idx[0];
    const float* hr = ref_lnf + (size_t)token * C;
    const float* hc = cur_lnf + (size_t)token * C;

    float d2 = 0.0f;
    float dot[NC];
    float g2[NC];
    #pragma unroll
    for (int k = 0; k < NC; ++k) {
        dot[k] = 0.0f;
        g2[k] = 0.0f;
    }

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = hc[c] - hr[c];
        d2 += d * d;
        float wa = wte[(size_t)winner * C + c];
        #pragma unroll
        for (int k = 0; k < NC; ++k) {
            int j = tk.idx[k + 1];
            float g = wa - wte[(size_t)j * C + c];
            dot[k] += g * d;
            g2[k] += g * g;
        }
    }

    int tid = threadIdx.x;
    sd2[tid] = d2;
    #pragma unroll
    for (int k = 0; k < NC; ++k) {
        sdot[k * blockDim.x + tid] = dot[k];
        sg2[k * blockDim.x + tid] = g2[k];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sd2[tid] += sd2[tid + stride];
            #pragma unroll
            for (int k = 0; k < NC; ++k) {
                sdot[k * blockDim.x + tid] += sdot[k * blockDim.x + tid + stride];
                sg2[k * blockDim.x + tid] += sg2[k * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float* zr = ref_logits + (size_t)token * Vp;
        const float* zc = cur_logits + (size_t)token * Vp;
        float dn = sqrtf(sd2[0]);
        float best_risk = -FLT_MAX;
        int best_k = -1;

        for (int k = 0; k < NC; ++k) {
            int j = tk.idx[k + 1];
            float margin = zr[winner] - zr[j];
            float risk = margin > 0.0f ? -sdot[k * blockDim.x] / margin : -FLT_MAX;
            if (risk > best_risk) {
                best_risk = risk;
                best_k = k;
            }
        }

        DecisionGeometryStats s{};
        s.delta_norm = dn;
        s.risk_max = best_risk;
        s.risk_candidate = best_k >= 0 ? tk.idx[best_k + 1] : -1;
        s.actual_flip = margins[token].ref1 != margins[token].cur1;
        s.predicted_flip = best_risk > 1.0f;
        s.current_winner_in_topk = 0;
        for (int k = 0; k < XRAY_TOPK; ++k) {
            if (tk.idx[k] == margins[token].cur1) s.current_winner_in_topk = 1;
        }

        if (best_k >= 0) {
            int j = tk.idx[best_k + 1];
            float margin0 = zr[winner] - zr[j];
            float margin1 = zc[winner] - zc[j];
            float pdm = sdot[best_k * blockDim.x];
            float gn = sqrtf(sg2[best_k * blockDim.x]);
            s.baseline_margin = margin0;
            s.current_margin = margin1;
            s.predicted_dmargin = pdm;
            s.exact_dmargin = margin1 - margin0;
            s.alignment = (gn > 0.0f && dn > 0.0f) ? pdm / (gn * dn) : 0.0f;
            s.boundary_distance = gn > 0.0f ? margin0 / gn : INFINITY;
            s.step_over_distance = s.boundary_distance > 0.0f ? dn / s.boundary_distance : 0.0f;
        }
        out[token] = s;
    }
}

static void report_geometry_case(const char* label,
                                 GPT2* model,
                                 DataLoader* loader,
                                 int B, int T,
                                 int layer,
                                 XrayComponent component,
                                 const float* ref_logits,
                                 const float* ref_lnf,
                                 const BaselineTopK* d_topk,
                                 MarginStats* d_margins,
                                 DecisionGeometryStats* d_geom) {
    const int N = B * T;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;
    const int C = model->config.channels;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(model, loader->inputs, NULL, B, T, layer, component);
    cudaCheck(cudaDeviceSynchronize());

    const int margin_block = 128;
    decision_margin_kernel<<<CEIL_DIV(N, margin_block), margin_block>>>(
        ref_logits, model->acts.output, d_margins, N, V, Vp);
    cudaCheck(cudaGetLastError());

    const int geom_block = 256;
    const size_t smem = (1 + 2 * (XRAY_TOPK - 1)) * geom_block * sizeof(float);
    decision_geometry_kernel<<<N, geom_block, smem>>>(
        ref_lnf, model->acts.lnf, ref_logits, model->acts.output,
        model->params.wte, d_topk, d_margins, d_geom, N, C, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    std::vector<MarginStats> margins(N);
    std::vector<DecisionGeometryStats> geom(N);
    cudaCheck(cudaMemcpy(margins.data(), d_margins, N * sizeof(MarginStats), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(geom.data(), d_geom, N * sizeof(DecisionGeometryStats), cudaMemcpyDeviceToHost));

    int flips = 0, predicted = 0, tp = 0, fp = 0, fn = 0, flip_in_topk = 0;
    double dnorm_flip = 0.0, dnorm_stable = 0.0;
    double risk_flip = 0.0, risk_stable = 0.0;
    int stable = 0;
    double identity_max = 0.0;

    for (int i = 0; i < N; ++i) {
        bool a = geom[i].actual_flip;
        bool p = geom[i].predicted_flip;
        flips += a;
        predicted += p;
        tp += a && p;
        fp += !a && p;
        fn += a && !p;
        flip_in_topk += a && geom[i].current_winner_in_topk;
        identity_max = fmax(identity_max, fabs((double)geom[i].predicted_dmargin - geom[i].exact_dmargin));
        if (a) {
            dnorm_flip += geom[i].delta_norm;
            risk_flip += geom[i].risk_max;
        } else {
            ++stable;
            dnorm_stable += geom[i].delta_norm;
            risk_stable += geom[i].risk_max;
        }
    }

    std::vector<int> risk_order(N), mag_order(N);
    std::iota(risk_order.begin(), risk_order.end(), 0);
    std::iota(mag_order.begin(), mag_order.end(), 0);
    std::sort(risk_order.begin(), risk_order.end(), [&](int a, int b) { return geom[a].risk_max > geom[b].risk_max; });
    std::sort(mag_order.begin(), mag_order.end(), [&](int a, int b) { return geom[a].delta_norm > geom[b].delta_norm; });
    std::vector<int> risk_rank(N), mag_rank(N);
    for (int r = 0; r < N; ++r) {
        risk_rank[risk_order[r]] = r + 1;
        mag_rank[mag_order[r]] = r + 1;
    }

    printf("[xray][decision-geometry-summary] %-11s component=%-7s L=%02d topk=%d flips=%d/%d predicted=%d tp=%d fp=%d fn=%d flipped_winner_in_topk=%d/%d max|linear_identity_error|=%.3e\n",
           label, component_name(component), layer, XRAY_TOPK, flips, N,
           predicted, tp, fp, fn, flip_in_topk, flips, identity_max);
    printf("[xray][decision-geometry-summary] %-11s mean_delta_norm flip=%.6e stable=%.6e mean_risk flip=%+.6e stable=%+.6e\n",
           label,
           flips ? dnorm_flip / flips : 0.0,
           stable ? dnorm_stable / stable : 0.0,
           flips ? risk_flip / flips : 0.0,
           stable ? risk_stable / stable : 0.0);

    for (int i = 0; i < N; ++i) {
        if (!geom[i].actual_flip) continue;
        printf("[xray][decision-geometry-flip] %-11s tok=%04d b=%d t=%d ref=%d runner=%d cur=%d risk_candidate=%d risk=%+.6f align=%+.6f delta_norm=%.6e boundary_dist=%.6e step/dist=%.6f margin0=%+.7f margin1=%+.7f pred_dm=%+.7f exact_dm=%+.7f risk_rank=%d mag_rank=%d cur_in_topk=%d\n",
               label, i, i / T, i % T,
               margins[i].ref1, margins[i].ref2, margins[i].cur1,
               geom[i].risk_candidate, geom[i].risk_max, geom[i].alignment,
               geom[i].delta_norm, geom[i].boundary_distance, geom[i].step_over_distance,
               geom[i].baseline_margin, geom[i].current_margin,
               geom[i].predicted_dmargin, geom[i].exact_dmargin,
               risk_rank[i], mag_rank[i], geom[i].current_winner_in_topk);
    }

    int shown = 0;
    for (int idx : risk_order) {
        if (geom[idx].actual_flip) continue;
        printf("[xray][decision-geometry-control] %-11s tok=%04d risk=%+.6f align=%+.6f delta_norm=%.6e boundary_dist=%.6e step/dist=%.6f margin0=%+.7f margin1=%+.7f candidate=%d risk_rank=%d mag_rank=%d\n",
               label, idx, geom[idx].risk_max, geom[idx].alignment,
               geom[idx].delta_norm, geom[idx].boundary_distance, geom[idx].step_over_distance,
               geom[idx].baseline_margin, geom[idx].current_margin,
               geom[idx].risk_candidate, risk_rank[idx], mag_rank[idx]);
        if (++shown == 5) break;
    }
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    if (B <= 0 || T <= 0) {
        fprintf(stderr, "usage: %s [B=4] [T=512]\n", argv[0]);
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

    const int N = B * T;
    const int C = model.config.channels;
    const int V = model.config.vocab_size;
    const int Vp = model.config.padded_vocab_size;
    const size_t logits_elems = (size_t)N * Vp;
    const size_t lnf_elems = (size_t)N * C;

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float* ref_logits = nullptr;
    float* ref_lnf = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&ref_lnf, lnf_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output, logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(ref_lnf, model.acts.lnf, lnf_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    BaselineTopK* d_topk = nullptr;
    MarginStats* d_margins = nullptr;
    DecisionGeometryStats* d_geom = nullptr;
    cudaCheck(cudaMalloc((void**)&d_topk, N * sizeof(BaselineTopK)));
    cudaCheck(cudaMalloc((void**)&d_margins, N * sizeof(MarginStats)));
    cudaCheck(cudaMalloc((void**)&d_geom, N * sizeof(DecisionGeometryStats)));

    const int topk_block = 128;
    baseline_topk_kernel<<<CEIL_DIV(N, topk_block), topk_block>>>(ref_logits, d_topk, N, V, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    printf("[xray][decision-geometry] hardware directional risk at final-layernorm representation\n");
    printf("[xray][decision-geometry] R_j = -(w_a-w_j)^T delta_h / (z_a-z_j); compare risk rank with raw ||delta_h|| rank\n");
    printf("[xray][decision-geometry] classifier is linear in h=lnf, so predicted margin delta is an exact geometry check, not a Taylor approximation\n");
    printf("[xray][decision-geometry] candidate set = baseline top-%d logits; this tests whether local decision geometry captures rare execution-induced flips\n", XRAY_TOPK);

    report_geometry_case("qkv-L02", &model, &loader, B, T, 2, XRAY_QKV,
                         ref_logits, ref_lnf, d_topk, d_margins, d_geom);
    report_geometry_case("fcproj-L00", &model, &loader, B, T, 0, XRAY_FCPROJ,
                         ref_logits, ref_lnf, d_topk, d_margins, d_geom);

    cudaCheck(cudaFree(d_geom));
    cudaCheck(cudaFree(d_margins));
    cudaCheck(cudaFree(d_topk));
    cudaCheck(cudaFree(ref_lnf));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
