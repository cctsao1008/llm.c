#pragma push_macro("main")
#undef main
#define main xray_downstream_linearity_embedded_main
#include "downstream_decision_linearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

struct RescueDecisionStats {
    int ref1;
    int cur1;
    int ref2;
    int cur2;
    float ref_margin;
    float cur_margin;
};

__global__ void xray_rescue_decision_kernel(const float* ref, const float* cur,
                                            RescueDecisionStats* out,
                                            int N, int V, int Vp) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= N) return;

    const float* r = ref + (size_t)token * Vp;
    const float* c = cur + (size_t)token * Vp;
    float r1v = -FLT_MAX, r2v = -FLT_MAX;
    float c1v = -FLT_MAX, c2v = -FLT_MAX;
    int r1 = -1, r2 = -1, c1 = -1, c2 = -1;

    for (int i = 0; i < V; ++i) {
        float rv = r[i];
        if (rv > r1v) {
            r2v = r1v; r2 = r1;
            r1v = rv; r1 = i;
        } else if (rv > r2v) {
            r2v = rv; r2 = i;
        }

        float cv = c[i];
        if (cv > c1v) {
            c2v = c1v; c2 = c1;
            c1v = cv; c1 = i;
        } else if (cv > c2v) {
            c2v = cv; c2 = i;
        }
    }

    RescueDecisionStats s;
    s.ref1 = r1; s.cur1 = c1;
    s.ref2 = r2; s.cur2 = c2;
    s.ref_margin = r1v - r2v;
    s.cur_margin = c1v - c2v;
    out[token] = s;
}

static std::vector<RescueDecisionStats> xray_compare_to_reference(
        const float* ref_logits, const GPT2* model,
        RescueDecisionStats* d_stats, int B, int T) {
    const int N = B * T;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;
    const int block = 128;
    xray_rescue_decision_kernel<<<CEIL_DIV(N, block), block>>>(
        ref_logits, model->acts.output, d_stats, N, V, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    std::vector<RescueDecisionStats> h(N);
    cudaCheck(cudaMemcpy(h.data(), d_stats, N * sizeof(RescueDecisionStats),
                         cudaMemcpyDeviceToHost));
    return h;
}

static int xray_count_flips(const std::vector<RescueDecisionStats>& s) {
    int n = 0;
    for (const auto& x : s) n += x.ref1 != x.cur1;
    return n;
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const float threshold = argc > 3 ? strtof(argv[3], nullptr) : 1.0e-3f;
    if (B <= 0 || T <= 0 || threshold < 0.0f) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [threshold=1e-3]\n", argv[0]);
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

    const int N = B * T;
    const int C = model.config.channels;
    const int Vp = model.config.padded_vocab_size;
    const size_t state_n = (size_t)N * C;
    const size_t state_bytes = state_n * sizeof(float);
    const size_t logits_elems = (size_t)N * Vp;

    // Reference = original/custom path.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    float* ref_logits = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output,
                         logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    RescueDecisionStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(RescueDecisionStats)));

    // Low-precision case = only fcproj L00 switches to cuBLAS TF32.
    cudaEvent_t ev0, ev1;
    cudaCheck(cudaEventCreate(&ev0));
    cudaCheck(cudaEventCreate(&ev1));
    cudaCheck(cudaEventRecord(ev0));
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T,
                                    0, XRAY_FCPROJ);
    cudaCheck(cudaEventRecord(ev1));
    cudaCheck(cudaEventSynchronize(ev1));
    float low_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&low_ms, ev0, ev1));

    // Preserve the actual L00 state needed for a real row-level rescue.
    float* cur_l00_residual2 = nullptr;
    float* cur_l00_gelu = nullptr;
    float* rescue_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&cur_l00_residual2, state_bytes));
    cudaCheck(cudaMalloc((void**)&cur_l00_gelu, (size_t)N * 4 * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&rescue_l00, state_bytes));
    cudaCheck(cudaMemcpy(cur_l00_residual2, model.acts.residual2,
                         state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(cur_l00_gelu, model.acts.fch_gelu,
                         (size_t)N * 4 * C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(rescue_l00, model.acts.residual3,
                         state_bytes, cudaMemcpyDeviceToDevice));

    cudaCheck(cudaEventRecord(ev0));
    std::vector<RescueDecisionStats> low =
        xray_compare_to_reference(ref_logits, &model, d_stats, B, T);
    cudaCheck(cudaEventRecord(ev1));
    cudaCheck(cudaEventSynchronize(ev1));
    float gate_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&gate_ms, ev0, ev1));

    std::vector<int> selected;
    selected.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (low[i].cur_margin <= threshold) selected.push_back(i);
    }

    const int low_flips = xray_count_flips(low);
    printf("[xray][actual-rescue] real mitigation prototype for the fcproj-L00 TF32 case\n");
    printf("[xray][actual-rescue] gate=current low-precision top1 margin <= %.9e\n", threshold);
    printf("[xray][actual-rescue] selected rows are recomputed with the original/custom FP32 fcproj kernel at L00, patched into residual3, then the exact downstream graph is replayed\n");
    printf("[xray][actual-rescue] causal note: replay currently covers the full BxT suffix because an L00 row can affect later positions through causal attention; this is a correctness prototype, not an optimized runtime\n");
    printf("[xray][actual-rescue-baseline] low_flips=%d/%d selected=%zu/%d coverage=%.6f low_forward_ms=%.3f gate_ms=%.3f\n",
           low_flips, N, selected.size(), N, (double)selected.size() / N,
           low_ms, gate_ms);

    for (int tok : selected) {
        printf("[xray][actual-rescue-selected] tok=%04d b=%d t=%d low_top1=%d ref_top1=%d low_margin=%.9e was_flip=%d\n",
               tok, tok / T, tok % T, low[tok].cur1, low[tok].ref1,
               low[tok].cur_margin, low[tok].cur1 != low[tok].ref1);
    }

    if (selected.empty()) {
        printf("[xray][actual-rescue-summary] no rows selected; nothing to rescue\n");
    } else {
        float* precise_row = nullptr;
        cudaCheck(cudaMalloc((void**)&precise_row, C * sizeof(float)));
        ParameterTensors p = model.params;
        float* fcprojw = p.fcprojw; // L00
        float* fcprojb = p.fcprojb;

        cudaCheck(cudaEventRecord(ev0));
        for (int tok : selected) {
            // One-token exact/custom recomputation of the L00 FC projection.
            matmul_forward(precise_row,
                           cur_l00_gelu + (size_t)tok * 4 * C,
                           fcprojw, fcprojb,
                           1, 1, 4 * C, C);
            residual_forward(rescue_l00 + (size_t)tok * C,
                             cur_l00_residual2 + (size_t)tok * C,
                             precise_row, C);
        }
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // Re-run layers 1..L-1 plus final norm/classifier from the patched L00 state.
        xray_forward_from_residual3(&model, rescue_l00, 0, B, T);
        cudaCheck(cudaEventRecord(ev1));
        cudaCheck(cudaEventSynchronize(ev1));
        float rescue_ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&rescue_ms, ev0, ev1));

        std::vector<RescueDecisionStats> rescued =
            xray_compare_to_reference(ref_logits, &model, d_stats, B, T);
        const int rescued_flips = xray_count_flips(rescued);

        int recovered = 0;
        int unresolved = 0;
        int new_flips = 0;
        int selected_recovered = 0;
        for (int i = 0; i < N; ++i) {
            const bool was_flip = low[i].cur1 != low[i].ref1;
            const bool is_flip = rescued[i].cur1 != rescued[i].ref1;
            if (was_flip && !is_flip) ++recovered;
            if (was_flip && is_flip) ++unresolved;
            if (!was_flip && is_flip) ++new_flips;
        }
        for (int tok : selected) {
            if (low[tok].cur1 != low[tok].ref1 &&
                rescued[tok].cur1 == rescued[tok].ref1) ++selected_recovered;
            printf("[xray][actual-rescue-result] tok=%04d ref=%d low=%d rescued=%d ref_margin=%.9e low_margin=%.9e rescued_margin=%.9e recovered=%d\n",
                   tok, rescued[tok].ref1, low[tok].cur1, rescued[tok].cur1,
                   rescued[tok].ref_margin, low[tok].cur_margin,
                   rescued[tok].cur_margin,
                   low[tok].cur1 != low[tok].ref1 && rescued[tok].cur1 == rescued[tok].ref1);
        }

        printf("[xray][actual-rescue-summary] initial_flips=%d rescued_flips=%d recovered=%d unresolved=%d new_flips=%d selected_flip_recovered=%d selected_rows=%zu coverage=%.6f rescue_suffix_ms=%.3f rescue_vs_low_ratio=%.3f\n",
               low_flips, rescued_flips, recovered, unresolved, new_flips,
               selected_recovered, selected.size(), (double)selected.size() / N,
               rescue_ms, low_ms > 0.0f ? rescue_ms / low_ms : 0.0f);
        printf("[xray][actual-rescue-interpretation] success requires the real patched/replayed result to reduce flips without creating collateral decision changes; low selected-row coverage alone does not imply low latency until the causal suffix replay is optimized\n");

        cudaCheck(cudaFree(precise_row));
    }

    cudaCheck(cudaEventDestroy(ev1));
    cudaCheck(cudaEventDestroy(ev0));
    cudaCheck(cudaFree(rescue_l00));
    cudaCheck(cudaFree(cur_l00_gelu));
    cudaCheck(cudaFree(cur_l00_residual2));
    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
