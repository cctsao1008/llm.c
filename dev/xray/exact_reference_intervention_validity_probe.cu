#pragma push_macro("main")
#undef main
#define main xray_subset_interaction_embedded_main
#include "adaptive_precision_subset_interaction_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void xray_row_diff(const float* a, const float* b, int n,
                          int* neq, double* rel_l2, double* max_abs) {
    std::vector<float> ha(n), hb(n);
    cudaCheck(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    long double d2 = 0.0L, a2 = 0.0L;
    int q = 0;
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        if (memcmp(&ha[i], &hb[i], sizeof(float)) != 0) ++q;
        const long double d = (long double)hb[i] - ha[i];
        d2 += d * d;
        a2 += (long double)ha[i] * ha[i];
        mx = fmax(mx, fabs((double)d));
    }
    *neq = q;
    *rel_l2 = a2 > 0.0L ? sqrt((double)(d2 / a2)) : sqrt((double)d2);
    *max_abs = mx;
}

static void xray_tensor_diff(const float* a, const float* b, size_t n,
                             int* neq, double* rel_l2, double* max_abs) {
    std::vector<float> ha(n), hb(n);
    cudaCheck(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    long double d2 = 0.0L, a2 = 0.0L;
    int q = 0;
    double mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&ha[i], &hb[i], sizeof(float)) != 0) ++q;
        const long double d = (long double)hb[i] - ha[i];
        d2 += d * d;
        a2 += (long double)ha[i] * ha[i];
        mx = fmax(mx, fabs((double)d));
    }
    *neq = q;
    *rel_l2 = a2 > 0.0L ? sqrt((double)(d2 / a2)) : sqrt((double)d2);
    *max_abs = mx;
}

static void xray_patch_exact_reference_rows(float* patched,
                                             const float* ref_l00,
                                             const int rows[4], int mask, int C) {
    for (int i = 0; i < 4; ++i) {
        if (((mask >> i) & 1) == 0) continue;
        const int tok = rows[i];
        cudaCheck(cudaMemcpy(patched + (size_t)tok * C,
                             ref_l00 + (size_t)tok * C,
                             C * sizeof(float), cudaMemcpyDeviceToDevice));
    }
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

    const int N = B * T;
    const int C = model.config.channels;
    const int Vp = model.config.padded_vocab_size;
    const size_t state_n = (size_t)N * C;
    const size_t state_bytes = state_n * sizeof(float);
    const size_t logits_n = (size_t)N * Vp;
    const size_t logits_bytes = logits_n * sizeof(float);

    // Full custom reference execution and exact L00 state snapshot.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    float *ref_logits = nullptr, *ref_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_bytes));
    cudaCheck(cudaMalloc((void**)&ref_l00, state_bytes));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output, logits_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(ref_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));

    // Alternate execution: only L00 fcproj uses cuBLAS TF32.
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    float *low_logits = nullptr, *low_l00 = nullptr, *low_residual2 = nullptr, *low_gelu = nullptr;
    cudaCheck(cudaMalloc((void**)&low_logits, logits_bytes));
    cudaCheck(cudaMalloc((void**)&low_l00, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_residual2, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_gelu, (size_t)N * 4 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(low_logits, model.acts.output, logits_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(low_l00, model.acts.residual3, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(low_residual2, model.acts.residual2, state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(low_gelu, model.acts.fch_gelu,
                         (size_t)N * 4 * C * sizeof(float), cudaMemcpyDeviceToDevice));

    RescueDecisionStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(RescueDecisionStats)));
    std::vector<RescueDecisionStats> low =
        xray_compare_to_reference(ref_logits, &model, d_stats, B, T);

    const int b = target / T;
    const int t = target % T;
    if (t < 3 || low[target].ref1 == low[target].cur1) {
        printf("[xray][intervention-validity] target=%d unsuitable t=%d ref=%d low=%d\n",
               target, t, low[target].ref1, low[target].cur1);
        return 0;
    }
    const int local_rows[4] = {t - 3, t - 2, t - 1, t};
    const int rows[4] = {b*T + t-3, b*T + t-2, b*T + t-1, b*T + t};
    const int ref_winner = low[target].ref1;
    const int low_winner = low[target].cur1;

    printf("[xray][intervention-validity] separate execution disagreement, intervention validity, and downstream interaction\n");
    printf("[xray][intervention-validity] target=%d b=%d t=%d ref=%d low=%d window=[%d,%d,%d,%d]\n",
           target, b, t, ref_winner, low_winner,
           local_rows[0], local_rows[1], local_rows[2], local_rows[3]);

    // Control 1: does replaying the untouched low checkpoint reproduce low logits?
    xray_forward_from_residual3(&model, low_l00, 0, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int replay_neq = 0;
    double replay_rel = 0.0, replay_max = 0.0;
    xray_tensor_diff(low_logits, model.acts.output, logits_n,
                     &replay_neq, &replay_rel, &replay_max);
    const double replay_pair = xray_margin_at(&model, target, ref_winner, low_winner);
    printf("[xray][intervention-validity-replay] unequal=%d/%zu rel_l2=%.9e max_abs=%.9e pair_margin=%+.9e\n",
           replay_neq, logits_n, replay_rel, replay_max, replay_pair);

    // Control 2: compare M=1 custom recomputation against the exact row produced by the full custom execution.
    float* one_row = nullptr;
    float* one_r3 = nullptr;
    cudaCheck(cudaMalloc((void**)&one_row, C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&one_r3, C * sizeof(float)));
    for (int i = 0; i < 4; ++i) {
        const int tok = rows[i];
        matmul_forward(one_row,
                       low_gelu + (size_t)tok * 4 * C,
                       model.params.fcprojw, model.params.fcprojb,
                       1, 1, 4 * C, C);
        residual_forward(one_r3,
                         low_residual2 + (size_t)tok * C,
                         one_row, C);
        cudaCheck(cudaDeviceSynchronize());
        int neq = 0;
        double rel = 0.0, mx = 0.0;
        xray_row_diff(ref_l00 + (size_t)tok * C, one_r3, C, &neq, &rel, &mx);
        printf("[xray][intervention-validity-row] local_t=%d global_tok=%d unequal=%d/%d rel_l2=%.9e max_abs=%.9e\n",
               local_rows[i], tok, neq, C, rel, mx);
    }

    // Exact binary intervention cube: low row <-> saved full-reference row. No recomputation shape change.
    float* patched = nullptr;
    cudaCheck(cudaMalloc((void**)&patched, state_bytes));
    double margin[16] = {};
    int top1[16] = {};
    int new_flips[16] = {};
    const int initial_flips = xray_count_flips(low);

    for (int mask = 0; mask < 16; ++mask) {
        cudaCheck(cudaMemcpy(patched, low_l00, state_bytes, cudaMemcpyDeviceToDevice));
        xray_patch_exact_reference_rows(patched, ref_l00, rows, mask, C);
        xray_forward_from_residual3(&model, patched, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<RescueDecisionStats> cur =
            xray_compare_to_reference(ref_logits, &model, d_stats, B, T);
        margin[mask] = xray_margin_at(&model, target, ref_winner, low_winner);
        top1[mask] = cur[target].cur1;
        int nf = 0;
        for (int q = 0; q < N; ++q) {
            const bool before = low[q].cur1 != low[q].ref1;
            const bool after = cur[q].cur1 != cur[q].ref1;
            if (!before && after) ++nf;
        }
        new_flips[mask] = nf;
        printf("[xray][exact-subset-result] mask=%2d n=%d pair_margin=%+.9e target_top1=%d baseline_restored=%d total_flips=%d new_flips=%d\n",
               mask, xray_popcount4(mask), margin[mask], top1[mask],
               top1[mask] == ref_winner ? 1 : 0, xray_count_flips(cur), nf);
    }

    int best_mask = -1, best_n = 99;
    for (int mask = 1; mask < 16; ++mask) {
        if (top1[mask] != ref_winner || new_flips[mask] != 0) continue;
        int n = xray_popcount4(mask);
        if (n < best_n) { best_n = n; best_mask = mask; }
    }
    printf("[xray][exact-subset-min] mask=%d n=%d pair_margin=%+.9e initial_flips=%d\n",
           best_mask, best_mask >= 0 ? best_n : -1,
           best_mask >= 0 ? margin[best_mask] : 0.0, initial_flips);

    double abs_by_order[5] = {};
    for (int mask = 1; mask < 16; ++mask) {
        double mu = 0.0;
        for (int sub = mask;; sub = (sub - 1) & mask) {
            const int parity = (xray_popcount4(mask) - xray_popcount4(sub)) & 1;
            mu += parity ? -margin[sub] : margin[sub];
            if (sub == 0) break;
        }
        abs_by_order[xray_popcount4(mask)] += fabs(mu);
        printf("[xray][exact-subset-mobius] mask=%2d order=%d coefficient=%+.9e\n",
               mask, xray_popcount4(mask), mu);
    }
    printf("[xray][exact-subset-order] abs_main=%.9e abs_pair=%.9e abs_triple=%.9e abs_fourway=%.9e\n",
           abs_by_order[1], abs_by_order[2], abs_by_order[3], abs_by_order[4]);

    cudaCheck(cudaFree(patched));
    cudaCheck(cudaFree(one_r3));
    cudaCheck(cudaFree(one_row));
    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(low_gelu));
    cudaCheck(cudaFree(low_residual2));
    cudaCheck(cudaFree(low_l00));
    cudaCheck(cudaFree(low_logits));
    cudaCheck(cudaFree(ref_l00));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
