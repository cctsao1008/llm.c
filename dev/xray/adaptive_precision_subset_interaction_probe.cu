#pragma push_macro("main")
#undef main
#define main xray_causal_cone_embedded_main
#include "adaptive_precision_causal_cone_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstdio>
#include <vector>

static int xray_popcount4(int mask) {
    int n = 0;
    for (int i = 0; i < 4; ++i) n += (mask >> i) & 1;
    return n;
}

static void xray_repair_l00_mask(GPT2* model,
                                 float* l00_residual2,
                                 const float* l00_gelu,
                                 float* patched_l00,
                                 const int rows[4],
                                 int mask,
                                 int C) {
    float* precise_row = nullptr;
    cudaCheck(cudaMalloc((void**)&precise_row, C * sizeof(float)));
    float* fcprojw = model->params.fcprojw;
    float* fcprojb = model->params.fcprojb;
    for (int i = 0; i < 4; ++i) {
        if (((mask >> i) & 1) == 0) continue;
        const int tok = rows[i];
        matmul_forward(precise_row,
                       l00_gelu + (size_t)tok * 4 * C,
                       fcprojw, fcprojb,
                       1, 1, 4 * C, C);
        residual_forward(patched_l00 + (size_t)tok * C,
                         l00_residual2 + (size_t)tok * C,
                         precise_row, C);
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFree(precise_row));
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
    const size_t logits_elems = (size_t)N * Vp;

    // Reference: all-original/custom forward.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    float* ref_logits = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output,
                         logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    RescueDecisionStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(RescueDecisionStats)));

    // Low path: only fcproj L00 is TF32.
    gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T,
                                    0, XRAY_FCPROJ);
    cudaCheck(cudaDeviceSynchronize());
    std::vector<RescueDecisionStats> low =
        xray_compare_to_reference(ref_logits, &model, d_stats, B, T);

    const int b = target / T;
    const int t = target % T;
    if (t < 3) {
        fprintf(stderr, "target local t=%d is too small for a 4-token recent window\n", t);
        return 3;
    }
    if (low[target].ref1 == low[target].cur1) {
        printf("[xray][subset-interaction] requested token=%d is not a flip in this run; ref=%d low=%d\n",
               target, low[target].ref1, low[target].cur1);
        return 0;
    }

    float* low_l00_residual2 = nullptr;
    float* low_l00_gelu = nullptr;
    float* low_l00_residual3 = nullptr;
    float* patched_l00 = nullptr;
    cudaCheck(cudaMalloc((void**)&low_l00_residual2, state_bytes));
    cudaCheck(cudaMalloc((void**)&low_l00_gelu, (size_t)N * 4 * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&low_l00_residual3, state_bytes));
    cudaCheck(cudaMalloc((void**)&patched_l00, state_bytes));
    cudaCheck(cudaMemcpy(low_l00_residual2, model.acts.residual2,
                         state_bytes, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(low_l00_gelu, model.acts.fch_gelu,
                         (size_t)N * 4 * C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(low_l00_residual3, model.acts.residual3,
                         state_bytes, cudaMemcpyDeviceToDevice));

    const int local_rows[4] = {t - 3, t - 2, t - 1, t};
    const int rows[4] = {
        b * T + local_rows[0], b * T + local_rows[1],
        b * T + local_rows[2], b * T + local_rows[3]
    };
    const int ref_winner = low[target].ref1;
    const int low_winner = low[target].cur1;

    printf("[xray][subset-interaction] exhaust all 16 repair subsets of the minimal known 4-token rescue window\n");
    printf("[xray][subset-interaction] target=%d b=%d t=%d ref=%d low=%d window=[%d,%d,%d,%d]\n",
           target, b, t, ref_winner, low_winner,
           local_rows[0], local_rows[1], local_rows[2], local_rows[3]);
    printf("[xray][subset-interaction] fixed pair margin = logit(ref_winner)-logit(low_winner); positive means the original winner beats the low-precision winner\n");
    printf("[xray][subset-interaction] mobius(mask)=sum_{sub subset mask} (-1)^(|mask|-|sub|) pair_margin(sub), exposing exact main/pair/triple/4-way repair interactions over this binary intervention cube\n");

    double margin[16] = {};
    int top1[16] = {};
    int total_flips[16] = {};
    int new_flips[16] = {};

    const int initial_flips = xray_count_flips(low);
    for (int mask = 0; mask < 16; ++mask) {
        cudaCheck(cudaMemcpy(patched_l00, low_l00_residual3,
                             state_bytes, cudaMemcpyDeviceToDevice));
        if (mask != 0) {
            xray_repair_l00_mask(&model, low_l00_residual2, low_l00_gelu,
                                 patched_l00, rows, mask, C);
        }
        xray_forward_from_residual3(&model, patched_l00, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());

        std::vector<RescueDecisionStats> repaired =
            xray_compare_to_reference(ref_logits, &model, d_stats, B, T);
        margin[mask] = xray_margin_at(&model, target, ref_winner, low_winner);
        top1[mask] = repaired[target].cur1;
        total_flips[mask] = xray_count_flips(repaired);
        int nf = 0;
        for (int i = 0; i < N; ++i) {
            const bool before = low[i].cur1 != low[i].ref1;
            const bool after = repaired[i].cur1 != repaired[i].ref1;
            if (!before && after) ++nf;
        }
        new_flips[mask] = nf;

        char bits[5];
        for (int i = 0; i < 4; ++i) bits[i] = ((mask >> i) & 1) ? '1' : '0';
        bits[4] = '\0';
        printf("[xray][subset-interaction-result] mask=%2d bits=%s n=%d rows={",
               mask, bits, xray_popcount4(mask));
        bool first = true;
        for (int i = 0; i < 4; ++i) {
            if (((mask >> i) & 1) == 0) continue;
            if (!first) printf(",");
            printf("%d", local_rows[i]);
            first = false;
        }
        printf("} target_top1=%d pair_margin=%+.9e target_rescued=%d total_flips=%d initial_flips=%d new_flips=%d\n",
               top1[mask], margin[mask], top1[mask] == ref_winner ? 1 : 0,
               total_flips[mask], initial_flips, new_flips[mask]);
    }

    int best_mask = -1;
    int best_n = 99;
    for (int mask = 1; mask < 16; ++mask) {
        if (top1[mask] != ref_winner || new_flips[mask] != 0) continue;
        const int n = xray_popcount4(mask);
        if (n < best_n) { best_n = n; best_mask = mask; }
    }
    if (best_mask >= 0) {
        printf("[xray][subset-interaction-min] mask=%d n=%d pair_margin=%+.9e\n",
               best_mask, best_n, margin[best_mask]);
    } else {
        printf("[xray][subset-interaction-min] no subset in the 4-token window rescues the target without collateral flips\n");
    }

    double mobius[16] = {};
    for (int mask = 0; mask < 16; ++mask) {
        double mu = 0.0;
        for (int sub = mask;; sub = (sub - 1) & mask) {
            const int parity = (xray_popcount4(mask) - xray_popcount4(sub)) & 1;
            mu += parity ? -margin[sub] : margin[sub];
            if (sub == 0) break;
        }
        mobius[mask] = mu;
        if (mask != 0) {
            printf("[xray][subset-interaction-mobius] mask=%2d order=%d coefficient=%+.9e\n",
                   mask, xray_popcount4(mask), mu);
        }
    }

    double abs_by_order[5] = {};
    for (int mask = 1; mask < 16; ++mask)
        abs_by_order[xray_popcount4(mask)] += fabs(mobius[mask]);
    printf("[xray][subset-interaction-order] abs_main=%.9e abs_pair=%.9e abs_triple=%.9e abs_fourway=%.9e\n",
           abs_by_order[1], abs_by_order[2], abs_by_order[3], abs_by_order[4]);

    cudaCheck(cudaFree(patched_l00));
    cudaCheck(cudaFree(low_l00_residual3));
    cudaCheck(cudaFree(low_l00_gelu));
    cudaCheck(cudaFree(low_l00_residual2));
    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
