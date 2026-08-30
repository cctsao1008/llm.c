#pragma push_macro("main")
#undef main
#define main xray_causal_cone_embedded_main
#include "adaptive_precision_causal_cone_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cstdio>
#include <vector>

struct SingletonInfluenceRow {
    int local_t;
    double pair_margin;
    double delta_margin;
    int target_top1;
    int target_rescued;
    int total_flips;
    int new_flips;
};

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
    if (low[target].ref1 == low[target].cur1) {
        printf("[xray][singleton-influence] requested token=%d is not a flip in this run; ref=%d low=%d\n",
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

    const int ref_winner = low[target].ref1;
    const int low_winner = low[target].cur1;
    const double low_pair_margin = xray_margin_at(&model, target, ref_winner, low_winner);
    const int initial_flips = xray_count_flips(low);

    printf("[xray][singleton-influence] map the exact causal effect of repairing each single L00 fcproj row in the target causal prefix\n");
    printf("[xray][singleton-influence] target=%d b=%d t=%d ref=%d low=%d low_pair_margin=%+.9e prefix=[0,%d]\n",
           target, b, t, ref_winner, low_winner, low_pair_margin, t);
    printf("[xray][singleton-influence] each intervention repairs exactly one row with the original/custom FP32 fcproj kernel, then replays layers 1..end\n");
    printf("[xray][singleton-influence] delta_margin = repaired fixed-pair margin - low fixed-pair margin; this is an exact intervention effect, not a gradient estimate\n");

    std::vector<SingletonInfluenceRow> rows;
    rows.reserve(t + 1);

    for (int local = 0; local <= t; ++local) {
        cudaCheck(cudaMemcpy(patched_l00, low_l00_residual3,
                             state_bytes, cudaMemcpyDeviceToDevice));
        const int tok = b * T + local;
        xray_repair_l00_rows(&model, low_l00_residual2, low_l00_gelu,
                             patched_l00, tok, tok, C);
        xray_forward_from_residual3(&model, patched_l00, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());

        std::vector<RescueDecisionStats> repaired =
            xray_compare_to_reference(ref_logits, &model, d_stats, B, T);
        const double pair_margin = xray_margin_at(&model, target, ref_winner, low_winner);
        int nf = 0;
        for (int i = 0; i < N; ++i) {
            const bool before = low[i].cur1 != low[i].ref1;
            const bool after = repaired[i].cur1 != repaired[i].ref1;
            if (!before && after) ++nf;
        }
        SingletonInfluenceRow r;
        r.local_t = local;
        r.pair_margin = pair_margin;
        r.delta_margin = pair_margin - low_pair_margin;
        r.target_top1 = repaired[target].cur1;
        r.target_rescued = repaired[target].cur1 == ref_winner ? 1 : 0;
        r.total_flips = xray_count_flips(repaired);
        r.new_flips = nf;
        rows.push_back(r);

        printf("[xray][singleton-influence-result] local_t=%3d distance=%3d pair_margin=%+.9e delta_margin=%+.9e target_top1=%d target_rescued=%d total_flips=%d new_flips=%d\n",
               local, t - local, r.pair_margin, r.delta_margin,
               r.target_top1, r.target_rescued, r.total_flips, r.new_flips);
    }

    std::vector<SingletonInfluenceRow> ranked = rows;
    std::sort(ranked.begin(), ranked.end(), [](const SingletonInfluenceRow& a,
                                                const SingletonInfluenceRow& b) {
        const double aa = fabs(a.delta_margin);
        const double bb = fabs(b.delta_margin);
        if (aa != bb) return aa > bb;
        return a.local_t > b.local_t;
    });

    const int topk = std::min<int>(12, ranked.size());
    for (int i = 0; i < topk; ++i) {
        const auto& r = ranked[i];
        printf("[xray][singleton-influence-rank] rank=%2d local_t=%3d distance=%3d delta_margin=%+.9e pair_margin=%+.9e target_rescued=%d new_flips=%d\n",
               i + 1, r.local_t, t - r.local_t,
               r.delta_margin, r.pair_margin, r.target_rescued, r.new_flips);
    }

    int rescuers = 0;
    int earliest = -1;
    int latest = -1;
    for (const auto& r : rows) {
        if (!r.target_rescued || r.new_flips != 0) continue;
        ++rescuers;
        if (earliest < 0) earliest = r.local_t;
        latest = r.local_t;
    }
    printf("[xray][singleton-influence-summary] causal_prefix_rows=%d singleton_rescuers_no_collateral=%d earliest_rescuer=%d latest_rescuer=%d\n",
           t + 1, rescuers, earliest, latest);

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
