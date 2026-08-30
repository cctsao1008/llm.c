#pragma push_macro("main")
#undef main
#define main xray_actual_rescue_embedded_main
#include "adaptive_precision_actual_rescue_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cstdio>
#include <vector>

static void xray_repair_l00_rows(GPT2* model,
                                 float* l00_residual2,
                                 const float* l00_gelu,
                                 float* patched_l00,
                                 int first_token, int last_token,
                                 int C) {
    if (first_token > last_token) return;
    float* precise_row = nullptr;
    cudaCheck(cudaMalloc((void**)&precise_row, C * sizeof(float)));
    float* fcprojw = model->params.fcprojw;
    float* fcprojb = model->params.fcprojb;
    for (int tok = first_token; tok <= last_token; ++tok) {
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
    const int requested_token = argc > 3 ? atoi(argv[3]) : 1186;
    if (B <= 0 || T <= 0 || requested_token < 0 || requested_token >= B * T) {
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

    const int target = requested_token;
    const int b = target / T;
    const int t = target % T;
    if (low[target].ref1 == low[target].cur1) {
        printf("[xray][causal-cone] requested token=%d is not a flip in this run; ref=%d low=%d\n",
               target, low[target].ref1, low[target].cur1);
        cudaCheck(cudaFree(d_stats));
        cudaCheck(cudaFree(ref_logits));
        dataloader_free(&loader);
        gpt2_free(&model);
        cublasCheck(cublasDestroy(cublas_handle));
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

    printf("[xray][causal-cone] localize how much of the L00 causal prefix must be repaired to recover a low-precision decision flip\n");
    printf("[xray][causal-cone] target=%d b=%d t=%d ref=%d low=%d ref_margin=%.9e low_margin=%.9e\n",
           target, b, t, low[target].ref1, low[target].cur1,
           low[target].ref_margin, low[target].cur_margin);
    printf("[xray][causal-cone] each trial starts from the exact low L00 residual3, repairs a nested recent suffix [t-width+1,t] inside the target sequence with custom FP32 fcproj, then replays layers 1..end\n");
    printf("[xray][causal-cone] width=t+1 repairs the complete causal prefix and is the correctness control; future positions and other batch elements remain low because causal attention cannot affect the target from them\n");

    std::vector<int> widths = {1, 2, 4, 8, 16, 32, 64, 128, t + 1};
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    widths.erase(std::remove_if(widths.begin(), widths.end(), [t](int w) {
        return w <= 0 || w > t + 1;
    }), widths.end());

    const int initial_flips = xray_count_flips(low);
    for (int width : widths) {
        cudaCheck(cudaMemcpy(patched_l00, low_l00_residual3,
                             state_bytes, cudaMemcpyDeviceToDevice));
        const int local_first = t - width + 1;
        const int first = b * T + local_first;
        const int last = b * T + t;
        xray_repair_l00_rows(&model, low_l00_residual2, low_l00_gelu,
                             patched_l00, first, last, C);
        xray_forward_from_residual3(&model, patched_l00, 0, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<RescueDecisionStats> repaired =
            xray_compare_to_reference(ref_logits, &model, d_stats, B, T);
        const int flips = xray_count_flips(repaired);
        const bool recovered = repaired[target].cur1 == repaired[target].ref1;
        int new_flips = 0;
        for (int i = 0; i < N; ++i) {
            const bool before = low[i].cur1 != low[i].ref1;
            const bool after = repaired[i].cur1 != repaired[i].ref1;
            if (!before && after) ++new_flips;
        }
        printf("[xray][causal-cone-result] width=%3d local_range=[%3d,%3d] repaired_rows=%3d target_rescued=%d target_top1=%d target_margin=%.9e total_flips=%d initial_flips=%d new_flips=%d\n",
               width, local_first, t, width, recovered ? 1 : 0,
               repaired[target].cur1, repaired[target].cur_margin,
               flips, initial_flips, new_flips);
    }

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
