#pragma push_macro("main")
#undef main
#define main xray_decision_margin_embedded_main
#include "decision_margin_probe.cu"
#undef main
#pragma pop_macro("main")

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

struct RescueRow {
    int token;
    MarginStats s;
};

static void xray_collect_rescue_case(const char* label,
                                     GPT2* model,
                                     DataLoader* loader,
                                     int B, int T,
                                     int layer,
                                     XrayComponent component,
                                     const float* ref_logits,
                                     MarginStats* d_stats) {
    const int N = B * T;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(model, loader->inputs, NULL, B, T, layer, component);
    cudaCheck(cudaDeviceSynchronize());

    const int block = 128;
    decision_margin_kernel<<<CEIL_DIV(N, block), block>>>(
        ref_logits, model->acts.output, d_stats, N, V, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    std::vector<MarginStats> stats(N);
    cudaCheck(cudaMemcpy(stats.data(), d_stats, N * sizeof(MarginStats), cudaMemcpyDeviceToHost));

    int flips = 0;
    for (int i = 0; i < N; ++i) flips += stats[i].ref1 != stats[i].cur1;

    std::vector<RescueRow> by_cur_margin;
    by_cur_margin.reserve(N);
    for (int i = 0; i < N; ++i) by_cur_margin.push_back(RescueRow{i, stats[i]});
    std::sort(by_cur_margin.begin(), by_cur_margin.end(), [](const RescueRow& a, const RescueRow& b) {
        if (a.s.cur_margin != b.s.cur_margin) return a.s.cur_margin < b.s.cur_margin;
        return a.token < b.token;
    });

    printf("[xray][precision-rescue] %-13s component=%-7s L=%02d flips=%d/%d\n",
           label, component_name(component), layer, flips, N);
    printf("[xray][precision-rescue] runtime gate uses only the low-precision result: rescue iff current top1 margin <= threshold\n");
    printf("[xray][precision-rescue] rescued tokens are scored as if a precise fallback restores the reference decision; this probe tests screening feasibility, not fallback latency\n");

    if (flips > 0) {
        float catch_all_threshold = 0.0f;
        for (int i = 0; i < N; ++i) {
            if (stats[i].ref1 != stats[i].cur1)
                catch_all_threshold = fmaxf(catch_all_threshold, stats[i].cur_margin);
        }
        int triggered = 0, caught = 0;
        for (int i = 0; i < N; ++i) {
            if (stats[i].cur_margin <= catch_all_threshold) {
                ++triggered;
                if (stats[i].ref1 != stats[i].cur1) ++caught;
            }
        }
        printf("[xray][precision-rescue-min] %-13s catch_all_threshold=%.9e triggered=%d/%d coverage=%.6f caught=%d/%d false_rescues=%d\n",
               label, catch_all_threshold, triggered, N, (double)triggered / N,
               caught, flips, triggered - caught);
    } else {
        printf("[xray][precision-rescue-min] %-13s no flips in this case; catch-all threshold undefined\n", label);
    }

    const float thresholds[] = {
        1.0e-4f, 2.0e-4f, 5.0e-4f, 1.0e-3f,
        2.0e-3f, 5.0e-3f, 1.0e-2f, 2.0e-2f
    };
    for (float th : thresholds) {
        int triggered = 0;
        int caught = 0;
        for (int i = 0; i < N; ++i) {
            if (stats[i].cur_margin <= th) {
                ++triggered;
                if (stats[i].ref1 != stats[i].cur1) ++caught;
            }
        }
        const int residual_flips = flips - caught;
        const double recall = flips ? (double)caught / flips : 0.0;
        const double rescue_precision = triggered ? (double)caught / triggered : 0.0;
        printf("[xray][precision-rescue-threshold] %-13s th=%.1e triggered=%4d/%d coverage=%.6f caught=%d/%d recall=%.6f rescue_precision=%.6f residual_flips=%d false_rescues=%d\n",
               label, th, triggered, N, (double)triggered / N,
               caught, flips, recall, rescue_precision, residual_flips,
               triggered - caught);
    }

    int rank = 0;
    for (const RescueRow& r : by_cur_margin) {
        ++rank;
        if (r.s.ref1 != r.s.cur1) {
            printf("[xray][precision-rescue-flip] %-13s tok=%04d b=%d t=%d ref=%d cur=%d ref_margin=%.9e cur_margin=%.9e current_margin_rank=%d/%d percentile=%.6f\n",
                   label, r.token, r.token / T, r.token % T,
                   r.s.ref1, r.s.cur1, r.s.ref_margin, r.s.cur_margin,
                   rank, N, (double)rank / N);
        }
    }
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
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
    const size_t logits_elems = (size_t)N * model.config.padded_vocab_size;

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float* ref_logits = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output,
                         logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    MarginStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(MarginStats)));

    printf("[xray][precision-rescue] test a deployable first-stage signal for adaptive precision: current low-precision top1 margin\n");
    printf("[xray][precision-rescue] question: can rare execution-induced decision flips be captured by rescuing only a small low-margin subset?\n");

    xray_collect_rescue_case("qkv-L02", &model, &loader, B, T, 2, XRAY_QKV, ref_logits, d_stats);
    xray_collect_rescue_case("fcproj-L00", &model, &loader, B, T, 0, XRAY_FCPROJ, ref_logits, d_stats);

    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
