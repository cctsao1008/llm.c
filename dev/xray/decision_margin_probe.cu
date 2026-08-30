#define XRAY_COMPONENT_SENSITIVITY_EMBEDDED
#include "component_sensitivity_probe.cu"
#undef XRAY_COMPONENT_SENSITIVITY_EMBEDDED

#include <cfloat>
#include <cstdio>
#include <vector>

struct MarginStats {
    int ref1;
    int ref2;
    int cur1;
    int cur2;
    float ref_margin;
    float cur_margin;
    float refpair_cur_margin;
    float d_ref1;
    float d_ref2;
};

__global__ void decision_margin_kernel(const float* ref, const float* cur,
                                       MarginStats* out, int N, int V, int Vp) {
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

    MarginStats s;
    s.ref1 = r1;
    s.ref2 = r2;
    s.cur1 = c1;
    s.cur2 = c2;
    s.ref_margin = r1v - r2v;
    s.cur_margin = c1v - c2v;
    s.refpair_cur_margin = c[r1] - c[r2];
    s.d_ref1 = c[r1] - r[r1];
    s.d_ref2 = c[r2] - r[r2];
    out[token] = s;
}

static void run_margin_case(const char* label, GPT2* model, DataLoader* loader,
                            int B, int T, int layer, XrayComponent component,
                            const float* ref_logits, MarginStats* d_stats) {
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

    std::vector<MarginStats> stats(N);
    cudaCheck(cudaMemcpy(stats.data(), d_stats, N * sizeof(MarginStats), cudaMemcpyDeviceToHost));

    int changed = 0;
    double changed_margin_sum = 0.0;
    double unchanged_margin_sum = 0.0;
    float changed_margin_min = FLT_MAX;
    float changed_margin_max = 0.0f;
    int unchanged = 0;

    for (int i = 0; i < N; ++i) {
        const MarginStats& s = stats[i];
        if (s.ref1 != s.cur1) {
            ++changed;
            changed_margin_sum += s.ref_margin;
            changed_margin_min = fminf(changed_margin_min, s.ref_margin);
            changed_margin_max = fmaxf(changed_margin_max, s.ref_margin);
            printf("[xray][decision-flip] %-13s tok=%04d b=%d t=%d ref=%d runner=%d -> cur=%d ref_margin=%+.7f cur_margin=%+.7f refpair_cur_margin=%+.7f d_ref1=%+.7f d_ref2=%+.7f\n",
                   label, i, i / T, i % T,
                   s.ref1, s.ref2, s.cur1,
                   s.ref_margin, s.cur_margin, s.refpair_cur_margin,
                   s.d_ref1, s.d_ref2);
        } else {
            ++unchanged;
            unchanged_margin_sum += s.ref_margin;
        }
    }

    printf("[xray][decision-summary] %-13s component=%-7s L=%02d flips=%d/%d mean_ref_margin_flipped=%.7f min=%.7f max=%.7f mean_ref_margin_unchanged=%.7f\n",
           label, component_name(component), layer, changed, N,
           changed ? changed_margin_sum / changed : 0.0,
           changed ? changed_margin_min : 0.0,
           changed ? changed_margin_max : 0.0,
           unchanged ? unchanged_margin_sum / unchanged : 0.0);
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
    const size_t logits_elems = (size_t)N * model.config.padded_vocab_size;

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float* ref_logits = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output,
                         logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    MarginStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(MarginStats)));

    printf("[xray][decision] test whether rare top-1 flips are explained by small baseline decision margins\n");
    printf("[xray][decision] candidates: qkv L02 (largest representation/output displacement) vs fcproj L00 (more top1 flips)\n");

    run_margin_case("qkv-L02", &model, &loader, B, T, 2, XRAY_QKV, ref_logits, d_stats);
    run_margin_case("fcproj-L00", &model, &loader, B, T, 0, XRAY_FCPROJ, ref_logits, d_stats);

    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
