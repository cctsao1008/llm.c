#define main xray_forward_ab_embedded_main
#include "forward_ab_probe.cu"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct DiffStats {
    double rel_l2;
    double mean_abs;
    double max_abs;
    double cosine;
};

static DiffStats compare_host(const float* ref, const float* cur, size_t n) {
    long double ref2 = 0.0L;
    long double cur2 = 0.0L;
    long double diff2 = 0.0L;
    long double dot = 0.0L;
    long double abs_sum = 0.0L;
    double max_abs = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double a = ref[i];
        double b = cur[i];
        double d = b - a;
        ref2 += (long double)a * a;
        cur2 += (long double)b * b;
        diff2 += (long double)d * d;
        dot += (long double)a * b;
        abs_sum += fabs(d);
        max_abs = std::max(max_abs, fabs(d));
    }

    double rel_l2 = ref2 > 0.0L ? sqrt((double)(diff2 / ref2)) : 0.0;
    double cosine = (ref2 > 0.0L && cur2 > 0.0L)
        ? (double)(dot / sqrt(ref2 * cur2))
        : 1.0;
    return DiffStats{rel_l2, (double)(abs_sum / n), max_abs, cosine};
}

static void copy_device(std::vector<float>& dst, const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void run_candidate(GPT2* model, int* inputs, int B, int T, int mode) {
    if (mode == 0) {
        cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
        gpt2_forward_cublas(model, inputs, NULL, B, T);
    } else {
        cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        gpt2_forward_cublas(model, inputs, NULL, B, T);
    }
    cudaCheck(cudaDeviceSynchronize());
}

static void print_stage(const char* mode, int layer, const char* stage,
                        const std::vector<float>& ref_all,
                        const float* cur_all,
                        size_t layer_elems) {
    std::vector<float> cur(layer_elems);
    cudaCheck(cudaMemcpy(cur.data(), cur_all + (size_t)layer * layer_elems,
                         layer_elems * sizeof(float), cudaMemcpyDeviceToHost));
    const float* ref = ref_all.data() + (size_t)layer * layer_elems;
    DiffStats s = compare_host(ref, cur.data(), layer_elems);
    printf("[xray][layer-diff] %-11s L=%02d %-9s rel_l2=%10.6e mean_abs=%10.6e max_abs=%10.6e cosine=%.10f\n",
           mode, layer, stage, s.rel_l2, s.mean_abs, s.max_abs, s.cosine);
}

static void summarize_residual_growth(const char* mode,
                                      const std::vector<float>& ref_residual3,
                                      const float* cur_residual3,
                                      int L, size_t layer_elems) {
    std::vector<float> cur(layer_elems);
    double first = 0.0;
    double last = 0.0;
    double peak = 0.0;
    int peak_layer = -1;

    for (int l = 0; l < L; ++l) {
        cudaCheck(cudaMemcpy(cur.data(), cur_residual3 + (size_t)l * layer_elems,
                             layer_elems * sizeof(float), cudaMemcpyDeviceToHost));
        const float* ref = ref_residual3.data() + (size_t)l * layer_elems;
        DiffStats s = compare_host(ref, cur.data(), layer_elems);
        if (l == 0) first = s.rel_l2;
        if (l == L - 1) last = s.rel_l2;
        if (s.rel_l2 > peak) {
            peak = s.rel_l2;
            peak_layer = l;
        }
    }

    double growth = first > 0.0 ? last / first : 0.0;
    printf("[xray][layer-summary] %-11s residual3 first=%10.6e last=%10.6e growth=%8.3fx peak=%10.6e@L%02d\n",
           mode, first, last, growth, peak, peak_layer);
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    if (B <= 0 || T <= 0) {
        fprintf(stderr, "usage: %s [B=4] [T=512]\n", argv[0]);
        return 2;
    }

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    cublasCheck(cublasCreate(&cublas_handle));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    DataLoader loader;
    dataloader_init(&loader,
                    "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
                    B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    // Baseline: exact original forward path. Its custom GEMMs use ordinary FP32
    // arithmetic while attention's cuBLAS operations retain TF32 mode on Ada.
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const size_t layer_elems = (size_t)B * T * C;
    const size_t all_elems = (size_t)L * layer_elems;

    std::vector<float> ref_attproj;
    std::vector<float> ref_fcproj;
    std::vector<float> ref_residual3;
    copy_device(ref_attproj, model.acts.attproj, all_elems);
    copy_device(ref_fcproj, model.acts.fcproj, all_elems);
    copy_device(ref_residual3, model.acts.residual3, all_elems);

    printf("[xray][layer-diff] device=%s cc=%d.%d B=%d T=%d layers=%d\n",
           prop.name, prop.major, prop.minor, B, T, L);
    printf("[xray][layer-diff] baseline=original custom forward; candidates replace all forward GEMMs with cuBLAS\n");
    printf("[xray][layer-diff] stages: attproj=attention branch output, fcproj=MLP branch output, residual3=block output\n");

    const char* names[2] = {"cuBLAS-FP32", "cuBLAS-TF32"};
    for (int mode = 0; mode < 2; ++mode) {
        run_candidate(&model, loader.inputs, B, T, mode);
        for (int l = 0; l < L; ++l) {
            print_stage(names[mode], l, "attproj", ref_attproj, model.acts.attproj, layer_elems);
            print_stage(names[mode], l, "fcproj", ref_fcproj, model.acts.fcproj, layer_elems);
            print_stage(names[mode], l, "residual3", ref_residual3, model.acts.residual3, layer_elems);
        }
        summarize_residual_growth(names[mode], ref_residual3, model.acts.residual3, L, layer_elems);
    }

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
