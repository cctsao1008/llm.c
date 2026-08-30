#define TESTING
#include "../../train_gpt2_fp32.cu"

#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct Shape {
    const char* name;
    int N;
    int C;
    int OC;
    int iters;
};

static double wall_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static void cublas_forward(float* out, const float* inp, const float* weight,
                           int N, int C, int OC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major: out[N,OC] = inp[N,C] * weight[OC,C]^T.
    // cuBLAS is column-major, so compute out^T = weight * inp^T.
    cublasCheck(cublasSgemm(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            OC, N, C,
                            &alpha,
                            weight, C,
                            inp, C,
                            &beta,
                            out, OC));
}

template <typename Fn>
static double bench(const char* range_name, Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaCheck(cudaDeviceSynchronize());

    nvtxRangePushA(range_name);
    double t0 = wall_ms();
    for (int i = 0; i < iters; ++i) fn();
    cudaCheck(cudaDeviceSynchronize());
    double t1 = wall_ms();
    nvtxRangePop();
    return (t1 - t0) / iters;
}

static void run_shape(const Shape& s) {
    size_t inp_elems = (size_t)s.N * s.C;
    size_t w_elems = (size_t)s.OC * s.C;
    size_t out_elems = (size_t)s.N * s.OC;

    float *inp = nullptr, *weight = nullptr, *out = nullptr;
    cudaCheck(cudaMalloc(&inp, inp_elems * sizeof(float)));
    cudaCheck(cudaMalloc(&weight, w_elems * sizeof(float)));
    cudaCheck(cudaMalloc(&out, out_elems * sizeof(float)));
    cudaCheck(cudaMemset(inp, 0, inp_elems * sizeof(float)));
    cudaCheck(cudaMemset(weight, 0, w_elems * sizeof(float)));
    cudaCheck(cudaMemset(out, 0, out_elems * sizeof(float)));

    char tag_custom[128], tag_tf32[128], tag_fp32[128];
    snprintf(tag_custom, sizeof(tag_custom), "xray/custom/%s", s.name);
    snprintf(tag_tf32, sizeof(tag_tf32), "xray/cublas_tf32/%s", s.name);
    snprintf(tag_fp32, sizeof(tag_fp32), "xray/cublas_fp32/%s", s.name);

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    double custom_ms = bench(tag_custom, [&] {
        int sqrt_block_size = 16;
        dim3 gridDim(CEIL_DIV(s.N, 8 * sqrt_block_size),
                     CEIL_DIV(s.OC, 8 * sqrt_block_size));
        dim3 blockDim(sqrt_block_size, sqrt_block_size);
        matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, nullptr, s.C, s.OC);
        cudaCheck(cudaGetLastError());
    }, 3, s.iters);

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    double tf32_ms = bench(tag_tf32, [&] {
        cublas_forward(out, inp, weight, s.N, s.C, s.OC);
    }, 3, s.iters);

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    double fp32_ms = bench(tag_fp32, [&] {
        cublas_forward(out, inp, weight, s.N, s.C, s.OC);
    }, 3, s.iters);

    double flops = 2.0 * (double)s.N * s.C * s.OC;
    auto tflops = [flops](double ms) { return flops / (ms * 1.0e9); };

    printf("[xray][matmul] %-10s N=%5d C=%4d OC=%5d | custom=%8.3f ms %6.2f TF/s | cuBLAS-TF32=%8.3f ms %6.2f TF/s | cuBLAS-FP32=%8.3f ms %6.2f TF/s | speedup TF32/custom=%5.2fx\n",
           s.name, s.N, s.C, s.OC,
           custom_ms, tflops(custom_ms),
           tf32_ms, tflops(tf32_ms),
           fp32_ms, tflops(fp32_ms),
           custom_ms / tf32_ms);

    cudaCheck(cudaFree(inp));
    cudaCheck(cudaFree(weight));
    cudaCheck(cudaFree(out));
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

    int N = B * T;
    printf("[xray][matmul] device=%s cc=%d.%d B=%d T=%d N=%d\n",
           prop.name, prop.major, prop.minor, B, T, N);
    printf("[xray][matmul] comparing the exact custom forward GEMM kernel used by train_gpt2_fp32.cu against cuBLAS on representative GPT-2 shapes\n");

    // Representative forward GEMMs in one GPT-2 block plus the vocabulary projection.
    std::vector<Shape> shapes = {
        {"qkv",        N, 768,  2304, 20},
        {"attproj",    N, 768,   768, 30},
        {"fc",         N, 768,  3072, 20},
        {"fcproj",     N, 3072,  768, 20},
        {"classifier", N, 768, 50304,  5},
    };

    for (const Shape& s : shapes) run_shape(s);

    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
