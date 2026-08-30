#define XRAY_COMPONENT_SENSITIVITY_EMBEDDED
#include "component_sensitivity_probe.cu"
#undef XRAY_COMPONENT_SENSITIVITY_EMBEDDED

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void xray_source_blend(float* out, const float* custom, const float* tf32,
                                  size_t n, float alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (alpha == 0.0f) out[i] = custom[i];
    else if (alpha == 1.0f) out[i] = tf32[i];
    else {
        float a = custom[i];
        out[i] = a + alpha * (tf32[i] - a);
    }
}

struct SourceStats {
    double gain;
    double linearity_residual;
    double cosine;
    double zero_fraction;
    double sign_mismatch_fraction;
};

static SourceStats source_stats(const std::vector<float>& custom,
                                const std::vector<float>& tf32,
                                const std::vector<float>& blended,
                                double alpha) {
    long double expected2 = 0.0L, actual2 = 0.0L, err2 = 0.0L, dot = 0.0L;
    size_t zero_count = 0, sign_mismatch = 0, nonzero_expected = 0;
    for (size_t i = 0; i < custom.size(); ++i) {
        long double d = (long double)tf32[i] - custom[i];
        long double e = (long double)alpha * d;
        long double a = (long double)blended[i] - custom[i];
        long double r = a - e;
        expected2 += e * e;
        actual2 += a * a;
        err2 += r * r;
        dot += a * e;
        if (e != 0.0L) {
            ++nonzero_expected;
            if (a == 0.0L) ++zero_count;
            else if ((a > 0.0L) != (e > 0.0L)) ++sign_mismatch;
        }
    }
    double gain = expected2 > 0.0L ? sqrt((double)(actual2 / expected2)) : 0.0;
    double linres = expected2 > 0.0L ? sqrt((double)(err2 / expected2)) : 0.0;
    double cosine = (actual2 > 0.0L && expected2 > 0.0L)
        ? (double)(dot / sqrt((double)(actual2 * expected2))) : 0.0;
    double zf = nonzero_expected ? (double)zero_count / nonzero_expected : 0.0;
    double sf = nonzero_expected ? (double)sign_mismatch / nonzero_expected : 0.0;
    return SourceStats{gain, linres, cosine, zf, sf};
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    const int selected_layer = 2;
    if (B <= 0 || T <= 0) return 2;

    cudaCheck(cudaSetDevice(0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    // Allocate activations and run the original graph once so L2's input residual exists.
    gpt2_forward(&model, loader.inputs, loader.targets, B, T);
    cudaCheck(cudaDeviceSynchronize());

    int C = model.config.channels;
    ParameterTensors p = model.params;
    ActivationTensors a = model.acts;
    size_t BTC = (size_t)B * T * C;
    const float* residual = a.residual3 + (size_t)(selected_layer - 1) * BTC;
    float* ln1 = a.ln1 + (size_t)selected_layer * BTC;
    float* ln1_mean = a.ln1_mean + (size_t)selected_layer * B * T;
    float* ln1_rstd = a.ln1_rstd + (size_t)selected_layer * B * T;
    const float* ln1w = p.ln1w + selected_layer * C;
    const float* ln1b = p.ln1b + selected_layer * C;
    const float* qkvw = p.qkvw + (size_t)selected_layer * 3 * C * C;
    const float* qkvb = p.qkvb + (size_t)selected_layer * 3 * C;

    layernorm_forward(ln1, ln1_mean, ln1_rstd, residual, ln1w, ln1b, B, T, C);

    size_t n = (size_t)B * T * 3 * C;
    float *d_custom=nullptr, *d_tf32=nullptr, *d_blend=nullptr;
    cudaCheck(cudaMalloc((void**)&d_custom, n*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_tf32, n*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_blend, n*sizeof(float)));
    matmul_forward(d_custom, ln1, qkvw, qkvb, B, T, C, 3*C);
    xray_matmul_cublas(d_tf32, ln1, qkvw, qkvb, B, T, C, 3*C);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> h_custom(n), h_tf32(n), h_blend(n);
    cudaCheck(cudaMemcpy(h_custom.data(), d_custom, n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_tf32.data(), d_tf32, n*sizeof(float), cudaMemcpyDeviceToHost));

    const float alphas[] = {-8.f,-4.f,-2.f,-1.f,-0.5f,-0.25f,0.125f,0.25f,0.5f,1.f,2.f,4.f,8.f,16.f};
    int block=256, grid=(int)((n+block-1)/block);
    printf("[xray][source-fidelity] source=qkv L02; compare realized float32 injection with ideal alpha*(tf32-custom)\n");
    for (float alpha : alphas) {
        xray_source_blend<<<grid,block>>>(d_blend,d_custom,d_tf32,n,alpha);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_blend.data(), d_blend, n*sizeof(float), cudaMemcpyDeviceToHost));
        SourceStats s = source_stats(h_custom,h_tf32,h_blend,alpha);
        printf("[xray][source-fidelity] alpha=%+7.3f gain=%8.4f linearity_residual=%10.6e cosine=% .9f zeroed=%8.4f%% sign_mismatch=%8.4f%%\n",
               alpha,s.gain,s.linearity_residual,s.cosine,100.0*s.zero_fraction,100.0*s.sign_mismatch_fraction);
    }

    cudaCheck(cudaFree(d_blend));
    cudaCheck(cudaFree(d_tf32));
    cudaCheck(cudaFree(d_custom));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
