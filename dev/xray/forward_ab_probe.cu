#define TESTING
#include "../../train_gpt2_fp32.cu"

#include <nvtx3/nvToolsExt.h>
#include <cmath>
#include <cstdio>

__global__ void xray_add_bias_kernel(float* out, const float* bias, int N, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC;
    if (idx < total) {
        out[idx] += bias[idx % OC];
    }
}

static void xray_matmul_cublas(float* out,
                               const float* inp,
                               const float* weight,
                               const float* bias,
                               int B, int T, int C, int OC) {
    const int N = B * T;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major out[N,OC] = inp[N,C] * weight[OC,C]^T.
    // cuBLAS is column-major, therefore compute out^T = weight * inp^T.
    cublasCheck(cublasSgemm(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            OC, N, C,
                            &alpha,
                            weight, C,
                            inp, C,
                            &beta,
                            out, OC));

    if (bias != nullptr) {
        const int block = 256;
        const int grid = CEIL_DIV(N * OC, block);
        xray_add_bias_kernel<<<grid, block>>>(out, bias, N, OC);
        cudaCheck(cudaGetLastError());
    }
}

static void gpt2_forward_cublas(GPT2 *model, int* inputs, int* targets, int B, int T) {
    if (model->params_memory == NULL) {
        fprintf(stderr, "xray: model not initialized\n");
        exit(EXIT_FAILURE);
    }

    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    for (int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) assert(0 <= targets[i] && targets[i] < V);
    }

    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) num_activations += model->act_sizes[i];
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    } else if (B != model->batch_size || T != model->seq_len) {
        fprintf(stderr, "xray: B/T mismatch\n");
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL) {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float* residual;

    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);

    for (int l = 0; l < L; l++) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3 * C * C;
        float* l_qkvb = params.qkvb + l * 3 * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4 * C * C;
        float* l_fcb = params.fcb + l * 4 * C;
        float* l_fcprojw = params.fcprojw + l * C * 4 * C;
        float* l_fcprojb = params.fcprojb + l * C;

        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3 * C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4 * C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        float* scratch = acts.output;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        xray_matmul_cublas(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        xray_matmul_cublas(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        xray_matmul_cublas(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
        xray_matmul_cublas(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }

    residual = acts.residual3 + (L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    xray_matmul_cublas(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    if (targets != NULL) {
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++) mean_loss += model->cpu_losses[i];
        model->mean_loss = mean_loss / (B * T);
    } else {
        model->mean_loss = -1.0f;
    }
}

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

template <typename Fn>
static double bench_forward(const char* tag, Fn&& fn, int warmup, int iters, float* loss_out) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaCheck(cudaDeviceSynchronize());
    nvtxRangePushA(tag);
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) fn();
    cudaCheck(cudaDeviceSynchronize());
    double t1 = now_ms();
    nvtxRangePop();
    *loss_out = 0.0f;
    return (t1 - t0) / iters;
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    int iters = argc > 3 ? atoi(argv[3]) : 8;
    if (B <= 0 || T <= 0 || iters <= 0) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [iters=8]\n", argv[0]);
        return 2;
    }

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    cublasCheck(cublasCreate(&cublas_handle));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    // Allocate/warm the original path once before timing either candidate.
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward(&model, loader.inputs, loader.targets, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float custom_loss = 0.0f;
    float fp32_loss = 0.0f;
    float tf32_loss = 0.0f;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    double custom_ms = bench_forward("xray/forward/custom", [&] {
        gpt2_forward(&model, loader.inputs, loader.targets, B, T);
    }, 2, iters, &custom_loss);
    custom_loss = model.mean_loss;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    double fp32_ms = bench_forward("xray/forward/cublas_fp32", [&] {
        gpt2_forward_cublas(&model, loader.inputs, loader.targets, B, T);
    }, 2, iters, &fp32_loss);
    fp32_loss = model.mean_loss;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    double tf32_ms = bench_forward("xray/forward/cublas_tf32", [&] {
        gpt2_forward_cublas(&model, loader.inputs, loader.targets, B, T);
    }, 2, iters, &tf32_loss);
    tf32_loss = model.mean_loss;

    printf("[xray][forward-ab] device=%s cc=%d.%d B=%d T=%d iters=%d\n",
           prop.name, prop.major, prop.minor, B, T, iters);
    printf("[xray][forward-ab] custom       %8.3f ms loss=%.8f\n", custom_ms, custom_loss);
    printf("[xray][forward-ab] cuBLAS-FP32  %8.3f ms loss=%.8f speedup=%5.2fx dloss=%+.8f\n",
           fp32_ms, fp32_loss, custom_ms / fp32_ms, fp32_loss - custom_loss);
    printf("[xray][forward-ab] cuBLAS-TF32  %8.3f ms loss=%.8f speedup=%5.2fx dloss=%+.8f\n",
           tf32_ms, tf32_loss, custom_ms / tf32_ms, tf32_loss - custom_loss);

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
