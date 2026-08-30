#define XRAY_COMPONENT_SENSITIVITY_EMBEDDED
#include "component_sensitivity_probe.cu"
#undef XRAY_COMPONENT_SENSITIVITY_EMBEDDED

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void xray_blend_kernel(float* out,
                                  const float* custom,
                                  const float* tf32,
                                  size_t n,
                                  float alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (alpha == 0.0f) {
        out[i] = custom[i];
    } else if (alpha == 1.0f) {
        out[i] = tf32[i];
    } else {
        float a = custom[i];
        out[i] = a + alpha * (tf32[i] - a);
    }
}

struct ResponseStats {
    double response_rel_l2;
    double gain_vs_alpha1;
    double linearity_residual;
};

static ResponseStats compare_response(const std::vector<float>& baseline,
                                      const std::vector<float>& alpha1,
                                      const std::vector<float>& current,
                                      double alpha) {
    long double base2 = 0.0L;
    long double d1_2 = 0.0L;
    long double da_2 = 0.0L;
    long double linerr2 = 0.0L;

    for (size_t i = 0; i < baseline.size(); ++i) {
        long double b = baseline[i];
        long double d1 = (long double)alpha1[i] - b;
        long double da = (long double)current[i] - b;
        long double e = da - (long double)alpha * d1;
        base2 += b * b;
        d1_2 += d1 * d1;
        da_2 += da * da;
        linerr2 += e * e;
    }

    double response_rel = base2 > 0.0L ? sqrt((double)(da_2 / base2)) : 0.0;
    long double expected2 = (long double)alpha * alpha * d1_2;
    double gain = expected2 > 0.0L ? sqrt((double)(da_2 / expected2)) : 0.0;
    double linres = expected2 > 0.0L ? sqrt((double)(linerr2 / expected2)) : 0.0;
    return ResponseStats{response_rel, gain, linres};
}

// Run the original forward graph, except at QKV of one selected layer we
// interpolate along the *natural implementation perturbation* produced by
// custom FP32 GEMM -> cuBLAS TF32 on the exact same input and weights:
//   qkv(alpha) = qkv_custom + alpha * (qkv_tf32 - qkv_custom)
// alpha=0 is the original custom execution; alpha=1 is exact cuBLAS-TF32 QKV.
static void gpt2_forward_qkv_natural_alpha(GPT2* model,
                                           int* inputs,
                                           int* targets,
                                           int B, int T,
                                           int selected_layer,
                                           float alpha,
                                           float* qkv_custom,
                                           float* qkv_tf32) {
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; ++i) num_activations += model->act_sizes[i];
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    }

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL) {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);

    const size_t qkv_elems = (size_t)B * T * 3 * C;
    const int blend_block = 256;
    const int blend_grid = (int)((qkv_elems + blend_block - 1) / blend_block);

    for (int l = 0; l < L; ++l) {
        float* residual = l == 0 ? acts.encoded : acts.residual3 + (size_t)(l - 1) * B * T * C;

        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float* l_qkvb = params.qkvb + (size_t)l * 3 * C;
        float* l_attprojw = params.attprojw + (size_t)l * C * C;
        float* l_attprojb = params.attprojb + (size_t)l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float* l_fcb = params.fcb + (size_t)l * 4 * C;
        float* l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float* l_fcprojb = params.fcprojb + (size_t)l * C;

        float* l_ln1 = acts.ln1 + (size_t)l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + (size_t)l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + (size_t)l * B * T;
        float* l_qkvr = acts.qkvr + (size_t)l * B * T * 3 * C;
        float* l_atty = acts.atty + (size_t)l * B * T * C;
        float* l_att = acts.att + (size_t)l * B * NH * T * T;
        float* l_attproj = acts.attproj + (size_t)l * B * T * C;
        float* l_residual2 = acts.residual2 + (size_t)l * B * T * C;
        float* l_ln2 = acts.ln2 + (size_t)l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + (size_t)l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + (size_t)l * B * T;
        float* l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float* l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float* l_fcproj = acts.fcproj + (size_t)l * B * T * C;
        float* l_residual3 = acts.residual3 + (size_t)l * B * T * C;
        float* scratch = acts.output;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);

        if (l == selected_layer) {
            matmul_forward(qkv_custom, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
            xray_matmul_cublas(qkv_tf32, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            xray_blend_kernel<<<blend_grid, blend_block>>>(scratch, qkv_custom, qkv_tf32, qkv_elems, alpha);
            cudaCheck(cudaGetLastError());
        } else {
            matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        }

        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }

    float* residual = acts.residual3 + (size_t)(L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                      residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    if (targets != NULL) {
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; ++i) mean_loss += model->cpu_losses[i];
        model->mean_loss = mean_loss / (B * T);
    } else {
        model->mean_loss = -1.0f;
    }
}

static void capture_final_residual(std::vector<float>& dst, GPT2* model, int B, int T) {
    int L = model->config.num_layers;
    int C = model->config.channels;
    size_t n = (size_t)B * T * C;
    dst.resize(n);
    const float* src = model->acts.residual3 + (size_t)(L - 1) * n;
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    const int selected_layer = 2;

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

    size_t qkv_elems = (size_t)B * T * 3 * model.config.channels;
    float* qkv_custom = nullptr;
    float* qkv_tf32 = nullptr;
    cudaCheck(cudaMalloc((void**)&qkv_custom, qkv_elems * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&qkv_tf32, qkv_elems * sizeof(float)));

    std::vector<float> baseline;
    std::vector<float> alpha1;

    gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                   B, T, selected_layer, 0.0f, qkv_custom, qkv_tf32);
    cudaCheck(cudaDeviceSynchronize());
    double loss0 = model.mean_loss;
    capture_final_residual(baseline, &model, B, T);

    gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                   B, T, selected_layer, 1.0f, qkv_custom, qkv_tf32);
    cudaCheck(cudaDeviceSynchronize());
    double loss1 = model.mean_loss;
    capture_final_residual(alpha1, &model, B, T);
    double dloss1 = loss1 - loss0;

    printf("[xray][natural-linearity] source=qkv L02 natural_delta=(cuBLAS-TF32 - custom-FP32) on identical inputs/weights\n");
    printf("[xray][natural-linearity] alpha=0 original custom; alpha=1 exact TF32 QKV; test whether downstream response scales linearly\n");
    printf("[xray][natural-linearity] baseline_loss=%.8f alpha1_loss=%.8f alpha1_dloss=%+.8e\n",
           loss0, loss1, dloss1);

    const float alphas[] = {-8.0f, -4.0f, -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
    const int na = (int)(sizeof(alphas) / sizeof(alphas[0]));

    for (int ai = 0; ai < na; ++ai) {
        float alpha = alphas[ai];
        std::vector<float> cur;
        double loss = 0.0;

        if (alpha == 1.0f) {
            cur = alpha1;
            loss = loss1;
        } else {
            gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                           B, T, selected_layer, alpha, qkv_custom, qkv_tf32);
            cudaCheck(cudaDeviceSynchronize());
            loss = model.mean_loss;
            capture_final_residual(cur, &model, B, T);
        }

        ResponseStats s = compare_response(baseline, alpha1, cur, alpha);
        double dloss = loss - loss0;
        double loss_gain = (alpha != 0.0f && dloss1 != 0.0) ? dloss / (alpha * dloss1) : 0.0;
        double loss_linerr = (alpha != 0.0f && dloss1 != 0.0)
            ? fabs(dloss - alpha * dloss1) / fabs(alpha * dloss1)
            : 0.0;

        printf("[xray][natural-linearity] alpha=%+6.2f final_residual_rel_l2=%10.6e gain_vs_linear=%8.4f linearity_residual=%10.6e dloss=%+.8e loss_gain=%+8.4f loss_linearity_residual=%10.6e\n",
               alpha, s.response_rel_l2, s.gain_vs_alpha1, s.linearity_residual,
               dloss, loss_gain, loss_linerr);
    }

    cudaCheck(cudaFree(qkv_tf32));
    cudaCheck(cudaFree(qkv_custom));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
