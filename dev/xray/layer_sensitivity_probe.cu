#define main xray_forward_ab_embedded_main
#include "forward_ab_probe.cu"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

enum XrayBranch {
    XRAY_BRANCH_ATTN = 0,
    XRAY_BRANCH_MLP = 1,
};

struct XrayDiff {
    double rel_l2;
    double mean_abs;
    double max_abs;
};

static XrayDiff compare_vec(const std::vector<float>& ref, const std::vector<float>& cur) {
    long double ref2 = 0.0L;
    long double diff2 = 0.0L;
    long double abs_sum = 0.0L;
    double max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = ref[i];
        double d = (double)cur[i] - a;
        ref2 += (long double)a * a;
        diff2 += (long double)d * d;
        abs_sum += fabs(d);
        max_abs = std::max(max_abs, fabs(d));
    }
    return XrayDiff{
        ref2 > 0.0L ? sqrt((double)(diff2 / ref2)) : 0.0,
        ref.empty() ? 0.0 : (double)(abs_sum / ref.size()),
        max_abs
    };
}

static void copy_device_vec(std::vector<float>& dst, const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void xray_choose_matmul(float* out,
                               const float* inp,
                               const float* weight,
                               const float* bias,
                               int B, int T, int C, int OC,
                               bool use_cublas) {
    if (use_cublas) {
        xray_matmul_cublas(out, inp, weight, bias, B, T, C, OC);
    } else {
        matmul_forward(out, inp, weight, bias, B, T, C, OC);
    }
}

// Run the original forward graph, except that exactly one transformer layer and
// one branch (attention or MLP) uses cuBLAS TF32 for its two GEMMs. The final
// vocabulary projection stays on the original custom kernel so the measured
// effect is attributable to the selected layer/branch only.
static void gpt2_forward_one_branch_tf32(GPT2* model, int* inputs, int* targets,
                                         int B, int T, int selected_layer,
                                         XrayBranch selected_branch) {
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

        bool perturb_attn = l == selected_layer && selected_branch == XRAY_BRANCH_ATTN;
        bool perturb_mlp = l == selected_layer && selected_branch == XRAY_BRANCH_MLP;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        xray_choose_matmul(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C, perturb_attn);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        xray_choose_matmul(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C, perturb_attn);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        xray_choose_matmul(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C, perturb_mlp);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
        xray_choose_matmul(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C, perturb_mlp);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }

    float* residual = acts.residual3 + (size_t)(L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                      residual, params.lnfw, params.lnfb, B, T, C);
    // Keep classifier identical to baseline: original custom FP32 GEMM.
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

#ifdef XRAY_LAYER_SENSITIVITY_EMBEDDED
int xray_layer_sensitivity_embedded_main(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
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
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);
    dataloader_next_batch(&loader);

    // Baseline is the unmodified original forward path.
    gpt2_forward(&model, loader.inputs, loader.targets, B, T);
    cudaCheck(cudaDeviceSynchronize());
    float baseline_loss = model.mean_loss;

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const size_t final_elems = (size_t)B * T * C;
    const float* final_residual = model.acts.residual3 + (size_t)(L - 1) * final_elems;
    std::vector<float> ref_final;
    copy_device_vec(ref_final, final_residual, final_elems);

    printf("[xray][layer-causal] device=%s cc=%d.%d B=%d T=%d layers=%d baseline_loss=%.8f\n",
           prop.name, prop.major, prop.minor, B, T, L, baseline_loss);
    printf("[xray][layer-causal] exactly one layer/branch switches from custom FP32 GEMMs to cuBLAS TF32; all other forward GEMMs stay original\n");

    double strongest_loss = -1.0;
    int strongest_loss_layer = -1;
    const char* strongest_loss_branch = "";
    double strongest_residual = -1.0;
    int strongest_residual_layer = -1;
    const char* strongest_residual_branch = "";

    for (int branch_i = 0; branch_i < 2; ++branch_i) {
        XrayBranch branch = (XrayBranch)branch_i;
        const char* branch_name = branch == XRAY_BRANCH_ATTN ? "attn" : "mlp";
        for (int l = 0; l < L; ++l) {
            cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
            gpt2_forward_one_branch_tf32(&model, loader.inputs, loader.targets, B, T, l, branch);
            cudaCheck(cudaDeviceSynchronize());

            std::vector<float> cur_final;
            copy_device_vec(cur_final,
                            model.acts.residual3 + (size_t)(L - 1) * final_elems,
                            final_elems);
            XrayDiff d = compare_vec(ref_final, cur_final);
            double dloss = (double)model.mean_loss - baseline_loss;

            printf("[xray][layer-causal] %-4s L=%02d final_residual rel_l2=%10.6e mean_abs=%10.6e max_abs=%10.6e dloss=%+.8f\n",
                   branch_name, l, d.rel_l2, d.mean_abs, d.max_abs, dloss);

            if (fabs(dloss) > strongest_loss) {
                strongest_loss = fabs(dloss);
                strongest_loss_layer = l;
                strongest_loss_branch = branch_name;
            }
            if (d.rel_l2 > strongest_residual) {
                strongest_residual = d.rel_l2;
                strongest_residual_layer = l;
                strongest_residual_branch = branch_name;
            }
        }
    }

    printf("[xray][layer-causal-summary] strongest |dloss|=%10.6e at %s L%02d; strongest final residual rel_l2=%10.6e at %s L%02d\n",
           strongest_loss, strongest_loss_branch, strongest_loss_layer,
           strongest_residual, strongest_residual_branch, strongest_residual_layer);

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
