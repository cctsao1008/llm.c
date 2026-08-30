#define XRAY_COMPONENT_SENSITIVITY_EMBEDDED
#include "component_sensitivity_probe.cu"
#undef XRAY_COMPONENT_SENSITIVITY_EMBEDDED

#include <cmath>
#include <cstdio>
#include <vector>

struct TokenOutputStats {
    float rel_l2;
    float max_abs;
    float target_delta;
    int top1_ref;
    int top1_cur;
};

__global__ void compare_logits_per_token(const float* ref,
                                         const float* cur,
                                         const int* targets,
                                         TokenOutputStats* out,
                                         int N, int V, int Vp) {
    int token = blockIdx.x;
    if (token >= N) return;

    extern __shared__ unsigned char smem_raw[];
    float* ref2_s = reinterpret_cast<float*>(smem_raw);
    float* diff2_s = ref2_s + blockDim.x;
    float* maxdiff_s = diff2_s + blockDim.x;
    float* maxref_s = maxdiff_s + blockDim.x;
    float* maxcur_s = maxref_s + blockDim.x;
    int* argref_s = reinterpret_cast<int*>(maxcur_s + blockDim.x);
    int* argcur_s = argref_s + blockDim.x;

    const float* r = ref + (size_t)token * Vp;
    const float* c = cur + (size_t)token * Vp;

    float ref2 = 0.0f;
    float diff2 = 0.0f;
    float maxdiff = 0.0f;
    float maxref = -FLT_MAX;
    float maxcur = -FLT_MAX;
    int argref = -1;
    int argcur = -1;

    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float rv = r[i];
        float cv = c[i];
        float d = cv - rv;
        ref2 += rv * rv;
        diff2 += d * d;
        maxdiff = fmaxf(maxdiff, fabsf(d));
        if (rv > maxref) { maxref = rv; argref = i; }
        if (cv > maxcur) { maxcur = cv; argcur = i; }
    }

    int tid = threadIdx.x;
    ref2_s[tid] = ref2;
    diff2_s[tid] = diff2;
    maxdiff_s[tid] = maxdiff;
    maxref_s[tid] = maxref;
    maxcur_s[tid] = maxcur;
    argref_s[tid] = argref;
    argcur_s[tid] = argcur;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ref2_s[tid] += ref2_s[tid + stride];
            diff2_s[tid] += diff2_s[tid + stride];
            maxdiff_s[tid] = fmaxf(maxdiff_s[tid], maxdiff_s[tid + stride]);
            if (maxref_s[tid + stride] > maxref_s[tid]) {
                maxref_s[tid] = maxref_s[tid + stride];
                argref_s[tid] = argref_s[tid + stride];
            }
            if (maxcur_s[tid + stride] > maxcur_s[tid]) {
                maxcur_s[tid] = maxcur_s[tid + stride];
                argcur_s[tid] = argcur_s[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int target = targets[token];
        TokenOutputStats s;
        s.rel_l2 = ref2_s[0] > 0.0f ? sqrtf(diff2_s[0] / ref2_s[0]) : 0.0f;
        s.max_abs = maxdiff_s[0];
        s.target_delta = c[target] - r[target];
        s.top1_ref = argref_s[0];
        s.top1_cur = argcur_s[0];
        out[token] = s;
    }
}

static void capture_losses(std::vector<float>& dst, GPT2* model, int N) {
    dst.assign(model->cpu_losses, model->cpu_losses + N);
}

static void report_case(const char* name,
                        GPT2* model,
                        DataLoader* loader,
                        int B, int T,
                        int layer,
                        XrayComponent component,
                        const float* ref_logits,
                        const std::vector<float>& ref_losses,
                        TokenOutputStats* d_stats) {
    const int N = B * T;
    const int V = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;

    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gpt2_forward_one_component_tf32(model, loader->inputs, loader->targets,
                                    B, T, layer, component);
    cudaCheck(cudaDeviceSynchronize());
    std::vector<float> cur_losses;
    capture_losses(cur_losses, model, N);

    gpt2_forward_one_component_tf32(model, loader->inputs, NULL,
                                    B, T, layer, component);
    cudaCheck(cudaDeviceSynchronize());

    int block = 256;
    size_t smem = (5 * block * sizeof(float)) + (2 * block * sizeof(int));
    compare_logits_per_token<<<N, block, smem>>>(ref_logits, model->acts.output,
                                                 model->targets, d_stats, N, V, Vp);
    cudaCheck(cudaGetLastError());
    std::vector<TokenOutputStats> stats(N);
    cudaCheck(cudaMemcpy(stats.data(), d_stats, N * sizeof(TokenOutputStats), cudaMemcpyDeviceToHost));

    double mean_rel = 0.0;
    double max_rel = -1.0;
    int max_rel_token = -1;
    double mean_abs_dloss = 0.0;
    double max_abs_dloss = -1.0;
    int max_loss_token = -1;
    double mean_abs_target_delta = 0.0;
    int top1_changes = 0;

    for (int i = 0; i < N; ++i) {
        mean_rel += stats[i].rel_l2;
        if (stats[i].rel_l2 > max_rel) {
            max_rel = stats[i].rel_l2;
            max_rel_token = i;
        }
        double dl = fabs((double)cur_losses[i] - ref_losses[i]);
        mean_abs_dloss += dl;
        if (dl > max_abs_dloss) {
            max_abs_dloss = dl;
            max_loss_token = i;
        }
        mean_abs_target_delta += fabs((double)stats[i].target_delta);
        if (stats[i].top1_ref != stats[i].top1_cur) ++top1_changes;
    }

    mean_rel /= N;
    mean_abs_dloss /= N;
    mean_abs_target_delta /= N;

    printf("[xray][output-causal] %-10s component=%-7s L=%02d mean_logit_rel_l2=%10.6e max_logit_rel_l2=%10.6e@tok%04d mean|dloss_tok|=%10.6e max|dloss_tok|=%10.6e@tok%04d mean|dtarget_logit|=%10.6e top1_changes=%d/%d\n",
           name, component_name(component), layer,
           mean_rel, max_rel, max_rel_token,
           mean_abs_dloss, max_abs_dloss, max_loss_token,
           mean_abs_target_delta, top1_changes, N);
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

    gpt2_forward(&model, loader.inputs, loader.targets, B, T);
    cudaCheck(cudaDeviceSynchronize());
    std::vector<float> ref_losses;
    capture_losses(ref_losses, &model, N);

    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float* ref_logits = nullptr;
    cudaCheck(cudaMalloc((void**)&ref_logits, logits_elems * sizeof(float)));
    cudaCheck(cudaMemcpy(ref_logits, model.acts.output,
                         logits_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    TokenOutputStats* d_stats = nullptr;
    cudaCheck(cudaMalloc((void**)&d_stats, N * sizeof(TokenOutputStats)));

    printf("[xray][output-causal] compare two previously discovered sensitive sites at token/logit level\n");
    printf("[xray][output-causal] question: broad representation displacement or task-directed output consequence?\n");

    report_case("residual-peak", &model, &loader, B, T,
                2, XRAY_QKV, ref_logits, ref_losses, d_stats);
    report_case("loss-peak", &model, &loader, B, T,
                0, XRAY_FCPROJ, ref_logits, ref_losses, d_stats);

    cudaCheck(cudaFree(d_stats));
    cudaCheck(cudaFree(ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
