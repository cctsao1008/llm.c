#define main xray_forward_ab_embedded_main
#include "forward_ab_probe.cu"
#undef main

#include <vector>

struct TrainRun {
    const char* name;
    double mean_ms;
    std::vector<float> losses;
};

enum ForwardMode {
    XRAY_CUSTOM = 0,
    XRAY_CUBLAS_FP32 = 1,
    XRAY_CUBLAS_TF32 = 2,
};

static void set_tf32() {
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

static void set_fp32() {
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
}

static void run_forward_mode(GPT2* model, DataLoader* loader, int B, int T, ForwardMode mode) {
    if (mode == XRAY_CUSTOM) {
        // This matches train_gpt2_fp32.cu on Ampere/Ada: custom forward GEMMs,
        // while attention/cuBLAS calls retain TF32 math mode.
        set_tf32();
        gpt2_forward(model, loader->inputs, loader->targets, B, T);
    } else if (mode == XRAY_CUBLAS_FP32) {
        set_fp32();
        gpt2_forward_cublas(model, loader->inputs, loader->targets, B, T);
    } else {
        set_tf32();
        gpt2_forward_cublas(model, loader->inputs, loader->targets, B, T);
    }
}

static TrainRun run_training_variant(const char* name, ForwardMode mode,
                                     int B, int T, int measured_steps) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    DataLoader loader;
    dataloader_init(&loader,
                    "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
                    B, T, 0, 1, 1);

    // One complete untimed step pays lazy activation/gradient/optimizer allocations.
    dataloader_next_batch(&loader);
    run_forward_mode(&model, &loader, B, T, mode);
    gpt2_zero_grad(&model);
    // Keep backward math mode identical across all variants. Only the forward GEMM
    // implementation/precision is under test here.
    set_tf32();
    gpt2_backward(&model);
    gpt2_update(&model, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
    cudaCheck(cudaDeviceSynchronize());

    TrainRun result{name, 0.0, {}};
    result.losses.reserve(measured_steps);

    double total_ms = 0.0;
    for (int step = 0; step < measured_steps; ++step) {
        dataloader_next_batch(&loader);

        nvtxRangePushA(name);
        double t0 = now_ms();
        run_forward_mode(&model, &loader, B, T, mode);
        float loss = model.mean_loss;
        gpt2_zero_grad(&model);
        set_tf32();
        gpt2_backward(&model);
        gpt2_update(&model, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 2);
        cudaCheck(cudaDeviceSynchronize());
        double t1 = now_ms();
        nvtxRangePop();

        total_ms += t1 - t0;
        result.losses.push_back(loss);
    }
    result.mean_ms = total_ms / measured_steps;

    dataloader_free(&loader);
    gpt2_free(&model);
    return result;
}

static void print_run(const TrainRun& r, double baseline_ms,
                      const std::vector<float>* baseline_losses) {
    double speedup = baseline_ms / r.mean_ms;
    double mean_abs_dloss = 0.0;
    double max_abs_dloss = 0.0;
    float final_delta = 0.0f;

    if (baseline_losses != nullptr && baseline_losses->size() == r.losses.size()) {
        for (size_t i = 0; i < r.losses.size(); ++i) {
            double d = fabs((double)r.losses[i] - (double)(*baseline_losses)[i]);
            mean_abs_dloss += d;
            if (d > max_abs_dloss) max_abs_dloss = d;
        }
        if (!r.losses.empty()) {
            mean_abs_dloss /= r.losses.size();
            final_delta = r.losses.back() - baseline_losses->back();
        }
    }

    printf("[xray][train-ab] %-13s mean=%8.3f ms speedup=%5.2fx first_loss=%.8f final_loss=%.8f",
           r.name, r.mean_ms, speedup,
           r.losses.empty() ? -1.0f : r.losses.front(),
           r.losses.empty() ? -1.0f : r.losses.back());
    if (baseline_losses != nullptr) {
        printf(" mean|dloss|=%.8g max|dloss|=%.8g final_dloss=%+.8f",
               mean_abs_dloss, max_abs_dloss, final_delta);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    int steps = argc > 3 ? atoi(argv[3]) : 8;
    if (B <= 0 || T <= 0 || steps <= 0) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [steps=8]\n", argv[0]);
        return 2;
    }

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    cublasCheck(cublasCreate(&cublas_handle));

    printf("[xray][train-ab] device=%s cc=%d.%d B=%d T=%d measured_steps=%d\n",
           prop.name, prop.major, prop.minor, B, T, steps);
    printf("[xray][train-ab] complete-step A/B: forward differs; backward+AdamW stay on the same TF32 path\n");

    TrainRun custom = run_training_variant("custom", XRAY_CUSTOM, B, T, steps);
    TrainRun fp32 = run_training_variant("cuBLAS-FP32", XRAY_CUBLAS_FP32, B, T, steps);
    TrainRun tf32 = run_training_variant("cuBLAS-TF32", XRAY_CUBLAS_TF32, B, T, steps);

    print_run(custom, custom.mean_ms, nullptr);
    print_run(fp32, custom.mean_ms, &custom.losses);
    print_run(tf32, custom.mean_ms, &custom.losses);

    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
