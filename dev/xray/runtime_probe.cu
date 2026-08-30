#define TESTING
#include "../../train_gpt2_fp32.cu"

#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <vector>

struct PhaseSample {
    const char* name;
    float gpu_ms;
    double wall_ms;
    size_t free_before;
    size_t free_after;
};

static double wall_now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static void get_mem(size_t* free_b, size_t* total_b) {
    cudaCheck(cudaMemGetInfo(free_b, total_b));
}

static void print_mem(const char* tag) {
    size_t free_b = 0, total_b = 0;
    get_mem(&free_b, &total_b);
    size_t used_b = total_b - free_b;
    printf("[xray][mem] %-18s used=%8.2f MiB free=%8.2f MiB total=%8.2f MiB\n",
           tag,
           used_b / 1048576.0,
           free_b / 1048576.0,
           total_b / 1048576.0);
}

template <typename Fn>
static PhaseSample measure_phase(const char* name, Fn&& fn) {
    // Deliberately isolate each semantic phase. This is a probe, not a benchmark.
    cudaCheck(cudaDeviceSynchronize());

    size_t free_before = 0, total_before = 0;
    get_mem(&free_before, &total_before);

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    double wall_start = wall_now_ms();
    nvtxRangePushA(name);
    cudaCheck(cudaEventRecord(start));
    fn();
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    nvtxRangePop();
    double wall_stop = wall_now_ms();

    float gpu_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&gpu_ms, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    size_t free_after = 0, total_after = 0;
    get_mem(&free_after, &total_after);

    return PhaseSample{name, gpu_ms, wall_stop - wall_start, free_before, free_after};
}

static void print_sample(const char* prefix, const PhaseSample& s) {
    double delta_mib = ((double)s.free_before - (double)s.free_after) / 1048576.0;
    double host_gap = s.wall_ms - s.gpu_ms;
    printf("[xray][%s] %-12s gpu=%8.3f ms wall=%8.3f ms wall-gpu=%8.3f ms mem_delta=%8.2f MiB\n",
           prefix, s.name, s.gpu_ms, s.wall_ms, host_gap, delta_mib);
}

static void setup_device() {
    int device_idx = 0;
    cudaCheck(cudaSetDevice(device_idx));

    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, device_idx));
    cublasCheck(cublasCreate(&cublas_handle));

    int enable_tf32 = prop.major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, math_mode));

    printf("[xray] device=%s cc=%d.%d TF32=%s\n",
           prop.name, prop.major, prop.minor, enable_tf32 ? "on" : "off");
}

static void cold_path_probe(GPT2* model, DataLoader* loader, int B, int T) {
    printf("\n[xray] === cold-path discovery ===\n");
    print_mem("before first batch");
    dataloader_next_batch(loader);

    PhaseSample fwd = measure_phase("forward", [&] {
        gpt2_forward(model, loader->inputs, loader->targets, B, T);
    });
    print_sample("cold", fwd);
    print_mem("after first forward");

    PhaseSample zero = measure_phase("zero_grad", [&] {
        gpt2_zero_grad(model);
    });
    print_sample("cold", zero);

    PhaseSample bwd = measure_phase("backward", [&] {
        gpt2_backward(model);
    });
    print_sample("cold", bwd);
    print_mem("after first backward");

    PhaseSample upd = measure_phase("adamw", [&] {
        gpt2_update(model, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
    });
    print_sample("cold", upd);
    print_mem("after first update");
}

static void steady_probe(GPT2* model, DataLoader* loader, int B, int T, int steps) {
    printf("\n[xray] === steady-state phase discovery (%d steps) ===\n", steps);

    std::vector<double> fwd_gpu, bwd_gpu, upd_gpu, zero_gpu;
    std::vector<double> fwd_wall, bwd_wall, upd_wall, zero_wall;

    for (int i = 0; i < steps; ++i) {
        dataloader_next_batch(loader);

        PhaseSample fwd = measure_phase("forward", [&] {
            gpt2_forward(model, loader->inputs, loader->targets, B, T);
        });
        PhaseSample zero = measure_phase("zero_grad", [&] {
            gpt2_zero_grad(model);
        });
        PhaseSample bwd = measure_phase("backward", [&] {
            gpt2_backward(model);
        });
        PhaseSample upd = measure_phase("adamw", [&] {
            gpt2_update(model, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, i + 2);
        });

        print_sample("steady", fwd);
        print_sample("steady", zero);
        print_sample("steady", bwd);
        print_sample("steady", upd);

        fwd_gpu.push_back(fwd.gpu_ms); fwd_wall.push_back(fwd.wall_ms);
        zero_gpu.push_back(zero.gpu_ms); zero_wall.push_back(zero.wall_ms);
        bwd_gpu.push_back(bwd.gpu_ms); bwd_wall.push_back(bwd.wall_ms);
        upd_gpu.push_back(upd.gpu_ms); upd_wall.push_back(upd.wall_ms);
    }

    auto mean = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double x : v) sum += x;
        return v.empty() ? 0.0 : sum / v.size();
    };

    double gpu_total = mean(fwd_gpu) + mean(zero_gpu) + mean(bwd_gpu) + mean(upd_gpu);
    double wall_total = mean(fwd_wall) + mean(zero_wall) + mean(bwd_wall) + mean(upd_wall);

    printf("\n[xray] === discovered phase means ===\n");
    printf("[xray][mean] forward   gpu=%8.3f ms wall=%8.3f ms\n", mean(fwd_gpu), mean(fwd_wall));
    printf("[xray][mean] zero_grad gpu=%8.3f ms wall=%8.3f ms\n", mean(zero_gpu), mean(zero_wall));
    printf("[xray][mean] backward  gpu=%8.3f ms wall=%8.3f ms\n", mean(bwd_gpu), mean(bwd_wall));
    printf("[xray][mean] adamw     gpu=%8.3f ms wall=%8.3f ms\n", mean(upd_gpu), mean(upd_wall));
    printf("[xray][mean] total     gpu=%8.3f ms wall=%8.3f ms\n", gpu_total, wall_total);
    if (gpu_total > 0.0) {
        printf("[xray][share] forward=%5.1f%% zero_grad=%5.1f%% backward=%5.1f%% adamw=%5.1f%%\n",
               100.0 * mean(fwd_gpu) / gpu_total,
               100.0 * mean(zero_gpu) / gpu_total,
               100.0 * mean(bwd_gpu) / gpu_total,
               100.0 * mean(upd_gpu) / gpu_total);
    }
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    int steps = argc > 3 ? atoi(argv[3]) : 5;
    if (B <= 0 || T <= 0 || steps <= 0) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [steps=5]\n", argv[0]);
        return 2;
    }

    setup_device();
    print_mem("process start");

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    print_mem("after model load");

    DataLoader train_loader;
    dataloader_init(&train_loader,
                    "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
                    B, T, 0, 1, 1);

    printf("[xray] probe B=%d T=%d steps=%d parameters=%zu\n",
           B, T, steps, model.num_parameters);

    cold_path_probe(&model, &train_loader, B, T);
    steady_probe(&model, &train_loader, B, T, steps);

    print_mem("before free");
    dataloader_free(&train_loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    print_mem("after free");
    return 0;
}
