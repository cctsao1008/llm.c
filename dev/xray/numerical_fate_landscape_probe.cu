#define XRAY_COMPONENT_SENSITIVITY_EMBEDDED
#include "component_sensitivity_probe.cu"
#undef XRAY_COMPONENT_SENSITIVITY_EMBEDDED

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static constexpr int XRAY_TOP2_BLOCK = 256;

__device__ inline void xray_consider_top2(float value, int index,
                                          float& best, int& best_i,
                                          float& second, int& second_i) {
    if (index < 0) return;
    const bool better_best = (value > best) || (value == best && index < best_i);
    if (better_best) {
        if (index != best_i) {
            second = best;
            second_i = best_i;
        }
        best = value;
        best_i = index;
    } else if (index != best_i) {
        const bool better_second = (value > second) || (value == second && index < second_i);
        if (better_second) {
            second = value;
            second_i = index;
        }
    }
}

__global__ void xray_top2_rows_kernel(const float* logits,
                                      int* top1_i, int* top2_i,
                                      float* top1_v, float* top2_v,
                                      int N, int V, int Vp) {
    const int row = blockIdx.x;
    if (row >= N) return;

    __shared__ float sbest[XRAY_TOP2_BLOCK];
    __shared__ float ssecond[XRAY_TOP2_BLOCK];
    __shared__ int sbest_i[XRAY_TOP2_BLOCK];
    __shared__ int ssecond_i[XRAY_TOP2_BLOCK];

    float best = -CUDART_INF_F;
    float second = -CUDART_INF_F;
    int best_i = INT_MAX;
    int second_i = INT_MAX;
    const float* rowp = logits + (size_t)row * Vp;

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        xray_consider_top2(rowp[v], v, best, best_i, second, second_i);
    }

    sbest[threadIdx.x] = best;
    ssecond[threadIdx.x] = second;
    sbest_i[threadIdx.x] = best_i;
    ssecond_i[threadIdx.x] = second_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float b = sbest[threadIdx.x];
            float s = ssecond[threadIdx.x];
            int bi = sbest_i[threadIdx.x];
            int si = ssecond_i[threadIdx.x];
            xray_consider_top2(sbest[threadIdx.x + stride], sbest_i[threadIdx.x + stride], b, bi, s, si);
            xray_consider_top2(ssecond[threadIdx.x + stride], ssecond_i[threadIdx.x + stride], b, bi, s, si);
            sbest[threadIdx.x] = b;
            ssecond[threadIdx.x] = s;
            sbest_i[threadIdx.x] = bi;
            ssecond_i[threadIdx.x] = si;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        top1_i[row] = sbest_i[0];
        top2_i[row] = ssecond_i[0];
        top1_v[row] = sbest[0];
        top2_v[row] = ssecond[0];
    }
}

__global__ void xray_pair_margin_kernel(const float* ref_logits,
                                        const float* alt_logits,
                                        const int* ref_top1,
                                        const int* ref_runner,
                                        const int* alt_top1,
                                        float* ref_pair,
                                        float* alt_pair,
                                        int* competitor,
                                        int N, int Vp) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const int a = ref_top1[row];
    const int b = (alt_top1[row] != a) ? alt_top1[row] : ref_runner[row];
    competitor[row] = b;
    const float* r = ref_logits + (size_t)row * Vp;
    const float* x = alt_logits + (size_t)row * Vp;
    ref_pair[row] = r[a] - r[b];
    alt_pair[row] = x[a] - x[b];
}

struct XrayTop2Host {
    std::vector<int> top1, top2;
    std::vector<float> v1, v2;
};

struct XrayTop2Device {
    int *top1 = nullptr, *top2 = nullptr;
    float *v1 = nullptr, *v2 = nullptr;
};

static void xray_alloc_top2(XrayTop2Device& d, int N) {
    cudaCheck(cudaMalloc((void**)&d.top1, (size_t)N * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d.top2, (size_t)N * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d.v1, (size_t)N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d.v2, (size_t)N * sizeof(float)));
}

static void xray_free_top2(XrayTop2Device& d) {
    cudaCheck(cudaFree(d.v2));
    cudaCheck(cudaFree(d.v1));
    cudaCheck(cudaFree(d.top2));
    cudaCheck(cudaFree(d.top1));
}

static XrayTop2Host xray_capture_top2(const float* logits,
                                      XrayTop2Device& d,
                                      int N, int V, int Vp) {
    xray_top2_rows_kernel<<<N, XRAY_TOP2_BLOCK>>>(logits, d.top1, d.top2, d.v1, d.v2, N, V, Vp);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    XrayTop2Host h;
    h.top1.resize(N); h.top2.resize(N); h.v1.resize(N); h.v2.resize(N);
    cudaCheck(cudaMemcpy(h.top1.data(), d.top1, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h.top2.data(), d.top2, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h.v1.data(), d.v1, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h.v2.data(), d.v2, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static int xray_top2_mismatches(const XrayTop2Host& a, const XrayTop2Host& b) {
    int neq = 0;
    for (size_t i = 0; i < a.top1.size(); ++i) {
        if (a.top1[i] != b.top1[i] || a.top2[i] != b.top2[i] ||
            a.v1[i] != b.v1[i] || a.v2[i] != b.v2[i]) ++neq;
    }
    return neq;
}

static double xray_median(std::vector<double> x) {
    if (x.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(x.begin(), x.end());
    const size_t n = x.size();
    return (n & 1) ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

struct XrayFamily {
    const char* name;
    XrayComponent component;
};

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batches = argc > 3 ? atoi(argv[3]) : 8;
    const char* csv_path = argc > 4 ? argv[4] : "/tmp/numerical_fate_landscape.csv";
    const double near_margin = argc > 5 ? atof(argv[5]) : 1.0e-2;
    const int repeats = argc > 6 ? atoi(argv[6]) : 2;
    if (B <= 0 || T <= 0 || batches <= 0 || repeats <= 0) {
        fprintf(stderr, "usage: %s [B=4] [T=512] [batches=8] [csv=/tmp/numerical_fate_landscape.csv] [near_margin=0.01] [repeats=2]\n", argv[0]);
        return 2;
    }

    const XrayFamily families[] = {
        {"l00-qkv", XRAY_QKV},
        {"l00-attproj", XRAY_ATTPROJ},
        {"l00-fc", XRAY_FC},
        {"l00-fcproj", XRAY_FCPROJ},
    };
    constexpr int NF = sizeof(families) / sizeof(families[0]);

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin", B, T, 0, 1, 1);

    const int V = model.config.vocab_size;
    const int Vp = model.config.padded_vocab_size;
    const int N = B * T;

    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
        perror("fopen csv");
        return 3;
    }
    fprintf(csv, "batch,family,scan_index,b,t,input_token,ref_top1,ref_runner,ref_top2_margin,alt_top1,alt_runner,alt_top2_margin,flip,competitor,ref_pair_margin,alt_pair_margin,pair_shift,near_ref,repeat_exact\n");

    // First forward allocates activations, including the large logits scratch/output.
    dataloader_next_batch(&loader);
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());

    float* d_ref_logits = nullptr;
    cudaCheck(cudaMalloc((void**)&d_ref_logits, (size_t)N * Vp * sizeof(float)));
    XrayTop2Device d_ref_top2, d_alt_top2;
    xray_alloc_top2(d_ref_top2, N);
    xray_alloc_top2(d_alt_top2, N);

    float *d_ref_pair = nullptr, *d_alt_pair = nullptr;
    int* d_competitor = nullptr;
    cudaCheck(cudaMalloc((void**)&d_ref_pair, (size_t)N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_alt_pair, (size_t)N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_competitor, (size_t)N * sizeof(int)));

    long long flips[NF] = {0,0,0,0};
    long long near_nonflip[NF] = {0,0,0,0};
    long long far_nonflip[NF] = {0,0,0,0};
    long long repeat_fail[NF] = {0,0,0,0};
    std::vector<double> flip_ref_margin[NF];
    std::vector<double> flip_abs_pair_shift[NF];
    std::vector<unsigned char> flip_mask((size_t)batches * N, 0);
    long long baseline_repeat_fail = 0;

    printf("[xray][fate-landscape] device=%s cc=%d.%d B=%d T=%d batches=%d families=%d repeats=%d near_margin=%.6g\n",
           prop.name, prop.major, prop.minor, B, T, batches, NF, repeats, near_margin);
    printf("[xray][fate-landscape] survey only: four L00 single-GEMM TF32 perturbation families; no CPU64 causal localization in this phase\n");
    printf("[xray][fate-landscape] csv=%s\n", csv_path);

    for (int batch = 0; batch < batches; ++batch) {
        if (batch > 0) dataloader_next_batch(&loader);

        // Baseline repeatability gate.
        XrayTop2Host ref_first;
        for (int r = 0; r < repeats; ++r) {
            gpt2_forward(&model, loader.inputs, NULL, B, T);
            cudaCheck(cudaDeviceSynchronize());
            XrayTop2Host cur = xray_capture_top2(model.acts.output, d_ref_top2, N, V, Vp);
            if (r == 0) ref_first = cur;
            else baseline_repeat_fail += xray_top2_mismatches(ref_first, cur);
        }
        const XrayTop2Host ref = xray_capture_top2(model.acts.output, d_ref_top2, N, V, Vp);
        cudaCheck(cudaMemcpy(d_ref_logits, model.acts.output,
                             (size_t)N * Vp * sizeof(float), cudaMemcpyDeviceToDevice));

        long long batch_flips[NF] = {0,0,0,0};

        for (int fi = 0; fi < NF; ++fi) {
            XrayTop2Host alt_first, alt;
            int family_repeat_mismatch = 0;
            for (int r = 0; r < repeats; ++r) {
                cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
                gpt2_forward_one_component_tf32(&model, loader.inputs, NULL, B, T, 0, families[fi].component);
                cudaCheck(cudaDeviceSynchronize());
                XrayTop2Host cur = xray_capture_top2(model.acts.output, d_alt_top2, N, V, Vp);
                if (r == 0) alt_first = cur;
                else family_repeat_mismatch += xray_top2_mismatches(alt_first, cur);
                alt = std::move(cur);
            }
            repeat_fail[fi] += family_repeat_mismatch;

            const int pair_block = 256;
            const int pair_grid = CEIL_DIV(N, pair_block);
            xray_pair_margin_kernel<<<pair_grid, pair_block>>>(
                d_ref_logits, model.acts.output,
                d_ref_top2.top1, d_ref_top2.top2, d_alt_top2.top1,
                d_ref_pair, d_alt_pair, d_competitor, N, Vp);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());

            std::vector<float> ref_pair(N), alt_pair(N);
            std::vector<int> competitor(N);
            cudaCheck(cudaMemcpy(ref_pair.data(), d_ref_pair, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(alt_pair.data(), d_alt_pair, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(competitor.data(), d_competitor, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

            for (int bt = 0; bt < N; ++bt) {
                const bool flip = ref.top1[bt] != alt.top1[bt];
                const double ref_margin = (double)ref.v1[bt] - (double)ref.v2[bt];
                const double alt_margin = (double)alt.v1[bt] - (double)alt.v2[bt];
                const double shift = (double)alt_pair[bt] - (double)ref_pair[bt];
                const bool near = ref_margin <= near_margin;
                if (flip) {
                    ++flips[fi];
                    ++batch_flips[fi];
                    flip_ref_margin[fi].push_back(ref_margin);
                    flip_abs_pair_shift[fi].push_back(std::fabs(shift));
                    flip_mask[(size_t)batch * N + bt] |= (unsigned char)(1u << fi);
                } else if (near) {
                    ++near_nonflip[fi];
                } else {
                    ++far_nonflip[fi];
                }

                const long long scan_index = (long long)batch * N + bt;
                const int bb = bt / T;
                const int tt = bt % T;
                const int input_token = loader.inputs[bt];
                fprintf(csv,
                        "%d,%s,%lld,%d,%d,%d,%d,%d,%.9e,%d,%d,%.9e,%d,%d,%.9e,%.9e,%.9e,%d,%d\n",
                        batch, families[fi].name, scan_index, bb, tt, input_token,
                        ref.top1[bt], ref.top2[bt], ref_margin,
                        alt.top1[bt], alt.top2[bt], alt_margin,
                        flip ? 1 : 0, competitor[bt],
                        (double)ref_pair[bt], (double)alt_pair[bt], shift,
                        near ? 1 : 0, family_repeat_mismatch == 0 ? 1 : 0);
            }
        }

        printf("[xray][fate-landscape-batch] batch=%d flips qkv=%lld attproj=%lld fc=%lld fcproj=%lld baseline_repeat_fail=%lld\n",
               batch, batch_flips[0], batch_flips[1], batch_flips[2], batch_flips[3], baseline_repeat_fail);
        fflush(csv);
    }

    long long union_flips = 0, multi_family_flips = 0, all_family_flips = 0;
    for (unsigned char m : flip_mask) {
        if (!m) continue;
        ++union_flips;
        int bits = 0;
        for (int i = 0; i < NF; ++i) bits += (m >> i) & 1u;
        if (bits >= 2) ++multi_family_flips;
        if (bits == NF) ++all_family_flips;
    }

    const long long total = (long long)batches * N;
    for (int fi = 0; fi < NF; ++fi) {
        printf("[xray][fate-landscape-family] family=%s tokens=%lld flips=%lld flip_rate=%.9e near_nonflip=%lld far_nonflip=%lld repeat_fail=%lld median_ref_margin_flip=%.9e median_abs_pair_shift_flip=%.9e\n",
               families[fi].name, total, flips[fi], total ? (double)flips[fi] / (double)total : 0.0,
               near_nonflip[fi], far_nonflip[fi], repeat_fail[fi],
               xray_median(flip_ref_margin[fi]), xray_median(flip_abs_pair_shift[fi]));
    }
    printf("[xray][fate-landscape-overlap] unique_positions=%lld union_flips=%lld multi_family_flips=%lld all_family_flips=%lld\n",
           total, union_flips, multi_family_flips, all_family_flips);
    printf("[xray][fate-landscape-summary] baseline_repeat_fail=%lld family_repeat_fail_total=%lld validity=%d csv=%s\n",
           baseline_repeat_fail,
           repeat_fail[0] + repeat_fail[1] + repeat_fail[2] + repeat_fail[3],
           (baseline_repeat_fail == 0 && repeat_fail[0] == 0 && repeat_fail[1] == 0 && repeat_fail[2] == 0 && repeat_fail[3] == 0) ? 1 : 0,
           csv_path);
    printf("[xray][fate-landscape-summary] interpretation gate: this is a population survey of final decision geometry only; it does not localize a causal layer/operator or establish a mechanism\n");

    fclose(csv);
    cudaCheck(cudaFree(d_competitor));
    cudaCheck(cudaFree(d_alt_pair));
    cudaCheck(cudaFree(d_ref_pair));
    xray_free_top2(d_alt_top2);
    xray_free_top2(d_ref_top2);
    cudaCheck(cudaFree(d_ref_logits));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
