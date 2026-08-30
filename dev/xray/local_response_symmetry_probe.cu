#pragma push_macro("main")
#undef main
#define main xray_natural_linearity_embedded_main
#include "natural_perturbation_linearity_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cmath>
#include <cstdio>
#include <vector>

struct LocalSymmetryStats {
    double plus_rel_l2;
    double minus_rel_l2;
    double odd_rel_l2;
    double even_rel_l2;
    double curvature_ratio;
    double slope_rel_diff;
    double slope_cosine;
};

static double vector_norm2(const std::vector<long double>& v) {
    long double s = 0.0L;
    for (long double x : v) s += x * x;
    return sqrt((double)s);
}

static LocalSymmetryStats local_symmetry_stats(const std::vector<float>& baseline,
                                               const std::vector<float>& plus,
                                               const std::vector<float>& minus,
                                               double alpha,
                                               const std::vector<long double>* slope_ref) {
    long double base2 = 0.0L;
    long double plus2 = 0.0L;
    long double minus2 = 0.0L;
    long double odd2 = 0.0L;
    long double even2 = 0.0L;

    std::vector<long double> slope(baseline.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        long double b = baseline[i];
        long double dp = (long double)plus[i] - b;
        long double dm = (long double)minus[i] - b;
        long double odd = 0.5L * ((long double)plus[i] - (long double)minus[i]);
        long double even = 0.5L * ((long double)plus[i] + (long double)minus[i]) - b;
        long double s = odd / (long double)alpha;

        base2 += b * b;
        plus2 += dp * dp;
        minus2 += dm * dm;
        odd2 += odd * odd;
        even2 += even * even;
        slope[i] = s;
    }

    double plus_rel = base2 > 0.0L ? sqrt((double)(plus2 / base2)) : 0.0;
    double minus_rel = base2 > 0.0L ? sqrt((double)(minus2 / base2)) : 0.0;
    double odd_rel = base2 > 0.0L ? sqrt((double)(odd2 / base2)) : 0.0;
    double even_rel = base2 > 0.0L ? sqrt((double)(even2 / base2)) : 0.0;
    double curvature_ratio = odd2 > 0.0L ? sqrt((double)(even2 / odd2)) : 0.0;

    double slope_rel_diff = 0.0;
    double slope_cosine = 1.0;
    if (slope_ref != nullptr) {
        long double ref2 = 0.0L;
        long double cur2 = 0.0L;
        long double diff2 = 0.0L;
        long double dot = 0.0L;
        for (size_t i = 0; i < slope.size(); ++i) {
            long double r = (*slope_ref)[i];
            long double c = slope[i];
            ref2 += r * r;
            cur2 += c * c;
            long double d = c - r;
            diff2 += d * d;
            dot += r * c;
        }
        slope_rel_diff = ref2 > 0.0L ? sqrt((double)(diff2 / ref2)) : 0.0;
        slope_cosine = (ref2 > 0.0L && cur2 > 0.0L)
            ? (double)(dot / sqrt((double)(ref2 * cur2))) : 0.0;
    }

    return LocalSymmetryStats{plus_rel, minus_rel, odd_rel, even_rel,
                              curvature_ratio, slope_rel_diff, slope_cosine};
}

static void make_central_slope(std::vector<long double>& slope,
                               const std::vector<float>& plus,
                               const std::vector<float>& minus,
                               double alpha) {
    slope.resize(plus.size());
    for (size_t i = 0; i < plus.size(); ++i) {
        slope[i] = ((long double)plus[i] - (long double)minus[i]) / (2.0L * (long double)alpha);
    }
}

static double mean_loss_double(const GPT2& model, int N) {
    long double sum = 0.0L;
    for (int i = 0; i < N; ++i) sum += (long double)model.cpu_losses[i];
    return (double)(sum / (long double)N);
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

    const int N = B * T;
    std::vector<float> baseline;
    gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                   B, T, selected_layer, 0.0f, qkv_custom, qkv_tf32);
    cudaCheck(cudaDeviceSynchronize());
    double loss0 = mean_loss_double(model, N);
    capture_final_residual(baseline, &model, B, T);

    const float alphas[] = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
    const int na = (int)(sizeof(alphas) / sizeof(alphas[0]));
    std::vector<long double> slope_ref;

    printf("[xray][local-response] source=qkv L02 natural TF32-custom direction; symmetric +/-alpha around original custom execution\n");
    printf("[xray][local-response] use odd response as local linear term and even response as curvature; slope_ref is central difference at alpha=0.125\n");
    printf("[xray][local-response] baseline_loss_double=%.12f\n", loss0);

    for (int ai = 0; ai < na; ++ai) {
        float alpha = alphas[ai];
        std::vector<float> plus;
        std::vector<float> minus;

        gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                       B, T, selected_layer, alpha, qkv_custom, qkv_tf32);
        cudaCheck(cudaDeviceSynchronize());
        double loss_plus = mean_loss_double(model, N);
        capture_final_residual(plus, &model, B, T);

        gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                       B, T, selected_layer, -alpha, qkv_custom, qkv_tf32);
        cudaCheck(cudaDeviceSynchronize());
        double loss_minus = mean_loss_double(model, N);
        capture_final_residual(minus, &model, B, T);

        if (ai == 0) make_central_slope(slope_ref, plus, minus, alpha);
        LocalSymmetryStats s = local_symmetry_stats(baseline, plus, minus, alpha,
                                                    ai == 0 ? nullptr : &slope_ref);

        double loss_odd = 0.5 * (loss_plus - loss_minus);
        double loss_even = 0.5 * (loss_plus + loss_minus) - loss0;
        double loss_curvature_ratio = fabs(loss_odd) > 0.0 ? fabs(loss_even / loss_odd) : 0.0;
        double loss_central_slope = loss_odd / alpha;

        printf("[xray][local-response] alpha=%6.3f plus_rel=%10.6e minus_rel=%10.6e odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e slope_rel_diff=%10.6e slope_cosine=% .9f loss_plus=%+.9e loss_minus=%+.9e loss_slope=%+.9e loss_curvature_ratio=%10.6e\n",
               alpha, s.plus_rel_l2, s.minus_rel_l2, s.odd_rel_l2, s.even_rel_l2,
               s.curvature_ratio, s.slope_rel_diff, s.slope_cosine,
               loss_plus - loss0, loss_minus - loss0, loss_central_slope,
               loss_curvature_ratio);
    }

    cudaCheck(cudaFree(qkv_tf32));
    cudaCheck(cudaFree(qkv_custom));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
