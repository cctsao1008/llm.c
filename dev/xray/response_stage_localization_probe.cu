#define main xray_local_response_embedded_main
#include "local_response_symmetry_probe.cu"
#undef main

#include <cstdio>
#include <vector>

struct StageCapture {
    const char* name;
    std::vector<float> baseline;
    std::vector<float> plus;
    std::vector<float> minus;
};

static void capture_device(std::vector<float>& dst, const float* src, size_t n) {
    dst.resize(n);
    cudaCheck(cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void capture_stage_baseline(std::vector<StageCapture>& stages, GPT2* model, int B, int T, int layer) {
    int C = model->config.channels;
    int NH = model->config.num_heads;
    int L = model->config.num_layers;
    size_t BTC = (size_t)B * T * C;
    size_t BHTT = (size_t)B * NH * T * T;

    stages.clear();
    stages.push_back({"qkvr", {}, {}, {}});
    stages.push_back({"att", {}, {}, {}});
    stages.push_back({"atty", {}, {}, {}});
    stages.push_back({"attproj", {}, {}, {}});
    stages.push_back({"residual2", {}, {}, {}});
    stages.push_back({"residual3-L02", {}, {}, {}});
    stages.push_back({"final-residual", {}, {}, {}});

    capture_device(stages[0].baseline, model->acts.qkvr + (size_t)layer * B * T * 3 * C, 3 * BTC);
    capture_device(stages[1].baseline, model->acts.att + (size_t)layer * BHTT, BHTT);
    capture_device(stages[2].baseline, model->acts.atty + (size_t)layer * BTC, BTC);
    capture_device(stages[3].baseline, model->acts.attproj + (size_t)layer * BTC, BTC);
    capture_device(stages[4].baseline, model->acts.residual2 + (size_t)layer * BTC, BTC);
    capture_device(stages[5].baseline, model->acts.residual3 + (size_t)layer * BTC, BTC);
    capture_device(stages[6].baseline, model->acts.residual3 + (size_t)(L - 1) * BTC, BTC);
}

static void capture_stage_side(std::vector<StageCapture>& stages, GPT2* model, int B, int T, int layer, bool plus_side) {
    int C = model->config.channels;
    int NH = model->config.num_heads;
    int L = model->config.num_layers;
    size_t BTC = (size_t)B * T * C;
    size_t BHTT = (size_t)B * NH * T * T;

    auto& qkvr = plus_side ? stages[0].plus : stages[0].minus;
    auto& att = plus_side ? stages[1].plus : stages[1].minus;
    auto& atty = plus_side ? stages[2].plus : stages[2].minus;
    auto& attproj = plus_side ? stages[3].plus : stages[3].minus;
    auto& residual2 = plus_side ? stages[4].plus : stages[4].minus;
    auto& residual3_l2 = plus_side ? stages[5].plus : stages[5].minus;
    auto& final_residual = plus_side ? stages[6].plus : stages[6].minus;

    capture_device(qkvr, model->acts.qkvr + (size_t)layer * B * T * 3 * C, 3 * BTC);
    capture_device(att, model->acts.att + (size_t)layer * BHTT, BHTT);
    capture_device(atty, model->acts.atty + (size_t)layer * BTC, BTC);
    capture_device(attproj, model->acts.attproj + (size_t)layer * BTC, BTC);
    capture_device(residual2, model->acts.residual2 + (size_t)layer * BTC, BTC);
    capture_device(residual3_l2, model->acts.residual3 + (size_t)layer * BTC, BTC);
    capture_device(final_residual, model->acts.residual3 + (size_t)(L - 1) * BTC, BTC);
}

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4;
    int T = argc > 2 ? atoi(argv[2]) : 512;
    const int layer = 2;
    if (B <= 0 || T <= 0) return 2;

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

    gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                   B, T, layer, 0.0f, qkv_custom, qkv_tf32);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<StageCapture> stages;
    capture_stage_baseline(stages, &model, B, T, layer);

    const float alphas[] = {0.125f, 0.25f, 0.5f, 1.0f};
    printf("[xray][stage-response] source=qkv L02 natural TF32-custom direction; locate where +/- symmetry first breaks\n");
    printf("[xray][stage-response] qkvr is post-permute Q/K/V; att is post-score+softmax attention matrix; atty is post attention-value GEMM\n");

    for (float alpha : alphas) {
        gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                       B, T, layer, alpha, qkv_custom, qkv_tf32);
        cudaCheck(cudaDeviceSynchronize());
        capture_stage_side(stages, &model, B, T, layer, true);

        gpt2_forward_qkv_natural_alpha(&model, loader.inputs, loader.targets,
                                       B, T, layer, -alpha, qkv_custom, qkv_tf32);
        cudaCheck(cudaDeviceSynchronize());
        capture_stage_side(stages, &model, B, T, layer, false);

        for (const StageCapture& stage : stages) {
            LocalSymmetryStats s = local_symmetry_stats(stage.baseline, stage.plus, stage.minus,
                                                        alpha, nullptr);
            printf("[xray][stage-response] alpha=%5.3f stage=%-14s plus_rel=%10.6e minus_rel=%10.6e odd_rel=%10.6e even_rel=%10.6e curvature_ratio=%10.6e\n",
                   alpha, stage.name, s.plus_rel_l2, s.minus_rel_l2,
                   s.odd_rel_l2, s.even_rel_l2, s.curvature_ratio);
        }
    }

    cudaCheck(cudaFree(qkv_tf32));
    cudaCheck(cudaFree(qkv_custom));
    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
