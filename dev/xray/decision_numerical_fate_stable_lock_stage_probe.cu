#pragma push_macro("main")
#undef main
#define main xray_l02_stage_embedded_main
#include "decision_numerical_fate_l02_stage_probe.cu"
#undef main
#pragma pop_macro("main")

#include <cstring>
#include <string>

struct XrayStableLockFamily {
    const char* name;
    XrayComponent component;
};

static const XrayStableLockFamily xray_stable_lock_families[] = {
    {"qkv", XRAY_QKV},
    {"attproj", XRAY_ATTPROJ},
    {"fc", XRAY_FC},
    {"fcproj", XRAY_FCPROJ},
};

struct XrayLockSpec {
    const XrayStableLockFamily* family;
    int layer;
};

static const XrayStableLockFamily* xray_find_lock_family(const std::string& name) {
    for (const auto& f : xray_stable_lock_families) {
        if (name == f.name) return &f;
    }
    return nullptr;
}

static bool xray_parse_lock_spec(const char* text, XrayLockSpec* out) {
    const std::string s(text ? text : "");
    const size_t colon = s.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= s.size()) return false;
    const std::string family_name = s.substr(0, colon);
    const XrayStableLockFamily* family = xray_find_lock_family(family_name);
    if (!family) return false;
    char* end = nullptr;
    const long layer = std::strtol(s.c_str() + colon + 1, &end, 10);
    if (!end || *end != '\0') return false;
    out->family = family;
    out->layer = (int)layer;
    return true;
}

static void xray_capture_layer_gpu_prefix(XrayL02GpuPrefix* p, GPT2* model,
                                          int layer, int b, int T, int P) {
    const int B = model->batch_size;
    const int C = model->config.channels;
    const int NH = model->config.num_heads;
    const size_t BTC = (size_t)B * T * C;

    xray_copy_gpu_prefix(p->input,
                         model->acts.residual3 + (size_t)(layer - 1) * BTC,
                         b, T, P, C);
    xray_copy_gpu_prefix(p->ln1,
                         model->acts.ln1 + (size_t)layer * BTC,
                         b, T, P, C);
    const float* qkvr_layer =
        model->acts.qkvr + (size_t)layer * B * T * 3 * C;
    xray_copy_qkv_canonical(p->qkv, qkvr_layer, B, b, T, P, C, NH);
    xray_copy_gpu_prefix(p->atty,
                         model->acts.atty + (size_t)layer * BTC,
                         b, T, P, C);
    xray_copy_gpu_prefix(p->attproj,
                         model->acts.attproj + (size_t)layer * BTC,
                         b, T, P, C);
    xray_copy_gpu_prefix(p->residual2,
                         model->acts.residual2 + (size_t)layer * BTC,
                         b, T, P, C);
    xray_copy_gpu_prefix(p->ln2,
                         model->acts.ln2 + (size_t)layer * BTC,
                         b, T, P, C);
    xray_copy_gpu_prefix(p->fch,
                         model->acts.fch + (size_t)layer * B * T * 4 * C,
                         b, T, P, 4 * C);
    xray_copy_gpu_prefix(p->gelu,
                         model->acts.fch_gelu + (size_t)layer * B * T * 4 * C,
                         b, T, P, 4 * C);
    xray_copy_gpu_prefix(p->fcproj,
                         model->acts.fcproj + (size_t)layer * BTC,
                         b, T, P, C);
    xray_copy_gpu_prefix(p->residual3,
                         model->acts.residual3 + (size_t)layer * BTC,
                         b, T, P, C);
}

static std::vector<double> xray_cpu64_finish_from_layer_stage(
    int layer, XrayL02Stage stage, const XrayL02GpuPrefix& g,
    const std::vector<XrayCpu64LayerWeights>& hw,
    const std::vector<float>& lnfw, const std::vector<float>& lnfb,
    int L, int P, int C, int NH) {

    const auto& w = hw[layer];
    std::vector<double> input = xray_to_double(g.input);
    std::vector<double> ln1, qkv, atty, attproj, residual2;
    std::vector<double> ln2, fch, gelu, fcproj, residual3;

    if (stage >= ST_LN1) ln1 = xray_to_double(g.ln1);
    else xray_cpu64_layernorm_rows(ln1, input, w.ln1w, w.ln1b, P, C);

    if (stage >= ST_QKV) qkv = xray_to_double(g.qkv);
    else xray_cpu64_matmul(qkv, ln1, w.qkvw, &w.qkvb, P, C, 3 * C);

    if (stage >= ST_ATTY) atty = xray_to_double(g.atty);
    else xray_cpu64_attention(atty, qkv, P, C, NH);

    if (stage >= ST_ATTPROJ) attproj = xray_to_double(g.attproj);
    else xray_cpu64_matmul(attproj, atty, w.attprojw, &w.attprojb, P, C, C);

    if (stage >= ST_RESIDUAL2) residual2 = xray_to_double(g.residual2);
    else xray_cpu64_add(residual2, input, attproj);

    if (stage >= ST_LN2) ln2 = xray_to_double(g.ln2);
    else xray_cpu64_layernorm_rows(ln2, residual2, w.ln2w, w.ln2b, P, C);

    if (stage >= ST_FCH) fch = xray_to_double(g.fch);
    else xray_cpu64_matmul(fch, ln2, w.fcw, &w.fcb, P, C, 4 * C);

    if (stage >= ST_GELU) gelu = xray_to_double(g.gelu);
    else xray_cpu64_gelu(gelu, fch);

    if (stage >= ST_FCPROJ) fcproj = xray_to_double(g.fcproj);
    else xray_cpu64_matmul(fcproj, gelu, w.fcprojw, &w.fcprojb, P, 4 * C, C);

    if (stage >= ST_RESIDUAL3) residual3 = xray_to_double(g.residual3);
    else xray_cpu64_add(residual3, residual2, fcproj);

    std::vector<double> residual = std::move(residual3);
    std::vector<double> a, q, y, ap, r2, n2, f, ge, fp, r3;
    for (int l = layer + 1; l < L; ++l) {
        const auto& wl = hw[l];
        xray_cpu64_layernorm_rows(a, residual, wl.ln1w, wl.ln1b, P, C);
        xray_cpu64_matmul(q, a, wl.qkvw, &wl.qkvb, P, C, 3 * C);
        xray_cpu64_attention(y, q, P, C, NH);
        xray_cpu64_matmul(ap, y, wl.attprojw, &wl.attprojb, P, C, C);
        xray_cpu64_add(r2, residual, ap);
        xray_cpu64_layernorm_rows(n2, r2, wl.ln2w, wl.ln2b, P, C);
        xray_cpu64_matmul(f, n2, wl.fcw, &wl.fcb, P, C, 4 * C);
        xray_cpu64_gelu(ge, f);
        xray_cpu64_matmul(fp, ge, wl.fcprojw, &wl.fcprojb, P, 4 * C, C);
        xray_cpu64_add(r3, r2, fp);
        residual.swap(r3);
    }

    std::vector<double> target(residual.end() - C, residual.end());
    return xray_cpu64_layernorm_row_double(target, lnfw, lnfb);
}

static XrayReadoutStats xray_cpu64_stats_from_lnf(
    const std::vector<double>& lnf,
    const std::vector<float>& wte,
    const std::vector<float>& gpu_logits,
    int V, int C, int ref_winner, int alt_winner) {
    return xray_cpu64_classifier(lnf, wte, gpu_logits,
                                 V, C, ref_winner, alt_winner);
}

int main(int argc, char** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 4;
    const int T = argc > 2 ? atoi(argv[2]) : 512;
    const int batch_index = argc > 3 ? atoi(argv[3]) : 5;
    const int target = argc > 4 ? atoi(argv[4]) : 134;

    if (B <= 0 || T <= 0 || batch_index < 0 || target < 0 || target >= B * T) {
        fprintf(stderr,
                "usage: %s [B=4] [T=512] [batch_index=5] [batch_local_token=134] [family:lock_layer ...]\n",
                argv[0]);
        return 2;
    }

    std::vector<XrayLockSpec> specs;
    if (argc > 5) {
        for (int i = 5; i < argc; ++i) {
            XrayLockSpec spec{};
            if (!xray_parse_lock_spec(argv[i], &spec)) {
                fprintf(stderr, "invalid lock spec '%s'; expected qkv:N|attproj:N|fc:N|fcproj:N\n",
                        argv[i]);
                return 2;
            }
            specs.push_back(spec);
        }
    } else {
        // Default is the validated scan_index=10374 stable-lock topology.
        specs.push_back({xray_find_lock_family("qkv"), 10});
        specs.push_back({xray_find_lock_family("attproj"), 7});
        specs.push_back({xray_find_lock_family("fc"), 2});
        specs.push_back({xray_find_lock_family("fcproj"), 3});
    }

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
                    B, T, 0, 1, 1);
    for (int i = 0; i <= batch_index; ++i) dataloader_next_batch(&loader);

    const int L = model.config.num_layers;
    const int C = model.config.channels;
    const int NH = model.config.num_heads;
    const int V = model.config.vocab_size;
    const int Vp = model.config.padded_vocab_size;
    const int b = target / T;
    const int t = target % T;
    const int P = t + 1;
    const long long scan_index = (long long)batch_index * B * T + target;

    for (const auto& spec : specs) {
        if (spec.layer <= 0 || spec.layer >= L) {
            fprintf(stderr, "lock layer %d for family %s is out of range; expected 1..%d\n",
                    spec.layer, spec.family->name, L - 1);
            return 2;
        }
    }

    // Reference decision for this case.
    gpt2_forward(&model, loader.inputs, NULL, B, T);
    cudaCheck(cudaDeviceSynchronize());
    int ref_winner = -1, ref_runner = -1;
    float ref_margin = 0.0f;
    xray_top2_at(&model, target, &ref_winner, &ref_runner, &ref_margin);

    std::vector<XrayCpu64LayerWeights> hw;
    xray_cpu64_load_suffix_weights(hw, &model);
    std::vector<float> wte((size_t)V * C), lnfw(C), lnfb(C);
    cudaCheck(cudaMemcpy(wte.data(), model.params.wte,
                         (size_t)V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfw.data(), model.params.lnfw,
                         C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lnfb.data(), model.params.lnfb,
                         C * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[xray][decision-lock-stage] device=%s cc=%d.%d B=%d T=%d batch=%d target=%d scan_index=%lld b=%d t=%d prefix=%d specs=%zu\n",
           prop.name, prop.major, prop.minor, B, T, batch_index, target,
           scan_index, b, t, P, specs.size());
    printf("[xray][decision-lock-stage] ref_top1=%d ref_runner=%d ref_top2_margin=%+.9e\n",
           ref_winner, ref_runner, (double)ref_margin);
    printf("[xray][decision-lock-stage] each stage point means: preserve the natural GPU prefix through that stage of the candidate lock layer, then evaluate all remaining computation in CPU64; nested switch deltas are not standalone operator contributions\n");

    int processed = 0;
    int disagreements = 0;
    int global_residual3_replay_exact = 1;
    int global_endpoint_valid = 1;
    int common_alt_winner = 1;
    int first_alt_winner = -1;

    for (const auto& spec : specs) {
        const auto& family = *spec.family;
        const int layer = spec.layer;
        ++processed;

        cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        gpt2_forward_one_component_tf32(&model, loader.inputs, NULL,
                                        B, T, 0, family.component);
        cudaCheck(cudaDeviceSynchronize());

        int alt_winner = -1, alt_runner = -1;
        float alt_margin = 0.0f;
        xray_top2_at(&model, target, &alt_winner, &alt_runner, &alt_margin);
        if (first_alt_winner < 0) first_alt_winner = alt_winner;
        else common_alt_winner &= (alt_winner == first_alt_winner);

        std::vector<float> alt_logits(V);
        cudaCheck(cudaMemcpy(alt_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        const double gpu_pair =
            (double)alt_logits[ref_winner] - (double)alt_logits[alt_winner];

        printf("[xray][decision-lock-stage-family] family=%-7s lock_layer=%02d ref=%d ref_runner=%d alt=%d alt_runner=%d alt_top2_margin=%+.9e gpu_pair=%+.9e disagreement=%d\n",
               family.name, layer, ref_winner, ref_runner,
               alt_winner, alt_runner, (double)alt_margin, gpu_pair,
               alt_winner != ref_winner ? 1 : 0);

        if (alt_winner == ref_winner) {
            printf("[xray][decision-lock-stage-family-summary] family=%s lock_layer=%d skipped=1 reason=no_final_decision_disagreement\n",
                   family.name, layer);
            continue;
        }
        ++disagreements;

        // Capture every exact GPU stage on the natural alternate trajectory.
        XrayL02GpuPrefix g;
        xray_capture_layer_gpu_prefix(&g, &model, layer, b, T, P);

        // Validity control 1: exact residual3 checkpoint must replay the natural
        // target vocabulary through the original GPU suffix bit-for-bit.
        const size_t state_n = (size_t)B * T * C;
        const size_t state_bytes = state_n * sizeof(float);
        float* d_checkpoint = nullptr;
        cudaCheck(cudaMalloc((void**)&d_checkpoint, state_bytes));
        cudaCheck(cudaMemcpy(d_checkpoint,
                             model.acts.residual3 + (size_t)layer * state_n,
                             state_bytes, cudaMemcpyDeviceToDevice));
        xray_forward_from_residual3(&model, d_checkpoint, layer, B, T);
        cudaCheck(cudaDeviceSynchronize());
        std::vector<float> replay_logits(V);
        cudaCheck(cudaMemcpy(replay_logits.data(),
                             model.acts.output + (size_t)target * Vp,
                             (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int replay_unequal = 0;
        double replay_max_abs = 0.0;
        xray_compare_gpu_logits(alt_logits, replay_logits,
                                &replay_unequal, &replay_max_abs);
        const int residual3_replay_exact = replay_unequal == 0;
        global_residual3_replay_exact &= residual3_replay_exact;
        cudaCheck(cudaFree(d_checkpoint));

        // Validity control 2: generic stage evaluator endpoints must agree with
        // the independently defined CPU64 residual-checkpoint evaluator.
        const std::vector<double> input_ref_lnf =
            xray_cpu64_continue_from_residual_checkpoint(
                g.input, layer - 1, hw, lnfw, lnfb, L, P, C, NH);
        const std::vector<double> residual3_ref_lnf =
            xray_cpu64_continue_from_residual_checkpoint(
                g.residual3, layer, hw, lnfw, lnfb, L, P, C, NH);
        const XrayReadoutStats input_ref_stats =
            xray_cpu64_stats_from_lnf(input_ref_lnf, wte, alt_logits,
                                      V, C, ref_winner, alt_winner);
        const XrayReadoutStats residual3_ref_stats =
            xray_cpu64_stats_from_lnf(residual3_ref_lnf, wte, alt_logits,
                                      V, C, ref_winner, alt_winner);

        std::vector<XrayReadoutStats> stage_stats;
        stage_stats.reserve(ST_COUNT);
        for (int si = 0; si < ST_COUNT; ++si) {
            const auto stage = (XrayL02Stage)si;
            const std::vector<double> lnf =
                xray_cpu64_finish_from_layer_stage(
                    layer, stage, g, hw, lnfw, lnfb, L, P, C, NH);
            const XrayReadoutStats stats =
                xray_cpu64_stats_from_lnf(lnf, wte, alt_logits,
                                          V, C, ref_winner, alt_winner);
            stage_stats.push_back(stats);
        }

        const double input_endpoint_err =
            stage_stats[ST_INPUT].pair_margin - input_ref_stats.pair_margin;
        const double residual3_endpoint_err =
            stage_stats[ST_RESIDUAL3].pair_margin - residual3_ref_stats.pair_margin;
        const int endpoint_valid =
            stage_stats[ST_INPUT].winner == input_ref_stats.winner &&
            stage_stats[ST_RESIDUAL3].winner == residual3_ref_stats.winner &&
            std::fabs(input_endpoint_err) <= 1.0e-12 &&
            std::fabs(residual3_endpoint_err) <= 1.0e-12;
        global_endpoint_valid &= endpoint_valid;

        int first_alt_stage = -1;
        int first_not_ref_stage = -1;
        int winner_changes = 0;
        int pair_sign_changes = 0;
        for (int si = 0; si < ST_COUNT; ++si) {
            const auto& s = stage_stats[(size_t)si];
            if (first_alt_stage < 0 && s.winner == alt_winner) first_alt_stage = si;
            if (first_not_ref_stage < 0 && s.winner != ref_winner) first_not_ref_stage = si;
            if (si > 0) {
                if (s.winner != stage_stats[(size_t)si - 1].winner) ++winner_changes;
                if (xray_sign64(s.pair_margin) !=
                    xray_sign64(stage_stats[(size_t)si - 1].pair_margin)) {
                    ++pair_sign_changes;
                }
            }
        }

        int stable_alt_from_stage = -1;
        if (stage_stats.back().winner == alt_winner) {
            stable_alt_from_stage = ST_COUNT - 1;
            while (stable_alt_from_stage > 0 &&
                   stage_stats[(size_t)stable_alt_from_stage - 1].winner == alt_winner) {
                --stable_alt_from_stage;
            }
        }

        int last_winner_change_stage = -1;
        int last_pair_sign_change_stage = -1;
        for (int si = 1; si < ST_COUNT; ++si) {
            if (stage_stats[(size_t)si].winner != stage_stats[(size_t)si - 1].winner) {
                last_winner_change_stage = si;
            }
            if (xray_sign64(stage_stats[(size_t)si].pair_margin) !=
                xray_sign64(stage_stats[(size_t)si - 1].pair_margin)) {
                last_pair_sign_change_stage = si;
            }
        }

        for (int si = 0; si < ST_COUNT; ++si) {
            const auto& s = stage_stats[(size_t)si];
            const double switch_delta = si == 0 ? 0.0 :
                s.pair_margin - stage_stats[(size_t)si - 1].pair_margin;
            printf("[xray][decision-lock-stage-point] family=%-7s layer=%02d stage=%-9s cpu64_top1=%d cpu64_runner=%d cpu64_pair=%+.9e matches_alt=%d is_ref=%d pair_sign=%+d switch_delta=%+.9e\n",
                   family.name, layer, xray_stage_name((XrayL02Stage)si),
                   s.winner, s.runner, s.pair_margin,
                   s.winner == alt_winner ? 1 : 0,
                   s.winner == ref_winner ? 1 : 0,
                   xray_sign64(s.pair_margin), switch_delta);
        }

        printf("[xray][decision-lock-stage-validation] family=%-7s layer=%02d residual3_gpu_replay_exact=%d replay_unequal=%d/%d replay_max_abs=%.9e input_ref_top1=%d input_stage_top1=%d input_ref_pair=%+.9e input_stage_pair=%+.9e input_endpoint_err=%+.9e residual3_ref_top1=%d residual3_stage_top1=%d residual3_ref_pair=%+.9e residual3_stage_pair=%+.9e residual3_endpoint_err=%+.9e endpoint_valid=%d\n",
               family.name, layer,
               residual3_replay_exact, replay_unequal, V, replay_max_abs,
               input_ref_stats.winner, stage_stats[ST_INPUT].winner,
               input_ref_stats.pair_margin, stage_stats[ST_INPUT].pair_margin,
               input_endpoint_err,
               residual3_ref_stats.winner, stage_stats[ST_RESIDUAL3].winner,
               residual3_ref_stats.pair_margin, stage_stats[ST_RESIDUAL3].pair_margin,
               residual3_endpoint_err, endpoint_valid);

        printf("[xray][decision-lock-stage-family-summary] family=%-7s lock_layer=%02d first_alt_stage=%s stable_alt_from_stage=%s last_winner_change_stage=%s winner_changes=%d pair_sign_changes=%d last_pair_sign_change_stage=%s residual3_gpu_replay_exact=%d endpoint_valid=%d\n",
               family.name, layer,
               first_alt_stage >= 0 ? xray_stage_name((XrayL02Stage)first_alt_stage) : "none",
               stable_alt_from_stage >= 0 ? xray_stage_name((XrayL02Stage)stable_alt_from_stage) : "none",
               last_winner_change_stage >= 0 ? xray_stage_name((XrayL02Stage)last_winner_change_stage) : "none",
               winner_changes, pair_sign_changes,
               last_pair_sign_change_stage >= 0 ? xray_stage_name((XrayL02Stage)last_pair_sign_change_stage) : "none",
               residual3_replay_exact, endpoint_valid);
    }

    printf("[xray][decision-lock-stage-summary] processed=%d disagreements=%d global_residual3_replay_exact=%d global_endpoint_valid=%d common_alt_winner=%d alt_winner=%d\n",
           processed, disagreements, global_residual3_replay_exact,
           global_endpoint_valid, common_alt_winner, first_alt_winner);
    printf("[xray][decision-lock-stage-summary] interpretation gate: use stable_alt_from_stage only when residual3 replay and endpoint validation pass; stage points are nested GPU-prefix/CPU64-suffix execution switches, not additive operator attributions; a shared stage class across families is evidence for a shared computational channel only at this tested case and depth set\n");

    dataloader_free(&loader);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
