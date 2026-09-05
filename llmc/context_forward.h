#ifndef LLMC_CONTEXT_FORWARD_H
#define LLMC_CONTEXT_FORWARD_H

/*
 * Context-only GPT-2 forward path for LSMM.
 *
 * This header is intended to be included after train_gpt2.c has defined the
 * GPT2 model and layer primitives. It computes the same Transformer state as
 * gpt2_forward() through the final layer norm, but deliberately stops before
 * vocabulary projection, softmax, and loss evaluation.
 *
 * The resulting contextual representation is model->acts.lnf with shape
 * [B][T][C]. Model parameters are unchanged.
 */
static void gpt2_forward_context(GPT2 *model, int *inputs, size_t B, size_t T) {
    if (model == NULL || inputs == NULL || B == 0 || T == 0) {
        return;
    }

    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    size_t V = model->config.vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    for (size_t i = 0; i < B * T; ++i) {
        assert(0 <= inputs[i] && inputs[i] < (int)V);
    }

    if (model->acts_memory == NULL) {
        model->batch_size = (int)B;
        model->seq_len = (int)T;

        fill_in_activation_sizes(model->act_sizes, model->config, (int)B, (int)T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; ++i) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        model->inputs = (int *)mallocCheck(B * T * sizeof(int));
        model->targets = (int *)mallocCheck(B * T * sizeof(int));
    } else if (B != (size_t)model->batch_size || T != (size_t)model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n",
               model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    memcpy(model->inputs, inputs, B * T * sizeof(int));

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float *residual;

    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, (int)B, (int)T, (int)C);

    for (size_t l = 0; l < L; ++l) {
        residual = l == 0
            ? acts.encoded
            : acts.residual3 + (l - 1) * B * T * C;

        float *l_ln1w = params.ln1w + l * C;
        float *l_ln1b = params.ln1b + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_qkvb = params.qkvb + l * 3 * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_attprojb = params.attprojb + l * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_ln2b = params.ln2b + l * C;
        float *l_fcw = params.fcw + l * 4 * C * C;
        float *l_fcb = params.fcb + l * 4 * C;
        float *l_fcprojw = params.fcprojw + l * C * 4 * C;
        float *l_fcprojb = params.fcprojb + l * C;

        float *l_ln1 = acts.ln1 + l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkv = acts.qkv + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_preatt = acts.preatt + l * B * NH * T * T;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_attproj = acts.attproj + l * B * T * C;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + l * B * T * C;
        float *l_residual3 = acts.residual3 + l * B * T * C;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd,
                          residual, l_ln1w, l_ln1b,
                          (int)B, (int)T, (int)C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb,
                       (int)B, (int)T, (int)C, (int)(3 * C));
        attention_forward(l_atty, l_preatt, l_att, l_qkv,
                          (int)B, (int)T, (int)C, (int)NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb,
                       (int)B, (int)T, (int)C, (int)C);
        residual_forward(l_residual2, residual, l_attproj, (int)(B * T * C));
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd,
                          l_residual2, l_ln2w, l_ln2b,
                          (int)B, (int)T, (int)C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb,
                       (int)B, (int)T, (int)C, (int)(4 * C));
        gelu_forward(l_fch_gelu, l_fch, (int)(B * T * 4 * C));
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb,
                       (int)B, (int)T, (int)(4 * C), (int)C);
        residual_forward(l_residual3, l_residual2, l_fcproj, (int)(B * T * C));
    }

    residual = acts.residual3 + (L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                      residual, params.lnfw, params.lnfb,
                      (int)B, (int)T, (int)C);

    model->mean_loss = -1.0f;
}

#endif /* LLMC_CONTEXT_FORWARD_H */
