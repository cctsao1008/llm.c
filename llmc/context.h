#ifndef LLMC_CONTEXT_H
#define LLMC_CONTEXT_H

#include <stddef.h>

/*
 * Read-only view of the final GPT-2 contextual representation.
 *
 * This header is intended to be included after train_gpt2.c has defined GPT2.
 * The view points directly at model->acts.lnf, the final layer-normalized
 * activation with shape [B][T][C], immediately before vocabulary projection.
 *
 * The returned pointer is owned by GPT2 and remains valid only while the model
 * and its activation storage remain alive and unchanged.
 */
typedef struct {
    const float *states;  /* [batch_size][seq_len][width] */
    size_t batch_size;
    size_t seq_len;
    size_t width;
} GPT2ContextView;

static inline int gpt2_context_view(const GPT2 *model, GPT2ContextView *view) {
    if (model == NULL || view == NULL) {
        return 0;
    }

    if (model->acts_memory == NULL || model->acts.lnf == NULL) {
        return 0;
    }

    view->states = model->acts.lnf;
    view->batch_size = (size_t)model->batch_size;
    view->seq_len = (size_t)model->seq_len;
    view->width = (size_t)model->config.channels;
    return 1;
}

#endif /* LLMC_CONTEXT_H */
