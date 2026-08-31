#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

static inline void xray_compare_gpu_logits(const std::vector<float>& ref,
                                           const std::vector<float>& cur,
                                           int* unequal,
                                           double* max_abs) {
    int neq = 0;
    double mx = 0.0;
    const size_t n = std::min(ref.size(), cur.size());
    for (size_t i = 0; i < n; ++i) {
        if (ref[i] != cur[i]) ++neq;
        mx = std::max(mx, std::fabs((double)cur[i] - (double)ref[i]));
    }
    neq += (int)(ref.size() > n ? ref.size() - n : cur.size() - n);
    *unequal = neq;
    *max_abs = mx;
}
