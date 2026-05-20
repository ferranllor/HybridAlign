#include "../include/definitions.h"

#pragma once

static inline DTYPEMATRIX max(DTYPEMATRIX a, DTYPEMATRIX b) {
    return (a > b) ? a : b;
}

static inline int get_diag_start(int d, int M, int N) {
    int l_min = (M < N) ? M : N;
    int l_max = (M > N) ? M : N;
    
    if (d <= l_min) {
        return (d * (d + 1)) / 2;
    } else if (d <= l_max) {
        return (l_min * (l_min + 1)) / 2 + (d - l_min) * (l_min + 1);
    } else {
        int rem = M + N - d + 1;
        return (M + 1) * (N + 1) - (rem * (rem + 1)) / 2;
    }
}

static inline int get_diagonal_index(int i, int j, int M, int N) {
    int d = i + j;
    int start = get_diag_start(d, M, N);
    int max_d_M = (0 > d - M) ? 0 : (d - M);
    return start + j - max_d_M;
}