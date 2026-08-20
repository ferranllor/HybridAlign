#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "definitions.h"
#pragma once

// Overridable from the command line (-DBLOCKSIZE=...) so tools/verify.sh can sweep them
#ifndef BLOCKSIZE
#define BLOCKSIZE 480
#endif

// Three rotating diagonals (d, d - 1, d - 2) is all the recurrence needs; the extra buffers only
// existed for the memcpy_async variant that is no longer in the kernel.
#ifndef N_BUFFERS
#define N_BUFFERS 3
#endif

// A band of the DP matrix always runs along the shorter side, so a block never has anything for
// more than min(M, N) threads to do: BLOCKSIZE is only the upper cap. The width is taken from the
// largest node of the level (the smaller ones in the same level just leave the tail idle, exactly
// as before) and rounded up to a whole warp.
static inline int band_width_for_level(int max_node_size, int M) {
    int lo = (max_node_size < M) ? max_node_size : M;
    int w = ((lo + 31) / 32) * 32;

    if (w < 32) w = 32;
    return (w < BLOCKSIZE) ? w : BLOCKSIZE;
}

// Dynamic shared memory of one block of such a level: topRow (only needed when the band runs along
// the rows, i.e. M < N), prevCol, the N_BUFFERS rotating diagonals, then the two sequences.
static inline int shared_bytes_for_level(int max_node_size, int M, int band_width) {
    int matrix_slots = (M + 1) + N_BUFFERS * (band_width + 2);
    if (max_node_size > M) matrix_slots += max_node_size + 1;

    return (int)(matrix_slots * sizeof(DTYPEMATRIX) + (M + max_node_size) * sizeof(DTYPEALPHABET));
}

__device__ static inline int get_diag_start_device(int d, int M, int N) {
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

__device__  static inline int get_diagonal_index_device(int i, int j, int M, int N) {
    int d = i + j;
    int start = get_diag_start_device(d, M, N);
    
    int j_base = (d - M > 0) ? (d - M) : 0;
    return start + (j - j_base);
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
    
    int j_base = (d - M > 0) ? (d - M) : 0;
    return start + (j - j_base);
}

static inline void reverse_string(char* str, int len) {
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}