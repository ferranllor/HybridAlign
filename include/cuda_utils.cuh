#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE 128

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