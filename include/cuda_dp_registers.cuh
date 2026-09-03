#pragma once
#include "definitions.h"
#include "cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// The DP core of the registers version, on ONE node, for ONE warp. It used to be the body of
// compute_dp_gpu_registers: it lives in a header now for the same reason the shared memory core
// does, the build has no device side linking, so the multi read version and the persistent kernels
// cannot reach a __device__ function sitting in another .cu.
//
// What the callers differ in is only where the columns are. A single sequence version passes
// col_off 0 and the graph's own last columns; the multi read version passes the offset of the read
// it is aligning, so the same node carries one last column per read.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// A band needs slots 0 .. stripe_height + 1: the cells it owns plus the two halos holding the
// neighbouring row or column it does not. Lane L holds slot L, so the band is two cells narrower
// than the warp and 30 + 2 lands exactly on 32.
#ifndef REG_BAND
#define REG_BAND (WARP_SIZE - 2)
#endif

// Shared memory of ONE warp: topRow (only when the band runs along the rows, i.e. M < N), prevCol,
// then the two sequences. The rotating diagonals are in registers, that is the point of it.
static inline int registers_dp_toprow_slots(int max_node_size, int M) {
    return (max_node_size > M) ? (max_node_size + 1) : 0;
}

static inline int registers_dp_elems_per_warp(int max_node_size, int M) {
    return registers_dp_toprow_slots(max_node_size, M) + (M + 1) + seq_slots_for_level(max_node_size, M);
}

__device__ static void compute_dp_node_registers(Node* node, int M,
                                                 const DTYPEALPHABET* __restrict query_rev,
                                                 size_t col_off, DTYPEMATRIX* mine, int toprow_slots,
                                                 int& best_max, int& best_d, int& best_j)
{
    const unsigned int mask = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // ------------------------------------------------- Initialize -------------------------------------------------

    int N = node->sequence.size;

    DTYPEMATRIX* __restrict last_col = &node->last_col[col_off];

    const DTYPEALPHABET* __restrict node_seq = node->sequence.sequence;

    DTYPEMATRIX* topRow = mine;
    DTYPEMATRIX* prevCol = &mine[toprow_slots];

    DTYPEALPHABET* node_shared = (DTYPEALPHABET*)&prevCol[M + 1];
    DTYPEALPHABET* query_shared = &node_shared[N];

    for (int i = lane; i < N; i += WARP_SIZE) node_shared[i] = node_seq[i];
    for (int i = lane; i < M; i += WARP_SIZE) query_shared[i] = query_rev[i];

    if (M < N)
        for (int j = lane; j <= N; j += WARP_SIZE) topRow[j] = 0;

    if (node->num_in == 0) {
        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = 0;
    }
    else if (node->num_in == 1) {
        const DTYPEMATRIX* __restrict prev_last = &node->v_in[0]->last_col[col_off];

        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = prev_last[i];
    }
    else {
        const DTYPEMATRIX* __restrict prev_last  = &node->v_in[0]->last_col[col_off];
        const DTYPEMATRIX* __restrict prev_last2 = &node->v_in[1]->last_col[col_off];

        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = max(prev_last[i], prev_last2[i]);

        for (int p = 2; p < node->num_in; ++p) {
            const DTYPEMATRIX* __restrict prev_lastp = &node->v_in[p]->last_col[col_off];
            for (int i = lane; i <= M; i += WARP_SIZE)
                prevCol[i] = max(prev_lastp[i], prevCol[i]);
        }
    }

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    int v_a = 0, v_p = 0, v_pp = 0;

    auto process_diagonal = [&](int klo, int khi, int current_d,
                            int node_off, int query_off,
                            int off_diag, int off_up, int off_left) {

        int up_p    = (off_up   == 0) ? v_p  : __shfl_down_sync(mask, v_p, 1);
        int left_p  = (off_left == 0) ? v_p  : __shfl_up_sync(mask, v_p, 1);
        int diag_pp = (off_diag == 1) ? __shfl_down_sync(mask, v_pp, 1)
                                      : __shfl_up_sync(mask, v_pp, 1);

        if (lane >= klo && lane <= khi) {
            int score = (node_shared[node_off + lane] == query_shared[query_off + lane]) ? MATCH : MISMATCH;

            int diagonal = diag_pp + score;
            int up       = up_p + GAP;
            int left     = left_p + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            v_a = res;

            int j = node_off + lane + 1;
            if (j == N) last_col[current_d - N] = res;

            if (res > local_max) {
                local_max = res;
                local_max_d = current_d;
                local_max_j = j;
            }
        }

        int tmp = v_pp;
        v_pp = v_p;
        v_p  = v_a;
        v_a  = tmp;
    };

    __syncwarp(mask);

    if (M >= N) {
        for (int startN = 0; startN < N; startN += REG_BAND) {
            int stripe_height = min(REG_BAND, N - startN);

            int js = startN + 1;
            int je = startN + stripe_height;

            __syncwarp(mask);

            v_a = 0; v_p = 0; v_pp = 0;
            if (lane == 0) v_p = prevCol[1];

            int d = js + 1;

            // --------------- Grow phase ------------------

            for (; d <= je; ++d) {
                int khi = d - 1 - startN;

                if (lane == 0)       v_a = prevCol[d - startN];
                if (lane == khi + 1) v_a = 0;

                process_diagonal(1, khi, d, startN - 1, M - d + startN, -1, 0, -1);
            }

            // --------------- Stable phase ------------------

            for (; d <= M; ++d) {
                if (lane == 0) v_a = prevCol[d - startN];

                process_diagonal(1, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                int handover = __shfl_sync(mask, v_p, stripe_height);
                if (lane == 0) prevCol[d - je] = handover;
            }

            // --------------- Shrink phase -----------------

            for (; d <= je + M; ++d) {
                int j_min = d - M;
                int klo = max(1, j_min - startN);

                if (lane == 0 && j_min <= startN) v_a = prevCol[d - startN];

                process_diagonal(klo, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                int handover = __shfl_sync(mask, v_p, stripe_height);
                if (lane == 0) prevCol[d - je] = handover;
            }
        }
    }
    else {
        for (int startM = 0; startM < M; startM += REG_BAND) {
            int stripe_height = min(REG_BAND, M - startM);

            int is = startM + 1;
            int ie = startM + stripe_height;

            __syncwarp(mask);

            v_a = 0; v_p = 0; v_pp = 0;
            if (lane == stripe_height + 1) { v_pp = prevCol[startM]; v_p = topRow[1]; }
            if (lane == stripe_height)     { v_p = prevCol[is]; }

            int d = is + 1;

            // --------------- Grow phase ------------------

            for (; d <= ie; ++d) {
                int klo = ie + 2 - d;

                if (lane == klo - 1)           v_a = prevCol[d];
                if (lane == stripe_height + 1) v_a = topRow[d - startM];

                process_diagonal(klo, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);
            }

            // --------------- Stable phase ------------------

            for (; d <= is + N; ++d) {
                if (lane == stripe_height + 1 && d - startM <= N) v_a = topRow[d - startM];

                process_diagonal(1, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                int handover = __shfl_sync(mask, v_p, 1);
                if (lane == 0) topRow[d - ie] = handover;
            }

            // --------------- Shrink phase -----------------

            for (; d <= ie + N; ++d) {
                int khi = ie + 1 - d + N;

                process_diagonal(1, khi, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                int handover = __shfl_sync(mask, v_p, 1);
                if (lane == 0) topRow[d - ie] = handover;
            }
        }
    }

    __syncwarp(mask);

    // ------------------ Reduce the local max ------------------

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        int cand_max = __shfl_down_sync(mask, local_max, off);
        int cand_d   = __shfl_down_sync(mask, local_max_d, off);
        int cand_j   = __shfl_down_sync(mask, local_max_j, off);

        bool is_greater = (cand_max > local_max) ||
                          (cand_max == local_max && cand_d != -1 &&
                              (local_max_d == -1 || cand_d < local_max_d ||
                                  (cand_d == local_max_d && cand_j < local_max_j)));

        if (is_greater) { local_max = cand_max; local_max_d = cand_d; local_max_j = cand_j; }
    }

    best_max = local_max;
    best_d   = local_max_d;
    best_j   = local_max_j;
}
