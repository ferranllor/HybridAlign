#pragma once
#include "definitions.h"
#include "cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// The DP core of the shared memory version, on ONE node. It used to be the body of
// compute_dp_gpu_shared_mem: it lives in a header now because the persistent kernel version has to
// call it from inside its own worker kernel, and the build has no device side linking (-rdc), so a
// __device__ function in another .cu is not reachable. Both callers get the same code this way.
//
// Everything it needs from the launch it reads from the launch: the band width is blockDim.x and
// the scratch space is the dynamic shared memory. The caller is therefore the one responsible for
// sizing both, with band_width_for_level() and shared_bytes_for_level().


// The rotating diagonals are band-width sized, so they live in dynamic shared memory as well: a
// level of short nodes then gets both a narrow block and a small footprint, and it is the footprint
// that decides how many of those blocks an SM can keep resident. Indexed as dpBuffers[buffer][k],
// like the static array it replaces.
struct DpBuffers {
    DTYPEMATRIX* base;
    int stride;

    __device__ DTYPEMATRIX* operator[](int b) const { return base + b * stride; }
};

__device__ static void compute_dp_node_shared_mem(Node* node, Sequence sequence, Sequence sequence_rev)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    int M = sequence.size;
    int N = node->sequence.size;
    DTYPEMATRIX* __restrict dp = node->dp_matrix;

    // The band is as wide as the block: the launch already sized the block after min(M, N) of the
    // widest node of this level, so BLOCKSIZE is never the width here, only its cap.
    int band_width = blockDim.x;

    extern __shared__ DTYPEMATRIX shared[];

    // topRow only ever carries something when the band runs along the rows (M < N). In the M >= N
    // case it is the top row of the matrix, which is all zeros for a local alignment and is never
    // written, so it is neither allocated nor read there.
    DTYPEMATRIX* topRow = shared;
    DTYPEMATRIX* prevCol = (M < N) ? &(shared[N+1]) : shared;

    DpBuffers dpBuffers = { &prevCol[M + 1], band_width + 2 };

    if (M < N)
        for (int j = threadIdx.x; j <= N; j += blockDim.x) topRow[j] = 0;

    if (node->num_in == 0) {
        for (int i = threadIdx.x; i <= M; i += blockDim.x) 
            prevCol[i] = 0;
    }
    else if (node->num_in == 1) {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;
        int prev_N = node->v_in[0]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            prevCol[i] = prev_dp[get_diagonal_index_device(i, prev_N, M, prev_N)];
    }
    else {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 
        int prev_N1 = node->v_in[0]->sequence.size;
        int prev_N2 = node->v_in[1]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x) {
            int score1 = prev_dp[get_diagonal_index_device(i, prev_N1, M, prev_N1)];
            int score2 = prev_dp2[get_diagonal_index_device(i, prev_N2, M, prev_N2)];
            prevCol[i] = max(score1, score2);
        }

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            int prev_Ni = node->v_in[i]->sequence.size;
            for (int j = threadIdx.x; j <= M; j += blockDim.x) {
                int act = get_diagonal_index_device(j, 0, M, N);
                prevCol[j] = max(prev_dp[get_diagonal_index_device(j, prev_Ni, M, prev_Ni)], prevCol[j]);
            }
        }
    }
    
    if (node->num_in == 0) {
        for (int i = threadIdx.x; i <= M; i += blockDim.x) 
            dp[get_diagonal_index_device(i, 0, M, N)] = 0;
        
        for (int j = threadIdx.x + 1; j <= N; j += blockDim.x) 
            dp[get_diagonal_index_device(0, j, M, N)] = 0;
    }
    else if (node->num_in == 1) {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;
        int prev_N = node->v_in[0]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            dp[get_diagonal_index_device(i, 0, M, N)] = prev_dp[get_diagonal_index_device(i, prev_N, M, prev_N)];
        
        for (int j = threadIdx.x + 1; j <= N; j += blockDim.x) 
            dp[get_diagonal_index_device(0, j, M, N)] = 0;
    }
    else {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 
        int prev_N1 = node->v_in[0]->sequence.size;
        int prev_N2 = node->v_in[1]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x) {
            int score1 = prev_dp[get_diagonal_index_device(i, prev_N1, M, prev_N1)];
            int score2 = prev_dp2[get_diagonal_index_device(i, prev_N2, M, prev_N2)];
            dp[get_diagonal_index_device(i, 0, M, N)] = max(score1, score2);
        }

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            int prev_Ni = node->v_in[i]->sequence.size;
            for (int j = threadIdx.x; j <= M; j += blockDim.x) {
                int act = get_diagonal_index_device(j, 0, M, N);
                dp[act] = max(prev_dp[get_diagonal_index_device(j, prev_Ni, M, prev_Ni)], dp[act]);
            }
        }
        for (int j = threadIdx.x + 1; j <= N; j += blockDim.x) dp[get_diagonal_index_device(0, j, M, N)] = 0;
    }

    DTYPEALPHABET* node_seq = (DTYPEALPHABET*)dpBuffers[N_BUFFERS]; // one past the last buffer
    DTYPEALPHABET* query_seq_rev = &node_seq[N];

    for (int i = threadIdx.x; i < node->sequence.size; i+= blockDim.x)
    {
        node_seq[i] = node->sequence.sequence[i];
    }

    for (int i = threadIdx.x; i < sequence_rev.size; i+= blockDim.x)
    {
        query_seq_rev[i] = sequence_rev.sequence[i];
    }

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    // The matrix is walked band by band: band_width consecutive columns at a time when M >= N,
    // band_width consecutive rows at a time when M < N (i.e. always along the shorter side, so a
    // whole anti-diagonal of the band fits in one dpBuffers row). Inside a band the local buffer
    // index k is tied to the band coordinate and *not* to the position inside the diagonal, so the
    // neighbour offsets stay constant for every diagonal and every phase:
    //
    //   M >= N : k = j - startN            -> up = [k], left = [k-1], diagonal = [k-1]
    //   M <  N : k = (startM + h) + 1 - i   -> up = [k+1], left = [k], diagonal = [k+1]
    //
    // k = 0 (resp. k = h + 1) is the halo slot holding the neighbouring column/row that the band
    // does not own: the carry column startN (prevCol) / the carry row startM (topRow), plus the
    // top row of the matrix where the diagonal has not reached the end of the band yet.

    int bufferAct, bufferPrev, bufferPrevPrev;
    int write_base; // dp[write_base + k] is where local index k of the current diagonal is stored

    auto process_diagonal = [&](int klo, int khi, int current_d,
                            int node_off, int query_off,
                            int off_diag, int off_up, int off_left) {
        int prev_max = local_max;

        // 1. Compute, and write straight through to global memory: a thread only ever reads back
        //    the cell it wrote itself, so the store needs no barrier and no second pass over the
        //    buffer. (The barrier below is for the *next* diagonal, which reads this one's cells
        //    across threads.)
        for (int k = klo + threadIdx.x; k <= khi; k += blockDim.x) {
            int score = (node_seq[node_off + k] == query_seq_rev[query_off + k]) ? MATCH : MISMATCH;

            int diagonal = dpBuffers[bufferPrevPrev][k + off_diag] + score;
            int up       = dpBuffers[bufferPrev][k + off_up] + GAP;
            int left     = dpBuffers[bufferPrev][k + off_left] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dpBuffers[bufferAct][k] = res;
            dp[write_base + k] = res;

            local_max = max(res, local_max);
        }

        if (prev_max != local_max) {
            local_max_d = current_d;
        }

        __syncthreads();

        // 2. Buffer Rotation. N_BUFFERS is not a power of two, so wrap with a select instead of
        //    paying for three integer modulos on every diagonal.
        bufferPrevPrev = bufferPrev;
        bufferPrev     = bufferAct;
        bufferAct      = (bufferAct + 1 == N_BUFFERS) ? 0 : bufferAct + 1;
    };


    __syncthreads();

    if (M >= N) {
        // Bands of columns [js, je]. prevCol carries column startN over from the previous band.
        for (int startN = 0; startN < N; startN += band_width) {
            int stripe_height = min(band_width, N - startN);

            int js = startN + 1;                // first column owned by the band
            int je = startN + stripe_height;    // last column owned by the band

            __syncthreads();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            // Seed the two diagonals that precede the band: (0, startN) and (0, startN + 1) sit on
            // the top row (all zeros), (1, startN) is the first cell of the carry column.
            if (threadIdx.x == 0) {
                dpBuffers[bufferPrevPrev][0] = 0;
                dpBuffers[bufferPrev][0]     = prevCol[1];
                dpBuffers[bufferPrev][1]     = 0;
            }

            __syncthreads();

            int d = js + 1;                                     // first diagonal that hits the band
            int diag_start = get_diag_start_device(d, M, N);    // where diagonal d starts in dp

            // --------------- Grow phase ------------------
            // The band's piece of the diagonal is still growing: it runs from the carry column down
            // to the top row of the matrix, which is still inside the band. Note this ends at je,
            // not at l_min: what grows is the band's piece, not the matrix diagonal.

            for (; d <= je; ++d) {
                int khi = d - 1 - startN;               // the cell above it is on the top row

                write_base = diag_start + startN;       // d <= M, so no column is missing yet

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][0]       = prevCol[d - startN];  // carry column
                    dpBuffers[bufferAct][khi + 1] = 0;                    // top row of the matrix
                }

                process_diagonal(1, khi, d, startN - 1, M - d + startN, -1, 0, -1);

                diag_start += d + 1;                    // diagonal d holds d + 1 cells (d <= N)
            }

            // --------------- Stable phase ------------------
            // The band is full: every diagonal crosses all of its columns, so the last one is
            // finished on each pass and can be handed over to the next band.

            for (; d <= M; ++d) {
                write_base = diag_start + startN;

                if (threadIdx.x == 0) dpBuffers[bufferAct][0] = prevCol[d - startN];

                process_diagonal(1, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                if (threadIdx.x == 0) prevCol[d - je] = dpBuffers[bufferPrev][stripe_height];

                diag_start += min(d, N) + 1;            // d + 1 while d <= N, then N + 1
            }

            // --------------- Shrink phase -----------------
            // Past d = M the matrix diagonals start losing their first column, so the band's piece
            // starts further and further in until only its last column is left.

            for (; d <= je + M; ++d) {
                int j_min = d - M;                      // first column stored on this diagonal
                int klo = max(1, j_min - startN);

                write_base = diag_start + startN - j_min;

                // The carry column only exists while its cell is still inside the matrix.
                if (threadIdx.x == 0 && j_min <= startN) dpBuffers[bufferAct][0] = prevCol[d - startN];

                process_diagonal(klo, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                if (threadIdx.x == 0) prevCol[d - je] = dpBuffers[bufferPrev][stripe_height];

                diag_start += M + N - d + 1;
            }
        }
    }
    else {
        // Bands of rows [is, ie]. topRow carries row startM over from the previous band, prevCol is
        // the column 0 boundary and is only read.
        for (int startM = 0; startM < M; startM += band_width) {
            int stripe_height = min(band_width, M - startM);

            int is = startM + 1;                // first row owned by the band
            int ie = startM + stripe_height;    // last row owned by the band

            __syncthreads();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            // Seed the two diagonals that precede the band: (startM, 0) and (startM, 1) are on the
            // carry row, (is, 0) is the first cell of the band on the column 0 boundary.
            if (threadIdx.x == 0) {
                dpBuffers[bufferPrevPrev][stripe_height + 1] = prevCol[startM];
                dpBuffers[bufferPrev][stripe_height + 1]     = topRow[1];
                dpBuffers[bufferPrev][stripe_height]         = prevCol[is];
            }

            __syncthreads();

            int d = is + 1;                                     // first diagonal that hits the band
            int diag_start = get_diag_start_device(d, M, N);    // where diagonal d starts in dp

            // --------------- Grow phase ------------------
            // Mirror image of the M >= N case: the band's piece runs from the column 0 boundary
            // down to the last row of the band, and grows until it covers every row of it.

            for (; d <= ie; ++d) {
                int klo = ie + 2 - d;                   // the cell left of it is on column 0

                write_base = diag_start + (d - ie - 1); // d <= M, so no column is missing yet

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][klo - 1]           = prevCol[d];          // column 0
                    dpBuffers[bufferAct][stripe_height + 1] = topRow[d - startM];  // carry row
                }

                process_diagonal(klo, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                diag_start += d + 1;                    // d <= M < N, the diagonal is still growing
            }

            // --------------- Stable phase ------------------
            // The band is full: the last row is finished on every diagonal and handed over to the
            // next band. The matrix diagonal itself may still grow, stay or shrink here - that only
            // shows up in its length and in how many columns it has already lost.

            for (; d <= is + N; ++d) {
                int j_min = max(0, d - M);              // first column stored on this diagonal

                write_base = diag_start + (d - ie - 1) - j_min;

                // The carry row stops once the diagonal runs past the end of the query.
                if (threadIdx.x == 0 && d - startM <= N)
                    dpBuffers[bufferAct][stripe_height + 1] = topRow[d - startM];

                process_diagonal(1, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                if (threadIdx.x == 0) topRow[d - ie] = dpBuffers[bufferPrev][1];

                diag_start += min(d, N) - j_min + 1;
            }

            // --------------- Shrink phase -----------------
            // The diagonal has run past the last column, so it now leaves the band row by row.

            for (; d <= ie + N; ++d) {
                int khi = ie + 1 - d + N;               // first row of the band it still reaches

                write_base = diag_start + M - ie - 1;   // = diag_start + (d - ie - 1) - (d - M)

                process_diagonal(1, khi, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                if (threadIdx.x == 0) topRow[d - ie] = dpBuffers[bufferPrev][1];

                diag_start += M + N - d + 1;
            }
        }
    }
    __syncthreads();

    // ------------------ Find j of local max ------------------

    __syncthreads();

    int t = threadIdx.x;

    // The band loops are over, so the rotating diagonals are free: they are band_width + 2 wide,
    // which is exactly one slot per thread plus change.
    DTYPEMATRIX* local_max_red   = dpBuffers[0];
    DTYPEMATRIX* local_max_d_red = dpBuffers[1];

    local_max_red[t] = local_max;
    local_max_d_red[t] = local_max_d;

    __syncthreads(); 

    for (unsigned int live = blockDim.x; live > 1; ) {
        unsigned int stride = (live + 1) / 2;

        if (t + stride < live) {
            int curr_max = local_max_red[t];
            int candidate = local_max_red[t + stride];
            
            int curr_d = local_max_d_red[t];
            int candidate_d = local_max_d_red[t + stride];

            bool is_greater = (candidate > curr_max);
            
            local_max_red[t]  = is_greater ? candidate : curr_max;
            local_max_d_red[t] = is_greater ? candidate_d : curr_d; 
        }
        __syncthreads();
        live = stride;
    }

    local_max = local_max_red[0];
    local_max_d = local_max_d_red[0];

    if (local_max_d != -1) {

        __shared__ int shared_min_j;
        if (threadIdx.x == 0) {
            shared_min_j = INT_MAX; 
        }

        __syncthreads();
        
        int final_d = local_max_d;
        int j_start = (local_max_d - M > 1) ? local_max_d - M : 1;
        int j_end = (local_max_d - 1 < N) ? local_max_d - 1 : N;

        int startCurr = get_diag_start_device(final_d, M, N);

        int off_curr = max(0, final_d - M);

        for (int j = j_start + threadIdx.x; j <= j_end; j += blockDim.x) {
            if (dp[startCurr + j - off_curr] == local_max) {
                atomicMin(&shared_min_j, j);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0)
            local_max_j = shared_min_j;
    }

    if (threadIdx.x == 0) {
        node->max_score = local_max;
        node->max_score_d = local_max_d;
        node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
        node->max_score_j = local_max_j;
    }

}
