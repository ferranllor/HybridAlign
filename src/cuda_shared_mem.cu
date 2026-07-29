#include "../include/cuda_shared_mem.cuh"
#include <cuda/barrier>

AlignmentResult gpu_align_shared_mem(Graph graph, Graph cudaGraph, Sequence sequence)
{
    Sequence sequence_rev;
    Sequence tmp;

    tmp.sequence = (char*)malloc(sequence.size * sizeof(char));
    for (int idx = 0; idx < sequence.size; ++idx) {
        tmp.sequence[idx] = sequence.sequence[sequence.size - 1 - idx];
    }

    sequence_rev.size = sequence.size;

    cudaMalloc((void**)&sequence_rev.sequence, sequence.size * sizeof(char));
    cudaMemcpy(sequence_rev.sequence, tmp.sequence, sequence.size * sizeof(char), cudaMemcpyHostToDevice);

    int num_levels = graph.nodes[graph.num_nodes-1].depth + 1;
    int* nodes_per_level = (int*)calloc(num_levels, sizeof(int));

    for (int n = 0; n < graph.num_nodes; n++) {
        int depth = graph.nodes[n].depth;
        nodes_per_level[depth]++;
    }

    cudaStream_t compute_stream, copy_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copy_stream);

    Node* device_nodes_pointers = (Node*)malloc(graph.num_nodes * sizeof(Node));
    cudaMemcpy(device_nodes_pointers, cudaGraph.nodes, graph.num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    Node* device_nodes_tmp = NULL;
    cudaMallocHost((void**)&device_nodes_tmp, graph.num_nodes * sizeof(Node));

    cudaError_t status;
    Node* act = cudaGraph.nodes;
    int node_offset = 0;

    cudaEvent_t compute_done;
    cudaEventCreate(&compute_done);

    const int BATCH_LEVELS = 8; // smaller = more overlap, larger = fewer commands
    int batch_start_offset = 0;   // node_offset at start of current batch
    Node* batch_start_act  = act; // device pointer at start of current batch
    int levels_in_batch = 0;

    for (int d = 0; d < num_levels; d++)
    {
        dim3 gridDim(nodes_per_level[d]);
        dim3 blockDim(BLOCKSIZE);

        // The kernel lays out topRow (N + 1), prevCol (M + 1), node_seq (N) and query_seq_rev (M)
        // in dynamic shared memory, so it has to be sized after the largest node of *this* level
        // (dpBuffers, the barriers and the reduction arrays are static shared memory).
        int max_node_size = 0;
        for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
            if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

        int dynamic_shared_mem_bytes = sizeof(DTYPEMATRIX) * (max_node_size + sequence.size + 2);
        dynamic_shared_mem_bytes += sizeof(DTYPEALPHABET) * (sequence.size + max_node_size);

        compute_dp_gpu_shared_mem<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

        cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
        }

        act = &act[nodes_per_level[d]];
        node_offset += nodes_per_level[d];
        levels_in_batch++;

        bool last_level = (d == num_levels - 1);
        bool batch_full = (levels_in_batch >= BATCH_LEVELS);

        if (batch_full || last_level) {
            // one sync point for the whole batch, not one per level
            cudaEventRecord(compute_done, compute_stream);
            cudaStreamWaitEvent(copy_stream, compute_done, 0);

            int batch_num_nodes = node_offset - batch_start_offset;
            size_t batch_nodes_size = (size_t)batch_num_nodes * sizeof(Node);

            cudaMemcpyAsync(&device_nodes_tmp[batch_start_offset],
                            batch_start_act,
                            batch_nodes_size,
                            cudaMemcpyDeviceToHost,
                            copy_stream);

            size_t batch_matrix_elements = 0;
            for (int n = batch_start_offset; n < node_offset; n++) {
                batch_matrix_elements += (size_t)(sequence.size + 2) * (graph.nodes[n].sequence.size + 2);
            }

            if (batch_matrix_elements > 0) {
                cudaMemcpyAsync(graph.nodes[batch_start_offset].dp_matrix,
                                device_nodes_pointers[batch_start_offset].dp_matrix,
                                batch_matrix_elements * sizeof(DTYPEMATRIX),
                                cudaMemcpyDeviceToHost,
                                copy_stream);
            }

            // reset batch trackers
            batch_start_offset = node_offset;
            batch_start_act = act;
            levels_in_batch = 0;
        }
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
    }

    graph.max_score = device_nodes_tmp[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        graph.nodes[n].max_score   = device_nodes_tmp[n].max_score;
        graph.nodes[n].max_score_d = device_nodes_tmp[n].max_score_d;
        graph.nodes[n].max_score_i = device_nodes_tmp[n].max_score_i;
        graph.nodes[n].max_score_j = device_nodes_tmp[n].max_score_j;

        if (device_nodes_tmp[n].max_score > graph.max_score) {
            graph.max_score = device_nodes_tmp[n].max_score;
            graph.max_score_node_id = n;
        }
    }

    cudaEventDestroy(compute_done);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    free(nodes_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);
    cudaFreeHost(device_nodes_tmp);
    free(device_nodes_pointers);

    return compute_traceback_gpu_shared_mem(graph, sequence);
}

__global__ void compute_dp_gpu_shared_mem(Node* node, Sequence sequence, Sequence sequence_rev)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    node = &node[blockIdx.x];
    
    int M = sequence.size;
    int N = node->sequence.size;
    DTYPEMATRIX* __restrict dp = node->dp_matrix;

    __shared__ DTYPEMATRIX dpBuffers[N_BUFFERS][BLOCKSIZE+2];
    __shared__ cuda::barrier<cuda::thread_scope_block> write_barriers[N_BUFFERS];

    if (threadIdx.x == 0) {
    for (int b = 0; b < N_BUFFERS; ++b) {
        init(&write_barriers[b], blockDim.x); 
    }
}

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* topRow = shared;
    DTYPEMATRIX* prevCol = &(shared[N+1]);

    //if (threadIdx.x == 0) printf("Holis\n");

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

    DTYPEALPHABET* node_seq = (DTYPEALPHABET*)&prevCol[M + 1];
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

    // The matrix is walked band by band: BLOCKSIZE consecutive columns at a time when M >= N,
    // BLOCKSIZE consecutive rows at a time when M < N (i.e. always along the shorter side, so a
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


    auto process_diagonal_tma = [&](int klo, int khi, int current_d,
                            int node_off, int query_off,
                            int off_diag, int off_up, int off_left) {

        write_barriers[bufferAct].wait(write_barriers[bufferAct].arrive());
        int prev_max = local_max;

        for (int k = klo + threadIdx.x; k <= khi; k += blockDim.x) {
            int score = (node_seq[node_off + k] == query_seq_rev[query_off + k]) ? MATCH : MISMATCH;

            int diagonal = dpBuffers[bufferPrevPrev][k + off_diag] + score;
            int up       = dpBuffers[bufferPrev][k + off_up] + GAP;
            int left     = dpBuffers[bufferPrev][k + off_left] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dpBuffers[bufferAct][k] = res;

            local_max = max(res, local_max);
        }

        if (prev_max != local_max) {
            local_max_d = current_d;
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            size_t copy_bytes = (khi - klo + 1) * sizeof(DTYPEMATRIX);

            cuda::memcpy_async(
                &dp[write_base + klo],
                &dpBuffers[bufferAct][klo],
                copy_bytes,
                write_barriers[bufferAct]
            );
        }

        bufferAct      = (bufferAct + 1) % N_BUFFERS;
        bufferPrev     = (bufferPrev + 1) % N_BUFFERS;
        bufferPrevPrev = (bufferPrevPrev + 1) % N_BUFFERS;
    };

    __syncthreads();

    if (M >= N) {
        // Bands of columns [js, je]. prevCol carries column startN over from the previous band.
        for (int startN = 0; startN < N; startN += BLOCKSIZE) {
            int stripe_height = min(BLOCKSIZE, N - startN);

            int js = startN + 1;                // first column owned by the band
            int je = startN + stripe_height;    // last column owned by the band

            __syncthreads();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            // Seed the two diagonals that precede the band: (0, startN) and (0, startN + 1) sit on
            // the top row, (1, startN) is the first cell of the carry column.
            if (threadIdx.x == 0) {
                dpBuffers[bufferPrevPrev][0] = topRow[startN];
                dpBuffers[bufferPrev][0]     = prevCol[1];
                dpBuffers[bufferPrev][1]     = topRow[startN + 1];
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
                    dpBuffers[bufferAct][khi + 1] = topRow[d];            // top row of the matrix
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
        for (int startM = 0; startM < M; startM += BLOCKSIZE) {
            int stripe_height = min(BLOCKSIZE, M - startM);

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
    __shared__ DTYPEMATRIX local_max_red[BLOCKSIZE];
    __shared__ DTYPEMATRIX local_max_d_red[BLOCKSIZE];

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

AlignmentResult compute_traceback_gpu_shared_mem(Graph graph, Sequence sequence) {
    Node* curr_node = &graph.nodes[graph.max_score_node_id];
    int i = curr_node->max_score_i; 
    int j = curr_node->max_score_j; 
    int M = sequence.size;

    int max_graph_seq_len = 0;
    for (int k = 0; k < graph.num_nodes; k++) {
        max_graph_seq_len += graph.nodes[k].sequence.size;
    }
    char* align_graph = (char*)malloc(M + max_graph_seq_len + 1);
    char* align_query = (char*)malloc(M + max_graph_seq_len + 1);
    int pos = 0; 

    while (curr_node != NULL) {
        int N = curr_node->sequence.size;
        int curr_score = curr_node->dp_matrix[get_diagonal_index(i, j, M, N)];
        if (curr_score <= 0) break;

        if (i > 0 && j > 0) {
            int score = (curr_node->sequence.sequence[j-1] == sequence.sequence[i-1]) ? MATCH : MISMATCH;
            
            int diag_score = curr_node->dp_matrix[get_diagonal_index(i - 1, j - 1, M, N)];
            int up_score   = curr_node->dp_matrix[get_diagonal_index(i - 1, j, M, N)];
            int left_score = curr_node->dp_matrix[get_diagonal_index(i, j - 1, M, N)];

            if (curr_score == diag_score + score) {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = sequence.sequence[i-1];
                i--; j--;
            } else if (curr_score == up_score + GAP) {
                align_graph[pos] = '-';
                align_query[pos] = sequence.sequence[i-1];
                i--;
            } else {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = '-';
                j--;
            }
            pos++;
            
        } else if (j == 0) { 
            if (curr_node->num_in > 0) {
                Node* best_prev = NULL;
                for (int p = 0; p < curr_node->num_in; p++) {
                    Node* prev = curr_node->v_in[p];
                    int prev_N = prev->sequence.size;
                    if (curr_score == prev->dp_matrix[get_diagonal_index(i, prev_N, M, prev_N)]) {
                        best_prev = prev;
                        break;
                    }
                }
                curr_node = best_prev;
                if (curr_node) j = curr_node->sequence.size;
            } else {
                while (i > 0) {
                    align_graph[pos] = '-';
                    align_query[pos] = sequence.sequence[i-1];
                    i--; pos++;
                }
                curr_node = NULL;
            }
        } else if (i == 0) { 
            while (j > 0) {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = '-';
                j--; pos++;
            }
            curr_node = NULL;
        }
    }
    
    align_graph[pos] = '\0'; 
    align_query[pos] = '\0';
    reverse_string(align_graph, pos);
    reverse_string(align_query, pos);

    AlignmentResult res;
    res.graph_align = align_graph;
    res.query_align = align_query;
    res.size = pos;

    return res;
}