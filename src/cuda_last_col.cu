#include "../include/cuda_last_col.cuh"

// *************************************************************************************************
//
//                                           Utils
//
// *************************************************************************************************

struct DpBuffersLastCol {
    DTYPEMATRIX* base;
    int stride;

    __device__ DTYPEMATRIX* operator[](int b) const { return base + b * stride; }
};

// Shared memory of one block: topRow (only when the band runs along the rows, i.e. M < N), prevCol,
// the N_BUFFERS rotating diagonals, then the two sequences.

static inline int shared_bytes_for_level_last_col(int max_node_size, int M, int band_width) {
    int matrix_slots = (M + 1) + N_BUFFERS * (band_width + 2);
    if (max_node_size > M) matrix_slots += max_node_size + 1;

    return (int)(matrix_slots * sizeof(DTYPEMATRIX) + (M + max_node_size) * sizeof(DTYPEALPHABET));
}

// *************************************************************************************************
//
//                                           Scheduler
//
// *************************************************************************************************

AlignmentResult gpu_align_last_col(Graph graph, Graph cudaGraph, Sequence sequence)
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

    cudaStream_t compute_stream, copy_stream; // It might not be much, but even though I don't need it anymore, it might still be better to have it than not, though
    cudaStreamCreate(&compute_stream);        // I have not tested this hypothesis. If you wanted to, most likely this would help on more discrete systems the most
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

    const int BATCH_LEVELS = 8;
    int batch_start_offset = 0;
    Node* batch_start_act  = act;
    int levels_in_batch = 0;

    for (int d = 0; d < num_levels; d++)
    {
        int max_node_size = 0;
        for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
            if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

        dim3 gridDim(nodes_per_level[d]);
        dim3 blockDim(band_width_for_level(max_node_size, sequence.size));

        int dynamic_shared_mem_bytes = shared_bytes_for_level_last_col(max_node_size, sequence.size, blockDim.x);

        compute_dp_gpu_last_col<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

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
            cudaEventRecord(compute_done, compute_stream);
            cudaStreamWaitEvent(copy_stream, compute_done, 0);

            int batch_num_nodes = node_offset - batch_start_offset;

            cudaMemcpyAsync(&device_nodes_tmp[batch_start_offset],
                            batch_start_act,
                            (size_t)batch_num_nodes * sizeof(Node),
                            cudaMemcpyDeviceToHost,
                            copy_stream);

            cudaMemcpyAsync(graph.nodes[batch_start_offset].last_col,
                            device_nodes_pointers[batch_start_offset].last_col,
                            (size_t)batch_num_nodes * (sequence.size + 1) * sizeof(DTYPEMATRIX),
                            cudaMemcpyDeviceToHost,
                            copy_stream);

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

    return compute_traceback_gpu_last_col(graph, sequence);
}

// *************************************************************************************************
//
//                                            Kernel
//
// *************************************************************************************************

// So I had a good idea. I remembered something that I overheard a while back, it basically was about recomputing
// the dp matrix during the traceback. I though it didn't make any sense, you need the dp matrices of each node/vertex's
// predecessor to be able to compute its dp matrix, and then I realised, I don't need the entire thing, only the last column!
//
// Basically, I realised every DP matrix is defined entirely by the elements of the last column from every predesecor node, so, this means
// that if we only store those, we can recompute the DP matriceas later without storing much data.
//
// The second interesting thing, is that, during the traceback phase, to decide to which predecesor node we will go from from the current
// node, we only need the elements of the last column!
//
//
//  +------------+
//  |          A1|
//  |          A2|  <---+
//  |          A3|  <---+
//  |          A4|      |       +------------+
//  +------------+      |       | C1         |
//                      |       | C2         |
//                      +------ | C3         | (optimal alignment goes through cell C3, and it looks at the cell on top C2, and cells max(A2, B2) & max(A3, B3))
//  +------------+      |       | C4         |
//  |          B1|      |       +------------+
//  |          B2|  <---+
//  |          B3|  <---+
//  |          B4|
//  +------------+
//
//
// With this, since the optimal alignment normally goes through a small percentage of the total nodes, we can just save those and recompute the DP matrices 
// where the alignment happens on the way back in the CPU. Since the percentage is small, even if the CPU can deliver less throughput, this barely accounts for <1% of exec time.
//
// However, worth noting this is not so true on real pangenome graphs, where the branching factor is small. It's why I made optimised kernels that
// move the DP matrix arround, and why systems like the DGX Spark is useful in the field, since the GPU can compute those much faster.
// This does not mean it's useless however, if, for example, you wanted to align a short sequence to a massive graph, you will compute a DP matrix for every node,
// but you know 99% of them will not be looked at. As with most things in life, wether this helps depends on the situation (or in this case, the dataset)
//
// So, big version here, I went from using 7,8GB of VRAM to align one sequence to a graph, and now this version only uses 30 MB
// This is great, it completly makes this compute bound, but, it kind of makes all the previous work obsolete, given I don't have
// to move data at all now :'). In any case, it's basically the same thing as the shared memory kernel that claude helped me
// fix, and this time it ONLY uses shared memory (aside from the reads of the previous cols of previous nodes at the start,a nd storing those last columns, of course).

__global__ void compute_dp_gpu_last_col(Node* node, Sequence sequence, Sequence sequence_rev)
{
    node = &node[blockIdx.x];

    // ------------------------------------------------- Initialize -------------------------------------------------

    int M = sequence.size;
    int N = node->sequence.size;

    DTYPEMATRIX* __restrict last_col = node->last_col;

    int band_width = blockDim.x;

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* topRow = shared;
    DTYPEMATRIX* prevCol = (M < N) ? &(shared[N+1]) : shared;

    DpBuffersLastCol dpBuffers = { &prevCol[M + 1], band_width + 2 };

    if (M < N)
        for (int j = threadIdx.x; j <= N; j += blockDim.x) topRow[j] = 0;

    if (node->num_in == 0) {
        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            prevCol[i] = 0;
    }
    else if (node->num_in == 1) {
        const DTYPEMATRIX* __restrict prev_last = node->v_in[0]->last_col;

        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            prevCol[i] = prev_last[i];
    }
    else {
        const DTYPEMATRIX* __restrict prev_last  = node->v_in[0]->last_col;
        const DTYPEMATRIX* __restrict prev_last2 = node->v_in[1]->last_col;

        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            prevCol[i] = max(prev_last[i], prev_last2[i]);

        for (int p = 2; p < node->num_in; ++p) {
            const DTYPEMATRIX* __restrict prev_lastp = node->v_in[p]->last_col;
            for (int i = threadIdx.x; i <= M; i += blockDim.x)
                prevCol[i] = max(prev_lastp[i], prevCol[i]);
        }
    }

    DTYPEALPHABET* node_seq = (DTYPEALPHABET*)dpBuffers[N_BUFFERS];
    DTYPEALPHABET* query_seq_rev = &node_seq[N];

    for (int i = threadIdx.x; i < node->sequence.size; i += blockDim.x)
        node_seq[i] = node->sequence.sequence[i];

    for (int i = threadIdx.x; i < sequence_rev.size; i += blockDim.x)
        query_seq_rev[i] = sequence_rev.sequence[i];

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    int bufferAct, bufferPrev, bufferPrevPrev;

    auto process_diagonal = [&](int klo, int khi, int current_d,
                            int node_off, int query_off,
                            int off_diag, int off_up, int off_left) {

        for (int k = klo + threadIdx.x; k <= khi; k += blockDim.x) {
            int score = (node_seq[node_off + k] == query_seq_rev[query_off + k]) ? MATCH : MISMATCH;

            int diagonal = dpBuffers[bufferPrevPrev][k + off_diag] + score;
            int up       = dpBuffers[bufferPrev][k + off_up] + GAP;
            int left     = dpBuffers[bufferPrev][k + off_left] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dpBuffers[bufferAct][k] = res;

            int j = node_off + k + 1;
            if (j == N) last_col[current_d - N] = res;

            if (res > local_max) {
                local_max = res;
                local_max_d = current_d;
                local_max_j = j;
            }
        }

        __syncthreads();

        bufferPrevPrev = bufferPrev;
        bufferPrev     = bufferAct;
        bufferAct      = (bufferAct + 1 == N_BUFFERS) ? 0 : bufferAct + 1;
    };

    __syncthreads();

    if (M >= N) {
        for (int startN = 0; startN < N; startN += band_width) {
            int stripe_height = min(band_width, N - startN);

            int js = startN + 1;
            int je = startN + stripe_height;

            __syncthreads();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            if (threadIdx.x == 0) {
                dpBuffers[bufferPrevPrev][0] = 0;
                dpBuffers[bufferPrev][0]     = prevCol[1];
                dpBuffers[bufferPrev][1]     = 0;
            }

            __syncthreads();

            int d = js + 1;

            // --------------- Grow phase ------------------

            for (; d <= je; ++d) {
                int khi = d - 1 - startN;

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][0]       = prevCol[d - startN];
                    dpBuffers[bufferAct][khi + 1] = 0;
                }

                process_diagonal(1, khi, d, startN - 1, M - d + startN, -1, 0, -1);
            }

            // --------------- Stable phase ------------------

            for (; d <= M; ++d) {
                if (threadIdx.x == 0) dpBuffers[bufferAct][0] = prevCol[d - startN];

                process_diagonal(1, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                if (threadIdx.x == 0) prevCol[d - je] = dpBuffers[bufferPrev][stripe_height];
            }

            // --------------- Shrink phase -----------------

            for (; d <= je + M; ++d) {
                int j_min = d - M;
                int klo = max(1, j_min - startN);

                if (threadIdx.x == 0 && j_min <= startN) dpBuffers[bufferAct][0] = prevCol[d - startN];

                process_diagonal(klo, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                if (threadIdx.x == 0) prevCol[d - je] = dpBuffers[bufferPrev][stripe_height];
            }
        }
    }
    else {
        for (int startM = 0; startM < M; startM += band_width) {
            int stripe_height = min(band_width, M - startM);

            int is = startM + 1;
            int ie = startM + stripe_height;

            __syncthreads();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            if (threadIdx.x == 0) {
                dpBuffers[bufferPrevPrev][stripe_height + 1] = prevCol[startM];
                dpBuffers[bufferPrev][stripe_height + 1]     = topRow[1];
                dpBuffers[bufferPrev][stripe_height]         = prevCol[is];
            }

            __syncthreads();

            int d = is + 1;

            // --------------- Grow phase ------------------

            for (; d <= ie; ++d) {
                int klo = ie + 2 - d;

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][klo - 1]           = prevCol[d];
                    dpBuffers[bufferAct][stripe_height + 1] = topRow[d - startM];
                }

                process_diagonal(klo, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);
            }

            // --------------- Stable phase ------------------

            for (; d <= is + N; ++d) {
                if (threadIdx.x == 0 && d - startM <= N)
                    dpBuffers[bufferAct][stripe_height + 1] = topRow[d - startM];

                process_diagonal(1, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                if (threadIdx.x == 0) topRow[d - ie] = dpBuffers[bufferPrev][1];
            }

            // --------------- Shrink phase -----------------

            for (; d <= ie + N; ++d) {
                int khi = ie + 1 - d + N;

                process_diagonal(1, khi, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                if (threadIdx.x == 0) topRow[d - ie] = dpBuffers[bufferPrev][1];
            }
        }
    }

    __syncthreads();

    // ------------------ Reduce the local max ------------------

    int t = threadIdx.x;

    DTYPEMATRIX* red_max = dpBuffers[0];
    DTYPEMATRIX* red_d   = dpBuffers[1];
    DTYPEMATRIX* red_j   = dpBuffers[2];

    red_max[t] = local_max;
    red_d[t]   = local_max_d;
    red_j[t]   = local_max_j;

    __syncthreads();

    for (unsigned int live = blockDim.x; live > 1; ) {
        unsigned int stride = (live + 1) / 2;

        if (t + stride < live) {
            int curr_max = red_max[t], cand_max = red_max[t + stride];
            int curr_d   = red_d[t],   cand_d   = red_d[t + stride];
            int curr_j   = red_j[t],   cand_j   = red_j[t + stride];

            bool is_greater = (cand_max > curr_max) ||
                              (cand_max == curr_max && cand_d != -1 &&
                                  (curr_d == -1 || cand_d < curr_d ||
                                      (cand_d == curr_d && cand_j < curr_j)));

            if (is_greater) { red_max[t] = cand_max; red_d[t] = cand_d; red_j[t] = cand_j; }
        }
        __syncthreads();
        live = stride;
    }

    if (threadIdx.x == 0) {
        int m = red_max[0], md = red_d[0], mj = red_j[0];

        node->max_score = m;
        node->max_score_d = md;
        node->max_score_i = (md != -1) ? (md - mj) : -1;
        node->max_score_j = (md != -1) ? mj : -1;
    }
}



// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Fills a plain row major (M + 1) x (N + 1) matrix for one node, the same recurrence the kernel
// runs. Column 0 is the best of the predecessors' last columns and row 0 is zero, which is exactly
// how the kernel seeds prevCol, so the matrix this produces is the one the kernel would have
// stored. Called only for the nodes the alignment actually crosses.

// Note here, claude did this, I see it's not exactly optimised, but meh, the time it takes is so low it does not need to be

static void recompute_node_dp(Node* node, Sequence sequence, DTYPEMATRIX* dp)
{
    int M = sequence.size;
    int N = node->sequence.size;
    int stride = N + 1;

    for (int j = 0; j <= N; j++) dp[j] = 0;

    for (int i = 1; i <= M; i++) {
        int boundary = 0;
        for (int p = 0; p < node->num_in; p++)
            if (node->v_in[p]->last_col[i] > boundary) boundary = node->v_in[p]->last_col[i];

        dp[i * stride] = boundary;
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            int score = (node->sequence.sequence[j-1] == sequence.sequence[i-1]) ? MATCH : MISMATCH;

            int diagonal = dp[(i-1) * stride + (j-1)] + score;
            int up       = dp[(i-1) * stride + j] + GAP;
            int left     = dp[i * stride + (j-1)] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[i * stride + j] = res;
        }
    }
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Same code as before, the only thing that now, I call the traceback function whenever I decide what node comes next.

AlignmentResult compute_traceback_gpu_last_col(Graph graph, Sequence sequence) {
    Node* curr_node = &graph.nodes[graph.max_score_node_id];
    int i = curr_node->max_score_i;
    int j = curr_node->max_score_j;
    int M = sequence.size;

    int max_graph_seq_len = 0;
    int max_node_size = 0;
    for (int k = 0; k < graph.num_nodes; k++) {
        max_graph_seq_len += graph.nodes[k].sequence.size;
        if (graph.nodes[k].sequence.size > max_node_size) max_node_size = graph.nodes[k].sequence.size;
    }

    char* align_graph = (char*)malloc(M + max_graph_seq_len + 1);
    char* align_query = (char*)malloc(M + max_graph_seq_len + 1);
    int pos = 0;

    DTYPEMATRIX* dp = (DTYPEMATRIX*)malloc((size_t)(M + 1) * (max_node_size + 1) * sizeof(DTYPEMATRIX));
    Node* loaded = NULL;

    while (curr_node != NULL) {
        int N = curr_node->sequence.size;
        int stride = N + 1;

        if (curr_node != loaded) {
            recompute_node_dp(curr_node, sequence, dp);
            loaded = curr_node;
        }

        int curr_score = dp[i * stride + j];
        if (curr_score <= 0) break;

        if (i > 0 && j > 0) {
            int score = (curr_node->sequence.sequence[j-1] == sequence.sequence[i-1]) ? MATCH : MISMATCH;

            int diag_score = dp[(i-1) * stride + (j-1)];
            int up_score   = dp[(i-1) * stride + j];

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
        }
        else if (j == 0) {
            if (curr_node->num_in > 0) {
                Node* best_prev = NULL;
                for (int p = 0; p < curr_node->num_in; p++) {
                    Node* prev = curr_node->v_in[p];
                    if (prev->last_col[i] == curr_score) {
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
        }
        else {
            while (j > 0) {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = '-';
                j--; pos++;
            }
            curr_node = NULL;
        }
    }

    free(dp);

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
