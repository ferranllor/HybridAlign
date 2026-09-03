#include "../include/cuda_warps.cuh"

#include "../include/hybrid_utils.cuh"

// *************************************************************************************************
//
//                                            Utils
//
// *************************************************************************************************

#define WARP_SIZE 32 // I mean, it's pretty constant, but magic numbers look bad, so i left it as a global def

struct DpBuffersWarps {
    DTYPEMATRIX* base;
    int stride;

    __device__ DTYPEMATRIX* operator[](int b) const { return base + b * stride; }
};

// Shared memory of one warp: topRow (only when the band runs along the rows, i.e. M < N), prevCol, the N_BUFFERS 
// rotating diagonals, then the two sequences. Exported so that the hybrid version launches this kernel with 
// exactly the size it expects instead of keeping its own copy.

int warps_toprow_slots(int max_node_size, int M) {
    return (max_node_size > M) ? (max_node_size + 1) : 0;
}

int warps_elems_per_warp(int max_node_size, int M, int band_width) {
    return warps_toprow_slots(max_node_size, M) + (M + 1) + N_BUFFERS * (band_width + 2)
           + seq_slots_for_level(max_node_size, M);
}

// *************************************************************************************************
//
//                                           Scheduler
//
// *************************************************************************************************

// I mean it's the same thing, but now I have WARPS_PER_BLOCK warps per block, so I have to change how I launch the kernels a bit. I also added some extra args
// to the kernel call. Honestly, it was laziness, but if you need those bytes, you can calculate it on inside the kernel intead. 
// Look at the kernel comment though, way more explanatory


AlignmentResult gpu_align_warps(Graph graph, Graph cudaGraph, Sequence sequence)
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

    const int BATCH_LEVELS = 8;
    int batch_start_offset = 0;
    Node* batch_start_act  = act;
    int levels_in_batch = 0;

    for (int d = 0; d < num_levels; d++)
    {
        int max_node_size = 0;
        for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
            if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

        int band_width = band_width_for_level(max_node_size, sequence.size);
        int toprow_slots = warps_toprow_slots(max_node_size, sequence.size);
        int elems_per_warp = warps_elems_per_warp(max_node_size, sequence.size, band_width);

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((nodes_per_level[d] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        int dynamic_shared_mem_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("cuda_warps", d, dynamic_shared_mem_bytes);

        compute_dp_gpu_warps<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
            act, nodes_per_level[d], sequence, sequence_rev, elems_per_warp, toprow_slots, band_width);

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

    return compute_traceback_gpu_warps(graph, sequence);
}


// *************************************************************************************************
//
//                                            Kernel
//
// *************************************************************************************************

// You know how I said doing a good shared mem version would make this 100x easier, well, I was wrong. It was 1000x easier, worked on the 3rd try, as hard as it is to believe.
//
// In any case, this is the first implementation that uses thread warps instead of thread blocks. There were
// two places I could put warps: 1. Multiple warps per node, 2. Give multiple nodes to a thread block, and each to a warp
// I was thinking of going for 1, but my life is complicated enough without thinking about the synchronisation that would involve.
// Then I looked at the dataset. A typical H100 has 132 SMs, at 64 max resident warps per SM, that makes 8448 Warps needing nodes to work on.
// A real sequence to graph aligner, does not do 1 alignment, but 10s of 1000s (150_10 has 18600 on the dense node). 
// We have enough work, instead we need less synchronisation. So I went for the 1 node per warp approach. 
//
// You may already see coming a first problem, each node is different in size, and so, what if a warp finishes earlier? Does it wait
// and do nothing until the slowest one finishes? Well, yes, and no. Remember the topological sort? Well, as a second key, I used the length
// of the sequence of the node, so, since warps have roughly the same amount of work, they should take the same amount of time (roughly).
// And now I see you saying: But, what if one warp gets the columns it needs earlier? Or what if the number of warps is more than the GPU can do in parallel?
// And to that I say, I know, I just don't know how to fix it :'). For now, just use a number of warps multiple of 4, since GPUs tent to have 4 warp schedulers
// with its associated resources in there, also, using many more than 4 would net be good for now, it's not like I have that much work on the dataset.
// 
// Now, onto how it works. This is pretty much the shared mem version, but using warps and syncwarp instead of syncthreads (for the most part) and thread 0 is now lane 0.
// What actually changed a bit is the last bit of the reduction, given that we have warps, now it's a shuffle reduction, courtesy of claude.
//
// Last note, I forgot to mention this is slower for GPU only. This is because before we had 160 threads working on a node, and now we have 32, even if individually faster.
// On the long tail of 1-4 ndoes per level of my datasets, that means the last part is slower than a pentium would be if it did it. (not really, but you get the point)
// In any case, hybrid versions rock for this, specially since I realised that I was compiling them without optimisations flags, and now they are FAST by comparison.

__global__ void compute_dp_gpu_warps(Node* nodes, int level_nodes, Sequence sequence,
                                     Sequence sequence_rev, int elems_per_warp, int toprow_slots,
                                     int band_width)
{
    const unsigned int mask = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warpIdx = threadIdx.x / WARP_SIZE;

    int nodeIdx = blockIdx.x * (blockDim.x / WARP_SIZE) + warpIdx;
    if (nodeIdx >= level_nodes) return;

    Node* node = &nodes[nodeIdx];

    // ------------------------------------------------- Initialize -------------------------------------------------

    int M = sequence.size;
    int N = node->sequence.size;

    DTYPEMATRIX* __restrict last_col = node->last_col;

    const DTYPEALPHABET* __restrict node_seq  = node->sequence.sequence;
    const DTYPEALPHABET* __restrict query_rev = sequence_rev.sequence;

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* mine = &shared[warpIdx * elems_per_warp];

    DTYPEMATRIX* topRow = mine;
    DTYPEMATRIX* prevCol = &mine[toprow_slots];

    DpBuffersWarps dpBuffers = { &prevCol[M + 1], band_width + 2 };

    DTYPEALPHABET* node_shared = (DTYPEALPHABET*)dpBuffers[N_BUFFERS];
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
        const DTYPEMATRIX* __restrict prev_last = node->v_in[0]->last_col;

        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = prev_last[i];
    }
    else {
        const DTYPEMATRIX* __restrict prev_last  = node->v_in[0]->last_col;
        const DTYPEMATRIX* __restrict prev_last2 = node->v_in[1]->last_col;

        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = max(prev_last[i], prev_last2[i]);

        for (int p = 2; p < node->num_in; ++p) {
            const DTYPEMATRIX* __restrict prev_lastp = node->v_in[p]->last_col;
            for (int i = lane; i <= M; i += WARP_SIZE)
                prevCol[i] = max(prev_lastp[i], prevCol[i]);
        }
    }

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    int bufferAct, bufferPrev, bufferPrevPrev;

    auto process_diagonal = [&](int klo, int khi, int current_d,
                            int node_off, int query_off,
                            int off_diag, int off_up, int off_left) {

        for (int k = klo + lane; k <= khi; k += WARP_SIZE) {
            int score = (node_shared[node_off + k] == query_shared[query_off + k]) ? MATCH : MISMATCH;

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

        __syncwarp();

        bufferPrevPrev = bufferPrev;
        bufferPrev     = bufferAct;
        bufferAct      = (bufferAct + 1 == N_BUFFERS) ? 0 : bufferAct + 1;
    };

    __syncwarp();

    if (M >= N) {
        for (int startN = 0; startN < N; startN += band_width) {
            int stripe_height = min(band_width, N - startN);

            int js = startN + 1;
            int je = startN + stripe_height;

            __syncwarp();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            if (lane == 0) {
                dpBuffers[bufferPrevPrev][0] = 0;
                dpBuffers[bufferPrev][0]     = prevCol[1];
                dpBuffers[bufferPrev][1]     = 0;
            }

            __syncwarp();

            int d = js + 1;

            // --------------- Grow phase ------------------

            for (; d <= je; ++d) {
                int khi = d - 1 - startN;

                if (lane == 0) {
                    dpBuffers[bufferAct][0]       = prevCol[d - startN];
                    dpBuffers[bufferAct][khi + 1] = 0;
                }

                process_diagonal(1, khi, d, startN - 1, M - d + startN, -1, 0, -1);
            }

            // --------------- Stable phase ------------------

            for (; d <= M; ++d) {
                if (lane == 0) dpBuffers[bufferAct][0] = prevCol[d - startN];

                process_diagonal(1, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                if (lane == 0) prevCol[d - je] = dpBuffers[bufferPrev][stripe_height];
            }

            // --------------- Shrink phase -----------------

            for (; d <= je + M; ++d) {
                int j_min = d - M;
                int klo = max(1, j_min - startN);

                if (lane == 0 && j_min <= startN) dpBuffers[bufferAct][0] = prevCol[d - startN];

                process_diagonal(klo, stripe_height, d, startN - 1, M - d + startN, -1, 0, -1);

                if (lane == 0) prevCol[d - je] = dpBuffers[bufferPrev][stripe_height];
            }
        }
    }
    else {
        for (int startM = 0; startM < M; startM += band_width) {
            int stripe_height = min(band_width, M - startM);

            int is = startM + 1;
            int ie = startM + stripe_height;

            __syncwarp();

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            if (lane == 0) {
                dpBuffers[bufferPrevPrev][stripe_height + 1] = prevCol[startM];
                dpBuffers[bufferPrev][stripe_height + 1]     = topRow[1];
                dpBuffers[bufferPrev][stripe_height]         = prevCol[is];
            }

            __syncwarp();

            int d = is + 1;

            // --------------- Grow phase ------------------

            for (; d <= ie; ++d) {
                int klo = ie + 2 - d;

                if (lane == 0) {
                    dpBuffers[bufferAct][klo - 1]           = prevCol[d];
                    dpBuffers[bufferAct][stripe_height + 1] = topRow[d - startM];
                }

                process_diagonal(klo, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);
            }

            // --------------- Stable phase ------------------

            for (; d <= is + N; ++d) {
                if (lane == 0 && d - startM <= N)
                    dpBuffers[bufferAct][stripe_height + 1] = topRow[d - startM];

                process_diagonal(1, stripe_height, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                if (lane == 0) topRow[d - ie] = dpBuffers[bufferPrev][1];
            }

            // --------------- Shrink phase -----------------

            for (; d <= ie + N; ++d) {
                int khi = ie + 1 - d + N;

                process_diagonal(1, khi, d, d - ie - 2, M - ie - 1, 1, 1, 0);

                if (lane == 0) topRow[d - ie] = dpBuffers[bufferPrev][1];
            }
        }
    }

    __syncwarp();

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

    if (lane == 0) {
        node->max_score = local_max;
        node->max_score_d = local_max_d;
        node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
        node->max_score_j = (local_max_d != -1) ? local_max_j : -1;
    }
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Nothing to see here, it's recycled from the last_col version, again thanks to claude for the recompute function.

static void recompute_node_dp_warps(Node* node, Sequence sequence, DTYPEMATRIX* dp)
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

AlignmentResult compute_traceback_gpu_warps(Graph graph, Sequence sequence) {
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
            recompute_node_dp_warps(curr_node, sequence, dp);
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
