#include "../include/cuda_registers.cuh"

#include "../include/hybrid_utils.cuh"

#define WARP_SIZE 32
#define REG_BAND (WARP_SIZE - 2)


// Shared memory of ONE warp: topRow (only when the band runs along the rows, i.e. M < N) and
// prevCol. The rotating diagonals are gone from here, that is the point of the version.
int registers_toprow_slots(int max_node_size, int M) {
    return (max_node_size > M) ? (max_node_size + 1) : 0;
}

int registers_elems_per_warp(int max_node_size, int M) {
    return registers_toprow_slots(max_node_size, M) + (M + 1) + seq_slots_for_level(max_node_size, M);
}

// *************************************************************************************************
//
//                                            Kernel
//
// *************************************************************************************************


// Okay, quick drawing here:
//
// x--------------------------------------------x
// |O0 A1 B2 C3                                 |
// |A0 B1 C2                                    |
// |B0 C1                                       |
// |C0                                          |
// |                                            |
// |                                            |
// |                                            |
// |                                            |
// x--------------------------------------------x
//
// Basically, the thing here is cell C2 needs cell B1, B2, and A1 to be computed. To do this, we grab previous buffers and shuffle them up or down to put those 3 elements
// somewhere thread 2 can use them. Thing is, we only want to use 1 shuffle, and to do that, we need halo values, 2 specifically. Now, imagine we have 4 threads in a warp.
// Basically, thread 0 and 4 cannot do any work, because both of them do not have a thread beside them to hand them their higher values:
//
//     Here are init values, normally all 0s      T0 
// x--------------------------------------------x 
// |O0 A1 B2 C3                                 | T1
// |A0 B1 C2                                    | T2
// |--------------------------------------------|     <- stride is here (2 ELEMS, 4 THREADS - 2)
// |B0 C1                                       | T3 
// |C0                                          |
// |                                            |
// |                                            |
// |                                            |
// |                                            |
// x--------------------------------------------x
//
// As you can clearly see now, T0 is basically only there to load and hold halo values in the digonal buffers, pretty much doing nothing most of the time, as is T3.
// Aside from this T1, and T2 are the only ones doing real work. Thankfully, with 32 threads, this scales great (30 working, 2 slacking, that's 93,75% working).
// 
// This is the reason why REG_BAND exists, and the way that the kernel works in short

__global__ void compute_dp_gpu_registers(Node* nodes, int level_nodes, Sequence sequence,
                                         Sequence sequence_rev, int elems_per_warp, int toprow_slots)
{
    const unsigned int mask = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;

    int node_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_in_block;
    if (node_idx >= level_nodes) return;

    Node* node = &nodes[node_idx];

    // ------------------------------------------------- Initialize -------------------------------------------------

    int M = sequence.size;
    int N = node->sequence.size;

    DTYPEMATRIX* __restrict last_col = node->last_col;

    const DTYPEALPHABET* __restrict node_seq  = node->sequence.sequence;
    const DTYPEALPHABET* __restrict query_rev = sequence_rev.sequence;

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* mine = &shared[warp_in_block * elems_per_warp];

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

    int v_a = 0, v_p = 0, v_pp = 0;

    auto process_diagonal = [&](int klo, int khi, int current_d,
                            int node_off, int query_off,
                            int off_diag, int off_up, int off_left) {

        int up_p    = (off_up   == 0) ? v_p  : __shfl_down_sync(mask, v_p, 1); // Do sync here instead of having an extra __syncwarps. There should be no divergence here
        int left_p  = (off_left == 0) ? v_p  : __shfl_up_sync(mask, v_p, 1);
        int diag_pp = (off_diag == 1) ? __shfl_down_sync(mask, v_pp, 1)
                                      : __shfl_up_sync(mask, v_pp, 1);

        if (lane >= klo && lane <= khi) {  // Should only affect in halo cases or grow/shrink
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

    if (lane == 0) {
        node->max_score = local_max;
        node->max_score_d = local_max_d;
        node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
        node->max_score_j = (local_max_d != -1) ? local_max_j : -1;
    }
}

// *************************************************************************************************
//
//                                           Scheduler
//
// *************************************************************************************************

// A level is now WARPS_PER_BLOCK nodes to a block instead of one, so the grid is that many times  smaller and the last block of a level is only partly filled,
// which is what level_nodes is for. Aside from this, everything should be really similar to previous schedulers

AlignmentResult gpu_align_registers(Graph graph, Graph cudaGraph, Sequence sequence)
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

        int toprow_slots = registers_toprow_slots(max_node_size, sequence.size);
        int elems_per_warp = registers_elems_per_warp(max_node_size, sequence.size);

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((nodes_per_level[d] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        int dynamic_shared_mem_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("cuda_registers", d, dynamic_shared_mem_bytes);

        compute_dp_gpu_registers<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
            act, nodes_per_level[d], sequence, sequence_rev, elems_per_warp, toprow_slots);

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

    return compute_traceback_gpu_registers(graph, sequence);
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Same thing here, this is reused code, with a changed name

static void recompute_node_dp_registers(Node* node, Sequence sequence, DTYPEMATRIX* dp)
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

AlignmentResult compute_traceback_gpu_registers(Graph graph, Sequence sequence) {
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
            recompute_node_dp_registers(curr_node, sequence, dp);
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
