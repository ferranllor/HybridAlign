#include "../include/cuda_multi.cuh"
#include "../include/hybrid_utils.cuh"

#define WARP_SIZE 32

struct DpBuffersMulti {
    DTYPEMATRIX* base;
    int stride;

    __device__ DTYPEMATRIX* operator[](int b) const { return base + b * stride; }
};

static inline int multi_toprow_slots(int max_node_size, int M) {
    return (max_node_size > M) ? (max_node_size + 1) : 0;
}

static inline int multi_elems_per_warp(int max_node_size, int M, int band_width) {
    return multi_toprow_slots(max_node_size, M) + (M + 1) + N_BUFFERS * (band_width + 2)
           + seq_slots_for_level(max_node_size, M);
}

// How many reads are aligned against the graph in one pass. Overridable from the environment so a
// sweep can find where the tail stops being the bottleneck.
int multi_num_reads(void)
{
    const char* e = getenv("NUM_READS");
    int r = e ? atoi(e) : 32;
    return (r < 1) ? 1 : r;
}

// *************************************************************************************************
//
//                                            Kernel
//
// *************************************************************************************************

// So, so far we've been trying to avoid having the GPU do the tail at all, because it's not very good at it,
// however, real genomics applications usually do multiple reads instead of one singular one because of the coarser 
// grain paralelism. It's for this reason (and mostly to get bigger numbers) that I've made this version with the help of claude.
//
// The idea is fairly simple, just do multiple reads and align them to the same graph. Then, let consecutive warps align different
// reads to the same node, and the result are tasks that are fairly homogeneous in size.
//
// This time however, you will see I used claude a lot. I figured it was code that did not affect performance, and I have other things 
// I want to look at, so I saved time here.
//
// You will see this version uses shared mem. I have another using registers, but for some reason, it is slower, hence the two versions

__global__ void compute_dp_gpu_multi(Node* nodes, int level_nodes, int num_reads,
                                     const DTYPEALPHABET* __restrict reads_rev, int M,
                                     int* __restrict out_max, int* __restrict out_d, int* __restrict out_j,
                                     int elems_per_warp, int toprow_slots, int band_width)
{
    const unsigned int mask = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;

    int pair = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_in_block;
    if (pair >= level_nodes * num_reads) return;

    int node_local = pair / num_reads;
    int read = pair - node_local * num_reads;

    Node* node = &nodes[node_local];

    // ------------------------------------------------- Initialize -------------------------------------------------

    int N = node->sequence.size;

    DTYPEMATRIX* __restrict last_col = &node->last_col[(size_t)read * (M + 1)];

    const DTYPEALPHABET* __restrict node_seq  = node->sequence.sequence;
    const DTYPEALPHABET* __restrict query_rev = &reads_rev[(size_t)read * M];

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* mine = &shared[warp_in_block * elems_per_warp];

    DTYPEMATRIX* topRow = mine;
    DTYPEMATRIX* prevCol = &mine[toprow_slots];

    DpBuffersMulti dpBuffers = { &prevCol[M + 1], band_width + 2 };

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
        const DTYPEMATRIX* __restrict prev_last = &node->v_in[0]->last_col[(size_t)read * (M + 1)];

        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = prev_last[i];
    }
    else {
        const DTYPEMATRIX* __restrict prev_last  = &node->v_in[0]->last_col[(size_t)read * (M + 1)];
        const DTYPEMATRIX* __restrict prev_last2 = &node->v_in[1]->last_col[(size_t)read * (M + 1)];

        for (int i = lane; i <= M; i += WARP_SIZE)
            prevCol[i] = max(prev_last[i], prev_last2[i]);

        for (int p = 2; p < node->num_in; ++p) {
            const DTYPEMATRIX* __restrict prev_lastp = &node->v_in[p]->last_col[(size_t)read * (M + 1)];
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

        __syncwarp(mask);

        bufferPrevPrev = bufferPrev;
        bufferPrev     = bufferAct;
        bufferAct      = (bufferAct + 1 == N_BUFFERS) ? 0 : bufferAct + 1;
    };

    __syncwarp(mask);

    if (M >= N) {
        for (int startN = 0; startN < N; startN += band_width) {
            int stripe_height = min(band_width, N - startN);

            int js = startN + 1;
            int je = startN + stripe_height;

            __syncwarp(mask);

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            if (lane == 0) {
                dpBuffers[bufferPrevPrev][0] = 0;
                dpBuffers[bufferPrev][0]     = prevCol[1];
                dpBuffers[bufferPrev][1]     = 0;
            }

            __syncwarp(mask);

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

            __syncwarp(mask);

            bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;

            if (lane == 0) {
                dpBuffers[bufferPrevPrev][stripe_height + 1] = prevCol[startM];
                dpBuffers[bufferPrev][stripe_height + 1]     = topRow[1];
                dpBuffers[bufferPrev][stripe_height]         = prevCol[is];
            }

            __syncwarp(mask);

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
        size_t slot = (size_t)node->id * num_reads + read;

        out_max[slot] = local_max;
        out_d[slot]   = local_max_d;
        out_j[slot]   = local_max_j;
    }
}

// *************************************************************************************************
//
//                                           Scheduler
//
// *************************************************************************************************

// The reads. Only the first one is real: the rest are copies of it with a deterministic scattering
// of substituted bases, because the datasets carry a single read per graph and what this version
// has to be measured on is how the tail behaves once there are many of them. Read 0 being untouched
// is also the correctness check, its alignment has to come out the same as every single sequence
// version produces.
// 
// This was done by claude, just generates some dummy sequences for my small dataset so we acn du multi sequence

void multi_build_reads(Sequence sequence, int num_reads, DTYPEALPHABET* reads, DTYPEALPHABET* reads_rev)
{
    const char bases[4] = { 'A', 'C', 'G', 'T' };
    int M = sequence.size;

    for (int r = 0; r < num_reads; r++) {
        DTYPEALPHABET* dst = &reads[(size_t)r * M];

        memcpy(dst, sequence.sequence, M * sizeof(DTYPEALPHABET));

        unsigned int h = 2166136261u ^ (unsigned int)r;
        for (int t = 0; r > 0 && t < M / 20; t++) {
            h = h * 16777619u + 2654435761u;
            dst[h % M] = bases[(h >> 16) & 3];
        }

        for (int i = 0; i < M; i++)
            reads_rev[(size_t)r * M + i] = dst[M - 1 - i];
    }
}

// One launch per level, as always, except the grid is now level_nodes * num_reads warps instead of
// level_nodes. That single change is the whole point: the 3733 tail levels used to launch one or
// two warps and leave the GPU idle, and they launch num_reads times that now.
//
// This was also done by claude (based on the previous versions)

AlignmentResult gpu_align_multi(Graph graph, Graph cudaGraph, Sequence sequence)
{
    int M = sequence.size;
    int num_reads = multi_num_reads();

    DTYPEALPHABET* reads = (DTYPEALPHABET*)malloc((size_t)num_reads * M);
    DTYPEALPHABET* reads_rev = (DTYPEALPHABET*)malloc((size_t)num_reads * M);
    multi_build_reads(sequence, num_reads, reads, reads_rev);

    DTYPEALPHABET* d_reads_rev = NULL;
    cudaMalloc((void**)&d_reads_rev, (size_t)num_reads * M);
    cudaMemcpy(d_reads_rev, reads_rev, (size_t)num_reads * M, cudaMemcpyHostToDevice);

    size_t slots = (size_t)graph.num_nodes * num_reads;

    int *d_max = NULL, *d_d = NULL, *d_j = NULL;
    cudaMalloc((void**)&d_max, slots * sizeof(int));
    cudaMalloc((void**)&d_d,   slots * sizeof(int));
    cudaMalloc((void**)&d_j,   slots * sizeof(int));

    int num_levels = graph.nodes[graph.num_nodes-1].depth + 1;
    int* nodes_per_level = (int*)calloc(num_levels, sizeof(int));

    for (int n = 0; n < graph.num_nodes; n++) nodes_per_level[graph.nodes[n].depth]++;

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    Node* device_nodes_pointers = (Node*)malloc(graph.num_nodes * sizeof(Node));
    cudaMemcpy(device_nodes_pointers, cudaGraph.nodes, graph.num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    Node* act = cudaGraph.nodes;
    int node_offset = 0;

    for (int d = 0; d < num_levels; d++)
    {
        int max_node_size = 0;
        for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
            if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

        int band_width = band_width_for_level(max_node_size, M);
        int toprow_slots = multi_toprow_slots(max_node_size, M);
        int elems_per_warp = multi_elems_per_warp(max_node_size, M, band_width);

        long long pairs = (long long)nodes_per_level[d] * num_reads;

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((unsigned int)((pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));

        int shared_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("cuda_multi", d, shared_bytes);

        compute_dp_gpu_multi<<<gridDim, blockDim, shared_bytes, compute_stream>>>(
            act, nodes_per_level[d], num_reads, d_reads_rev, M, d_max, d_d, d_j,
            elems_per_warp, toprow_slots, band_width);

        cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
        }

        act = &act[nodes_per_level[d]];
        node_offset += nodes_per_level[d];
    }

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
    }

    int* h_max = (int*)malloc(slots * sizeof(int));
    int* h_d   = (int*)malloc(slots * sizeof(int));
    int* h_j   = (int*)malloc(slots * sizeof(int));

    cudaMemcpy(h_max, d_max, slots * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d,   d_d,   slots * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_j,   d_j,   slots * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(graph.nodes[0].last_col,
               device_nodes_pointers[0].last_col,
               slots * (M + 1) * sizeof(DTYPEMATRIX),
               cudaMemcpyDeviceToHost);

    AlignmentResult res = compute_traceback_gpu_multi(graph, sequence, 0, num_reads, h_max, h_d, h_j, reads);

    for (int r = 1; r < num_reads; r++) {
        AlignmentResult other = compute_traceback_gpu_multi(graph, sequence, r, num_reads, h_max, h_d, h_j, reads);
        free(other.graph_align);
        free(other.query_align);
    }

    cudaStreamDestroy(compute_stream);
    cudaFree(d_reads_rev); cudaFree(d_max); cudaFree(d_d); cudaFree(d_j);
    free(h_max); free(h_d); free(h_j);
    free(nodes_per_level); free(device_nodes_pointers);
    free(reads); free(reads_rev);

    return res;
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Same rebuild as the single sequence versions, on one read's slice of the last columns. Every read
// walks its own path through the graph, so this runs once per read, and each one recomputes only
// the one or two nodes its own alignment crosses.
//
// Done by claude as well

static void recompute_node_dp_multi(Node* node, const DTYPEALPHABET* query, int M, int read,
                                    int num_reads, DTYPEMATRIX* dp)
{
    int N = node->sequence.size;
    int stride = N + 1;
    size_t off = (size_t)read * (M + 1);

    for (int j = 0; j <= N; j++) dp[j] = 0;

    for (int i = 1; i <= M; i++) {
        int boundary = 0;
        for (int p = 0; p < node->num_in; p++)
            if (node->v_in[p]->last_col[off + i] > boundary) boundary = node->v_in[p]->last_col[off + i];

        dp[i * stride] = boundary;
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            int score = (node->sequence.sequence[j-1] == query[i-1]) ? MATCH : MISMATCH;

            int diagonal = dp[(i-1) * stride + (j-1)] + score;
            int up       = dp[(i-1) * stride + j] + GAP;
            int left     = dp[i * stride + (j-1)] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[i * stride + j] = res;
        }
    }
}

AlignmentResult compute_traceback_gpu_multi(Graph graph, Sequence sequence, int read, int num_reads,
                                            const int* max_score, const int* max_d, const int* max_j,
                                            const DTYPEALPHABET* reads)
{
    int M = sequence.size;
    const DTYPEALPHABET* query = &reads[(size_t)read * M];
    size_t off = (size_t)read * (M + 1);

    int best = -1, best_node = 0;
    for (int n = 0; n < graph.num_nodes; n++) {
        int v = max_score[(size_t)n * num_reads + read];
        if (v > best) { best = v; best_node = n; }
    }

    Node* curr_node = &graph.nodes[best_node];
    int d = max_d[(size_t)best_node * num_reads + read];
    int j = max_j[(size_t)best_node * num_reads + read];
    int i = (d != -1) ? (d - j) : -1;

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

    while (curr_node != NULL && i >= 0) {
        int N = curr_node->sequence.size;
        int stride = N + 1;

        if (curr_node != loaded) {
            recompute_node_dp_multi(curr_node, query, M, read, num_reads, dp);
            loaded = curr_node;
        }

        int curr_score = dp[i * stride + j];
        if (curr_score <= 0) break;

        if (i > 0 && j > 0) {
            int score = (curr_node->sequence.sequence[j-1] == query[i-1]) ? MATCH : MISMATCH;

            int diag_score = dp[(i-1) * stride + (j-1)];
            int up_score   = dp[(i-1) * stride + j];

            if (curr_score == diag_score + score) {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = query[i-1];
                i--; j--;
            } else if (curr_score == up_score + GAP) {
                align_graph[pos] = '-';
                align_query[pos] = query[i-1];
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
                    if (prev->last_col[off + i] == curr_score) { best_prev = prev; break; }
                }
                curr_node = best_prev;
                if (curr_node) j = curr_node->sequence.size;
            } else {
                while (i > 0) {
                    align_graph[pos] = '-';
                    align_query[pos] = query[i-1];
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
