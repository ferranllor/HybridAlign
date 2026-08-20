#include "../include/cuda_no_copy.cuh"

#include "../include/cuda_naive.cuh"
#include "../include/cuda_async_batching.cuh"
#include "../include/cuda_shared_mem.cuh"

#include <time.h>

// Set HYBRID_STATS=1 in the environment for a per alignment breakdown on stderr, same as the
// hybrid versions.

static inline double no_copy_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Everything these three share: the reversed query, the level table and, at the end, the scan for
// the best node. There is nothing to copy back, so the "finish" step is only a synchronise.

static void no_copy_prologue(Graph graph, Sequence sequence, Sequence* sequence_rev,
                             int* num_levels, int** nodes_per_level, int** max_node_size_per_level)
{
    Sequence tmp;

    tmp.sequence = (char*)malloc(sequence.size * sizeof(char));
    for (int idx = 0; idx < sequence.size; ++idx) {
        tmp.sequence[idx] = sequence.sequence[sequence.size - 1 - idx];
    }

    sequence_rev->size = sequence.size;
    cudaMalloc((void**)&sequence_rev->sequence, sequence.size * sizeof(char));
    cudaMemcpy(sequence_rev->sequence, tmp.sequence, sequence.size * sizeof(char), cudaMemcpyHostToDevice);
    free(tmp.sequence);

    *num_levels = graph.nodes[graph.num_nodes-1].depth + 1;
    *nodes_per_level = (int*)calloc(*num_levels, sizeof(int));
    *max_node_size_per_level = (int*)calloc(*num_levels, sizeof(int));

    for (int n = 0; n < graph.num_nodes; n++) {
        int depth = graph.nodes[n].depth;
        (*nodes_per_level)[depth]++;
        if (graph.nodes[n].sequence.size > (*max_node_size_per_level)[depth])
            (*max_node_size_per_level)[depth] = graph.nodes[n].sequence.size;
    }
}

static AlignmentResult no_copy_epilogue(Graph graph, Sequence sequence, Sequence sequence_rev,
                                        int* nodes_per_level, int* max_node_size_per_level,
                                        cudaStream_t compute_stream, double t_launch, int launches)
{
    double t0 = no_copy_now_ms();

    cudaError_t status = cudaStreamSynchronize(compute_stream);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
    }

    double t_wait = no_copy_now_ms() - t0;

    // The kernels wrote the scores into the shared node structs, so they are already here.
    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        if (graph.nodes[n].max_score > graph.max_score) {
            graph.max_score = graph.nodes[n].max_score;
            graph.max_score_node_id = n;
        }
    }

    cudaStreamDestroy(compute_stream);
    cudaFree(sequence_rev.sequence);
    free(nodes_per_level);
    free(max_node_size_per_level);

    t0 = no_copy_now_ms();
    AlignmentResult res = compute_traceback_no_copy(graph, sequence);
    double t_traceback = no_copy_now_ms() - t0;

    if (getenv("HYBRID_STATS") != NULL) {
        fprintf(stderr, "[no_copy] launch %6.2f ms (%d launches) | wait %7.2f ms | traceback %5.2f ms\n",
                t_launch, launches, t_wait, t_traceback);
    }

    return res;
}

// --------------------------------------------------------------------------------- one per node

AlignmentResult gpu_align_no_copy_naive(Graph graph, Graph cudaGraph, Sequence sequence)
{
    Sequence sequence_rev;
    int num_levels;
    int* nodes_per_level;
    int* max_node_size_per_level;

    no_copy_prologue(graph, sequence, &sequence_rev, &num_levels, &nodes_per_level, &max_node_size_per_level);

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    double t0 = no_copy_now_ms();

    dim3 blockDim(BLOCKSIZE);
    Node* act = cudaGraph.nodes;

    for (int d = 0; d < num_levels; d++)
    {
        for (int t = 0; t < nodes_per_level[d]; t++)
        {
            compute_dp_gpu_naive<<<1, blockDim, 0, compute_stream>>>(&act[t], sequence, sequence_rev);
        }

        // no synchronise here: one stream already runs the levels in order

        act = &act[nodes_per_level[d]];
    }

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
    }

    return no_copy_epilogue(graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, graph.num_nodes);
}

// -------------------------------------------------------------------------------- one per level

AlignmentResult gpu_align_no_copy_level(Graph graph, Graph cudaGraph, Sequence sequence)
{
    Sequence sequence_rev;
    int num_levels;
    int* nodes_per_level;
    int* max_node_size_per_level;

    no_copy_prologue(graph, sequence, &sequence_rev, &num_levels, &nodes_per_level, &max_node_size_per_level);

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    double t0 = no_copy_now_ms();

    dim3 blockDim(BLOCKSIZE);
    Node* act = cudaGraph.nodes;

    for (int d = 0; d < num_levels; d++)
    {
        dim3 gridDim(nodes_per_level[d]);

        compute_dp_gpu_async_batching<<<gridDim, blockDim, 0, compute_stream>>>(act, sequence, sequence_rev);

        act = &act[nodes_per_level[d]];
    }

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
    }

    return no_copy_epilogue(graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels);
}

// ------------------------------------------------------- one per level, shared memory kernel

AlignmentResult gpu_align_no_copy_shared_mem(Graph graph, Graph cudaGraph, Sequence sequence)
{
    Sequence sequence_rev;
    int num_levels;
    int* nodes_per_level;
    int* max_node_size_per_level;

    no_copy_prologue(graph, sequence, &sequence_rev, &num_levels, &nodes_per_level, &max_node_size_per_level);

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    double t0 = no_copy_now_ms();

    Node* act = cudaGraph.nodes;

    for (int d = 0; d < num_levels; d++)
    {
        dim3 gridDim(nodes_per_level[d]);
        dim3 blockDim(band_width_for_level(max_node_size_per_level[d], sequence.size));

        int dynamic_shared_mem_bytes = shared_bytes_for_level(max_node_size_per_level[d], sequence.size, blockDim.x);

        compute_dp_gpu_shared_mem<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

        act = &act[nodes_per_level[d]];
    }

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
    }

    return no_copy_epilogue(graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels);
}

AlignmentResult compute_traceback_no_copy(Graph graph, Sequence sequence) {
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
