#include "../include/hybrid_pinned.cuh"
#include "../include/cuda_shared_mem.cuh"

#include "../include/definitions.h"
extern "C" {
#include "../include/cpu_simd_parallel_node.h"
}

#include <time.h>

// The same alignment loop is used by hybrid version 2 (pinned host memory, init_pinned_graph) and
// version 3 (managed memory with the migration policy fixed, init_advised_graph). Nothing in here
// depends on which one allocated the graph, which is the point: running both is a controlled
// experiment on the memory kind alone.
//
// Compared to hybrid_unified it changes two things about *when* it talks to the driver, both of
// which only ever remove work:
//
//   * it synchronises only when a kernel is actually in flight, instead of once per CPU level
//     (150_10 has ~3730 CPU levels but only ~11 GPU ones),
//   * it opens one OpenMP parallel region per run of consecutive CPU levels instead of one per
//     level, using the implicit barrier of "omp for" to keep the levels ordered.
//
// Set HYBRID_STATS=1 in the environment to get a per alignment breakdown on stderr.

static inline double hybrid_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

AlignmentResult gpu_align_hybrid_pinned(Graph graph, Graph cudaGraph, Sequence sequence)
{
    bool stats = (getenv("HYBRID_STATS") != NULL);
    double t_launch = 0.0, t_sync = 0.0, t_cpu = 0.0, t_traceback = 0.0;
    double t_total = hybrid_now_ms();
    int gpu_levels = 0, cpu_levels = 0, syncs = 0;

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

    // Gathered up front, in plain host memory: reading it from the node structs inside the loop
    // would touch the shared graph while a kernel is running.
    int* max_node_size_per_level = (int*)calloc(num_levels, sizeof(int));

    for (int n = 0; n < graph.num_nodes; n++) {
        int depth = graph.nodes[n].depth;
        if (graph.nodes[n].sequence.size > max_node_size_per_level[depth])
            max_node_size_per_level[depth] = graph.nodes[n].sequence.size;
    }

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    cudaError_t status;
    Node* act = graph.nodes; // host and device walk the very same array
    int node_offset = 0;
    bool gpu_work_pending = false;

    for (int d = 0; d < num_levels; d++)
    {
        if (nodes_per_level[d] >= HYBRID_MIN_NODES)
        {
            double t0 = hybrid_now_ms();

            dim3 gridDim(nodes_per_level[d]);
            dim3 blockDim(BLOCKSIZE);

            int dynamic_shared_mem_bytes = sizeof(DTYPEMATRIX) * (max_node_size_per_level[d] + sequence.size + 2);
            dynamic_shared_mem_bytes += sizeof(DTYPEALPHABET) * (sequence.size + max_node_size_per_level[d]);

            compute_dp_gpu_shared_mem<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

            cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
            }

            gpu_work_pending = true;
            gpu_levels++;

            act = &act[nodes_per_level[d]];
            node_offset += nodes_per_level[d];

            t_launch += hybrid_now_ms() - t0;
        }
        else
        {
            // Whole run of consecutive CPU levels, so the threads are gathered once for all of them
            int d_end = d;
            while (d_end < num_levels && nodes_per_level[d_end] < HYBRID_MIN_NODES) d_end++;

            // Nothing to copy, but the levels do read what the kernels wrote, and only a kernel
            // that is still in flight has to be waited for.
            if (gpu_work_pending)
            {
                double t0 = hybrid_now_ms();

                status = cudaStreamSynchronize(compute_stream);
                if (status != cudaSuccess) {
                    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
                }

                gpu_work_pending = false;
                syncs++;
                t_sync += hybrid_now_ms() - t0;
            }

            double t0 = hybrid_now_ms();

            #pragma omp parallel num_threads(HYBRID_CPU_THREADS)
            {
                int level_offset = node_offset;

                for (int l = d; l < d_end; l++)
                {
                    // the implicit barrier at the end of "omp for" is what orders the levels
                    #pragma omp for schedule(dynamic)
                    for (int t = 0; t < nodes_per_level[l]; t++)
                    {
                        compute_dp_cpu_simd_parallel_node(&graph.nodes[level_offset + t], sequence);
                    }

                    level_offset += nodes_per_level[l];
                }
            }

            t_cpu += hybrid_now_ms() - t0;

            for (int l = d; l < d_end; l++) {
                act = &act[nodes_per_level[l]];
                node_offset += nodes_per_level[l];
                cpu_levels++;
            }

            d = d_end - 1; // the loop's ++d moves past the run
        }
    }

    if (gpu_work_pending)
    {
        double t0 = hybrid_now_ms();

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
        }

        syncs++;
        t_sync += hybrid_now_ms() - t0;
    }

    // Both sides wrote their scores into the same structs, so there is nothing to merge.
    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        if (graph.nodes[n].max_score > graph.max_score) {
            graph.max_score = graph.nodes[n].max_score;
            graph.max_score_node_id = n;
        }
    }

    cudaStreamDestroy(compute_stream);
    free(nodes_per_level);
    free(max_node_size_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);

    double t0 = hybrid_now_ms();
    AlignmentResult res = compute_traceback_hybrid_pinned(graph, sequence);
    t_traceback = hybrid_now_ms() - t0;

    if (stats) {
        fprintf(stderr, "[hybrid] total %7.2f ms | launch %6.2f ms (%d levels) | wait %7.2f ms (%d syncs) "
                        "| cpu %7.2f ms (%d levels) | traceback %5.2f ms\n",
                hybrid_now_ms() - t_total, t_launch, gpu_levels, t_sync, syncs,
                t_cpu, cpu_levels, t_traceback);
    }

    return res;
}

AlignmentResult compute_traceback_hybrid_pinned(Graph graph, Sequence sequence) {
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
