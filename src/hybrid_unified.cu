#include "../include/hybrid_unified.cuh"
#include "../include/cuda_shared_mem.cuh"

#include "../include/definitions.h"
extern "C" {
#include "../include/cpu_simd_parallel_node.h"
}

AlignmentResult gpu_align_hybrid_unified(Graph graph, Graph cudaGraph, Sequence sequence)
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

    // The kernel needs the widest node of the level to size its dynamic shared memory. Reading it
    // from the node structs inside the loop would touch managed pages while a kernel is running, so
    // it is gathered once, up front, into plain host memory.
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

    for (int d = 0; d < num_levels; d++)
    {
        if (nodes_per_level[d] >= HYBRID_MIN_NODES)
        {
            dim3 gridDim(nodes_per_level[d]);
            dim3 blockDim(BLOCKSIZE);

            int dynamic_shared_mem_bytes = sizeof(DTYPEMATRIX) * (max_node_size_per_level[d] + sequence.size + 2);
            dynamic_shared_mem_bytes += sizeof(DTYPEALPHABET) * (sequence.size + max_node_size_per_level[d]);

            compute_dp_gpu_shared_mem<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

            cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
            }
        }
        else
        {
            // Nothing to copy: the level reads the matrices the kernels just wrote, in place. Only
            // the ordering has to be enforced.
            status = cudaStreamSynchronize(compute_stream);
            if (status != cudaSuccess) {
                fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
            }

            if (nodes_per_level[d] > 1)
            {
                #pragma omp parallel for schedule(dynamic) num_threads(16)
                for (int t = 0; t < nodes_per_level[d]; t++)
                {
                    compute_dp_cpu_simd_parallel_node(&graph.nodes[node_offset + t], sequence);
                }
            }
            else
            {
                compute_dp_cpu_simd_parallel_node(&graph.nodes[node_offset], sequence);
            }
        }

        act = &act[nodes_per_level[d]];
        node_offset += nodes_per_level[d];
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
    }

    // Both sides wrote their scores into the same structs, so there is nothing to merge either.
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

    return compute_traceback_hybrid_unified(graph, sequence);
}

AlignmentResult compute_traceback_hybrid_unified(Graph graph, Sequence sequence) {
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
