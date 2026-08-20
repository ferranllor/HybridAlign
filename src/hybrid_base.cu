#include "../include/hybrid_base.cuh"
#include "../include/cuda_shared_mem.cuh"
#include "../include/nvtx_ranges.cuh"

#include "../include/definitions.h"
extern "C" {
#include "../include/cpu_simd_parallel_node.h"
}

AlignmentResult gpu_align_hybrid_base(Graph graph, Graph cudaGraph, Sequence sequence)
{
    NVTX_PUSH("setup", NVTX_COL_SETUP);

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

    // Last level that still runs on the GPU. Once it is done nothing has to travel back to the
    // device any more, so the CPU levels after it keep their matrices on the host.
    int last_gpu_level = -1;
    for (int d = 0; d < num_levels; d++)
        if (nodes_per_level[d] >= HYBRID_MIN_NODES) last_gpu_level = d;

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

    int batch_start_offset = 0;     // node_offset at start of current batch
    Node* batch_start_act  = act;   // device pointer at start of current batch
    size_t batch_bytes = 0;         // matrices already queued in this batch

    NVTX_POP(); // setup
    NVTX_PUSH("align loop", NVTX_COL_SETUP);

    for (int d = 0; d < num_levels; d++)
    {
        bool level_on_gpu = (nodes_per_level[d] >= HYBRID_MIN_NODES);

        if (level_on_gpu)
        {
            NVTX_PUSHF(NVTX_COL_GPU, "gpu launch L%d (%d nodes)", d, nodes_per_level[d]);

            int max_node_size = 0;
            for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
                if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

            dim3 gridDim(nodes_per_level[d]);
            dim3 blockDim(band_width_for_level(max_node_size, sequence.size));

            int dynamic_shared_mem_bytes = shared_bytes_for_level(max_node_size, sequence.size, blockDim.x);

            compute_dp_gpu_shared_mem<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

            cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
            }

            for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
                batch_bytes += (size_t)(sequence.size + 2) * (graph.nodes[n].sequence.size + 2) * sizeof(DTYPEMATRIX);

            act = &act[nodes_per_level[d]];
            node_offset += nodes_per_level[d];

            NVTX_POP(); // gpu launch
        }

        // The batch has to be on the host before the next CPU level reads it, and the levels are
        // otherwise flushed once they are big enough to be worth a copy command of their own.
        bool next_level_on_cpu = (d + 1 < num_levels) && (nodes_per_level[d + 1] < HYBRID_MIN_NODES);
        bool last_level = (d == num_levels - 1);

        if (batch_bytes > 0 && (batch_bytes >= BATCH_BYTES || next_level_on_cpu || last_level))
        {
            NVTX_PUSHF(NVTX_COL_COPY, "queue D2H batch (%zu KB)", batch_bytes / 1024);

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

            cudaMemcpyAsync(graph.nodes[batch_start_offset].dp_matrix,
                            device_nodes_pointers[batch_start_offset].dp_matrix,
                            batch_bytes,
                            cudaMemcpyDeviceToHost,
                            copy_stream);

            // reset batch trackers
            batch_start_offset = node_offset;
            batch_start_act = act;
            batch_bytes = 0;

            NVTX_POP(); // queue D2H batch
        }

        if (!level_on_gpu)
        {
            // Everything this level reads (the last column of each predecessor) has to have landed
            // on the host already.
            NVTX_PUSH("wait for D2H", NVTX_COL_SYNC);

            status = cudaStreamSynchronize(copy_stream);
            if (status != cudaSuccess) {
                fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
            }

            NVTX_POP(); // wait for D2H

            NVTX_PUSHF(NVTX_COL_CPU, "cpu level L%d (%d nodes)", d, nodes_per_level[d]);

            // Half of these levels hold a single node, and opening a parallel region for one node
            // costs more than the node does.
            if (nodes_per_level[d] > 1)
            {
                #pragma omp parallel for schedule(dynamic) num_threads(2)
                for (int t = 0; t < nodes_per_level[d]; t++)
                {
                    compute_dp_cpu_simd_parallel_node(&graph.nodes[node_offset + t], sequence);
                }
            }
            else
            {
                compute_dp_cpu_simd_parallel_node(&graph.nodes[node_offset], sequence);
            }

            NVTX_POP(); // cpu level

            // Only worth uploading if some later level still runs on the device.
            if (d < last_gpu_level)
            {
                NVTX_PUSHF(NVTX_COL_COPY, "H2D level L%d", d);

                size_t level_bytes = 0;
                for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++)
                    level_bytes += (size_t)(sequence.size + 2) * (graph.nodes[n].sequence.size + 2) * sizeof(DTYPEMATRIX);

                cudaMemcpy(device_nodes_pointers[node_offset].dp_matrix,
                           graph.nodes[node_offset].dp_matrix,
                           level_bytes,
                           cudaMemcpyHostToDevice);

                NVTX_POP(); // H2D level
            }

            act = &act[nodes_per_level[d]];
            node_offset += nodes_per_level[d];

            // the CPU wrote straight into the host matrices, so the batch restarts after it
            batch_start_offset = node_offset;
            batch_start_act = act;
        }
    }

    NVTX_POP(); // align loop
    NVTX_PUSH("final sync", NVTX_COL_SYNC);

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
    }

    NVTX_POP(); // final sync
    NVTX_PUSH("merge scores", NVTX_COL_REDUCE);

    // The GPU levels wrote their scores into the device node structs, the CPU levels wrote theirs
    // straight into the host ones, so only the former have to be taken over.
    node_offset = 0;
    for (int d = 0; d < num_levels; d++) {
        if (nodes_per_level[d] >= HYBRID_MIN_NODES) {
            for (int n = node_offset; n < node_offset + nodes_per_level[d]; n++) {
                graph.nodes[n].max_score   = device_nodes_tmp[n].max_score;
                graph.nodes[n].max_score_d = device_nodes_tmp[n].max_score_d;
                graph.nodes[n].max_score_i = device_nodes_tmp[n].max_score_i;
                graph.nodes[n].max_score_j = device_nodes_tmp[n].max_score_j;
            }
        }
        node_offset += nodes_per_level[d];
    }

    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        if (graph.nodes[n].max_score > graph.max_score) {
            graph.max_score = graph.nodes[n].max_score;
            graph.max_score_node_id = n;
        }
    }

    NVTX_POP(); // merge scores
    NVTX_PUSH("teardown", NVTX_COL_SETUP);

    cudaEventDestroy(compute_done);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    free(nodes_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);
    cudaFreeHost(device_nodes_tmp);
    free(device_nodes_pointers);

    NVTX_POP(); // teardown
    NVTX_PUSH("traceback", NVTX_COL_TRACEBACK);

    AlignmentResult res = compute_traceback_hybrid_base(graph, sequence);

    NVTX_POP(); // traceback

    return res;
}

AlignmentResult compute_traceback_hybrid_base(Graph graph, Sequence sequence) {
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
