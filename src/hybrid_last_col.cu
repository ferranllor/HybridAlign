#include "../include/hybrid_last_col.cuh"
#include "../include/cuda_warps.cuh"
#include "../include/cuda_registers.cuh"
#include "../include/nvtx_ranges.cuh"

extern "C" {
#include "../include/cpu_last_col.h"
}

#include <time.h>

static inline double hybrid_last_col_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// *************************************************************************************************
//
//                                          Align loop
//
// *************************************************************************************************

// I guess this one was only logical. Now that we don't have to move almost any data, having a hybrid version always works,
// given the lack of paralelism of the long tail in the datasets

static AlignmentResult gpu_align_hybrid_last_col(Graph graph, Graph cudaGraph, Sequence sequence,
                                                 bool use_registers)
{
    bool stats = (getenv("HYBRID_STATS") != NULL);
    double t_launch = 0.0, t_sync = 0.0, t_cpu = 0.0, t_traceback = 0.0;
    double t_total = hybrid_last_col_now_ms();
    int gpu_levels = 0, cpu_levels = 0, syncs = 0;

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

    int* max_node_size_per_level = (int*)calloc(num_levels, sizeof(int));
    int max_node_size_all = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        int depth = graph.nodes[n].depth;
        if (graph.nodes[n].sequence.size > max_node_size_per_level[depth])
            max_node_size_per_level[depth] = graph.nodes[n].sequence.size;
        if (graph.nodes[n].sequence.size > max_node_size_all)
            max_node_size_all = graph.nodes[n].sequence.size;
    }

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    cudaError_t status;
    Node* act = graph.nodes;
    int node_offset = 0;
    bool gpu_work_pending = false;

    NVTX_POP(); // setup
    NVTX_PUSH("align loop", NVTX_COL_SETUP);

    for (int d = 0; d < num_levels; d++)
    {
        if (nodes_per_level[d] >= HYBRID_MIN_NODES)
        {
            NVTX_PUSHF(NVTX_COL_GPU, "gpu launch L%d (%d nodes)", d, nodes_per_level[d]);

            double t0 = hybrid_last_col_now_ms();

            int max_node_size = max_node_size_per_level[d];

            dim3 blockDim(WARPS_PER_BLOCK * 32);
            dim3 gridDim((nodes_per_level[d] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

            if (use_registers) {
                int toprow_slots = registers_toprow_slots(max_node_size, sequence.size);
                int elems_per_warp = registers_elems_per_warp(max_node_size, sequence.size);
                int shared_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
                check_shared_fits(use_registers ? "hybrid registers" : "hybrid warps", d, shared_bytes);

                compute_dp_gpu_registers<<<gridDim, blockDim, shared_bytes, compute_stream>>>(
                    act, nodes_per_level[d], sequence, sequence_rev, elems_per_warp, toprow_slots);
            } else {
                int band_width = band_width_for_level(max_node_size, sequence.size);
                int toprow_slots = warps_toprow_slots(max_node_size, sequence.size);
                int elems_per_warp = warps_elems_per_warp(max_node_size, sequence.size, band_width);
                int shared_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
                check_shared_fits(use_registers ? "hybrid registers" : "hybrid warps", d, shared_bytes);

                compute_dp_gpu_warps<<<gridDim, blockDim, shared_bytes, compute_stream>>>(
                    act, nodes_per_level[d], sequence, sequence_rev, elems_per_warp, toprow_slots, band_width);
            }

            cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
            }

            gpu_work_pending = true;
            gpu_levels++;

            act = &act[nodes_per_level[d]];
            node_offset += nodes_per_level[d];

            t_launch += hybrid_last_col_now_ms() - t0;

            NVTX_POP(); // gpu launch
        }
        else
        {
            int d_end = d;
            while (d_end < num_levels && nodes_per_level[d_end] < HYBRID_MIN_NODES) d_end++;

            if (gpu_work_pending)
            {
                NVTX_PUSH("wait for kernels", NVTX_COL_SYNC);

                double t0 = hybrid_last_col_now_ms();

                status = cudaStreamSynchronize(compute_stream);
                if (status != cudaSuccess) {
                    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
                }

                gpu_work_pending = false;
                syncs++;
                t_sync += hybrid_last_col_now_ms() - t0;

                NVTX_POP(); // wait for kernels
            }

            NVTX_PUSHF(NVTX_COL_CPU, "cpu levels L%d-%d", d, d_end - 1);

            double t0 = hybrid_last_col_now_ms();

            #pragma omp parallel num_threads(HYBRID_CPU_THREADS)
            {
                int level_offset = node_offset;
                DTYPEMATRIX* scratch = (DTYPEMATRIX*)malloc(3 * (size_t)(max_node_size_all + 2) * sizeof(DTYPEMATRIX));

                for (int l = d; l < d_end; l++)
                {
                    #pragma omp for schedule(dynamic)
                    for (int t = 0; t < nodes_per_level[l]; t++)
                    {
                        Node* node = &graph.nodes[level_offset + t];
                        int best, bd, bj;

                        compute_dp_cpu_last_col(node, sequence, tmp.sequence, 0, scratch, &best, &bd, &bj);

                        node->max_score = best;
                        node->max_score_d = bd;
                        node->max_score_i = (bd != -1) ? (bd - bj) : -1;
                        node->max_score_j = (bd != -1) ? bj : -1;
                    }

                    level_offset += nodes_per_level[l];
                }

                free(scratch);
            }

            t_cpu += hybrid_last_col_now_ms() - t0;

            NVTX_POP(); // cpu levels

            for (int l = d; l < d_end; l++) {
                act = &act[nodes_per_level[l]];
                node_offset += nodes_per_level[l];
                cpu_levels++;
            }

            d = d_end - 1;
        }
    }

    NVTX_POP(); // align loop

    if (gpu_work_pending)
    {
        NVTX_PUSH("final sync", NVTX_COL_SYNC);

        double t0 = hybrid_last_col_now_ms();

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
        }

        syncs++;
        t_sync += hybrid_last_col_now_ms() - t0;

        NVTX_POP(); // final sync
    }

    NVTX_PUSH("merge scores", NVTX_COL_REDUCE);

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

    cudaStreamDestroy(compute_stream);
    free(nodes_per_level);
    free(max_node_size_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);

    NVTX_POP(); // teardown
    NVTX_PUSH("traceback", NVTX_COL_TRACEBACK);

    double t0 = hybrid_last_col_now_ms();
    AlignmentResult res = compute_traceback_hybrid_last_col(graph, sequence);
    t_traceback = hybrid_last_col_now_ms() - t0;

    NVTX_POP(); // traceback

    if (stats) {
        fprintf(stderr, "[hybrid %s] total %7.2f ms | launch %6.2f ms (%d levels) | wait %7.2f ms (%d syncs) "
                        "| cpu %7.2f ms (%d levels) | traceback %5.2f ms\n",
                use_registers ? "registers" : "warps",
                hybrid_last_col_now_ms() - t_total, t_launch, gpu_levels, t_sync, syncs,
                t_cpu, cpu_levels, t_traceback);
    }

    return res;
}

AlignmentResult gpu_align_hybrid_warps(Graph graph, Graph cudaGraph, Sequence sequence)
{
    return gpu_align_hybrid_last_col(graph, cudaGraph, sequence, false);
}

AlignmentResult gpu_align_hybrid_registers(Graph graph, Graph cudaGraph, Sequence sequence)
{
    return gpu_align_hybrid_last_col(graph, cudaGraph, sequence, true);
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

AlignmentResult compute_traceback_hybrid_last_col(Graph graph, Sequence sequence)
{
    Node* start = &graph.nodes[graph.max_score_node_id];

    return traceback_last_col(graph, sequence.sequence, sequence.size, 0,
                              start, start->max_score_i, start->max_score_j);
}
