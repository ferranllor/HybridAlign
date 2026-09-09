#include "../include/cuda_registers_merged_req.cuh"

extern "C" {
#include "../include/cpu_last_col.h"
}

#include "../include/cuda_dp_registers_merged_req.cuh"
#include "../include/hybrid_utils.cuh"

#define WARP_SIZE 32
#define REG_BAND (WARP_SIZE - 2)

// How many diagonals share one halo load and one handover store. A warp is 32 lanes, so 32
// consecutive diagonals is exactly what one coalesced access covers.
#define REQ_BLOCK WARP_SIZE


// Shared memory of ONE warp: topRow (only when the band runs along the rows, i.e. M < N) and
// prevCol. The rotating diagonals are gone from here, that is the point of the version.
int registers_merged_req_toprow_slots(int max_node_size, int M) {
    return (max_node_size > M) ? (max_node_size + 1) : 0;
}

int registers_merged_req_elems_per_warp(int max_node_size, int M) {
    return registers_merged_req_toprow_slots(max_node_size, M) + (M + 1) + seq_slots_for_level(max_node_size, M);
}

// *************************************************************************************************
//
//                                            Kernel
//
// *************************************************************************************************

// This version is rather simple, remember how we had to, at every diagonal, get one singular halo value? It was driving the 
// LSU units mad, so much so they were the bottleneck. I added this merged_req version, which just modified the kernel to request and store those halo
// elements 32 at a time, which is much easier on the LSUs. The thing is that this kernel now went from 50-ish registers -> 70-ish registers.
// This is now a problem because it lowers occupancy, so we'll have to see how to work on that... 

__global__ void compute_dp_gpu_registers_merged_req(Node* nodes, int level_nodes, Sequence sequence,
                                                Sequence sequence_rev, int elems_per_warp, int toprow_slots)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;

    int node_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_in_block;
    if (node_idx >= level_nodes) return;

    Node* node = &nodes[node_idx];

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* mine = &shared[warp_in_block * elems_per_warp];

    int local_max, local_max_d, local_max_j;

    compute_dp_node_registers_merged_req(node, sequence.size, sequence_rev.sequence, 0,
                                         mine, toprow_slots, local_max, local_max_d, local_max_j);

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

AlignmentResult gpu_align_registers_merged_req(Graph graph, Graph cudaGraph, Sequence sequence)
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

        int toprow_slots = registers_merged_req_toprow_slots(max_node_size, sequence.size);
        int elems_per_warp = registers_merged_req_elems_per_warp(max_node_size, sequence.size);

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((nodes_per_level[d] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        int dynamic_shared_mem_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("cuda_registers_merged_req", d, dynamic_shared_mem_bytes);

        compute_dp_gpu_registers_merged_req<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
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

    return compute_traceback_gpu_registers_merged_req(graph, sequence);
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// The kernel leaves the same last columns behind as every other last column version, so this is the
// shared walk back from cpu_last_col.c.

AlignmentResult compute_traceback_gpu_registers_merged_req(Graph graph, Sequence sequence)
{
    Node* start = &graph.nodes[graph.max_score_node_id];

    return traceback_last_col(graph, sequence.sequence, sequence.size, 0,
                              start, start->max_score_i, start->max_score_j);
}
