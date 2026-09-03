#include "../include/cuda_multi_registers.cuh"
#include "../include/cuda_multi.cuh"
#include "../include/cuda_dp_registers.cuh"

#include "../include/hybrid_utils.cuh"

// *************************************************************************************************
//
//                                            Kernel
//
// *************************************************************************************************

// Same kernel as multi, but using the registers implementation instead of shared mem. Should be faster, but for some reason, it seems slower.

__global__ void compute_dp_gpu_multi_registers(Node* nodes, int level_nodes, int num_reads,
                                               const DTYPEALPHABET* __restrict reads_rev, int M,
                                               int* __restrict out_max, int* __restrict out_d,
                                               int* __restrict out_j, int elems_per_warp, int toprow_slots)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;

    int pair = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_in_block;
    if (pair >= level_nodes * num_reads) return;

    int node_local = pair / num_reads;
    int read = pair - node_local * num_reads;

    Node* node = &nodes[node_local];

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* mine = &shared[warp_in_block * elems_per_warp];

    int local_max, local_max_d, local_max_j;

    compute_dp_node_registers(node, M, &reads_rev[(size_t)read * M], (size_t)read * (M + 1),
                              mine, toprow_slots, local_max, local_max_d, local_max_j);

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

// Version 0's scheduler with the band gone: the registers core fixes its own band at REG_BAND, so
// a level only decides how much shared memory a warp needs and how many pairs there are.

AlignmentResult gpu_align_multi_registers(Graph graph, Graph cudaGraph, Sequence sequence)
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

        int toprow_slots = registers_dp_toprow_slots(max_node_size, M);
        int elems_per_warp = registers_dp_elems_per_warp(max_node_size, M);

        long long pairs = (long long)nodes_per_level[d] * num_reads;

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((unsigned int)((pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));

        int shared_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("cuda_multi_registers", d, shared_bytes);

        compute_dp_gpu_multi_registers<<<gridDim, blockDim, shared_bytes, compute_stream>>>(
            act, nodes_per_level[d], num_reads, d_reads_rev, M, d_max, d_d, d_j,
            elems_per_warp, toprow_slots);

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
