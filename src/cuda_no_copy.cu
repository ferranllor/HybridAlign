#include "../include/cuda_no_copy.cuh"

#include "../include/cuda_naive.cuh"
#include "../include/cuda_async_batching.cuh"
#include "../include/cuda_shared_mem.cuh"
#include "../include/cuda_last_col.cuh"
#include "../include/cuda_warps.cuh"
#include "../include/cuda_registers.cuh"
#include "../include/cuda_persistent_kernels.cuh"
#include "../include/cuda_dp_registers.cuh"

#include "../include/hybrid_utils.cuh"

extern "C" {
#include "../include/cpu_last_col.h"
}

#include <sched.h>
#include <time.h>

// Set HYBRID_STATS=1 in the environment for a per alignment breakdown on stderr, same as the
// hybrid versions.

static inline double no_copy_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Everything these share: the reversed query, the level table and, at the end, the scan for the
// best node. There is nothing to copy back, so the "finish" step is only a synchronise.

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

// use_last_col picks which traceback runs at the end: the versions built on a full score matrix
// read it straight out of the shared graph, the ones that only left a last column behind recompute
// the handful of nodes the alignment actually crosses.

static AlignmentResult no_copy_epilogue(const char* who, Graph graph, Sequence sequence, Sequence sequence_rev,
                                        int* nodes_per_level, int* max_node_size_per_level,
                                        cudaStream_t compute_stream, double t_launch, int launches,
                                        bool use_last_col)
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
    AlignmentResult res = use_last_col ? compute_traceback_no_copy_last_col(graph, sequence)
                                       : compute_traceback_no_copy(graph, sequence);
    double t_traceback = no_copy_now_ms() - t0;

    if (getenv("HYBRID_STATS") != NULL) {
        fprintf(stderr, "[no_copy %s] launch %6.2f ms (%d launches) | wait %7.2f ms | traceback %5.2f ms\n",
                who, t_launch, launches, t_wait, t_traceback);
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

    return no_copy_epilogue("naive", graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, graph.num_nodes, false);
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

    return no_copy_epilogue("level", graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels, false);
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

    return no_copy_epilogue("shared_mem", graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels, false);
}

// -------------------------------------------------- one per level, last column only, shared mem

// From here on the versions keep only a last column per node, so nothing near the 7.8 GB of score
// matrices is written and the traceback recomputes the nodes it walks. Same kernels as mode 1
// versions 8 to 11: what disappears is the batched copy of the columns and of the node structs,
// since the graph the kernels write is already the one the traceback reads.

AlignmentResult gpu_align_no_copy_last_col(Graph graph, Graph cudaGraph, Sequence sequence)
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

        compute_dp_gpu_last_col<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(act, sequence, sequence_rev);

        act = &act[nodes_per_level[d]];
    }

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
    }

    return no_copy_epilogue("last_col", graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels, true);
}

// ------------------------------------------------------------- one warp per node, shared memory

AlignmentResult gpu_align_no_copy_warps(Graph graph, Graph cudaGraph, Sequence sequence)
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
        int max_node_size = max_node_size_per_level[d];

        int band_width = band_width_for_level(max_node_size, sequence.size);
        int toprow_slots = warps_toprow_slots(max_node_size, sequence.size);
        int elems_per_warp = warps_elems_per_warp(max_node_size, sequence.size, band_width);

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((nodes_per_level[d] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        int dynamic_shared_mem_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("no_copy warps", d, dynamic_shared_mem_bytes);

        compute_dp_gpu_warps<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
            act, nodes_per_level[d], sequence, sequence_rev, elems_per_warp, toprow_slots, band_width);

        act = &act[nodes_per_level[d]];
    }

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
    }

    return no_copy_epilogue("warps", graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels, true);
}

// ------------------------------------------------------------------ one warp per node, shuffles

AlignmentResult gpu_align_no_copy_registers(Graph graph, Graph cudaGraph, Sequence sequence)
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
        int max_node_size = max_node_size_per_level[d];

        int toprow_slots = registers_toprow_slots(max_node_size, sequence.size);
        int elems_per_warp = registers_elems_per_warp(max_node_size, sequence.size);

        dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
        dim3 gridDim((nodes_per_level[d] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        int dynamic_shared_mem_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);
        check_shared_fits("no_copy registers", d, dynamic_shared_mem_bytes);

        compute_dp_gpu_registers<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
            act, nodes_per_level[d], sequence, sequence_rev, elems_per_warp, toprow_slots);

        act = &act[nodes_per_level[d]];
    }

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
    }

    return no_copy_epilogue("registers", graph, sequence, sequence_rev, nodes_per_level, max_node_size_per_level,
                            compute_stream, no_copy_now_ms() - t0, num_levels, true);
}

// --------------------------------------------------------------------------- persistent kernels

// Same workers as mode 1 version 11, on the registers core, one warp each. The whole "STEP 3" of
// that version, the batched copy of the columns and of the node structs, is gone: draining the
// rings at the end of a level is all that is left between two levels, and the scores are already
// in the shared node structs when the last worker of the graph finishes.

static void* alloc_shared_flags(size_t bytes)
{
    void* p = NULL;
    cudaError_t status = cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (status != cudaSuccess) {
        fprintf(stderr, "communicator allocation failed: %s\n", cudaGetErrorString(status));
        return NULL;
    }
    return p;
}

AlignmentResult gpu_align_no_copy_persistent_kernels(Graph graph, Graph cudaGraph, Sequence sequence)
{
    Sequence sequence_rev;
    int num_levels;
    int* nodes_per_level;
    int* max_node_size_per_level;

    no_copy_prologue(graph, sequence, &sequence_rev, &num_levels, &nodes_per_level, &max_node_size_per_level);

    cudaStream_t compute_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

    // STEP 0: create the communication structs and other mechanisms for that. As default, use
    // NKERNELS 0, which means as many as SMs we have.

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // A worker is handed nodes of any level, so its shared memory has to fit the largest one.
    int max_node_size = 0;
    for (int d = 0; d < num_levels; d++)
        if (max_node_size_per_level[d] > max_node_size) max_node_size = max_node_size_per_level[d];

    int toprow_slots = registers_dp_toprow_slots(max_node_size, sequence.size);
    int elems_per_warp = registers_dp_elems_per_warp(max_node_size, sequence.size);

    int block_threads = WARPS_PER_BLOCK * WARP_SIZE;
    int dynamic_shared_mem_bytes = WARPS_PER_BLOCK * elems_per_warp * sizeof(DTYPEMATRIX);

    if (dynamic_shared_mem_bytes > 48 * 1024) {
        cudaError_t attr_status = cudaFuncSetAttribute(worker, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_mem_bytes);
        if (attr_status != cudaSuccess) {
            fprintf(stderr, "%d bytes of shared memory per block is over what this device allows: %s\n",
                    dynamic_shared_mem_bytes, cudaGetErrorString(attr_status));
        }
    }

    // NKERNELS 0 means "fill the device". We will launch as many blocks as can reside on the GPU
    int num_blocks = NKERNELS;
    if (num_blocks <= 0) {
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, worker, block_threads, dynamic_shared_mem_bytes);

        if (blocks_per_sm < 1) blocks_per_sm = 1;
        num_blocks = prop.multiProcessorCount * blocks_per_sm;
    }

    if (num_blocks * WARPS_PER_BLOCK > graph.num_nodes)
        num_blocks = (graph.num_nodes + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (num_blocks < 1) num_blocks = 1;

    int num_workers = num_blocks * WARPS_PER_BLOCK;

    Communicator** communicators = (Communicator**)alloc_shared_flags(num_workers * sizeof(Communicator*));

    for (int k = 0; k < num_workers; k++) {
        communicators[k] = (Communicator*)alloc_shared_flags(sizeof(Communicator));

        Communicator* comm = communicators[k];

        // Each gets its own cache line, otherwise, we might have false sharing issues.
        comm->done        = (bool*)alloc_shared_flags(CACHELINE);
        comm->work_top    = (int*) alloc_shared_flags(CACHELINE);
        comm->work_bottom = (int*) alloc_shared_flags(CACHELINE);
        comm->workPool    = (int*) alloc_shared_flags(WORKPOOLSIZE * sizeof(int));

        comm->done[0] = false;
        comm->work_top[0] = 0;
        comm->work_bottom[0] = 0;

        for (int w = 0; w < WORKPOOLSIZE; w++)
            comm->workPool[w] = -1;
    }

    double t0 = no_copy_now_ms();

    // STEP 1: Launch kernels

    dim3 gridDim(num_blocks);
    dim3 blockDim(block_threads);

    worker<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
        communicators, cudaGraph.nodes, sequence, sequence_rev, elems_per_warp, toprow_slots);

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Error when launching the persistent kernels: %s\n", cudaGetErrorString(launch_status));
    }

    // STEP 2: Traverse graph & assign work:

    int node_offset = 0; // node ids are global, the ring carries them as such

    for (int l = 0; l < num_levels; l++)
    {
        int k = 0;

        for (int n = 0; n < nodes_per_level[l];)
        {
            volatile int* work_top    = communicators[k]->work_top;
            volatile int* work_bottom = communicators[k]->work_bottom;

            int top = work_top[0];
            int work_next = (top + 1) % WORKPOOLSIZE;

            if (work_next != work_bottom[0]) // this ring has room
            {
                communicators[k]->workPool[top] = node_offset + n;

                // Make sure it's updated before the work pointer reaches the GPU.
                __sync_synchronize();
                work_top[0] = work_next;

                ++n;
            }

            k = (k + 1) % num_workers;
        }

        // Wait for all workers to finish working on the level
        for (int w = 0; w < num_workers; w++) {
            volatile int* work_top    = communicators[w]->work_top;
            volatile int* work_bottom = communicators[w]->work_bottom;

            while (work_bottom[0] != work_top[0]) {
                sched_yield();
            }
        }

        // and there is no STEP 3 here: the columns are already where the traceback wants them, and
        // so are the scores the workers wrote into the shared node structs.

        node_offset += nodes_per_level[l];
    }

    // Every ring is drained at this point, so the workers are all parked in their spin loop and the
    // flag is the only thing they are still waiting for.
    for (int k = 0; k < num_workers; k++) {
        volatile bool* done = communicators[k]->done;
        __sync_synchronize();
        done[0] = true;
    }

    double t_launch = no_copy_now_ms() - t0;

    AlignmentResult res = no_copy_epilogue("persistent_kernels", graph, sequence, sequence_rev,
                                           nodes_per_level, max_node_size_per_level,
                                           compute_stream, t_launch, num_levels, true);

    for (int k = 0; k < num_workers; k++) {
        cudaFreeHost(communicators[k]->done);
        cudaFreeHost(communicators[k]->work_top);
        cudaFreeHost(communicators[k]->work_bottom);
        cudaFreeHost(communicators[k]->workPool);
        cudaFreeHost(communicators[k]);
    }

    cudaFreeHost(communicators);

    return res;
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

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

// The walk back for versions 7 to 10, which left only a last column per node behind. It is the one
// in cpu_last_col.c, the same the CPU and the hybrid versions call: the shared graph means the
// columns the kernels wrote are already the ones it recomputes from.

AlignmentResult compute_traceback_no_copy_last_col(Graph graph, Sequence sequence)
{
    Node* start = &graph.nodes[graph.max_score_node_id];

    return traceback_last_col(graph, sequence.sequence, sequence.size, 0,
                              start, start->max_score_i, start->max_score_j);
}
