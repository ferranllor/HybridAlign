#include "../include/cuda_persistent_kernels.cuh"
#include "../include/cuda_registers.cuh"
#include "../include/cuda_dp_registers.cuh"

#include "../include/hybrid_utils.cuh"

#include <sched.h>

// I didn't mention this version because it was a failed experiment, I could not get it to perform well 
// in the time I had, so it's more of a leftover than anything...

// The persistent workers on the registers core. A worker used to be a thread block of band_width
// threads filling a whole dp matrix; it is a warp now, doing one node with the three diagonals in
// registers and leaving only its last column behind. So a block of WARPS_PER_BLOCK warps is
// WARPS_PER_BLOCK independent workers, each with its own ring, and the block barriers the old
// worker needed are gone with them: lane 0 polls the ring and the shuffles are the rest of the
// synchronisation.

__global__ void worker(Communicator** communicators, Node* nodes, Sequence sequence,
                       Sequence sequence_rev, int elems_per_warp, int toprow_slots)
{
    const unsigned int mask = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;

    int worker_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_in_block; // Worker id, each warp gets one

    Communicator* comm = communicators[worker_id];

    volatile int*  work_top    = comm->work_top;
    volatile int*  work_bottom = comm->work_bottom;
    volatile int*  workPool    = comm->workPool;
    volatile bool* done        = comm->done;

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* mine = &shared[warp_in_block * elems_per_warp];

    while (true)
    {
        int node_id = -1;

        if (lane == 0)
        {
            int bottom = work_bottom[0];

            while (true)
            {
                if (bottom != work_top[0]) { node_id = workPool[bottom]; break; } // got work
                if (done[0] && bottom == work_top[0]) break;                      // re-read: nothing left

                __nanosleep(BACKOFF_NS); // wait for a bit, no need to saturate the controller with high-latency requests
            }
        }

        node_id = __shfl_sync(mask, node_id, 0); // only lane 0 polls, the rest of the warp reads its verdict here

        if (node_id < 0) break;

        int local_max, local_max_d, local_max_j;

        compute_dp_node_registers(&nodes[node_id], sequence.size, sequence_rev.sequence, 0,
                                  mine, toprow_slots, local_max, local_max_d, local_max_j);

        __syncwarp(mask);

        // Publish the node before publishing the fact that it is finished.

        if (lane == 0) {
            Node* node = &nodes[node_id];

            node->max_score = local_max;
            node->max_score_d = local_max_d;
            node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
            node->max_score_j = (local_max_d != -1) ? local_max_j : -1;

            __threadfence_system();
            work_bottom[0] = (work_bottom[0] + 1) % WORKPOOLSIZE;
        }

        __syncwarp(mask);
    }

    return;
}

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

AlignmentResult gpu_align_persistent_kernels(Graph graph, Graph cudaGraph, Sequence sequence)
{
    Sequence sequence_rev;
    Sequence tmp;

    tmp.sequence = (char*)malloc(sequence.size * sizeof(char));
    for (int idx = 0; idx < sequence.size; ++idx) { // The sequence has to be reversed first, since we're reading it in reverse inside the kernel
        tmp.sequence[idx] = sequence.sequence[sequence.size - 1 - idx];
    }

    sequence_rev.size = sequence.size;

    cudaMalloc((void**)&sequence_rev.sequence, sequence.size * sizeof(char));
    cudaMemcpy(sequence_rev.sequence, tmp.sequence, sequence.size * sizeof(char), cudaMemcpyHostToDevice);

    int num_levels = graph.nodes[graph.num_nodes-1].depth + 1;
    int* nodes_per_level = (int*)calloc(num_levels, sizeof(int));

    for (int n = 0; n < graph.num_nodes; n++) {
        int depth = graph.nodes[n].depth;
        nodes_per_level[depth]++; // Set up counts of nodes per level for the scheduler later
    }

    // Set up a copy and compute stream, we we may want to sync the copy stream, and obiously
    // persistent kernels will not end until all work is done.
    cudaStream_t compute_stream, copy_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);

    // STEP 0: create the communication structs and other mechanisms for that. As default, use NKERNELS 0, which means as many as SMs we have. Any more could be useful, but needs exploration, may be done in future implmentations

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // We need to get tha max node size if we want to know the max amount of shared mem we will use
    int max_node_size = 0;
    for (int n = 0; n < graph.num_nodes; n++)
        if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

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

        if (blocks_per_sm < 1) blocks_per_sm = 1; // the query says the kernel does not fit; one block is still resident
        num_blocks = prop.multiProcessorCount * blocks_per_sm;
    }

    // no point in more workers than nodes, but a block is WARPS_PER_BLOCK of them and cannot be split
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

    Node* device_nodes_pointers = (Node*)malloc(graph.num_nodes * sizeof(Node));
    cudaMemcpy(device_nodes_pointers, cudaGraph.nodes, graph.num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    Node* device_nodes_tmp = NULL;
    cudaMallocHost((void**)&device_nodes_tmp, graph.num_nodes * sizeof(Node));

    // STEP 1: Launch kernels

    dim3 gridDim(num_blocks);
    dim3 blockDim(block_threads);

    worker<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(
        communicators, cudaGraph.nodes, sequence, sequence_rev, elems_per_warp, toprow_slots);

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        fprintf(stderr, "Error when launching the persistent kernels: %s\n", cudaGetErrorString(launch_status));
    }

    cudaError_t status;

    const int BATCH_LEVELS = 8; // smaller = more overlap, larger = fewer commands

    int batch_start_offset = 0;
    int levels_in_batch = 0;

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

        // STEP 3: Copy the last columns back to main mem

        node_offset += nodes_per_level[l];
        levels_in_batch++;

        bool last_level = (l == num_levels - 1);
        bool batch_full = (levels_in_batch >= BATCH_LEVELS);

        if (batch_full || last_level) {
            int batch_num_nodes = node_offset - batch_start_offset;

            cudaMemcpyAsync(&device_nodes_tmp[batch_start_offset],
                            &cudaGraph.nodes[batch_start_offset],
                            (size_t)batch_num_nodes * sizeof(Node),
                            cudaMemcpyDeviceToHost,
                            copy_stream);

            cudaMemcpyAsync(graph.nodes[batch_start_offset].last_col,
                            device_nodes_pointers[batch_start_offset].last_col,
                            (size_t)batch_num_nodes * (sequence.size + 1) * sizeof(DTYPEMATRIX),
                            cudaMemcpyDeviceToHost,
                            copy_stream);

            batch_start_offset = node_offset;
            levels_in_batch = 0;
        }
    }

    // Every ring is drained at this point, so the workers are all parked in their spin loop and the
    // flag is the only thing they are still waiting for.
    for (int k = 0; k < num_workers; k++) {
        volatile bool* done = communicators[k]->done;
        __sync_synchronize(); // Claude did this. Honestly, I don't get it, but it works so that's ok i guess
        done[0] = true;
    }

    // LAST STEP: Find max score, free struff, and compute traceback.

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

    for (int k = 0; k < num_workers; k++) {
        cudaFreeHost(communicators[k]->done);
        cudaFreeHost(communicators[k]->work_top);
        cudaFreeHost(communicators[k]->work_bottom);
        cudaFreeHost(communicators[k]->workPool);
        cudaFreeHost(communicators[k]);
    }

    cudaFreeHost(communicators);

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    free(nodes_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);
    cudaFreeHost(device_nodes_tmp);
    free(device_nodes_pointers);

    return compute_traceback_gpu_registers(graph, sequence);
}
