#include "../include/cuda_shared_mem.cuh"
#include "../include/cuda_persistent_kernels.cuh"
#include "../include/cuda_dp_shared_mem.cuh"

#include <sched.h>


__global__ void worker(Communicator** communicators, Node* nodes, Sequence sequence, Sequence sequence_rev)
{
    Communicator* comm = communicators[blockIdx.x]; // Worker id, each thread block gets one, so we use the blockId

    volatile int*  work_top    = comm->work_top;
    volatile int*  work_bottom = comm->work_bottom;
    volatile int*  workPool    = comm->workPool;
    volatile bool* done        = comm->done;

    __shared__ int s_node_id; // only thread 0 polls, the rest of the block reads its verdict here

    while (true)
    {
        if (threadIdx.x == 0)
        {
            int bottom = work_bottom[0];
            int node_id = -1;

            while (true)
            {
                if (bottom != work_top[0]) { node_id = workPool[bottom]; break; } // got work
                if (done[0] && bottom == work_top[0]) break;                      // re-read: nothing left

                __nanosleep(BACKOFF_NS); // wait for a bit, no need to saturate the controller with high-latency requests
            }

            s_node_id = node_id;
        }

        __syncthreads();

        int node_id = s_node_id;
        if (node_id < 0) break;

        compute_dp_node_shared_mem(&nodes[node_id], sequence, sequence_rev);

        __syncthreads();

        // Publish the node before publishing the fact that it is finished.
        
        if (threadIdx.x == 0) {
            __threadfence_system();
            work_bottom[0] = (work_bottom[0] + 1) % WORKPOOLSIZE;
        }

        __syncthreads();
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

    int band = band_width_for_level(max_node_size, sequence.size);
    int dynamic_shared_mem_bytes = shared_bytes_for_level(max_node_size, sequence.size, band);

    if (dynamic_shared_mem_bytes > 48 * 1024) {
        cudaError_t attr_status = cudaFuncSetAttribute(worker, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_mem_bytes);
        if (attr_status != cudaSuccess) {
            fprintf(stderr, "%d bytes of shared memory per block is over what this device allows: %s\n",
                    dynamic_shared_mem_bytes, cudaGetErrorString(attr_status));
        }
    }

    // NKERNELS 0 means "fill the device". We will launch as many blocks as can reside on the GPU
    int num_kernels = NKERNELS;
    if (num_kernels <= 0) {
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, worker, band, dynamic_shared_mem_bytes);

        if (blocks_per_sm < 1) blocks_per_sm = 1; // the query says the kernel does not fit; one block is still resident
        num_kernels = prop.multiProcessorCount * blocks_per_sm;
    }

    if (num_kernels > graph.num_nodes) num_kernels = graph.num_nodes; // no point in more workers than nodes

    Communicator** communicators = (Communicator**)alloc_shared_flags(num_kernels * sizeof(Communicator*));

    for (int k = 0; k < num_kernels; k++) {
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

    dim3 gridDim(num_kernels);
    dim3 blockDim(band);

    worker<<<gridDim, blockDim, dynamic_shared_mem_bytes, compute_stream>>>(communicators, cudaGraph.nodes, sequence, sequence_rev);

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

            k = (k + 1) % num_kernels;
        }

        // Wait for all workers to finish working on the level
        for (int w = 0; w < num_kernels; w++) {
            volatile int* work_top    = communicators[w]->work_top;
            volatile int* work_bottom = communicators[w]->work_bottom;

            while (work_bottom[0] != work_top[0]) {
                sched_yield();
            }
        }

        // STEP 3: Copy the dp matrices back to main mem

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

            size_t batch_matrix_elements = 0;
            for (int n = batch_start_offset; n < node_offset; n++)
                batch_matrix_elements += (size_t)(sequence.size + 2) * (graph.nodes[n].sequence.size + 2);

            if (batch_matrix_elements > 0) {
                cudaMemcpyAsync(graph.nodes[batch_start_offset].dp_matrix,
                                device_nodes_pointers[batch_start_offset].dp_matrix,
                                batch_matrix_elements * sizeof(DTYPEMATRIX),
                                cudaMemcpyDeviceToHost,
                                copy_stream);
            }

            batch_start_offset = node_offset;
            levels_in_batch = 0;
        }
    }

    // Every ring is drained at this point, so the workers are all parked in their spin loop and the
    // flag is the only thing they are still waiting for.
    for (int k = 0; k < num_kernels; k++) {
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

    for (int k = 0; k < num_kernels; k++) {
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

    return compute_traceback_gpu_shared_mem(graph, sequence);
}
