#include "../include/cuda_shared_mem.cuh"

AlignmentResult gpu_align_shared_mem(Graph graph, Graph cudaGraph, Sequence sequence)
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

    cudaError_t status;
    Node* act = cudaGraph.nodes;

    for (int d = 0; d < num_levels; d++)
    {
        dim3 gridDim(nodes_per_level[d]);
        dim3 blockDim(BLOCKSIZE);
        
        int dynamic_shared_mem_bytes = sizeof(DTYPEMATRIX) * (BLOCKSIZE + BLOCKSIZE + graph.nodes[d].sequence.size + graph.nodes[d].sequence.size + sequence.size + 3 + ((BLOCKSIZE + 2) * 3));

        compute_dp_gpu_shared_mem<<<gridDim, blockDim, dynamic_shared_mem_bytes>>>(act, sequence, sequence_rev);
        
        cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
        }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(status));
        }

        act = &act[nodes_per_level[d]];
    }

    /*
    int mean_nodes_per_level = 0;

    for (int d = 0; d < num_levels; d++) {
        mean_nodes_per_level += nodes_per_level[d];
    }

    printf("Mean nodes per level: %f\n", (double)mean_nodes_per_level/(double)num_levels);
    printf("Num nodes: %d\n", graph.num_nodes);

    printf("Num nodes level 2: %d\n", nodes_per_level[2]);
    printf("Num nodes level 4: %d\n", nodes_per_level[4]);
    printf("Num nodes level 8: %d\n", nodes_per_level[8]);
    printf("Num nodes level 16: %d\n", nodes_per_level[16]);
    printf("Num nodes level 32: %d\n", nodes_per_level[32]);
    printf("Num nodes level 64: %d\n", nodes_per_level[64]);
    */

    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;
    for (int n = 1; n < graph.num_nodes; n++)
    {
        if (graph.nodes[n].max_score > graph.max_score) { 
            graph.max_score = graph.nodes[n].max_score; 
            graph.max_score_node_id = n;
        }
    }

    free(nodes_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);

    Node* device_nodes_scratch = (Node*)malloc(graph.num_nodes * sizeof(Node));
    cudaMemcpy(device_nodes_scratch, cudaGraph.nodes, graph.num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    graph.max_score = device_nodes_scratch[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        graph.nodes[n].max_score   = device_nodes_scratch[n].max_score;
        graph.nodes[n].max_score_d = device_nodes_scratch[n].max_score_d;
        graph.nodes[n].max_score_i = device_nodes_scratch[n].max_score_i;
        graph.nodes[n].max_score_j = device_nodes_scratch[n].max_score_j;
        
        if (device_nodes_scratch[n].max_score > graph.max_score) { 
            graph.max_score = device_nodes_scratch[n].max_score; 
            graph.max_score_node_id = n;
        }

        size_t matrix_size = (sequence.size + 2) * (graph.nodes[n].sequence.size + 2);
        cudaMemcpy(graph.nodes[n].dp_matrix, device_nodes_scratch[n].dp_matrix, 
                   matrix_size * sizeof(DTYPEMATRIX), cudaMemcpyDeviceToHost);
    }

    // Clean up local tracking structures
    free(device_nodes_scratch);

    return compute_traceback_gpu_shared_mem(graph, sequence);
}

__global__ void compute_dp_gpu_shared_mem(Node* node, Sequence sequence, Sequence sequence_rev)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    node = &node[blockIdx.x];
    
    int M = sequence.size;
    int N = node->sequence.size;
    DTYPEMATRIX* __restrict dp = node->dp_matrix;

    const int nBuffers = 3;

    __shared__ DTYPEMATRIX dpBuffers[nBuffers][BLOCKSIZE+2];

    extern __shared__ DTYPEMATRIX shared[];

    DTYPEMATRIX* topRow = shared;
    DTYPEMATRIX* prevCol = &(shared[N+1]);

    //if (threadIdx.x == 0) printf("Holis\n");

    for (int j = threadIdx.x; j <= N; j += blockDim.x) topRow[j] = 0;

    if (node->num_in == 0) {
        for (int i = threadIdx.x; i <= M; i += blockDim.x) 
            prevCol[i] = 0;
    }
    else if (node->num_in == 1) {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;
        int prev_N = node->v_in[0]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            prevCol[i] = prev_dp[get_diagonal_index_device(i, prev_N, M, prev_N)];
    }
    else {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 
        int prev_N1 = node->v_in[0]->sequence.size;
        int prev_N2 = node->v_in[1]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x) {
            int score1 = prev_dp[get_diagonal_index_device(i, prev_N1, M, prev_N1)];
            int score2 = prev_dp2[get_diagonal_index_device(i, prev_N2, M, prev_N2)];
            prevCol[i] = max(score1, score2);
        }

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            int prev_Ni = node->v_in[i]->sequence.size;
            for (int j = threadIdx.x; j <= M; j += blockDim.x) {
                int act = get_diagonal_index_device(j, 0, M, N);
                prevCol[j] = max(prev_dp[get_diagonal_index_device(j, prev_Ni, M, prev_Ni)], prevCol[j]);
            }
        }
    }
    
    if (node->num_in == 0) {
        for (int i = threadIdx.x; i <= M; i += blockDim.x) 
            dp[get_diagonal_index_device(i, 0, M, N)] = 0;
        
        for (int j = threadIdx.x + 1; j <= N; j += blockDim.x) 
            dp[get_diagonal_index_device(0, j, M, N)] = 0;
    }
    else if (node->num_in == 1) {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;
        int prev_N = node->v_in[0]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x)
            dp[get_diagonal_index_device(i, 0, M, N)] = prev_dp[get_diagonal_index_device(i, prev_N, M, prev_N)];
        
        for (int j = threadIdx.x + 1; j <= N; j += blockDim.x) 
            dp[get_diagonal_index_device(0, j, M, N)] = 0;
    }
    else {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 
        int prev_N1 = node->v_in[0]->sequence.size;
        int prev_N2 = node->v_in[1]->sequence.size;

        for (int i = threadIdx.x; i <= M; i += blockDim.x) {
            int score1 = prev_dp[get_diagonal_index_device(i, prev_N1, M, prev_N1)];
            int score2 = prev_dp2[get_diagonal_index_device(i, prev_N2, M, prev_N2)];
            dp[get_diagonal_index_device(i, 0, M, N)] = max(score1, score2);
        }

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            int prev_Ni = node->v_in[i]->sequence.size;
            for (int j = threadIdx.x; j <= M; j += blockDim.x) {
                int act = get_diagonal_index_device(j, 0, M, N);
                dp[act] = max(prev_dp[get_diagonal_index_device(j, prev_Ni, M, prev_Ni)], dp[act]);
            }
        }
        for (int j = threadIdx.x + 1; j <= N; j += blockDim.x) dp[get_diagonal_index_device(0, j, M, N)] = 0;
    }

    char* __restrict node_seq = node->sequence.sequence; // TODO: move this into shared memory.
    char* __restrict query_seq_rev = sequence_rev.sequence;

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    int l_min = (M < N) ? M : N;
    int l_max = (M > N) ? M : N;  

    int startCurr, startPrev;
    int bufferAct, bufferPrev, bufferPrevPrev;
    int k_start;

    auto process_stripe_small = [&](int blockStart, int blockEnd, int current_d,
                            int j_offset, int i_offset, 
                            int off_diag, int off_up, int off_left) {
        int prev_max = local_max;

        // 1. Compute Phase
        for (int k = blockStart + threadIdx.x; k < blockEnd; k += blockDim.x) {
            int j = j_offset + k;
            int i = i_offset + k;

            int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

            int diagonal = dpBuffers[bufferPrevPrev][k + off_diag] + score;
            int up       = dpBuffers[bufferPrev][k + off_up] + GAP;
            int left     = dpBuffers[bufferPrev][k + off_left] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dpBuffers[bufferAct][k] = res;

            local_max = max(res, local_max);
        }

        if (prev_max != local_max) {
            local_max_d = current_d;
        }

        // 2. Async write to global memory
        __syncthreads();

        for (int k = blockStart + threadIdx.x; k < blockEnd; k += blockDim.x)
            dp[startCurr + k_start + k] = dpBuffers[bufferAct][k];
        
        __syncthreads(); 

        // 3. Buffer Rotation
        bufferAct      = (bufferAct + 1) % N_BUFFERS;
        bufferPrev     = (bufferPrev + 1) % N_BUFFERS;
        bufferPrevPrev = (bufferPrevPrev + 1) % N_BUFFERS;
    };

    __syncthreads();

    for (int startM = 2; startM < M; startM += BLOCKSIZE) {
        startCurr = get_diag_start_device(startM - 1, M, N); // starts at 1
        startPrev = get_diag_start_device(startM - 2, M, N); // starts at 0

        int stripe_height = min(BLOCKSIZE, M - startM + 2);

        int d = startM; // StartM corresponds exactly to the diagonal where we want to start. This is the reason we start at 2, to offset halo values

        k_start = 0;

        bufferAct = 2, bufferPrev = 1, bufferPrevPrev = 0;
        
        if (threadIdx.x == 0) {
            dpBuffers[bufferPrevPrev][0] = dp[startPrev];
            dpBuffers[bufferPrev][1] = dp[startCurr + 1];
            dpBuffers[bufferPrev][0] = dp[startCurr];
        }

        __syncthreads();

        // --------------- Grow phase ------------------

        for (; d < l_min + 2 - 1; ++d) {
            startPrev = startCurr;
            startCurr = startCurr + d;

            int d_size = d + 1;
            int blockStart = 1 + k_start;
            int blockEnd = min(blockStart + stripe_height, d_size - 1);

            if (threadIdx.x == 0) {
                dpBuffers[bufferAct][0] = prevCol[d];
                dpBuffers[bufferAct][blockEnd] = topRow[d - 2];
            }
            __syncthreads();

            process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        k_start - 1, k_start + M - d, 
                        -1, 0, -1);

            if (d > (startM - 2 + stripe_height)) k_start++;

            if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];
        }

        int offset_col1 = (l_min == N);
        int offset_row1 = (l_min == M);

        if (N >= M) {

            // --------------- Stable phase ------------------
            
            int d_size = l_min + 1;
            int blockStart = offset_col1 + k_start;
            int blockEnd = min(blockStart + stripe_height, d_size - offset_row1);

            if (d < l_min + 2)
            {
                startPrev = startCurr;
                startCurr = startCurr + l_min + 1;

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][blockEnd] = topRow[d - 1];
                }

                __syncthreads();

                process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        k_start + d - M - offset_col1, k_start, 
                        0, 1, 0);

                if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];
                
                ++d;
            }

            for (; d < l_max + 2 - 1; ++d) {
                startPrev = startCurr;
                startCurr = startCurr + l_min + 1;

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][blockEnd] = topRow[d - 1];
                }

                __syncthreads();

                process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        k_start + d - M - offset_col1, k_start, 
                        1, 1, 0);
                
                if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];
            }

            // --------------- Shrink phase -----------------

            blockStart = k_start;

            for (; d < startM + stripe_height + N + 2 - 2; ++d) { 
                startPrev = startCurr;
                startCurr = startCurr + (M + N) - d + 2;

                d_size = (M + N) - d + 1;
                blockEnd = min(blockStart + stripe_height, d_size);
                
                process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        k_start + d - M, k_start, 
                        1, 1, 0);

                if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];
            }
        }
        else {
            // --------------- Stable phase ------------------

            int d_size = l_min + 1;

            for (; (d < l_max + 2 - 1) && (d < startM + N + stripe_height + 1); ++d) {
                startPrev = startCurr;
                startCurr = startCurr + l_min + 1;

                int blockStart = offset_col1 + k_start;
                int blockEnd = min(blockStart + stripe_height, d_size - offset_row1);

                if (threadIdx.x == 0) {
                    dpBuffers[bufferAct][0] = prevCol[d];
                }

                __syncthreads();

                if (threadIdx.x == 0 && (blockStart < 0 || ((blockEnd - blockStart) > blockDim.x))) printf("blockStart: %d, blockEnd: %d, d_size %d, stripe_height %d\n", blockStart, blockEnd, d_size, stripe_height);

                process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        k_start - offset_col1, k_start + M - d, 
                        -1, 0, -1);
                
                if (d > (startM - 2 + stripe_height)) k_start++;
                if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];
            }

            // --------------- Shrink phase -----------------

            int blockStart = k_start;

            if ((d < l_max + 2) && (d < startM + N + stripe_height + 1))
            {
                startPrev = startCurr;
                startCurr = startCurr + (M + N) - d + 2;

                d_size = (M + N) - d + 1;
                int blockEnd = min(blockStart + stripe_height, d_size);

                process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        d - M - 1 + k_start, k_start, 
                        0, 1, 0);
                
                if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];

                ++d;
            }

            for (; d < (M + N + 2 - 1) && (d < startM + N + stripe_height + 1); ++d) {
                startPrev = startCurr;
                startCurr = startCurr + (M + N) - d + 2;

                d_size = (M + N) - d + 1;
                int blockEnd = min(blockStart + stripe_height, d_size);

                process_stripe_small(blockStart - k_start, blockEnd - k_start, d, 
                        k_start + d - M - 1, k_start, 
                        1, 1, 0);

                if (threadIdx.x == 0) topRow[d - (startM + stripe_height - 2)] = dpBuffers[bufferPrev][0];
            }
        }
    }

    // ------------------ Find j of local max ------------------

    __syncthreads();

    int t = threadIdx.x;
    __shared__ DTYPEMATRIX local_max_red[BLOCKSIZE];
    __shared__ DTYPEMATRIX local_max_d_red[BLOCKSIZE];

    local_max_red[t] = local_max;
    local_max_d_red[t] = local_max_d;

    __syncthreads(); 

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            int curr_max = local_max_red[t];
            int candidate = local_max_red[t + stride];
            
            int curr_d = local_max_d_red[t];
            int candidate_d = local_max_d_red[t + stride];

            bool is_greater = (candidate > curr_max);
            
            local_max_red[t]  = is_greater ? candidate : curr_max;
            local_max_d_red[t] = is_greater ? candidate_d : curr_d; 
        }
        __syncthreads();
    }

    local_max = local_max_red[0];
    local_max_d = local_max_d_red[0];

    if (local_max_d != -1) {

        __shared__ int shared_min_j;
        if (threadIdx.x == 0) {
            shared_min_j = INT_MAX; 
        }

        __syncthreads();
        
        int final_d = local_max_d;
        int j_start = (local_max_d - M > 1) ? local_max_d - M : 1;
        int j_end = (local_max_d - 1 < N) ? local_max_d - 1 : N;

        int startCurr = get_diag_start_device(final_d, M, N);

        int off_curr = max(0, final_d - M);

        for (int j = j_start + threadIdx.x; j <= j_end; j += blockDim.x) {
            if (dp[startCurr + j - off_curr] == local_max) {
                atomicMin(&shared_min_j, j);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0)
            local_max_j = shared_min_j;
    }

    if (threadIdx.x == 0) {
        node->max_score = local_max;
        node->max_score_d = local_max_d;
        node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
        node->max_score_j = local_max_j;
    }

}

AlignmentResult compute_traceback_gpu_shared_mem(Graph graph, Sequence sequence) {
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