#include "../include/cuda_parallel_async.cuh"

AlignmentResult gpu_align_parallel_async(Graph graph, Graph cudaGraph, Sequence sequence)
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

    Node* device_nodes_tmp = (Node*)malloc(graph.num_nodes * sizeof(Node));

    cudaError_t status;
    Node* act = cudaGraph.nodes;
    int node_offset = 0;

    for (int d = 0; d < num_levels; d++)
    {
        dim3 gridDim(nodes_per_level[d]);
        dim3 blockDim(BLOCKSIZE);

        compute_dp_gpu_parallel_async<<<gridDim, blockDim, 0, compute_stream>>>(act, sequence, sequence_rev);
        
        cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(launch_status));
        }

        cudaEvent_t compute_done;
        cudaEventCreate(&compute_done);
        cudaEventRecord(compute_done, compute_stream);

        cudaStreamWaitEvent(copy_stream, compute_done, 0);

        size_t level_nodes_size = nodes_per_level[d] * sizeof(Node);
        cudaMemcpyAsync(&device_nodes_tmp[node_offset], act, level_nodes_size, cudaMemcpyDeviceToHost, copy_stream);

        for (int i = 0; i < nodes_per_level[d]; i++) {
            int n = node_offset + i;
            size_t matrix_size = (sequence.size + 2) * (graph.nodes[n].sequence.size + 2);
            
            cudaMemcpyAsync(graph.nodes[n].dp_matrix, device_nodes_pointers[n].dp_matrix, 
                            matrix_size * sizeof(DTYPEMATRIX), cudaMemcpyDeviceToHost, copy_stream);
        }

        cudaEventDestroy(compute_done);

        act = &act[nodes_per_level[d]];
        node_offset += nodes_per_level[d];
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

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    free(nodes_per_level);
    cudaFree(sequence_rev.sequence);
    free(tmp.sequence);
    free(device_nodes_tmp);
    free(device_nodes_pointers);

    return compute_traceback_gpu_parallel_async(graph, sequence);
}

__global__ void compute_dp_gpu_parallel_async(Node* node, Sequence sequence, Sequence sequence_rev)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    node = &node[blockIdx.x];
    
    int M = sequence.size;
    int N = node->sequence.size;
    DTYPEMATRIX* __restrict dp = node->dp_matrix;
    
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

    // ------------------------------------------------- Compute -------------------------------------------------
    
    char* __restrict node_seq = node->sequence.sequence;
    char* __restrict query_seq_rev = sequence_rev.sequence;

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    int l_min = (M < N) ? M : N;
    int l_max = (M > N) ? M : N;

    int startCurr = 1;
    int startPrev = 0;
    int startPrevPrev;

    int d = 2;

    // --------------- Grow phase ------------------

    for (; d < l_min + 2 - 1; ++d) { // +2 because starting d offset, -1 because grow is of size l_min - 1
        startPrevPrev = startPrev;
        startPrev = startCurr;
        startCurr = startCurr + d;

        int prev_max = local_max;
        int d_size = d + 1; // D_size is always the size of the current diagonal, and takes into account halo elements. Grow phase always has first row and columns, so +1, as d is always d_size -1 (so -1 + 2 = 1)

        for (int k = 1 + threadIdx.x; k < d_size - 1; k += blockDim.x) { // +1 for first column. -1 for first row
            int j = k - 1;
            int i = M - d + k;

            int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

            int diagonal    = dp[startPrevPrev + k - 1] + score;
            int up          = dp[startPrev + k] + GAP;
            int left        = dp[startPrev + k - 1] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[startCurr + k] = res;

            local_max = max(res, local_max);
        }

        if (prev_max != local_max)
        {
            local_max_d = d;
        }

        __syncthreads();
    }

    // --------------- Stable phase ------------------

    int offset_col1 = (l_min == N); // If l_min = N, means we will always have the first col in our diagonal, elements are redundant so we start at index 1 to avoid them
    int offset_row1 = (l_min == M); // Same thing, but with the first row, so we ignore the last element on the diagonal

    if (N >= M) {
        // Whiever is reading this, ignore this block, its just a matter of transitioning to a different way of indexing, because stuff is 
        // not in memory as it should be for the math to be pretty. This still counts as stable phase for all intents and purposes.
        // you will see that when the max is M this phase starts later. This happens because this shift is linked to when the first 
        // column stops being there. Maybe you should look for a different way that is more "consistent"? Idk, things like this (chapuzas) make me think
        // that I'm looking at the problem the wrong way... In any case, for now, it works, so everything is good.
        {
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + l_min + 1; // Offset of 1, since we will always have elements from top row or first column during stable phase

            int prev_max = local_max;
            int d_size = l_min + 1; // +1 for one of the two offsets, whichever applies

            for (int k = offset_col1 + threadIdx.x; k < d_size - offset_row1; k += blockDim.x) { 
                int j = d - M + k - offset_col1;
                int i = k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k] + score;
                int up          = dp[startPrev + k + 1] + GAP;
                int left        = dp[startPrev + k] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();

            ++d;
        }

        for (; d < l_max + 2 - 1; ++d) { // +2 because starting d offset, -1 because grow is of size l_min - 1, and stable is of size l_max - l_min elements, so (l_max - l_min) + l_min - 1 = l_max - 1
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + l_min + 1; // Offset of 1, since we will always have elements from top row or first column during stable phase

            int prev_max = local_max;
            int d_size = l_min + 1; // +1 for one of the two offsets, whichever applies

            for (int k = offset_col1 + threadIdx.x; k < d_size - offset_row1; k += blockDim.x) { 
                int j = d - M + k - offset_col1;
                int i = k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k + 1] + score;
                int up          = dp[startPrev + k + 1] + GAP;
                int left        = dp[startPrev + k] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();
        }

        // --------------- Shrink phase -----------------

        for (; d < M + N + 2 - 1; ++d) { // +2 because of offset, l_min is of size l_min - 1, and stable of l_max - l_min. Shrink is of l_min, so that gives 2 + (l_min - 1) + (l_max - l_min) + l_min = 2 - 1 + l_max + l_min or M + N + 2 - 1
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + (M + N) - d + 2;

            int prev_max = local_max;
            int d_size = (M + N) - d + 1;
            
            for (int k = threadIdx.x; k < d_size; k += blockDim.x) {
                int j = d - M + k;
                int i = k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k + 1] + score;
                int up          = dp[startPrev + k + 1] + GAP;
                int left        = dp[startPrev + k] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();
        }
    }
    else
    {
        // --------------- Stable phase ------------------
        
        for (; d < l_max + 2 - 1; ++d) { // +2 because starting d offset, -1 because grow is of size l_min - 1, and stable is of size l_max - l_min elements, so (l_max - l_min) + l_min - 1 = l_max - 1
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + l_min + 1; // Offset of 1, since we will always have elements from top row or first column during stable phase

            int prev_max = local_max;
            int d_size = l_min + 1; // +1 for one of the two offsets, whichever applies

            for (int k = offset_col1 + threadIdx.x; k < d_size - offset_row1; k += blockDim.x) { 
                int j = k - offset_col1;
                int i = M - d + k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k - 1] + score;
                int up          = dp[startPrev + k] + GAP;
                int left        = dp[startPrev + k - 1] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();
        }

        // --------------- Shrink phase -----------------

        // You can find and explanation for this thing on the if branch on top, so do that if you're wandering what this is :)
        {
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + (M + N) - d + 2;

            int prev_max = local_max;
            int d_size = (M + N) - d + 1;
            
            for (int k = threadIdx.x; k < d_size; k += blockDim.x) {
                int j = d - M - 1 + k;
                int i = k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k] + score;
                int up          = dp[startPrev + k + 1] + GAP;
                int left        = dp[startPrev + k] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();

            ++d;
        }

        for (; d < M + N + 2 - 1; ++d) { // +2 because of offset, l_min is of size l_min - 1, and stable of l_max - l_min. Shrink is of l_min, so that gives 2 + (l_min - 1) + (l_max - l_min) + l_min = 2 - 1 + l_max + l_min or M + N + 2 - 1
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + (M + N) - d + 2;

            int prev_max = local_max;
            int d_size = (M + N) - d + 1;
            
            for (int k = threadIdx.x; k < d_size; k += blockDim.x) {
                int j = d - M - 1 + k;
                int i = k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k + 1] + score;
                int up          = dp[startPrev + k + 1] + GAP;
                int left        = dp[startPrev + k] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();
        }
    }

       
    // ------------------ Find j of local max ------------------

    __syncthreads();

    int t = threadIdx.x;
    __shared__ int local_max_red[BLOCKSIZE];
    __shared__ int local_max_d_red[BLOCKSIZE];

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

        startCurr = get_diag_start_device(final_d, M, N);

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

AlignmentResult compute_traceback_gpu_parallel_async(Graph graph, Sequence sequence) {
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