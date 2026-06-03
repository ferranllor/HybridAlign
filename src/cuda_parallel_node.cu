#include "../include/cuda_parallel_node.cuh"

AlignmentResult gpu_align_parallel_node(Graph graph, Graph cudaGraph, Sequence sequence)
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

        compute_dp_gpu_parallel_node<<<gridDim, blockDim>>>(act, sequence, sequence_rev);
        
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

    return compute_traceback_gpu_parallel_node(graph, sequence);
}

__global__ void compute_dp_gpu_parallel_node(Node* node, Sequence sequence, Sequence sequence_rev)
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

    for (; d <= l_min; ++d) {
        startPrevPrev = startPrev;
        startPrev = startCurr;
        startCurr = startCurr + d;

        int prev_max = local_max;
        int d_size = d - 1;

        for (int k = 1 + threadIdx.x; k <= d_size; k += blockDim.x) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
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

    if (l_min == N) {
        for (; d <= l_max; ++d) {
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + N + 1;

            int prev_max = local_max;
            int d_size = N;

            for (int k = 1 + threadIdx.x; k <= d_size; k += blockDim.x) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
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
    }
    else {
        for (; d <= l_max; ++d) {
            startPrevPrev = startPrev;
            startPrev = startCurr;
            startCurr = startCurr + M + 1;

            int prev_max = local_max;
            int d_size = M;

            int off_curr = max(0, d - M);
            int off_up   = max(0, d - 1 - M);
            int off_diag = max(0, d - 2 - M);

            for (int k = off_curr + threadIdx.x; k < off_curr + d_size; k += blockDim.x) { 
                int j = k - 1;
                int i = M - d + k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k - 1 - off_diag] + score;
                int up          = dp[startPrev + k - off_up] + GAP;
                int left        = dp[startPrev + k - 1 - off_up] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k - off_curr] = res;

                local_max = max(res, local_max);
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }

            __syncthreads();
        }
    }

    // --------------- Shrink phase ------------------

    for (; d <= (M+N); ++d) {
        startPrevPrev = startPrev;
        startPrev = startCurr;
        startCurr = startCurr + (M + N) - d + 2;

        int prev_max = local_max;
        int d_size = (M + N) - d + 1;

        int off_curr = max(0, d - M);
        int off_up   = max(0, d - 1 - M);
        int off_diag = max(0, d - 2 - M);

        for (int k = off_curr + threadIdx.x; k < off_curr + d_size; k += blockDim.x) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
            int j = k - 1;
            int i = M - d + k;

            int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

            int diagonal    = dp[startPrevPrev + k - 1 - off_diag] + score;
            int up          = dp[startPrev + k - off_up] + GAP;
            int left        = dp[startPrev + k - 1 - off_up] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[startCurr + k - off_curr] = res;

            local_max = max(res, local_max);
        }

        if (prev_max != local_max)
        {
            local_max_d = d;
        }

        __syncthreads();
    }

    // ------------------ Find j of local max ------------------

    __syncthreads();

    int t = threadIdx.x;
    __shared__ int local_max_red[BLOCKSIZE];
    __shared__ int local_max_d_red[BLOCKSIZE];

    local_max_red[t] = local_max;
    local_max_d_red[t] = local_max_d;

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

    __syncthreads();

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

AlignmentResult compute_traceback_gpu_parallel_node(Graph graph, Sequence sequence) {
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