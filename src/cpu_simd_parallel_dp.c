#include "../include/cpu_simd_parallel_dp.h"
#include "../include/cpu_utils.h"

AlignmentResult cpu_align_simd_parallel_dp(Graph graph, Sequence sequence)
{
    compute_dp_cpu_simd_parallel_dp(&graph.nodes[0], sequence);
    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 1; n < graph.num_nodes; n++)
    {
        compute_dp_cpu_simd_parallel_dp(&graph.nodes[n], sequence);

        if (graph.nodes[n].max_score > graph.max_score) { 
            graph.max_score = graph.nodes[n].max_score; 
            graph.max_score_node_id = n;
        }
    }

    return compute_traceback_cpu_simd_parallel_dp(graph, sequence);
}


void compute_dp_cpu_simd_parallel_dp(Node* node, Sequence sequence)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    int M = sequence.size;
    int N = node->sequence.size;
    DTYPEMATRIX* __restrict dp = node->dp_matrix;
    
    if (node->num_in == 0) {
        for (int i = 0; i <= M; ++i) dp[get_diagonal_index(i, 0, M, N)] = 0;
        for (int j = 1; j <= N; ++j) dp[get_diagonal_index(0, j, M, N)] = 0;
    }
    else if (node->num_in == 1) {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;
        int prev_N = node->v_in[0]->sequence.size;

        for (int i = 0; i <= M; ++i) {
            dp[get_diagonal_index(i, 0, M, N)] = prev_dp[get_diagonal_index(i, prev_N, M, prev_N)];
        }
        for (int j = 1; j <= N; ++j) dp[get_diagonal_index(0, j, M, N)] = 0;
    }
    else {
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 
        int prev_N1 = node->v_in[0]->sequence.size;
        int prev_N2 = node->v_in[1]->sequence.size;

        for (int i = 0; i <= M; ++i) {
            int score1 = prev_dp[get_diagonal_index(i, prev_N1, M, prev_N1)];
            int score2 = prev_dp2[get_diagonal_index(i, prev_N2, M, prev_N2)];
            dp[get_diagonal_index(i, 0, M, N)] = max(score1, score2);
        }

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            int prev_Ni = node->v_in[i]->sequence.size;
            for (int j = 0; j <= M; ++j) {
                int act = get_diagonal_index(j, 0, M, N);
                dp[act] = max(prev_dp[get_diagonal_index(j, prev_Ni, M, prev_Ni)], dp[act]);
            }
        }
        for (int j = 1; j <= N; ++j) dp[get_diagonal_index(0, j, M, N)] = 0;
    }

    // ------------------------------------------------- Compute -------------------------------------------------
    
    char* __restrict node_seq = node->sequence.sequence;
    char* __restrict query_seq = sequence.sequence;

    char* __restrict query_seq_rev = (char*)malloc(M * sizeof(char));
    for (int idx = 0; idx < M; ++idx) {
        query_seq_rev[idx] = query_seq[M - 1 - idx];
    }

    int n_threads = 2; // Even with 1 its 2x worse just because of openMP overhead :(

    int* local_max_shared = (int*)malloc(sizeof(int) * n_threads);
    int* local_max_d_shared = (int*)malloc(sizeof(int) * n_threads);
    int local_max_j = -1;

    #pragma omp parallel num_threads(n_threads)
    {
        int t = omp_get_thread_num();

        int local_max = -1;
        int local_max_d = -1;

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

            #pragma omp for 
            for (int k = 1; k <= d_size; ++k) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
                int j = k - 1;
                int i = M - d + k;
                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k - 1] + score;
                int up          = dp[startPrev + k] + GAP;
                int left        = dp[startPrev + k - 1] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k] = res;

                if (res > local_max) {
                    local_max = res;
                }
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }
        }

        // --------------- Stable phase ------------------

        if (l_min == N) {

                for (; d <= l_max; ++d) {
                    startPrevPrev = startPrev;
                    startPrev = startCurr;
                    startCurr = startCurr + N + 1;

                    int prev_max = local_max;
                    int d_size = N;

                    #pragma omp for
                    for (int k = 1; k <= d_size; ++k) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
                        int j = k - 1;
                        int i = M - d + k;

                        int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                        int diagonal    = dp[startPrevPrev + k - 1] + score;
                        int up          = dp[startPrev + k] + GAP;
                        int left        = dp[startPrev + k - 1] + GAP;

                        int res = max(max(diagonal, 0), max(up, left));
                        dp[startCurr + k] = res;

                        if (res > local_max) {
                            local_max = res;
                        }
                    }

                    if (prev_max != local_max)
                    {
                        local_max_d = d;
                    }
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

                #pragma omp for
                for (int k = off_curr; k < off_curr + d_size; ++k) { 
                    int j = k - 1;
                    int i = M - d + k;

                    int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                    int diagonal    = dp[startPrevPrev + k - 1 - off_diag] + score;
                    int up          = dp[startPrev + k - off_up] + GAP;
                    int left        = dp[startPrev + k - 1 - off_up] + GAP;

                    int res = max(max(diagonal, 0), max(up, left));
                    dp[startCurr + k - off_curr] = res;

                    if (res > local_max) {
                        local_max = res;
                    }
                }

                if (prev_max != local_max)
                {
                    local_max_d = d;
                }
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

            #pragma omp for
            for (int k = off_curr; k < off_curr + d_size; ++k) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
                int j = k - 1;
                int i = M - d + k;

                int score = (node_seq[j] == query_seq_rev[i]) ? MATCH : MISMATCH;

                int diagonal    = dp[startPrevPrev + k - 1 - off_diag] + score;
                int up          = dp[startPrev + k - off_up] + GAP;
                int left        = dp[startPrev + k - 1 - off_up] + GAP;

                int res = max(max(diagonal, 0), max(up, left));
                dp[startCurr + k - off_curr] = res;

                if (res > local_max) {
                    local_max = res;
                }
            }

            if (prev_max != local_max)
            {
                local_max_d = d;
            }
        }

        local_max_shared[t] = local_max;
        local_max_d_shared[t] = local_max_d;
    }

    int local_max = -1;
    int local_max_d = -1;

    for (int i = 0; i < n_threads; i++)
    {
        if (local_max_shared[i] > local_max)
        {
            local_max = local_max_shared[i];
            local_max_d = local_max_d_shared[i];
        }
    }

    free(query_seq_rev);

    // ------------------ Find j of local max ------------------

    if (local_max_d != -1) {
        
        int final_d = local_max_d;
        int j_start = (local_max_d - M > 1) ? local_max_d - M : 1;
        int j_end = (local_max_d - 1 < N) ? local_max_d - 1 : N;

        int startCurr = get_diag_start(final_d, M, N);

        int off_curr = max(0, final_d - M); 

        for (int j = j_start; j <= j_end; j++) {
            if (dp[startCurr + j - off_curr] == local_max) {
                local_max_j = j;
                break;
            }
        }
    }

    node->max_score = local_max;
    node->max_score_d = local_max_d;
    node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
    node->max_score_j = local_max_j;
}

AlignmentResult compute_traceback_cpu_simd_parallel_dp(Graph graph, Sequence sequence) {
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