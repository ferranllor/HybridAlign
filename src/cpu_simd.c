#include "../include/cpu_simd.h"

AlignmentResult cpu_align_simd(Graph graph, Sequence sequence)
{
    compute_dp_cpu_simd(&graph.nodes[0], sequence);
    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 1; n < graph.num_nodes; n++)
    {
        compute_dp_cpu_simd(&graph.nodes[n], sequence);

        if (graph.nodes[n].max_score > graph.max_score) { // Update where to start traceback
            graph.max_score = graph.nodes[n].max_score; 
            graph.max_score_node_id = n;
        }
    }

    // Section 2: Traceback
    return compute_traceback_cpu_simd(graph, sequence);
}

void reverse_string(char* str, int len) {
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}


void compute_dp_cpu_simd(Node* node, Sequence sequence)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    int M = sequence.size;
    int N = node->sequence_size;

    node->dp_matrix = (DTYPEMATRIX*)malloc((M + 1) * (N + 1) * sizeof(DTYPEMATRIX));
    
    if (node->num_in == 0) {
        DTYPEMATRIX* curr_dp = node->dp_matrix;

        // We initialize the boundary column (j = 0) for all rows (i)
        for (int i = 0; i <= M; ++i) {
            int act = get_diagonal_index(i, 0, M, N);
            curr_dp[act] = 0;
        }

        for (int j = 1; j <= N; ++j) {
            int act = get_diagonal_index(0, j, M, N);
            curr_dp[act] = 0;
        }
    }
    else if (node->num_in == 1) {
        DTYPEMATRIX* curr_dp = node->dp_matrix;
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;
        int prev_N = node->v_in[0]->sequence_size;

        for (int i = 0; i <= M; ++i) {
            int act = get_diagonal_index(i, 0, M, N);
            // Inherit from the last column (j = prev_N) of the predecessor matrix
            int offset = get_diagonal_index(i, prev_N, M, prev_N);
            curr_dp[act] = prev_dp[offset];
        }

        for (int j = 1; j <= N; ++j) {
            int act = get_diagonal_index(0, j, M, N);
            curr_dp[act] = 0;
        }
    }
    else {
        DTYPEMATRIX* curr_dp = node->dp_matrix;
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 

        int prev_N1 = node->v_in[0]->sequence_size;
        int prev_N2 = node->v_in[1]->sequence_size;

        for (int i = 0; i <= M; ++i) {
            int act = get_diagonal_index(i, 0, M, N);
            int offset = get_diagonal_index(i, prev_N1, M, prev_N1);
            int offset2 = get_diagonal_index(i, prev_N2, M, prev_N2);
            
            curr_dp[act] = max(prev_dp[offset], prev_dp2[offset2]);
        }

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            int prev_Ni = node->v_in[i]->sequence_size;

            for (int j = 0; j <= M; ++j) {
                int act = get_diagonal_index(j, 0, M, N);
                int offset = get_diagonal_index(j, prev_Ni, M, prev_Ni);
                
                curr_dp[act] = max(prev_dp[offset], curr_dp[act]);
            }
        }

        for (int j = 1; j <= N; ++j) {
            int act = get_diagonal_index(0, j, M, N);
            curr_dp[act] = 0;
        }
    }

    // ------------------------------------------------- Compute -------------------------------------------------
    
    DTYPEMATRIX* __restrict dp = node->dp_matrix;
    char* __restrict node_seq = node->sequence;
    char* __restrict query_seq = malloc(sizeof(char) * sequence.size);

    for (int i = 0; i < sequence.size; ++i)
        query_seq[i] = sequence.sequence[i];

    reverse_string(query_seq, sequence.size);

    int local_max = -1;
    int local_max_d = -1;
    int local_max_j = -1;

    int l_min = (M < N) ? M : N;
    int l_max = (M > N) ? M : N;

    int startCurr = get_diag_start(1, M, N);
    int startPrev = get_diag_start(0, M, N);
    int startPrevPrev;
    int d = 2;

    // --------------- Grow phase ------------------

    for (; d <= l_min; d++) {

        startPrevPrev = startPrev;
        startPrev = startCurr;
        startCurr = get_diag_start(d, M, N);

        int prev_max = local_max;
        int j_start = 1;
        int j_end = d - 1;

        for (int j = 1; j < d; j++) { // TODO: Iterate over every 8 elements and look for local max j after, that way we can do SIMD and keep max j
            int i = d - j;
            int score = (node_seq[j-1] == query_seq[i-1]) ? MATCH : MISMATCH;

            int diagonal    = dp[startPrevPrev + j - 1] + score;
            int up          = dp[startPrev + j] + GAP;
            int left        = dp[startPrev + j - 1] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[startCurr + j] = res;

            if (res > local_max) {
                local_max = res;
            }
        }

        if (prev_max != local_max)
        {
            local_max_d = d;
        }
    }

    // ------------------ Stable phase ------------------

    for (; d <= l_max; d++) {
        
        startPrevPrev = startPrev;
        startPrev = startCurr;
        startCurr = get_diag_start(d, M, N);

        int prev_max = local_max;
        int j_start = (d - M > 1) ? d - M : 1;
        int j_end = (d - 1 < N) ? d - 1 : N;

        for (int j = j_start; j <= j_end; j++) {
            int i = d - j;
            int score = (node_seq[j-1] == query_seq[i-1]) ? MATCH : MISMATCH;

            int diagonal    = dp[startPrevPrev + j - 1] + score;
            int up          = dp[startPrev + j] + GAP;
            int left        = dp[startPrev + j - 1] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[startCurr + j] = res;

            if (res > local_max) {
                local_max = res;
            }
        }

        if (prev_max != local_max) {
            local_max_d = d;
        }
    }

    // ------------------ Shrink Phase ------------------

    int total_diagonals = M + N;
    for (; d <= total_diagonals; d++) {
        startPrevPrev = startPrev;
        startPrev = startCurr;
        startCurr = get_diag_start(d, M, N);

        int prev_max = local_max;
        int j_start = (d - M > 1) ? d - M : 1;
        int j_end = (d - 1 < N) ? d - 1 : N;

        for (int j = j_start; j <= j_end; j++) {
            int i = d - j;
            int score = (node_seq[j-1] == query_seq[i-1]) ? MATCH : MISMATCH;

            int diagonal    = dp[startPrevPrev + j - 1] + score;
            int up          = dp[startPrev + j] + GAP;
            int left        = dp[startPrev + j - 1] + GAP;

            int res = max(max(diagonal, 0), max(up, left));
            dp[startCurr + j] = res;

            if (res > local_max) {
                local_max = res;
            }
        }

        if (prev_max != local_max) {
            local_max_d = d;
        }
    }

    // ------------------ Find j of local max ------------------

    if (local_max_d != -1) {
        
        int final_d = local_max_d;
        int j_start = (local_max_d - M > 1) ? local_max_d - M : 1;
        int j_end = (local_max_d - 1 < N) ? local_max_d - 1 : N;

        startCurr = get_diag_start(final_d, M, N);

        for (int j = j_start; j <= j_end; j++) {
            if (dp[startCurr + j] == local_max) {
                local_max_j = j;
                break;
            }
        }
    }

    node->max_score = local_max;
    node->max_score_i = (local_max_d != -1) ? (local_max_d - local_max_j) : -1;
    node->max_score_j = local_max_j;
}

AlignmentResult compute_traceback_cpu_simd(Graph graph, Sequence sequence) {
    // Start from the last node in the topological sort
    Node* curr_node = &graph.nodes[graph.max_score_node_id];
    int i = curr_node->max_score_i;
    int j = curr_node->max_score_j;
    int row_width = sequence.size + 1;

    char* align_graph = malloc(graph.num_nodes + sequence.size + 1);
    char* align_query = malloc(graph.num_nodes + sequence.size + 1);
    int pos = 0;

    while (curr_node != NULL) {
        int score_now = curr_node->dp_matrix[i * row_width + j];
        if (score_now <= 0) break;

        if (i > 0 && j > 0) {
            int score = (curr_node->sequence[i-1] == sequence.sequence[j-1]) ? MATCH : MISMATCH;
            if (score_now == curr_node->dp_matrix[(i-1) * row_width + (j-1)] + score) {
                align_graph[pos] = curr_node->sequence[i-1];
                align_query[pos] = sequence.sequence[j-1];
                i--; j--;
            } else if (score_now == curr_node->dp_matrix[(i-1) * row_width + j] + GAP) {
                align_graph[pos] = curr_node->sequence[i-1];
                align_query[pos] = '-';
                i--;
            } else {
                align_graph[pos] = '-';
                align_query[pos] = sequence.sequence[j-1];
                j--;
            }
            pos++;
        } else if (i > 0) { // j == 0
            align_graph[pos] = curr_node->sequence[i-1];
            align_query[pos] = '-';
            i--; pos++;
        } else { // i == 0, jump to another node
            if (curr_node->num_in > 0) {
                Node* best_prev = NULL;
                for (int p = 0; p < curr_node->num_in; p++) {
                    Node* prev = curr_node->v_in[p];
                    if (score_now == prev->dp_matrix[prev->sequence_size * row_width + j]) {
                        best_prev = prev;
                        break;
                    }
                }
                curr_node = best_prev;
                if (curr_node) i = curr_node->sequence_size;
            } else {
                while (j > 0) {
                    align_graph[pos] = '-';
                    align_query[pos] = sequence.sequence[j-1];
                    j--; pos++;
                }
                curr_node = NULL;
            }
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
