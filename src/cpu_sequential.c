#include "../include/cpu_sequential.h"

AlignmentResult cpu_align_sequential(Graph graph, Sequence sequence)
{
    compute_dp_cpu_sequential(&graph.nodes[0], sequence);
    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 1; n < graph.num_nodes; n++)
    {
        compute_dp_cpu_sequential(&graph.nodes[n], sequence);

        if (graph.nodes[n].max_score > graph.max_score) { // Update where to start traceback
            graph.max_score = graph.nodes[n].max_score; 
            graph.max_score_node_id = n;
        }
    }

    // Section 2: Traceback
    return compute_traceback_cpu_sequential(graph, sequence);
}

void compute_dp_cpu_sequential(Node* node, Sequence sequence)
{
    // ------------------------------------------------- Initialize -------------------------------------------------

    node->dp_matrix = (DTYPEMATRIX*)malloc((sequence.size + 1) * (node->sequence_size + 1) * sizeof(DTYPEMATRIX));
    
    if (node->num_in == 0) {
        // We do not have to inherit from another matrix, init to zeroes
        DTYPEMATRIX* curr_dp = node->dp_matrix; // Just to make sure the compiler does not think that since node is a pointer the dp_matrix we're doing can change midway. We want simd if possible

        for (int i = 0; i <= sequence.size; ++i)
            curr_dp[i] = 0;
    }
    else if (node->num_in == 1){
        DTYPEMATRIX* curr_dp = node->dp_matrix;
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix;

        int offset = node->v_in[0]->sequence_size * (sequence.size + 1);

        for (int i = 0; i <= sequence.size; ++i)
            curr_dp[i] = prev_dp[offset + i]; //max(prev_dp[offset + i], 0); 
    }
    else {
        DTYPEMATRIX* curr_dp = node->dp_matrix;
        DTYPEMATRIX* prev_dp = node->v_in[0]->dp_matrix; 
        DTYPEMATRIX* prev_dp2 = node->v_in[1]->dp_matrix; 

        int offset = node->v_in[0]->sequence_size * (sequence.size + 1);
        int offset2 = node->v_in[1]->sequence_size * (sequence.size + 1);

        for (int j = 0; j <= sequence.size; ++j)
            curr_dp[j] = max(prev_dp[offset + j], prev_dp2[offset2 + j]); 

        for (int i = 2; i < node->num_in; ++i) {
            prev_dp = node->v_in[i]->dp_matrix;
            offset = node->v_in[i]->sequence_size * (sequence.size + 1);

            for (int j = 0; j <= sequence.size; ++j)
                curr_dp[j] = max(prev_dp[offset + j], curr_dp[j]);
        }
    }

    // ------------------------------------------------- Compute -------------------------------------------------
    
    DTYPEMATRIX* dp = node->dp_matrix;
    char* node_seq = node->sequence;
    char* query_seq = sequence.sequence;

    int local_max = -1;
    int local_max_i = -1;
    int local_max_j = -1;

    for (int i = 1; i <= node->sequence_size; i++) {
        int curr_row = i * (sequence.size + 1);
        int prev_row = (i - 1) * (sequence.size + 1);
        
        dp[curr_row] = dp[prev_row];

        for (int j = 1; j <= sequence.size; j++) {

            int score = (node_seq[i-1] == query_seq[j-1]) ? MATCH : MISMATCH;

            int diagonal = dp[prev_row + (j - 1)] + score;
            int up = dp[prev_row + j] + GAP;
            int left = dp[curr_row + (j - 1)] + GAP;

            int res = max(0, max(diagonal, max(up, left)));
            
            dp[curr_row + j] = res;

            if (res > local_max)
            {
                local_max = res;
                local_max_i = i;
                local_max_j = j;
            }

            // OFC, this whole loop is bad, and meant to give us better looking numbers ;)
        }
    }

    node->max_score = local_max;
    node->max_score_i = local_max_i;
    node->max_score_j = local_max_j;
}

void reverse_string(char* str, int len) {
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}

AlignmentResult compute_traceback_cpu_sequential(Graph graph, Sequence sequence) {
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
