#include "../include/cpu_last_col.h"
#include "../include/cpu_utils.h"

// *************************************************************************************************
//
//                                          Forward pass
//
// *************************************************************************************************

// One node against one query, producing exactly what the GPU kernels produce: the node's last
// column and the query's local maximum inside that node. Nothing else is kept, so the sweep only
// holds three anti diagonals at a time and the scratch is 3 * (N + 2) ints, which belongs to the
// thread rather than to the node.
//
// It sweeps anti diagonals rather than rows. Along a row every cell depends on the one to its left,
// so a row sweep is a serial chain the compiler cannot vectorise; along an anti diagonal the cells
// are independent and all three neighbours sit on the two previous diagonals, so the inner loop is
// a straight vector operation. Written as rows this was six times slower.
//
// The query is walked backwards so both sequence reads move forward with j, which is what the
// pre-reversed copy is for, and the maximum is a vector reduction with the scalar search for its
// column only run on the rare diagonal that improves it.
//
// col_offset is where this query's column lives inside a node's last_col. It is zero when there is
// a single query, and read * (M + 1) when a batch of them shares the graph, which is the only
// difference between the single sequence and the multiple sequence versions.
//
// The maximum is resolved the way the kernels resolve it, highest score then earliest anti diagonal
// then leftmost column, so a node computed here and a node computed on the GPU report the same cell
// and the traceback cannot tell which side produced it.

void compute_dp_cpu_last_col(Node* node, Sequence sequence, const DTYPEALPHABET* query_rev,
                             size_t col_offset, DTYPEMATRIX* scratch,
                             int* best_score, int* best_d, int* best_j)
{
    int M = sequence.size;
    int N = node->sequence.size;
    int stride = N + 2;

    const DTYPEALPHABET* node_seq = node->sequence.sequence;
    DTYPEMATRIX* last_col = &node->last_col[col_offset];

    DTYPEMATRIX* prev2 = scratch;
    DTYPEMATRIX* prev1 = &scratch[stride];
    DTYPEMATRIX* curr  = &scratch[2 * stride];

    for (int j = 0; j < 3 * stride; j++) scratch[j] = 0;

    int boundary1 = 0;
    for (int p = 0; p < node->num_in; p++)
        if (node->v_in[p]->last_col[col_offset + 1] > boundary1)
            boundary1 = node->v_in[p]->last_col[col_offset + 1];

    prev1[0] = boundary1;
    last_col[0] = 0;

    int best = -1, bd = -1, bj = -1;

    for (int d = 2; d <= M + N; d++) {
        int jlo = (d - M > 1) ? (d - M) : 1;
        int jhi = (N < d - 1) ? N : (d - 1);

        if (d <= M) {
            int boundary = 0;
            for (int p = 0; p < node->num_in; p++)
                if (node->v_in[p]->last_col[col_offset + d] > boundary)
                    boundary = node->v_in[p]->last_col[col_offset + d];

            curr[0] = boundary;
        }

        if (d <= N) curr[d] = 0;

        int dmax = -1;

        for (int j = jlo; j <= jhi; j++) {
            int score = (node_seq[j-1] == query_rev[M - d + j]) ? MATCH : MISMATCH;

            int diagonal = prev2[j-1] + score;
            int up       = prev1[j] + GAP;
            int left     = prev1[j-1] + GAP;

            int res = diagonal > 0 ? diagonal : 0;
            if (up > res) res = up;
            if (left > res) res = left;

            curr[j] = res;
            if (res > dmax) dmax = res;
        }

        if (dmax > best) {
            best = dmax;
            bd = d;
            for (int j = jlo; j <= jhi; j++) if (curr[j] == dmax) { bj = j; break; }
        }

        if (jhi == N && d - N >= 1) last_col[d - N] = curr[N];

        DTYPEMATRIX* roll = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr  = roll;
    }

    *best_score = best;
    *best_d = bd;
    *best_j = bj;
}

// *************************************************************************************************
//
//                                          Align loop
//
// *************************************************************************************************

// The CPU baseline for the last column versions, so that they have something to be compared
// against that keeps as little as they do.
//
// It is parallelised over the nodes of a level, the same decomposition cpu_simd_parallel_node uses,
// because that is the honest single sequence comparison: the GPU versions get one block or one warp
// per node of a level, and this gets one thread per node of a level. It inherits the same problem
// too, 150_10's 3733 one node levels leave it with a single thread's worth of work each, which is
// exactly what the hybrid and the multiple sequence versions exist to fix.
//
// The nodes are already in topological order, so the levels are contiguous runs of the array and
// one OpenMP region covers the whole graph, with the implicit barrier of "omp for" ordering them.

AlignmentResult cpu_align_last_col(Graph graph, Sequence sequence)
{
    int M = sequence.size;

    DTYPEALPHABET* query_rev = (DTYPEALPHABET*)malloc(M * sizeof(DTYPEALPHABET));
    for (int i = 0; i < M; i++) query_rev[i] = sequence.sequence[M - 1 - i];

    int num_levels = graph.nodes[graph.num_nodes-1].depth + 1;
    int* nodes_per_level = (int*)calloc(num_levels, sizeof(int));

    int max_node_size = 0;
    for (int n = 0; n < graph.num_nodes; n++) {
        nodes_per_level[graph.nodes[n].depth]++;
        if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;
    }

    #pragma omp parallel
    {
        DTYPEMATRIX* scratch = (DTYPEMATRIX*)malloc(3 * (size_t)(max_node_size + 2) * sizeof(DTYPEMATRIX));
        int level_offset = 0;

        for (int l = 0; l < num_levels; l++)
        {
            #pragma omp for schedule(dynamic)
            for (int t = 0; t < nodes_per_level[l]; t++)
            {
                Node* node = &graph.nodes[level_offset + t];
                int best, bd, bj;

                compute_dp_cpu_last_col(node, sequence, query_rev, 0, scratch, &best, &bd, &bj);

                node->max_score = best;
                node->max_score_d = bd;
                node->max_score_i = (bd != -1) ? (bd - bj) : -1;
                node->max_score_j = (bd != -1) ? bj : -1;
            }

            level_offset += nodes_per_level[l];
        }

        free(scratch);
    }

    graph.max_score = graph.nodes[0].max_score;
    graph.max_score_node_id = 0;

    for (int n = 0; n < graph.num_nodes; n++) {
        if (graph.nodes[n].max_score > graph.max_score) {
            graph.max_score = graph.nodes[n].max_score;
            graph.max_score_node_id = n;
        }
    }

    Node* start = &graph.nodes[graph.max_score_node_id];
    AlignmentResult res = traceback_last_col(graph, sequence.sequence, M, 0,
                                             start, start->max_score_i, start->max_score_j);

    free(nodes_per_level);
    free(query_rev);

    return res;
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Fills a plain row major (M + 1) x (N + 1) matrix for one node, the same recurrence the forward
// pass runs. Column 0 is the best of the predecessors' last columns and row 0 is zero, which is
// exactly how the forward pass seeds itself, so this produces the matrix that would have been
// stored. Called only for the nodes the alignment actually crosses.

void recompute_node_dp_last_col(Node* node, const DTYPEALPHABET* query, int M,
                                size_t col_offset, DTYPEMATRIX* dp)
{
    int N = node->sequence.size;
    int stride = N + 1;

    for (int j = 0; j <= N; j++) dp[j] = 0;

    for (int i = 1; i <= M; i++) {
        int boundary = 0;
        for (int p = 0; p < node->num_in; p++)
            if (node->v_in[p]->last_col[col_offset + i] > boundary)
                boundary = node->v_in[p]->last_col[col_offset + i];

        dp[i * stride] = boundary;
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            int score = (node->sequence.sequence[j-1] == query[i-1]) ? MATCH : MISMATCH;

            int diagonal = dp[(i-1) * stride + (j-1)] + score;
            int up       = dp[(i-1) * stride + j] + GAP;
            int left     = dp[i * stride + (j-1)] + GAP;

            dp[i * stride + j] = max(max(diagonal, 0), max(up, left));
        }
    }
}

// The walk back, shared by every version that keeps only last columns: the CPU baseline, both
// hybrids and the multiple sequence versions. None of them stored a matrix, so it rebuilds one node
// at a time and moves to the predecessor holding the score when it reaches column 0. The alignment
// is local, so it crosses one or two nodes before it hits a zero and recomputes about 77k cells,
// against the 1.95 billion the forward pass would have had to store to avoid it.
//
// Where the walk starts is the caller's business, because that is the only thing that differs
// between them: a single query reads it out of the node, a batch reads it out of its own array.

AlignmentResult traceback_last_col(Graph graph, const DTYPEALPHABET* query, int M,
                                   size_t col_offset, Node* start_node, int start_i, int start_j)
{
    Node* curr_node = start_node;
    int i = start_i;
    int j = start_j;

    int max_graph_seq_len = 0;
    int max_node_size = 0;
    for (int k = 0; k < graph.num_nodes; k++) {
        max_graph_seq_len += graph.nodes[k].sequence.size;
        if (graph.nodes[k].sequence.size > max_node_size) max_node_size = graph.nodes[k].sequence.size;
    }

    char* align_graph = (char*)malloc(M + max_graph_seq_len + 1);
    char* align_query = (char*)malloc(M + max_graph_seq_len + 1);
    int pos = 0;

    DTYPEMATRIX* dp = (DTYPEMATRIX*)malloc((size_t)(M + 1) * (max_node_size + 1) * sizeof(DTYPEMATRIX));
    Node* loaded = NULL;

    while (curr_node != NULL && i >= 0) {
        int N = curr_node->sequence.size;
        int stride = N + 1;

        if (curr_node != loaded) {
            recompute_node_dp_last_col(curr_node, query, M, col_offset, dp);
            loaded = curr_node;
        }

        int curr_score = dp[i * stride + j];
        if (curr_score <= 0) break;

        if (i > 0 && j > 0) {
            int score = (curr_node->sequence.sequence[j-1] == query[i-1]) ? MATCH : MISMATCH;

            int diag_score = dp[(i-1) * stride + (j-1)];
            int up_score   = dp[(i-1) * stride + j];

            if (curr_score == diag_score + score) {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = query[i-1];
                i--; j--;
            } else if (curr_score == up_score + GAP) {
                align_graph[pos] = '-';
                align_query[pos] = query[i-1];
                i--;
            } else {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = '-';
                j--;
            }
            pos++;
        }
        else if (j == 0) {
            if (curr_node->num_in > 0) {
                Node* best_prev = NULL;
                for (int p = 0; p < curr_node->num_in; p++) {
                    Node* prev = curr_node->v_in[p];
                    if (prev->last_col[col_offset + i] == curr_score) { best_prev = prev; break; }
                }
                curr_node = best_prev;
                if (curr_node) j = curr_node->sequence.size;
            } else {
                while (i > 0) {
                    align_graph[pos] = '-';
                    align_query[pos] = query[i-1];
                    i--; pos++;
                }
                curr_node = NULL;
            }
        }
        else {
            while (j > 0) {
                align_graph[pos] = curr_node->sequence.sequence[j-1];
                align_query[pos] = '-';
                j--; pos++;
            }
            curr_node = NULL;
        }
    }

    free(dp);

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
