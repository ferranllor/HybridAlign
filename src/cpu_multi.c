#include "../include/cpu_multi.h"
#include "../include/cpu_utils.h"
#include "../include/cpu_last_col.h"

// *************************************************************************************************
//
//                                            Reads
//
// *************************************************************************************************

// The same batch the GPU version builds, base for base: only the first read is real and the rest
// are copies of it with a deterministic scattering of substituted bases. It has to be the same
// scheme and the same seed, otherwise the two modes would not be aligning the same thing and the
// throughput numbers would not be comparable. Read 0 being untouched is also the correctness
// check, its alignment has to come out the same as every single sequence version produces.

static void cpu_multi_build_reads(Sequence sequence, int num_reads, DTYPEALPHABET* reads)
{
    const char bases[4] = { 'A', 'C', 'G', 'T' };
    int M = sequence.size;

    for (int r = 0; r < num_reads; r++) {
        DTYPEALPHABET* dst = &reads[(size_t)r * M];

        memcpy(dst, sequence.sequence, M * sizeof(DTYPEALPHABET));

        unsigned int h = 2166136261u ^ (unsigned int)r;
        for (int t = 0; r > 0 && t < M / 20; t++) {
            h = h * 16777619u + 2654435761u;
            dst[h % M] = bases[(h >> 16) & 3];
        }
    }
}

// *************************************************************************************************
//
//                                          Align loop
//
// *************************************************************************************************

// The multiple sequence version of the last column CPU baseline. The per node sweep, the matrix
// rebuild and the walk back are all the ones cpu_last_col already defines: aligning R queries
// against one graph does not change any of them, it only changes which column of a node each one
// writes, which is what col_offset is for.
//
// What does change is the decomposition. cpu_last_col parallelises over the nodes of a level, which
// leaves it a single thread on 150_10's 3733 one node levels; here a thread takes a whole query and
// walks the entire graph with it, so there is no barrier anywhere in the run and the tail is as
// parallel as the head. That is deliberately the CPU's best case, so that what the comparison
// against mode 4 measures is throughput per query rather than who copes better with a bad
// dependency graph.
//
// The queries are independent, and the nodes are already in topological order, so a plain forward
// scan of the array visits every predecessor before its successors.

AlignmentResult cpu_align_multi(Graph graph, Sequence sequence)
{
    int M = sequence.size;

    const char* e = getenv("NUM_READS");
    int num_reads = e ? atoi(e) : 32;
    if (num_reads < 1) num_reads = 1;

    DTYPEALPHABET* reads = (DTYPEALPHABET*)malloc((size_t)num_reads * M);
    DTYPEALPHABET* reads_rev = (DTYPEALPHABET*)malloc((size_t)num_reads * M);

    cpu_multi_build_reads(sequence, num_reads, reads);

    for (int r = 0; r < num_reads; r++)
        for (int i = 0; i < M; i++)
            reads_rev[(size_t)r * M + i] = reads[(size_t)r * M + M - 1 - i];

    size_t slots = (size_t)graph.num_nodes * num_reads;

    int* max_score = (int*)malloc(slots * sizeof(int));
    int* max_d     = (int*)malloc(slots * sizeof(int));
    int* max_j     = (int*)malloc(slots * sizeof(int));

    int max_node_size = 0;
    for (int n = 0; n < graph.num_nodes; n++)
        if (graph.nodes[n].sequence.size > max_node_size) max_node_size = graph.nodes[n].sequence.size;

    #pragma omp parallel
    {
        DTYPEMATRIX* scratch = (DTYPEMATRIX*)malloc(3 * (size_t)(max_node_size + 2) * sizeof(DTYPEMATRIX));

        #pragma omp for schedule(dynamic)
        for (int r = 0; r < num_reads; r++) {
            const DTYPEALPHABET* query_rev = &reads_rev[(size_t)r * M];
            size_t col_offset = (size_t)r * (M + 1);

            for (int n = 0; n < graph.num_nodes; n++) {
                size_t slot = (size_t)n * num_reads + r;

                compute_dp_cpu_last_col(&graph.nodes[n], sequence, query_rev, col_offset, scratch,
                                        &max_score[slot], &max_d[slot], &max_j[slot]);
            }
        }

        free(scratch);
    }

    AlignmentResult res = compute_traceback_cpu_multi(graph, sequence, 0, num_reads,
                                                      max_score, max_d, max_j, reads);

    for (int r = 1; r < num_reads; r++) {
        AlignmentResult other = compute_traceback_cpu_multi(graph, sequence, r, num_reads,
                                                            max_score, max_d, max_j, reads);
        free(other.graph_align);
        free(other.query_align);
    }

    free(max_score); free(max_d); free(max_j);
    free(reads); free(reads_rev);

    return res;
}

// *************************************************************************************************
//
//                                           Traceback
//
// *************************************************************************************************

// Only the starting cell differs from the single query case: a batch keeps its maxima in its own
// arrays instead of in the node, so this finds the best node for one query and hands the walk to
// traceback_last_col with that query's column offset.

AlignmentResult compute_traceback_cpu_multi(Graph graph, Sequence sequence, int read, int num_reads,
                                            const int* max_score, const int* max_d, const int* max_j,
                                            const DTYPEALPHABET* reads)
{
    int M = sequence.size;
    const DTYPEALPHABET* query = &reads[(size_t)read * M];

    int best = -1, best_node = 0;
    for (int n = 0; n < graph.num_nodes; n++) {
        int v = max_score[(size_t)n * num_reads + read];
        if (v > best) { best = v; best_node = n; }
    }

    int d = max_d[(size_t)best_node * num_reads + read];
    int j = max_j[(size_t)best_node * num_reads + read];
    int i = (d != -1) ? (d - j) : -1;

    return traceback_last_col(graph, query, M, (size_t)read * (M + 1),
                              &graph.nodes[best_node], i, j);
}
