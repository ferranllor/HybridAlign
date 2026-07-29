// ---------------------------------------------------------------------------------------------
//  verify_dp - correctness harness
//
//  Runs the CPU sequential aligner and one GPU version on the same graph/sequence pair and
//  compares them cell by cell:
//
//    * every DP matrix, entry by entry (the CPU stores them row major, the GPU anti-diagonal
//      major, so the indices are translated with get_diagonal_index())
//    * the per node maximum score
//    * the final alignment (identical strings, and if they differ, whether they still score the
//      same - traceback tie breaks are allowed to differ, scores are not)
//
//  Build:  make -C tools            (or: tools/verify.sh, which also sweeps BLOCKSIZE)
//  Usage:  tools/bin/verify_dp <dataset> <mode> <version> [max_reports] [--csv]
//          mode is the same as bin/main: 1 = GPU only, 2 = hybrid CPU-GPU
//          e.g. tools/bin/verify_dp 500_10 1 6
//               tools/bin/verify_dp 500_10 2 0
//
//  Exit code is 0 when the GPU matches the CPU, 1 otherwise, so it can be used in a script.
// ---------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "definitions.h"
#include "commons.cuh"
#include "cuda_utils.cuh"

extern "C" {
#include "cpu_sequential.h"
}
#include "cuda_naive.cuh"
#include "cuda_parallel_node.cuh"
#include "cuda_parallel_async.cuh"
#include "cuda_async_monolithic.cuh"
#include "cuda_async_batching.cuh"
#include "cuda_shared_mem.cuh"
#include "hybrid_base.cuh"
#include "hybrid_unified.cuh"
#include "hybrid_pinned.cuh"
#include "cuda_no_copy.cuh"

// ------------------------------------------------------------------ input (mirrors main.c)

static int get_node_index(IDMap* map, int map_size, const char* name) {
    for (int i = 0; i < map_size; i++) if (strcmp(map[i].name, name) == 0) return map[i].index;
    return -1;
}

static int read_gfa_graph(const char* filename, Graph* graph) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: could not open %s\n", filename); return -1; }

    static char line[65536];
    int node_count = 0;
    while (fgets(line, sizeof(line), fp)) if (line[0] == 'S') node_count++;

    graph->nodes = (Node*)calloc(node_count, sizeof(Node));
    IDMap* id_map = (IDMap*)malloc(node_count * sizeof(IDMap));
    graph->num_nodes = node_count;

    rewind(fp);
    int current_node = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != 'S') continue;
        strtok(line, "\t\n");
        char* name = strtok(NULL, "\t\n");
        char* sequence = strtok(NULL, "\t\n");
        if (!name || !sequence) continue;

        strncpy(id_map[current_node].name, name, 255);
        id_map[current_node].index = current_node;
        graph->nodes[current_node].id = current_node;
        graph->nodes[current_node].sequence.size = strlen(sequence);
        graph->nodes[current_node].sequence.sequence = (char*)malloc(strlen(sequence) + 1);
        strcpy(graph->nodes[current_node].sequence.sequence, sequence);
        current_node++;
    }

    rewind(fp);
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != 'L') continue;
        strtok(line, "\t\n");
        char* from_name = strtok(NULL, "\t\n");
        strtok(NULL, "\t\n");
        char* to_name = strtok(NULL, "\t\n");
        if (!from_name || !to_name) continue;

        int u = get_node_index(id_map, node_count, from_name);
        int v = get_node_index(id_map, node_count, to_name);
        if (u == -1 || v == -1) continue;

        Node* s = &graph->nodes[u];
        Node* t = &graph->nodes[v];
        s->v_out = (Node**)realloc(s->v_out, (s->num_out + 1) * sizeof(Node*));
        s->v_out[s->num_out++] = t;
        t->v_in = (Node**)realloc(t->v_in, (t->num_in + 1) * sizeof(Node*));
        t->v_in[t->num_in++] = s;
    }

    free(id_map);
    fclose(fp);
    return 0;
}

static int read_input_sequence_old(const char* filename, Sequence* seq, Sequence* seq_mod) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: could not open %s\n", filename); return -1; }

    static char line[16384];
    if (fgets(line, sizeof(line), fp)) {
        int id1, id2;
        static char s1[16384], s2[16384];
        if (sscanf(line, "%d %d %s %s", &id1, &id2, s1, s2) == 4) {
            seq->size = strlen(s1);
            seq->sequence = (char*)malloc(seq->size + 1);
            strcpy(seq->sequence, s1);
            seq_mod->size = strlen(s2);
            seq_mod->sequence = (char*)malloc(seq_mod->size + 1);
            strcpy(seq_mod->sequence, s2);
        }
    }
    fclose(fp);
    return 0;
}

static int sort_graph_topologically(Graph* graph) {
    if (graph->num_nodes <= 1) return 0;

    int* order = (int*)malloc(graph->num_nodes * sizeof(int));
    int* temp_indegree = (int*)malloc(graph->num_nodes * sizeof(int));
    int* queue = (int*)malloc(graph->num_nodes * sizeof(int));
    int* depth = (int*)calloc(graph->num_nodes, sizeof(int));

    int head = 0, tail = 0;
    for (int i = 0; i < graph->num_nodes; i++) {
        temp_indegree[i] = graph->nodes[i].num_in;
        if (!temp_indegree[i]) queue[tail++] = i;
    }

    int count = 0;
    while (head < tail) {
        int u_idx = queue[head++];
        order[count++] = u_idx;
        Node* u = &graph->nodes[u_idx];
        for (int i = 0; i < u->num_out; i++) {
            int v_idx = u->v_out[i]->id;
            if (depth[u_idx] + 1 > depth[v_idx]) depth[v_idx] = depth[u_idx] + 1;
            if (--temp_indegree[v_idx] == 0) queue[tail++] = v_idx;
        }
    }
    if (count < graph->num_nodes) { fprintf(stderr, "Error: cycle detected\n"); return -1; }

    int* old_to_new = (int*)malloc(graph->num_nodes * sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) old_to_new[order[i]] = i;

    Node* sorted_nodes = (Node*)malloc(graph->num_nodes * sizeof(Node));
    for (int i = 0; i < graph->num_nodes; i++) {
        int old_idx = order[i];
        sorted_nodes[i] = graph->nodes[old_idx];
        sorted_nodes[i].id = i;
        sorted_nodes[i].depth = depth[old_idx];
    }
    for (int i = 0; i < graph->num_nodes; i++) {
        Node* curr = &sorted_nodes[i];
        for (int j = 0; j < curr->num_in; j++)  curr->v_in[j]  = &sorted_nodes[old_to_new[curr->v_in[j]->id]];
        for (int j = 0; j < curr->num_out; j++) curr->v_out[j] = &sorted_nodes[old_to_new[curr->v_out[j]->id]];
    }

    free(graph->nodes);
    graph->nodes = sorted_nodes;
    free(order); free(temp_indegree); free(queue); free(old_to_new); free(depth);
    return 0;
}

// ------------------------------------------------------------------ helpers

static const char* version_name(int mode, int v) {
    if (mode == 3) {
        switch (v) {
            case 0: case 1: return "no_copy_naive";
            case 2: case 3: case 4: case 5: return "no_copy_level";
            case 6: return "no_copy_shared_mem";
            default: return "unknown";
        }
    }
    if (mode == 2) {
        switch (v) {
            case 0: return "hybrid_base";
            case 1: return "hybrid_unified";
            case 2: return "hybrid_pinned";
            case 3: return "hybrid_advised";
            default: return "unknown";
        }
    }
    switch (v) {
        case 0: case 1: return "naive";
        case 2: return "parallel_node";
        case 3: return "parallel_async";
        case 4: return "async_monolithic";
        case 5: return "async_batching";
        case 6: return "shared_mem";
        default: return "unknown";
    }
}

// Smith-Waterman score of an already built alignment, used to tell a different-but-equivalent
// traceback (allowed) from a wrong one (not allowed).
static int alignment_score(const char* g, const char* q) {
    int score = 0;
    for (int i = 0; g[i]; i++) {
        if (g[i] == '-' || q[i] == '-') score += GAP;
        else score += (g[i] == q[i]) ? MATCH : MISMATCH;
    }
    return score;
}

// ------------------------------------------------------------------ main

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dataset> <mode 1|2|3> <version> [max_reports] [--csv]\n", argv[0]);
        fprintf(stderr, "   ex: %s 500_10 1 6   (GPU shared_mem)\n", argv[0]);
        fprintf(stderr, "       %s 500_10 2 0   (hybrid base)\n", argv[0]);
        return 2;
    }

    const char* dataset = argv[1];
    int mode = atoi(argv[2]);
    int gpu_version = atoi(argv[3]);
    int max_reports = (argc > 4 && argv[4][0] != '-') ? atoi(argv[4]) : 10;
    bool csv = false;
    for (int a = 4; a < argc; a++) if (!strcmp(argv[a], "--csv")) csv = true;

    char gpath[1024], spath[1024];
    snprintf(gpath, sizeof(gpath), "datasets/graphs/old/%s.graph", dataset);
    snprintf(spath, sizeof(spath), "datasets/sequences/old/S_%s.seq", dataset);

    Graph gcpu, ggpu, cudaGraph;
    Sequence seq, seq_mod;
    if (read_gfa_graph(gpath, &gcpu) || sort_graph_topologically(&gcpu)) return 2;
    if (read_gfa_graph(gpath, &ggpu) || sort_graph_topologically(&ggpu)) return 2;
    if (read_input_sequence_old(spath, &seq, &seq_mod)) return 2;

    int M = seq.size;

    // NOTE: init_cpu_graph() lives in commons.cuh and has no return statement; compiled as C++
    // that is UB and traps at runtime, so the plain host allocation is inlined here instead.
    for (int n = 0; n < gcpu.num_nodes; n++) {
        size_t ms = (size_t)(M + 2) * (gcpu.nodes[n].sequence.size + 2);
        gcpu.nodes[n].dp_matrix = (DTYPEMATRIX*)malloc(ms * sizeof(DTYPEMATRIX));
    }

    AlignmentResult rcpu = cpu_align_sequential(gcpu, seq);

    AlignmentResult rgpu;

    if (mode == 3) {
        init_shared_graph(&ggpu, &cudaGraph, M);

        switch (gpu_version) {
            case 0: case 1: rgpu = gpu_align_no_copy_naive(ggpu, cudaGraph, seq); break;
            case 2: case 3: case 4: case 5: rgpu = gpu_align_no_copy_level(ggpu, cudaGraph, seq); break;
            case 6: rgpu = gpu_align_no_copy_shared_mem(ggpu, cudaGraph, seq); break;
            default: fprintf(stderr, "unknown no-copy version %d\n", gpu_version); return 2;
        }
    }
    else if (mode == 2) {
        switch (gpu_version) {
            case 0: init_hybrid_graph(&ggpu, &cudaGraph, M); break;
            case 1: init_unified_graph(&ggpu, &cudaGraph, M); break;
            case 2: init_pinned_graph(&ggpu, &cudaGraph, M); break;
            case 3: init_advised_graph(&ggpu, &cudaGraph, M); break;
            default: fprintf(stderr, "unknown hybrid version %d\n", gpu_version); return 2;
        }

        switch (gpu_version) {
            case 0: rgpu = gpu_align_hybrid_base(ggpu, cudaGraph, seq); break;
            case 1: rgpu = gpu_align_hybrid_unified(ggpu, cudaGraph, seq); break;
            case 2: case 3: rgpu = gpu_align_hybrid_pinned(ggpu, cudaGraph, seq); break;
            default: fprintf(stderr, "unknown hybrid version %d\n", gpu_version); return 2;
        }
    } else {
        init_cpu_graph_pinned(&ggpu, M);
        init_gpu_graph(&ggpu, &cudaGraph, M);

        switch (gpu_version) {
            case 0: case 1: rgpu = gpu_align_naive(ggpu, cudaGraph, seq); break;
            case 2: rgpu = gpu_align_parallel_node(ggpu, cudaGraph, seq); break;
            case 3: rgpu = gpu_align_parallel_async(ggpu, cudaGraph, seq); break;
            case 4: rgpu = gpu_align_async_monolithic(ggpu, cudaGraph, seq); break;
            case 5: rgpu = gpu_align_async_batching(ggpu, cudaGraph, seq); break;
            case 6: rgpu = gpu_align_shared_mem(ggpu, cudaGraph, seq); break;
            default: fprintf(stderr, "unknown GPU version %d\n", gpu_version); return 2;
        }
    }

    // ---------------- DP matrices, cell by cell ----------------
    //   CPU: dp[i_node * (M + 1) + j_query]      (row major)
    //   GPU: dp[get_diagonal_index(i_query, j_node, M, N)]   (anti-diagonal major)

    long long bad_nodes = 0, bad_cells = 0, bad_scores = 0;
    long long tot_class[3] = {0, 0, 0}, bad_class[3] = {0, 0, 0}; // 0: N<M, 1: N==M, 2: N>M
    int reports = 0;

    for (int n = 0; n < gcpu.num_nodes; n++) {
        int N = gcpu.nodes[n].sequence.size;
        DTYPEMATRIX* a = gcpu.nodes[n].dp_matrix;
        DTYPEMATRIX* b = ggpu.nodes[n].dp_matrix;

        int klass = (N < M) ? 0 : ((N == M) ? 1 : 2);
        tot_class[klass]++;

        bool node_bad = false;
        int first_i = -1, first_j = -1;
        for (int inode = 0; inode <= N; inode++) {
            for (int jq = 0; jq <= M; jq++) {
                if (a[inode * (M + 1) + jq] != b[get_diagonal_index(jq, inode, M, N)]) {
                    bad_cells++;
                    if (!node_bad) { first_i = inode; first_j = jq; node_bad = true; }
                }
            }
        }

        if (gcpu.nodes[n].max_score != ggpu.nodes[n].max_score) bad_scores++;

        if (node_bad) {
            bad_nodes++;
            bad_class[klass]++;
            if (!csv && reports < max_reports) {
                reports++;
                printf("  node %d (N=%d, M=%d, %s, bands=%d, num_in=%d): first mismatch at "
                       "node_j=%d query_i=%d (d=%d) cpu=%d gpu=%d\n",
                       n, N, M, (M >= N ? "M>=N" : "M<N"),
                       ((M >= N ? N : M) + BLOCKSIZE - 1) / BLOCKSIZE, gcpu.nodes[n].num_in,
                       first_i, first_j, first_i + first_j,
                       a[first_i * (M + 1) + first_j],
                       b[get_diagonal_index(first_j, first_i, M, N)]);
            }
        }
    }

    int cpu_score = alignment_score(rcpu.graph_align, rcpu.query_align);
    int gpu_score = alignment_score(rgpu.graph_align, rgpu.query_align);
    bool same_align = (strcmp(rcpu.graph_align, rgpu.graph_align) == 0 &&
                       strcmp(rcpu.query_align, rgpu.query_align) == 0);
    bool ok = (bad_cells == 0 && bad_scores == 0 && cpu_score == gpu_score);

    if (csv) {
        // dataset,mode,version,name,blocksize,nodes,M,bad_nodes,bad_cells,bad_max_scores,
        // cpu_len,gpu_len,cpu_score,gpu_score,identical_alignment,ok
        printf("%s,%d,%d,%s,%d,%d,%d,%lld,%lld,%lld,%d,%d,%d,%d,%d,%d\n",
               dataset, mode, gpu_version, version_name(mode, gpu_version), BLOCKSIZE,
               gcpu.num_nodes, M, bad_nodes, bad_cells, bad_scores,
               rcpu.size, rgpu.size, cpu_score, gpu_score, same_align ? 1 : 0, ok ? 1 : 0);
    } else {
        printf("dataset=%s mode=%d version=%d (%s) nodes=%d M=%d BLOCKSIZE=%d\n",
               dataset, mode, gpu_version, version_name(mode, gpu_version), gcpu.num_nodes, M, BLOCKSIZE);
        printf("DP mismatches : %lld / %d nodes, %lld cells\n", bad_nodes, gcpu.num_nodes, bad_cells);
        printf("  by shape    : N<M %lld/%lld | N==M %lld/%lld | N>M %lld/%lld\n",
               bad_class[0], tot_class[0], bad_class[1], tot_class[1], bad_class[2], tot_class[2]);
        printf("max_score mismatches: %lld / %d nodes\n", bad_scores, gcpu.num_nodes);
        printf("alignment     : cpu len=%d score=%d | gpu len=%d score=%d | identical=%s\n",
               rcpu.size, cpu_score, rgpu.size, gpu_score, same_align ? "yes" : "no");
        if (!same_align && cpu_score == gpu_score)
            printf("                (different traceback, same score - equivalent optimum)\n");
        printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    }

    return ok ? 0 : 1;
}
