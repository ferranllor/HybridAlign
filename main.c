#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "include/my_time_lib.h"
#include "include/cpu_sequential.h"
#include "include/cpu_simd.h"
#include "include/cpu_simd_parallel_dp.h"
#include "include/definitions.h"

// *************************************************************************************************
//
//                                             Utils
//
// *************************************************************************************************

int get_node_index(IDMap* map, int map_size, const char* name) {
    for (int i = 0; i < map_size; i++) {
        if (strcmp(map[i].name, name) == 0) return map[i].index;
    }
    return -1; // Not found
}

int read_gfa_graph(const char* filename, Graph* graph) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return -1;
    }

    char line[65536];
    int node_count = 0;

    // Pass 1: Count nodes to allocate memory
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'S') {
            node_count++;
        }
    }

    graph->nodes = (Node*)calloc(node_count, sizeof(Node));
    IDMap* id_map = (IDMap*)malloc(node_count * sizeof(IDMap));
    if (!graph->nodes || !id_map) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return -1;
    }
    
    graph->num_nodes = node_count;
    
    // Pass 2: Parse Segments (Nodes)
    rewind(fp);
    int current_node = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'S') {
            // GFA Format: S \t <Name> \t <Sequence> \t [Optional Tags]
            char* type = strtok(line, "\t\n");
            char* name = strtok(NULL, "\t\n");
            char* sequence = strtok(NULL, "\t\n");
            
            if (name && sequence) {
                // Map the string name to our internal contiguous integer ID
                strncpy(id_map[current_node].name, name, 255);
                id_map[current_node].index = current_node;

                graph->nodes[current_node].id = current_node; 
                graph->nodes[current_node].sequence.size = strlen(sequence);
                
                // If sequence is "*", it means sequence is omitted in GFA, handle appropriately
                if (strcmp(sequence, "*") == 0) {
                    graph->nodes[current_node].sequence.size = 0;
                    graph->nodes[current_node].sequence.sequence = NULL;
                } else {
                    graph->nodes[current_node].sequence.sequence = (char*)malloc((graph->nodes[current_node].sequence.size + 1) * sizeof(char));
                    strcpy(graph->nodes[current_node].sequence.sequence, sequence);
                }
                current_node++;
            }
        } 
    }

    // Pass 3: Parse Links (Edges)
    rewind(fp);
    
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'L') {
            // GFA Format: L \t <From> \t <FromOrient> \t <To> \t <ToOrient> \t <Overlap> \t [Optional Tags]
            char* type = strtok(line, "\t\n");
            char* from_name = strtok(NULL, "\t\n");
            char* from_orient = strtok(NULL, "\t\n");
            char* to_name = strtok(NULL, "\t\n");
            char* to_orient = strtok(NULL, "\t\n");
            char* overlap = strtok(NULL, "\t\n");
            
            if (from_name && to_name) {
                int u_idx = get_node_index(id_map, node_count, from_name);
                int v_idx = get_node_index(id_map, node_count, to_name);

                if (u_idx != -1 && v_idx != -1) {
                    Node* source = &graph->nodes[u_idx];
                    Node* target = &graph->nodes[v_idx];

                    // Add target to source's outgoing list
                    source->v_out = (Node**)realloc(source->v_out, (source->num_out + 1) * sizeof(Node*));
                    source->v_out[source->num_out] = target;
                    source->num_out++;
                    
                    // Add source to target's incoming list
                    target->v_in = (Node**)realloc(target->v_in, (target->num_in + 1) * sizeof(Node*));
                    target->v_in[target->num_in] = source;
                    target->num_in++;
                } else {
                    fprintf(stderr, "Warning: Link references unknown segments: %s -> %s\n", from_name, to_name);
                }
            }
        }
    }
    
    free(id_map);
    fclose(fp);
    return 0;
}

static void strip_newline(char *str) {
    int len = strlen(str);
    while (len > 0 && (str[len - 1] == '\n' || str[len - 1] == '\r')) {
        str[len - 1] = '\0';
        len--;
    }
}

int read_input_sequence_old(char* filename, Sequence* seq, Sequence* seq_mod)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return -1;
    }

    char line[4096];

    if (fgets(line, sizeof(line), fp)) {
        int id1, id2;
        char s1[2048], s2[2048];

        if (sscanf(line, "%d %d %s %s", &id1, &id2, s1, s2) == 4) {
            
            // Base sequence
            seq->size = strlen(s1);
            seq->sequence = (char*)malloc((seq->size + 1) * sizeof(char));
            if (seq->sequence) {
                strcpy(seq->sequence, s1);
            }

            // Ground truth
            seq_mod->size = strlen(s2);
            seq_mod->sequence = (char*)malloc((seq_mod->size + 1) * sizeof(char));
            if (seq_mod->sequence) {
                strcpy(seq_mod->sequence, s2);
            }
        }
    }

    fclose(fp);

    return 0;
}

int read_input_sequence(char* fastqfile, char* truthfile, Sequence* seq, Sequence* seq_mod)
{
    FILE *fq_fp = fopen(fastqfile, "r");
    if (!fq_fp) {
        fprintf(stderr, "Error: Could not open FASTQ file %s\n", fastqfile);
        return -1;
    }

    FILE *tr_fp = fopen(truthfile, "r");
    if (!tr_fp) {
        fprintf(stderr, "Error: Could not open Ground Truth file %s\n", truthfile);
        fclose(fq_fp);
        return -1;
    }

    char line1[2048], line2[2048], line3[2048], line4[2048];
    char tr_line[4096];
    int status = -1;

    // 1. Read one record from FASTQ (4 lines)
    if (fgets(line1, sizeof(line1), fq_fp) &&
        fgets(line2, sizeof(line2), fq_fp) &&
        fgets(line3, sizeof(line3), fq_fp) &&
        fgets(line4, sizeof(line4), fq_fp)) {
        
        strip_newline(line2); // line2 contains the read sequence

        // 2. Read corresponding line from Ground Truth TSV
        if (fgets(tr_line, sizeof(tr_line), tr_fp)) {
            strip_newline(tr_line);
            
            // Split TSV line into ID and True Sequence
            char* tr_id = strtok(tr_line, "\t");
            char* tr_seq = strtok(NULL, "\t");

            if (tr_seq) {
                // Base sequence (with simulated sequencing errors)
                seq->size = strlen(line2);
                seq->sequence = (char*)malloc((seq->size + 1) * sizeof(char));
                if (seq->sequence) {
                    strcpy(seq->sequence, line2);
                }

                // Ground truth sequence (error-free graph path sequence)
                seq_mod->size = strlen(tr_seq);
                seq_mod->sequence = (char*)malloc((seq_mod->size + 1) * sizeof(char));
                if (seq_mod->sequence) {
                    strcpy(seq_mod->sequence, tr_seq);
                }
                
                status = 0; // Successfully populated both structures
            }
        }
    }

    fclose(fq_fp);
    fclose(tr_fp);

    return status;
}

int sort_graph_topologically(Graph* graph) 
{
    if (graph->num_nodes <= 1) return 0;

    int* order = (int*)malloc(graph->num_nodes * sizeof(int));
    int* temp_indegree = (int*)malloc(graph->num_nodes * sizeof(int));
    int* queue = (int*)malloc(graph->num_nodes * sizeof(int));
    int head = 0, tail = 0;

    for (int i = 0; i < graph->num_nodes; i++) {
        temp_indegree[i] = graph->nodes[i].num_in;
        if (temp_indegree[i] == 0) queue[tail++] = i;
    }

    int count = 0;
    while (head < tail) {
        int u_idx = queue[head++];
        order[count++] = u_idx;
        Node* u = &graph->nodes[u_idx];
        for (int i = 0; i < u->num_out; i++) {
            int v_idx = u->v_out[i]->id; 
            if (--temp_indegree[v_idx] == 0) queue[tail++] = v_idx;
        }
    }

    if (count < graph->num_nodes) {
        fprintf(stderr, "Error: Cycle detected!!\n");
        return -1;
    }

    int* old_to_new = (int*)malloc(graph->num_nodes * sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) {
        old_to_new[order[i]] = i;
    }

    Node* sorted_nodes = (Node*)malloc(graph->num_nodes * sizeof(Node));
    for (int i = 0; i < graph->num_nodes; i++) {
        int old_idx = order[i];
        sorted_nodes[i] = graph->nodes[old_idx];
        sorted_nodes[i].id = i; // Update ID to match new index
    }

    for (int i = 0; i < graph->num_nodes; i++) {
        Node* curr = &sorted_nodes[i];
        
        // Update incoming pointers
        for (int j = 0; j < curr->num_in; j++) {
            int old_neighbor_idx = curr->v_in[j]->id;
            int new_neighbor_idx = old_to_new[old_neighbor_idx];
            curr->v_in[j] = &sorted_nodes[new_neighbor_idx];
        }

        // Update outgoing pointers
        for (int j = 0; j < curr->num_out; j++) {
            int old_neighbor_idx = curr->v_out[j]->id;
            int new_neighbor_idx = old_to_new[old_neighbor_idx];
            curr->v_out[j] = &sorted_nodes[new_neighbor_idx];
        }
    }

    free(graph->nodes); // Free the old array
    graph->nodes = sorted_nodes;

    free(order);
    free(temp_indegree);
    free(queue);
    free(old_to_new);

    return 0;
}

void verify_alignment(const char* align_graph, const char* align_query, Sequence ground_truth) {
    int matches = 0;
    int alignment_len = strlen(align_query);
    
    for (int i = 0; i < alignment_len; i++) {
        if (align_graph[i] == align_query[i] && align_graph[i] != '-') {
            matches++;
        }
    }

    float identity = (float)matches / alignment_len * 100.0f;

    char* cleaned_query = malloc(alignment_len + 1);
    int k = 0;
    for (int i = 0; i < alignment_len; i++) {
        if (align_query[i] != '-') {
            cleaned_query[k++] = align_query[i];
        }
    }
    cleaned_query[k] = '\0';

    printf("\n--- Alignment Verification ---\n");
    printf("Alignment Length: %d\n", alignment_len);
    printf("Identity Score:   %.2f%%\n", identity);

    if (strcmp(cleaned_query, ground_truth.sequence) == 0) {
        printf("Integrity Check:  PASSED (Query matches ground truth)\n");
    } else {
        printf("Integrity Check:  FAILED (Query was corrupted during alignment)\n");
    }
    printf("------------------------------\n");

    free(cleaned_query);
}

void initGraph(Graph* graph, int seqSize)
{
    for (int n = 0; n < graph->num_nodes; n++) {
        // Allocate with extra padding to safeguard the final diagonal tail offsets
        size_t matrix_size = (seqSize + 2) * (graph->nodes[n].sequence.size + 2); 
        graph->nodes[n].dp_matrix = (DTYPEMATRIX*)malloc(matrix_size * sizeof(DTYPEMATRIX));
    }
}

// *************************************************************************************************
//
//                                             Main
//
// *************************************************************************************************

int main() {
    Graph graph;
    Sequence sequence, sequence_mod;
    
    /*
    if (read_gfa_graph("datasets/graphs/old/20_10.graph", &graph) != 0) { 
        fprintf(stderr, "Error encountered while reading input graph\n"); return -1; 
    }
    
    if (read_input_sequence_old("datasets/sequences/old/S_20_10.seq", &sequence, &sequence_mod) != 0) { 
        fprintf(stderr, "Error encountered while reading input sequence\n"); return -2; 
    }
    */
    
    if (read_gfa_graph("datasets/graphs/cactus-BRCA2.gfa", &graph) != 0) { 
        fprintf(stderr, "Error encountered while reading input graph\n"); return -1; 
    }

    if (read_input_sequence("datasets/sequences/cactus-BRCA2-1500.fq", "datasets/sequences/cactus-BRCA2-1500.tsv", &sequence, &sequence_mod) != 0) { 
        fprintf(stderr, "Error encountered while reading input sequence\n"); return -2; 
    }

    if (sort_graph_topologically(&graph) != 0) { 
        fprintf(stderr, "Error encountered while sorting input graph\n"); return -3; 
    }

    printf("Successfully loaded input, proceeding with verification run.\n");

    initGraph(&graph, sequence.size);

    // Verify

    AlignmentResult res = cpu_align_simd_parallel_dp(graph, sequence);
    verify_alignment(res.graph_align, res.query_align, sequence_mod);

    printf("Graph Alignment: %s\n", res.graph_align);
    printf("Query Alignment: %s\n", res.query_align);

    free(res.graph_align);
    free(res.query_align);

    printf("Test passed. Prociding with timed executions\n");

    // Time executions

    double timers[NITER];
    TIMER_DEF(0);
    
    for (int i=-WARMUP; i<NITER; i++) {

    
        TIMER_START(0);
        AlignmentResult res = cpu_align_simd_parallel_dp(graph, sequence);
        TIMER_STOP(0);

        free(res.graph_align);
        free(res.query_align);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }

    double a_mean = arithmetic_mean(timers, NITER);
    fprintf(stdout, "Arithmetic Mean: %lf\n", a_mean);


    //Maybe calculate bandwith? Idk, might be useful... or not.

    //double bytes_cpu_sequential = nnz * (sizeof(float) + sizeof(int) + sizeof(int) + sizeof(float));
    //double bandwidth_coo = bytes_coo / a_mean / 1.e9;
    //fprintf(stdout, "My GEMM-COO bandwidth %lf GB/s\n", bandwidth_coo);

    for (int i = 0; i < graph.num_nodes; i++) {
        free(graph.nodes[i].sequence.sequence);
        if(graph.nodes[i].dp_matrix) free(graph.nodes[i].dp_matrix);
        if(graph.nodes[i].v_in) free(graph.nodes[i].v_in);
        if(graph.nodes[i].v_out) free(graph.nodes[i].v_out);
    }
    free(graph.nodes);
    free(sequence.sequence);
    free(sequence_mod.sequence);
    
    return 0;
}