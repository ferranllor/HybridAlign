#pragma once
#include "../include/definitions.h"

void compute_dp_cpu_last_col(Node* node, Sequence sequence, const DTYPEALPHABET* query_rev,
                             size_t col_offset, DTYPEMATRIX* scratch,
                             int* best_score, int* best_d, int* best_j);

void recompute_node_dp_last_col(Node* node, const DTYPEALPHABET* query, int M,
                                size_t col_offset, DTYPEMATRIX* dp);

AlignmentResult traceback_last_col(Graph graph, const DTYPEALPHABET* query, int M,
                                   size_t col_offset, Node* start_node, int start_i, int start_j);

AlignmentResult cpu_align_last_col(Graph graph, Sequence sequence);
