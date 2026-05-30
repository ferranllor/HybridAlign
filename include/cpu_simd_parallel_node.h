#pragma once
#include "../include/definitions.h"
#include "../include/utils.h"

AlignmentResult cpu_align_simd_parallel_node(Graph graph, Sequence sequence);
void compute_dp_cpu_simd_parallel_node(Node* node, Sequence sequence);
AlignmentResult compute_traceback_cpu_simd_parallel_node(Graph graph, Sequence sequence);
