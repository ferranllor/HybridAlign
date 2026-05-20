#pragma once
#include "../include/definitions.h"
#include "../include/utils.h"

AlignmentResult cpu_align_simd(Graph graph, Sequence sequence);
void compute_dp_cpu_simd(Node* node, Sequence sequence);
AlignmentResult compute_traceback_cpu_simd(Graph graph, Sequence sequence);
void reverse_string(char* str, int len);
