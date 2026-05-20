#pragma once
#include "../include/definitions.h"
#include "../include/utils.h"

AlignmentResult cpu_align_sequential(Graph graph, Sequence sequence);
void compute_dp_cpu_sequential(Node* node, Sequence sequence);
AlignmentResult compute_traceback_cpu_sequential(Graph graph, Sequence sequence);
void reverse_string(char* str, int len);
