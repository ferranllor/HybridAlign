#pragma once
#include "../include/definitions.h"

AlignmentResult cpu_align_multi(Graph graph, Sequence sequence);
AlignmentResult compute_traceback_cpu_multi(Graph graph, Sequence sequence, int read, int num_reads,
                                            const int* max_score, const int* max_d, const int* max_j,
                                            const DTYPEALPHABET* reads);
