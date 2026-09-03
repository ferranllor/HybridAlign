#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_MULTI_H
#define CUDA_MULTI_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_multi(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult compute_traceback_gpu_multi(Graph graph, Sequence sequence, int read, int num_reads,
                                            const int* max_score, const int* max_d, const int* max_j,
                                            const DTYPEALPHABET* reads);

int multi_num_reads(void);
void multi_build_reads(Sequence sequence, int num_reads, DTYPEALPHABET* reads, DTYPEALPHABET* reads_rev);

#ifdef __cplusplus
}
#endif

#endif
