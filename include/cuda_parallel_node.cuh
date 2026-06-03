#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_PARALLEL_NODE_H
#define CUDA_PARALLEL_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_parallel_node(Graph graph, Graph cudaGraph, Sequence sequence);
__global__ void compute_dp_gpu_parallel_node(Node* node, Sequence sequence, Sequence sequence_rev);
AlignmentResult compute_traceback_gpu_parallel_node(Graph graph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif

