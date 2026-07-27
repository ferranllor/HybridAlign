#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_ASYNC_MONOLITHIC_H
#define CUDA_ASYNC_MONOLITHIC_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_async_monolithic(Graph graph, Graph cudaGraph, Sequence sequence);
__global__ void compute_dp_gpu_async_monolithic(Node* node, Sequence sequence, Sequence sequence_rev);
AlignmentResult compute_traceback_gpu_async_monolithic(Graph graph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif

