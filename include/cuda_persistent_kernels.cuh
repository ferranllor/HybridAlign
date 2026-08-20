#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_PERSISTENT_KERNELS_H
#define CUDA_PERSISTENT_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_persistent_kernels(Graph graph, Graph cudaGraph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif

