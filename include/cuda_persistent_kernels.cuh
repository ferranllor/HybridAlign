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

// Exported so that the no-copy version of mode 3 launches the very same workers instead of keeping
// its own copy of them: what changes there is only the scheduler around the rings, not the worker.
__global__ void worker(Communicator** communicators, Node* nodes, Sequence sequence,
                       Sequence sequence_rev, int elems_per_warp, int toprow_slots);

#ifdef __cplusplus
}
#endif

#endif

