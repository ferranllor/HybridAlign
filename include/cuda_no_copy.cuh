#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_NO_COPY_H
#define CUDA_NO_COPY_H

#ifdef __cplusplus
extern "C" {
#endif

// GPU only versions for a machine where the CPU and the GPU share one memory. The graph is
// allocated once (init_shared_graph) and both sides use the same pointers, so every copy in
// versions 0 to 6 disappears and only the launch strategy is left:
//
//   version 0, 1  ->  one launch per node          (what cuda_naive does)
//   version 2 - 5 ->  one launch per level         (cuda_parallel_node ... cuda_async_batching:
//                                                   they differ only in how they schedule the
//                                                   copies, which no longer exist)
//   version 6     ->  one launch per level, shared memory kernel (cuda_shared_mem)
//
// The dependencies between levels are enforced by the stream order, so there is no per level
// synchronise either: one wait at the end is enough.

AlignmentResult gpu_align_no_copy_naive(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_level(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_shared_mem(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult compute_traceback_no_copy(Graph graph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif
