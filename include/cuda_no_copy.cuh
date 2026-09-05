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
// allocated once and both sides use the same pointers, so every copy disappears and only the
// launch strategy is left. One version here per GPU version of mode 1, in the same order:
//
//   version 0, 1  ->  one launch per node          (what cuda_naive does)
//   version 2 - 5 ->  one launch per level         (cuda_parallel_node ... cuda_async_batching:
//                                                   they differ only in how they schedule the
//                                                   copies, which no longer exist)
//   version 6     ->  one launch per level, shared memory kernel (cuda_shared_mem, mode 1 v7)
//   version 7     ->  one launch per level, last column only     (cuda_last_col, mode 1 v8)
//   version 8     ->  one warp per node, shared memory diagonals (cuda_warps, mode 1 v9)
//   version 9     ->  one warp per node, diagonals in registers  (cuda_registers, mode 1 v10)
//   version 10    ->  persistent warps fed by rings              (cuda_persistent_kernels, v11)
//
// Versions 0 to 6 keep a whole score matrix per node (init_shared_graph); 7 to 10 keep only a last
// column (init_shared_graph_last_col) and let the traceback recompute the nodes it walks, which is
// the only reason the graph fits in a shared allocation at all on the big datasets.
//
// The dependencies between levels are enforced by the stream order, so there is no per level
// synchronise either: one wait at the end is enough. The persistent version is the exception, it
// has to drain the rings of a level before handing out the next one.

AlignmentResult gpu_align_no_copy_naive(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_level(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_shared_mem(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_last_col(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_warps(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_registers(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_no_copy_persistent_kernels(Graph graph, Graph cudaGraph, Sequence sequence);

AlignmentResult compute_traceback_no_copy(Graph graph, Sequence sequence);
AlignmentResult compute_traceback_no_copy_last_col(Graph graph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif
