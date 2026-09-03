#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"
#include "../include/hybrid_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef HYBRID_LAST_COL_H
#define HYBRID_LAST_COL_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_hybrid_warps(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult gpu_align_hybrid_registers(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult compute_traceback_hybrid_last_col(Graph graph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif
