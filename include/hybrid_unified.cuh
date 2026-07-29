#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"
#include "../include/hybrid_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef HYBRID_UNIFIED_H
#define HYBRID_UNIFIED_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_hybrid_unified(Graph graph, Graph cudaGraph, Sequence sequence);
AlignmentResult compute_traceback_hybrid_unified(Graph graph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif
