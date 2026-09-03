#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_WARPS_H
#define CUDA_WARPS_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_warps(Graph graph, Graph cudaGraph, Sequence sequence);
__global__ void compute_dp_gpu_warps(Node* nodes, int level_nodes, Sequence sequence, Sequence sequence_rev, int elems_per_warp, int toprow_slots, int band_width);
AlignmentResult compute_traceback_gpu_warps(Graph graph, Sequence sequence);

int warps_toprow_slots(int max_node_size, int M);
int warps_elems_per_warp(int max_node_size, int M, int band_width);

#ifdef __cplusplus
}
#endif

#endif
