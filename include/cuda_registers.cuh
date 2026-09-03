#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_REGISTERS_H
#define CUDA_REGISTERS_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_registers(Graph graph, Graph cudaGraph, Sequence sequence);
__global__ void compute_dp_gpu_registers(Node* nodes, int level_nodes, Sequence sequence, Sequence sequence_rev, int elems_per_warp, int toprow_slots);
AlignmentResult compute_traceback_gpu_registers(Graph graph, Sequence sequence);

int registers_toprow_slots(int max_node_size, int M);
int registers_elems_per_warp(int max_node_size, int M);

#ifdef __cplusplus
}
#endif

#endif
