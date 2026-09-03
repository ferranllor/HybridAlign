#pragma once
#include "../include/definitions.h"
#include "../include/cuda_utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDA_MULTI_REGISTERS_H
#define CUDA_MULTI_REGISTERS_H

#ifdef __cplusplus
extern "C" {
#endif

AlignmentResult gpu_align_multi_registers(Graph graph, Graph cudaGraph, Sequence sequence);

#ifdef __cplusplus
}
#endif

#endif
