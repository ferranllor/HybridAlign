#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma once

// Levels holding fewer nodes than this are computed on the CPU instead of the GPU: one block per
// node means a small level leaves the GPU almost empty, and the launch plus the level sync cost
// more than the whole level does on 2 CPU threads.
// Overridable from the command line (-DHYBRID_MIN_NODES=...) so the sweeps can move it.
#ifndef HYBRID_MIN_NODES
#define HYBRID_MIN_NODES 8
#endif

// Levels are grouped into one copy command until their matrices reach this many bytes. Keeping the
// batches bounded in size (instead of in number of levels) is what lets the first results come back
// early: the first levels are the widest ones, so a fixed level count puts most of the graph in the
// very first copy and nothing can start on the CPU until it lands.
#ifndef BATCH_BYTES
#define BATCH_BYTES ((size_t)64 * 1024 * 1024)
#endif
