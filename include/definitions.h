#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>

#pragma once

#define DTYPEALPHABET char
#define DTYPEMATRIX short

// Highest valid version for each mode, used by main.c to validate the arguments and to print the
// usage. Bump these when a new version is added to the switches in main.c.
#define CPU_MAX_VERSION    5
#define GPU_MAX_VERSION    11
#define HYBRID_MAX_VERSION 5
#define NOCOPY_MAX_VERSION 10
#define MULTI_MAX_VERSION  1

#define WARMUP 1
#define NITER 3
#define CHUNKSIZE 32
#define WORKPOOLSIZE 32
#define NKERNELS 0

// How many node ids the host tries to drop into a worker's ring per visit. One doorbell write and
// one read of the worker's side of the ring then cover BATCHSIZE nodes instead of one, which is
// what the wide levels care about. Capped by the ring at WORKPOOLSIZE - 1.
#ifndef BATCHSIZE
#define BATCHSIZE 8
#endif
#define CACHELINE 128    // doorbells are padded to this so two workers never share a line
#define BACKOFF_NS 128   // how long a worker parks between two polls of its ring
#define WATCHDOG_SPINS 2000000L // PK_DEBUG only: yields to wait on a level before crying deadlock

// *************************************************************************************************
//
//                                       Struct definitions
//
// *************************************************************************************************

#define MATCH (DTYPEMATRIX)1
#define MISMATCH (DTYPEMATRIX)-1
#define GAP (DTYPEMATRIX)-1

typedef struct Node Node;
typedef struct Graph Graph;
typedef struct Sequence Sequence;

typedef struct Sequence { 
    DTYPEALPHABET* sequence; int size;
} Sequence;

typedef struct Node {
    int id, depth;

    Node** v_in; int num_in;
    Node** v_out; int num_out;

    Sequence sequence;
    DTYPEMATRIX* dp_matrix;

    // The one column of scores a successor has to read to seed its own boundary, used by the last column versions instead of dp_matrix.
    DTYPEMATRIX* last_col;

    int max_score; int max_score_i; int max_score_j; int max_score_d;
} Node;

typedef struct Communicator {
    bool *done;
    int *work_top, *work_bottom;
    int *workPool;

    // TODO: Add stuff to comunicate to GPU that some columns are already done on the CPU, mainly small diagonals, and maybe also tell to stop earlier
    // Optimisation idea, if latency is a problem, maybe try to put everything in the same cache line, except for work pool if too big I guess.
} Communicator;

typedef struct Graph { 
    Node* nodes; int num_nodes;

    int max_score; int max_score_node_id;
} Graph;

typedef struct {
    DTYPEALPHABET* graph_align;
    DTYPEALPHABET* query_align;

    int size;
} AlignmentResult;

// Helper structure to map GFA string IDs to integer IDs
typedef struct {
    char name[256];
    int index;
} IDMap;