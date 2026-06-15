#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>

#pragma once

#define DTYPEALPHABET char
#define DTYPEMATRIX int

#define WARMUP 2
#define NITER 10
#define CHUNKSIZE 1024

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

    int max_score; int max_score_i; int max_score_j; int max_score_d;
} Node;

typedef struct Communicator {
    int id;
    bool* job_ready, *job_done, *done;

    int* nodeId;

    // TODO: Add stuff to comunicate to GPU that some columns are already done on the CPU, mainly small diagonals, and maybe also tell to stop earlier
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