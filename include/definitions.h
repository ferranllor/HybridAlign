#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#pragma once

#define DTYPEALPHABET char
#define DTYPEMATRIX int
#define INT_MIN 2147483647/2

#define WARMUP 2
#define NITER 10

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

typedef struct Node {
    int id;

    Node** v_in; int num_in;
    Node** v_out; int num_out;

    DTYPEALPHABET* sequence; int sequence_size;
    DTYPEMATRIX* dp_matrix;

    int max_score; int max_score_i; int max_score_j;
} Node;

typedef struct Graph { 
    Node* nodes; int num_nodes;

    int max_score; int max_score_node_id;
} Graph;

typedef struct Sequence { 
    char* sequence; int size;
} Sequence;

typedef struct {
    char* graph_align;
    char* query_align;

    int size;
} AlignmentResult;

// Helper structure to map GFA string IDs to integer IDs
typedef struct {
    char name[256];
    int index;
} IDMap;