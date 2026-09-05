#include "../include/definitions.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma once

#ifndef COMMONS_H
#define COMMONS_H

#ifdef __cplusplus
extern "C" {
#endif

int init_cpu_graph(Graph* graph, int seqSize) 
{
    for (int n = 0; n < graph->num_nodes; n++) {
        // Allocate with extra padding to safeguard the final diagonal tail offsets
        size_t matrix_size = (seqSize + 2) * (graph->nodes[n].sequence.size + 2); 
        graph->nodes[n].dp_matrix = (DTYPEMATRIX*)malloc(matrix_size * sizeof(DTYPEMATRIX));
    }
}

int free_cpu_graph(Graph* graph) 
{
    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->nodes[i].sequence.sequence);
        if(graph->nodes[i].dp_matrix) free(graph->nodes[i].dp_matrix);
        if(graph->nodes[i].v_in) free(graph->nodes[i].v_in);
        if(graph->nodes[i].v_out) free(graph->nodes[i].v_out);
    }
    free(graph->nodes);
}


int init_gpu_graph(Graph* graph, Graph* cudaGraph, int seqSize)
{
    cudaError_t cudaStatus;

    cudaGraph->num_nodes = graph->num_nodes;
    cudaGraph->max_score = graph->max_score;
    cudaGraph->max_score_node_id = graph->max_score_node_id;

    Node* device_nodes = (Node*)malloc(graph->num_nodes * sizeof(Node));
    if (device_nodes == NULL) { return -1; }

    size_t num_elems = 0;

    for (int n = 0; n < graph->num_nodes; n++)
        num_elems += graph->nodes[n].sequence.size + 2;

    num_elems *= (seqSize + 2);

    DTYPEMATRIX* dpmatrices = NULL;

    cudaStatus = cudaMalloc((void**)&dpmatrices, num_elems * sizeof(DTYPEMATRIX));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "dp_matrix allocation failed!\n"); return -2; }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* d_node = &device_nodes[n];

        d_node->id = h_node->id;
        d_node->num_in = h_node->num_in;
        d_node->num_out = h_node->num_out;

        d_node->dp_matrix = dpmatrices;

        size_t matrix_size = (seqSize + 2) * (h_node->sequence.size + 2);
        dpmatrices = &dpmatrices[matrix_size];

        d_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMalloc((void**)&d_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }
        
        cudaStatus = cudaMemcpy(d_node->sequence.sequence, h_node->sequence.sequence, 
                                h_node->sequence.size * sizeof(DTYPEALPHABET), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence memcpy failed!\n"); return -3; }

        if (h_node->num_in > 0) {
            cudaStatus = cudaMalloc((void**)&d_node->v_in, h_node->num_in * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            d_node->v_in = NULL;
        }

        if (h_node->num_out > 0) {
            cudaStatus = cudaMalloc((void**)&d_node->v_out, h_node->num_out * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            d_node->v_out = NULL;
        }
    }

    cudaStatus = cudaMalloc((void**)&cudaGraph->nodes, graph->num_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaGraph->nodes allocation failed!\n"); return -2; }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* d_node = &device_nodes[n];

        if (h_node->num_in > 0) {
            Node** tmp_v_in = (Node**)malloc(h_node->num_in * sizeof(Node*));
            for (int i = 0; i < h_node->num_in; i++) {
                int neighbor_id = h_node->v_in[i]->id; 
                tmp_v_in[i] = &cudaGraph->nodes[neighbor_id];
            }
            cudaMemcpy(d_node->v_in, tmp_v_in, h_node->num_in * sizeof(Node*), cudaMemcpyHostToDevice);
            free(tmp_v_in);
        }

        if (h_node->num_out > 0) {
            Node** tmp_v_out = (Node**)malloc(h_node->num_out * sizeof(Node*));
            for (int i = 0; i < h_node->num_out; i++) {
                int neighbor_id = h_node->v_out[i]->id;
                tmp_v_out[i] = &cudaGraph->nodes[neighbor_id];
            }
            cudaMemcpy(d_node->v_out, tmp_v_out, h_node->num_out * sizeof(Node*), cudaMemcpyHostToDevice);
            free(tmp_v_out);
        }
    }

    cudaStatus = cudaMemcpy(cudaGraph->nodes, device_nodes, graph->num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "Final nodes array copy failed!\n"); return -3; }

    free(device_nodes);

    return 0;
}

int free_gpu_graph(Graph* cudaGraph)
{
    Node* device_nodes = (Node*)malloc(cudaGraph->num_nodes * sizeof(Node));

    cudaError_t status = cudaMemcpy(device_nodes, cudaGraph->nodes, 
                                    cudaGraph->num_nodes * sizeof(Node), 
                                    cudaMemcpyDeviceToHost);
    
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed during free routine. Memory might leak!\n");
        free(device_nodes);
        return -1;
    }

    cudaFree(device_nodes[0].dp_matrix);

    for (int n = 0; n < cudaGraph->num_nodes; n++) {
        Node* node = &device_nodes[n];

        cudaFree(node->sequence.sequence);
        cudaFree(node->v_in);
        cudaFree(node->v_out);
    }


    cudaFree(cudaGraph->nodes);
    free(device_nodes);
    
    cudaGraph->num_nodes = 0;

    return 0;
}


// *************************************************************************************************
//
//                              Last column graph (mode 1, v7 v8 v9)
//
// *************************************************************************************************
//
// The forward pass of the last column versions keeps no matrix at all. A node hands its successors
// one thing, the M + 1 scores of its rightmost column, and that is the only thing that has to
// survive it: 151 ints a node, 30 MB for 150_10, against the 7.80 GB the score matrices took. The
// alignment itself is recovered afterwards by recomputing the two or three nodes the path actually
// crosses, on the CPU, which is why nothing else is stored.
//
// Both sides are one contiguous allocation of num_nodes * (seqSize + 1) ints, zeroed once so that
// last_col[0], the top row, is already the zero the recurrence wants.

int init_cpu_graph_last_col(Graph* graph, int seqSize)
{
    size_t total_cols = (size_t)graph->num_nodes * (seqSize + 1);

    DTYPEMATRIX* cols = NULL;
    if (cudaMallocHost((void**)&cols, total_cols * sizeof(DTYPEMATRIX)) != cudaSuccess) return -1;

    memset(cols, 0, total_cols * sizeof(DTYPEMATRIX));

    for (int n = 0; n < graph->num_nodes; n++) {
        graph->nodes[n].last_col  = cols;
        graph->nodes[n].dp_matrix = NULL;

        cols += (seqSize + 1);
    }

    return 0;
}

int free_cpu_graph_last_col(Graph* graph)
{
    if (graph->num_nodes > 0) cudaFreeHost(graph->nodes[0].last_col);

    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->nodes[i].sequence.sequence);
        if (graph->nodes[i].v_in) free(graph->nodes[i].v_in);
        if (graph->nodes[i].v_out) free(graph->nodes[i].v_out);
    }
    free(graph->nodes);

    return 0;
}

int init_gpu_graph_last_col(Graph* graph, Graph* cudaGraph, int seqSize)
{
    cudaError_t cudaStatus;

    cudaGraph->num_nodes = graph->num_nodes;
    cudaGraph->max_score = graph->max_score;
    cudaGraph->max_score_node_id = graph->max_score_node_id;

    Node* device_nodes = (Node*)malloc(graph->num_nodes * sizeof(Node));
    if (device_nodes == NULL) { return -1; }

    size_t total_cols = (size_t)graph->num_nodes * (seqSize + 1);

    DTYPEMATRIX* lastcols = NULL;

    cudaStatus = cudaMalloc((void**)&lastcols, total_cols * sizeof(DTYPEMATRIX));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "last_col allocation failed!\n"); return -2; }

    cudaMemset(lastcols, 0, total_cols * sizeof(DTYPEMATRIX));

    DTYPEMATRIX* col_walk = lastcols;

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* d_node = &device_nodes[n];

        d_node->id = h_node->id;
        d_node->num_in = h_node->num_in;
        d_node->num_out = h_node->num_out;

        d_node->dp_matrix = NULL;
        d_node->last_col  = col_walk;

        col_walk += (seqSize + 1);

        d_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMalloc((void**)&d_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }

        cudaStatus = cudaMemcpy(d_node->sequence.sequence, h_node->sequence.sequence,
                                h_node->sequence.size * sizeof(DTYPEALPHABET), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence memcpy failed!\n"); return -3; }

        if (h_node->num_in > 0) {
            cudaStatus = cudaMalloc((void**)&d_node->v_in, h_node->num_in * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            d_node->v_in = NULL;
        }

        if (h_node->num_out > 0) {
            cudaStatus = cudaMalloc((void**)&d_node->v_out, h_node->num_out * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            d_node->v_out = NULL;
        }
    }

    cudaStatus = cudaMalloc((void**)&cudaGraph->nodes, graph->num_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaGraph->nodes allocation failed!\n"); return -2; }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* d_node = &device_nodes[n];

        if (h_node->num_in > 0) {
            Node** tmp_v_in = (Node**)malloc(h_node->num_in * sizeof(Node*));
            for (int i = 0; i < h_node->num_in; i++)
                tmp_v_in[i] = &cudaGraph->nodes[h_node->v_in[i]->id];
            cudaMemcpy(d_node->v_in, tmp_v_in, h_node->num_in * sizeof(Node*), cudaMemcpyHostToDevice);
            free(tmp_v_in);
        }

        if (h_node->num_out > 0) {
            Node** tmp_v_out = (Node**)malloc(h_node->num_out * sizeof(Node*));
            for (int i = 0; i < h_node->num_out; i++)
                tmp_v_out[i] = &cudaGraph->nodes[h_node->v_out[i]->id];
            cudaMemcpy(d_node->v_out, tmp_v_out, h_node->num_out * sizeof(Node*), cudaMemcpyHostToDevice);
            free(tmp_v_out);
        }
    }

    cudaStatus = cudaMemcpy(cudaGraph->nodes, device_nodes, graph->num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "Final nodes array copy failed!\n"); return -3; }

    free(device_nodes);

    return 0;
}

int free_gpu_graph_last_col(Graph* cudaGraph)
{
    Node* device_nodes = (Node*)malloc(cudaGraph->num_nodes * sizeof(Node));

    cudaError_t status = cudaMemcpy(device_nodes, cudaGraph->nodes,
                                    cudaGraph->num_nodes * sizeof(Node),
                                    cudaMemcpyDeviceToHost);

    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed during free routine. Memory might leak!\n");
        free(device_nodes);
        return -1;
    }

    cudaFree(device_nodes[0].last_col);

    for (int n = 0; n < cudaGraph->num_nodes; n++) {
        cudaFree(device_nodes[n].sequence.sequence);
        cudaFree(device_nodes[n].v_in);
        cudaFree(device_nodes[n].v_out);
    }

    cudaFree(cudaGraph->nodes);
    free(device_nodes);

    cudaGraph->num_nodes = 0;

    return 0;
}

int init_cpu_graph_pinned(Graph* graph, int seqSize) 
{

    size_t num_elems = 0;
    for (int n = 0; n < graph->num_nodes; n++) {
        num_elems += (graph->nodes[n].sequence.size + 2);
    }
    num_elems *= (seqSize + 2);

    DTYPEMATRIX* dpmatrices = NULL;
    cudaMallocHost((void**)&dpmatrices, num_elems * sizeof(DTYPEMATRIX));

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* node = &graph->nodes[n];
        // Allocate with extra padding to safeguard the final diagonal tail offsets
        
        graph->nodes[n].dp_matrix = dpmatrices;

        size_t matrix_size = (seqSize + 2) * (graph->nodes[n].sequence.size + 2);
        dpmatrices = &dpmatrices[matrix_size];

        char* tmp;
        cudaMallocHost((void**)&tmp, node->sequence.size * sizeof(char));
        memcpy(tmp, node->sequence.sequence, node->sequence.size * sizeof(char));

        free(node->sequence.sequence);
        node->sequence.sequence = tmp;
    }

    return 0;
}

int free_cpu_graph_pinned(Graph* graph)
{
    cudaFreeHost(graph->nodes[0].dp_matrix); // original adress of the buffer containing all dp matrices

    for (int i = 0; i < graph->num_nodes; i++) {
        cudaFreeHost(graph->nodes[i].sequence.sequence);
        free(graph->nodes[i].v_in);
        free(graph->nodes[i].v_out);
    }
    free(graph->nodes);

    return 0;
}


int init_hybrid_graph(Graph* graph, Graph* cudaGraph, int seqSize)
{
    // A hybrid run needs both sides at once: the pinned host matrices the CPU levels write into and
    // the device matrices the GPU levels write into, laid out identically (same node order, same
    // padding), so moving a level across is one copy of a contiguous range.
    int status = init_cpu_graph_pinned(graph, seqSize);
    if (status != 0) return status;

    return init_gpu_graph(graph, cudaGraph, seqSize);
}

int free_hybrid_graph(Graph* graph, Graph* cudaGraph)
{
    free_gpu_graph(cudaGraph);

    return free_cpu_graph_pinned(graph);
}


int init_unified_graph(Graph* graph, Graph* cudaGraph, int seqSize)
{
    cudaError_t cudaStatus;

    // One graph for both sides. The node array, the sequences and the matrices are all managed, so
    // the CPU levels and the kernels work on the very same structs: there is no host copy and no
    // device copy to keep in sync, which is the whole point on a machine where both processors sit
    // behind the same memory.
    Node* unified_nodes = NULL;
    cudaStatus = cudaMallocManaged((void**)&unified_nodes, graph->num_nodes * sizeof(Node), cudaMemAttachGlobal);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "unified nodes allocation failed!\n"); return -2; }

    size_t num_elems = 0;

    for (int n = 0; n < graph->num_nodes; n++)
        num_elems += graph->nodes[n].sequence.size + 2;

    num_elems *= (seqSize + 2);

    DTYPEMATRIX* dpmatrices = NULL;

    cudaStatus = cudaMallocManaged((void**)&dpmatrices, num_elems * sizeof(DTYPEMATRIX), cudaMemAttachGlobal);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "dp_matrix allocation failed!\n"); return -2; }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* u_node = &unified_nodes[n];

        u_node->id = h_node->id;
        u_node->depth = h_node->depth;
        u_node->num_in = h_node->num_in;
        u_node->num_out = h_node->num_out;

        u_node->dp_matrix = dpmatrices;

        size_t matrix_size = (seqSize + 2) * (h_node->sequence.size + 2);
        dpmatrices = &dpmatrices[matrix_size];

        u_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMallocManaged((void**)&u_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET), cudaMemAttachGlobal);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }

        memcpy(u_node->sequence.sequence, h_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));

        if (h_node->num_in > 0) {
            cudaStatus = cudaMallocManaged((void**)&u_node->v_in, h_node->num_in * sizeof(Node*), cudaMemAttachGlobal);
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            u_node->v_in = NULL;
        }

        if (h_node->num_out > 0) {
            cudaStatus = cudaMallocManaged((void**)&u_node->v_out, h_node->num_out * sizeof(Node*), cudaMemAttachGlobal);
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            u_node->v_out = NULL;
        }
    }

    // Neighbours point into the unified array itself, so the same pointer is valid on both sides.
    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* u_node = &unified_nodes[n];

        for (int i = 0; i < h_node->num_in; i++)
            u_node->v_in[i] = &unified_nodes[h_node->v_in[i]->id];

        for (int i = 0; i < h_node->num_out; i++)
            u_node->v_out[i] = &unified_nodes[h_node->v_out[i]->id];
    }

    // The graph read from the file is replaced by the unified one, and the device graph *is* it.
    for (int n = 0; n < graph->num_nodes; n++) {
        free(graph->nodes[n].sequence.sequence);
        free(graph->nodes[n].v_in);
        free(graph->nodes[n].v_out);
    }
    free(graph->nodes);

    graph->nodes = unified_nodes;

    cudaGraph->nodes = unified_nodes;
    cudaGraph->num_nodes = graph->num_nodes;
    cudaGraph->max_score = graph->max_score;
    cudaGraph->max_score_node_id = graph->max_score_node_id;

    return 0;
}

int init_pinned_graph(Graph* graph, Graph* cudaGraph, int seqSize)
{
    cudaError_t cudaStatus;

    // Same single shared graph as init_unified_graph, but in pinned host memory instead of managed
    // memory. Unified addressing lets the kernels dereference these pointers directly, and because
    // the pages are page locked in system memory they never migrate: the CPU and the GPU always
    // find the data where it already is. Nothing is copied and nothing moves, so a hand over
    // between a GPU level and a CPU level costs exactly one stream synchronise.
    Node* pinned_nodes = NULL;
    cudaStatus = cudaMallocHost((void**)&pinned_nodes, graph->num_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "pinned nodes allocation failed!\n"); return -2; }

    size_t num_elems = 0;

    for (int n = 0; n < graph->num_nodes; n++)
        num_elems += graph->nodes[n].sequence.size + 2;

    num_elems *= (seqSize + 2);

    DTYPEMATRIX* dpmatrices = NULL;

    cudaStatus = cudaMallocHost((void**)&dpmatrices, num_elems * sizeof(DTYPEMATRIX));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "dp_matrix allocation failed!\n"); return -2; }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* p_node = &pinned_nodes[n];

        p_node->id = h_node->id;
        p_node->depth = h_node->depth;
        p_node->num_in = h_node->num_in;
        p_node->num_out = h_node->num_out;

        p_node->dp_matrix = dpmatrices;

        size_t matrix_size = (seqSize + 2) * (h_node->sequence.size + 2);
        dpmatrices = &dpmatrices[matrix_size];

        p_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMallocHost((void**)&p_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }

        memcpy(p_node->sequence.sequence, h_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));

        if (h_node->num_in > 0) {
            cudaStatus = cudaMallocHost((void**)&p_node->v_in, h_node->num_in * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            p_node->v_in = NULL;
        }

        if (h_node->num_out > 0) {
            cudaStatus = cudaMallocHost((void**)&p_node->v_out, h_node->num_out * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            p_node->v_out = NULL;
        }
    }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* p_node = &pinned_nodes[n];

        for (int i = 0; i < h_node->num_in; i++)
            p_node->v_in[i] = &pinned_nodes[h_node->v_in[i]->id];

        for (int i = 0; i < h_node->num_out; i++)
            p_node->v_out[i] = &pinned_nodes[h_node->v_out[i]->id];
    }

    for (int n = 0; n < graph->num_nodes; n++) {
        free(graph->nodes[n].sequence.sequence);
        free(graph->nodes[n].v_in);
        free(graph->nodes[n].v_out);
    }
    free(graph->nodes);

    graph->nodes = pinned_nodes;

    cudaGraph->nodes = pinned_nodes;
    cudaGraph->num_nodes = graph->num_nodes;
    cudaGraph->max_score = graph->max_score;
    cudaGraph->max_score_node_id = graph->max_score_node_id;

    return 0;
}

// *************************************************************************************************
//
//                             Multi sequence graph (mode 4)
//
// *************************************************************************************************
//
// Same as the last column graph, with a read dimension: a node carries num_reads last columns
// instead of one, laid out read after read so that consecutive warps of the kernel, which take
// consecutive reads of the same node, land on neighbouring slices. R * 30 MB for 150_10, which is
// the reason the batch is possible at all - the score matrices were 7.80 GB a read.

int init_cpu_graph_multi(Graph* graph, int seqSize, int num_reads)
{
    size_t total = (size_t)graph->num_nodes * num_reads * (seqSize + 1);

    DTYPEMATRIX* cols = NULL;
    if (cudaMallocHost((void**)&cols, total * sizeof(DTYPEMATRIX)) != cudaSuccess) return -1;

    memset(cols, 0, total * sizeof(DTYPEMATRIX));

    for (int n = 0; n < graph->num_nodes; n++) {
        graph->nodes[n].last_col  = cols;
        graph->nodes[n].dp_matrix = NULL;

        cols += (size_t)num_reads * (seqSize + 1);
    }

    return 0;
}

// Plain host allocation of the same layout, for the CPU baseline of mode 0. Nothing on that path
// ever touches the device, so page locking the pages would only make a large batch fail to
// allocate for no benefit.

int init_cpu_graph_multi_plain(Graph* graph, int seqSize, int num_reads)
{
    size_t total = (size_t)graph->num_nodes * num_reads * (seqSize + 1);

    DTYPEMATRIX* cols = (DTYPEMATRIX*)calloc(total, sizeof(DTYPEMATRIX));
    if (cols == NULL) return -1;

    for (int n = 0; n < graph->num_nodes; n++) {
        graph->nodes[n].last_col  = cols;
        graph->nodes[n].dp_matrix = NULL;

        cols += (size_t)num_reads * (seqSize + 1);
    }

    return 0;
}

int free_cpu_graph_multi_plain(Graph* graph)
{
    if (graph->num_nodes > 0) free(graph->nodes[0].last_col);

    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->nodes[i].sequence.sequence);
        if (graph->nodes[i].v_in) free(graph->nodes[i].v_in);
        if (graph->nodes[i].v_out) free(graph->nodes[i].v_out);
    }
    free(graph->nodes);

    return 0;
}

int free_cpu_graph_multi(Graph* graph)
{
    return free_cpu_graph_last_col(graph);
}

int init_gpu_graph_multi(Graph* graph, Graph* cudaGraph, int seqSize, int num_reads)
{
    cudaError_t cudaStatus;

    cudaGraph->num_nodes = graph->num_nodes;

    Node* device_nodes = (Node*)malloc(graph->num_nodes * sizeof(Node));
    if (device_nodes == NULL) { return -1; }

    size_t total = (size_t)graph->num_nodes * num_reads * (seqSize + 1);

    DTYPEMATRIX* lastcols = NULL;
    cudaStatus = cudaMalloc((void**)&lastcols, total * sizeof(DTYPEMATRIX));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "last_col allocation failed!\n"); return -2; }

    cudaMemset(lastcols, 0, total * sizeof(DTYPEMATRIX));

    DTYPEMATRIX* col_walk = lastcols;

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* d_node = &device_nodes[n];

        d_node->id = h_node->id;
        d_node->num_in = h_node->num_in;
        d_node->num_out = h_node->num_out;

        d_node->dp_matrix = NULL;
        d_node->last_col  = col_walk;

        col_walk += (size_t)num_reads * (seqSize + 1);

        d_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMalloc((void**)&d_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }

        cudaMemcpy(d_node->sequence.sequence, h_node->sequence.sequence,
                   h_node->sequence.size * sizeof(DTYPEALPHABET), cudaMemcpyHostToDevice);

        if (h_node->num_in > 0) {
            cudaMalloc((void**)&d_node->v_in, h_node->num_in * sizeof(Node*));
        } else d_node->v_in = NULL;

        if (h_node->num_out > 0) {
            cudaMalloc((void**)&d_node->v_out, h_node->num_out * sizeof(Node*));
        } else d_node->v_out = NULL;
    }

    cudaStatus = cudaMalloc((void**)&cudaGraph->nodes, graph->num_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaGraph->nodes allocation failed!\n"); return -2; }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* d_node = &device_nodes[n];

        if (h_node->num_in > 0) {
            Node** tmp_v_in = (Node**)malloc(h_node->num_in * sizeof(Node*));
            for (int i = 0; i < h_node->num_in; i++) tmp_v_in[i] = &cudaGraph->nodes[h_node->v_in[i]->id];
            cudaMemcpy(d_node->v_in, tmp_v_in, h_node->num_in * sizeof(Node*), cudaMemcpyHostToDevice);
            free(tmp_v_in);
        }

        if (h_node->num_out > 0) {
            Node** tmp_v_out = (Node**)malloc(h_node->num_out * sizeof(Node*));
            for (int i = 0; i < h_node->num_out; i++) tmp_v_out[i] = &cudaGraph->nodes[h_node->v_out[i]->id];
            cudaMemcpy(d_node->v_out, tmp_v_out, h_node->num_out * sizeof(Node*), cudaMemcpyHostToDevice);
            free(tmp_v_out);
        }
    }

    cudaMemcpy(cudaGraph->nodes, device_nodes, graph->num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    free(device_nodes);

    return 0;
}

int free_gpu_graph_multi(Graph* cudaGraph)
{
    return free_gpu_graph_last_col(cudaGraph);
}

// *************************************************************************************************
//
//                          Pinned last column graph (mode 2, v4 v5)
//
// *************************************************************************************************
//
// One shared graph in pinned host memory, like init_pinned_graph, carrying last columns instead of
// score matrices. Unified addressing lets the kernels dereference these pointers directly and the
// page locked pages never migrate, so a hand over between a GPU level and a CPU level costs one
// stream synchronise and no copy at all. What crosses the boundary is 604 bytes a node, which is
// why pinned memory is enough here and nothing more elaborate is needed.

int init_pinned_graph_last_col(Graph* graph, Graph* cudaGraph, int seqSize)
{
    cudaError_t cudaStatus;

    Node* pinned_nodes = NULL;
    cudaStatus = cudaMallocHost((void**)&pinned_nodes, graph->num_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "pinned nodes allocation failed!\n"); return -2; }

    size_t total_cols = (size_t)graph->num_nodes * (seqSize + 1);

    DTYPEMATRIX* lastcols = NULL;
    cudaStatus = cudaMallocHost((void**)&lastcols, total_cols * sizeof(DTYPEMATRIX));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "last_col allocation failed!\n"); return -2; }

    memset(lastcols, 0, total_cols * sizeof(DTYPEMATRIX));

    DTYPEMATRIX* col_walk = lastcols;

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* p_node = &pinned_nodes[n];

        p_node->id = h_node->id;
        p_node->depth = h_node->depth;
        p_node->num_in = h_node->num_in;
        p_node->num_out = h_node->num_out;

        p_node->dp_matrix = NULL;
        p_node->last_col = col_walk;

        col_walk += (seqSize + 1);

        p_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMallocHost((void**)&p_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }

        memcpy(p_node->sequence.sequence, h_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));

        if (h_node->num_in > 0) {
            cudaStatus = cudaMallocHost((void**)&p_node->v_in, h_node->num_in * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            p_node->v_in = NULL;
        }

        if (h_node->num_out > 0) {
            cudaStatus = cudaMallocHost((void**)&p_node->v_out, h_node->num_out * sizeof(Node*));
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            p_node->v_out = NULL;
        }
    }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* p_node = &pinned_nodes[n];

        for (int i = 0; i < h_node->num_in; i++)
            p_node->v_in[i] = &pinned_nodes[h_node->v_in[i]->id];

        for (int i = 0; i < h_node->num_out; i++)
            p_node->v_out[i] = &pinned_nodes[h_node->v_out[i]->id];
    }

    for (int n = 0; n < graph->num_nodes; n++) {
        free(graph->nodes[n].sequence.sequence);
        free(graph->nodes[n].v_in);
        free(graph->nodes[n].v_out);
    }
    free(graph->nodes);

    graph->nodes = pinned_nodes;

    cudaGraph->nodes = pinned_nodes;
    cudaGraph->num_nodes = graph->num_nodes;
    cudaGraph->max_score = graph->max_score;
    cudaGraph->max_score_node_id = graph->max_score_node_id;

    return 0;
}

int free_pinned_graph_last_col(Graph* graph, Graph* cudaGraph)
{
    if (graph->num_nodes > 0) cudaFreeHost(graph->nodes[0].last_col);

    for (int n = 0; n < graph->num_nodes; n++) {
        cudaFreeHost(graph->nodes[n].sequence.sequence);
        cudaFreeHost(graph->nodes[n].v_in);
        cudaFreeHost(graph->nodes[n].v_out);
    }

    cudaFreeHost(graph->nodes);

    graph->nodes = NULL;
    graph->num_nodes = 0;
    cudaGraph->nodes = NULL;
    cudaGraph->num_nodes = 0;

    return 0;
}

int free_pinned_graph(Graph* graph, Graph* cudaGraph)
{
    cudaFreeHost(graph->nodes[0].dp_matrix); // original adress of the buffer containing all dp matrices

    for (int n = 0; n < graph->num_nodes; n++) {
        cudaFreeHost(graph->nodes[n].sequence.sequence);
        cudaFreeHost(graph->nodes[n].v_in);
        cudaFreeHost(graph->nodes[n].v_out);
    }

    cudaFreeHost(graph->nodes);

    graph->nodes = NULL;
    cudaGraph->nodes = NULL;
    cudaGraph->num_nodes = 0;

    return 0;
}

int free_unified_graph(Graph* graph, Graph* cudaGraph)
{
    cudaFree(graph->nodes[0].dp_matrix); // original adress of the buffer containing all dp matrices

    for (int n = 0; n < graph->num_nodes; n++) {
        cudaFree(graph->nodes[n].sequence.sequence);
        cudaFree(graph->nodes[n].v_in);
        cudaFree(graph->nodes[n].v_out);
    }

    cudaFree(graph->nodes);

    graph->nodes = NULL;
    cudaGraph->nodes = NULL;
    cudaGraph->num_nodes = 0;

    return 0;
}

// Pins the migration policy of a managed range: the pages stay in system memory and the device
// gets a mapping to them, instead of the driver moving the range to whoever touched it last.
// CUDA 13 changed cudaMemAdvise to take a cudaMemLocation, so both spellings are kept.
static void keep_in_system_memory(void* ptr, size_t bytes)
{
    int device = 0;
    cudaGetDevice(&device);

#if CUDART_VERSION >= 13000
    struct cudaMemLocation host_location;
    struct cudaMemLocation device_location;

    host_location.type = cudaMemLocationTypeHost;
    host_location.id = 0;
    device_location.type = cudaMemLocationTypeDevice;
    device_location.id = device;

    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, host_location);
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device_location);
#else
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device);
#endif
}

int init_advised_graph(Graph* graph, Graph* cudaGraph, int seqSize)
{
    // Managed memory again, but with the migration policy nailed down: the pages are told to stay
    // in system memory and to be mapped into the device page tables. On a machine where both
    // processors sit behind the same memory that is what "shared" should mean - without the advice
    // the driver is free to migrate a page to whoever touched it last, which is what makes the
    // hand over between GPU and CPU levels cost a variable amount of time.
    int status = init_unified_graph(graph, cudaGraph, seqSize);
    if (status != 0) return status;

    size_t num_elems = 0;
    for (int n = 0; n < graph->num_nodes; n++)
        num_elems += graph->nodes[n].sequence.size + 2;
    num_elems *= (seqSize + 2);

    keep_in_system_memory(graph->nodes[0].dp_matrix, num_elems * sizeof(DTYPEMATRIX));
    keep_in_system_memory(graph->nodes, graph->num_nodes * sizeof(Node));

    return 0;
}

int free_advised_graph(Graph* graph, Graph* cudaGraph)
{
    return free_unified_graph(graph, cudaGraph);
}

// *************************************************************************************************
//
//                          Shared graph, last column layout (mode 3)
//
// *************************************************************************************************
//
// Same idea as init_unified_graph / init_pinned_graph, but carrying one last column per node
// instead of a full score matrix, which is what the last column, warps, registers and persistent
// kernel cores expect. 30 MB instead of 7.8 GB for 150_10, and since the graph is shared there is
// nothing to copy back before the traceback recomputes the few nodes it walks.

int init_unified_graph_last_col(Graph* graph, Graph* cudaGraph, int seqSize)
{
    cudaError_t cudaStatus;

    Node* unified_nodes = NULL;
    cudaStatus = cudaMallocManaged((void**)&unified_nodes, graph->num_nodes * sizeof(Node), cudaMemAttachGlobal);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "unified nodes allocation failed!\n"); return -2; }

    size_t total_cols = (size_t)graph->num_nodes * (seqSize + 1);

    DTYPEMATRIX* lastcols = NULL;
    cudaStatus = cudaMallocManaged((void**)&lastcols, total_cols * sizeof(DTYPEMATRIX), cudaMemAttachGlobal);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "last_col allocation failed!\n"); return -2; }

    memset(lastcols, 0, total_cols * sizeof(DTYPEMATRIX));

    DTYPEMATRIX* col_walk = lastcols;

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* u_node = &unified_nodes[n];

        u_node->id = h_node->id;
        u_node->depth = h_node->depth;
        u_node->num_in = h_node->num_in;
        u_node->num_out = h_node->num_out;

        u_node->dp_matrix = NULL;
        u_node->last_col = col_walk;

        col_walk += (seqSize + 1);

        u_node->sequence.size = h_node->sequence.size;
        cudaStatus = cudaMallocManaged((void**)&u_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET), cudaMemAttachGlobal);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "sequence allocation failed!\n"); return -2; }

        memcpy(u_node->sequence.sequence, h_node->sequence.sequence, h_node->sequence.size * sizeof(DTYPEALPHABET));

        if (h_node->num_in > 0) {
            cudaStatus = cudaMallocManaged((void**)&u_node->v_in, h_node->num_in * sizeof(Node*), cudaMemAttachGlobal);
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            u_node->v_in = NULL;
        }

        if (h_node->num_out > 0) {
            cudaStatus = cudaMallocManaged((void**)&u_node->v_out, h_node->num_out * sizeof(Node*), cudaMemAttachGlobal);
            if (cudaStatus != cudaSuccess) { return -2; }
        } else {
            u_node->v_out = NULL;
        }
    }

    for (int n = 0; n < graph->num_nodes; n++) {
        Node* h_node = &graph->nodes[n];
        Node* u_node = &unified_nodes[n];

        for (int i = 0; i < h_node->num_in; i++)
            u_node->v_in[i] = &unified_nodes[h_node->v_in[i]->id];

        for (int i = 0; i < h_node->num_out; i++)
            u_node->v_out[i] = &unified_nodes[h_node->v_out[i]->id];
    }

    for (int n = 0; n < graph->num_nodes; n++) {
        free(graph->nodes[n].sequence.sequence);
        free(graph->nodes[n].v_in);
        free(graph->nodes[n].v_out);
    }
    free(graph->nodes);

    graph->nodes = unified_nodes;

    cudaGraph->nodes = unified_nodes;
    cudaGraph->num_nodes = graph->num_nodes;
    cudaGraph->max_score = graph->max_score;
    cudaGraph->max_score_node_id = graph->max_score_node_id;

    return 0;
}

int free_unified_graph_last_col(Graph* graph, Graph* cudaGraph)
{
    if (graph->num_nodes > 0) cudaFree(graph->nodes[0].last_col);

    for (int n = 0; n < graph->num_nodes; n++) {
        cudaFree(graph->nodes[n].sequence.sequence);
        cudaFree(graph->nodes[n].v_in);
        cudaFree(graph->nodes[n].v_out);
    }

    cudaFree(graph->nodes);

    graph->nodes = NULL;
    graph->num_nodes = 0;
    cudaGraph->nodes = NULL;
    cudaGraph->num_nodes = 0;

    return 0;
}

int init_advised_graph_last_col(Graph* graph, Graph* cudaGraph, int seqSize)
{
    int status = init_unified_graph_last_col(graph, cudaGraph, seqSize);
    if (status != 0) return status;

    size_t total_cols = (size_t)graph->num_nodes * (seqSize + 1);

    keep_in_system_memory(graph->nodes[0].last_col, total_cols * sizeof(DTYPEMATRIX));
    keep_in_system_memory(graph->nodes, graph->num_nodes * sizeof(Node));

    return 0;
}

int free_advised_graph_last_col(Graph* graph, Graph* cudaGraph)
{
    return free_unified_graph_last_col(graph, cudaGraph);
}

// One shared graph for the GPU only versions of mode 3. Which allocator is behind it is chosen at
// run time, because the answer is a property of the machine and not of the algorithm:
//
//   SHARED_MEM_KIND=advised   managed + preferred location host + accessed by device  (default)
//   SHARED_MEM_KIND=pinned    cudaMallocHost, the pages are page locked and never move
//   SHARED_MEM_KIND=managed   plain cudaMallocManaged, the driver migrates as it sees fit
//
// The name is printed once so a log always says which one produced the numbers.
int init_shared_graph(Graph* graph, Graph* cudaGraph, int seqSize)
{
    const char* kind = getenv("SHARED_MEM_KIND");
    if (kind == NULL) kind = "advised";

    fprintf(stdout, "Shared graph allocated with SHARED_MEM_KIND=%s\n", kind);

    if (strcmp(kind, "pinned") == 0)  return init_pinned_graph(graph, cudaGraph, seqSize);
    if (strcmp(kind, "managed") == 0) return init_unified_graph(graph, cudaGraph, seqSize);

    return init_advised_graph(graph, cudaGraph, seqSize);
}

int free_shared_graph(Graph* graph, Graph* cudaGraph)
{
    const char* kind = getenv("SHARED_MEM_KIND");
    if (kind == NULL) kind = "advised";

    if (strcmp(kind, "pinned") == 0) return free_pinned_graph(graph, cudaGraph);

    return free_unified_graph(graph, cudaGraph);
}

// The same choice of allocator, for the versions of mode 3 that keep only a last column per node.
int init_shared_graph_last_col(Graph* graph, Graph* cudaGraph, int seqSize)
{
    const char* kind = getenv("SHARED_MEM_KIND");
    if (kind == NULL) kind = "advised";

    fprintf(stdout, "Shared last column graph allocated with SHARED_MEM_KIND=%s\n", kind);

    if (strcmp(kind, "pinned") == 0)  return init_pinned_graph_last_col(graph, cudaGraph, seqSize);
    if (strcmp(kind, "managed") == 0) return init_unified_graph_last_col(graph, cudaGraph, seqSize);

    return init_advised_graph_last_col(graph, cudaGraph, seqSize);
}

int free_shared_graph_last_col(Graph* graph, Graph* cudaGraph)
{
    const char* kind = getenv("SHARED_MEM_KIND");
    if (kind == NULL) kind = "advised";

    if (strcmp(kind, "pinned") == 0) return free_pinned_graph_last_col(graph, cudaGraph);

    return free_unified_graph_last_col(graph, cudaGraph);
}

#ifdef __cplusplus
}
#endif

#endif

