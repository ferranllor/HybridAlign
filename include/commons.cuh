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
    
#ifdef __cplusplus
}
#endif

#endif

