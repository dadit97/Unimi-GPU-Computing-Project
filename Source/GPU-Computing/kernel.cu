#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__device__ bool areAllTrue(bool* vector, int dimension) {
    for (int i = 0; i < dimension; i++) {
        if (vector[i] == false) return false;
    }
    return true;
}

__device__ void initializeBoolVector(bool* vector, int dimension) {
    for (int i = 0; i < dimension; i++) {
        vector[i] = false;
    }
}

__global__ void shortestPathsParallel(int* matrix, int dimension, int* results) {
    // Each Thread computes the problem for the node with its tID index
    int tID = threadIdx.x;

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)malloc((dimension) * sizeof(bool));
    initializeBoolVector(Vt, dimension);
    Vt[tID] = true;

    // l vector initialization
    int* l = (int*)malloc(dimension * sizeof(int));
    
    // Getting direct connections with source node
    for (int i = 0; i < dimension; i++) {
        l[i] = matrix[tID * dimension + i];
    }
    
    // while V != Vt
    while (!areAllTrue(Vt, dimension)) {

        int closestWeigth = 999999999;
        int closestIndex = tID;

        // Find the next vertex closest to source node
        for (int i = 0; i < dimension; i++) {
            if (Vt[i] == true) continue;
            if (l[i] < closestWeigth) {
                closestWeigth = l[i];
                closestIndex = i;
                
            }
        }
        
        // Add closest vertex to Vt
        Vt[closestIndex] = true;
        
        // Recompute l
        for (int i = 0; i < dimension; i++) {
            if (Vt[i] == true) continue;
            int uvWeight = matrix[closestIndex * dimension + i];
            l[i] = min(l[i], l[closestIndex] + uvWeight);
        }
    }

    for (int i = 0; i < dimension; i++) {
        results[tID * dimension + i] = l[i];
    }

    free(Vt);
    free(l);
}

__global__ void shortestPathsParallelV2(int* matrix, int dimension, int* results) {
    // Each Block computes the problem for the node with its blockID index
    // Max threads per block = 1024
    // Numero di thread per blocco = 1024/numero nodi 
    int tID = threadIdx.x;
    int bID = blockIdx.x;

    // Shared memory initialization
    extern __shared__ int s[];
    int* sharedData = s;

    // l vector initialization
    int* l = (int*)&sharedData[0];

    // min vector initialization
    int* min = (int*)&l[dimension];

    // in-place reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tID;
        if (index < blockDim.x)
            l[index] = atomicMin(l[index], l[index + stride]);
        // synchronize within threadblock
        __syncthreads();
    }

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)&min[dimension];
    
    Vt[tID] = false;
    __syncthreads();

    if (tID == 0) {
        Vt[bID] = true;
    }
    __syncthreads();


    // Getting direct connections with source node
    l[tID] = matrix[bID * dimension + tID];

    // Filling min shared vector
    min[tID] = matrix[bID * dimension + tID];

    __syncthreads();

    //FINO A QUI OK

    // while V != Vt
    while (!areAllTrue(Vt, dimension)) {

        int closestWeigth = 999999999;
        int closestIndex = tID;

        // Find the next vertex closest to source node
        if (Vt[tID] != true) {
            if (l[tID] < closestWeigth) {
                closestWeigth = l[tID];
                closestIndex = tID;
            }
        }
        // Add closest vertex to Vt
        Vt[closestIndex] = true;
        __syncthreads();

        // Recompute l
        if (Vt[tID] != true) {
            int uvWeight = matrix[closestIndex * dimension + tID];
            l[tID] = min(l[tID], l[closestIndex] + uvWeight);
        }
        __syncthreads();
    }

    results[bID * dimension + tID] = l[tID];
}