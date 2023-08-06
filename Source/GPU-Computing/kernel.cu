#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#ifdef __CUDACC__
#define __syncthreads() __syncthreads()
#else
#define __syncthreads()
#endif

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

// FIXME
__device__ int minIndex(int* vector, int indexA, int indexB) {
    if (vector[indexA] == 0) return indexB + blockDim.x;
    if (vector[indexB == 0]) return indexA + blockDim.x;
    return vector[indexA] < vector[indexB] ? indexA + blockDim.x : indexB + blockDim.x;
}

__device__ int minWithoutZero(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;
    return min(a, b);
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
    // Numero di thread per blocco = min(1024/numero nodi, nodi di una riga)
    int tID = threadIdx.x;
    int bID = blockIdx.x;

    // Shared memory initialization
    extern __shared__ int s[];
    int* sharedData = s;

    // l vector initialization
    int* l = (int*)&sharedData[0];

    // minimum vector initialization, first half are values second half are indexes
    int* minimum = (int*)&l[blockDim.x];

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)&minimum[blockDim.x * 2];
    
    Vt[tID] = false;

    if (tID == 0) {
        Vt[bID] = true;
    }
    __syncthreads();
    
    // Getting direct connections with source node
    l[tID] = matrix[bID * dimension + tID];

    __syncthreads();

    //FASE 1 : ricerca lineare del nodo più vicino localmente
    // IN QUESTO CASO HO UN THREAD PER NODO, QUINDI IL LOCALE E' GIA' NOTO

    //FASE 2 : Riduzione per trovare il più vicino globale

    while (true) {

        if (areAllTrue(Vt, dimension)) break;
        __syncthreads();

        // Restoring min shared vector from l vector
        minimum[tID] = Vt[tID] ? 99999999 : l[tID];
        minimum[tID + blockDim.x] = tID;

        __syncthreads();

        if (bID == 0) {
            printf("%d - %d | %d ||", minimum[tID], minimum[blockDim.x + tID], Vt[tID]);
            //printf("%d | ", Vt[tID]);
            if (tID == blockDim.x - 1) printf("\n");
        }

        __syncthreads();

        // in-place reduction
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            // convert tid into local array index
            int index = 2 * stride * tID;
            if (index < blockDim.x) {
                //FIXME
                minimum[index + blockDim.x] = minIndex(minimum, index, index + stride);
                minimum[index] = minWithoutZero(minimum[index], minimum[index + stride]);
            }
            // synchronize within threadblock
            __syncthreads();
        }

        // Add closest vertex to Vt
        if (tID == 0) {
            Vt[minimum[blockDim.x]] = true;
        }
        __syncthreads();

        // Recompute l
        if (!Vt[tID]) {
            int uvWeight = matrix[minimum[blockDim.x] * dimension + tID];
            l[tID] = min(l[tID], l[minimum[blockDim.x]] + uvWeight);
        }

        //if (bID == 0 && tID == 0) printf("l[tID]: %d, vT[tID]: %d, closestIndex: %d\n", l[tID], Vt[tID], minimum[blockDim.x]);

        __syncthreads();
    }

    results[bID * dimension + tID] = l[tID];

    free(Vt);
    free(l);
    free(minimum);

    /* PARTE VECCHIA
    // while V != Vt
    while (!areAllTrue(Vt, dimension)) {

        int closestWeigth = 999999999;
        int closestIndex = tID;

        

        // Find the next vertex closest to source node
        if (Vt[tID] != true) {

            // in-place reduction
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                // convert tid into local array index
                int index = 2 * stride * tID;
                if (index < blockDim.x)
                    min[index] = atomicMin(min[index], min[index + stride]);
                // synchronize within threadblock
                __syncthreads();
            }

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
    } */
}