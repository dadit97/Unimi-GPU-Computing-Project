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


__device__ int minIndexWithVt(int value, int min, int nodeIndexA, int nodeIndexB, bool VtA, bool VtB) {
    if (VtA && !VtB) return nodeIndexB;
    if (VtB && !VtA) return nodeIndexA;
    return value == min ? nodeIndexA : nodeIndexB;
}

__device__ int minWithVt(int a, int b, bool VtA, bool VtB) {
    if (VtA && !VtB) return b;
    if (VtB && !VtA) return a;
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

__global__ void shortestPathsParallelV2(int* matrix, int* results) {
    // Each Block computes the problem for the node with its blockID index
    // Max threads per block = 1024
    // Numero di thread per blocco = min(1024/numero nodi, nodi di una riga)
    int tID = threadIdx.x;
    int bID = blockIdx.x;
    int bDim = blockDim.x;

    // Shared memory initialization
    extern __shared__ int s[];
    int* sharedData = s;

    // l vector initialization
    int* l = (int*)&sharedData[0];

    // minimum vector initialization, first half are values second half are indexes
    int* minimum = (int*)&l[bDim];

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)&minimum[bDim * 2];

    bool* stopCycle = (bool*)&Vt[bDim];
    
    Vt[tID] = false;

    if (tID == 0) {
        Vt[bID] = true;
    }
    __syncthreads();
    
    // Getting direct connections with source node
    l[tID] = matrix[bID * bDim + tID];

    __syncthreads();

    int whileCounter = 0;
    int forCounter = 0;
    
    while (true) {

        whileCounter++;
        __syncthreads();
        if (bID == 0) {
            printf("%d", Vt[tID]);
            if (tID == bDim - 1) printf("\n\n");
        }

        if (tID == 0) {
            stopCycle[0] = areAllTrue(Vt, bDim);
        }
        __syncthreads();

        if (stopCycle[0] || whileCounter > 140) {
            break;
        }
        __syncthreads();

        // Restoring min shared vector from l vector
        minimum[tID] = l[tID];
        minimum[tID + bDim] = tID;

        __syncthreads();

        
        // in-place reduction
        for (int stride = 1; stride < bDim; stride *= 2) {
            forCounter++;
            // convert tid into local array index
            int index = 2 * stride * tID;
            if (index < bDim) {
                
                int localMin = minWithVt(
                    minimum[index],
                    minimum[index + stride],
                    Vt[minimum[index + bDim]],
                    Vt[minimum[index + stride + bDim]]
                );

                int localMinIndex = minIndexWithVt(
                    minimum[index],
                    localMin,
                    minimum[index + bDim],
                    minimum[index + stride + bDim],
                    Vt[minimum[index + bDim]],
                    Vt[minimum[index + stride + bDim]]
                );

                /*if (bID == 3 && tID == 1 && whileCounter == 7 && forCounter == 2) {
                    printf("minimum[index]:%d, minimum[index + stride]:%d, %d %d, %d %d\n", minimum[index],
                        minimum[index + stride],
                        Vt[minimum[index + bDim]],
                        Vt[minimum[index + stride + bDim]],
                        index,
                        stride);
                }*/
                minimum[index + bDim] = localMinIndex;
                minimum[index] = localMin;
                //if (index + stride + bDim > 15)printf("%d, %d, %d, %d, %d, %d, %d, %d\n", localMinIndex, minimum[index], localMin, minimum[index + bDim], minimum[index + stride + bDim], index, stride, bDim);
                //printf("%d\n", minimum[12]);
                //printf("test\n");
            }
            // synchronize within threadblock
            __syncthreads();
        }
        forCounter = 0;

        /*if (bID == 0) {
            if (tID == 0) printf("\n\n");
            __syncthreads();
            printf("%d - %d | %d ||", minimum[tID], minimum[bDim + tID], Vt[tID]);
        }*/

        __syncthreads();

        // Add closest vertex to Vt
        if (tID == 0) {
            /*if (Vt[minimum[bDim]]) {
                for (int i = 0; i < bDim; i++) {
                    Vt[i] = true;
                }
            }
            else {
                Vt[minimum[bDim]] = true;
            }*/
            Vt[minimum[bDim]] = true;
        }
        //if (bID == 3) printf("%d ", Vt[tID]);
        //if (bID == 3 && tID == bDim - 1) printf("\n");
        __syncthreads();

        //if (bID == 3) printf("%d ", l[tID]);
        //if (bID == 3 && tID == bDim - 1) printf("\n\n");

        // Recompute l
        if (!Vt[tID]) {

            /*if (bID == 0) {
                if (tID == 0) printf("\n\n\n");
                printf("%d,%d ", Vt[tID], l[tID]);
            }
            __syncthreads();*/

            int uvWeight = matrix[minimum[bDim] * bDim + tID];
            l[tID] = min(l[tID], l[minimum[bDim]] + uvWeight);
            __syncthreads();
        }

        //if (bID == 3) printf("%d ", l[tID]);
        //if (bID == 3 && tID == bDim - 1) printf("   %d ", minimum[bDim]);
        //if (bID == 3 && tID == bDim - 1) printf("\n\n");
        
        //if (bID == 3) printf("%d-%d ", l[tID], Vt[tID]);
        //if (bID == 3 && tID == bDim - 1) printf("\n\n");
        __syncthreads();
        //if (bID == 0) printf("%d ", Vt[tID]);
        // 
        //if (bID == 0 && tID == bDim - 1) printf("\n");
        /*__syncthreads();
        if (bID == 0) printf("%d ", l[tID]);
        if (bID == 0 && tID == bDim - 1) printf("\n\n\n");*/
    }
    
    results[bID * bDim + tID] = l[tID];
}
