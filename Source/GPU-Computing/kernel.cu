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


/*__device__ int minIndexWithVt(int value, int min, int nodeIndexA, int nodeIndexB, bool VtA, bool VtB) {
    if (VtA && !VtB) return nodeIndexB;
    if (VtB && !VtA) return nodeIndexA;
    return value == min ? nodeIndexA : nodeIndexB;
}*/
__device__ int minIndexWithVt(int a, int b, int nodeIndexA, int nodeIndexB, bool VtA, bool VtB) {
    if (VtA && !VtB) return nodeIndexB;
    if (VtB && !VtA) return nodeIndexA;
    int minimum = min(a, b);
    return minimum == a ? nodeIndexA : nodeIndexB;
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
    
    while (true) {

        whileCounter++;

        if (tID == 0) {
            stopCycle[0] = areAllTrue(Vt, bDim);
        }
        __syncthreads();

        if (stopCycle[0]) {
            /*if (bID == 5 && tID == 0) {
                for (int i = 0; i < bDim; i++) {
                    printf("%d", Vt[i]);
                }
                printf(" %d-%d\n", minimum[0], minimum[bDim]);
                for (int i = 0; i < bDim; i++) {
                    printf("%d", l[i]);
                }
                printf(" %d\n\n", whileCounter);
            }*/
            break;
        }
        __syncthreads();

        // Restoring min shared vector from l vector
        minimum[tID] = l[tID];
        minimum[tID + bDim] = tID;

        __syncthreads();

        
        // in-place reduction
        for (int stride = 1; stride < bDim; stride *= 2) {
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
                    minimum[index + stride],
                    minimum[index + bDim],
                    minimum[index + stride + bDim],
                    Vt[minimum[index + bDim]],
                    Vt[minimum[index + stride + bDim]]
                );
                if (index + stride >= bDim) printf("%d", index + stride);

                /*if (bID == 5 && tID == 0 && whileCounter == 31) {

                    for (int i = 0; i < bDim; i++) {
                        printf("%d", Vt[i]);
                    }
                    printf(" %d-%d\n", minimum[0], minimum[bDim]);
                    for (int i = 0; i < bDim; i++) {
                        printf("%d", l[i]);
                    }
                    printf(" %d\n\n", whileCounter);

                    printf("%d %d - %d %d - %d %d\n", minimum[index],
                        minimum[index + bDim],
                        minimum[index + stride],
                        minimum[index + stride + bDim],
                        Vt[minimum[index + bDim]],
                        Vt[minimum[index + stride + bDim]]);
                    printf("%d %d %d %d\n\n", localMin, localMinIndex, index, stride);
                }
                __syncthreads();*/
                minimum[index + bDim] = localMinIndex;
                minimum[index] = localMin;
            }
            // synchronize within threadblock
            __syncthreads();
        }

        __syncthreads();

        // Add closest vertex to Vt
        if (tID == 0) {
            if (Vt[minimum[bDim]] == true) printf("error");
            Vt[minimum[bDim]] = true;
        }
        __syncthreads();

        /*if (bID == 5 && tID == 0) {
            for (int i = 0; i < bDim; i++) {
                printf("%d", Vt[i]);
            }
            printf(" %d-%d\n", minimum[bDim], minimum[0]);
            for (int i = 0; i < bDim; i++) {
                printf("%d", l[i]);
            }
            printf(" %d\n\n", whileCounter);
        }*/
        __syncthreads();

        // Recompute l
        if (!Vt[tID]) {

            int uvWeight = matrix[minimum[bDim] * bDim + tID];
            l[tID] = min(l[tID], l[minimum[bDim]] + uvWeight);
            __syncthreads();
        }

        __syncthreads();
    }
    
    results[bID * bDim + tID] = l[tID];
}
