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
    // Each Thread computes the problem for the node with its calculated tID index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dimension) {

        // Boolean vector simulating the Vt set initialization
        bool* Vt = (bool*)malloc((dimension) * sizeof(bool));
        initializeBoolVector(Vt, dimension);
        Vt[idx] = true;
        //printf("Ciao: %d\n", idx); // OK
        // l vector initialization
        int* l = (int*)malloc(dimension * sizeof(int));
        
        // Getting direct connections with source node
        for (int i = 0; i < dimension; i++) {
            if ((idx * dimension + i) >= (dimension * dimension)) printf("Value exceeded\n");
            l[i] = matrix[idx * dimension + i];
        }
        printf("tId: %d\n", idx);
        // while V != Vt
        while (!areAllTrue(Vt, dimension)) {

            int closestWeigth = 99999;
            int closestIndex = idx;

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
            results[idx * dimension + i] = l[i];
        }

        free(Vt);
        free(l);
    }
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
    int* minimumValues = (int*)&l[bDim];
    int* minimumIndexes = (int*)&minimumValues[bDim];

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)&minimumIndexes[bDim];

    bool* stopCycle = (bool*)&Vt[bDim];
    
    Vt[tID] = false;

    if (tID == 0) {
        Vt[bID] = true;
    }
    __syncthreads();
    
    // Getting direct connections with source node
    l[tID] = matrix[bID * bDim + tID];

    __syncthreads();
    
    while (true) {

        if (tID == 0) {
            stopCycle[0] = areAllTrue(Vt, bDim);
        }
        __syncthreads();

        if (stopCycle[0]) {
            break;
        }
        __syncthreads();

        // Restoring min shared vector from l vector
        minimumValues[tID] = l[tID];
        minimumIndexes[tID] = tID;
        
        // in-place reduction
        for (int stride = 1; stride < bDim; stride *= 2) {
            // convert tid into local array index
            int index = 2 * stride * tID;
            if (index < bDim) {
                
                int localMin = minWithVt(
                    minimumValues[index],
                    minimumValues[index + stride],
                    Vt[minimumIndexes[index]],
                    Vt[minimumIndexes[index + stride]]
                );

                int localMinIndex = minIndexWithVt(
                    minimumValues[index],
                    minimumValues[index + stride],
                    minimumIndexes[index],
                    minimumIndexes[index + stride],
                    Vt[minimumIndexes[index]],
                    Vt[minimumIndexes[index + stride]]
                );
                //if (index + stride >= bDim) printf("%d", index + stride);

                /*if (bID == 0 && tID == 0 && whileCounter == 100) {

                    for (int i = 0; i < bDim; i++) {
                        printf("%d", Vt[i]);
                    }
                    printf(" %d-%d\n", minimumValues[0], minimumIndexes[0]);
                    for (int i = 0; i < bDim; i++) {
                        printf("%d", l[i]);
                    }
                    printf(" %d\n\n", whileCounter);

                    printf("%d %d - %d %d - %d %d\n",
                        minimumValues[index],
                        minimumIndexes[index],
                        minimumValues[index + stride],
                        minimumIndexes[index + stride],
                        Vt[minimumIndexes[index]],
                        Vt[minimumIndexes[index + stride]]);
                    printf("%d %d %d %d\n\n", localMin, localMinIndex, index, stride);
                }*/
                minimumIndexes[index] = localMinIndex;
                minimumValues[index] = localMin;
            }
            // synchronize within threadblock
            __syncthreads();
        }

        /*if (bID == 0 && tID == 0) {

            for (int i = 0; i < bDim; i++) {
                printf("%d", Vt[i]);
            }
            printf(" %d-%d\n", minimumValues[0], minimumIndexes[0]);
        }

        __syncthreads();*/

        // Add closest vertex to Vt
        if (tID == 0) {
            Vt[minimumIndexes[0]] = true;
        }
        __syncthreads();

        // Recompute l
        if (!Vt[tID]) {

            int uvWeight = matrix[minimumIndexes[0] * bDim + tID];
            l[tID] = min(l[tID], l[minimumIndexes[0]] + uvWeight);
        }

        __syncthreads();
    }
    
    results[bID * bDim + tID] = l[tID];
}
