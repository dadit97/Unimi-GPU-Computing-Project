#include <stdio.h>
#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define MAXWEIGHT 5

using namespace std;



int adj_matrix[10][10] = {
    {0, 1, 5, 4, 3, 2, 4, 2, 3, 10},
    {1, 0, 5, 10, 5, 3, 10, 2, 10, 5},
    {5, 5, 0, 1, 10, 3, 10, 10, 10, 4},
    {4, 10, 1, 0, 10, 10, 5, 10, 10, 3},
    {3, 5,10,10 ,0 ,10 ,10 ,10 ,5 ,2},
    {2 ,3 ,3 ,10 ,10 ,0 ,10 ,10 ,10 ,5},
    {4 ,10 ,10 ,5 ,10 ,10 ,0 ,10 ,2 ,1},
    {2 ,2 ,10 ,10 ,10 ,10 ,10 ,0 ,5 ,5},
    {3 ,10 ,10 ,10 ,5 ,10 ,2 ,5 ,0 ,10},
    {10 ,5 ,4 ,3 ,2 ,5 ,1 ,5 ,10 ,0}
};

__global__ void printThreadMatrixRow(int* matrix, int dimension) {
    int tID = threadIdx.x;
    for (int i = 0; i < dimension; i++) {
        printf("Thread %d, valore %d,%d: %d\n", tID, tID, i,  matrix[tID * dimension + i]);
    }
}

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

__global__ void shortestPaths(int* matrix, int dimension, int* results) {
    // Each Thread computes the problem for the node with its tID index
    int tID = threadIdx.x;
    int sourceNodeIndex = tID;

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)malloc((dimension) * sizeof(bool));
    initializeBoolVector(Vt, dimension);
    Vt[sourceNodeIndex] = true;

    // l vector initialization
    int* l = (int*)malloc(dimension * sizeof(int));
    
    // Getting direct connections with source node
    for (int i = 0; i < dimension; i++) {
        l[i] = matrix[sourceNodeIndex * dimension + i];
    }
    
    // while V != Vt
    while (!areAllTrue(Vt, dimension)) {

        int closestWeigth = 999999999;
        int closestIndex = sourceNodeIndex;

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

        /*if (tID == 4) {
            printf("Thread %d, closestIndex:%d,  closestWeigth:%d\n", tID, closestIndex, closestWeigth);
            printf("l: ");
            for (int i = 0; i < dimension; i++) {
                printf("%d ", l[i]);
            }
            printf("\n");
            printf("Vt: ");
            for (int i = 0; i < dimension; i++) {
                printf("%d ", Vt[i]);
            }
            printf("\n");
        }*/
        
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

int main(void) {
    cudaSetDevice(0);

    int nodes = 10;

    int* matrix = (int*)malloc(nodes * nodes * sizeof(int*));
    int index = 0;
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            matrix[index] = adj_matrix[i][j];
            index++;
        }
    }

    int* gpu_matrix;
    cudaError_t cudaError = cudaMalloc(&gpu_matrix, nodes * nodes * sizeof(int));

    if (cudaError != cudaSuccess) {
        printf("Errore during matrix allocation on GPU: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Matrix allocation on GPU completed\n");

    cudaError = cudaMemcpy(gpu_matrix, matrix, nodes * nodes * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        printf("Error during matrix copy on GPU: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Matrix copy on GPU completed\n");
    free(matrix);

    int* resultsMatrix;
    int* results = (int*)malloc(nodes * sizeof(int));
    cudaError = cudaMalloc(&resultsMatrix, nodes * nodes * sizeof(int));

    if (cudaError != cudaSuccess) {
        printf("Error during results matrix allocation on GPU: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Results matrix allocation completed\n");
    
    auto begin = std::chrono::high_resolution_clock::now();
    shortestPaths<<<1, nodes >>>(gpu_matrix, nodes, resultsMatrix);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    cudaError = cudaMemcpy(results, resultsMatrix, nodes * nodes * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        printf("Error during results copy on Host: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Results copy on Host completed\n");

    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d ", results[i * nodes + j]);
        }
        printf("\n");
    }

    printf("Kernel execution time: %f\n", elapsed.count());

    cudaFree(resultsMatrix);
    free(results);
    return 0;
}