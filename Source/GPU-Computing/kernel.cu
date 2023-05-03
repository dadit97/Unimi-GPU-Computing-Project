#include <stdio.h>
#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

bool allTrue(bool* vector, int dimension) {
    for (int i = 0; i < dimension; i++) {
        if (vector[i] == false) return false;
    }
    return true;
}

void shortestPathsSequential(int* matrix, int dimension, int* results) {
    for (int node = 0; node < dimension; node++) {
        // Boolean vector simulating the Vt set initialization
        bool* Vt = (bool*)malloc((dimension) * sizeof(bool));
        for (int i = 0; i < dimension; i++) {
            Vt[i] = false;
        }
        Vt[node] = true;

        // l vector initialization
        int* l = (int*)malloc(dimension * sizeof(int));

        // Getting direct connections with source node
        for (int i = 0; i < dimension; i++) {
            l[i] = matrix[node * dimension + i];
        }

        // while V != Vt
        while (!allTrue(Vt, dimension)) {

            int closestWeigth = 999999999;
            int closestIndex = node;

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

            for (int i = 0; i < dimension; i++) {
                if (Vt[i] == true) continue;
                int uvWeight = matrix[closestIndex * dimension + i];
                l[i] = min(l[i], l[closestIndex] + uvWeight);
            }
        }

        for (int i = 0; i < dimension; i++) {
            results[node * dimension + i] = l[i];
        }

        free(Vt);
        free(l);
    }
}

__global__ void shortestPathsParallel(int* matrix, int dimension, int* results) {
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

    //Kernel part
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
    

    int* resultsMatrix;
    int* results = (int*)malloc(nodes * nodes * sizeof(int));
    cudaError = cudaMalloc(&resultsMatrix, nodes * nodes * sizeof(int));

    if (cudaError != cudaSuccess) {
        printf("Error during results matrix allocation on GPU: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Results matrix allocation completed\n");
    
    shortestPathsParallel<<<1, nodes >>>(gpu_matrix, nodes, resultsMatrix);
    cudaDeviceSynchronize();

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

    cudaFree(resultsMatrix);

    //Sequential part

    // Results matrix re-initialization
    for (int i = 0; i < nodes * nodes; i++) {
        results[i] = 0;
    }

    shortestPathsSequential(matrix, nodes, results);

    printf("Sequential algorithm completed\n");
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d ", results[i * nodes + j]);
        }
        printf("\n");
    }
    
    free(results);
    free(matrix);
    return 0;
}