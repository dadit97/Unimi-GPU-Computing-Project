#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define MAXWEIGHT 5

using namespace std;



int adj_matrix[10][10] = {
    {0, 1, 5, 4, 3, 2, 4, 2, 3, -1},
    {1, 0, 5, -1, 5, 3, -1, 2, -1, 5},
    {5, 5, 0, 1, -1, 3, -1, -1, -1, 4},
    {4, -1, 1, 0, -1, -1, 5, -1, -1, 3},
    {3, 5,-1,-1 ,0 ,-1 ,-1 ,-1 ,5 ,2},
    {2 ,3 ,3 ,-1 ,-1 ,0 ,-1 ,-1 ,-1 ,5},
    {4 ,-1 ,-1 ,5 ,-1 ,-1 ,0 ,-1 ,2 ,1},
    {2 ,2 ,-1 ,-1 ,-1 ,-1 ,-1 ,0 ,5 ,5},
    {3 ,-1 ,-1 ,-1 ,5 ,-1 ,2 ,5 ,0 ,-1},
    {1 ,3 ,-1 ,1 ,4 ,-1 ,2 ,4 ,1 ,0}
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

__global__ void shortestPath(int* matrix, int dimension) {
    // Each Thread computes the problem for the node with its tID index
    int tID = threadIdx.x;
    int sourceNodeIndex = tID;

    // Boolean vector simulating the Vt set initialization
    bool* Vt = (bool*)malloc((dimension) * sizeof(bool));
    initializeBoolVector(Vt, dimension);
    Vt[sourceNodeIndex] = true;

    // l vector initialization
    int* l = (int*)malloc(dimension * sizeof(int));
    
    // Getting direct connections with source node maintaining the column fixed
    // and jumping from row to row
    for (int i = 0; i < dimension; i++) {
        if (Vt[i] == true) continue;
        else {
            l[i] = matrix[sourceNodeIndex + dimension * i];
        }
    }

    // while V != Vt
    while (!areAllTrue(Vt, dimension)) {
        // Find the next vertex closest to source node
        int closestWeigth = MAXWEIGHT;
        int closestIndex = sourceNodeIndex;
        for (int i = 0; i < dimension; i++) {
            if (Vt[i] == true) continue;
            if (l[i] != -1 && l[i] < closestWeigth) {
                closestWeigth = l[i];
                closestIndex = i;
            }
        }
        //printf("Thread %d, closest index: %d, closest weight: %d \n", tID, closestIndex, closestWeigth);

        // Add closest vertex to Vt
        Vt[closestIndex] = true;

        // Recompute l
        for (int i = 0; i < dimension; i++) {
            if (Vt[i] == true) continue;
            int uvWeight = matrix[closestIndex * dimension + i];
            if (l[i] > (l[closestIndex] + uvWeight)) {
                l[i] = l[closestIndex] + uvWeight;
            }
        }
    }
    for (int i = 0; i < dimension; i++) {
        if (tID == true) continue;
        printf("Thread %d, l index: %d, weight: %d \n", tID, i, l[i]);
    }
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
        printf("Errore durante l'allocazione della memoria sulla GPU: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Allocazione della memoria sulla GPU completata\n");

    cudaError = cudaMemcpy(gpu_matrix, matrix, nodes * nodes * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        printf("Errore durante la copia della matrice sulla GPU: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
    printf("Copia della matrice sulla GPU completata\n");
    free(matrix);
        
    shortestPath<<<1, nodes >>>(gpu_matrix, nodes);

    cudaDeviceSynchronize();
    return 0;
}