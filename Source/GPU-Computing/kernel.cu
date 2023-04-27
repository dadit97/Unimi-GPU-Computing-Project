#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    {1 ,3 ,-1 ,1 ,4 ,-1 ,2 ,4 ,1 ,-1}
};

__global__ void helloFromGPU(void) {
    int tID = threadIdx.x;
    printf("Hello World from GPU (I'am thread = %d)!\n", tID);
}

__global__ void printThreadMatrixRow(int* matrix, int dimension) {
    int tID = threadIdx.x;
    for (int i = 0; i < 10; i++) {
        printf("Thread %d, valore %d,%d: %d\n", tID, tID, i,  matrix[tID * dimension + i]);
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
    cudaError_t mallocError = cudaMalloc(&gpu_matrix, nodes * nodes * sizeof(int));

    if (mallocError != cudaSuccess) {
        printf("Errore durante l'allocazione della memoria sulla GPU: %s\n", cudaGetErrorString(mallocError));
        exit(1);
    }
    printf("Allocazione della memoria sulla GPU completata\n");

    mallocError = cudaMemcpy(gpu_matrix, matrix, nodes * nodes * sizeof(int), cudaMemcpyHostToDevice);
    if (mallocError != cudaSuccess) {
        printf("Errore durante la copia della matrice sulla GPU: %s\n", cudaGetErrorString(mallocError));
        exit(1);
    }
    printf("Copia della matrice sulla GPU completata\n");
    free(matrix);
        
    //helloFromGPU << <1, 10 >> > ();
    printThreadMatrixRow<<<1, 10 >>> (gpu_matrix, nodes);
    cudaDeviceSynchronize();
    return 0;
}