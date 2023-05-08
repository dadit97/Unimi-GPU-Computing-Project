#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "data.h"
#include "kernel.h"
#include "sequential.h"

using namespace std;

int main(void) {

    //Kernel part
    cudaSetDevice(0);

    int nodes = 10;
    int* matrix = (int*)malloc(nodes * nodes * sizeof(int*));
    for (int i = 0; i < nodes; i++) {
        matrix[i] = 999999999;
    }

    printf("Generating random graph\n");
    generateRandomGraph(matrix, nodes);
    printf("Initial matrix loaded\n");

    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d ", matrix[i * nodes + j]);
        }
        printf("\n");
    }

    /*int index = 0;
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            matrix[index] = adj_matrix[i][j];
            index++;
        }
    }*/

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

    using clock = std::chrono::system_clock;
    using ms = std::chrono::duration<double, std::milli>;
    auto before = clock::now();

    shortestPathsParallel << <1, nodes >> > (gpu_matrix, nodes, resultsMatrix);
    cudaError = cudaGetLastError();

    if (cudaError != cudaSuccess) {
        printf("Error during kernel launch: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        printf("Kernel syncronization returned error: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    ms duration = clock::now() - before;

    cudaError = cudaMemcpy(results, resultsMatrix, nodes * nodes * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        printf("Error during results copy on Host: %s\n", cudaGetErrorString(cudaError));
    }
    printf("Results copy on Host completed\n");

    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d ", results[i * nodes + j]);
        }
        printf("\n");
    }

    cudaFree(resultsMatrix);
    printf("Kernel execution time: %f milliseconds\n", duration.count());

    //Sequential part

    // Results matrix re-initialization
    for (int i = 0; i < nodes * nodes; i++) {
        results[i] = 0;
    }

    before = clock::now();
    shortestPathsSequential(matrix, nodes, results);
    duration = clock::now() - before;

    /*printf("Sequential algorithm completed\n");
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d ", results[i * nodes + j]);
        }
        printf("\n");
    }*/

    printf("Sequential execution time: %f milliseconds\n", duration.count());
    free(results);
    free(matrix);
    return 0;
}