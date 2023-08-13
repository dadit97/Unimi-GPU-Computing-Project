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

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("Max Threads per Block:%d\n", props.maxThreadsPerBlock);
    printf("Max Blocks per Multiprocessor:%d\n", props.maxBlocksPerMultiProcessor);
    printf("Max Shared Memory size per Block:%d\n", props.sharedMemPerBlock);

    int nodes = 512;
    int* matrix = (int*)malloc(nodes * nodes * sizeof(int*));
    for (int i = 0; i < nodes; i++) {
        matrix[i] = 9;
    }

    printf("Shared Memory size per Block:%d bytes\n", sizeof(int) * nodes + sizeof(int) * nodes * 2 + sizeof(bool) * nodes + sizeof(bool));
    generateRandomGraph(matrix, nodes);
    printf("Initial matrix loaded\n");

    /*for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d", matrix[i * nodes + j]);
        }
        printf("\n");
    }*/

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

    //KERNEL V1 PART

    printf("\n\nKERNEL V1 PART\n\n");

    shortestPathsParallel <<<1, nodes >>> (gpu_matrix, nodes, resultsMatrix);
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

    /*for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d", results[i * nodes + j]);
        }
        printf("\n");
    }*/

    printf("Kernel execution time: %f milliseconds\n", duration.count());

    // KERNEL V2 PART

    printf("\n\nKERNEL V2 PART\n\n");

    before = clock::now();
    shortestPathsParallelV2 <<<nodes, nodes, sizeof(int) * nodes + sizeof(int) * nodes * 2 + sizeof(bool) * nodes + sizeof(bool) >> > (gpu_matrix, resultsMatrix);
    cudaError = cudaGetLastError();

    if (cudaError != cudaSuccess) {
        printf("Error during kernel V2 launch: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    cudaError = cudaPeekAtLastError();
    if (cudaError != cudaSuccess) {
        printf("Error during kernel V2 execution: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        printf("Kernel V2 syncronization returned error: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    duration = clock::now() - before;

    cudaError = cudaMemcpy(results, resultsMatrix, nodes * nodes * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        printf("Error during results copy on Host: %s\n", cudaGetErrorString(cudaError));
    }
    printf("Results copy on Host completed\n");

    /*for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d", results[i * nodes + j]);
        }
        printf("\n");
    }*/

    cudaFree(resultsMatrix);
    cudaFree(gpu_matrix);
    printf("Kernel V2 execution time: %f milliseconds\n", duration.count());

    //SEQUENTIAL PART

    // Results matrix re-initialization
    for (int i = 0; i < nodes * nodes; i++) {
        results[i] = 0;
    }

    printf("\n\nSEQUENTIAL PART\n\n");

    before = clock::now();
    shortestPathsSequential(matrix, nodes, results);
    duration = clock::now() - before;

    /*printf("Sequential algorithm completed\n");
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            printf("%d", results[i * nodes + j]);
        }
        printf("\n");
    }*/

    printf("Sequential execution time: %f milliseconds\n", duration.count());
    free(results);
    free(matrix);
    return 0;
}