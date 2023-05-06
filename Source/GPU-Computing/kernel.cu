#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <random>
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

void getDataFromFile(int* matrix, int dimension, char* path) {

    FILE* file = NULL;
    file = fopen(path, "r");
    char buffer[80];
    char* token;
    int number = 0;
    int n0 = 0;
    int n1 = 0;
    int n2 = 0;
    int counter = 0;

    while (fgets(buffer, 80, file)) {
        token = strtok(buffer, "\t");
        while (token) {
            number = atoi(token);
            if (counter == 0) n0 = number;
            if (counter == 1) n1 = number;
            if (counter == 2) n2 = number;
            token = strtok(NULL, "\t");
            counter++;
        }
        printf("matrix[%d] = %d\n", n0 * dimension + n1, n2);
        matrix[n0 * dimension + n1] = n2;
        counter = 0;
    }
}

int getRandomWeight() {
    std::vector<int> list{ 1, 2, 3, 4, 5, 999999999 };
    int index = rand() % list.size();
    return list[index];
}

void generateRandomGraph(int* matrix, int dimension) {
    int weight = 0;
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            if (i == j) weight = 0;
            else weight = getRandomWeight();
            matrix[i * dimension + j] = weight;
            matrix[j * dimension + i] = weight;
        }
    }
}

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
    
    shortestPathsParallel<<<1, nodes >>>(gpu_matrix, nodes, resultsMatrix);
    cudaDeviceSynchronize();

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