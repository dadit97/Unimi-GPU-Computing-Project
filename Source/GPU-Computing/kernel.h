#pragma once

using namespace std;

__device__ extern bool areAllTrue(bool* vector, int dimension);

__device__ extern void initializeBoolVector(bool* vector, int dimension);

__global__ extern void shortestPathsParallel(int* matrix, int dimension, int* results);

__global__ extern void shortestPathsParallelV2(int* matrix, int dimension, int* results);