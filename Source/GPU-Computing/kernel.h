#pragma once

using namespace std;

__device__ extern bool areAllTrue(bool* vector, int dimension);

__device__ extern void initializeBoolVector(bool* vector, int dimension);

__device__ extern int minIndexWithVt(int a, int b, int nodeIndexA, int nodeIndexB, bool VtA, bool VtB);

__device__ extern int minWithVt(int a, int b, bool VtA, bool VtB);

__global__ extern void shortestPathsParallel(int* matrix, int dimension, int* results, int* lArray, bool* VtArray);

__global__ extern void shortestPathsParallelV2(int* matrix, int* results);