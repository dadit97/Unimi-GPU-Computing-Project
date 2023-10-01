#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

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

/*void shortestPathsListSequential(int* adiacencyList, int* weightsList, int* offsetList, int dimension, int* results) {
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
        int directConnections = 0;
        if (node == dimension - 1) {
            directConnections = dimension - offsetList[node];
        }
        else {
            directConnections = offsetList[node + 1] - offsetList[node];
        }
        int* connections = &adiacencyList[offsetList[node]];
        int* connectionsWeights = &weightsList[offsetList[node]];

        for (int i = 0; i < dimension; i++) {
            bool isConnected = false;
            for (int j = 0; i < directConnections; i++) {
                if (i == connections[j]) {
                    l[i] = connectionsWeights[j];
                    isConnected = true;
                    break;;
                }
            }
            if (isConnected) continue;;
            l[i] = 999999999;
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
}*/
