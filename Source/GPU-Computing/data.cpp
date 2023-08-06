#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <list>
#include <vector>

using namespace std;

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

void getBasicGraph(int* matrix) {
    matrix[0] = 0;
    matrix[1] = 4;
    matrix[2] = 1;
    matrix[3] = 3;
    matrix[4] = 4;
    matrix[5] = 0;
    matrix[6] = 2;
    matrix[7] = 999999999;
    matrix[8] = 1;
    matrix[9] = 2;
    matrix[10] = 0;
    matrix[11] = 1;
    matrix[12] = 3;
    matrix[13] = 999999999;
    matrix[14] = 1;
    matrix[15] = 0;
}

//Not used, replaced by random generation
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