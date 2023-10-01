#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <list>
#include <vector>

using namespace std;

int getRandomWeight() {
    std::vector<int> list{ 1, 2, 3, 4, 5, 999999 };
    int index = rand() % list.size();
    return list[index];
}

void generateRandomGraph(int* matrix, int dimension) {
    srand(1);//(time(NULL));
    rand();
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

void getBasicGraphV2(int* matrix) {
    matrix[0] = 0;
    matrix[1] = 2;
    matrix[2] = 4;
    matrix[3] = 99;
    matrix[4] = 99;
    matrix[5] = 99;
    matrix[6] = 99;
    matrix[7] = 99;

    matrix[8] = 2;
    matrix[9] = 0;
    matrix[10] = 99;
    matrix[11] = 1;
    matrix[12] = 99;
    matrix[13] = 99;
    matrix[14] = 99;
    matrix[15] = 2;

    matrix[16] = 4;
    matrix[17] = 99;
    matrix[18] = 0;
    matrix[19] = 2;
    matrix[20] = 99;
    matrix[21] = 99;
    matrix[22] = 1;
    matrix[23] = 3;

    matrix[24] = 99;
    matrix[25] = 1;
    matrix[26] = 2;
    matrix[27] = 0;
    matrix[28] = 2;
    matrix[29] = 5;
    matrix[30] = 4;
    matrix[31] = 99;

    matrix[32] = 99;
    matrix[33] = 99;
    matrix[34] = 99;
    matrix[35] = 2;
    matrix[36] = 0;
    matrix[37] = 2;
    matrix[38] = 99;
    matrix[39] = 99;

    matrix[40] = 99;
    matrix[41] = 99;
    matrix[42] = 99;
    matrix[43] = 5;
    matrix[44] = 2;
    matrix[45] = 0;
    matrix[46] = 1;
    matrix[47] = 99;

    matrix[48] = 99;
    matrix[49] = 99;
    matrix[50] = 1;
    matrix[51] = 4;
    matrix[52] = 99;
    matrix[53] = 1;
    matrix[54] = 0;
    matrix[55] = 99;

    matrix[56] = 99;
    matrix[57] = 2;
    matrix[58] = 3;
    matrix[59] = 99;
    matrix[60] = 99;
    matrix[61] = 99;
    matrix[62] = 99;
    matrix[63] = 0;
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