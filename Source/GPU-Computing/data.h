#pragma once

using namespace std;

extern int getRandomWeight();

extern void generateRandomGraph(int* matrix, int dimension);

extern void getBasicGraph(int* matrix);

//Not used, replaced by random generation
extern void getDataFromFile(int* matrix, int dimension, char* path);