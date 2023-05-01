#ifndef UTILITY_CUH
#define UTILITY_CUH

#include "master.cuh"

#define FIELD_SIZE_MIN 1
#define FIELD_SIZE_MAX 1024

#define GRID_DIM_MIN 1
#define GRID_DIM_MAX 1024

#define BLOCK_DIM_MIN 1
#define BLOCK_DIM_MAX 1024

#define DEVICE_NUMBER 0

int isDeviceValid();
int handleArgs(int argc, char **argv, int *fieldSize, int *gridDimVal, int *blockDimVal);
void setDims(dim3 *gridDimStruct, dim3 *blockDimStruct, int gridDimVal, int blockDimVal);
int handleMemoryAlloc(int **dField, int **dStates, int fieldBytes);
int handleMemoryFree(int *dField, int *dStates);
void printField(int *field, int size);
void setSingleElementOnDevice(int *array, int index, int value);

#endif
