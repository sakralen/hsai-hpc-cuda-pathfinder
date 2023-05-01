#ifndef RANDHANDLER_CUH
#define RANDHANDLER_CUH

#include "master.cuh"
#include "lee.cuh"

#include <curand.h>
#include <curand_kernel.h>

#define RAND_LOW -1
#define RAND_HIGH 10

int generateField(int* dField, int* dStates, int fieldSize, dim3* gridDimStruct, dim3* blockDimStruct);

#endif
