#ifndef RANDHANDLER_CUH
#define RANDHANDLER_CUH

#include "master.cuh"
#include "pathfinder.cuh"

#include <curand.h>
#include <curand_kernel.h>

#define RAND_LOW -1
#define RAND_HIGH 10

int generateFieldGpu(int *dField, int *dStates, int fieldSize, dim3 *gridDimStruct, dim3 *blockDimStruct);
int generateFieldCpu(int* dField, int* dStates, int fieldSize);

#endif
