#ifndef LEE_H
#define LEE_H

#include "master.cuh"
#include "utility.cuh"

#define BARRIER -1
#define NOT_VISITED 1
#define ON_FRONTIER 2
#define VISITED 3

__global__ void propagateWave(int fieldSize, int *dField, int *dStates, int* canPropagateFurther);
__forceinline__ __device__ int isVerticalAdjacentValid(int index, int offset, int *dStates, int fieldSize);
__forceinline__ __device__ int isHorizontalAdjacentValid(int index, int offset, int *dStates, int fieldSize);

#endif
