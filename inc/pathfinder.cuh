#ifndef PATHFINDER_CUH
#define PATHFINDER_CUH

#include "master.cuh"
#include "utility.cuh"

#define BARRIER -1
#define NOT_VISITED 1
#define ON_FRONTIER 2
#define VISITED 3

#define DELTA_SRC_DST fieldSize

__global__ void propagateWave(int dstLinear, int fieldSize, int *fieldDevice, int *statesDevice, int *dCanPropagateFurther, int *dIsDstReached);
__forceinline__ __device__ int isVerticalAdjacentValid(int index, int offset, int *dStates, int fieldSize);
__forceinline__ __device__ int isHorizontalAdjacentValid(int index, int offset, int *dStates, int fieldSize);
int execPathfinder(int srcLinearIndex, int dstLinearIndex, int fieldSize, int *dField, int *dStates, dim3 gridDim, dim3 blockDim); // TODO: add src and dest points handling
void generateSrcAndDest(int *srcLinearIndex, int *dstLinearIndex, int fieldSize);

#endif
