#include "lee.cuh"

__forceinline__ __device__ int isVerticalAdjacentValid(int index, int offset, int *statesDevice, int fieldSize)
{
    return (((index + offset) >= 0) && ((index + offset) < fieldSize * fieldSize)
            // && (statesDevice[index + offset] != ON_FRONTIER)
            // && (statesDevice[index + offset] != VISITED));
            && (statesDevice[index + offset] == NOT_VISITED));
}

__forceinline__ __device__ int isHorizontalAdjacentValid(int index, int offset, int *statesDevice, int fieldSize)
{
    return ((index / fieldSize) == ((index + offset) / fieldSize)
            // && (statesDevice[index + offset] != ON_FRONTIER)
            // && (statesDevice[index + offset] != VISITED));
            && (statesDevice[index + offset] == NOT_VISITED));
}

__global__ void propagateWave(int fieldSize, int *fieldDevice, int *statesDevice, int *dCanPropagateFurther)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int linearIndex = x + y * gridDim.x * blockDim.x;

    if (linearIndex < 0 || linearIndex >= fieldSize * fieldSize)
    {
        return;
    }

    if (statesDevice[linearIndex] != ON_FRONTIER)
    {
        return;
    }

    //__shared__ int isBlockNotTrapped;
    int isThreadNotTrapped = FALSE;

    statesDevice[linearIndex] = VISITED;

    __syncthreads();

    if (isHorizontalAdjacentValid(linearIndex, -1, statesDevice, fieldSize))
    {
        statesDevice[linearIndex - 1] = ON_FRONTIER;
        fieldDevice[linearIndex - 1] = fieldDevice[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }
    if (isHorizontalAdjacentValid(linearIndex, 1, statesDevice, fieldSize))
    {
        statesDevice[linearIndex + 1] = ON_FRONTIER;
        fieldDevice[linearIndex + 1] = fieldDevice[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }
    if (isVerticalAdjacentValid(linearIndex, -fieldSize, statesDevice, fieldSize))
    {
        statesDevice[linearIndex - fieldSize] = ON_FRONTIER;
        fieldDevice[linearIndex - fieldSize] = fieldDevice[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }
    if (isVerticalAdjacentValid(linearIndex, fieldSize, statesDevice, fieldSize))
    {
        statesDevice[linearIndex + fieldSize] = ON_FRONTIER;
        fieldDevice[linearIndex + fieldSize] = fieldDevice[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }

    //__syncthreads();
    // atomicOr(&isBlockNotTrapped, isThreadNotTrapped);
    __syncthreads();
    atomicOr(dCanPropagateFurther, isThreadNotTrapped);
}
