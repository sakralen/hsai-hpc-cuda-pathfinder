#include "pathfinder.cuh"

__forceinline__ __device__ int isVerticalAdjacentValid(int index, int offset, int *dStates, int fieldSize)
{
    return (((index + offset) >= 0) && ((index + offset) < fieldSize * fieldSize)
            // && (dStates[index + offset] != ON_FRONTIER)
            // && (dStates[index + offset] != VISITED));
            && (dStates[index + offset] == NOT_VISITED));
}

__forceinline__ __device__ int isHorizontalAdjacentValid(int index, int offset, int *dStates, int fieldSize)
{
    return ((index / fieldSize) == ((index + offset) / fieldSize)
            // && (dStates[index + offset] != ON_FRONTIER)
            // && (dStates[index + offset] != VISITED));
            && (dStates[index + offset] == NOT_VISITED));
}

__global__ void propagateWave(int dstLinearIndex, int fieldSize, int *dField, int *dStates, int *dCanPropagateFurther, int *dIsDstReached)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int linearIndex = x + y * gridDim.x * blockDim.x;

    if (linearIndex < 0 || linearIndex >= fieldSize * fieldSize)
    {
        return;
    }

    if (dStates[linearIndex] != ON_FRONTIER)
    {
        return;
    }

    //__shared__ int isBlockNotTrapped;
    int isThreadNotTrapped = FALSE;

    dStates[linearIndex] = VISITED;

    __syncthreads();

    if (isHorizontalAdjacentValid(linearIndex, -1, dStates, fieldSize))
    {
        dStates[linearIndex - 1] = ON_FRONTIER;
        dField[linearIndex - 1] = dField[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }
    if (isHorizontalAdjacentValid(linearIndex, 1, dStates, fieldSize))
    {
        dStates[linearIndex + 1] = ON_FRONTIER;
        dField[linearIndex + 1] = dField[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }
    if (isVerticalAdjacentValid(linearIndex, -fieldSize, dStates, fieldSize))
    {
        dStates[linearIndex - fieldSize] = ON_FRONTIER;
        dField[linearIndex - fieldSize] = dField[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }
    if (isVerticalAdjacentValid(linearIndex, fieldSize, dStates, fieldSize))
    {
        dStates[linearIndex + fieldSize] = ON_FRONTIER;
        dField[linearIndex + fieldSize] = dField[linearIndex] + 1;
        isThreadNotTrapped = TRUE;
    }

    int didThreadReachDst = (linearIndex == dstLinearIndex);

    __syncthreads();
    atomicOr(dCanPropagateFurther, isThreadNotTrapped);

    __syncthreads();
    atomicOr(dIsDstReached, didThreadReachDst);
}

void execPathfinder(int srcLinearIndex, int dstLinearIndex, int fieldSize, int *dField, int *dStates, dim3 gridDim, dim3 blockDim)
{
    // Setting src:
    setSingleElementOnDevice(dStates, srcLinearIndex, ON_FRONTIER);
    setSingleElementOnDevice(dField, srcLinearIndex, 0);

    // Setting dst:
    setSingleElementOnDevice(dStates, dstLinearIndex, NOT_VISITED);

    // Setting flags:
    int *dCanPropagateFurther = NULL;
    cudaMalloc(&dCanPropagateFurther, sizeof(int));
    // cudaMemset(dCanPropagateFurther, FALSE, sizeof(int));

    int hCanPropagateFurther = FALSE;

    int *dIsDstReached = NULL;
    cudaMalloc(&dIsDstReached, sizeof(int));
    // cudaMemset(dIsDstReached, FALSE, sizeof(int));

    int hIsDstReached = FALSE;

    // Pathfinder:
    do
    {
        cudaMemset(dCanPropagateFurther, FALSE, sizeof(int));
        cudaMemset(dIsDstReached, FALSE, sizeof(int));

        propagateWave<<<gridDim, blockDim>>>(dstLinearIndex, fieldSize, dField, dStates, dCanPropagateFurther, dIsDstReached);

        cudaMemcpy(&hCanPropagateFurther, dCanPropagateFurther, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&hIsDstReached, dIsDstReached, sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG_MODE
        int fieldBytes = fieldSize * fieldSize * sizeof(int);

        int *hField = (int *)malloc(fieldBytes);
        cudaMemcpy(hField, dField, fieldBytes, cudaMemcpyDeviceToHost);
        printField(hField, fieldSize);
        printf("\n");
        cudaMemcpy(hField, dStates, fieldBytes, cudaMemcpyDeviceToHost);
        printField(hField, fieldSize);
        free(hField);

        printf("hCanPropagateFurther val: %d\n", hCanPropagateFurther);
        printf("hIsDstReached val: %d\n", hIsDstReached);
        printf("\n");
#endif
    } while ((hIsDstReached == FALSE) && (hCanPropagateFurther == TRUE)); // TODO: Probably should check this condition

    cudaFree(dCanPropagateFurther);
    cudaFree(dIsDstReached);
}
