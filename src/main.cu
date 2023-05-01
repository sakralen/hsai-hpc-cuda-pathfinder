#include "master.cuh"
#include "utility.cuh"
#include "fieldgenerator.cuh"
#include "lee.cuh"

#define TMP_SRC_LINEAR_INDEX 7

// "h" at the start of a variable means that it is allocated on host;
// "d" -- on device.

int main(int argc, char **argv)
{
    if (!isDeviceValid())
    {
        return 1;
    }

    int fieldSize = 0;
    int gridDimVal = 0;
    int blockDimVal = 0;

    if (!handleArgs(argc, argv, &fieldSize, &gridDimVal, &blockDimVal))
    {
        return 1;
    }

    dim3 gridDim;
    dim3 blockDim;

    setDims(&gridDim, &blockDim, gridDimVal, blockDimVal);

    int fieldBytes = fieldSize * fieldSize * sizeof(int);
    int *dField = NULL;
    int *dStates = NULL;

    if (!handleMemoryAlloc(&dField, &dStates, fieldBytes))
    {
        return 1;
    }

    if (!generateField(dField, dStates, fieldSize, &gridDim, &blockDim))
    {
        handleMemoryFree(dField, dStates);
        return 1;
    }

    // Debug stuff
    setSingleElementOnDevice(dStates, TMP_SRC_LINEAR_INDEX, ON_FRONTIER);
    propagateWave<<<gridDim, blockDim>>>(fieldSize, dField, dStates);

    int *hField = (int *)malloc(fieldBytes);
    cudaMemcpy(hField, dField, fieldBytes, cudaMemcpyDeviceToHost);
    printField(hField, fieldSize);
    printf("\n");
    cudaMemcpy(hField, dStates, fieldBytes, cudaMemcpyDeviceToHost);
    printField(hField, fieldSize);
    free(hField);
    // !Debug stuff

    handleMemoryFree(dField, dStates);

    return 0;
}
