#include "master.cuh"
#include "utility.cuh"
#include "fieldgenerator.cuh"
#include "pathfinder.cuh"

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

    // Temporary setting src and dst points
    srand(time(NULL));
    int srcLinearIndex = 0;
    int dstLinearIndex = 0;
    generateSrcAndDest(&srcLinearIndex, &dstLinearIndex, fieldSize);

    execPathfinder(srcLinearIndex, dstLinearIndex, fieldSize, dField, dStates, gridDim, blockDim);

    handleMemoryFree(dField, dStates);

    return 0;
}
