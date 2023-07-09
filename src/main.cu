#include "master.cuh"
#include "utility.cuh"
#include "fieldgenerator.cuh"
#include "pathfinder.cuh"

// "h" at the start of a variable means that it is allocated on host;
// "d" -- on device.

// TODO: rework field gen to make distrubtion closer to 1/12 barrier/field

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

    if (!generateFieldCpu(dField, dStates, fieldSize))
    {
        handleMemoryFree(dField, dStates);
        return 1;
    }

    srand(time(NULL));
    int srcLinearIndex = 0;
    int dstLinearIndex = 0;
    generateSrcAndDest(&srcLinearIndex, &dstLinearIndex, fieldSize);

    float elapsedTime = 0.;

    int pathLength = execPathfinder(srcLinearIndex, dstLinearIndex, fieldSize, dField, dStates, gridDim, blockDim, &elapsedTime);
    if (pathLength)
    {
        printf("Path's length is %d\nElapsed time is %.2f ms\n", pathLength, elapsedTime);
    }
    else
    {
        printf("Path does not exist!\n");
    }

    handleMemoryFree(dField, dStates);

    return 0;
}
