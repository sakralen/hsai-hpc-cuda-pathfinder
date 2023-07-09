#include "master.cuh"
#include "utility.cuh"
#include "fieldgenerator.cuh"
#include "pathfinder.cuh"

#define RUNS_COUNT 10

typedef struct
{
    int fieldSize;
    int gridDimVal;
    int blockDimVal;
} ExperimentData;

void execExperiment(ExperimentData data, int runsCount)
{
    printf("----------------------------------------------------------------------------\n");
    printf("Experiment parameters: field size is %d, grid size is %d, block size is %d\n", data.fieldSize, data.gridDimVal, data.blockDimVal);

    int successCount = 0;
    int failureCount = 0;
    float average = 0.;

    while (successCount < runsCount)
    {
        int fieldSize = data.fieldSize;
        int gridDimVal = data.gridDimVal;
        int blockDimVal = data.blockDimVal;

        dim3 gridDim;
        dim3 blockDim;

        setDims(&gridDim, &blockDim, gridDimVal, blockDimVal);

        int fieldBytes = fieldSize * fieldSize * sizeof(int);
        int *dField = NULL;
        int *dStates = NULL;

        if (!handleMemoryAlloc(&dField, &dStates, fieldBytes))
        {
            //return 1;
            continue;
        }

        if (!generateFieldCpu(dField, dStates, fieldSize))
        {
            handleMemoryFree(dField, dStates);
            //return 1;
            continue;
        }

        srand(time(NULL));
        int srcLinearIndex = 0;
        int dstLinearIndex = 0;
        generateSrcAndDest(&srcLinearIndex, &dstLinearIndex, fieldSize);

        float elapsedTime = 0.;

        int pathLength = execPathfinder(srcLinearIndex, dstLinearIndex, fieldSize, dField, dStates, gridDim, blockDim, &elapsedTime);
        if (pathLength > 0)
        {
            //printf("Path's length is %d\nElapsed time is %.2f ms\n", pathLength, elapsedTime);
            successCount++;
            average += pathLength;
        }
        else
        {
            //printf("Path does not exist!\n");
            failureCount++;
        }

        handleMemoryFree(dField, dStates);
    }

    printf("Successes: %d\n", successCount);
    printf("Failures: %d\n", failureCount);
    average /= runsCount;
    printf("Average time: %.2f\n", average);
    printf("----------------------------------------------------------------------------\n\n");
}

int main(int argc, char **argv)
{
    if (!isDeviceValid())
    {
        return 1;
    }

    ExperimentData experiments[] = 
    {
        {100, 5, 5}, {100, 5, 10}, {100, 5, 15}, {100, 5, 20}, {100, 5, 25}, {100, 5, 30},
        {100, 10, 5}, {100, 10, 10}, {100, 10, 15}, {100, 10, 20}, {100, 10, 25}, {100, 10, 30},
        {100, 50, 5}, {100, 50, 10}, {100, 50, 15}, {100, 50, 20}, {100, 50, 25}, {100, 50, 30},
        {100, 100, 5}, {100, 100, 10}, {100, 100, 15}, {100, 100, 20}, {100, 100, 25}, {100, 100, 30}
    };

    for (int i = 0; i < 20; i++)
    {
        execExperiment(experiments[i], RUNS_COUNT);
    }
    
    return 0;
}
