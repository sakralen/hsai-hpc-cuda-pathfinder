#include "master.cuh"
#include "utility.cuh"
#include "fieldgenerator.cuh"
#include "pathfinder.cuh"

#define RUNS_COUNT 10
#define UNIQUE_FIELDS_COUNT 3
#define UNIQUE_GRID_NS_COUNT 3
#define UNIQUE_BLOCK_NS_COUNT 3

const int kFieldNs[] = {100, 225, 1000};
const int kGridNs[] = {1, 2, 4, 16, 32, 64};
const int kBlockNs[] = {1, 2, 4, 16, 32, 64};

// const int kFieldNs[] = {5, 10, 20};
// const int kGridNs[] = {1, 4, 16};
// const int kBlockNs[] = {1, 4, 16};

void execExperiment(int fieldSize, int gridDimVal, int blockDimVal, int runsCount)
{
    printf("----------------------------------------------------------------------------\n");
    printf("Experiment parameters: field size is %d, grid size is %d, block size is %d\n", fieldSize, gridDimVal, blockDimVal);

    int successCount = 0;
    int failureCount = 0;
    float average = 0.;

    while (successCount < runsCount)
    {
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

        int srcLinearIndex = 0;
        int dstLinearIndex = 0;
        generateSrcAndDest(&srcLinearIndex, &dstLinearIndex, fieldSize);

        float elapsedTime = 0.;

        int pathLength = execPathfinder(srcLinearIndex, dstLinearIndex, fieldSize, dField, dStates, gridDim, blockDim, &elapsedTime);
        if (pathLength > 0)
        {
            //printf("Path's length is %d\nElapsed time is %.2f ms\n", pathLength, elapsedTime);
            successCount++;
            average += elapsedTime;
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
    printf("Average time: %.2f ms\n", average);
    printf("----------------------------------------------------------------------------\n\n");
}

int main(int argc, char **argv)
{
    if (!isDeviceValid())
    {
        return 1;
    }

    srand(time(NULL));

    for (int i = 0; i < UNIQUE_FIELDS_COUNT; i++) {
        for (int j = 0; j < UNIQUE_GRID_NS_COUNT; j++) {
            for (int k = 0; k < UNIQUE_BLOCK_NS_COUNT; k++) {
                execExperiment(kFieldNs[i], kGridNs[j], kBlockNs[k], RUNS_COUNT);
            }           
        }
    }
    
    return 0;
}
