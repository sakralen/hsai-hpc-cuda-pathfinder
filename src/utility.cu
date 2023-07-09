#include "utility.cuh"

int isDeviceValid()
{
    if (cudaSetDevice(DEVICE_NUMBER) != cudaSuccess)
    {
        printf("Error: cudaSetDevice failed\n");
        return FALSE;
    }

    cudaDeviceProp property;
    cudaGetDeviceProperties(&property, DEVICE_NUMBER);

    if (property.totalGlobalMem < (FIELD_SIZE_MIN * FIELD_SIZE_MAX))
    {
        printf("Error: not enough global memory\n");
        return FALSE;
    }

    return TRUE;
}

int handleArgs(int argc, char **argv, int *fieldSize, int *gridDimVal, int *blockDimVal)
{
    if (argc != 4)
    {
        printf("%s usage: fieldSize gridDimVal blockDimVal", argv[0]);
        return FALSE;
    }

    printf("Passed args are:\n");
    for (int i = 0; i < argc; i++)
    {
        printf("%s ", argv[i]);
    }
    printf("\n");
    

    *fieldSize = atoi(argv[1]);
    *gridDimVal = atoi(argv[2]);
    *blockDimVal = atoi(argv[3]);

    if (*fieldSize < FIELD_SIZE_MIN || *fieldSize > FIELD_SIZE_MAX)
    {
        printf("Error: fieldSize should be in the range of %d to %d\n", FIELD_SIZE_MIN, FIELD_SIZE_MAX);
        return FALSE;
    }

    if (*gridDimVal < GRID_DIM_MIN || *gridDimVal > GRID_DIM_MAX)
    {
        printf("Error: gridDimVal should be in the range of %d to %d\n", GRID_DIM_MIN, GRID_DIM_MAX);
        return FALSE;
    }

    if (*blockDimVal < BLOCK_DIM_MIN || *blockDimVal > BLOCK_DIM_MAX)
    {
        printf("Error: blockDimVal should be in the range of %d to %d\n", BLOCK_DIM_MIN, BLOCK_DIM_MAX);
        return FALSE; 
    }

    if (((*blockDimVal) * (*blockDimVal) * (*gridDimVal) * (*gridDimVal)) < ((*fieldSize) * (*fieldSize))) {
        printf("Error: not enough threads for this field size (blockDimVal^2 * gridDimVal^2 < fieldSize^2\n)");
        return FALSE; 
    }

    return TRUE;
}

void setDims(dim3 *gridDimStruct, dim3 *blockDimStruct, int gridDimVal, int blockDimVal)
{
    gridDimStruct->x = gridDimVal;
    gridDimStruct->y = gridDimVal;

    blockDimStruct->x = blockDimVal;
    blockDimStruct->y = blockDimVal;
}

int handleMemoryAlloc(int **fieldDevice, int **statesDevice, int fieldBytes)
{
    if (cudaMalloc(fieldDevice, fieldBytes) != cudaSuccess)
    {
        printf("Error: failed to allocate field[] on the device\n");
        return FALSE;
    }

    if (cudaMemset(*fieldDevice, 0, fieldBytes) != cudaSuccess)
    {
        printf("Error: failed to memset field[] on the device\n");
        return FALSE;
    }

    if (cudaMalloc(statesDevice, fieldBytes) != cudaSuccess)
    {
        printf("Error: failed to allocate states[] on the device\n");
        return FALSE;
    }

    if (cudaMemset(*statesDevice, 0, fieldBytes) != cudaSuccess)
    {
        printf("Error: failed to memset states[] on the device\n");
        return FALSE;
    }

    return TRUE;
}

int handleMemoryFree(int *fieldDevice, int *statesDevice)
{
    int result = TRUE;

    if (cudaFree(fieldDevice) != cudaSuccess)
    {
        printf("Error: failed to free fieldDevice[]\n");
        result = FALSE;
    }

    if (cudaFree(statesDevice) != cudaSuccess)
    {
        printf("Error: failed to free statesDevice[]\n");
        result = FALSE;
    }

    return result;
}

void printField(int *field, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%2d ", field[i * size + j]);
        }
        printf("\n");
    }
}

__global__ void setSingleElement(int *array, int index, int value)
{
    array[index] = value;
}

void setSingleElementOnDevice(int *array, int index, int value)
{
    setSingleElement<<<1, 1>>>(array, index, value);
}
