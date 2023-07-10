#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(int* dField, int fieldSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int linearIndex = x + y * gridDim.x * blockDim.x;

    while (!(linearIndex < 0 || linearIndex >= fieldSize * fieldSize)) {
        dField[linearIndex] = 1;
        linearIndex += gridDim.x;
    }
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

int main(int argc, char **argv) {

    int fieldSize = atoi(argv[1]);
    int gridDimVal = atoi(argv[2]);
    int blockDimVal = atoi(argv[3]);

    int fieldBytes = fieldSize * fieldSize * sizeof(int);

    int* hField = (int*)malloc(fieldBytes);

    int* dField;
    cudaMalloc(&dField, fieldBytes);
    cudaMemset(dField, 0, fieldBytes);

    cudaMemcpy(hField, dField, fieldBytes, cudaMemcpyDeviceToHost);
    // printField(hField, fieldSize);
    // printf("\n");
    
    dim3 gridDim(gridDimVal, gridDimVal);
    dim3 blockDim(blockDimVal, blockDimVal);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    kernel<<<gridDim, blockDim>>>(dField, fieldSize);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(hField, dField, fieldBytes, cudaMemcpyDeviceToHost);
    // printField(hField, fieldSize);
    printf("\n");
    printf("Elapsed time is %.5f\n", elapsedTime);

    cudaFree(dField);
    free(hField);

    return 0;
}
