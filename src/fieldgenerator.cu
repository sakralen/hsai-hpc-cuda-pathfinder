#include "fieldgenerator.cuh"

__global__ void init(unsigned int seed, curandState_t* curandStates, int fieldSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    if (offset < 0 || offset >= fieldSize * fieldSize) {
        return;
    }

    curand_init(seed, offset, 0, &curandStates[offset]);
}

__global__ void generate(curandState_t* curandStates, int* dField, int* dStates, int fieldSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    if (offset < 0 || offset >= fieldSize * fieldSize) {
        return;
    }

    int generated = (curand(&curandStates[offset]) % (RAND_HIGH - RAND_LOW + 1)) + RAND_LOW;
    dField[offset] = (generated == -1) ? BARRIER : 0; 
    dStates[offset] = (generated == -1) ? BARRIER : NOT_VISITED;
}

// curandState_t* curandStates has nothing to do with dStates.
// This pointer is necessary part of cuRAND execution.
int generateField(int* dField, int* dStates, int fieldSize, dim3* gridDimStruct, dim3* blockDimStruct) {
    // int fieldBytes = fieldSize * fieldSize * sizeof(int);
    curandState_t* curandStates; 

    if (cudaMalloc(&curandStates, fieldSize * fieldSize * sizeof(curandState_t)) != cudaSuccess) {
        printf("Error: failed to allocate curandStates[] on the device\n");
        return FALSE;
    }

    init<<<*gridDimStruct, *blockDimStruct>>>(time(NULL), curandStates, fieldSize);
    generate<<<*gridDimStruct, *blockDimStruct>>>(curandStates, dField, dStates, fieldSize);

    // if (cudaMemcpy(dStates, dField, fieldBytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
    //     printf("Error: failed to copy dField[] to dStates[]\n");

    //     cudaFree(curandStates);
    //     return FALSE;
    // }

    cudaFree(curandStates);
    return TRUE;
}
