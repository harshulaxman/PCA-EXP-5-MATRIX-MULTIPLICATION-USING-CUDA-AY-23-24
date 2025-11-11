# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME: Harsshitha lakshmanan</h3>
<h3>ENTER YOUR REGISTER NO: 212223230075</h3>
<h3>DATE: 11/11/25</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
```
!nvidia-smi
!nvcc --version

!pip -q install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

%%writefile matmul.cu
#include <cstdio>
#include <cuda_runtime.h>

#ifndef CHECK
#define CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

#define SIZE 4
#define BLOCK_SIZE 2

// Matrix multiplication C = A x B, square SIZE x SIZE
__global__ void matrixMultiply(const int* __restrict__ a,
                               const int* __restrict__ b,
                               int* __restrict__ c,
                               int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        #pragma unroll
        for (int k = 0; k < SIZE; ++k) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main()
{
    const int N = SIZE * SIZE;
    const size_t BYTES = N * sizeof(int);

    // Host buffers
    int hA[N], hB[N], hC[N];

    // Init A and B
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            hA[i * SIZE + j] = i + j;
            hB[i * SIZE + j] = i - j;
        }
    }

    // Device buffers
    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK(cudaMalloc((void**)&dA, BYTES));
    CHECK(cudaMalloc((void**)&dB, BYTES));
    CHECK(cudaMalloc((void**)&dC, BYTES));

    CHECK(cudaMemcpy(dA, hA, BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, BYTES, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((SIZE + block.x - 1) / block.x,
              (SIZE + block.y - 1) / block.y);

    // CUDA event timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matrixMultiply<<<grid, block>>>(dA, dB, dC, SIZE);
    CHECK(cudaEventRecord(stop));

    // Sync + check kernel error
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaMemcpy(hC, dC, BYTES, cudaMemcpyDeviceToHost));

    // Print result
    printf("Result Matrix (SIZE=%d):\n", SIZE);
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", hC[i * SIZE + j]);
        }
        printf("\n");
    }
    printf("Kernel time: %.6f ms\n", ms);

    // Cleanup
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));

    return 0;
}

!nvcc -arch=sm_75 matmul.cu -o matmul
!nvprof ./matmul
!nvprof --print-gpu-trace ./matmul
```

## OUTPUT:
<img width="1122" height="560" alt="image" src="https://github.com/user-attachments/assets/16827ae9-3d18-4d08-b831-665216a47458" />


## RESULT:
Thus the program has been executed by using CUDA to mulptiply two matrices. It is observed that there are variations in host and device elapsed time. Device took ______________time and host took ___________time.
