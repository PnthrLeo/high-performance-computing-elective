#include "stdio.h"
#include "assert.h"
#include "math.h"

#define N 100000
#define BLOCK_SIZE 128
#define GRID_SIZE 128
#define MAX_ERR 1e-6

__global__ void add(int *a, int *b, int *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < n)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    a = (int *)malloc(sizeof(int) * N);
    b = (int *)malloc(sizeof(int) * N);
    c = (int *)malloc(sizeof(int) * N);

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, sizeof(int) * N);
    cudaMalloc((void **)&d_b, sizeof(int) * N);
    cudaMalloc((void **)&d_c, sizeof(int) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Executing kernel
    add<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    // Transfer data back to host memory
    cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; i++)
    {
        assert(abs(c[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

    // Deallocate host memory
    free(a);
    free(b);
    free(c);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
