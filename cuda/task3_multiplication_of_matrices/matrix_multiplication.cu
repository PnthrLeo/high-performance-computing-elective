#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"
#include <iostream>
#include <chrono>

#define BLOCK_SIZE 1024
#define N 1005
#define M 1023 // M <= 1024
#define K 1231
#define MAX_ERR 1e-6

__global__ void mult_matrix_on_gpu(int *mat_a, int *mat_b, int *mat_c, int n, int m, int k)
{
    __shared__ int cache[BLOCK_SIZE];
    int cache_index = threadIdx.x;
    unsigned int idi = threadIdx.x;
    unsigned int idj = blockIdx.x;
    unsigned int idl = blockIdx.y;

    int temp = 0;
    if (idj < n && idl < k && idi < m)
    {
        temp = mat_a[idj * m + idi] * mat_b[idi * k + idl];
    }

    // set the cache values
    cache[cache_index] = temp;

    // synchronize threads in this block
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cache_index < i)
        {
            cache[cache_index] += cache[cache_index + i];
        }
        __syncthreads();
        i /= 2;
    }

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    if (cache_index == 0)
    {
        mat_c[idj * k + idl] = cache[0];
    }
}

int **init_matrix(unsigned int height, unsigned int width)
{
    int **mat = (int **)malloc(sizeof(int *) * height);
    mat[0] = (int *)malloc(sizeof(int) * height * width);
    for (int i = 1; i < height; i++)
    {
        mat[i] = mat[i - 1] + width;
    }
    return mat;
}

int **fill_matrix(int **mat, unsigned int height, unsigned int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            mat[i][j] = rand();
        }
    }
    return mat;
}

void free_matrix(int **mat, unsigned int height, unsigned int width)
{
    free(mat[0]);
    free(mat);
}

std::chrono::high_resolution_clock::time_point get_time_in_milliseconds()
{
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    return tp;
}

void verify(int **a, int **b, int **c, int n, int m, int k)
{
    int sum;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            sum = 0;
            for (int l = 0; l < m; l++)
            {
                sum += a[i][l] * b[l][j];
            }
            assert(abs(c[i][j] - sum) < MAX_ERR);
        }
    }
}

int main()
{
    int **a, **b, **c;
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    a = init_matrix(N, M);
    b = init_matrix(M, K);
    c = init_matrix(N, K);

    // Initialize host arrays
    a = fill_matrix(a, N, M);
    b = fill_matrix(b, M, K);

    // Allocate device memory
    cudaMalloc((void **)&d_a, sizeof(int) * N * M);
    cudaMalloc((void **)&d_b, sizeof(int) * M * K);
    cudaMalloc((void **)&d_c, sizeof(int) * N * K);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a[0], sizeof(int) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b[0], sizeof(int) * M * K, cudaMemcpyHostToDevice);

    // Set kernel configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid(N, K);

    // Executing kernel
    std::chrono::high_resolution_clock::time_point start_time = get_time_in_milliseconds();
    mult_matrix_on_gpu<<<grid, block>>>(d_a, d_b, d_c, N, M, K);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end_time = get_time_in_milliseconds();

    // Transfer data back to host memory
    cudaMemcpy(c[0], d_c, sizeof(int) * N * K, cudaMemcpyDeviceToHost);

    // Verification
    verify(a, b, c, N, M, K);

    printf("PASSED\n");
    std::chrono::duration<double, std::milli> time_span = end_time - start_time;
    std::cout << "Elapsed time: " << time_span.count() << " ms" << std::endl;

    // Deallocate host memory
    free_matrix(a, N, M);
    free_matrix(b, M, K);
    free_matrix(c, N, K);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
