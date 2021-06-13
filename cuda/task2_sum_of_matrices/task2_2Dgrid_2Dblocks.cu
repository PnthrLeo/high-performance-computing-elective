#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"

#define MAX_ERR 1e-6

__global__ void sum_matrix_on_gpu(int *mat_a, int *mat_b, int *mat_c, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        mat_c[idx] = mat_a[idx] + mat_b[idx];
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

int main()
{
    // Set size of matrices
    int nx = 1 << 14;
    int ny = 1 << 14;

    int **a, **b, **c;
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    a = init_matrix(ny, nx);
    b = init_matrix(ny, nx);
    c = init_matrix(ny, nx);

    // Initialize host arrays
    a = fill_matrix(a, ny, nx);
    b = fill_matrix(b, ny, nx);

    // Allocate device memory
    cudaMalloc((void **)&d_a, sizeof(int) * nx * ny);
    cudaMalloc((void **)&d_b, sizeof(int) * nx * ny);
    cudaMalloc((void **)&d_c, sizeof(int) * nx * ny);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a[0], sizeof(int) * nx * ny, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b[0], sizeof(int) * nx * ny, cudaMemcpyHostToDevice);

    // Set kernel configuration
    dim3 block(32, 32);
    dim3 grid((nx + block.x) / block.x, (ny + block.y) / block.y);

    // Executing kernel
    sum_matrix_on_gpu<<<grid, block>>>(d_a, d_b, d_c, nx, ny);

    // Transfer data back to host memory
    cudaMemcpy(c[0], d_c, sizeof(int) * nx * ny, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            assert(abs(c[i][j] - a[i][j] - b[i][j]) < MAX_ERR);
        }
    }

    printf("PASSED\n");

    // Deallocate host memory
    free_matrix(a, ny, nx);
    free_matrix(b, ny, nx);
    free_matrix(c, ny, nx);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
