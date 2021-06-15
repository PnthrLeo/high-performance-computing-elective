#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"
#include <iostream>
#include <chrono>

#define STRING_LENGTH 2000000000
#define HIST_SIZE 256 // should be less then 257

__global__ void calculate_hist(unsigned char *str, int str_length, int *hist)
{
    __shared__ int loc_hist[HIST_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_local = threadIdx.x;

    // init local histogram
    if (tid_local < HIST_SIZE) {
        loc_hist[tid_local] = 0;
    }

    __syncthreads();

    // count symbol in local histogram
    if (tid < str_length) {
        atomicAdd(&loc_hist[str[tid]], 1);
    }

    __syncthreads();

    // send counts to global histrogram
    if (tid_local < HIST_SIZE) {
        atomicAdd(&hist[tid_local], loc_hist[tid_local]);
    }
}

unsigned char *generate_random_string(unsigned int length)
{
    unsigned char *str = (unsigned char *)malloc(sizeof(unsigned char) * length);

    for (int i = 0; i < length; i++)
    {
        str[i] = rand() % HIST_SIZE;
    }

    return str;
}

int *init_hist()
{
    int *hist = (int *)malloc(sizeof(int) * HIST_SIZE);
    
    for (int i = 0; i < HIST_SIZE; i++) {
        hist[i] = 0;
    }

    return hist;
}

void generate_kernel_config(int length_of_str, int *grid_size, int *block_size) 
{   
    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, 0);
    int max_threads_per_block = device.maxThreadsPerBlock;

    if (length_of_str / max_threads_per_block > 0) {
        *grid_size = length_of_str / max_threads_per_block + 1;
        *block_size = max_threads_per_block;
    }
    else if (length_of_str < 256) {
        *grid_size = 1;
        *block_size = 256;
    }
    else {
        *grid_size = 1;
        *block_size = length_of_str;
    }
}

std::chrono::high_resolution_clock::time_point get_time_in_milliseconds()
{
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    return tp;
}

void verify(unsigned char *str, unsigned int length, int *hist)
{
    int *true_hist = init_hist();

    for (int i = 0; i < length; i++)
    {
        true_hist[str[i]] += 1;
    }

    for (int i = 0; i < HIST_SIZE; i++)
    {
        assert(hist[i] == true_hist[i]);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main()
{   
    int grid_size, block_size;
    unsigned char *str, *d_str;
    int *hist, *d_hist;

    str = generate_random_string(STRING_LENGTH);
    hist = init_hist();

    // Allocate device memory
    cudaMalloc((void **)&d_str, sizeof(unsigned char) * STRING_LENGTH);
    cudaMalloc((void **)&d_hist, sizeof(int) * HIST_SIZE);

    // Transfer data from host to device memory
    cudaMemcpy(d_str, str, sizeof(unsigned char) * STRING_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, hist, sizeof(int) * HIST_SIZE, cudaMemcpyHostToDevice);

    // Set kernel configuration
    generate_kernel_config(STRING_LENGTH, &grid_size, &block_size);
    dim3 grid(grid_size);
    dim3 block(block_size);
    // Executing kernel
    std::chrono::high_resolution_clock::time_point start_time = get_time_in_milliseconds();
    calculate_hist<<<grid, block>>>(d_str, STRING_LENGTH, d_hist);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point end_time = get_time_in_milliseconds();

    // Transfer data back to host memory
    cudaMemcpy(hist, d_hist, sizeof(int) * HIST_SIZE, cudaMemcpyDeviceToHost);

    // Verification
    verify(str, STRING_LENGTH, hist);

    printf("PASSED\n");
    std::chrono::duration<double, std::milli> time_span = end_time - start_time;
    std::cout << "Elapsed time: " << time_span.count() << " ms" << std::endl;

    // Deallocate host memory
    free(str);
    free(hist);

    // Deallocate device memory
    cudaFree(d_str);
    cudaFree(d_hist);

    return 0;
}
