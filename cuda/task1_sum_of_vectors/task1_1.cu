#include "stdio.h"
#include "assert.h"
#include "math.h"

#define N 100
#define MAX_ERR 1e-6

__global__ void add(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N];
	int *d_a, *d_b, *d_c;

    // Initialize host arrays
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
	
	// Allocate device memory
	cudaMalloc((void**)&d_a, sizeof(int) * N);
    cudaMalloc((void**)&d_b, sizeof(int) * N);
    cudaMalloc((void**)&d_c, sizeof(int) * N);
	
	// Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
    

    /*
        TEST 1
    */
    // Executing kernel 
    add<<<1,N>>>(d_a, d_b, d_c, N);
	
	// Transfer data back to host memory
    cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++) {
        assert(abs(c[i] - a[i] - b[i]) < MAX_ERR);
    }
	
	printf("<1, N> case PASSED\n");
	
	// Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);


    /*
        TEST 2
    */
    // Executing kernel 
    add<<<N,1>>>(d_a, d_b, d_c, N);
	
	// Transfer data back to host memory
    cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(abs(c[i] - a[i] - b[i]) < MAX_ERR);
    }
	
	printf("<N, 1> case PASSED\n");
	
	// Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}