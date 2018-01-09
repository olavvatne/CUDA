
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define SIZE 1024

// Global functions - kernels. Device code. Run on GPU. Code that run on CPU is host code
__global__
void VectorAdd(int *a, int *b, int *c, int n) {
	int i = threadIdx.x; // A readonly variable

	// if more threads than elements
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	int *a, *b, *c;

	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
		
	}

	VectorAdd <<<1, SIZE>>> (a, b, c, SIZE);

	cudaDeviceSynchronize();
	for (int i = 0; i < 10; ++i) {
		printf("c[%d] = %d\n", i, c[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}
