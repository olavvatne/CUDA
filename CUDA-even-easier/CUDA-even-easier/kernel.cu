
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n; i += stride) {
			y[i] = x[i] + y[i];
		}

}

int main(void) {
	int N = 1 << 20;
	float *x, *y;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	printf("Runs with %d blocks \n", numBlocks);
	// Run kernel on 1M elements on the GPU
	add <<<numBlocks, blockSize>>>(N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	for (int i = 0; i < 10; i++) {
		printf("[%d] is %f \n", i, y[i]);
	}

	cudaFree(x);
	cudaFree(y);

	return 0;
}