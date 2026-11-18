
#include <iostream>
#include <cuda_runtime.h>

#define N 1000000

// GPU function (Kernel)
__global__ void addArrays(
    float *A,
    float *B,
    float *C
) {
    // Identify which element this GPU thread processes
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate CPU memory
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate GPU memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy CPU data to GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, N * sizeof(float),
               cudaMemcpyHostToDevice);

    // Configure GPU execution
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    // Execute GPU kernel
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C
    );

    // Wait for GPU execution
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(h_C, d_C, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Display sample results
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i] << " + "
                  << h_B[i] << " = "
                  << h_C[i] << std::endl;
    }

    // Release GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Release CPU memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
