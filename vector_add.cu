#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void addArrays(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(int argc, char** argv) {
    const size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1'000'000;
    if (n == 0) {
        std::cerr << "Element count must be greater than zero." << std::endl;
        return EXIT_FAILURE;
    }

    const size_t bytes = n * sizeof(float);
    std::vector<float> hA(n), hB(n), hC(n);
    for (size_t i = 0; i < n; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n"
              << "Compute capability: " << prop.major << "." << prop.minor << "\n"
              << "Elements: " << n << "\n";

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = static_cast<int>((n + threadsPerBlock - 1) / threadsPerBlock);

    addArrays<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    bool valid = true;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(hC[i] - (hA[i] + hB[i])) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << std::endl;
            valid = false;
            break;
        }
    }

    for (size_t i = 0; i < (n < 10 ? n : 10); ++i)
        std::cout << hA[i] << " + " << hB[i] << " = " << hC[i] << "\n";

    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
