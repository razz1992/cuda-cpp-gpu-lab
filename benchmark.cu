#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void addArrays(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 10000000;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    if (!n || iterations < 1 || n > static_cast<size_t>(-1) / sizeof(float)) {
        std::cerr << "Usage: ./benchmark [elements>0] [iterations>0]\n";
        return EXIT_FAILURE;
    }

    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), cpu(n), gpu(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 1000) * 0.5f;
        b[i] = static_cast<float>(i % 500) * 0.25f;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n"
              << "Elements: " << n << ", iterations: " << iterations << "\n";

    // CPU baseline: average of repeated full vector additions.
    auto cpuStart = std::chrono::steady_clock::now();
    for (int k = 0; k < iterations; ++k)
        for (size_t i = 0; i < n; ++i) cpu[i] = a[i] + b[i];
    auto cpuEnd = std::chrono::steady_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count() / iterations;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    // Warm-up avoids including first-launch initialization in the timed kernel.
    addArrays<<<blocks, threads>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int k = 0; k < iterations; ++k)
        addArrays<<<blocks, threads>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernelTotalMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTotalMs, start, stop));
    const double kernelMs = kernelTotalMs / iterations;

    // End-to-end GPU path: host-to-device copies + kernel + device-to-host copy.
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e2eStart = std::chrono::steady_clock::now();
    for (int k = 0; k < iterations; ++k) {
        CUDA_CHECK(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice));
        addArrays<<<blocks, threads>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(gpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e2eEnd = std::chrono::steady_clock::now();
    const double e2eMs = std::chrono::duration<double, std::milli>(e2eEnd - e2eStart).count() / iterations;

    bool valid = true;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > 1e-5f) { valid = false; break; }
    }

    std::cout << std::fixed << std::setprecision(4)
              << "CPU average:                 " << cpuMs << " ms\n"
              << "GPU kernel average:          " << kernelMs << " ms\n"
              << "GPU end-to-end average:      " << e2eMs << " ms\n"
              << "CPU / GPU kernel speedup:    " << cpuMs / kernelMs << "x (kernel-only; excludes transfers)\n"
              << "CPU / GPU end-to-end ratio:  " << cpuMs / e2eMs << "x\n"
              << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
