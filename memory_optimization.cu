#include <cuda_runtime.h>
#include <algorithm>
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

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool validate(const std::vector<float>& expected, const float* actual, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(expected[i] - actual[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": expected "
                      << expected[i] << ", got " << actual[i] << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 10000000;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 50;
    if (!n || iterations < 1 || n > static_cast<size_t>(-1) / sizeof(float)) {
        std::cerr << "Usage: ./memory_optimization [elements>0] [iterations>0]\n";
        return EXIT_FAILURE;
    }

    const size_t bytes = n * sizeof(float);
    std::vector<float> hostA(n), hostB(n), expected(n), resultA(n), resultB(n);
    for (size_t i = 0; i < n; ++i) {
        hostA[i] = static_cast<float>(i % 1000) * 0.5f;
        hostB[i] = static_cast<float>(i % 500) * 0.25f;
        expected[i] = hostA[i] + hostB[i];
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n"
              << "Elements: " << n << ", iterations: " << iterations
              << ", buffer size per array: " << (bytes / (1024.0 * 1024.0)) << " MiB\n";

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    // Warm up the CUDA context and kernel before collecting timings.
    float *warmA = nullptr, *warmB = nullptr, *warmC = nullptr;
    CUDA_CHECK(cudaMalloc(&warmA, bytes));
    CUDA_CHECK(cudaMalloc(&warmB, bytes));
    CUDA_CHECK(cudaMalloc(&warmC, bytes));
    CUDA_CHECK(cudaMemcpy(warmA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(warmB, hostB.data(), bytes, cudaMemcpyHostToDevice));
    addArrays<<<blocks, threads>>>(warmA, warmB, warmC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(warmA));
    CUDA_CHECK(cudaFree(warmB));
    CUDA_CHECK(cudaFree(warmC));

    // Approach A: allocate and free device buffers on every iteration.
    auto aStart = Clock::now();
    for (int k = 0; k < iterations; ++k) {
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hostA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hostB.data(), bytes, cudaMemcpyHostToDevice));
        addArrays<<<blocks, threads>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(resultA.data(), dC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double approachAMs = elapsedMs(aStart, Clock::now()) / iterations;

    // Approach B: allocate device buffers once; reuse them for every iteration.
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    auto bStart = Clock::now();
    for (int k = 0; k < iterations; ++k) {
        CUDA_CHECK(cudaMemcpy(dA, hostA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hostB.data(), bytes, cudaMemcpyHostToDevice));
        addArrays<<<blocks, threads>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(resultB.data(), dC, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double approachBMs = elapsedMs(bStart, Clock::now()) / iterations;

    // Approach C: pinned host buffers + async copies and one CUDA stream.
    float *pinnedA = nullptr, *pinnedB = nullptr, *pinnedC = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinnedA, bytes));
    CUDA_CHECK(cudaMallocHost(&pinnedB, bytes));
    CUDA_CHECK(cudaMallocHost(&pinnedC, bytes));
    std::copy(hostA.begin(), hostA.end(), pinnedA);
    std::copy(hostB.begin(), hostB.end(), pinnedB);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto cStart = Clock::now();
    for (int k = 0; k < iterations; ++k) {
        CUDA_CHECK(cudaMemcpyAsync(dA, pinnedA, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, pinnedB, bytes, cudaMemcpyHostToDevice, stream));
        addArrays<<<blocks, threads, 0, stream>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(pinnedC, dC, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double approachCMs = elapsedMs(cStart, Clock::now()) / iterations;

    const bool validA = validate(expected, resultA.data(), n);
    const bool validB = validate(expected, resultB.data(), n);
    const bool validC = validate(expected, pinnedC, n);

    std::cout << std::fixed << std::setprecision(4)
              << "\nAverage end-to-end time per iteration\n"
              << "A. Allocate/free every iteration: " << approachAMs << " ms\n"
              << "B. Reuse device allocations:       " << approachBMs << " ms\n"
              << "C. Pinned + async stream:          " << approachCMs << " ms\n"
              << "\nRelative improvement vs A\n"
              << "B: " << (approachAMs / approachBMs) << "x\n"
              << "C: " << (approachAMs / approachCMs) << "x\n"
              << "\nValidation A/B/C: "
              << (validA ? "PASS" : "FAIL") << " / "
              << (validB ? "PASS" : "FAIL") << " / "
              << (validC ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(pinnedA));
    CUDA_CHECK(cudaFreeHost(pinnedB));
    CUDA_CHECK(cudaFreeHost(pinnedC));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return (validA && validB && validC) ? EXIT_SUCCESS : EXIT_FAILURE;
}
