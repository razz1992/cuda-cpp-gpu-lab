// Experiment 4: Memory coalescing and access patterns
// Compares contiguous reads against strided reads while writing contiguous output.
// Usage: ./memory_access_patterns [output_elements] [stride] [iterations]
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
  std::exit(EXIT_FAILURE); }} while(0)

__global__ void coalescedReadCopy(const float* input, float* output, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i];
}

__global__ void stridedReadCopy(const float* input, float* output, size_t n, size_t stride) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i * stride];
}

int main(int argc, char** argv) {
  size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 4000000ULL;
  size_t stride = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 16ULL;
  int iterations = (argc > 3) ? std::atoi(argv[3]) : 50;
  if (n == 0 || stride < 1 || iterations < 1) {
    std::cerr << "Usage: ./memory_access_patterns [output_elements>0] [stride>=1] [iterations>0]\n";
    return 1;
  }

  size_t inputCount = n * stride;
  std::vector<float> hInput(inputCount);
  for (size_t i=0; i<inputCount; ++i) hInput[i] = static_cast<float>((i % 1024) * 0.25f);

  float *dInput=nullptr, *dOutput=nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInput), inputCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutput), n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dInput, hInput.data(), inputCount*sizeof(float), cudaMemcpyHostToDevice));

  const int threads=256;
  const int blocks=(int)((n+threads-1)/threads);
  coalescedReadCopy<<<blocks,threads>>>(dInput,dOutput,n);
  stridedReadCopy<<<blocks,threads>>>(dInput,dOutput,n,stride);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto runCoalesced = [&]() {
    coalescedReadCopy<<<blocks,threads>>>(dInput,dOutput,n);
  };
  auto runStrided = [&]() {
    stridedReadCopy<<<blocks,threads>>>(dInput,dOutput,n,stride);
  };

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  auto benchmark = [&](auto kernelLaunch, const char* label) {
    for (int w=0; w<5; ++w) kernelLaunch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i=0; i<iterations; ++i) kernelLaunch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    std::vector<float> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(),dOutput,n*sizeof(float),cudaMemcpyDeviceToHost));
    size_t errors=0;
    const bool isStrided = (label[0] == 'S');
    for(size_t i=0;i<n;++i) {
      float expected = isStrided ? hInput[i * stride] : hInput[i];
      if(std::fabs(result[i]-expected)>1e-5f) ++errors;
    }
    std::cout << label << " avg kernel time: " << (ms/iterations) << " ms"
              << " | validation: " << (errors==0 ? "PASS" : "FAIL") << " (errors=" << errors << ")\n";
    return ms/iterations;
  };

  std::cout << "GPU: "; cudaDeviceProp prop{}; int device=0; CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop,device)); std::cout << prop.name << "\n";
  std::cout << "Output elements: " << n << ", stride: " << stride
            << ", input elements: " << inputCount << ", iterations: " << iterations << "\n";
  std::cout << "Threads/block: " << threads << ", blocks: " << blocks << "\n\n";

  float coalescedMs=benchmark(runCoalesced,"Coalesced (contiguous read)");
  float stridedMs=benchmark(runStrided,"Strided (every stride-th read)");
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nStrided / coalesced time ratio: " << (stridedMs/coalescedMs) << "x\n";
  std::cout << "Approx. useful read bandwidth (coalesced): " << ((n*sizeof(float)/1e6)/(coalescedMs/1000.0)) << " GB/s\n";
  std::cout << "Approx. useful read bandwidth (strided): " << ((n*sizeof(float)/1e6)/(stridedMs/1000.0)) << " GB/s\n";

  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dInput)); CUDA_CHECK(cudaFree(dOutput));
  return 0;
}
