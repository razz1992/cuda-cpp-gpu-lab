#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t err=(call); if(err!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(err)<<" at "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(EXIT_FAILURE);} } while(0)

__global__ void addChunk(const float* a,const float* b,float* c,size_t n){
    size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) c[i]=a[i]+b[i];
}
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
bool validate(const std::vector<float>& expected,const float* actual,size_t n){
    for(size_t i=0;i<n;++i) if(std::fabs(expected[i]-actual[i])>1e-5f){std::cerr<<"Mismatch at "<<i<<" expected "<<expected[i]<<" got "<<actual[i]<<"\n";return false;}
    return true;
}
int main(int argc,char** argv){
    const size_t n=argc>1?std::strtoull(argv[1],nullptr,10):50000000;
    const size_t chunk=argc>2?std::strtoull(argv[2],nullptr,10):1000000;
    const int iterations=argc>3?std::atoi(argv[3]):10;
    const int streamCount=argc>4?std::atoi(argv[4]):2;
    if(!n||!chunk||iterations<1||streamCount<2){std::cerr<<"Usage: ./stream_pipeline [elements>0] [chunk_elements>0] [iterations>0] [streams>=2]\n";return EXIT_FAILURE;}
    const size_t bytes=n*sizeof(float);
    std::vector<float> hostA(n),hostB(n),expected(n);
    for(size_t i=0;i<n;++i){hostA[i]=static_cast<float>(i%1000)*0.5f;hostB[i]=static_cast<float>(i%500)*0.25f;expected[i]=hostA[i]+hostB[i];}
    float *pA=nullptr,*pB=nullptr,*pOut=nullptr;
    CUDA_CHECK(cudaMallocHost(&pA,bytes));CUDA_CHECK(cudaMallocHost(&pB,bytes));CUDA_CHECK(cudaMallocHost(&pOut,bytes));
    std::copy(hostA.begin(),hostA.end(),pA);std::copy(hostB.begin(),hostB.end(),pB);
    int device=0;cudaDeviceProp prop{};CUDA_CHECK(cudaGetDevice(&device));CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    std::cout<<"GPU: "<<prop.name<<" (compute "<<prop.major<<"."<<prop.minor<<")\nElements: "<<n<<" | chunk: "<<chunk<<" | iterations: "<<iterations<<" | streams: "<<streamCount<<"\n";

    // Single-stream baseline: pinned host buffers, whole-array copies, same iteration count.
    float *dA=nullptr,*dB=nullptr,*dC=nullptr;CUDA_CHECK(cudaMalloc(&dA,bytes));CUDA_CHECK(cudaMalloc(&dB,bytes));CUDA_CHECK(cudaMalloc(&dC,bytes));
    constexpr int threads=256;int blocks=static_cast<int>((n+threads-1)/threads);
    addChunk<<<1,1>>>(dA,dB,dC,0);CUDA_CHECK(cudaDeviceSynchronize()); // context warmup; no dereference
    cudaStream_t baseline;CUDA_CHECK(cudaStreamCreate(&baseline));
    auto baseStart=Clock::now();
    for(int k=0;k<iterations;++k){
        CUDA_CHECK(cudaMemcpyAsync(dA,pA,bytes,cudaMemcpyHostToDevice,baseline));
        CUDA_CHECK(cudaMemcpyAsync(dB,pB,bytes,cudaMemcpyHostToDevice,baseline));
        addChunk<<<blocks,threads,0,baseline>>>(dA,dB,dC,n);CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(pOut,dC,bytes,cudaMemcpyDeviceToHost,baseline));
        CUDA_CHECK(cudaStreamSynchronize(baseline));
    }
    double baselineMs=ms(baseStart,Clock::now())/iterations;
    bool baselineValid=validate(expected,pOut,n);

    // Multi-stream pipeline: independent chunk buffers; round-robin stream submission.
    std::vector<cudaStream_t> streams(streamCount);
    std::vector<float*> chunkA(streamCount,nullptr),chunkB(streamCount,nullptr),chunkC(streamCount,nullptr);
    for(int s=0;s<streamCount;++s){CUDA_CHECK(cudaStreamCreate(&streams[s]));CUDA_CHECK(cudaMalloc(&chunkA[s],chunk*sizeof(float)));CUDA_CHECK(cudaMalloc(&chunkB[s],chunk*sizeof(float)));CUDA_CHECK(cudaMalloc(&chunkC[s],chunk*sizeof(float)));}
    const size_t numChunks=(n+chunk-1)/chunk;
    auto pipeStart=Clock::now();
    for(int k=0;k<iterations;++k){
        for(size_t ch=0;ch<numChunks;++ch){
            int s=static_cast<int>(ch%streamCount);
            size_t offset=ch*chunk;size_t count=std::min(chunk,n-offset);size_t chunkBytes=count*sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(chunkA[s],pA+offset,chunkBytes,cudaMemcpyHostToDevice,streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(chunkB[s],pB+offset,chunkBytes,cudaMemcpyHostToDevice,streams[s]));
            int chunkBlocks=static_cast<int>((count+threads-1)/threads);
            addChunk<<<chunkBlocks,threads,0,streams[s]>>>(chunkA[s],chunkB[s],chunkC[s],count);CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(pOut+offset,chunkC[s],chunkBytes,cudaMemcpyDeviceToHost,streams[s]));
        }
        for(int s=0;s<streamCount;++s)CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }
    double pipelineMs=ms(pipeStart,Clock::now())/iterations;
    bool pipelineValid=validate(expected,pOut,n);

    std::cout<<std::fixed<<std::setprecision(4)<<"\nAverage end-to-end time per iteration\n"
      <<"Single-stream whole-array: "<<baselineMs<<" ms\nMulti-stream chunk pipeline: "<<pipelineMs<<" ms\n"
      <<"Pipeline speedup: "<<baselineMs/pipelineMs<<"x\nValidation baseline/pipeline: "<<(baselineValid?"PASS":"FAIL")<<" / "<<(pipelineValid?"PASS":"FAIL")<<"\n";
    for(int s=0;s<streamCount;++s){CUDA_CHECK(cudaFree(chunkA[s]));CUDA_CHECK(cudaFree(chunkB[s]));CUDA_CHECK(cudaFree(chunkC[s]));CUDA_CHECK(cudaStreamDestroy(streams[s]));}
    CUDA_CHECK(cudaStreamDestroy(baseline));CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));CUDA_CHECK(cudaFreeHost(pA));CUDA_CHECK(cudaFreeHost(pB));CUDA_CHECK(cudaFreeHost(pOut));
    return (baselineValid&&pipelineValid)?EXIT_SUCCESS:EXIT_FAILURE;
}
