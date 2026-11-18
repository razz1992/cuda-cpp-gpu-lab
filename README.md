# CUDA C++ GPU experiments

A small CUDA C++ project for learning GPU execution and benchmarking on Kaggle.

## Files
- `vector_add.cu`: basic CUDA vector addition, device information, CUDA error checks, and correctness validation.
- `benchmark.cu`: CPU baseline vs GPU kernel-only and GPU end-to-end timing, with result validation.
- `KAGGLE_NOTEBOOK_GUIDE.md`: copy/paste Kaggle notebook cells.
- `1.cpp`: original example retained unchanged.

## Quick start on Kaggle
Enable **GPU** in Notebook Settings, then run:
```bash
!nvidia-smi
!nvcc --version
!git clone https://github.com/razz1992/cuda-cpp-gpu-lab.git
%cd cuda-cpp-gpu-lab
!nvcc -O2 -std=c++17 vector_add.cu -o vector_add
!./vector_add
!nvcc -O2 -std=c++17 benchmark.cu -o benchmark
!./benchmark
```

Benchmark syntax: `./benchmark [element_count] [iterations]`.
For example: `!./benchmark 10000000 100`.

## Benchmark interpretation
Use **GPU end-to-end** when comparing with CPU for application-level performance. **GPU kernel-only** excludes host/device transfer time and is intended to isolate kernel execution. Small vectors may not benefit from GPU execution because launch and transfer overhead can dominate.

GPU model, driver, CUDA Toolkit version, and compiler availability depend on the Kaggle session allocation.
