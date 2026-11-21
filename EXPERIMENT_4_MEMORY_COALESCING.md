# Experiment 4 — Memory Coalescing & Access Patterns

## Goal
Compare contiguous (coalesced) global-memory reads with a deliberately strided read pattern. Both kernels write the same-sized contiguous output array and perform one float read + one float write per output element. The strided case reads one float every `stride` elements from a larger input allocation.

## Why this matters
CUDA global-memory transactions are most efficient when threads in a warp access nearby addresses. For contiguous reads, adjacent threads access adjacent 4-byte floats. In the strided kernel, thread lane 0 reads input[0], lane 1 reads input[stride], lane 2 reads input[2*stride], and so on. This can reduce useful bytes per memory transaction and lower effective throughput.

## Kaggle steps
Enable GPU accelerator and run:

```bash
!git clone https://github.com/razz1992/cuda-cpp-gpu-lab.git
%cd cuda-cpp-gpu-lab
!nvidia-smi
!nvcc -O2 -std=c++17 memory_access_patterns.cu -o memory_access_patterns
!./memory_access_patterns 4000000 16 50
```

Arguments: `[output_elements] [stride] [iterations]`.
- `output_elements`: number of floats copied into the output.
- `stride`: source index spacing in the strided kernel. Try 1, 2, 4, 8, 16, and 32.
- `iterations`: repetitions used for event timing.

The default input allocation is 4M x 16 floats (~256 MB). Reduce output_elements if GPU memory is limited, particularly when testing a larger stride.

## What the benchmark measures
- CUDA event timing around repeated kernel launches only. Host-to-device initialization and device-to-host validation are outside the timed region.
- Both kernels have the same output count and contiguous output stores.
- Validation is performed against each kernel's expected source values.
- Reported useful read bandwidth counts only the requested float payload, not the additional memory traffic caused by striding. It is a useful-throughput estimate, not a physical DRAM bandwidth measurement.

## Experiments
Run the same output count and iterations while changing stride:

```bash
!./memory_access_patterns 4000000 1 50
!./memory_access_patterns 4000000 2 50
!./memory_access_patterns 4000000 4 50
!./memory_access_patterns 4000000 8 50
!./memory_access_patterns 4000000 16 50
!./memory_access_patterns 4000000 32 50
```

Use stride 1 as the contiguous-access control. For stride 1, the two kernels intentionally access the same pattern and should have similar timings; they may differ slightly due to measurement noise or launch/code differences.

## Interpretation caution
This is an illustrative access-pattern microbenchmark, not a perfectly isolated measurement of DRAM transactions. Cache behavior, compiler code generation, memory alignment, GPU load, and benchmark noise affect the results. Use Nsight Compute or Nsight Systems if available to inspect memory transactions and cache behavior directly.

## Record results
| Stride | Coalesced kernel (ms) | Strided kernel (ms) | Ratio | Validation |
|---|---:|---:|---:|---|
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 32 | | | | |
