# CUDA C++ GPU experiments

A practical CUDA C++ learning project for Kaggle NVIDIA GPUs.

## Experiments
- `vector_add.cu`: introductory CUDA vector addition and correctness check.
- `benchmark.cu`: CPU vs GPU kernel-only and end-to-end timing.
- `memory_optimization.cu`: compares device allocation on every iteration, reusable device allocations, and pinned host memory with asynchronous transfers on a CUDA stream.

## Kaggle quick start
Enable **Notebook Settings → Accelerator → GPU**, then import `KAGGLE_NOTEBOOK_GUIDE.ipynb` or run:
```bash
!git clone https://github.com/razz1992/cuda-cpp-gpu-lab.git
%cd cuda-cpp-gpu-lab
!nvidia-smi
!nvcc --version
!nvcc -O2 -std=c++17 memory_optimization.cu -o memory_optimization
!./memory_optimization
```

Experiment 2 syntax: `./memory_optimization [element_count] [iterations]`.
Try `!./memory_optimization 1000000 50`, `!./memory_optimization 10000000 50`, and `!./memory_optimization 50000000 20`.

## What to compare
- **A — Allocate/free each iteration:** includes three device allocations and frees per iteration, transfers, and kernel execution.
- **B — Reuse device allocations:** device buffers are allocated once outside the timed loop; transfers and kernel execute each iteration.
- **C — Pinned + async stream:** pinned host buffers are allocated once; asynchronous H2D copies, kernel, and D2H copy are queued in one stream and synchronized after the batch.

Approach C measures batched stream throughput, so compare it as a throughput experiment rather than assuming its per-iteration wall-clock time is identical in methodology to A/B. Pinned memory is a limited resource; allocate only what is needed. A single stream preserves operation order but does not demonstrate overlap between independent copy and compute operations.

## Benchmark discipline
Warm up the CUDA context, run several iterations, compare correctness, and repeat measurements. Results depend on GPU allocation, host CPU, driver, CUDA Toolkit, and session load. Avoid interpreting one run as a universal performance result.
