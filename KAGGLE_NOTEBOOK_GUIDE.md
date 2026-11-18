# Kaggle notebook guide

The recommended path is to import `KAGGLE_NOTEBOOK_GUIDE.ipynb` and run its cells in order.

## Experiment 1 — baseline
The notebook verifies the GPU and CUDA compiler, clones or pulls the repository, compiles `vector_add.cu` and `benchmark.cu`, and compares CPU and GPU timings.

## Experiment 2 — GPU memory optimization
Compile and run:
```bash
!nvcc -O2 -std=c++17 memory_optimization.cu -o memory_optimization
!./memory_optimization
```
Command format: `./memory_optimization [element_count] [iterations]`.

Suggested sizes:
```bash
!./memory_optimization 1000000 50
!./memory_optimization 10000000 50
!./memory_optimization 50000000 20
```

### Three approaches
1. **A — Allocate/free every iteration:** allocate three device buffers, copy input, launch the kernel, copy output, and free buffers on every iteration.
2. **B — Reuse device allocations:** allocate device buffers once, then reuse them while copying input and output for each iteration.
3. **C — Pinned host memory + asynchronous stream:** allocate page-locked host buffers once, enqueue asynchronous host-to-device copies, kernel execution, and device-to-host copy in one CUDA stream; synchronize after the batch.

### What to focus on
- Compare A vs B to isolate the cost of repeated `cudaMalloc`/`cudaFree`.
- Compare B vs C to study pinned host memory and asynchronous stream submission.
- Verify all three validation statuses are PASS.
- Repeat each size several times and record GPU model, element count, iterations, and timing.
- C is a batched stream-throughput measurement, while A/B synchronize through blocking copies each iteration. Treat this as an educational comparison, not a perfectly controlled microbenchmark.
- Pinned memory consumes a limited system resource. Use it selectively.
- The current one-stream pipeline serializes H2D → kernel → D2H; overlapping transfers and computation requires independent buffers and a more advanced multi-stream/double-buffered design.

Kaggle sessions are ephemeral. Save important results before the session ends.
