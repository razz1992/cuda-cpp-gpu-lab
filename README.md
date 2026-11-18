# CUDA C++ GPU experiments

A minimal CUDA C++ project for testing NVIDIA GPU execution on Kaggle.

## Files
- `vector_add.cu`: vector addition kernel, device information, CUDA error checks, and output validation.
- `1.cpp`: original example retained unchanged.

## Kaggle GPU test
1. Open Kaggle Notebooks and create a new notebook.
2. In Notebook Settings, set Accelerator to **GPU**.
3. Run:
   ```bash
   !nvidia-smi
   !nvcc --version
   ```
4. Upload `vector_add.cu` as a notebook input, or clone this repository.
5. Compile and run:
   ```bash
   !nvcc -O2 -std=c++17 vector_add.cu -o vector_add
   !./vector_add
   ```
6. Try larger workloads:
   ```bash
   !./vector_add 10000000
   !./vector_add 50000000
   ```

## Notes
- The code uses CUDA C++ kernel syntax, so compile with `nvcc`, not a regular C++ compiler.
- GPU availability, GPU model, driver and CUDA Toolkit version depend on the Kaggle session allocation.
- The sample verifies correctness; it is not a rigorous CPU-vs-GPU benchmark. Kernel launch and host/device transfer overhead matter, especially for small arrays.
- Kaggle sessions are ephemeral. Save notebook outputs and code to a persistent Kaggle Dataset or Git repository as needed.
