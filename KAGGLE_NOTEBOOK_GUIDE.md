# Kaggle notebook cells

Run these cells in order after setting **Notebook Settings → Accelerator → GPU**.

## 1. Verify the allocated NVIDIA GPU and CUDA compiler
```bash
!nvidia-smi
!nvcc --version
!which nvcc
```

If `nvcc` is missing, the session has no CUDA C++ compiler available. A CUDA-enabled PyTorch install does not necessarily include the compiler.

## 2. Clone or update the project
For a fresh notebook:
```bash
!git clone https://github.com/razz1992/cuda-cpp-gpu-lab.git
%cd cuda-cpp-gpu-lab
```
If already cloned in the current session:
```bash
%cd /kaggle/working/cuda-cpp-gpu-lab
!git pull origin main
```

## 3. Compile and run the correctness example
```bash
!nvcc -O2 -std=c++17 vector_add.cu -o vector_add
!./vector_add
!./vector_add 10000000
```

## 4. Compile and run CPU-vs-GPU benchmark
```bash
!nvcc -O2 -std=c++17 benchmark.cu -o benchmark
!./benchmark
```
Arguments are `./benchmark [element_count] [iterations]`. Suggested experiments:
```bash
!./benchmark 1000000 100
!./benchmark 10000000 100
!./benchmark 50000000 30
```

## 5. Interpret the measurements
- **CPU average**: repeated CPU vector addition, averaged per iteration.
- **GPU kernel average**: CUDA kernel execution only, measured with CUDA events; excludes host/device transfers.
- **GPU end-to-end average**: host-to-device copies + kernel + device-to-host copy, averaged per iteration.
- Compare CPU with GPU end-to-end for a practical comparison. Kernel-only speedup is useful for understanding GPU compute but is not an apples-to-apples application latency comparison.
- Small workloads may be slower on GPU because launch and transfer overhead dominate. Results vary by GPU model, allocation, system load, and memory behavior.
- The CPU and GPU results are validated before the program exits.

Kaggle sessions are ephemeral. Save important output or notebook versions outside the running session.
