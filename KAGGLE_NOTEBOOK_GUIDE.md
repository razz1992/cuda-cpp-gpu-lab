# Kaggle notebook cells

## 1. Verify the allocated NVIDIA GPU
```bash
!nvidia-smi
```

## 2. Verify the CUDA compiler
```bash
!nvcc --version
```

## 3. Get the source
Choose one option:

Upload the `vector_add.cu` file through the notebook's Add Input panel, then locate it:
```bash
!find /kaggle/input -name 'vector_add.cu'
```

Or clone the Git repository (replace with your repository URL):
```bash
!git clone YOUR_GIT_REPOSITORY_URL
%cd YOUR_REPOSITORY_PATH
```

## 4. Compile
```bash
!nvcc -O2 -std=c++17 vector_add.cu -o vector_add
```

## 5. Execute and validate
```bash
!./vector_add
!./vector_add 10000000
!./vector_add 50000000
```

## 6. Inspect GPU utilization
Run while a sufficiently long GPU workload is executing:
```bash
!nvidia-smi
```

If `nvcc` is unavailable, confirm that the notebook accelerator is set to GPU and inspect `!which nvcc`. Do not assume installing a Python CUDA package installs the CUDA C++ compiler.
