# DLL-Autograd

A lightweight tensor library with automatic differentiation, written in C++ with Python bindings. Inspired by PyTorch's API, DLL-Autograd provides a `Tensor` class that tracks computation graphs and supports backpropagation through arbitrary expressions.

The C++ core is exposed to Python via [pybind11](https://github.com/pybind/pybind11), giving near-native performance with a familiar Pythonic interface. Matrix multiplication is accelerated by [OpenBLAS](https://www.openblas.net/) or [CLBlast](https://github.com/CNugteren/CLBlast), and element-wise operations are parallelized with OpenMP or offloaded to the GPU via OpenCL.

## Features

**Tensor operations** with broadcasting support:
- Arithmetic: `+`, `-`, `*`, `/`, `**`, `@` (matmul)
- Unary: `exp`, `log`, `sqrt`, `cbrt`, `abs`, `sin`, `cos`, `transpose`
- Activations: `relu`, `tanh`, `sigmoid`, `softmax`
- Reductions: `sum`, `mean`, `prod`, `max`, `min`, `var`, `std` (global or along a dimension)
- Indexing and slicing

**Cross-Vendor GPU Acceleration** (OpenCL + CLBlast):
- Supports AMD, NVIDIA, and Intel GPUs.
- Seamless device management: `.to("gpu")`, `.cpu()`, `.is_gpu()`.
- Automatic mixed-device handling with performance warnings.

**Automatic differentiation**:
- Reverse-mode autodiff (backpropagation) through all supported operations.
- `requires_grad` flag to selectively track gradients.
- `backward()` to compute gradients, `zero_grad()` to reset.

## Example

```python
import DLL

# Core neural network components on GPU
w1 = DLL.randn((30, 20), std=0.1).to("gpu"); w1.requires_grad = True
b1 = DLL.randn((20,), std=0.1).to("gpu");    b1.requires_grad = True
w2 = DLL.randn((20, 1), std=0.1).to("gpu");  w2.requires_grad = True
b2 = DLL.randn((1,), std=0.1).to("gpu");     b2.requires_grad = True

x = DLL.randn((8, 30)).to("gpu")   # Input batch on GPU
y = DLL.randn((8, 1)).to("gpu")    # Targets on GPU

# Forward pass (executed on GPU kernels)
h = (x @ w1 + b1).relu()
pred = (h @ w2 + b2).sigmoid()
loss = ((y - pred) ** 2).mean()

# Backward pass (autograd on GPU)
loss.backward()

# Update weights
lr = 0.01
w1 = w1 - lr * w1.grad
```

## Requirements

### Core Dependencies
- **C++17** compiler (GCC or Clang)
- **CMake** ≥ 3.15
- **OpenBLAS** — optimized matrix multiplication for CPU.
- **OpenMP** — parallel CPU threading.
- **Python** ≥ 3.14 with [uv](https://github.com/astral-sh/uv).

### GPU Support (Optional)
- **OpenCL 1.2** Runtimes (ROCm for AMD, CUDA for NVIDIA, or Intel Compute Runtime).
- **CLBlast** — high-performance BLAS library for OpenCL.

Install on Ubuntu/Debian:
```bash
sudo apt install opencl-headers ocl-icd-opencl-dev libclblast-dev
```

## Building

Compile the extension:

```bash
# For CPU-only build
cmake -S . -B build && cmake --build build

# For GPU-enabled build (automatically detects OpenCL/CLBlast)
cmake -S . -B build -DDLL_USE_GPU=ON && cmake --build build
```

## GPU Hardware Overrides (AMD RDNA2/3)

For consumer AMD GPUs (like the RX 6000/7000 series), the ROCm OpenCL stack might not officially support your specific device ID (e.g., `gfx1031`). To enable support, set the following environment variable in your shell:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## Running Tests

```bash
# Run all tests (including GPU if available)
uv run python -m pytest Tests/

# Run specifically GPU tests
uv run python -m pytest -m gpu Tests/
```
