# DLL-Autograd

A lightweight tensor library with automatic differentiation, written in C++ with Python bindings. Inspired by PyTorch's API, DLL-Autograd provides a `Tensor` class that tracks computation graphs and supports backpropagation through arbitrary expressions.

The C++ core is exposed to Python via [pybind11](https://github.com/pybind/pybind11), giving near-native performance with a familiar Pythonic interface. Matrix multiplication is accelerated by [OpenBLAS](https://www.openblas.net/), and element-wise operations are parallelized with OpenMP.

## Features

**Tensor operations** with broadcasting support:
- Arithmetic: `+`, `-`, `*`, `/`, `**`, `@` (matmul)
- Unary: `exp`, `log`, `sqrt`, `cbrt`, `abs`, `sin`, `cos`, `transpose`
- Activations: `relu`, `tanh`, `sigmoid`, `softmax`
- Reductions: `sum`, `mean`, `prod`, `max`, `min`, `var`, `std` (global or along a dimension)
- Indexing and slicing

**Automatic differentiation**:
- Reverse-mode autodiff (backpropagation) through all supported operations
- `requires_grad` flag to selectively track gradients
- `backward()` to compute gradients, `zero_grad()` to reset

**Random tensor generation**: `rand`, `randn` with configurable parameters and seeding.

## Example

```python
import DLL

# Define a simple 2-layer neural network
w1 = DLL.randn((30, 20), std=0.1); w1.requires_grad = True
b1 = DLL.randn((20,), std=0.1);    b1.requires_grad = True
w2 = DLL.randn((20, 1), std=0.1);  w2.requires_grad = True
b2 = DLL.randn((1,), std=0.1);     b2.requires_grad = True

x = DLL.randn((8, 30))   # batch of 8, 30 features
y = DLL.randn((8, 1))    # targets

# Forward pass
h = (x @ w1 + b1).relu()
pred = (h @ w2 + b2).sigmoid()
loss = ((y - pred) ** 2).mean()

# Backward pass — gradients are computed for all tensors with requires_grad=True
loss.backward()

# Update weights
lr = 0.01
w1 = w1 - lr * w1.grad
```

See [Examples/breast_cancer.py](Examples/breast_cancer.py) for a full training loop on real data.

## Requirements

- **C++17** compiler (GCC or Clang)
- **CMake** ≥ 3.15
- **OpenBLAS** — used for optimized matrix multiplication (`cblas_sgemm`). Install with:
  ```bash
  # Debian/Ubuntu
  sudo apt install libopenblas-dev
  ```
- **OpenMP** — used for parallelizing element-wise and reduction operations. Typically bundled with GCC; on macOS install via `brew install libomp`.
- **Python** ≥ 3.14 with dependencies managed by [uv](https://github.com/astral-sh/uv) (see `uv.lock`)

## Building

Compile the C++ extension:

```bash
cmake -S . -B build && cmake --build build
```

## Running Tests

```bash
python -m pytest Tests/
```
