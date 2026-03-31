import DLL
import pytest
import numpy as np

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not DLL.gpu_available(), reason="GPU not available")
]

def test_gpu_available():
    assert DLL.gpu_available(), "GPU should be available after build with OpenCL"
    print(f"Testing on device: {DLL.gpu_device_name()}")

def test_gpu_add():
    a = DLL.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]).to("gpu")
    b = DLL.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2]).to("gpu")
    c = a + b
    expected = np.array([6, 8, 10, 12])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"
    assert c.device == "gpu"

def test_gpu_matmul():
    a = DLL.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]).to("gpu")
    b = DLL.Tensor([1.0, 0.0, 0.0, 1.0], [2, 2]).to("gpu") # Identity
    c = a @ b
    expected = np.array([1, 2, 3, 4])
    assert np.allclose(c.data, expected)
    assert c.device == "gpu"

def test_gpu_autograd():
    a = DLL.Tensor([2.0], [1]).to("gpu")
    a.requires_grad = True
    b = a * a # x^2
    b.backward()
    assert a.grad.data[0] == 4.0

def test_mixed_device():
    a = DLL.Tensor([1.0], [1]).to("gpu")
    b = DLL.Tensor([2.0], [1]) # CPU
    # This should trigger auto-transfer to GPU
    with pytest.warns(UserWarning, match="Tensors on different devices"):
        c = a + b
    assert c.device == "gpu"
    assert c.data[0] == 3.0
