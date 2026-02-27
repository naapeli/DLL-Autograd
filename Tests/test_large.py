import pytest
import torch
from DLL import rand

rtol = 1e-3
atol = 1e-4 

def sync_tensors(dll_tensor):
    t = torch.tensor(dll_tensor.data, dtype=torch.float32).reshape(dll_tensor.shape)
    t.requires_grad = True
    return t

@pytest.fixture(params=[
    ([1024, 1024], [1024, 1024]),          # 1. Large 2D exact match
    ([256, 256, 64], [256, 256, 64]),      # 2. Large 3D exact match
    ([1024, 512], [1, 512]),               # 3. Large 2D broadcast row
    ([128, 128, 128], [128, 1, 128]),      # 4. Large 3D internal broadcast
    ([32, 64, 128, 128], [32, 1, 1, 128])  # 5. Large 4D multi-axis broadcast
])
def large_elementwise_shapes(request):
    return request.param

@pytest.fixture(params=[
    ([1024, 512], [512, 1024]),            # 1. Standard large 2D matmul
    ([512, 2048], [2048, 256]),            # 2. Asymmetric large 2D matmul
    ([64, 256, 128], [64, 128, 256]),      # 3. Batched large 3D matmul
    ([16, 32, 128, 64], [16, 32, 64, 128]) # 4. Batched large 4D matmul
])
def large_matmul_shapes(request):
    return request.param

def test_large_add(large_elementwise_shapes):
    shape_a, shape_b = large_elementwise_shapes
    a_dll = rand(shape_a)
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll + b_dll
    res_t = a_t + b_t
    
    assert res_dll.shape == list(res_t.shape), "Forward shape mismatch in Add"
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_large_mul(large_elementwise_shapes):
    shape_a, shape_b = large_elementwise_shapes
    a_dll = rand(shape_a)
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll * b_dll
    res_t = a_t * b_t
    
    assert res_dll.shape == list(res_t.shape), "Forward shape mismatch in Mul"
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_large_pow(large_elementwise_shapes):
    shape_a, shape_b = large_elementwise_shapes
    
    a_dll = rand(shape_a) + 0.5 
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll ** b_dll
    res_t = a_t ** b_t
    
    assert res_dll.shape == list(res_t.shape), "Forward shape mismatch in Pow"
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_large_matmul(large_matmul_shapes):
    shape_a, shape_b = large_matmul_shapes
    
    a_dll = rand(shape_a)
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll @ b_dll
    res_t = a_t @ b_t
    
    assert res_dll.shape == list(res_t.shape), "Forward shape mismatch in Matmul"
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)