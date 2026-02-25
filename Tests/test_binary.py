import pytest
import torch
from DLL import rand

rtol = 1e-3
atol = 1e-5

def sync_tensors(dll_tensor):
    t = torch.tensor(dll_tensor.data, dtype=torch.float32).reshape(dll_tensor.shape)
    t.requires_grad = True
    return t

@pytest.fixture(params=[
    ([3, 4], [3, 4]),          # 1. No broadcasting (Exact match)
    ([3, 4], [1, 4]),          # 2. Broadcast row (2D + 1D effectively)
    ([3, 4], [3, 1]),          # 3. Broadcast column
    ([4, 1], [1, 5]),          # 4. Outer product (Both broadcast to [4, 5])
    ([2, 3, 4], [4]),          # 5. Different ranks (3D + 1D)
    ([2, 3, 4], [1, 3, 1]),    # 6. Different ranks with internal broadcasting
    ([1], [4, 4]),             # 7. Scalar-like to Matrix
    ([5, 1, 4, 1], [1, 3, 1, 2]) # 8. 4D chaotic multi-axis broadcasting
])
def broadcast_shapes(request):
    return request.param

def test_broadcast_add(broadcast_shapes):
    shape_a, shape_b = broadcast_shapes
    a_dll = rand(shape_a)
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll + b_dll
    res_t = a_t + b_t
    
    assert res_dll.shape == list(res_t.shape), "Forward shape mismatch"
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_broadcast_mul(broadcast_shapes):
    shape_a, shape_b = broadcast_shapes
    a_dll = rand(shape_a)
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll * b_dll
    res_t = a_t * b_t
    
    assert res_dll.shape == list(res_t.shape)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_broadcast_pow(broadcast_shapes):
    shape_a, shape_b = broadcast_shapes
    
    a_dll = rand(shape_a) + 0.5 
    b_dll = rand(shape_b)
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = a_dll ** b_dll
    res_t = a_t ** b_t
    
    assert res_dll.shape == list(res_t.shape)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_broadcast_complex_expression(broadcast_shapes):
    shape_a, shape_b = broadcast_shapes
    
    a_dll = rand(shape_a) + 0.5
    b_dll = rand(shape_b) + 0.5
    
    a_dll.requires_grad = True
    b_dll.requires_grad = True
    
    a_t = sync_tensors(a_dll)
    b_t = sync_tensors(b_dll)
    
    res_dll = (a_dll - b_dll) / (a_dll * b_dll)
    res_t = (a_t - b_t) / (a_t * b_t)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)