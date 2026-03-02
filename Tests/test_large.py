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
    ([1, 32, 256, 64], [2, 32, 256, 64]),      # 2. Large 3D exact match
    ([1024, 512], [1, 512]),               # 3. Large 2D broadcast row
    ([128, 128, 128], [128, 1, 128]),      # 4. Large 3D internal broadcast
    ([2, 64, 128, 128], [2, 1, 1, 128])  # 5. Large 4D multi-axis broadcast
])
def large_elementwise_shapes(request):
    return request.param

@pytest.fixture(params=[
    ([1024, 512], [512, 1024]),            # 1. Standard large 2D matmul
    ([512, 2048], [2048, 256]),            # 2. Asymmetric large 2D matmul
    ([64, 256, 128], [64, 128, 256]),      # 3. Batched large 3D matmul
    ([2, 32, 128, 64], [2, 32, 64, 128]) # 4. Batched large 4D matmul
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
    
    # exact comparison of every element takes a lot of time. Hence we compare the sums.
    # assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    # assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    # assert a_dll.grad.sum().data[0] == pytest.approx(a_t.grad.flatten().sum().item(), rel=rtol, abs=atol)
    # assert b_dll.grad.sum().data[0] == pytest.approx(b_t.grad.flatten().sum().item(), rel=rtol, abs=atol)
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(b_dll.grad.data, dtype=torch.float32).reshape(b_dll.shape), b_t.grad, rtol=rtol, atol=atol)

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
    
    # assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    # assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(b_dll.grad.data, dtype=torch.float32).reshape(b_dll.shape), b_t.grad, rtol=rtol, atol=atol)

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
    
    # assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    # assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(b_dll.grad.data, dtype=torch.float32).reshape(b_dll.shape), b_t.grad, rtol=rtol, atol=atol)

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
    
    # assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    # assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(b_dll.grad.data, dtype=torch.float32).reshape(b_dll.shape), b_t.grad, rtol=rtol, atol=atol)
