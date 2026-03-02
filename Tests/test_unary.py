import pytest
import torch
import random
from DLL import rand


rtol = 1e-3
atol = 1e-4

def sync_tensors(dll_tensor):
    t = torch.tensor(dll_tensor.data, dtype=torch.float32).reshape(dll_tensor.shape)
    t.requires_grad = True
    return t

@pytest.fixture(params=range(15))
def setup_mixed_tensors():
    rank = random.randint(1, 5)
    shape = [random.randint(2, 6) for _ in range(rank)]
    a_dll = (rand(shape) * 2.0) - 1.0 
    return a_dll, sync_tensors(a_dll)

@pytest.fixture(params=range(10))
def setup_positive_tensors():
    rank = random.randint(1, 5)
    shape = [random.randint(2, 6) for _ in range(rank)]
    a_dll = rand(shape) + 0.5 
    return a_dll, sync_tensors(a_dll)

@pytest.fixture(params=range(15))
def setup_ndim_tensors():
    rank = random.randint(2, 5)
    shape = [random.randint(2, 6) for _ in range(rank)]
    a_dll = (rand(shape) * 2.0) - 1.0
    return a_dll, sync_tensors(a_dll)

@pytest.mark.parametrize("op_name", ["relu", "tanh", "sigmoid", "exp", "abs", "sin", "cos"])
def test_mixed_unary_ops(setup_mixed_tensors, op_name):
    a_dll, a_t = setup_mixed_tensors
    a_dll.requires_grad = True
    res_dll = getattr(a_dll, op_name)()
    res_t = getattr(a_t, op_name)()
    
    assert res_dll.shape == list(res_t.shape)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)

@pytest.mark.parametrize("op_name", ["log", "sqrt", "cbrt"])
def test_positive_unary_ops(setup_positive_tensors, op_name):
    a_dll, a_t = setup_positive_tensors
    a_dll.requires_grad = True
    res_dll = getattr(a_dll, op_name)()
    
    if op_name == "cbrt":
        res_t = torch.pow(a_t, 1.0 / 3.0)
    else:
        res_t = getattr(a_t, op_name)()
        
    assert res_dll.shape == list(res_t.shape)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)

def test_softmax(setup_mixed_tensors):
    a_dll, a_t = setup_mixed_tensors
    a_dll.requires_grad = True
    dim = random.randint(-len(a_dll.shape), len(a_dll.shape) - 1)
    res_dll = a_dll.softmax(dim=dim)
    res_t = a_t.softmax(dim=dim)
    
    assert res_dll.shape == list(res_t.shape)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)

def test_transpose(setup_ndim_tensors):
    a_dll, a_t = setup_ndim_tensors
    a_dll.requires_grad = True
    dim0 = random.randint(-len(a_dll.shape), len(a_dll.shape) - 1)
    dim1 = random.randint(-len(a_dll.shape), len(a_dll.shape) - 1)
    
    res_dll = a_dll.transpose(dim0, dim1)
    res_t = a_t.transpose(dim0, dim1)
    
    assert res_dll.shape == list(res_t.shape)
    
    loss_weights = rand(res_dll.shape)
    loss_t = sync_tensors(loss_weights)
    
    (res_dll * loss_weights).sum().backward()
    (res_t * loss_t).sum().backward()
    
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)

def test_chained_unary_ops(setup_positive_tensors):
    a_dll, a_t = setup_positive_tensors
    a_dll.requires_grad = True
    
    res_dll = a_dll.exp().sqrt().log().sin().relu()
    res_t = a_t.exp().sqrt().log().sin().relu()
    
    assert res_dll.shape == list(res_t.shape)
    
    res_dll.sum().backward()
    res_t.sum().backward()
    
    torch.testing.assert_close(torch.tensor(res_dll.data, dtype=torch.float32).reshape(res_dll.shape), res_t, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch.tensor(a_dll.grad.data, dtype=torch.float32).reshape(a_dll.shape), a_t.grad, rtol=rtol, atol=atol)
