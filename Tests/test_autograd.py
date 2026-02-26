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

@pytest.fixture(params=range(50))
def setup_tensors():
    rank = random.randint(1, 4)
    shape = [random.randint(1, 6) for _ in range(rank)]
    
    a_dll = rand(shape) + 0.5
    b_dll = rand(shape) + 0.5
    
    a_torch = sync_tensors(a_dll)
    b_torch = sync_tensors(b_dll)
    
    return a_dll, b_dll, a_torch, b_torch

@pytest.fixture(params=[
    ([3, 4], [4, 5]),
    ([2, 3, 4], [2, 4, 5]), 
    ([10, 3, 4], [4, 5]),
    ([3, 4], [10, 4, 5]),
    ([6, 3, 4], [3, 4, 7, 5, 6, 4, 2]),
    ([2, 1, 3, 4], [1, 5, 4, 2]),
    ([1, 4, 3, 4], [2, 1, 4, 2]),
    ([2, 3, 1, 2, 4], [1, 1, 5, 4, 3])
])
def setup_matmul_tensors(request):
    shape_a, shape_b = request.param
    
    a_dll = rand(shape_a)
    b_dll = rand(shape_b)
    
    a_torch = sync_tensors(a_dll)
    b_torch = sync_tensors(b_dll)
    
    return a_dll, b_dll, a_torch, b_torch

def test_matmul_forward_backward(setup_matmul_tensors):
    a_dll, b_dll, a_t, b_t = setup_matmul_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    res_dll = a_dll @ b_dll
    res_t = a_t @ b_t

    assert res_dll.shape == list(res_t.shape), f"Shape mismatch: DLL={res_dll.shape}, Torch={list(res_t.shape)}"
    assert res_dll.data == pytest.approx(res_t.flatten().tolist(), rel=rtol, abs=atol)

    res_dll.sum().backward()
    res_t.sum().backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_matmul_in_computational_graph(setup_matmul_tensors):
    a_dll, b_dll, a_t, b_t = setup_matmul_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    res_dll = ((a_dll @ b_dll) + 1.0) ** 2.0
    L_dll = res_dll.sum()
    L_dll.backward()

    res_t = ((a_t @ b_t) + 1.0) ** 2.0
    L_t = res_t.sum()
    L_t.backward()

    assert L_dll.data[0] == pytest.approx(L_t.item(), rel=rtol)
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_torch_parity_complex(setup_tensors):
    a_dll, b_dll, a_t, b_t = setup_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    res_dll = (((a_dll * b_dll) + (a_dll ** 2.0)) / (a_dll - b_dll)).sum()
    res_dll.backward()

    res_t = (((a_t * b_t) + (a_t ** 2.0)) / (a_t - b_t)).sum()
    res_t.backward()

    assert res_dll.data[0] == pytest.approx(res_t.item(), rel=rtol)
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_exponential_chaos(setup_tensors):
    a_dll, b_dll, a_t, b_t = setup_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    L_dll = ((a_dll ** b_dll) + (b_dll ** a_dll)).sum()
    L_dll.backward()

    L_t = ((a_t ** b_t) + (b_t ** a_t)).sum()
    L_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_multi_path_accumulation(setup_tensors):
    a_dll, _, a_t, _ = setup_tensors
    a_dll.requires_grad = True
    
    L_dll = (((a_dll * a_dll) + a_dll) * a_dll).sum()
    L_dll.backward()

    L_t = (((a_t * a_t) + a_t) * a_t).sum()
    L_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_nested_quotient_stability(setup_tensors):
    a_dll, _, a_t, _ = setup_tensors
    a_dll.requires_grad = True

    L_dll = (1.0 / (1.0 + (1.0 / a_dll))).sum()
    L_dll.backward()

    L_t = (1.0 / (1.0 + (1.0 / a_t))).sum()
    L_t.backward()

    assert L_dll.data[0] == pytest.approx(L_t.item(), rel=rtol)
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_long_chain_rule(setup_tensors):
    a_dll, _, a_t, _ = setup_tensors
    a_dll.requires_grad = True

    y_dll = ((((a_dll * 2.0) + 1.0) ** 2.0) / 3.0).sum()
    y_dll.backward()

    y_t = ((((a_t * 2.0) + 1.0) ** 2.0) / 3.0).sum()
    y_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_variable_base_and_exponent_mixed(setup_tensors):
    a_dll, b_dll, a_t, b_t = setup_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    L_dll = ((a_dll ** b_dll) + (2.0 ** a_dll) - (b_dll ** 3.0)).sum()
    L_dll.backward()

    L_t = ((a_t ** b_t) + (2.0 ** a_t) - (b_t ** 3.0)).sum()
    L_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_reciprocal_multiplication(setup_tensors):
    a_dll, b_dll, a_t, b_t = setup_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    L_dll = (a_dll * (1.0 / b_dll)).sum()
    L_dll.backward()

    L_t = (a_t * (1.0 / b_t)).sum()
    L_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_diamond_graph(setup_tensors):
    a_dll, _, a_t, _ = setup_tensors
    a_dll.requires_grad = True

    c_dll = a_dll + 2.0
    d_dll = a_dll * 3.0
    L_dll = (c_dll * d_dll).sum()
    L_dll.backward()

    c_t = a_t + 2.0
    d_t = a_t * 3.0
    L_t = (c_t * d_t).sum()
    L_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)
