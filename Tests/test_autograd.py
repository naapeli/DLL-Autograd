import pytest
import torch
import random
from DLL import Tensor, rand


rtol = 1e-3

def to_torch(dll_tensor):
    t = torch.tensor(dll_tensor.data, requires_grad=True, dtype=torch.float32)
    return t

@pytest.fixture(params=range(50))
def setup_scalars():
    a_val = random.uniform(0.5, 2.0)
    b_val = random.uniform(0.5, 2.0)
    
    a_dll = Tensor([a_val], [1, 1])
    b_dll = Tensor([b_val], [1, 1])
    
    return a_dll, b_dll, a_val, b_val

def test_torch_parity_complex(setup_scalars):
    a_dll, b_dll, a_val, b_val = setup_scalars
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    res_dll = ((a_dll * b_dll) + (a_dll ** 2.0)) / (a_dll - b_dll)
    res_dll.backward()

    a_torch = torch.tensor([a_val], requires_grad=True)
    b_torch = torch.tensor([b_val], requires_grad=True)
    
    res_torch = ((a_torch * b_torch) + (a_torch ** 2.0)) / (a_torch - b_torch)
    res_torch.backward()

    assert res_dll.data[0] == pytest.approx(res_torch.item(), rel=rtol)
    assert a_dll.grad.data[0] == pytest.approx(a_torch.grad.item(), rel=rtol)
    assert b_dll.grad.data[0] == pytest.approx(b_torch.grad.item(), rel=rtol)

def test_exponential_chaos(setup_scalars):
    a_dll, b_dll, a_val, b_val = setup_scalars
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    L_dll = (a_dll ** b_dll) + (b_dll ** a_dll)
    L_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    b_t = torch.tensor([b_val], requires_grad=True)
    L_t = (a_t ** b_t) + (b_t ** a_t)
    L_t.backward()

    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)
    assert b_dll.grad.data[0] == pytest.approx(b_t.grad.item(), rel=rtol)

def test_multi_path_accumulation(setup_scalars):
    a_dll, _, a_val, _ = setup_scalars
    a_dll.requires_grad = True
    
    L_dll = ((a_dll * a_dll) + a_dll) * a_dll
    L_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    L_t = ((a_t * a_t) + a_t) * a_t
    L_t.backward()

    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)

def test_nested_quotient_stability(setup_scalars):
    a_dll, _, a_val, _ = setup_scalars
    a_dll.requires_grad = True

    L_dll = 1.0 / (1.0 + (1.0 / a_dll))
    L_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    L_t = 1.0 / (1.0 + (1.0 / a_t))
    L_t.backward()

    assert L_dll.data[0] == pytest.approx(L_t.item(), rel=rtol)
    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)

def test_long_chain_rule(setup_scalars):
    a_dll, _, a_val, _ = setup_scalars
    a_dll.requires_grad = True

    y_dll = (((a_dll * 2.0) + 1.0) ** 2.0) / 3.0
    y_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    y_t = (((a_t * 2.0) + 1.0) ** 2.0) / 3.0
    y_t.backward()

    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)

def test_variable_base_and_exponent_mixed(setup_scalars):
    a_dll, b_dll, a_val, b_val = setup_scalars
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    L_dll = (a_dll ** b_dll) + (2.0 ** a_dll) - (b_dll ** 3.0)
    L_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    b_t = torch.tensor([b_val], requires_grad=True)
    L_t = (a_t ** b_t) + (2.0 ** a_t) - (b_t ** 3.0)
    L_t.backward()

    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)
    assert b_dll.grad.data[0] == pytest.approx(b_t.grad.item(), rel=rtol)

def test_reciprocal_multiplication(setup_scalars):
    a_dll, b_dll, a_val, b_val = setup_scalars
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    L_dll = a_dll * (1.0 / b_dll)
    L_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    b_t = torch.tensor([b_val], requires_grad=True)
    L_t = a_t * (1.0 / b_t)
    L_t.backward()

    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)
    assert b_dll.grad.data[0] == pytest.approx(b_t.grad.item(), rel=rtol)

def test_diamond_graph(setup_scalars):
    a_dll, _, a_val, _ = setup_scalars
    a_dll.requires_grad = True

    c_dll = a_dll + 2.0
    d_dll = a_dll * 3.0
    L_dll = c_dll * d_dll
    L_dll.backward()

    a_t = torch.tensor([a_val], requires_grad=True)
    c_t = a_t + 2.0
    d_t = a_t * 3.0
    L_t = c_t * d_t
    L_t.backward()

    assert a_dll.grad.data[0] == pytest.approx(a_t.grad.item(), rel=rtol)
