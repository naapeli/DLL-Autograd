import pytest
import torch
import random
from DLL import rand


rtol = 1e-4

def sync_tensors(dll_tensor):
    t = torch.tensor(dll_tensor.data, dtype=torch.float32).reshape(dll_tensor.shape)
    t.requires_grad = True
    return t

def apply_random_reduction(dll_t, torch_t):
    reduction_type = random.choice(["sum", "mean", "prod", "max", "min", "var", "std"])
    mode = random.randint(0, 1)
    
    kwargs = {}
    if reduction_type in ["var", "std"]:
        kwargs["unbiased"] = random.choice([True, False])

    if mode == 0:
        res_dll = getattr(dll_t, reduction_type)(**kwargs)
        res_torch = getattr(torch_t, reduction_type)(**kwargs)
    else:
        dim = random.randint(0, len(dll_t.shape) - 1)
        keepdim = random.choice([True, False])
        res_dll = getattr(dll_t, reduction_type)(dim=dim, keepdim=keepdim, **kwargs)
        res_torch = getattr(torch_t, reduction_type)(dim=dim, keepdim=keepdim, **kwargs)
        
        if reduction_type in ["max", "min"]:
            res_torch = res_torch.values
            
    return res_dll, res_torch

@pytest.fixture(params=range(50))
def setup_tensors():
    rank = random.randint(1, 4)
    shape = [random.randint(2, 6) for _ in range(rank)] 
    
    a_dll = rand(shape)
    b_dll = rand(shape)
    
    a_torch = sync_tensors(a_dll)
    b_torch = sync_tensors(b_dll)
    
    return a_dll, b_dll, a_torch, b_torch

def test_randomized_reductions(setup_tensors):
    a_dll, b_dll, a_t, b_t = setup_tensors
    a_dll.requires_grad = True
    b_dll.requires_grad = True

    mid_dll = (a_dll * b_dll) + (a_dll ** 2.0)
    mid_t = (a_t * b_t) + (a_t ** 2.0)

    red_dll, red_t = apply_random_reduction(mid_dll, mid_t)

    if len(red_dll.shape) > 1 or (len(red_dll.shape) == 1 and red_dll.shape[0] > 1):
        final_dll = red_dll.sum()
        final_t = red_t.sum()
    else:
        final_dll = red_dll
        final_t = red_t

    final_dll.backward()
    final_t.backward()

    assert final_dll.data[0] == pytest.approx(final_t.item(), rel=rtol)
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=rtol)
    assert b_dll.grad.data == pytest.approx(b_t.grad.flatten().tolist(), rel=rtol, abs=rtol)

def test_random_nested_reductions(setup_tensors):
    a_dll, _, a_t, _ = setup_tensors
    if len(a_dll.shape) < 2: return
    
    a_dll.requires_grad = True
    
    red_type_1 = random.choice(["sum", "mean", "prod", "max", "min", "var", "std"])
    dim = random.randint(0, len(a_dll.shape) - 1)
    keepdim = random.choice([True, False])

    kwargs_1 = {}
    if red_type_1 in ["var", "std"]:
        kwargs_1["unbiased"] = random.choice([True, False])

    mid_dll = getattr(a_dll, red_type_1)(dim=dim, keepdim=keepdim, **kwargs_1)
    mid_t = getattr(a_t, red_type_1)(dim=dim, keepdim=keepdim, **kwargs_1)
    
    if red_type_1 in ["max", "min"]:
        mid_t = mid_t.values

    red_type_2 = random.choice(["sum", "mean", "prod", "max", "min", "var", "std"])
    
    kwargs_2 = {}
    if red_type_2 in ["var", "std"]:
        kwargs_2["unbiased"] = random.choice([True, False])
        
    L_dll = getattr(mid_dll, red_type_2)(**kwargs_2)
    L_t = getattr(mid_t, red_type_2)(**kwargs_2)

    L_dll.backward()
    L_t.backward()

    assert L_dll.data[0] == pytest.approx(L_t.item(), rel=rtol)
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=rtol)