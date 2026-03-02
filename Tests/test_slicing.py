import pytest
import torch

from DLL import rand, randn


rtol = 1e-3
atol = 1e-4

def sync_tensors(dll_tensor):
    t = torch.tensor(dll_tensor.data, dtype=torch.float32).reshape(dll_tensor.shape)
    t.requires_grad = True
    return t

@pytest.fixture
def complex_tensor():
    shape = (4, 5, 6)
    a_dll = randn(shape)
    a_torch = sync_tensors(a_dll)
    return a_dll, a_torch

@pytest.mark.parametrize("slices", [
    (slice(0, 2),),                # Basic 1D slice on 3D tensor
    (0, slice(1, 3)),              # Integer index + slice
    (slice(1, 4), slice(0, 2)),    # Multiple slices
    (1, 2, 1),                     # Full integer indexing (item extraction)
    (slice(None), slice(1, 2), 0), # Mix of colon, slice, and integer
    (slice(0, 4, 2), slice(None)), # Slicing with steps (if supported)
])
def test_slicing_forward_backward(complex_tensor, slices):
    a_dll, a_t = complex_tensor
    a_dll.requires_grad = True

    res_dll = a_dll[slices]
    res_t = a_t[slices]

    assert res_dll.shape == list(res_t.shape), f"Shape mismatch: DLL={res_dll.shape}, Torch={list(res_t.shape)}"
    
    assert res_dll.data == pytest.approx(res_t.flatten().tolist(), rel=rtol, abs=atol)

    res_dll.sum().backward()
    res_t.sum().backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_slicing_assignment_parity():
    shape = (5, 5)
    a_dll = rand(shape)
    a_t = sync_tensors(a_dll)
    a_dll.requires_grad = True

    sub_dll = a_dll[1:4, 1:4] * 2.0
    sub_t = a_t[1:4, 1:4] * 2.0

    L_dll = sub_dll.sum()
    L_t = sub_t.sum()
    
    L_dll.backward()
    L_t.backward()

    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_transpose_and_slice():
    a_dll = rand((4, 4))
    a_t = sync_tensors(a_dll)
    a_dll.requires_grad = True

    res_dll = a_dll.transpose()[0:2, :]
    res_t = a_t.t()[0:2, :]

    assert res_dll.shape == list(res_t.shape)
    assert res_dll.data == pytest.approx(res_t.flatten().tolist(), rel=rtol, abs=atol)

    res_dll.sum().backward()
    res_t.sum().backward()
    
    assert a_dll.grad.data == pytest.approx(a_t.grad.flatten().tolist(), rel=rtol, abs=atol)

def test_negative_indexing():
    a_dll = rand((5, 5))
    a_t = sync_tensors(a_dll)
    
    res_dll = a_dll[-1, 2:-1]
    res_t = a_t[-1, 2:-1]
    
    assert res_dll.shape == list(res_t.shape)
    assert res_dll.data == pytest.approx(res_t.flatten().tolist())
