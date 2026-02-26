import DLL
import torch
import numpy as np
from time import perf_counter


shape = (2048, 2048)  # (4096, 4096)  # (256, 512)
shapeT = (shape[1], shape[0])
rtol, atol = 1e-3, 1e-3

a_dll = DLL.randn(shape)
b_dll = DLL.randn(shapeT)
a_dll.requires_grad = True
b_dll.requires_grad = True

a_torch = torch.tensor(a_dll.data, dtype=torch.float32).reshape(shape)
b_torch = torch.tensor(b_dll.data, dtype=torch.float32).reshape(shapeT)
a_torch.requires_grad = True
b_torch.requires_grad = True

print("Computation started")

start_dll = perf_counter()
c_dll = a_dll @ b_dll
c_dll.sum().backward()
stop_dll = perf_counter()

print("computation ended")

start_torch = perf_counter()
c_torch = a_torch @ b_torch
c_torch.sum().backward()
stop_torch = perf_counter()

dll_out = np.array(c_dll.data).reshape(c_dll.shape)
torch_out = c_torch.detach().numpy()
forward_passed = np.allclose(dll_out, torch_out, rtol=rtol, atol=atol)

dll_grad_a = np.array(a_dll.grad.data).reshape(a_dll.shape)
torch_grad_a = a_torch.grad.numpy()
grad_a_passed = np.allclose(dll_grad_a, torch_grad_a, rtol=rtol, atol=atol)

dll_grad_b = np.array(b_dll.grad.data).reshape(b_dll.shape)
torch_grad_b = b_torch.grad.numpy()
grad_b_passed = np.allclose(dll_grad_b, torch_grad_b, rtol=rtol, atol=atol)

print(f"DLL Time:   {stop_dll - start_dll:.4f}s")
print(f"Torch Time: {stop_torch - start_torch:.4f}s")
print("-" * 30)
print(f"Forward Match:  {forward_passed}")
print(f"Grad A Match:   {grad_a_passed}")
print(f"Grad B Match:   {grad_b_passed}")

if not forward_passed:
    print(f"Max Diff Forward: {np.max(np.abs(dll_out - torch_out))}")
