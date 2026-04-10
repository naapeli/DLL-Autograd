import DLL

from time import perf_counter

from tqdm import tqdm
import torch


print(DLL.gpu_available())

n = 10000
K = 10
x = DLL.randn((n, n)).to("gpu")
y = DLL.randn((n, n)).to("gpu")


start = perf_counter()
for _ in tqdm(range(K)):
    z = (x @ y).sum()
    z.backward()
print(f"Time: {perf_counter() - start}")

x = x.cpu()
y = y.cpu()

start = perf_counter()
for _ in tqdm(range(K)):
    z = (x @ y).sum()
    z.backward()
print(f"Time: {perf_counter() - start}")

x = torch.tensor(x.data, requires_grad=True).reshape((n, n))
y = torch.tensor(y.data, requires_grad=True).reshape((n, n))

start = perf_counter()
for _ in tqdm(range(K)):
    z = (x @ y).sum()
    z.backward()
print(f"Time: {perf_counter() - start}")
