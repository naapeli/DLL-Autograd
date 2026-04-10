from DLL import randn

from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm


n = 10000  # 1000
k = 10  # 1000
backprop = False
x = randn((n, n))
y = randn((n, n))
if backprop:
    x.requires_grad = True
    y.requires_grad = True

start = perf_counter()
for i in tqdm(range(k)):
    z = (x @ y).sum()
    if backprop: z.backward()
print(f"Time:  {perf_counter() - start}")


x = np.array(x.data).reshape((n, n))
y = np.array(y.data).reshape((n, n))

start = perf_counter()
for _ in tqdm(range(k)):
    z = (x @ y).sum()
print(f"Time:  {perf_counter() - start}")


x = torch.tensor(x)
y = torch.tensor(y)
x.requires_grad = True
y.requires_grad = True

start = perf_counter()
for i in tqdm(range(k)):
    z = (x @ y).sum()
    if backprop: z.backward()
print(f"Time:  {perf_counter() - start}")
