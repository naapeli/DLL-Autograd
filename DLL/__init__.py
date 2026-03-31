from math import prod

from ._C import Tensor, rand, randn, gpu_available, gpu_device_name
from . import Random


__all__ = ["Tensor", "zeros", "ones", "rand", "randn", "Random", "gpu_available", "gpu_device_name"]

def zeros(shape):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([0] * n_elements, shape)

def ones(shape):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([1] * n_elements, shape)
