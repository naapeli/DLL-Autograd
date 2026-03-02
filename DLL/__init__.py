from math import prod

from ._C import Tensor, rand, randn
from . import Random


__all__ = ["Tensor", "zeros", "ones", "rand", "randn", "Random"]

def zeros(shape):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([0] * n_elements, shape)

def ones(shape):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([1] * n_elements, shape)
