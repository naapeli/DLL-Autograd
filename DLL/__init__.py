from math import prod

from ._c_tensor import Tensor


__all__ = ["Tensor", "zeros", "ones"]

def zeros(shape):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([0] * n_elements, shape)

def ones(shape):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([1] * n_elements, shape)
