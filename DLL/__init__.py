from math import prod, sqrt
from random import uniform, gauss

import os
import sys
from pathlib import Path

def _setup_dll_env():
    if sys.platform != "win32":
        return

    pkg_root = Path(__file__).parent
    lib_dir = pkg_root / "lib"

    if lib_dir.exists():
        os.add_dll_directory(str(lib_dir))

_setup_dll_env()


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

def rand(shape, min=0, max=1):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([uniform(min, max) for _ in range(n_elements)], shape)

def randn(shape, mu=0, var=1):
    shape = shape if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([gauss(mu, sqrt(var)) for _ in range(n_elements)], shape)
