from math import prod
from typing import Any, Optional, Union, List

from ._C import Tensor, rand, randn, gpu_available, gpu_device_name, cat, stack
from . import Random, linalg
from .constants import zeros, ones, eye
from .linalg import matmul, transpose, dot, vdot, diag


__all__ = [
    "Tensor", 
    "zeros", 
    "ones", 
    "eye", 
    "rand", 
    "randn", 
    "Random", 
    "gpu_available", 
    "gpu_device_name", 
    "linalg",
    "matmul",
    "transpose",
    "dot",
    "vdot",
    "diag",
    "cat",
    "stack",
]
