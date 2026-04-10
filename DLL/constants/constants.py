from math import prod
from typing import Union, List, Tuple

from .. import Tensor


def zeros(shape: Union[int, List[int], Tuple[int, ...]]) -> Tensor:
    """
    Creates a tensor filled with zeros.
    
    Args:
        shape (int | list[int] | tuple[int, ...]): The shape of the tensor.
        
    Returns:
        Tensor: A tensor of zeros with the specified shape.
    """
    shape = list(shape) if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([0.0] * n_elements, shape)

def ones(shape: Union[int, List[int], Tuple[int, ...]]) -> Tensor:
    """
    Creates a tensor filled with ones.
    
    Args:
        shape (int | list[int] | tuple[int, ...]): The shape of the tensor.
        
    Returns:
        Tensor: A tensor of ones with the specified shape.
    """
    shape = list(shape) if isinstance(shape, list | tuple) else [shape]
    n_elements = prod(shape)
    return Tensor([1.0] * n_elements, shape)

def eye(n: int) -> Tensor:
    """
    Creates an identity matrix of size n x n.
    
    Args:
        n (int): The number of rows and columns.
        
    Returns:
        Tensor: An n x n identity matrix.
    """
    return ones(n).diag()
