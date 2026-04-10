from typing import Any, Optional, Union, Tuple
from .. import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the matrix product of two tensors.
    """
    return a @ b

def transpose(a: Tensor, dim0: int = -2, dim1: int = -1) -> Tensor:
    """
    Returns a tensor that is a transposed version of the input.
    """
    return a.transpose(dim0, dim1)

def norm(a: Tensor, ord: Union[int, float, str, None] = None, dim: Optional[Union[int, Tuple[int, int]]] = None, keepdim: bool = False) -> Tensor:
    """
    Computes the matrix or vector norm.
    """
    if ord is None:
        ord = 'fro' if (dim is None and len(a.shape) > 1) or (isinstance(dim, tuple) and len(dim) == 2) else 2

    if dim is None:
        if ord is None or ord == 'fro' or (ord == 2 and len(a.shape) == 1):
            return (a**2.0).sum(keepdim=keepdim).sqrt()
        if len(a.shape) == 1:
            dim = 0
        elif len(a.shape) == 2:
            dim = (0, 1)
        else:
            raise ValueError("linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D")

    if isinstance(dim, int):
        # Vector norm over a specific dimension
        if ord == 2:
            return (a**2.0).sum(dim=dim, keepdim=keepdim).sqrt()
        elif ord == 1:
            return a.abs().sum(dim=dim, keepdim=keepdim)
        elif ord == float('inf'):
            return a.abs().max(dim=dim, keepdim=keepdim)
        elif ord == -float('inf'):
            return a.abs().min(dim=dim, keepdim=keepdim)
        elif isinstance(ord, (int, float)):
            return (a.abs()**float(ord)).sum(dim=dim, keepdim=keepdim)**(1.0/float(ord))
        else:
            raise ValueError(f"Invalid norm order {ord} for vector norm")

    elif isinstance(dim, tuple) and len(dim) == 2:
        # Matrix norm over two specific dimensions
        if ord == 'fro':
            d_first, d_second = sorted(dim, reverse=True)
            res = (a**2.0).sum(dim=d_first, keepdim=keepdim)
            res = res.sum(dim=d_second, keepdim=keepdim)
            return res.sqrt()
        elif ord == 1:
            # max of column sum
            res = a.abs().sum(dim=dim[0], keepdim=keepdim)
            d_max = dim[1] if keepdim or dim[1] < dim[0] else dim[1] - 1
            return res.max(dim=d_max, keepdim=keepdim)
        elif ord == float('inf'):
            # max of row sum
            res = a.abs().sum(dim=dim[1], keepdim=keepdim)
            d_max = dim[0] if keepdim or dim[0] < dim[1] else dim[0] - 1
            return res.max(dim=d_max, keepdim=keepdim)
        elif ord == -1:
            # min of column sum
            res = a.abs().sum(dim=dim[0], keepdim=keepdim)
            d_max = dim[1] if keepdim or dim[1] < dim[0] else dim[1] - 1
            return res.min(dim=d_max, keepdim=keepdim)
        elif ord == -float('inf'):
            # min of row sum
            res = a.abs().sum(dim=dim[1], keepdim=keepdim)
            d_max = dim[0] if keepdim or dim[0] < dim[1] else dim[0] - 1
            return res.min(dim=d_max, keepdim=keepdim)
        elif ord == 2:
            return svd(a)[1].max(keepdim=keepdim)
        elif ord == -2:
            return svd(a)[1].min(keepdim=keepdim)
        elif ord == 'nuc':
            return svd(a)[1].sum(keepdim=keepdim)
        else:
            raise ValueError(f"Invalid norm order {ord} for matrix norm")
    
    raise ValueError("Invalid dim argument")

def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot product of two tensors.
    """
    return (a * b).sum()

def vdot(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot product of two vectors.
    """
    return (a * b).sum()

def inv(a: Tensor) -> Tensor:
    """Computes the inverse of a square matrix."""
    return a.inv()

def det(a: Tensor) -> Tensor:
    """Computes the determinant of a square matrix."""
    return a.det()

def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solves a linear system of equations AX = B."""
    return A.solve(b)

def cholesky(a: Tensor) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric positive-definite matrix."""
    return a.cholesky()

def svd(a: Tensor, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the Singular Value Decomposition."""
    return a.svd(full_matrices=full_matrices)

def lu(a: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes the LU decomposition with partial pivoting.
    Returns (P, L, U) such that A = P @ L @ U.
    L is lower triangular with unit diagonal, U is upper triangular.
    P is a permutation matrix (not differentiable). L and U support autograd.
    """
    return a.lu()

def qr(a: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes the reduced QR decomposition.
    Returns (Q, R) such that A = Q @ R.
    Q has orthonormal columns (M×K), R is upper triangular (K×N), where K = min(M, N).
    """
    return a.qr()

def eig(a: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes eigenvalues and eigenvectors of a symmetric matrix.
    Returns (eigenvalues, eigenvectors) where eigenvectors are columns of V.
    The input matrix must be symmetric. Eigenvalues are returned in ascending order.
    """
    return a.eig()

def diag(a: Tensor, diagonal: int = 0) -> Tensor:
    """
    If input is a 1D tensor, returns a 2D tensor with the elements on the diagonal.
    If input is a 2D tensor, returns the diagonal elements as a 1D tensor.
    """
    return a.diag(diagonal=diagonal)

def trace(a: Tensor) -> Tensor:
    """Computes the sum of the elements on the main diagonal of a matrix."""
    return diag(a).sum(dim=-1)

def outer(a: Tensor, b: Tensor) -> Tensor:
    """Computes the outer product of two vectors."""
    if len(a.shape) != 1 or len(b.shape) != 1:
        # Fallback to general broadcast if not vectors? 
        # PyTorch outer is strictly for vectors.
        pass
    return a.unsqueeze(-1) * b.unsqueeze(-2)

def kron(a: Tensor, b: Tensor) -> Tensor:
    """Computes the Kronecker product of two matrices."""
    a_shape = a.shape
    b_shape = b.shape
    # (..., M, N) (..., P, Q) -> (..., M*P, N*Q)
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    out_shape = list(a_shape[:-2]) + [a_shape[-2] * b_shape[-2], a_shape[-1] * b_shape[-1]]
    return res.reshape(out_shape)

def matrix_power(a: Tensor, n: int) -> Tensor:
    """Computes the n-th power of a square matrix (for integer n)."""
    if not isinstance(n, int):
        raise ValueError("matrix_power only supports integer exponents n.")
    if n == 0:
        # Identity
        from ..constants import eye
        I = eye(a.shape[-1], a.shape[-1]).to(a.device)
        # Broadcast to batch shape if needed
        if len(a.shape) > 2:
            ones_batch = (a * 0.0 + 1.0).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True) # Trick to get batch shape
            return I * ones_batch
        return I
    if n < 0:
        return matrix_power(a.inv(), -n)
    
    res = None
    curr = a
    while n > 0:
        if n % 2 == 1:
            res = curr if res is None else res @ curr
        if n > 1:
            curr = curr @ curr
        n //= 2
    return res

def cond(a: Tensor, ord: Optional[Union[str, int, float]] = None) -> Tensor:
    """Computes the condition number of a matrix."""
    if ord is None or ord == 2:
        s = svd(a)[1]
        return s.max(dim=-1) / s.min(dim=-1)
    return norm(a, ord=ord) * norm(a.inv(), ord=ord)

def matrix_exp(a: Tensor) -> Tensor:
    """Computes the matrix exponential of a square matrix."""
    return a.matrix_exp()

def lu_factor(a: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes the LU factorization of a matrix."""
    return a.lu_factor()

def lu_solve(b: Tensor, LU: Tensor, pivots: Tensor, adjoint: bool = False) -> Tensor:
    """Solves a linear system AX = B given LU factorization."""
    return b.lu_solve(LU, pivots, adjoint=adjoint)

def cholesky_solve(b: Tensor, L: Tensor) -> Tensor:
    """Solves a linear system AX = B given Cholesky factor L."""
    return b.cholesky_solve(L)

def lstsq(A: Tensor, B: Tensor) -> Tensor:
    """Computes the least-squares solution to AX = B."""
    return A.lstsq(B)

# Aliases
chol = cholesky
