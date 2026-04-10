import DLL
from DLL import rand
import numpy as np
import torch
import pytest

rtol = 1e-3
atol = 1e-4

def sync_tensors(dll_tensor):
    t = torch.tensor(dll_tensor.data, dtype=torch.float32).reshape(dll_tensor.shape)
    t.requires_grad = True
    return t

def test_top_level_exports():
    assert hasattr(DLL, "matmul")
    assert hasattr(DLL, "transpose")
    assert hasattr(DLL, "dot")
    assert hasattr(DLL, "vdot")

def test_matmul():
    a = rand([3, 4])
    b = rand([4, 5])
    a.requires_grad = True
    b.requires_grad = True
    
    a_t = sync_tensors(a)
    b_t = sync_tensors(b)
    
    c = DLL.matmul(a, b)
    c_t = torch.matmul(a_t, b_t)
    
    assert np.allclose(c.data, c_t.detach().flatten().numpy(), atol=1e-3)
    
    c.sum().backward()
    c_t.sum().backward()
    
    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3)
    assert np.allclose(b.grad.data, b_t.grad.flatten().numpy(), atol=1e-3)

def test_transpose():
    a = rand([3, 4])
    a.requires_grad = True
    a_t = sync_tensors(a)
    
    b = DLL.transpose(a)
    b_t = a_t.transpose(-2, -1)
    
    assert np.allclose(b.data, b_t.detach().flatten().numpy(), atol=1e-3)
    
    b.sum().backward()
    b_t.sum().backward()
    
    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3)

def test_norm():
    a = rand([4, 5]) + 1.0
    a.requires_grad = True
    a_t = sync_tensors(a)
    
    a3 = rand([3, 4, 5]) + 1.0
    a3.requires_grad = True
    a3_t = sync_tensors(a3)
    
    test_cases_2d = [
        (None, None, False),
        (1, None, False),
        (float('inf'), None, False),
        (-float('inf'), None, False),
        ('fro', None, False),
    ]
    
    for ord, dim, keepdim in test_cases_2d:
        a.zero_grad()
        a_t.grad = None
        
        n = DLL.linalg.norm(a, ord=ord, dim=dim, keepdim=keepdim)
        n_t = torch.linalg.norm(a_t, ord=ord, dim=dim, keepdim=keepdim)
        
        assert np.allclose(n.data, n_t.detach().flatten().numpy(), atol=1e-3), f"Forward failed 2D ord={ord} dim={dim} keepdim={keepdim}"
        
        n.sum().backward()
        n_t.sum().backward()
        
        assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3), f"Backward failed 2D ord={ord} dim={dim} keepdim={keepdim}\nDLL:{a.grad.data}\nPyTorch:{a_t.grad.flatten().numpy()}"

    test_cases_3d = [
        # Vector norms
        (2, 1, False),
        (2, -1, True),
        (1, 0, False),
        (float('inf'), 2, False),
        (-float('inf'), 1, False),
        (3, 0, True),
        
        # Matrix norms
        ('fro', (1, 2), False),
        (1, (0, 2), False),
        (float('inf'), (1, 0), True),
        (-1, (0, 1), False),
        (-float('inf'), (2, 1), True),
    ]
    
    for ord, dim, keepdim in test_cases_3d:
        a3.zero_grad()
        a3_t.grad = None
        
        n = DLL.linalg.norm(a3, ord=ord, dim=dim, keepdim=keepdim)
        n_t = torch.linalg.norm(a3_t, ord=ord, dim=dim, keepdim=keepdim)
        
        assert np.allclose(n.data, n_t.detach().flatten().numpy(), atol=1e-3), f"Forward failed 3D ord={ord} dim={dim} keepdim={keepdim}"
        
        n.sum().backward()
        n_t.sum().backward()
        
        assert np.allclose(a3.grad.data, a3_t.grad.flatten().numpy(), atol=1e-3), f"Backward failed 3D ord={ord} dim={dim} keepdim={keepdim}\nDLL:{a3.grad.data}\nPyTorch:{a3_t.grad.flatten().numpy()}"

def test_dot():
    a = rand([3])
    b = rand([3])
    a.requires_grad = True
    b.requires_grad = True
    a_t = sync_tensors(a)
    b_t = sync_tensors(b)
    
    d = DLL.dot(a, b)
    d_t = torch.dot(a_t, b_t)
    
    assert np.isclose(d.item(), d_t.item(), atol=1e-3)
    
    d.sum().backward()
    d_t.backward()
    
    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3)
    assert np.allclose(b.grad.data, b_t.grad.flatten().numpy(), atol=1e-3)

def test_vdot():
    a = rand([3])
    b = rand([3])
    a.requires_grad = True
    b.requires_grad = True
    a_t = sync_tensors(a)
    b_t = sync_tensors(b)
    
    d = DLL.linalg.vdot(a, b)
    d_t = torch.vdot(a_t, b_t)
    
    assert np.isclose(d.item(), d_t.item(), atol=1e-3)
    
    d.sum().backward()
    d_t.backward()
    
    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3)
    assert np.allclose(b.grad.data, b_t.grad.flatten().numpy(), atol=1e-3)

def test_inv():
    n = 3
    a_np = np.random.randn(n, n).astype(np.float32)
    a_np += np.eye(n, dtype=np.float32) * n
    
    a = DLL.Tensor(a_np.tolist())
    a.requires_grad = True
    a_t = sync_tensors(a)
    
    a_inv = DLL.linalg.inv(a)
    a_inv_t = torch.linalg.inv(a_t)
    
    assert np.allclose(a_inv.data, a_inv_t.detach().flatten().numpy(), atol=1e-3)
    
    a_inv.sum().backward()
    a_inv_t.sum().backward()
    
    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3)

def test_det():
    n = 3
    a_np = np.random.randn(n, n).astype(np.float32)
    a_np += np.eye(n, dtype=np.float32) * n
    
    a = DLL.Tensor(a_np.tolist())
    a.requires_grad = True
    a_t = sync_tensors(a)
    
    d = DLL.linalg.det(a)
    d_t = torch.linalg.det(a_t)
    
    assert np.isclose(d.item(), d_t.item(), atol=1e-3)
    
    d.sum().backward()
    d_t.backward()
    

def test_svd():
    a = rand([4, 4]) + 1.0
    a.requires_grad = True
    a_t = sync_tensors(a)
    
    U, S, Vt = DLL.linalg.svd(a)
    U_t, S_t, Vt_t = torch.linalg.svd(a_t, full_matrices=False)
    
    # Check forward
    # Note: U and Vt signs can be flipped compared to PyTorch, but S should be identical.
    assert np.allclose(S.data, S_t.detach().flatten().numpy(), atol=1e-3)
    
    # Reconstruct A = U S V^T and verify
    # A_rec = U @ diag(S) @ Vt
    # We can just do matmul manually
    S_mat = DLL.Tensor(np.diag(S.data).flatten().tolist(), [4, 4])
    A_rec = U @ S_mat @ Vt
    assert np.allclose(A_rec.data, a.data, atol=1e-3)
    
    # Check backward pass
    a.zero_grad()
    a_t.grad = None
    
    loss = (U.abs() * S).sum() + (S**3).sum() + (Vt.abs() * S).sum()
    loss_t = (U_t.abs() * S_t).sum() + (S_t**3).sum() + (Vt_t.abs() * S_t).sum()
    
    loss.backward()
    loss_t.backward()
    
    # The gradients should match perfectly since we regularized the matrix
    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3)

    # Test full_matrices with non-square matrices
    for shape in ([5, 3], [3, 5]):
        M, N = shape
        K = min(M, N)
        b = rand(shape) + 1.0
        b_t = sync_tensors(b)

        # full_matrices=True
        U_f, S_f, Vt_f = DLL.linalg.svd(b, full_matrices=True)
        U_ft, S_ft, Vt_ft = torch.linalg.svd(b_t, full_matrices=True)
        assert U_f.shape == [M, M], f"U full shape mismatch: {U_f.shape} vs [{M}, {M}]"
        assert Vt_f.shape == [N, N], f"Vt full shape mismatch: {Vt_f.shape} vs [{N}, {N}]"
        assert np.allclose(S_f.data, S_ft.detach().numpy(), atol=1e-3)

        # full_matrices=False
        U_r, S_r, Vt_r = DLL.linalg.svd(b, full_matrices=False)
        U_rt, S_rt, Vt_rt = torch.linalg.svd(b_t, full_matrices=False)
        assert U_r.shape == [M, K], f"U reduced shape mismatch: {U_r.shape} vs [{M}, {K}]"
        assert Vt_r.shape == [K, N], f"Vt reduced shape mismatch: {Vt_r.shape} vs [{K}, {N}]"
        assert np.allclose(S_r.data, S_rt.detach().numpy(), atol=1e-3)

        # Reconstruction with reduced SVD: A ≈ U @ diag(S) @ Vt
        S_diag = DLL.diag(S_r)
        A_rec = U_r @ S_diag @ Vt_r
        assert np.allclose(A_rec.data, b.data, atol=1e-3), f"Reconstruction failed for shape {shape}"

def test_eig():
    # Create a symmetric matrix with distinct eigenvalues
    a_np = np.random.randn(4, 4).astype(np.float32)
    a_np = (a_np + a_np.T) / 2.0
    a_np += np.eye(4) * 4  # make well-conditioned

    a = DLL.Tensor(a_np.flatten().tolist(), [4, 4])
    a.requires_grad = True
    a_t = sync_tensors(a)

    L, V = DLL.linalg.eig(a)
    L_t, V_t = torch.linalg.eigh(a_t)
    
    # Forward: eigenvalues should match
    assert np.allclose(L.data, L_t.detach().numpy(), atol=1e-3), "Eigenvalues mismatch"

    # Reconstruction: A = V diag(L) V^T
    L_diag = DLL.diag(L)
    A_rec = V @ L_diag @ V.transpose(0, 1)
    assert np.allclose(A_rec.data, a.data, atol=1e-3), "Reconstruction failed"

    # Backward: use sign-invariant loss (V**2 is invariant to column sign flips)
    a.zero_grad()
    a_t.grad = None

    loss = (V**2 * L).sum() + (L**3).sum()
    loss_t = (V_t**2 * L_t).sum() + (L_t**3).sum()

    loss.backward()
    loss_t.backward()

    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3), \
        f"Eig gradient mismatch: max diff = {np.abs(np.array(a.grad.data) - a_t.grad.flatten().numpy()).max()}"

def test_lu():
    a_np = np.random.randn(4, 4).astype(np.float32)

    a = DLL.Tensor(a_np.flatten().tolist(), [4, 4])
    a.requires_grad = True
    a_t = sync_tensors(a)

    P, L, U = DLL.linalg.lu(a)
    P_t, L_t, U_t = torch.linalg.lu(a_t)

    # Reconstruction: A = P @ L @ U
    A_rec = P @ L @ U
    assert np.allclose(A_rec.data, a.data, atol=1e-3), "LU reconstruction failed"

    # L is lower triangular with unit diagonal
    L_np = np.array(L.data).reshape(4, 4)
    assert np.allclose(np.diag(L_np), 1.0), "L diagonal not unit"
    assert np.allclose(np.triu(L_np, 1), 0.0), "L upper triangle not zero"

    # U is upper triangular
    U_np = np.array(U.data).reshape(4, 4)
    assert np.allclose(np.tril(U_np, -1), 0.0), "U lower triangle not zero"

    # P is a permutation matrix
    P_np = np.array(P.data).reshape(4, 4)
    assert np.allclose(P_np.sum(axis=0), 1.0), "P columns don't sum to 1"
    assert np.allclose(P_np.sum(axis=1), 1.0), "P rows don't sum to 1"

    # Backward through L and U
    a.zero_grad()
    a_t.grad = None

    loss = (L**2).sum() + (U**3).sum()
    loss_t = (L_t**2).sum() + (U_t**3).sum()

    loss.backward()
    loss_t.backward()

    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3), \
        f"LU gradient mismatch: max diff = {np.abs(np.array(a.grad.data) - a_t.grad.flatten().numpy()).max()}"

def test_qr():
    # Square matrix
    a_np = np.random.randn(4, 4).astype(np.float32)
    a = DLL.Tensor(a_np.flatten().tolist(), [4, 4])
    a.requires_grad = True
    a_t = sync_tensors(a)

    Q, R = DLL.linalg.qr(a)
    Q_t, R_t = torch.linalg.qr(a_t)

    # Forward: reconstruction
    A_rec = Q @ R
    assert np.allclose(A_rec.data, a.data, atol=1e-3), "QR reconstruction failed (square)"

    # Q^T Q = I
    QtQ = np.array((Q.transpose(0, 1) @ Q).data).reshape(4, 4)
    assert np.allclose(QtQ, np.eye(4), atol=1e-3), "Q not orthogonal (square)"

    # R is upper triangular
    R_np = np.array(R.data).reshape(4, 4)
    assert np.allclose(np.tril(R_np, -1), 0.0, atol=1e-3), "R not upper triangular"

    # Backward
    loss = (Q**2).sum() + (R**2).sum()
    loss_t = (Q_t**2).sum() + (R_t**2).sum()
    loss.backward()
    loss_t.backward()

    assert np.allclose(a.grad.data, a_t.grad.flatten().numpy(), atol=1e-3), \
        f"QR gradient mismatch (square): max diff = {np.abs(np.array(a.grad.data) - a_t.grad.flatten().numpy()).max()}"

    # Tall matrix (M > N)
    b_np = np.random.randn(6, 3).astype(np.float32)
    b = DLL.Tensor(b_np.flatten().tolist(), [6, 3])
    b.requires_grad = True
    b_t = sync_tensors(b)

    Q2, R2 = DLL.linalg.qr(b)
    Q2_t, R2_t = torch.linalg.qr(b_t)

    assert Q2.shape == [6, 3] and R2.shape == [3, 3], f"Wrong shapes: Q={Q2.shape}, R={R2.shape}"

    B_rec = Q2 @ R2
    assert np.allclose(B_rec.data, b.data, atol=1e-3), "QR reconstruction failed (tall)"

    loss2 = (Q2**2).sum() + (R2**3).sum()
    loss2_t = (Q2_t**2).sum() + (R2_t**3).sum()
    loss2.backward()
    loss2_t.backward()

    assert np.allclose(b.grad.data, b_t.grad.flatten().numpy(), atol=1e-3), \
        f"QR gradient mismatch (tall): max diff = {np.abs(np.array(b.grad.data) - b_t.grad.flatten().numpy()).max()}"

def test_solve():
    n = 3
    A_np = np.random.randn(n, n).astype(np.float32)
    A_np += np.eye(n, dtype=np.float32) * n
    b_np = np.random.randn(n, 1).astype(np.float32)
    
    A = DLL.Tensor(A_np.tolist())
    A.requires_grad = True
    b = DLL.Tensor(b_np.tolist())
    b.requires_grad = True
    
    A_t = sync_tensors(A)
    b_t = sync_tensors(b)
    
    x = DLL.linalg.solve(A, b)
    x_t = torch.linalg.solve(A_t, b_t)
    
    assert np.allclose(x.data, x_t.detach().flatten().numpy(), atol=1e-3)
    
    x.sum().backward()
    x_t.sum().backward()
    
    assert np.allclose(b.grad.data, b_t.grad.flatten().numpy(), atol=1e-3)
    assert np.allclose(A.grad.data, A_t.grad.flatten().numpy(), atol=1e-3)

def test_cholesky():
    # Symmetric positive-definite matrix
    n = 3
    A_np = np.random.randn(n, n).astype(np.float32)
    A_np = A_np @ A_np.T + np.eye(n, dtype=np.float32) * 1e-3
    
    A = DLL.Tensor(A_np.tolist())
    A.requires_grad = True
    A_t = sync_tensors(A)
    
    L = DLL.linalg.cholesky(A)
    L_t = torch.linalg.cholesky(A_t)
    
    assert np.allclose(L.data, L_t.detach().flatten().numpy(), atol=1e-3)
    
    L.sum().backward()
    L_t.sum().backward()
    
    assert np.allclose(A.grad.data, A_t.grad.flatten().numpy(), atol=1e-3)
