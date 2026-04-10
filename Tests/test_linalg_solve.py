import torch
import numpy as np
import DLL
import unittest


class TestLinalgSolve(unittest.TestCase):
    def test_lu_solve(self):
        A_np = np.random.randn(3, 3).astype(np.float32)
        B_np = np.random.randn(3, 2).astype(np.float32)
        
        A_dll = DLL.Tensor(A_np.flatten().tolist(), [3, 3])
        B_dll = DLL.Tensor(B_np.flatten().tolist(), [3, 2])
        A_dll.requires_grad = True
        B_dll.requires_grad = True
        
        LU_dll, piv_dll = DLL.linalg.lu_factor(A_dll)
        X_dll = DLL.linalg.lu_solve(B_dll, LU_dll, piv_dll)
        
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        LU_torch, piv_torch = torch.linalg.lu_factor(A_torch)
        X_torch = torch.linalg.lu_solve(LU_torch, piv_torch, B_torch)
        
        if not np.allclose(np.array(X_dll.data).reshape(3, 2), X_torch.detach().numpy(), atol=1e-5):
            print("X_dll:", np.array(X_dll.data).reshape(3, 2))
            print("X_torch:", X_torch.detach().numpy())
            self.assertTrue(False)
        
        # Backward
        grad_out = np.random.randn(3, 2).astype(np.float32)
        X_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [3, 2]))
        X_torch.backward(torch.tensor(grad_out))
        
        if not np.allclose(B_dll.grad.data, B_torch.grad.flatten().numpy(), atol=1e-4):
            print("B_grad_dll:", np.array(B_dll.grad.data).reshape(3, 2))
            print("B_grad_torch:", B_torch.grad.numpy())
            self.assertTrue(False)
        if not np.allclose(A_dll.grad.data, A_torch.grad.flatten().numpy(), atol=1e-4):
            print("A_grad_dll:", np.array(A_dll.grad.shape))
            print("A_grad_dll val:", np.array(A_dll.grad.data).reshape(3, 3))
            print("A_grad_torch val:", A_torch.grad.numpy())
            self.assertTrue(False)

    def test_cholesky_solve(self):
        # SPD matrix
        R = np.random.randn(3, 3).astype(np.float32)
        A_np = (R @ R.T).astype(np.float32) + 5.0 * np.eye(3).astype(np.float32)
        B_np = np.random.randn(3, 1).astype(np.float32)
        
        A_dll = DLL.Tensor(A_np.flatten().tolist(), [3, 3])
        B_dll = DLL.Tensor(B_np.flatten().tolist(), [3, 1])
        A_dll.requires_grad = True
        B_dll.requires_grad = True
        
        L_dll = DLL.linalg.cholesky(A_dll)
        X_chol_dll = DLL.linalg.cholesky_solve(B_dll, L_dll)
        
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        L_torch = torch.linalg.cholesky(A_torch)
        # torch.cholesky_solve takes (B, L)
        X_torch = torch.cholesky_solve(B_torch, L_torch, upper=False)
        
        self.assertTrue(np.allclose(np.array(X_chol_dll.data).reshape(3, 1), X_torch.detach().numpy(), atol=1e-4))
        
        # Backward
        grad_out = np.random.randn(3, 1).astype(np.float32)
        X_chol_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [3, 1]))
        X_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(B_dll.grad.data, B_torch.grad.flatten().numpy(), atol=1e-3))
        self.assertTrue(np.allclose(A_dll.grad.data, A_torch.grad.flatten().numpy(), atol=1e-3))

    def test_lstsq(self):
        A_np = np.random.randn(5, 3).astype(np.float32)
        B_np = np.random.randn(5, 2).astype(np.float32)
        
        A_dll = DLL.Tensor(A_np.flatten().tolist(), [5, 3])
        B_dll = DLL.Tensor(B_np.flatten().tolist(), [5, 2])
        A_dll.requires_grad = True
        B_dll.requires_grad = True
        
        X_dll = DLL.linalg.lstsq(A_dll, B_dll)
        
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        X_torch = torch.linalg.lstsq(A_torch, B_torch).solution
        
        self.assertTrue(np.allclose(np.array(X_dll.data).reshape(3, 2), X_torch.detach().numpy(), atol=1e-3))
        
        # Backward
        grad_out = np.random.randn(3, 2).astype(np.float32)
        X_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [3, 2]))
        X_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(B_dll.grad.data, B_torch.grad.flatten().numpy(), atol=1e-2))
        self.assertTrue(np.allclose(A_dll.grad.data, A_torch.grad.flatten().numpy(), atol=1e-2))

    def test_lu_solve_batched(self):
        A_np = np.random.randn(2, 3, 3).astype(np.float32)
        B_np = np.random.randn(2, 3, 2).astype(np.float32)
        
        A_dll = DLL.Tensor(A_np.flatten().tolist(), [2, 3, 3])
        B_dll = DLL.Tensor(B_np.flatten().tolist(), [2, 3, 2])
        A_dll.requires_grad = True
        B_dll.requires_grad = True
        
        LU_dll, piv_dll = DLL.linalg.lu_factor(A_dll)
        X_dll = DLL.linalg.lu_solve(B_dll, LU_dll, piv_dll)
        
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        LU_torch, piv_torch = torch.linalg.lu_factor(A_torch)
        X_torch = torch.linalg.lu_solve(LU_torch, piv_torch, B_torch)
        
        self.assertTrue(np.allclose(np.array(X_dll.data).reshape(2, 3, 2), X_torch.detach().numpy(), atol=1e-4))
        
        # Backward
        grad_out = np.random.randn(2, 3, 2).astype(np.float32)
        X_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [2, 3, 2]))
        X_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(B_dll.grad.data, B_torch.grad.flatten().numpy(), atol=1e-3))
        self.assertTrue(np.allclose(A_dll.grad.data, A_torch.grad.flatten().numpy(), atol=1e-3))

    def test_cholesky_solve_batched(self):
        # 2 batches of 3x3 SPD matrices
        R = np.random.randn(2, 3, 3).astype(np.float32)
        A_np = np.zeros((2, 3, 3), dtype=np.float32)
        for b in range(2):
            A_np[b] = R[b] @ R[b].T + 5.0 * np.eye(3).astype(np.float32)
        B_np = np.random.randn(2, 3, 1).astype(np.float32)
        
        A_dll = DLL.Tensor(A_np.flatten().tolist(), [2, 3, 3])
        B_dll = DLL.Tensor(B_np.flatten().tolist(), [2, 3, 1])
        A_dll.requires_grad = True
        B_dll.requires_grad = True
        
        L_dll = DLL.linalg.cholesky(A_dll)
        X_chol_dll = DLL.linalg.cholesky_solve(B_dll, L_dll)
        
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        L_torch = torch.linalg.cholesky(A_torch)
        X_torch = torch.cholesky_solve(B_torch, L_torch, upper=False)
        
        self.assertTrue(np.allclose(np.array(X_chol_dll.data).reshape(2, 3, 1), X_torch.detach().numpy(), atol=1e-4))
        
        # Backward
        grad_out = np.random.randn(2, 3, 1).astype(np.float32)
        X_chol_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [2, 3, 1]))
        X_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(B_dll.grad.data, B_torch.grad.flatten().numpy(), atol=1e-3))
        self.assertTrue(np.allclose(A_dll.grad.data, A_torch.grad.flatten().numpy(), atol=1e-3))

    def test_lstsq_batched(self):
        A_np = np.random.randn(2, 5, 3).astype(np.float32)
        B_np = np.random.randn(2, 5, 2).astype(np.float32)
        
        A_dll = DLL.Tensor(A_np.flatten().tolist(), [2, 5, 3])
        B_dll = DLL.Tensor(B_np.flatten().tolist(), [2, 5, 2])
        A_dll.requires_grad = True
        B_dll.requires_grad = True
        
        X_dll = DLL.linalg.lstsq(A_dll, B_dll)
        
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        X_torch = torch.linalg.lstsq(A_torch, B_torch).solution
        
        self.assertTrue(np.allclose(np.array(X_dll.data).reshape(2, 3, 2), X_torch.detach().numpy(), atol=1e-3))
        
        # Backward
        grad_out = np.random.randn(2, 3, 2).astype(np.float32)
        X_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [2, 3, 2]))
        X_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(B_dll.grad.data, B_torch.grad.flatten().numpy(), atol=1e-2))
        self.assertTrue(np.allclose(A_dll.grad.data, A_torch.grad.flatten().numpy(), atol=1e-2))

if __name__ == "__main__":
    unittest.main()
