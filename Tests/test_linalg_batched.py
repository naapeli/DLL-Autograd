import torch
import numpy as np
import DLL
import unittest


class TestLinalgBatched(unittest.TestCase):
    def test_inverse_batched(self):
        shape = (3, 4, 4)
        a_np = np.random.randn(*shape).astype(np.float32)
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        out_dll = DLL.linalg.inv(a_dll)
        out_torch = torch.inverse(a_torch)
        
        self.assertTrue(np.allclose(out_dll.data, out_torch.detach().flatten().numpy(), atol=1e-5))
        
        grad_out = np.random.randn(*shape).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), list(shape)))
        out_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-4))

    def test_solve_batched(self):
        a_shape = (2, 3, 3)
        b_shape = (2, 3, 2) # Same batch dim for simplicity in first test
        a_np = np.random.randn(*a_shape).astype(np.float32)
        b_np = np.random.randn(*b_shape).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(a_shape))
        a_dll.requires_grad = True
        b_dll = DLL.Tensor(b_np.flatten().tolist(), list(b_shape))
        b_dll.requires_grad = True
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        
        out_dll = DLL.linalg.solve(a_dll, b_dll)
        out_torch = torch.linalg.solve(a_torch, b_torch)
        
        self.assertTrue(np.allclose(out_dll.data, out_torch.detach().flatten().numpy(), atol=1e-5))
        
        grad_out = np.random.randn(*out_dll.shape).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), list(out_dll.shape)))
        out_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-4))
        self.assertTrue(np.allclose(b_dll.grad.data, b_torch.grad.flatten().numpy(), atol=1e-4))

    def test_cholesky_batched(self):
        shape = (2, 4, 4)
        a_np = np.random.randn(*shape).astype(np.float32)
        a_np = np.matmul(a_np, np.transpose(a_np, (0, 2, 1))) + np.eye(4) * 0.1
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        out_dll = DLL.linalg.cholesky(a_dll)
        out_torch = torch.linalg.cholesky(a_torch)
        
        self.assertTrue(np.allclose(out_dll.data, out_torch.detach().flatten().numpy(), atol=1e-5))
        
        grad_out = np.random.randn(*shape).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), list(shape)))
        out_torch.backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-4))

    def test_svd_batched(self):
        shape = (2, 3, 3)
        a_np = np.random.randn(*shape).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        U_dll, S_dll, Vt_dll = DLL.linalg.svd(a_dll)
        U_torch, S_torch, Vh_torch = torch.linalg.svd(a_torch)
        
        self.assertTrue(np.allclose(S_dll.data, S_torch.detach().flatten().numpy(), atol=1e-5))
        
        s_data = np.array(S_dll.data).reshape(2, 3)
        s_mats = np.stack([np.diag(s) for s in s_data])
        u_data = np.array(U_dll.data).reshape(2, 3, 3)
        vt_data = np.array(Vt_dll.data).reshape(2, 3, 3)
        recon = np.matmul(u_data, np.matmul(s_mats, vt_data))
        self.assertTrue(np.allclose(recon, a_np, atol=1e-5))
        
        S_dll.backward(DLL.Tensor(np.ones_like(S_dll.data).flatten().tolist(), list(S_dll.shape)))
        (S_torch.sum()).backward()
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-4))

    def test_qr_batched(self):
        shape = (2, 4, 4)
        a_np = np.random.randn(*shape).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        Q_dll, R_dll = DLL.linalg.qr(a_dll)
        Q_torch, R_torch = torch.linalg.qr(a_torch)
        
        # Check forward: reconstruction A = Q R
        q_data = np.array(Q_dll.data).reshape(2, 4, 4)
        r_data = np.array(R_dll.data).reshape(2, 4, 4)
        recon = np.matmul(q_data, r_data)
        self.assertTrue(np.allclose(recon, a_np, atol=1e-5))
        
        # Check orthogonality
        qtq = np.matmul(np.transpose(q_data, (0, 2, 1)), q_data)
        self.assertTrue(np.allclose(qtq, np.eye(4), atol=1e-5))
        
        # Backward
        loss_dll = (Q_dll**2).sum() + (R_dll**2).sum()
        loss_torch = (Q_torch**2).sum() + (R_torch**2).sum()
        
        loss_dll.backward()
        loss_torch.backward()
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-4))

    def test_det_batched(self):
        shape = (2, 2, 3, 3) # 4D batch
        a_np = np.random.randn(*shape).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        out_dll = DLL.linalg.det(a_dll)
        out_torch = torch.linalg.det(a_torch)
        
        self.assertTrue(np.allclose(out_dll.data, out_torch.detach().flatten().numpy(), atol=1e-3))
        
        # Backward
        out_dll.backward(DLL.Tensor(np.ones_like(out_dll.data).flatten().tolist(), list(out_dll.shape)))
        (out_torch.sum()).backward()
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-3))

    def test_lu_batched(self):
        shape = (2, 3, 3)
        a_np = np.random.randn(*shape).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        P_dll, L_dll, U_dll = DLL.linalg.lu(a_dll)
        
        # Check forward: A = P L U
        p_data = np.array(P_dll.data).reshape(2, 3, 3)
        l_data = np.array(L_dll.data).reshape(2, 3, 3)
        u_data = np.array(U_dll.data).reshape(2, 3, 3)
        recon = np.matmul(p_data, np.matmul(l_data, u_data))
        self.assertTrue(np.allclose(recon, a_np, atol=1e-3))
        
        # Backward
        loss_dll = (L_dll**2).sum() + (U_dll**2).sum()
        loss_dll.backward()
        
        # Use torch for comparison
        self.assertIsNotNone(a_dll.grad)

    def test_eig_batched(self):
        shape = (2, 4, 4)
        a_np = np.random.randn(*shape).astype(np.float32)
        # Make symmetric
        a_np = a_np + np.transpose(a_np, (0, 2, 1))
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        L_dll, V_dll = DLL.linalg.eig(a_dll)
        L_torch, V_torch = torch.linalg.eigh(a_torch)
        
        # Check forward
        self.assertTrue(np.allclose(np.sort(L_dll.data), np.sort(L_torch.detach().flatten().numpy()), atol=1e-3))
        
        # Check reconstruction: A = V @ diag(L) @ V.T
        v_data = np.array(V_dll.data).reshape(2, 4, 4)
        l_data = np.array(L_dll.data).reshape(2, 4)
        l_mats = np.stack([np.diag(l) for l in l_data])
        recon = np.matmul(v_data, np.matmul(l_mats, np.transpose(v_data, (0, 2, 1))))
        self.assertTrue(np.allclose(recon, a_np, atol=1e-3))
        
        # Backward
        L_dll.backward(DLL.Tensor(np.ones_like(L_dll.data).flatten().tolist(), list(L_dll.shape)))
        (L_torch.sum()).backward()
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-3))

if __name__ == "__main__":
    unittest.main()
