import torch
import numpy as np
import DLL
import unittest

class TestLinalgAdvanced(unittest.TestCase):
    def test_trace(self):
        shape = (2, 3, 3)
        a_np = np.random.randn(*shape).astype(np.float32)
        a_dll = DLL.Tensor(a_np.flatten().tolist(), list(shape))
        a_dll.requires_grad = True
        
        out_dll = DLL.linalg.trace(a_dll)
        out_torch = torch.tensor(a_np).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        
        self.assertTrue(np.allclose(out_dll.data, out_torch.numpy(), atol=1e-5))
        
        out_dll.backward(DLL.Tensor(np.ones_like(out_dll.data).flatten().tolist(), list(out_dll.shape)))
        # grad of trace is Identity
        a_grad_np = np.array(a_dll.grad.data).reshape(shape)
        for b in range(2):
            self.assertTrue(np.allclose(a_grad_np[b], np.eye(3), atol=1e-5))

    def test_outer(self):
        a_np = np.random.randn(4).astype(np.float32)
        b_np = np.random.randn(5).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.tolist(), [4])
        a_dll.requires_grad = True
        b_dll = DLL.Tensor(b_np.tolist(), [5])
        b_dll.requires_grad = True
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        
        out_dll = DLL.linalg.outer(a_dll, b_dll)
        out_torch = torch.outer(a_torch, b_torch)
        
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(4, 5), out_torch.detach().numpy(), atol=1e-5))
        
        out_grad = np.random.randn(4, 5).astype(np.float32)
        out_dll.backward(DLL.Tensor(out_grad.flatten().tolist(), [4, 5]))
        out_torch.backward(torch.tensor(out_grad))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(b_dll.grad.data, b_torch.grad.detach().numpy(), atol=1e-5))

    def test_kron(self):
        a_np = np.random.randn(2, 2).astype(np.float32)
        b_np = np.random.randn(3, 3).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 2])
        a_dll.requires_grad = True
        b_dll = DLL.Tensor(b_np.flatten().tolist(), [3, 3])
        b_dll.requires_grad = True
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        
        out_dll = DLL.linalg.kron(a_dll, b_dll)
        out_torch = torch.kron(a_torch, b_torch)
        
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(6, 6), out_torch.detach().numpy(), atol=1e-4))
        
        out_grad = np.random.randn(6, 6).astype(np.float32)
        out_dll.backward(DLL.Tensor(out_grad.flatten().tolist(), [6, 6]))
        out_torch.backward(torch.tensor(out_grad))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-3))
        self.assertTrue(np.allclose(b_dll.grad.data, b_torch.grad.flatten().numpy(), atol=1e-3))

    def test_matrix_power(self):
        a_np = np.random.randn(3, 3).astype(np.float32)
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [3, 3])
        a_dll.requires_grad = True
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        # n = 3
        out_dll = DLL.linalg.matrix_power(a_dll, 3)
        out_torch = torch.linalg.matrix_power(a_torch, 3)
        
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(3, 3), out_torch.detach().numpy(), atol=1e-3))
        
        out_grad = np.random.randn(3, 3).astype(np.float32)
        out_dll.backward(DLL.Tensor(out_grad.flatten().tolist(), [3, 3]))
        out_torch.backward(torch.tensor(out_grad))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-3))

    def test_cond(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 2])
        
        c = DLL.linalg.cond(a_dll)
        c_ref = np.linalg.cond(a_np)
        
        self.assertAlmostEqual(c.item(), c_ref, places=3)

    def test_matrix_exp(self):
        a_np = np.random.randn(3, 3).astype(np.float32) * 0.5
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [3, 3])
        a_dll.requires_grad = True
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        
        out_dll = DLL.linalg.matrix_exp(a_dll)
        out_torch = torch.matrix_exp(a_torch)
        
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(3, 3), out_torch.detach().numpy(), atol=1e-3))
        
        # Backward
        grad_out = np.random.randn(3, 3).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [3, 3]))
        out_torch.backward(torch.tensor(grad_out))
        
        # Check if approx correct
        if not np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=5e-2):
            print(f"\nMatrix Exp Grad DLL: \n{np.array(a_dll.grad.data).reshape(3,3)}")
            print(f"Matrix Exp Grad Torch: \n{a_torch.grad.numpy()}")
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-1))

    def test_reshape_squeeze_unsqueeze(self):
        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 3, 4])
        a_dll.requires_grad = True
        
        # Reshape
        b_dll = a_dll.reshape([6, 4])
        self.assertEqual(b_dll.shape, [6, 4])
        
        # Unsqueeze
        c_dll = b_dll.unsqueeze(1)
        self.assertEqual(c_dll.shape, [6, 1, 4])
        
        # Squeeze
        d_dll = c_dll.squeeze(1)
        self.assertEqual(d_dll.shape, [6, 4])
        
        d_grad = np.random.randn(6, 4).astype(np.float32)
        d_dll.backward(DLL.Tensor(d_grad.flatten().tolist(), [6, 4]))
        self.assertTrue(np.allclose(a_dll.grad.data, d_grad.flatten(), atol=1e-5))

if __name__ == "__main__":
    unittest.main()
