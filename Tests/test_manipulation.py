import torch
import numpy as np
import DLL
import unittest


class TestManipulation(unittest.TestCase):
    def test_cat_2d(self):
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 4).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 3])
        b_dll = DLL.Tensor(b_np.flatten().tolist(), [2, 4])
        a_dll.requires_grad = True
        b_dll.requires_grad = True
        
        # Dim 1
        out_dll = DLL.cat([a_dll, b_dll], dim=1)
        out_torch = torch.cat([torch.tensor(a_np), torch.tensor(b_np)], dim=1)
        
        self.assertEqual(out_dll.shape, [2, 7])
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(2, 7), out_torch.numpy(), atol=1e-5))
        
        # Backward
        grad_out = np.random.randn(2, 7).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [2, 7]))
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        torch.cat([a_torch, b_torch], dim=1).backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(b_dll.grad.data, b_torch.grad.flatten().numpy(), atol=1e-5))

    def test_cat_3d_dim0(self):
        a_np = np.random.randn(2, 2, 2).astype(np.float32)
        b_np = np.random.randn(3, 2, 2).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 2, 2])
        b_dll = DLL.Tensor(b_np.flatten().tolist(), [3, 2, 2])
        a_dll.requires_grad = True
        b_dll.requires_grad = True
        
        out_dll = DLL.cat([a_dll, b_dll], dim=0)
        out_torch = torch.cat([torch.tensor(a_np), torch.tensor(b_np)], dim=0)
        
        self.assertEqual(out_dll.shape, [5, 2, 2])
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(5, 2, 2), out_torch.numpy(), atol=1e-5))
        
        grad_out = np.random.randn(5, 2, 2).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [5, 2, 2]))
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        torch.cat([a_torch, b_torch], dim=0).backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-5))

    def test_stack(self):
        a_np = np.random.randn(2, 2).astype(np.float32)
        b_np = np.random.randn(2, 2).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 2])
        b_dll = DLL.Tensor(b_np.flatten().tolist(), [2, 2])
        a_dll.requires_grad = True
        b_dll.requires_grad = True
        
        out_dll = DLL.stack([a_dll, b_dll], dim=0)
        out_torch = torch.stack([torch.tensor(a_np), torch.tensor(b_np)], dim=0)
        
        self.assertEqual(out_dll.shape, [2, 2, 2])
        self.assertTrue(np.allclose(np.array(out_dll.data).reshape(2, 2, 2), out_torch.numpy(), atol=1e-5))
        
        grad_out = np.random.randn(2, 2, 2).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [2, 2, 2]))
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        torch.stack([a_torch, b_torch], dim=0).backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.data, a_torch.grad.flatten().numpy(), atol=1e-5))

    def test_cat_gpu(self):
        if not DLL.gpu_available():
            self.skipTest("GPU not available")
        
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 4).astype(np.float32)
        
        a_dll = DLL.Tensor(a_np.flatten().tolist(), [2, 3]).to("gpu")
        b_dll = DLL.Tensor(b_np.flatten().tolist(), [2, 4]).to("gpu")
        a_dll.requires_grad = True
        b_dll.requires_grad = True
        
        out_dll = DLL.cat([a_dll, b_dll], dim=1)
        self.assertEqual(out_dll.device, "gpu")
        
        out_cpu = np.array(out_dll.cpu().data).reshape(2, 7)
        out_torch = torch.cat([torch.tensor(a_np), torch.tensor(b_np)], dim=1)
        self.assertTrue(np.allclose(out_cpu, out_torch.numpy(), atol=1e-5))
        
        grad_out = np.random.randn(2, 7).astype(np.float32)
        out_dll.backward(DLL.Tensor(grad_out.flatten().tolist(), [2, 7]).to("gpu"))
        
        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)
        torch.cat([a_torch, b_torch], dim=1).backward(torch.tensor(grad_out))
        
        self.assertTrue(np.allclose(a_dll.grad.cpu().data, a_torch.grad.flatten().numpy(), atol=1e-5))

if __name__ == "__main__":
    unittest.main()
