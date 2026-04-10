#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "ops/binary.h"
#include "ops/unary.h"
#include "ops/reduce.h"
#include "ops/linalg.h"
#include "Random/randomTensor.h"
#include "Random/random.h"

#ifdef DLL_GPU_ENABLED
#include "gpu_context.h"
#endif

namespace py = pybind11;

#include "ops/manipulation.h"

PYBIND11_MODULE(_C, m) {
    m.doc() = "DLL Autograd C++ extension";
    
    m.def("cat", &cat, py::arg("tensors"), py::arg("dim") = 0);
    m.def("stack", &stack, py::arg("tensors"), py::arg("dim") = 0);

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>>(), py::arg("data"), py::arg("shape"))
        .def(py::init<py::list>(), py::arg("data"))

        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("data", &Tensor::get_data)
        .def_readonly("device", &Tensor::device)
        
        .def("__getitem__", &Tensor::slice)

        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("grad", [](Tensor& t) { return t.grad; })
        .def("backward", &Tensor::backward, py::arg("grad") = nullptr)
        .def("zero_grad", &Tensor::zero_grad)

        .def("item", &Tensor::item)
        .def("__repr__", &Tensor::repr)

        // Device management
        .def("to", &Tensor::to, py::arg("device"))
        .def("cpu", &Tensor::cpu)
        .def("is_gpu", &Tensor::is_gpu)

        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("squeeze", &Tensor::squeeze, py::arg("dim") = -1)
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"))

        .def("transpose", [](const std::shared_ptr<Tensor>& a, int dim0, int dim1) { return transpose(a, dim0, dim1); }, py::arg("dim0") = -2, py::arg("dim1") = -1)
        .def("exp", [](const std::shared_ptr<Tensor>& a) { return exp(a); })
        .def("log", [](const std::shared_ptr<Tensor>& a) { return log(a); })
        .def("sqrt", [](const std::shared_ptr<Tensor>& a) { return sqrt(a); })
        .def("cbrt", [](const std::shared_ptr<Tensor>& a) { return cbrt(a); })
        .def("abs", [](const std::shared_ptr<Tensor>& a) { return abs(a); })
        .def("sin", [](const std::shared_ptr<Tensor>& a) { return sin(a); })
        .def("cos", [](const std::shared_ptr<Tensor>& a) { return cos(a); })
        .def("relu", [](const std::shared_ptr<Tensor>& a) { return relu(a); })
        .def("tanh", [](const std::shared_ptr<Tensor>& a) { return tanh(a); })
        .def("sigmoid", [](const std::shared_ptr<Tensor>& a) { return sigmoid(a); })
        .def("softmax", [](const std::shared_ptr<Tensor>& a, int dim) { return softmax(a, dim); }, py::arg("dim") = -1)

        .def("__add__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return add(a, b); })
        .def("__add__", [](const std::shared_ptr<Tensor>& a, float scalar) { return add_scalar(a, scalar); })
        .def("__radd__", [](const std::shared_ptr<Tensor>& a, float scalar) { return add_scalar(a, scalar); })

        .def("__sub__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return sub(a, b); })
        .def("__sub__", [](const std::shared_ptr<Tensor>& a, float scalar) { return sub_scalar(a, scalar); })
        .def("__rsub__", [](const std::shared_ptr<Tensor>& a, float scalar) { return rsub_scalar(a, scalar); })

        .def("__mul__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return mul(a, b); })
        .def("__mul__", [](const std::shared_ptr<Tensor>& a, float scalar) { return mul_scalar(a, scalar); })
        .def("__rmul__", [](const std::shared_ptr<Tensor>& a, float scalar) { return mul_scalar(a, scalar); })
        .def("__matmul__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return matmul(a, b); })

        .def("__truediv__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return div(a, b); })
        .def("__truediv__", [](const std::shared_ptr<Tensor>& a, float scalar) { return div_scalar(a, scalar); })
        .def("__rtruediv__", [](const std::shared_ptr<Tensor>& a, float scalar) { return rdiv_scalar(a, scalar); })

        .def("__pow__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return pow(a, b); })
        .def("__pow__", [](const std::shared_ptr<Tensor>& a, float scalar) { return pow_scalar(a, scalar); })
        .def("__rpow__", [](const std::shared_ptr<Tensor>& a, float scalar) { return rpow_scalar(a, scalar); })

        .def("sum", [](const std::shared_ptr<Tensor>& a, bool keepdim) { return sum(a, keepdim); }, py::arg("keepdim") = false)
        .def("sum", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim) { return sum(a, dim, keepdim); }, py::arg("dim"), py::arg("keepdim") = false)
        .def("mean", [](const std::shared_ptr<Tensor>& a, bool keepdim) { return mean(a, keepdim); }, py::arg("keepdim") = false)
        .def("mean", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim) { return mean(a, dim, keepdim); }, py::arg("dim"), py::arg("keepdim") = false)
        .def("prod", [](const std::shared_ptr<Tensor>& a, bool keepdim) { return prod(a, keepdim); }, py::arg("keepdim") = false)
        .def("prod", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim) { return prod(a, dim, keepdim); }, py::arg("dim"), py::arg("keepdim") = false)
        .def("max", [](const std::shared_ptr<Tensor>& a, bool keepdim) { return max(a, keepdim); }, py::arg("keepdim") = false)
        .def("max", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim) { return max(a, dim, keepdim); }, py::arg("dim"), py::arg("keepdim") = false)
        .def("min", [](const std::shared_ptr<Tensor>& a, bool keepdim) { return min(a, keepdim); }, py::arg("keepdim") = false)
        .def("min", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim) { return min(a, dim, keepdim); }, py::arg("dim"), py::arg("keepdim") = false)
        .def("var", [](const std::shared_ptr<Tensor>& a, bool keepdim, bool unbiased) { return var(a, keepdim, unbiased); }, py::arg("keepdim") = false, py::arg("unbiased") = true)
        .def("var", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim, bool unbiased) { return var(a, dim, keepdim, unbiased); }, py::arg("dim"), py::arg("keepdim") = false, py::arg("unbiased") = true)
        .def("std", [](const std::shared_ptr<Tensor>& a, bool keepdim, bool unbiased) { return std_dev(a, keepdim, unbiased); }, py::arg("keepdim") = false, py::arg("unbiased") = true)
        .def("std", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim, bool unbiased) { return std_dev(a, dim, keepdim, unbiased); }, py::arg("dim"), py::arg("keepdim") = false, py::arg("unbiased") = true)
        
        // Advanced linalg
        .def("inv", [](const std::shared_ptr<Tensor>& a) { return inverse(a); })
        .def("det", [](const std::shared_ptr<Tensor>& a) { return determinant(a); })
        .def("solve", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return solve(a, b); }, py::arg("b"))
        .def("cholesky", [](const std::shared_ptr<Tensor>& a) { return cholesky(a); })
        .def("svd", [](const std::shared_ptr<Tensor>& a, bool full_matrices) { return svd(a, full_matrices); }, py::arg("full_matrices") = true)
        .def("diag", [](const std::shared_ptr<Tensor>& a, int diagonal) { return diag(a, diagonal); }, py::arg("diagonal") = 0)
        .def("eig", [](const std::shared_ptr<Tensor>& a) { return eig(a); })
        .def("lu", [](const std::shared_ptr<Tensor>& a) { return lu(a); })
        .def("qr", [](const std::shared_ptr<Tensor>& a) { return qr(a); })
        .def("matrix_exp", [](const std::shared_ptr<Tensor>& a) { return matrix_exp(a); })
        .def("lu_factor", [](const std::shared_ptr<Tensor>& a) { return lu_factor(a); })
        .def("lu_solve", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& lu, const std::shared_ptr<Tensor>& piv, bool adjoint) {
            return lu_solve(a, lu, piv, adjoint);
        }, py::arg("lu"), py::arg("pivots"), py::arg("adjoint") = false)
        .def("cholesky_solve", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& l) { return cholesky_solve(a, l); }, py::arg("l"))
        .def("lstsq", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return lstsq(a, b); }, py::arg("b"));
    
    m.def("rand", [](std::vector<int> shape, float min = 0.0, float max = 1.0) { return randomTensor::rand(shape, min, max); }, py::arg("shape"), py::arg("min") = 0, py::arg("max") = 1);
    m.def("randn", [](std::vector<int> shape, float mean = 0.0, float stddev = 1.0) { return randomTensor::randn(shape, mean, stddev); }, py::arg("shape"), py::arg("mu") = 0, py::arg("std") = 1);
    m.def("seed", [](uint32_t seed){ randomGen::set_seed(seed); }, py::arg("seed"));

#ifdef DLL_GPU_ENABLED
    m.def("gpu_available", []() { return GPUContext::instance().is_available(); });
    m.def("gpu_device_name", []() { return GPUContext::instance().device_name(); });
#else
    m.def("gpu_available", []() { return false; });
    m.def("gpu_device_name", []() { return std::string("No GPU support (built without OpenCL)"); });
#endif
}
