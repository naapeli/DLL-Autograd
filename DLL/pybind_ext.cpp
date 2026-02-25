#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "ops/binary.h"


namespace py = pybind11;

PYBIND11_MODULE(_c_tensor, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>>(), py::arg("data"), py::arg("shape"))
        
        // Properties
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("strides", &Tensor::get_strides)
        .def_property_readonly("data", &Tensor::get_data)
        .def("item", &Tensor::item)
        .def("transpose", &Tensor::transpose)
        .def("__repr__", &Tensor::repr)

        // Addition
        .def("__add__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return add(a, b); })
        .def("__add__", [](const std::shared_ptr<Tensor>& a, float scalar) { return add_scalar(a, scalar); })
        .def("__radd__", [](const std::shared_ptr<Tensor>& a, float scalar) { return add_scalar(a, scalar); })

        // Subtraction
        .def("__sub__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return sub(a, b); })
        .def("__sub__", [](const std::shared_ptr<Tensor>& a, float scalar) { return sub_scalar(a, scalar); })
        .def("__rsub__", [](const std::shared_ptr<Tensor>& a, float scalar) { return rsub_scalar(a, scalar); })

        // Multiplication
        .def("__mul__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return mul(a, b); })
        .def("__mul__", [](const std::shared_ptr<Tensor>& a, float scalar) { return mul_scalar(a, scalar); })
        .def("__rmul__", [](const std::shared_ptr<Tensor>& a, float scalar) { return mul_scalar(a, scalar); })

        // Division
        .def("__truediv__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return div(a, b); })
        .def("__truediv__", [](const std::shared_ptr<Tensor>& a, float scalar) { return div_scalar(a, scalar); })
        .def("__rtruediv__", [](const std::shared_ptr<Tensor>& a, float scalar) { return rdiv_scalar(a, scalar); })

        // Power
        .def("__pow__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return pow(a, b); })
        .def("__pow__", [](const std::shared_ptr<Tensor>& a, float scalar) { return pow_scalar(a, scalar); })
        .def("__rpow__", [](const std::shared_ptr<Tensor>& a, float scalar) { return rpow_scalar(a, scalar); })
        ;
}
