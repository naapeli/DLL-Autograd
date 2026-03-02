#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "ops/binary.h"
#include "ops/unary.h"
#include "ops/reduce.h"
#include "Random/randomTensor.h"


namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>>(), py::arg("data"), py::arg("shape"))

        // .def("__getitem__", [](std::shared_ptr<Tensor> self, py::object key) {
        //     py::tuple indices;
            
        //     // 1. Handle single items vs tuples
        //     if (py::isinstance<py::tuple>(key)) {
        //         indices = key.cast<py::tuple>();
        //     } else {
        //         indices = py::make_tuple(key);
        //     }

        //     if (indices.size() > self->shape.size()) {
        //         throw std::invalid_argument("Too many indices for tensor.");
        //     }

        //     std::vector<int> new_shape;
        //     std::vector<int> new_strides;
        //     int new_offset = self->offset; // Starting from the current view's offset

        //     // 2. Iterate through dimensions
        //     for (size_t i = 0; i < self->shape.size(); ++i) {
        //         // If user omitted trailing dimensions (e.g., tensor[0]), treat them as ":"
        //         if (i >= indices.size()) {
        //             new_shape.push_back(self->shape[i]);
        //             new_strides.push_back(self->strides[i]);
        //             continue;
        //         }

        //         py::object item = indices[i];

        //         if (py::isinstance<py::int_>(item)) {
        //             // --- Handle single integer (e.g., 2 or -1) ---
        //             int idx = item.cast<int>();
        //             if (idx < 0) idx += self->shape[i];
        //             if (idx < 0 || idx >= self->shape[i]) {
        //                 throw std::out_of_range("Index out of bounds");
        //             }
                    
        //             new_offset += idx * self->strides[i];
        //             // Dimension is dropped, so we don't push to new_shape/new_strides

        //         } else if (py::isinstance<py::slice>(item)) {
        //             // --- Handle slice (e.g., 0:5 or 4:10) ---
        //             py::slice slice_obj = item.cast<py::slice>();
        //             size_t start, stop, step, slicelength;
                    
        //             // pybind11's magic method to calculate exact bounds based on dim size!
        //             if (!slice_obj.compute(self->shape[i], &start, &stop, &step, &slicelength)) {
        //                 throw py::error_already_set();
        //             }

        //             if (slicelength > 0) {
        //                 new_offset += start * self->strides[i];
        //                 new_shape.push_back(slicelength);
                        
        //                 // Extract original step to handle negative strides properly
        //                 int raw_step = 1; 
        //                 if (!slice_obj.attr("step").is_none()) {
        //                     raw_step = slice_obj.attr("step").cast<int>();
        //                 }
        //                 new_strides.push_back(self->strides[i] * raw_step);
        //             } else {
        //                 new_shape.push_back(0);
        //                 new_strides.push_back(self->strides[i]);
        //             }
        //         } else {
        //             throw std::invalid_argument("Invalid index type. Expected int or slice.");
        //         }
        //     }

        //     // Return a new zero-copy view!
        //     return std::make_shared<Tensor>(self->data, new_shape, new_strides, new_offset);
        // })
        .def("__getitem__", [](std::shared_ptr<Tensor> self, py::object key) {
            py::tuple indices;
            if (py::isinstance<py::tuple>(key)) {
                indices = key.cast<py::tuple>();
            } else {
                indices = py::make_tuple(key);
            }

            if (indices.size() > self->shape.size()) {
                throw std::invalid_argument("Too many indices for tensor.");
            }

            std::vector<int> new_shape;
            std::vector<int> new_strides;
            int new_offset = self->offset;

            // 1. Calculate the shape and strides of the slice (same as before)
            for (size_t i = 0; i < self->shape.size(); ++i) {
                if (i >= indices.size()) {
                    new_shape.push_back(self->shape[i]);
                    new_strides.push_back(self->strides[i]);
                    continue;
                }

                py::object item = indices[i];

                if (py::isinstance<py::int_>(item)) {
                    int idx = item.cast<int>();
                    if (idx < 0) idx += self->shape[i];
                    if (idx < 0 || idx >= self->shape[i]) throw std::out_of_range("Index out of bounds");
                    new_offset += idx * self->strides[i];
                } else if (py::isinstance<py::slice>(item)) {
                    py::slice slice_obj = item.cast<py::slice>();
                    size_t start, stop, step, slicelength;
                    
                    if (!slice_obj.compute(self->shape[i], &start, &stop, &step, &slicelength)) {
                        throw py::error_already_set();
                    }

                    if (slicelength > 0) {
                        new_offset += start * self->strides[i];
                        new_shape.push_back(slicelength);
                        int raw_step = 1; 
                        if (!slice_obj.attr("step").is_none()) raw_step = slice_obj.attr("step").cast<int>();
                        new_strides.push_back(self->strides[i] * raw_step);
                    } else {
                        new_shape.push_back(0);
                        new_strides.push_back(self->strides[i]);
                    }
                } else {
                    throw std::invalid_argument("Invalid index type. Expected int or slice.");
                }
            }

            // 2. DEEP COPY: Extract the elements into a new contiguous vector
            int total_elements = 1;
            for (int dim : new_shape) {
                total_elements *= dim;
            }

            std::vector<float> copied_data(total_elements);

            if (total_elements > 0) {
                // Map the new flat index to the old flat index using the calculated strides
                for (int i = 0; i < total_elements; ++i) {
                    int old_flat_index = new_offset;
                    int temp = i;
                    
                    // Work backwards through dimensions to calculate coordinates
                    for (int j = new_shape.size() - 1; j >= 0; --j) {
                        int coord = temp % new_shape[j];
                        old_flat_index += coord * new_strides[j];
                        temp /= new_shape[j];
                    }
                    copied_data[i] = self->data->at(old_flat_index);
                }
            }

            // 3. Return a completely new contiguous Tensor
            return std::make_shared<Tensor>(copied_data, new_shape);
        })
        
        // Properties
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("strides", &Tensor::get_strides)
        .def_property_readonly("data", &Tensor::get_data)

        // backprop
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("grad", [](Tensor& t) { return t.grad; })
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)

        // utils
        .def("item", &Tensor::item)
        .def("__repr__", &Tensor::repr)

        // unary operations
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

        // binary operations
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

        // reduce
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
        .def("std", [](const std::shared_ptr<Tensor>& a, int dim, bool keepdim, bool unbiased) { return std_dev(a, dim, keepdim, unbiased); }, py::arg("dim"), py::arg("keepdim") = false, py::arg("unbiased") = true);
    
    m.def("rand", [](std::vector<int> shape, float min = 0.0, float max = 1.0) { return randomTensor::rand(shape, min, max); }, py::arg("shape"), py::arg("min") = 0, py::arg("max") = 1);
    m.def("randn", [](std::vector<int> shape, float mean = 0.0, float stddev = 1.0) { return randomTensor::randn(shape, mean, stddev); }, py::arg("shape"), py::arg("mu") = 0, py::arg("std") = 1);
}
