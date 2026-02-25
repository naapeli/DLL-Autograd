#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::shared_ptr<std::vector<float>> data;
    std::vector<int> shape;
    std::vector<int> strides;

    Tensor(std::vector<float> d, std::vector<int> s) : shape(s) {
        data = std::make_shared<std::vector<float>>(d);
        
        int total_elements = 1;
        for (int dim : shape) total_elements *= dim;
        if (data->size() != total_elements) {
            throw std::invalid_argument("Data size does not match shape.");
        }

        strides.resize(shape.size());
        int current_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
    }

    Tensor(std::shared_ptr<std::vector<float>> shared_data, std::vector<int> s, std::vector<int> st) : data(shared_data), shape(s), strides(st) {}

    std::vector<int> get_shape() const { return shape; }
    std::vector<int> get_strides() const { return strides; }    
    std::vector<float> get_data() const { return *data; } 

    float item(const std::vector<int>& indices) const {
        int flat_index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_index += indices[i] * strides[i];
        }
        return (*data)[flat_index];
    }

    std::shared_ptr<Tensor> transpose() {
        if (shape.size() != 2) {
            throw std::runtime_error("This simple transpose only supports 2D matrices right now.");
        }
        std::vector<int> new_shape = {shape[1], shape[0]};
        std::vector<int> new_strides = {strides[1], strides[0]};
        return std::make_shared<Tensor>(data, new_shape, new_strides);
    }

    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other) {
        if (shape != other->shape) {
            throw std::invalid_argument("Shapes must match for this simple addition.");
        }
        if (strides != other->strides) {
            throw std::invalid_argument("Strides must match. Adding transposed matrices requires an N-dimensional iterator.");
        }
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = (*data)[i] + (*other->data)[i];
        }
        return std::make_shared<Tensor>(result_data, shape);
    }
    std::shared_ptr<Tensor> add_scalar(float scalar) {
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = (*data)[i] + scalar;
        }
        return std::make_shared<Tensor>(result_data, shape);
    }

    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& other) {
        auto negative_other = other->mul_scalar(-1.0f);
        return this->add(negative_other);
    }
    std::shared_ptr<Tensor> sub_scalar(float scalar) {
        return this->add_scalar(-scalar);
    }
    std::shared_ptr<Tensor> rsub_scalar(float scalar) {
        return this->mul_scalar(-1.0f)->add_scalar(scalar);
    }

    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& other) {
        if (shape != other->shape) {
            throw std::invalid_argument("Shapes must match for this simple addition.");
        }
        if (strides != other->strides) {
            throw std::invalid_argument("Strides must match. Adding transposed matrices requires an N-dimensional iterator.");
        }
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = (*data)[i] * (*other->data)[i];
        }
        return std::make_shared<Tensor>(result_data, shape);
    }
    std::shared_ptr<Tensor> mul_scalar(float scalar) {
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = (*data)[i] * scalar;
        }
        return std::make_shared<Tensor>(result_data, shape);
    }

    std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& other) {
        if (shape != other->shape) {
            throw std::invalid_argument("Shapes must match for this simple addition.");
        }
        if (strides != other->strides) {
            throw std::invalid_argument("Strides must match. Adding transposed matrices requires an N-dimensional iterator.");
        }
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = std::pow((*data)[i], (*other->data)[i]);
        }
        return std::make_shared<Tensor>(result_data, shape);
    }
    std::shared_ptr<Tensor> pow_scalar(float scalar) {
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = std::pow((*data)[i], scalar);
        }
        return std::make_shared<Tensor>(result_data, shape);
    }
    std::shared_ptr<Tensor> rpow_scalar(float scalar) {
        std::vector<float> result_data(data->size());
        for (size_t i = 0; i < data->size(); ++i) {
            result_data[i] = std::pow(scalar, (*data)[i]);
        }
        return std::make_shared<Tensor>(result_data, shape);
    }

    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& other) {
        auto inverse_other = other->pow_scalar(-1.0f);
        return this->mul(inverse_other);
    }
    std::shared_ptr<Tensor> div_scalar(float scalar) {
        return this->mul_scalar(1.0f / scalar);
    }
    std::shared_ptr<Tensor> rdiv_scalar(float scalar) {
        return this->pow_scalar(-1.0f)->mul_scalar(scalar);
    }
};

PYBIND11_MODULE(_c_tensor, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>>(), py::arg("data"), py::arg("shape"))
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("strides", &Tensor::get_strides)
        .def_property_readonly("data", &Tensor::get_data)
        .def("item", &Tensor::item)

        .def("transpose", &Tensor::transpose)

        .def("__add__", &Tensor::add)
        .def("__add__", &Tensor::add_scalar)
        .def("__radd__", &Tensor::add_scalar)

        .def("__mul__", &Tensor::mul)
        .def("__mul__", &Tensor::mul_scalar)
        .def("__rmul__", &Tensor::mul_scalar)

        .def("__sub__", &Tensor::sub)
        .def("__sub__", &Tensor::sub_scalar)
        .def("__rsub__", &Tensor::rsub_scalar)

        .def("__pow__", &Tensor::pow)
        .def("__pow__", &Tensor::pow_scalar)
        .def("__rpow__", &Tensor::rpow_scalar)

        .def("__truediv__", &Tensor::div)
        .def("__truediv__", &Tensor::div_scalar)
        .def("__rtruediv__", &Tensor::rdiv_scalar)
        ;
}
