#include "tensor.h"
#include "ops/binary.h"
#include "ops/reduce.h"
#include <memory>
#include <cmath>
#include <stdexcept>


std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a, int dim0, int dim1) {
    int ndim = a->shape.size();
    if (ndim < 2) {
        throw std::runtime_error("Transpose requires at least 2 dimensions.");
    }
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::out_of_range("Dimension out of range for transpose.");
    }
    std::vector<int> new_shape = a->shape;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::vector<int> new_strides(ndim);
    int current = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        new_strides[i] = current;
        current *= new_shape[i];
    }
    int num_elements = current;
    auto out_data = std::make_shared<std::vector<float>>(num_elements);
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    std::vector<int> a_contig_strides(ndim);
    current = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        a_contig_strides[i] = current;
        current *= a->shape[i];
    }
    for (int i = 0; i < num_elements; ++i) {
        int temp = i;
        int a_idx = 0;
        int out_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            int coord = temp / a_contig_strides[d];
            temp %= a_contig_strides[d];
            a_idx += coord * a->strides[d];
            int target_dim = d;
            if (d == dim0) target_dim = dim1;
            else if (d == dim1) target_dim = dim0;
            out_idx += coord * new_strides[target_dim];
        }
        out_ptr[out_idx] = a_ptr[a_idx];
    }
    auto out = std::make_shared<Tensor>(out_data, new_shape, new_strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, dim0, dim1, a_contig_strides, num_elements, ndim]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            for (int i = 0; i < num_elements; ++i) {
                int temp = i;
                int a_idx = 0;
                int out_idx = 0;
                for (int d = 0; d < ndim; ++d) {
                    int coord = temp / a_contig_strides[d];
                    temp %= a_contig_strides[d];
                    a_idx += coord * a->grad->strides[d];
                    int target_dim = d;
                    if (d == dim0) target_dim = dim1;
                    else if (d == dim1) target_dim = dim0;
                    out_idx += coord * out->grad->strides[target_dim];
                }
                a_grad_ptr[a_idx] += out_grad_ptr[out_idx];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::exp(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_ptr[i] * out_grad_ptr[i];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::log(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] / (a_ptr[i] + 1e-8f); 
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::sqrt(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] / (2.0f * out_ptr[i] + 1e-8f);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> cbrt(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::cbrt(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                float y = out_ptr[i];
                a_grad_ptr[i] += out_grad_ptr[i] / (3.0f * y * y + 1e-8f);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> abs(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::abs(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                float val = a_ptr[i];
                float sign = (val > 0.0f) - (val < 0.0f);
                a_grad_ptr[i] += out_grad_ptr[i] * sign;
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::sin(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] * std::cos(a_ptr[i]);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::cos(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] * -std::sin(a_ptr[i]);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = a_ptr[i] > 0.0f ? a_ptr[i] : 0.0f;
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                if (out_ptr[i] > 0.0f) {
                    a_grad_ptr[i] += out_grad_ptr[i];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::tanh(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                float y = out_ptr[i];
                a_grad_ptr[i] += (1.0f - y * y) * out_grad_ptr[i];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& a) {
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = 1.0f / (1.0f + std::exp(-a_ptr[i]));
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape, a->strides);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            for (size_t i = 0; i < a->data->size(); ++i) {
                float y = out_ptr[i];
                a_grad_ptr[i] += y * (1.0f - y) * out_grad_ptr[i];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a, int dim) {
    if (dim < 0) dim += a->shape.size();
    auto m = max(a, dim, true); 
    auto shifted = sub(a, m);
    auto e = exp(shifted);
    auto s = sum(e, dim, true);
    return div(e, s);
}
