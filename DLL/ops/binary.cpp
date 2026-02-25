#include "ops/binary.h"
#include <stdexcept>
#include <cmath>


void check_compatibility(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) {
        throw std::invalid_argument("Shapes must match for this simple operation.");
    }
    if (a->strides != b->strides) {
        throw std::invalid_argument("Strides must match. Needs an N-dimensional iterator.");
    }
}

struct BroadcastInfo {
    std::vector<int> out_shape;
    std::vector<int> out_strides;
    std::vector<int> strides_a;
    std::vector<int> strides_b;
    int out_size;
};

BroadcastInfo setup_broadcast(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    int ndim_a = a->shape.size();
    int ndim_b = b->shape.size();
    int ndim_out = std::max(ndim_a, ndim_b);
    BroadcastInfo info;
    info.out_shape.resize(ndim_out);
    info.out_strides.resize(ndim_out);
    info.strides_a.resize(ndim_out, 0);
    info.strides_b.resize(ndim_out, 0);
    for (int i = 0; i < ndim_out; ++i) {
        int dim_a = (i < ndim_out - ndim_a) ? 1 : a->shape[i - (ndim_out - ndim_a)];
        int dim_b = (i < ndim_out - ndim_b) ? 1 : b->shape[i - (ndim_out - ndim_b)];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument("Shapes are not broadcastable.");
        }
        info.out_shape[i] = std::max(dim_a, dim_b);
        int stride_a = (i < ndim_out - ndim_a) ? 0 : a->strides[i - (ndim_out - ndim_a)];
        int stride_b = (i < ndim_out - ndim_b) ? 0 : b->strides[i - (ndim_out - ndim_b)];
        info.strides_a[i] = (dim_a == 1) ? 0 : stride_a;
        info.strides_b[i] = (dim_b == 1) ? 0 : stride_b;
    }
    int current_stride = 1;
    for (int i = ndim_out - 1; i >= 0; --i) {
        info.out_strides[i] = current_stride;
        current_stride *= info.out_shape[i];
    }
    info.out_size = current_stride;
    return info;
}

// --- Addition ---
std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    BroadcastInfo info = setup_broadcast(a, b);
    std::vector<float> result_data(info.out_size);
    for (int i = 0; i < info.out_size; ++i) {
        int temp = i, idx_a = 0, idx_b = 0;
        for (int d = 0; d < (int)info.out_shape.size(); ++d) {
            int coord = temp / info.out_strides[d];
            temp %= info.out_strides[d];
            idx_a += coord * info.strides_a[d];
            idx_b += coord * info.strides_b[d];
        }
        result_data[i] = (*a->data)[idx_a] + (*b->data)[idx_b];
    }
    auto out = std::make_shared<Tensor>(result_data, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        out->_backward = [out, a, b, info]() {
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            for (int i = 0; i < info.out_size; ++i) {
                int temp = i, idx_a = 0, idx_b = 0;
                for (int d = 0; d < (int)info.out_shape.size(); ++d) {
                    int coord = temp / info.out_strides[d];
                    temp %= info.out_strides[d];
                    idx_a += coord * info.strides_a[d];
                    idx_b += coord * info.strides_b[d];
                }
                float grad_out = (*out->grad->data)[i];
                if (a->requires_grad) (*a->grad->data)[idx_a] += grad_out;
                if (b->requires_grad) (*b->grad->data)[idx_b] += grad_out;
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = (*a->data)[i] + scalar;
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            for (size_t i = 0; i < out->grad->data->size(); ++i) {
                (*a->grad->data)[i] += (*out->grad->data)[i];
            }
        };
    }
    return out;
}

// --- Multiplication ---
std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    BroadcastInfo info = setup_broadcast(a, b);
    std::vector<float> result_data(info.out_size);
    for (int i = 0; i < info.out_size; ++i) {
        int temp = i, idx_a = 0, idx_b = 0;
        for (int d = 0; d < (int)info.out_shape.size(); ++d) {
            int coord = temp / info.out_strides[d];
            temp %= info.out_strides[d];
            idx_a += coord * info.strides_a[d];
            idx_b += coord * info.strides_b[d];
        }
        result_data[i] = (*a->data)[idx_a] * (*b->data)[idx_b];
    }
    auto out = std::make_shared<Tensor>(result_data, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        out->_backward = [out, a, b, info]() {
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            for (int i = 0; i < info.out_size; ++i) {
                int temp = i, idx_a = 0, idx_b = 0;
                for (int d = 0; d < (int)info.out_shape.size(); ++d) {
                    int coord = temp / info.out_strides[d];
                    temp %= info.out_strides[d];
                    idx_a += coord * info.strides_a[d];
                    idx_b += coord * info.strides_b[d];
                }
                float grad_out = (*out->grad->data)[i];
                if (a->requires_grad) (*a->grad->data)[idx_a] += (*b->data)[idx_b] * grad_out;
                if (b->requires_grad) (*b->grad->data)[idx_b] += (*a->data)[idx_a] * grad_out;
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = (*a->data)[i] * scalar;
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, scalar]() {
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += scalar * (*out->grad->data)[i];
                }
            }
        };
    }
    return out;
}

// --- Subtraction ---
std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return add(a, mul_scalar(b, -1.0f));
}
std::shared_ptr<Tensor> sub_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return add_scalar(a, -scalar);
}
std::shared_ptr<Tensor> rsub_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return add_scalar(mul_scalar(a, -1.0f), scalar);
}

// --- Power ---
std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    BroadcastInfo info = setup_broadcast(a, b);
    std::vector<float> result_data(info.out_size);
    for (int i = 0; i < info.out_size; ++i) {
        int temp = i, idx_a = 0, idx_b = 0;
        for (int d = 0; d < (int)info.out_shape.size(); ++d) {
            int coord = temp / info.out_strides[d];
            temp %= info.out_strides[d];
            idx_a += coord * info.strides_a[d];
            idx_b += coord * info.strides_b[d];
        }
        result_data[i] = std::pow((*a->data)[idx_a], (*b->data)[idx_b]);
    }
    auto out = std::make_shared<Tensor>(result_data, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        out->_backward = [out, a, b, info]() {
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            for (int i = 0; i < info.out_size; ++i) {
                int temp = i, idx_a = 0, idx_b = 0;
                for (int d = 0; d < (int)info.out_shape.size(); ++d) {
                    int coord = temp / info.out_strides[d];
                    temp %= info.out_strides[d];
                    idx_a += coord * info.strides_a[d];
                    idx_b += coord * info.strides_b[d];
                }
                float grad_out = (*out->grad->data)[i];
                float val_a = (*a->data)[idx_a];
                float val_b = (*b->data)[idx_b];
                if (a->requires_grad) {
                    (*a->grad->data)[idx_a] += val_b * std::pow(val_a, val_b - 1.0f) * grad_out;
                }
                if (b->requires_grad) {
                    (*b->grad->data)[idx_b] += (*out->data)[i] * std::log(val_a + 1e-8f) * grad_out;
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> pow_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = std::pow((*a->data)[i], scalar);
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, scalar]() {
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += scalar * std::pow((*a->data)[i], scalar - 1) * (*out->grad->data)[i];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> rpow_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = std::pow(scalar, (*a->data)[i]);
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, scalar]() {
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += (*out->data)[i] * std::log(scalar + 1e-8f) * (*out->grad->data)[i];
                }
            }
        };
    }
    return out;
}

// --- Division ---
std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return mul(a, pow_scalar(b, -1.0f));
}
std::shared_ptr<Tensor> div_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return mul_scalar(a, 1.0f / scalar);
}
std::shared_ptr<Tensor> rdiv_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return mul_scalar(pow_scalar(a, -1.0f), scalar);
}
