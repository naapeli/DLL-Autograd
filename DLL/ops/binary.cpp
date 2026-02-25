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

// --- Addition ---
std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    check_compatibility(a, b);
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = (*a->data)[i] + (*b->data)[i];
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        out->_backward = [out, a, b]() {
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += (*out->grad->data)[i];
                }
            }
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*b->grad->data)[i] += (*out->grad->data)[i];
                }
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
    check_compatibility(a, b);
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = (*a->data)[i] * (*b->data)[i];
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        out->_backward = [out, a, b]() {
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += (*b->data)[i] * (*out->grad->data)[i];
                }
            }
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*b->grad->data)[i] += (*a->data)[i] * (*out->grad->data)[i];
                }
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
    check_compatibility(a, b);
    std::vector<float> result_data(a->data->size());
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = std::pow((*a->data)[i], (*b->data)[i]);
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        out->_backward = [out, a, b]() {
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += (*b->data)[i] * std::pow((*a->data)[i], (*b->data)[i] - 1) * (*out->grad->data)[i];
                }
            }
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*b->grad->data)[i] += (*out->data)[i] * std::log((*a->data)[i] + 1e-8f) * (*out->grad->data)[i];
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
