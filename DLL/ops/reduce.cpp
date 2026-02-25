#include "ops/reduce.h"
#include "ops/binary.h"
#include <stdexcept>
#include <limits>


std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, bool keepdim) {
    float total = 0;
    for (float val : *a->data) {
        total += val;
    }

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }

    auto out = std::make_shared<Tensor>(std::vector<float>{total}, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            float upstream_grad = (*out->grad->data)[0];
            for (size_t i = 0; i < a->grad->data->size(); ++i) {
                (*a->grad->data)[i] += upstream_grad;
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a->shape[i]);
        }
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int current_stride = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = current_stride;
        current_stride *= out_shape[i];
    }

    std::vector<float> out_data(out_size, 0.0f);
    
    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a->strides[d]) % a->shape[d];
            temp_i %= a->strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        out_data[out_idx] += (*a->data)[i];
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, dim, out_strides, keepdim]() {
            if (!a->grad) a->zero_grad();            
            for (int i = 0; i < (int)a->data->size(); ++i) {
                int temp_i = i;
                int out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a->strides[d]) % a->shape[d];
                    temp_i %= a->strides[d];
                    if (d != dim) {
                        int out_d = (keepdim || d < dim) ? d : d - 1;
                        out_idx += coord * out_strides[out_d];
                    }
                }
                (*a->grad->data)[i] += (*out->grad->data)[out_idx];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> prod(const std::shared_ptr<Tensor>& a, bool keepdim) {
    float total = 1.0f;
    for (float val : *a->data) {
        total *= val;
    }
    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }
    auto out = std::make_shared<Tensor>(std::vector<float>{total}, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a]() {
            if (!a->grad) a->zero_grad();
            int zero_count = 0;
            float non_zero_prod = 1.0f;
            for (float val : *a->data) {
                if (val == 0.0f) {
                    zero_count++;
                } else {
                    non_zero_prod *= val;
                }
            }
            float upstream_grad = (*out->grad->data)[0];
            float out_val = (*out->data)[0];
            for (size_t i = 0; i < a->data->size(); ++i) {
                float val = (*a->data)[i];
                float local_grad = 0.0f;
                if (zero_count == 0) {
                    local_grad = out_val / val;
                } else if (zero_count == 1 && val == 0.0f) {
                    local_grad = non_zero_prod;
                }
                (*a->grad->data)[i] += upstream_grad * local_grad;
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> prod(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a->shape[i]);
        }
    }
    if (out_shape.empty()) out_shape = {1};
    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int current_stride = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = current_stride;
        current_stride *= out_shape[i];
    }
    std::vector<float> out_data(out_size, 1.0f);
    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a->strides[d]) % a->shape[d];
            temp_i %= a->strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        out_data[out_idx] *= (*a->data)[i];
    }
    auto out = std::make_shared<Tensor>(out_data, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, dim, out_strides, keepdim]() {
            if (!a->grad) a->zero_grad();
            int out_size = out->data->size();
            std::vector<int> zero_counts(out_size, 0);
            std::vector<float> non_zero_prods(out_size, 1.0f);
            for (int i = 0; i < (int)a->data->size(); ++i) {
                int temp_i = i;
                int out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a->strides[d]) % a->shape[d];
                    temp_i %= a->strides[d];
                    if (d != dim) {
                        int out_d = (keepdim || d < dim) ? d : d - 1;
                        out_idx += coord * out_strides[out_d];
                    }
                }
                float val = (*a->data)[i];
                if (val == 0.0f) {
                    zero_counts[out_idx]++;
                } else {
                    non_zero_prods[out_idx] *= val;
                }
            }
            for (int i = 0; i < (int)a->data->size(); ++i) {
                int temp_i = i;
                int out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a->strides[d]) % a->shape[d];
                    temp_i %= a->strides[d];
                    if (d != dim) {
                        int out_d = (keepdim || d < dim) ? d : d - 1;
                        out_idx += coord * out_strides[out_d];
                    }
                }
                float val = (*a->data)[i];
                float local_grad = 0.0f;
                int z_count = zero_counts[out_idx];
                if (z_count == 0) {
                    local_grad = (*out->data)[out_idx] / val;
                } else if (z_count == 1 && val == 0.0f) {
                    local_grad = non_zero_prods[out_idx];
                }

                (*a->grad->data)[i] += (*out->grad->data)[out_idx] * local_grad;
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, bool keepdim) {
    int total_elements = 1;
    for (int dim_size : a->shape) {
        total_elements *= dim_size;
    }
    auto sum_tensor = sum(a, keepdim);
    return div_scalar(sum_tensor, static_cast<float>(total_elements));
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    int dim_size = a->shape[dim];
    auto sum_tensor = sum(a, dim, keepdim);
    return div_scalar(sum_tensor, static_cast<float>(dim_size));
}

std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& a, bool keepdim) {
    float max_val = -std::numeric_limits<float>::infinity();
    int argmax = -1;

    for (int i = 0; i < (int)a->data->size(); ++i) {
        if ((*a->data)[i] > max_val) {
            max_val = (*a->data)[i];
            argmax = i;
        }
    }

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }

    auto out = std::make_shared<Tensor>(std::vector<float>{max_val}, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, argmax]() {
            if (!a->grad) a->zero_grad();
            if (argmax != -1) {
                (*a->grad->data)[argmax] += (*out->grad->data)[0];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    
    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a->shape[i]);
        }
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int current_stride = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = current_stride;
        current_stride *= out_shape[i];
    }

    std::vector<float> out_data(out_size, -std::numeric_limits<float>::infinity());
    std::vector<int> argmax(out_size, -1);

    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a->strides[d]) % a->shape[d];
            temp_i %= a->strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        
        float val = (*a->data)[i];
        if (val > out_data[out_idx]) {
            out_data[out_idx] = val;
            argmax[out_idx] = i; 
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, argmax]() {
            if (!a->grad) a->zero_grad();
            for (int out_idx = 0; out_idx < (int)out->data->size(); ++out_idx) {
                int winner_idx = argmax[out_idx];
                if (winner_idx != -1) {
                    (*a->grad->data)[winner_idx] += (*out->grad->data)[out_idx];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> min(const std::shared_ptr<Tensor>& a, bool keepdim) {
    float min_val = std::numeric_limits<float>::infinity();
    int argmin = -1;

    for (int i = 0; i < (int)a->data->size(); ++i) {
        if ((*a->data)[i] < min_val) {
            min_val = (*a->data)[i];
            argmin = i;
        }
    }

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }

    auto out = std::make_shared<Tensor>(std::vector<float>{min_val}, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, argmin]() {
            if (!a->grad) a->zero_grad();
            if (argmin != -1) {
                (*a->grad->data)[argmin] += (*out->grad->data)[0];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> min(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    
    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a->shape[i]);
        }
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int current_stride = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = current_stride;
        current_stride *= out_shape[i];
    }

    std::vector<float> out_data(out_size, std::numeric_limits<float>::infinity());
    std::vector<int> argmin(out_size, -1);

    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a->strides[d]) % a->shape[d];
            temp_i %= a->strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        
        float val = (*a->data)[i];
        if (val < out_data[out_idx]) {
            out_data[out_idx] = val;
            argmin[out_idx] = i; 
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, argmin]() {
            if (!a->grad) a->zero_grad();
            for (int out_idx = 0; out_idx < (int)out->data->size(); ++out_idx) {
                int winner_idx = argmin[out_idx];
                if (winner_idx != -1) {
                    (*a->grad->data)[winner_idx] += (*out->grad->data)[out_idx];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& a, bool keepdim, bool unbiased) {
    auto mu = mean(a, true);
    auto diff = sub(a, mu);
    auto variance = pow_scalar(diff, 2);

    int n_elements = 1;
    for (int dim_size : a->shape) { n_elements *= dim_size; }
    int normalizer = (unbiased) ? n_elements - 1 : n_elements;
    return div_scalar(sum(variance, keepdim), static_cast<float>(normalizer));
}

std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& a, int dim, bool keepdim, bool unbiased) {
    auto mu = mean(a, dim, true);
    auto diff = sub(a, mu);
    auto variance = pow_scalar(diff, 2);
    int normalizer = (unbiased) ? (a->shape)[dim] - 1 : (a->shape)[dim];
    return div_scalar(sum(variance, dim, keepdim), static_cast<float>(normalizer));
}

std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a, bool keepdim, bool unbiased) {
    return pow_scalar(var(a, keepdim, unbiased), 0.5f);
}

std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a, int dim, bool keepdim, bool unbiased) {
    return pow_scalar(var(a, dim, keepdim, unbiased), 0.5f);
}
