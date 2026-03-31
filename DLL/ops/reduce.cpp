#include "ops/reduce.h"
#include "ops/binary.h"
#include "ops/unary.h"
#include <stdexcept>
#include <limits>
#include <omp.h>

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, bool keepdim) {
    float total = 0;
    
    #pragma omp parallel for simd reduction(+:total) if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        total += (*a->data)[i];
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
            
            #pragma omp parallel for simd if(a->grad->data->size() > 16384)
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

    std::vector<int> a_strides(a->shape.size());
    int a_current_stride = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) {
        a_strides[i] = a_current_stride;
        a_current_stride *= a->shape[i];
    }

    std::vector<float> out_data(out_size, 0.0f);
    
    #pragma omp parallel for if(a->data->size() > 16384)
    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a_strides[d]) % a->shape[d];
            temp_i %= a_strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        #pragma omp atomic
        out_data[out_idx] += (*a->data)[i];
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, dim, out_strides, a_strides, keepdim]() {
            if (!a->grad) a->zero_grad();            
            
            #pragma omp parallel for if(a->data->size() > 16384)
            for (int i = 0; i < (int)a->data->size(); ++i) {
                int temp_i = i;
                int out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a_strides[d]) % a->shape[d];
                    temp_i %= a_strides[d];
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
    
    #pragma omp parallel for simd reduction(*:total) if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        total *= (*a->data)[i];
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
            
            #pragma omp parallel for reduction(+:zero_count) reduction(*:non_zero_prod) if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                float val = (*a->data)[i];
                if (val == 0.0f) {
                    zero_count++;
                } else {
                    non_zero_prod *= val;
                }
            }
            float upstream_grad = (*out->grad->data)[0];
            float out_val = (*out->data)[0];
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
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

    std::vector<int> a_strides(a->shape.size());
    int a_current_stride = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) {
        a_strides[i] = a_current_stride;
        a_current_stride *= a->shape[i];
    }

    std::vector<float> out_data(out_size, 1.0f);
    
    #pragma omp parallel for if(a->data->size() > 16384)
    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a_strides[d]) % a->shape[d];
            temp_i %= a_strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        #pragma omp atomic
        out_data[out_idx] *= (*a->data)[i];
    }
    auto out = std::make_shared<Tensor>(out_data, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, dim, out_strides, a_strides, keepdim]() {
            if (!a->grad) a->zero_grad();
            int out_size = out->data->size();
            std::vector<int> zero_counts(out_size, 0);
            std::vector<float> non_zero_prods(out_size, 1.0f);
            
            #pragma omp parallel for if(a->data->size() > 16384)
            for (int i = 0; i < (int)a->data->size(); ++i) {
                int temp_i = i;
                int out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a_strides[d]) % a->shape[d];
                    temp_i %= a_strides[d];
                    if (d != dim) {
                        int out_d = (keepdim || d < dim) ? d : d - 1;
                        out_idx += coord * out_strides[out_d];
                    }
                }
                float val = (*a->data)[i];
                if (val == 0.0f) {
                    #pragma omp atomic
                    zero_counts[out_idx]++;
                } else {
                    #pragma omp atomic
                    non_zero_prods[out_idx] *= val;
                }
            }
            
            #pragma omp parallel for if(a->data->size() > 16384)
            for (int i = 0; i < (int)a->data->size(); ++i) {
                int temp_i = i;
                int out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a_strides[d]) % a->shape[d];
                    temp_i %= a_strides[d];
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

    #pragma omp parallel if(a->data->size() > 16384)
    {
        float local_max = -std::numeric_limits<float>::infinity();
        int local_argmax = -1;
        
        #pragma omp for
        for (int i = 0; i < (int)a->data->size(); ++i) {
            if ((*a->data)[i] > local_max) {
                local_max = (*a->data)[i];
                local_argmax = i;
            }
        }
        
        #pragma omp critical
        {
            if (local_max > max_val) {
                max_val = local_max;
                argmax = local_argmax;
            }
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

    std::vector<int> a_strides(a->shape.size());
    int a_current_stride = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) {
        a_strides[i] = a_current_stride;
        a_current_stride *= a->shape[i];
    }

    std::vector<float> out_data(out_size, -std::numeric_limits<float>::infinity());
    std::vector<int> argmax(out_size, -1);

    #pragma omp parallel for if(a->data->size() > 16384)
    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a_strides[d]) % a->shape[d];
            temp_i %= a_strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        
        float val = (*a->data)[i];
        if (val > out_data[out_idx]) {
            #pragma omp critical
            {
                if (val > out_data[out_idx]) {
                    out_data[out_idx] = val;
                    argmax[out_idx] = i; 
                }
            }
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, argmax]() {
            if (!a->grad) a->zero_grad();
            #pragma omp parallel for if(out->data->size() > 16384)
            for (int out_idx = 0; out_idx < (int)out->data->size(); ++out_idx) {
                int winner_idx = argmax[out_idx];
                if (winner_idx != -1) {
                    #pragma omp atomic
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

    #pragma omp parallel if(a->data->size() > 16384)
    {
        float local_min = std::numeric_limits<float>::infinity();
        int local_argmin = -1;
        
        #pragma omp for
        for (int i = 0; i < (int)a->data->size(); ++i) {
            if ((*a->data)[i] < local_min) {
                local_min = (*a->data)[i];
                local_argmin = i;
            }
        }
        
        #pragma omp critical
        {
            if (local_min < min_val) {
                min_val = local_min;
                argmin = local_argmin;
            }
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

    std::vector<int> a_strides(a->shape.size());
    int a_current_stride = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) {
        a_strides[i] = a_current_stride;
        a_current_stride *= a->shape[i];
    }

    std::vector<float> out_data(out_size, std::numeric_limits<float>::infinity());
    std::vector<int> argmin(out_size, -1);

    #pragma omp parallel for if(a->data->size() > 16384)
    for (int i = 0; i < (int)a->data->size(); ++i) {
        int temp_i = i;
        int out_idx = 0;
        for (int d = 0; d < (int)a->shape.size(); ++d) {
            int coord = (temp_i / a_strides[d]) % a->shape[d];
            temp_i %= a_strides[d];
            if (d != dim) {
                int out_d = (keepdim || d < dim) ? d : d - 1;
                out_idx += coord * out_strides[out_d];
            }
        }
        
        float val = (*a->data)[i];
        if (val < out_data[out_idx]) {
            #pragma omp critical
            {
                if (val < out_data[out_idx]) {
                    out_data[out_idx] = val;
                    argmin[out_idx] = i; 
                }
            }
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        out->_backward = [out, a, argmin]() {
            if (!a->grad) a->zero_grad();
            #pragma omp parallel for if(out->data->size() > 16384)
            for (int out_idx = 0; out_idx < (int)out->data->size(); ++out_idx) {
                int winner_idx = argmin[out_idx];
                if (winner_idx != -1) {
                    #pragma omp atomic
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
    if (dim < 0) dim += a->shape.size();
    auto mu = mean(a, dim, true);
    auto diff = sub(a, mu);
    auto variance = pow_scalar(diff, 2);
    int normalizer = (unbiased) ? (a->shape)[dim] - 1 : (a->shape)[dim];
    return div_scalar(sum(variance, dim, keepdim), static_cast<float>(normalizer));
}

std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a, bool keepdim, bool unbiased) {
    return sqrt(var(a, keepdim, unbiased));
}

std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a, int dim, bool keepdim, bool unbiased) {
    return sqrt(var(a, dim, keepdim, unbiased));
}
