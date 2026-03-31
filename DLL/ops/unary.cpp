#include "tensor.h"
#include "ops/binary.h"
#include "ops/reduce.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <omp.h>

#ifdef DLL_GPU_ENABLED
#include "ops/gpu_unary.h"
#endif

std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a, int dim0, int dim1) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return transpose_gpu(a, dim0, dim1);
#endif
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
    
    std::vector<int> a_strides(ndim);
    int current_a = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        a_strides[i] = current_a;
        current_a *= a->shape[i];
    }
    
    std::vector<int> out_strides(ndim);
    int current_out = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        out_strides[i] = current_out;
        current_out *= new_shape[i];
    }
    
    int num_elements = current_a;
    auto out_data = std::make_shared<std::vector<float>>(num_elements);
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for if(num_elements > 16384)
    for (int i = 0; i < num_elements; ++i) {
        int temp = i;
        int a_idx = 0;
        int out_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            int coord = temp / a_strides[d];
            temp %= a_strides[d];
            a_idx += coord * a_strides[d];
            int target_dim = (d == dim0) ? dim1 : (d == dim1 ? dim0 : d);
            out_idx += coord * out_strides[target_dim];
        }
        out_ptr[out_idx] = a_ptr[a_idx];
    }
    
    auto out = std::make_shared<Tensor>(out_data, new_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, dim0, dim1, a_strides, out_strides, num_elements, ndim]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            
            #pragma omp parallel for if(num_elements > 16384)
            for (int i = 0; i < num_elements; ++i) {
                int temp = i;
                int a_idx = 0;
                int out_idx = 0;
                for (int d = 0; d < ndim; ++d) {
                    int coord = temp / a_strides[d];
                    temp %= a_strides[d];
                    a_idx += coord * a_strides[d];
                    int target_dim = (d == dim0) ? dim1 : (d == dim1 ? dim0 : d);
                    out_idx += coord * out_strides[target_dim];
                }
                #pragma omp atomic
                a_grad_ptr[a_idx] += out_grad_ptr[out_idx];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return exp_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::exp(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_ptr[i] * out_grad_ptr[i];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return log_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::log(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] / (a_ptr[i] + 1e-8f); 
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return sqrt_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::sqrt(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] / (2.0f * out_ptr[i] + 1e-8f);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> cbrt(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return cbrt_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::cbrt(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                float y = out_ptr[i];
                a_grad_ptr[i] += out_grad_ptr[i] / (3.0f * y * y + 1e-8f);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> abs(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return abs_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::abs(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
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
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return sin_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::sin(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] * std::cos(a_ptr[i]);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return cos_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::cos(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_ptr = a->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                a_grad_ptr[i] += out_grad_ptr[i] * -std::sin(a_ptr[i]);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return relu_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = a_ptr[i] > 0.0f ? a_ptr[i] : 0.0f;
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
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
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return tanh_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        out_ptr[i] = std::tanh(a_ptr[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
            for (size_t i = 0; i < a->data->size(); ++i) {
                float y = out_ptr[i];
                a_grad_ptr[i] += (1.0f - y * y) * out_grad_ptr[i];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& a) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return sigmoid_gpu(a);
#endif
    auto out_data = std::make_shared<std::vector<float>>(a->data->size());
    float* a_ptr = a->data->data();
    float* out_ptr = out_data->data();
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        float x = a_ptr[i];
        if (x >= 0.0f) {
            float z = std::exp(-x);
            out_ptr[i] = 1.0f / (1.0f + z);
        } else {
            float z = std::exp(x);
            out_ptr[i] = z / (1.0f + z);
        }
    }
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            float* a_grad_ptr = a->grad->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* out_ptr = out->data->data();
            
            #pragma omp parallel for simd if(a->data->size() > 16384)
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
