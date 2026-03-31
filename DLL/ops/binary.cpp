#include "ops/binary.h"
#include <stdexcept>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <memory>
#include <cblas.h>

#ifdef DLL_GPU_ENABLED
#include "ops/gpu_binary.h"
#endif

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
    
    std::vector<int> contig_strides_a(ndim_a, 1);
    for (int i = ndim_a - 2; i >= 0; --i) contig_strides_a[i] = contig_strides_a[i+1] * a->shape[i+1];

    std::vector<int> contig_strides_b(ndim_b, 1);
    for (int i = ndim_b - 2; i >= 0; --i) contig_strides_b[i] = contig_strides_b[i+1] * b->shape[i+1];

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
        int stride_a = (i < ndim_out - ndim_a) ? 0 : contig_strides_a[i - (ndim_out - ndim_a)];
        int stride_b = (i < ndim_out - ndim_b) ? 0 : contig_strides_b[i - (ndim_out - ndim_b)];
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

std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu() || b->is_gpu()) {
        auto b2 = (a->is_gpu() && !b->is_gpu()) ? ensure_same_device(a, b) : b;
        auto a2 = (!a->is_gpu() && b2->is_gpu()) ? ensure_same_device(b2, a) : a;
        return add_gpu(a2, b2);
    }
#endif
    BroadcastInfo info = setup_broadcast(a, b);
    std::vector<float> result_data(info.out_size);
    
    bool a_contig = (info.strides_a == info.out_strides);
    bool b_contig = (info.strides_b == info.out_strides);
    
    if (a_contig && b_contig) {
        #pragma omp parallel for simd if(info.out_size > 16384)
        for (int i = 0; i < info.out_size; ++i) {
            result_data[i] = (*a->data)[i] + (*b->data)[i];
        }
    } else {
        #pragma omp parallel for if(info.out_size > 16384)
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
    }
    auto out = std::make_shared<Tensor>(result_data, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, b, info, a_contig, b_contig]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            
            if (a_contig && b_contig) {
                #pragma omp parallel for simd if(info.out_size > 16384)
                for (int i = 0; i < info.out_size; ++i) {
                    float grad_out = (*out->grad->data)[i];
                    if (a->requires_grad) (*a->grad->data)[i] += grad_out;
                    if (b->requires_grad) (*b->grad->data)[i] += grad_out;
                }
            } else {
                #pragma omp parallel for if(info.out_size > 16384)
                for (int i = 0; i < info.out_size; ++i) {
                    int temp = i, idx_a = 0, idx_b = 0;
                    for (int d = 0; d < (int)info.out_shape.size(); ++d) {
                        int coord = temp / info.out_strides[d];
                        temp %= info.out_strides[d];
                        idx_a += coord * info.strides_a[d];
                        idx_b += coord * info.strides_b[d];
                    }
                    float grad_out = (*out->grad->data)[i];
                    if (a->requires_grad) {
                        if (a_contig) { (*a->grad->data)[idx_a] += grad_out; }
                        else { 
                            #pragma omp atomic
                            (*a->grad->data)[idx_a] += grad_out; 
                        }
                    }
                    if (b->requires_grad) {
                        if (b_contig) { (*b->grad->data)[idx_b] += grad_out; }
                        else { 
                            #pragma omp atomic
                            (*b->grad->data)[idx_b] += grad_out; 
                        }
                    }
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return add_scalar_gpu(a, scalar);
#endif
    std::vector<float> result_data(a->data->size());
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = (*a->data)[i] + scalar;
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (!a->grad) a->zero_grad();
            
            #pragma omp parallel for simd if(out->grad->data->size() > 16384)
            for (size_t i = 0; i < out->grad->data->size(); ++i) {
                (*a->grad->data)[i] += (*out->grad->data)[i];
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu() || b->is_gpu()) {
        auto b2 = (a->is_gpu() && !b->is_gpu()) ? ensure_same_device(a, b) : b;
        auto a2 = (!a->is_gpu() && b2->is_gpu()) ? ensure_same_device(b2, a) : a;
        return mul_gpu(a2, b2);
    }
#endif
    BroadcastInfo info = setup_broadcast(a, b);
    std::vector<float> result_data(info.out_size);
    
    bool a_contig = (info.strides_a == info.out_strides);
    bool b_contig = (info.strides_b == info.out_strides);
    
    if (a_contig && b_contig) {
        #pragma omp parallel for simd if(info.out_size > 16384)
        for (int i = 0; i < info.out_size; ++i) {
            result_data[i] = (*a->data)[i] * (*b->data)[i];
        }
    } else {
        #pragma omp parallel for if(info.out_size > 16384)
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
    }
    auto out = std::make_shared<Tensor>(result_data, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, b, info, a_contig, b_contig]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            
            if (a_contig && b_contig) {
                #pragma omp parallel for simd if(info.out_size > 16384)
                for (int i = 0; i < info.out_size; ++i) {
                    float grad_out = (*out->grad->data)[i];
                    if (a->requires_grad) (*a->grad->data)[i] += (*b->data)[i] * grad_out;
                    if (b->requires_grad) (*b->grad->data)[i] += (*a->data)[i] * grad_out;
                }
            } else {
                #pragma omp parallel for if(info.out_size > 16384)
                for (int i = 0; i < info.out_size; ++i) {
                    int temp = i, idx_a = 0, idx_b = 0;
                    for (int d = 0; d < (int)info.out_shape.size(); ++d) {
                        int coord = temp / info.out_strides[d];
                        temp %= info.out_strides[d];
                        idx_a += coord * info.strides_a[d];
                        idx_b += coord * info.strides_b[d];
                    }
                    float grad_out = (*out->grad->data)[i];
                    if (a->requires_grad) {
                        float val = (*b->data)[idx_b] * grad_out;
                        if (a_contig) { (*a->grad->data)[idx_a] += val; }
                        else { 
                            #pragma omp atomic
                            (*a->grad->data)[idx_a] += val; 
                        }
                    }
                    if (b->requires_grad) {
                        float val = (*a->data)[idx_a] * grad_out;
                        if (b_contig) { (*b->grad->data)[idx_b] += val; }
                        else { 
                            #pragma omp atomic
                            (*b->grad->data)[idx_b] += val; 
                        }
                    }
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return mul_scalar_gpu(a, scalar);
#endif
    std::vector<float> result_data(a->data->size());
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = (*a->data)[i] * scalar;
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, scalar]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                
                #pragma omp parallel for simd if(out->grad->data->size() > 16384)
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += scalar * (*out->grad->data)[i];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return add(a, mul_scalar(b, -1.0f));
}

std::shared_ptr<Tensor> sub_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return add_scalar(a, -scalar);
}

std::shared_ptr<Tensor> rsub_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return add_scalar(mul_scalar(a, -1.0f), scalar);
}

std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu() || b->is_gpu()) {
        auto b2 = (a->is_gpu() && !b->is_gpu()) ? ensure_same_device(a, b) : b;
        auto a2 = (!a->is_gpu() && b2->is_gpu()) ? ensure_same_device(b2, a) : a;
        return pow_gpu(a2, b2);
    }
#endif
    BroadcastInfo info = setup_broadcast(a, b);
    std::vector<float> result_data(info.out_size);
    
    bool a_contig = (info.strides_a == info.out_strides);
    bool b_contig = (info.strides_b == info.out_strides);

    if (a_contig && b_contig) {
        #pragma omp parallel for simd if(info.out_size > 16384)
        for (int i = 0; i < info.out_size; ++i) {
            result_data[i] = std::pow((*a->data)[i], (*b->data)[i]);
        }
    } else {
        #pragma omp parallel for if(info.out_size > 16384)
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
    }
    auto out = std::make_shared<Tensor>(result_data, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, b, info, a_contig, b_contig]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            
            if (a_contig && b_contig) {
                #pragma omp parallel for simd if(info.out_size > 16384)
                for (int i = 0; i < info.out_size; ++i) {
                    float grad_out = (*out->grad->data)[i];
                    float val_a = (*a->data)[i];
                    float val_b = (*b->data)[i];
                    if (a->requires_grad) {
                        (*a->grad->data)[i] += val_b * std::pow(val_a, val_b - 1.0f) * grad_out;
                    }
                    if (b->requires_grad) {
                        (*b->grad->data)[i] += (*out->data)[i] * std::log(val_a + 1e-8f) * grad_out;
                    }
                }
            } else {
                #pragma omp parallel for if(info.out_size > 16384)
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
                        float val = val_b * std::pow(val_a, val_b - 1.0f) * grad_out;
                        if (a_contig) { (*a->grad->data)[idx_a] += val; }
                        else { 
                            #pragma omp atomic
                            (*a->grad->data)[idx_a] += val; 
                        }
                    }
                    if (b->requires_grad) {
                        float val = (*out->data)[i] * std::log(val_a + 1e-8f) * grad_out;
                        if (b_contig) { (*b->grad->data)[idx_b] += val; }
                        else { 
                            #pragma omp atomic
                            (*b->grad->data)[idx_b] += val; 
                        }
                    }
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> pow_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return pow_scalar_gpu(a, scalar);
#endif
    std::vector<float> result_data(a->data->size());
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = std::pow((*a->data)[i], scalar);
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, scalar]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                
                #pragma omp parallel for simd if(out->grad->data->size() > 16384)
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += scalar * std::pow((*a->data)[i], scalar - 1) * (*out->grad->data)[i];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> rpow_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu()) return rpow_scalar_gpu(a, scalar);
#endif
    std::vector<float> result_data(a->data->size());
    
    #pragma omp parallel for simd if(a->data->size() > 16384)
    for (size_t i = 0; i < a->data->size(); ++i) {
        result_data[i] = std::pow(scalar, (*a->data)[i]);
    }
    auto out = std::make_shared<Tensor>(result_data, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, scalar]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                
                #pragma omp parallel for simd if(out->grad->data->size() > 16384)
                for (size_t i = 0; i < out->grad->data->size(); ++i) {
                    (*a->grad->data)[i] += (*out->data)[i] * std::log(scalar + 1e-8f) * (*out->grad->data)[i];
                }
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return mul(a, pow_scalar(b, -1.0f));
}

std::shared_ptr<Tensor> div_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return mul_scalar(a, 1.0f / scalar);
}

std::shared_ptr<Tensor> rdiv_scalar(const std::shared_ptr<Tensor>& a, float scalar) {
    return mul_scalar(pow_scalar(a, -1.0f), scalar);
}

std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
#ifdef DLL_GPU_ENABLED
    if (a->is_gpu() || b->is_gpu()) {
        auto b2 = (a->is_gpu() && !b->is_gpu()) ? ensure_same_device(a, b) : b;
        auto a2 = (!a->is_gpu() && b2->is_gpu()) ? ensure_same_device(b2, a) : a;
        return matmul_gpu(a2, b2);
    }
#endif
    int ndim_a = a->shape.size();
    int ndim_b = b->shape.size();
    if (ndim_a < 2 || ndim_b < 2) {
        throw std::invalid_argument("Matmul requires both tensors to have at least 2 dimensions.");
    }
    int M = a->shape[ndim_a - 2];
    int K_a = a->shape[ndim_a - 1];
    int K_b = b->shape[ndim_b - 2];
    int N = b->shape[ndim_b - 1];
    if (K_a != K_b) {
        throw std::invalid_argument("Matmul inner dimensions must match.");
    }
    int K = K_a;

    std::vector<int> contig_strides_a(ndim_a, 1);
    for (int i = ndim_a - 2; i >= 0; --i) contig_strides_a[i] = contig_strides_a[i+1] * a->shape[i+1];

    std::vector<int> contig_strides_b(ndim_b, 1);
    for (int i = ndim_b - 2; i >= 0; --i) contig_strides_b[i] = contig_strides_b[i+1] * b->shape[i+1];

    int batch_ndim_a = ndim_a - 2;
    int batch_ndim_b = ndim_b - 2;
    int batch_ndim = std::max(batch_ndim_a, batch_ndim_b);
    std::vector<int> out_batch_shape(batch_ndim);
    std::vector<int> a_batch_strides(batch_ndim, 0);
    std::vector<int> b_batch_strides(batch_ndim, 0);
    std::vector<int> out_batch_strides(batch_ndim, 0);
    
    for (int i = 0; i < batch_ndim; ++i) {
        int dim_a = (i < batch_ndim - batch_ndim_a) ? 1 : a->shape[i - (batch_ndim - batch_ndim_a)];
        int dim_b = (i < batch_ndim - batch_ndim_b) ? 1 : b->shape[i - (batch_ndim - batch_ndim_b)];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument("Matmul batch shapes are not broadcastable.");
        }
        out_batch_shape[i] = std::max(dim_a, dim_b);
        int stride_a = (i < batch_ndim - batch_ndim_a || dim_a == 1) ? 0 : contig_strides_a[i - (batch_ndim - batch_ndim_a)];
        int stride_b = (i < batch_ndim - batch_ndim_b || dim_b == 1) ? 0 : contig_strides_b[i - (batch_ndim - batch_ndim_b)];
        a_batch_strides[i] = stride_a;
        b_batch_strides[i] = stride_b;
    }
    
    int num_matrices = 1;
    for (int i = batch_ndim - 1; i >= 0; --i) {
        out_batch_strides[i] = num_matrices;
        num_matrices *= out_batch_shape[i];
    }
    
    std::vector<int> out_shape = out_batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    int out_size = num_matrices * M * N;
    std::vector<float> out_data(out_size, 0.0f);
    
    float* a_ptr = a->data->data();
    float* b_ptr = b->data->data();
    float* out_ptr = out_data.data();
    
    for (int b_idx = 0; b_idx < num_matrices; ++b_idx) {
        int temp = b_idx;
        int offset_a = 0;
        int offset_b = 0;
        for (int d = 0; d < batch_ndim; ++d) {
            int coord = temp / out_batch_strides[d];
            temp %= out_batch_strides[d];
            offset_a += coord * a_batch_strides[d];
            offset_b += coord * b_batch_strides[d];
        }
        int offset_out = b_idx * M * N;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    a_ptr + offset_a, K,
                    b_ptr + offset_b, N,
                    0.0f,
                    out_ptr + offset_out, N);
    }
    
    auto out = std::make_shared<Tensor>(out_data, out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, b, num_matrices, M, K, N, batch_ndim, out_batch_strides, a_batch_strides, b_batch_strides]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();
            float* a_ptr = a->data->data();
            float* b_ptr = b->data->data();
            float* out_grad_ptr = out->grad->data->data();
            float* a_grad_ptr = a->requires_grad ? a->grad->data->data() : nullptr;
            float* b_grad_ptr = b->requires_grad ? b->grad->data->data() : nullptr;
            for (int b_idx = 0; b_idx < num_matrices; ++b_idx) {
                int temp = b_idx;
                int offset_a = 0, offset_b = 0;
                for (int d = 0; d < batch_ndim; ++d) {
                    int coord = temp / out_batch_strides[d];
                    temp %= out_batch_strides[d];
                    offset_a += coord * a_batch_strides[d];
                    offset_b += coord * b_batch_strides[d];
                }
                int offset_out = b_idx * M * N;
                if (a->requires_grad) {
                    // grad_a += grad_out @ b^T
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                M, K, N,
                                1.0f,
                                out_grad_ptr + offset_out, N,
                                b_ptr + offset_b, N,
                                1.0f,
                                a_grad_ptr + offset_a, K);
                }
                if (b->requires_grad) {
                    // grad_b += a^T @ grad_out
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                K, N, M,
                                1.0f,
                                a_ptr + offset_a, K,
                                out_grad_ptr + offset_out, N,
                                1.0f,
                                b_grad_ptr + offset_b, N);
                }
            }
        };
    }
    return out;
}
