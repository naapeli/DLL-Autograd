#include "ops/manipulation.h"
#ifdef DLL_GPU_ENABLED
#include "ops/gpu_manipulation.h"
#endif
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <omp.h>

std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("cat requires at least one tensor.");
    }
    
    // Check devices and ensure all on same device as first tensor
    std::string target_device = tensors[0]->device;
    std::vector<std::shared_ptr<Tensor>> processed_tensors;
    processed_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t->device != target_device) {
            processed_tensors.push_back(t->to(target_device));
        } else {
            processed_tensors.push_back(t);
        }
    }

#ifdef DLL_GPU_ENABLED
    if (target_device == "gpu") {
        return cat_gpu(processed_tensors, dim);
    }
#endif

    int ndim = processed_tensors[0]->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("Dimension out of range for cat.");
    
    std::vector<int> out_shape = processed_tensors[0]->shape;
    int target_dim_total = 0;
    for (size_t i = 0; i < processed_tensors.size(); ++i) {
        target_dim_total += processed_tensors[i]->shape[dim];
        for (int d = 0; d < ndim; ++d) {
            if (d != dim && processed_tensors[i]->shape[d] != processed_tensors[0]->shape[d]) {
                throw std::invalid_argument("Tensors must have matching shapes except in the concatenated dimension.");
            }
        }
    }
    out_shape[dim] = target_dim_total;
    
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];

    auto out_data = std::make_shared<std::vector<float>>(outer_size * target_dim_total * inner_size);
    
    std::vector<int> target_dims;
    for (const auto& t : processed_tensors) target_dims.push_back(t->shape[dim]);
    
    std::vector<int> target_offsets(processed_tensors.size() + 1, 0);
    for (size_t i = 0; i < processed_tensors.size(); ++i) target_offsets[i+1] = target_offsets[i] + target_dims[i];
    
    for (size_t i = 0; i < processed_tensors.size(); ++i) {
        processed_tensors[i]->ensure_cpu_data();
        float* src_base = processed_tensors[i]->data->data();
        float* dst_base = out_data->data();
        int t_dim = target_dims[i];
        
        #pragma omp parallel for if(outer_size * t_dim * inner_size > 16384)
        for (int o = 0; o < outer_size; ++o) {
            float* src = src_base + o * t_dim * inner_size;
            float* dst = dst_base + (o * target_dim_total + target_offsets[i]) * inner_size;
            std::memcpy(dst, src, t_dim * inner_size * sizeof(float));
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data, out_shape);
    
    bool any_requires_grad = false;
    for (const auto& t : processed_tensors) if (t->requires_grad) any_requires_grad = true;
    
    if (any_requires_grad) {
        out->requires_grad = true;
        out->_prev = processed_tensors;
        out->_backward = [weak_out = std::weak_ptr<Tensor>(out), processed_tensors, dim, outer_size, target_dim_total, inner_size, target_offsets, target_dims]() {
            auto out = weak_out.lock();
            if (!out || !out->grad) return;
            out->grad->ensure_cpu_data();
            float* grad_out_base = out->grad->data->data();
            
            for (size_t i = 0; i < processed_tensors.size(); ++i) {
                if (processed_tensors[i]->requires_grad) {
                    if (!processed_tensors[i]->grad) processed_tensors[i]->zero_grad();
                    processed_tensors[i]->grad->ensure_cpu_data();
                    float* grad_in_base = processed_tensors[i]->grad->data->data();
                    int t_dim = target_dims[i];
                    
                    #pragma omp parallel for if(outer_size * t_dim * inner_size > 16384)
                    for (int o = 0; o < outer_size; ++o) {
                        float* src = grad_out_base + (o * target_dim_total + target_offsets[i]) * inner_size;
                        float* dst = grad_in_base + o * t_dim * inner_size;
                        for (int j = 0; j < t_dim * inner_size; ++j) {
                            dst[j] += src[j];
                        }
                    }
                }
            }
        };
    }
    
    return out;
}

std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int dim) {
    if (tensors.empty()) throw std::invalid_argument("stack requires at least one tensor.");
    
    std::vector<std::shared_ptr<Tensor>> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(t->unsqueeze(dim));
    }
    return cat(unsqueezed, dim);
}
