#ifdef DLL_GPU_ENABLED
#include "ops/gpu_manipulation.h"
#include "gpu_context.h"
#include <stdexcept>
#include <cmath>

static size_t round_up(size_t n, size_t wg) {
    return ((n + wg - 1) / wg) * wg;
}

std::shared_ptr<Tensor> cat_gpu(const std::vector<std::shared_ptr<Tensor>>& tensors, int dim) {
    auto& ctx = GPUContext::instance();
    int ndim = tensors[0]->shape.size();
    std::vector<int> out_shape = tensors[0]->shape;
    int target_dim_total = 0;
    for (const auto& t : tensors) target_dim_total += t->shape[dim];
    out_shape[dim] = target_dim_total;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];

    auto out_buf = std::make_shared<GPUBuffer>(outer_size * target_dim_total * inner_size);
    cl::Kernel k = ctx.get_kernel("cat_kernel");

    int current_offset = 0;
    std::vector<int> target_dims;
    std::vector<int> target_offsets;

    for (const auto& t : tensors) {
        int t_dim = t->shape[dim];
        target_dims.push_back(t_dim);
        target_offsets.push_back(current_offset);

        k.setArg(0, t->gpu_data->get());
        k.setArg(1, out_buf->get());
        k.setArg(2, outer_size);
        k.setArg(3, t_dim);
        k.setArg(4, target_dim_total);
        k.setArg(5, inner_size);
        k.setArg(6, current_offset);

        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, 
            cl::NDRange(round_up(outer_size, 16), round_up(t_dim * inner_size, 16)), 
            cl::NDRange(16, 16));
        
        current_offset += t_dim;
    }
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, out_shape);
    
    bool any_requires_grad = false;
    for (const auto& t : tensors) if (t->requires_grad) any_requires_grad = true;

    if (any_requires_grad) {
        out->requires_grad = true;
        out->_prev = tensors;
        out->_backward = [weak_out = std::weak_ptr<Tensor>(out), tensors, dim, outer_size, target_dim_total, inner_size, target_offsets, target_dims]() {
            auto out = weak_out.lock();
            if (!out || !out->grad) return;
            auto& ctx = GPUContext::instance();
            cl::Kernel k_back = ctx.get_kernel("cat_backward_kernel");

            for (size_t i = 0; i < tensors.size(); ++i) {
                if (tensors[i]->requires_grad) {
                    if (!tensors[i]->grad) tensors[i]->zero_grad();
                    
                    k_back.setArg(0, tensors[i]->grad->gpu_data->get());
                    k_back.setArg(1, out->grad->gpu_data->get());
                    k_back.setArg(2, outer_size);
                    k_back.setArg(3, target_dims[i]);
                    k_back.setArg(4, target_dim_total);
                    k_back.setArg(5, inner_size);
                    k_back.setArg(6, target_offsets[i]);

                    ctx.get_queue().enqueueNDRangeKernel(k_back, cl::NullRange, 
                        cl::NDRange(round_up(outer_size, 16), round_up(target_dims[i] * inner_size, 16)), 
                        cl::NDRange(16, 16));
                }
            }
            ctx.finish();
        };
    }

    return out;
}

#endif
