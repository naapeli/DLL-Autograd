#ifdef DLL_GPU_ENABLED

#include "ops/gpu_reduce.h"
#include "ops/gpu_binary.h"
#include "ops/gpu_unary.h"
#include "gpu_context.h"
#include <stdexcept>
#include <limits>
#include <cmath>

static size_t round_up(size_t n, size_t wg) {
    return ((n + wg - 1) / wg) * wg;
}

// Helper to upload int vector to GPU
static cl::Buffer upload_ints(const std::vector<int>& v) {
    auto& ctx = GPUContext::instance();
    cl::Buffer buf = ctx.alloc(v.size() * sizeof(int));
    ctx.get_queue().enqueueWriteBuffer(buf, CL_TRUE, 0, v.size() * sizeof(int), v.data());
    return buf;
}

// ============================================================
// SUM
// ============================================================

std::shared_ptr<Tensor> sum_gpu(const std::shared_ptr<Tensor>& a, bool keepdim) {
    // Download to CPU, compute, re-upload result (sum_all is memory-bound anyway)
    // For large tensors, we use a two-pass GPU reduction
    auto& ctx = GPUContext::instance();
    int n = a->numel();

    // Use workgroup reduction kernel
    int wg_size = 256;
    int num_groups = (n + wg_size - 1) / wg_size;

    // First pass: partial sums per workgroup
    auto partials_buf = std::make_shared<GPUBuffer>(num_groups);
    cl::Kernel k = ctx.get_kernel("sum_all");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, partials_buf->get());
    k.setArg(2, cl::Local(wg_size * sizeof(float)));
    k.setArg(3, n);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(num_groups * wg_size), cl::NDRange(wg_size));
    ctx.finish();

    // Read partials back and sum on CPU (num_groups is small)
    std::vector<float> partials(num_groups);
    partials_buf->read_to(partials.data(), num_groups);
    float total = 0;
    for (float v : partials) total += v;

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }

    // Create result on GPU
    auto out_buf = std::make_shared<GPUBuffer>(&total, 1);
    auto out = std::make_shared<Tensor>(out_buf, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, n]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();

            // Read upstream grad (scalar)
            float upstream_grad;
            out->grad->gpu_data->read_to(&upstream_grad, 1);

            cl::Kernel k = ctx.get_kernel("sum_backward_all");
            k.setArg(0, a->grad->gpu_data->get());
            k.setArg(1, upstream_grad);
            k.setArg(2, n);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

std::shared_ptr<Tensor> sum_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    auto& ctx = GPUContext::instance();

    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int cs = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) { out_strides[i] = cs; cs *= out_shape[i]; }

    std::vector<int> a_strides(a->shape.size());
    int acs = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) { a_strides[i] = acs; acs *= a->shape[i]; }

    int a_size = a->numel();

    // Initialize output buffer with zeros
    auto out_buf = std::make_shared<GPUBuffer>(out_size);
    cl::Kernel fill = ctx.get_kernel("fill_kernel");
    fill.setArg(0, out_buf->get());
    fill.setArg(1, 0.0f);
    fill.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));

    // Upload stride info
    cl::Buffer gpu_a_strides = upload_ints(a_strides);
    cl::Buffer gpu_out_strides = upload_ints(out_strides);
    cl::Buffer gpu_a_shape = upload_ints(a->shape);

    int ndim = a->shape.size();
    cl::Kernel k = ctx.get_kernel("sum_dim");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, out_buf->get());
    k.setArg(2, gpu_a_strides);
    k.setArg(3, gpu_out_strides);
    k.setArg(4, gpu_a_shape);
    k.setArg(5, dim);
    k.setArg(6, ndim);
    k.setArg(7, a_size);
    k.setArg(8, (int)keepdim);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(a_size, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;

        out->_backward = [weak_out, a, dim, gpu_out_strides, gpu_a_strides, gpu_a_shape, ndim, a_size, keepdim]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();

            cl::Kernel k = ctx.get_kernel("sum_backward_dim");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->grad->gpu_data->get());
            k.setArg(2, gpu_a_strides);
            k.setArg(3, gpu_out_strides);
            k.setArg(4, gpu_a_shape);
            k.setArg(5, dim);
            k.setArg(6, ndim);
            k.setArg(7, a_size);
            k.setArg(8, (int)keepdim);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(a_size, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

// ============================================================
// MAX
// ============================================================

std::shared_ptr<Tensor> max_gpu(const std::shared_ptr<Tensor>& a, bool keepdim) {
    // Global max — download and compute on CPU, then upload result
    a->ensure_cpu_data();
    int n = a->numel();
    float max_val = -std::numeric_limits<float>::infinity();
    int argmax = -1;
    for (int i = 0; i < n; ++i) {
        if ((*a->data)[i] > max_val) { max_val = (*a->data)[i]; argmax = i; }
    }

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }
    auto out_buf = std::make_shared<GPUBuffer>(&max_val, 1);
    auto out = std::make_shared<Tensor>(out_buf, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, argmax]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            if (!a->grad) a->zero_grad();
            // Download grad, modify, re-upload
            a->grad->ensure_cpu_data();
            out->grad->ensure_cpu_data();
            if (argmax != -1) {
                (*a->grad->data)[argmax] += (*out->grad->data)[0];
            }
            // Re-upload
            auto& ctx = GPUContext::instance();
            int n = a->numel();
            a->grad->gpu_data = std::make_shared<GPUBuffer>(a->grad->data->data(), n);
        };
    }
    return out;
}

std::shared_ptr<Tensor> max_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    auto& ctx = GPUContext::instance();

    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int cs = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) { out_strides[i] = cs; cs *= out_shape[i]; }

    std::vector<int> a_strides(a->shape.size());
    int acs = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) { a_strides[i] = acs; acs *= a->shape[i]; }

    int a_size = a->numel();

    // Init output with -inf
    auto out_buf = std::make_shared<GPUBuffer>(out_size);
    float neg_inf = -std::numeric_limits<float>::infinity();
    cl::Kernel fill = ctx.get_kernel("fill_kernel");
    fill.setArg(0, out_buf->get());
    fill.setArg(1, neg_inf);
    fill.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));

    // Init argmax with -1
    cl::Buffer gpu_argmax = ctx.alloc(out_size * sizeof(int));
    cl::Kernel fill_int = ctx.get_kernel("fill_int_kernel");
    fill_int.setArg(0, gpu_argmax);
    fill_int.setArg(1, -1);
    fill_int.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill_int, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));

    cl::Buffer gpu_a_strides = upload_ints(a_strides);
    cl::Buffer gpu_out_strides = upload_ints(out_strides);
    cl::Buffer gpu_a_shape = upload_ints(a->shape);

    int ndim = a->shape.size();
    cl::Kernel k = ctx.get_kernel("max_dim");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, out_buf->get());
    k.setArg(2, gpu_argmax);
    k.setArg(3, gpu_a_strides);
    k.setArg(4, gpu_out_strides);
    k.setArg(5, gpu_a_shape);
    k.setArg(6, dim);
    k.setArg(7, ndim);
    k.setArg(8, a_size);
    k.setArg(9, (int)keepdim);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(a_size, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, gpu_argmax, out_size]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();

            cl::Kernel k = ctx.get_kernel("max_backward");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->grad->gpu_data->get());
            k.setArg(2, gpu_argmax);
            k.setArg(3, out_size);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

// ============================================================
// MIN
// ============================================================

std::shared_ptr<Tensor> min_gpu(const std::shared_ptr<Tensor>& a, bool keepdim) {
    a->ensure_cpu_data();
    int n = a->numel();
    float min_val = std::numeric_limits<float>::infinity();
    int argmin = -1;
    for (int i = 0; i < n; ++i) {
        if ((*a->data)[i] < min_val) { min_val = (*a->data)[i]; argmin = i; }
    }

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }
    auto out_buf = std::make_shared<GPUBuffer>(&min_val, 1);
    auto out = std::make_shared<Tensor>(out_buf, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, argmin]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            if (!a->grad) a->zero_grad();
            a->grad->ensure_cpu_data();
            out->grad->ensure_cpu_data();
            if (argmin != -1) {
                (*a->grad->data)[argmin] += (*out->grad->data)[0];
            }
            auto& ctx = GPUContext::instance();
            int n = a->numel();
            a->grad->gpu_data = std::make_shared<GPUBuffer>(a->grad->data->data(), n);
        };
    }
    return out;
}

std::shared_ptr<Tensor> min_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    auto& ctx = GPUContext::instance();

    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int cs = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) { out_strides[i] = cs; cs *= out_shape[i]; }

    std::vector<int> a_strides(a->shape.size());
    int acs = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) { a_strides[i] = acs; acs *= a->shape[i]; }

    int a_size = a->numel();

    auto out_buf = std::make_shared<GPUBuffer>(out_size);
    float pos_inf = std::numeric_limits<float>::infinity();
    cl::Kernel fill = ctx.get_kernel("fill_kernel");
    fill.setArg(0, out_buf->get());
    fill.setArg(1, pos_inf);
    fill.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));

    cl::Buffer gpu_argmin = ctx.alloc(out_size * sizeof(int));
    cl::Kernel fill_int = ctx.get_kernel("fill_int_kernel");
    fill_int.setArg(0, gpu_argmin);
    fill_int.setArg(1, -1);
    fill_int.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill_int, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));

    cl::Buffer gpu_a_strides = upload_ints(a_strides);
    cl::Buffer gpu_out_strides = upload_ints(out_strides);
    cl::Buffer gpu_a_shape = upload_ints(a->shape);

    int ndim = a->shape.size();
    cl::Kernel k = ctx.get_kernel("min_dim");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, out_buf->get());
    k.setArg(2, gpu_argmin);
    k.setArg(3, gpu_a_strides);
    k.setArg(4, gpu_out_strides);
    k.setArg(5, gpu_a_shape);
    k.setArg(6, dim);
    k.setArg(7, ndim);
    k.setArg(8, a_size);
    k.setArg(9, (int)keepdim);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(a_size, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        // Reuse max_backward kernel (same logic)
        out->_backward = [weak_out, a, gpu_argmin, out_size]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();

            cl::Kernel k = ctx.get_kernel("max_backward");  // same kernel works for min
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->grad->gpu_data->get());
            k.setArg(2, gpu_argmin);
            k.setArg(3, out_size);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

// ============================================================
// PROD
// ============================================================

std::shared_ptr<Tensor> prod_gpu(const std::shared_ptr<Tensor>& a, bool keepdim) {
    // Global product — compute on CPU, return as GPU tensor
    a->ensure_cpu_data();
    int n = a->numel();
    float total = 1.0f;
    for (int i = 0; i < n; ++i) total *= (*a->data)[i];

    std::vector<int> out_shape;
    if (keepdim) { out_shape.assign(a->shape.size(), 1); } else { out_shape = {1}; }
    auto out_buf = std::make_shared<GPUBuffer>(&total, 1);
    auto out = std::make_shared<Tensor>(out_buf, out_shape);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, total]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            if (!a->grad) a->zero_grad();
            a->grad->ensure_cpu_data();
            out->grad->ensure_cpu_data();
            float upstream_grad = (*out->grad->data)[0];
            int n = a->numel();
            // Count zeros
            int zero_count = 0;
            float non_zero_prod = 1.0f;
            for (int i = 0; i < n; ++i) {
                float val = (*a->data)[i];
                if (val == 0.0f) zero_count++;
                else non_zero_prod *= val;
            }
            for (int i = 0; i < n; ++i) {
                float val = (*a->data)[i];
                float local_grad = 0.0f;
                if (zero_count == 0) local_grad = total / val;
                else if (zero_count == 1 && val == 0.0f) local_grad = non_zero_prod;
                (*a->grad->data)[i] += upstream_grad * local_grad;
            }
            a->grad->gpu_data = std::make_shared<GPUBuffer>(a->grad->data->data(), n);
        };
    }
    return out;
}

std::shared_ptr<Tensor> prod_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim) {
    if (dim < 0) dim += a->shape.size();
    auto& ctx = GPUContext::instance();

    std::vector<int> out_shape;
    for (int i = 0; i < (int)a->shape.size(); ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape = {1};

    int out_size = 1;
    for (int s : out_shape) out_size *= s;
    std::vector<int> out_strides(out_shape.size());
    int cs = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) { out_strides[i] = cs; cs *= out_shape[i]; }

    std::vector<int> a_strides(a->shape.size());
    int acs = 1;
    for (int i = (int)a->shape.size() - 1; i >= 0; --i) { a_strides[i] = acs; acs *= a->shape[i]; }

    int a_size = a->numel();

    // Init with 1.0
    auto out_buf = std::make_shared<GPUBuffer>(out_size);
    cl::Kernel fill = ctx.get_kernel("fill_kernel");
    fill.setArg(0, out_buf->get());
    fill.setArg(1, 1.0f);
    fill.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));

    cl::Buffer gpu_a_strides = upload_ints(a_strides);
    cl::Buffer gpu_out_strides = upload_ints(out_strides);
    cl::Buffer gpu_a_shape = upload_ints(a->shape);

    int ndim = a->shape.size();
    cl::Kernel k = ctx.get_kernel("prod_dim");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, out_buf->get());
    k.setArg(2, gpu_a_strides);
    k.setArg(3, gpu_out_strides);
    k.setArg(4, gpu_a_shape);
    k.setArg(5, dim);
    k.setArg(6, ndim);
    k.setArg(7, a_size);
    k.setArg(8, (int)keepdim);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(a_size, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, out_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, dim, out_size, gpu_a_strides, gpu_out_strides, gpu_a_shape, ndim, a_size, keepdim]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            // Prod backward is complex — fall back to CPU computation
            if (!a->grad) a->zero_grad();
            a->ensure_cpu_data();
            a->grad->ensure_cpu_data();
            out->ensure_cpu_data();
            out->grad->ensure_cpu_data();

            std::vector<int> a_strides_loc(a->shape.size());
            int acs_l = 1;
            for (int i = (int)a->shape.size() - 1; i >= 0; --i) { a_strides_loc[i] = acs_l; acs_l *= a->shape[i]; }

            std::vector<int> out_shape_loc;
            for (int i = 0; i < (int)a->shape.size(); ++i) {
                if (i == dim) { if (keepdim) out_shape_loc.push_back(1); }
                else out_shape_loc.push_back(a->shape[i]);
            }
            if (out_shape_loc.empty()) out_shape_loc = {1};
            std::vector<int> out_strides_loc(out_shape_loc.size());
            int cs_l = 1;
            for (int i = (int)out_shape_loc.size() - 1; i >= 0; --i) { out_strides_loc[i] = cs_l; cs_l *= out_shape_loc[i]; }

            std::vector<int> zero_counts(out_size, 0);
            std::vector<float> non_zero_prods(out_size, 1.0f);

            for (int i = 0; i < a_size; ++i) {
                int temp_i = i, out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a_strides_loc[d]) % a->shape[d];
                    temp_i %= a_strides_loc[d];
                    if (d != dim) {
                        int out_d = (keepdim || d < dim) ? d : d - 1;
                        out_idx += coord * out_strides_loc[out_d];
                    }
                }
                float val = (*a->data)[i];
                if (val == 0.0f) zero_counts[out_idx]++;
                else non_zero_prods[out_idx] *= val;
            }

            for (int i = 0; i < a_size; ++i) {
                int temp_i = i, out_idx = 0;
                for (int d = 0; d < (int)a->shape.size(); ++d) {
                    int coord = (temp_i / a_strides_loc[d]) % a->shape[d];
                    temp_i %= a_strides_loc[d];
                    if (d != dim) {
                        int out_d = (keepdim || d < dim) ? d : d - 1;
                        out_idx += coord * out_strides_loc[out_d];
                    }
                }
                float val = (*a->data)[i];
                float local_grad = 0.0f;
                if (zero_counts[out_idx] == 0) local_grad = (*out->data)[out_idx] / val;
                else if (zero_counts[out_idx] == 1 && val == 0.0f) local_grad = non_zero_prods[out_idx];
                (*a->grad->data)[i] += (*out->grad->data)[out_idx] * local_grad;
            }
            // Re-upload grad to GPU
            auto& ctx = GPUContext::instance();
            a->grad->gpu_data = std::make_shared<GPUBuffer>(a->grad->data->data(), a->numel());
        };
    }
    return out;
}

#endif // DLL_GPU_ENABLED
