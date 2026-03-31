#ifdef DLL_GPU_ENABLED

#include "ops/gpu_unary.h"
#include "ops/gpu_binary.h"
#include "ops/gpu_reduce.h"
#include "gpu_context.h"
#include <stdexcept>

static size_t round_up(size_t n, size_t wg) {
    return ((n + wg - 1) / wg) * wg;
}

// Generic template for simple unary ops: forward + backward
// forward_kernel takes (a, out, n)
// backward_kernel takes (out_grad, extra_data, a_grad, n) where extra_data is out_data or a_data
static std::shared_ptr<Tensor> unary_op_gpu(const std::shared_ptr<Tensor>& a,
                                              const std::string& fwd_name,
                                              const std::string& bwd_name,
                                              bool bwd_uses_output) {
    auto& ctx = GPUContext::instance();
    int n = a->numel();
    auto out_buf = std::make_shared<GPUBuffer>(n);

    cl::Kernel k = ctx.get_kernel(fwd_name);
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, out_buf->get());
    k.setArg(2, n);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, bwd_name, bwd_uses_output, n]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();

            cl::Kernel k = ctx.get_kernel(bwd_name);
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, bwd_uses_output ? out->gpu_data->get() : a->gpu_data->get());
            k.setArg(2, a->grad->gpu_data->get());
            k.setArg(3, n);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

std::shared_ptr<Tensor> exp_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "exp_kernel", "exp_backward", true);
}

std::shared_ptr<Tensor> log_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "log_kernel", "log_backward", false);
}

std::shared_ptr<Tensor> sqrt_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "sqrt_kernel", "sqrt_backward", true);
}

std::shared_ptr<Tensor> cbrt_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "cbrt_kernel", "cbrt_backward", true);
}

std::shared_ptr<Tensor> abs_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "abs_kernel", "abs_backward", false);
}

std::shared_ptr<Tensor> sin_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "sin_kernel", "sin_backward", false);
}

std::shared_ptr<Tensor> cos_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "cos_kernel", "cos_backward", false);
}

std::shared_ptr<Tensor> relu_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "relu_kernel", "relu_backward", true);
}

std::shared_ptr<Tensor> tanh_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "tanh_kernel", "tanh_backward", true);
}

std::shared_ptr<Tensor> sigmoid_gpu(const std::shared_ptr<Tensor>& a) {
    return unary_op_gpu(a, "sigmoid_kernel", "sigmoid_backward", true);
}

std::shared_ptr<Tensor> transpose_gpu(const std::shared_ptr<Tensor>& a, int dim0, int dim1) {
    int ndim = a->shape.size();
    if (ndim < 2) throw std::runtime_error("Transpose requires at least 2 dimensions.");
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
        throw std::out_of_range("Dimension out of range for transpose.");

    std::vector<int> new_shape = a->shape;
    std::swap(new_shape[dim0], new_shape[dim1]);

    std::vector<int> a_strides(ndim);
    int current_a = 1;
    for (int i = ndim - 1; i >= 0; --i) { a_strides[i] = current_a; current_a *= a->shape[i]; }

    std::vector<int> out_strides(ndim);
    int current_out = 1;
    for (int i = ndim - 1; i >= 0; --i) { out_strides[i] = current_out; current_out *= new_shape[i]; }

    int num_elements = current_a;
    auto& ctx = GPUContext::instance();
    auto out_buf = std::make_shared<GPUBuffer>(num_elements);

    // Upload stride/shape info
    cl::Buffer gpu_a_strides = ctx.alloc(ndim * sizeof(int));
    cl::Buffer gpu_out_strides = ctx.alloc(ndim * sizeof(int));
    cl::Buffer gpu_shape = ctx.alloc(ndim * sizeof(int));
    ctx.get_queue().enqueueWriteBuffer(gpu_a_strides, CL_TRUE, 0, ndim * sizeof(int), a_strides.data());
    ctx.get_queue().enqueueWriteBuffer(gpu_out_strides, CL_TRUE, 0, ndim * sizeof(int), out_strides.data());
    ctx.get_queue().enqueueWriteBuffer(gpu_shape, CL_TRUE, 0, ndim * sizeof(int), a->shape.data());

    cl::Kernel k = ctx.get_kernel("transpose_kernel");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, out_buf->get());
    k.setArg(2, gpu_a_strides);
    k.setArg(3, gpu_out_strides);
    k.setArg(4, gpu_shape);
    k.setArg(5, dim0);
    k.setArg(6, dim1);
    k.setArg(7, ndim);
    k.setArg(8, num_elements);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(num_elements, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, new_shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, dim0, dim1, gpu_a_strides, gpu_out_strides, num_elements, ndim]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();

            cl::Kernel k = ctx.get_kernel("transpose_backward");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->grad->gpu_data->get());
            k.setArg(2, gpu_a_strides);
            k.setArg(3, gpu_out_strides);
            k.setArg(4, dim0);
            k.setArg(5, dim1);
            k.setArg(6, ndim);
            k.setArg(7, num_elements);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(num_elements, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

#endif // DLL_GPU_ENABLED
