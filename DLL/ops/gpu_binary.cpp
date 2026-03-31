#ifdef DLL_GPU_ENABLED

#include "ops/gpu_binary.h"
#include "gpu_context.h"
#include <clblast.h>
#include <stdexcept>
#include <algorithm>

// Helper: compute broadcast info and upload stride buffers to GPU
struct GPUBroadcastInfo {
    std::vector<int> out_shape;
    std::vector<int> out_strides;
    std::vector<int> strides_a;
    std::vector<int> strides_b;
    int out_size;
    bool a_contig;
    bool b_contig;

    // GPU buffers for stride info
    cl::Buffer gpu_out_strides;
    cl::Buffer gpu_strides_a;
    cl::Buffer gpu_strides_b;
};

static size_t round_up(size_t n, size_t wg) {
    return ((n + wg - 1) / wg) * wg;
}

static GPUBroadcastInfo setup_gpu_broadcast(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    int ndim_a = a->shape.size();
    int ndim_b = b->shape.size();
    int ndim_out = std::max(ndim_a, ndim_b);

    std::vector<int> contig_strides_a(ndim_a, 1);
    for (int i = ndim_a - 2; i >= 0; --i) contig_strides_a[i] = contig_strides_a[i+1] * a->shape[i+1];

    std::vector<int> contig_strides_b(ndim_b, 1);
    for (int i = ndim_b - 2; i >= 0; --i) contig_strides_b[i] = contig_strides_b[i+1] * b->shape[i+1];

    GPUBroadcastInfo info;
    info.out_shape.resize(ndim_out);
    info.out_strides.resize(ndim_out);
    info.strides_a.resize(ndim_out, 0);
    info.strides_b.resize(ndim_out, 0);

    for (int i = 0; i < ndim_out; ++i) {
        int dim_a = (i < ndim_out - ndim_a) ? 1 : a->shape[i - (ndim_out - ndim_a)];
        int dim_b = (i < ndim_out - ndim_b) ? 1 : b->shape[i - (ndim_out - ndim_b)];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            throw std::invalid_argument("Shapes are not broadcastable.");
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
    info.a_contig = (info.strides_a == info.out_strides);
    info.b_contig = (info.strides_b == info.out_strides);

    // Upload stride info to GPU
    auto& ctx = GPUContext::instance();
    info.gpu_out_strides = ctx.alloc(ndim_out * sizeof(int));
    info.gpu_strides_a = ctx.alloc(ndim_out * sizeof(int));
    info.gpu_strides_b = ctx.alloc(ndim_out * sizeof(int));
    ctx.get_queue().enqueueWriteBuffer(info.gpu_out_strides, CL_TRUE, 0, ndim_out * sizeof(int), info.out_strides.data());
    ctx.get_queue().enqueueWriteBuffer(info.gpu_strides_a, CL_TRUE, 0, ndim_out * sizeof(int), info.strides_a.data());
    ctx.get_queue().enqueueWriteBuffer(info.gpu_strides_b, CL_TRUE, 0, ndim_out * sizeof(int), info.strides_b.data());

    return info;
}

// ============================================================
// ADD
// ============================================================

std::shared_ptr<Tensor> add_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    auto& ctx = GPUContext::instance();
    GPUBroadcastInfo info = setup_gpu_broadcast(a, b);

    auto out_buf = std::make_shared<GPUBuffer>(info.out_size);

    if (info.a_contig && info.b_contig) {
        cl::Kernel k = ctx.get_kernel("add_contiguous");
        k.setArg(0, a->gpu_data->get());
        k.setArg(1, b->gpu_data->get());
        k.setArg(2, out_buf->get());
        k.setArg(3, info.out_size);
        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(info.out_size, 256)), cl::NDRange(256));
    } else {
        int ndim = info.out_shape.size();
        cl::Kernel k = ctx.get_kernel("add_broadcast");
        k.setArg(0, a->gpu_data->get());
        k.setArg(1, b->gpu_data->get());
        k.setArg(2, out_buf->get());
        k.setArg(3, info.gpu_out_strides);
        k.setArg(4, info.gpu_strides_a);
        k.setArg(5, info.gpu_strides_b);
        k.setArg(6, ndim);
        k.setArg(7, info.out_size);
        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(info.out_size, 256)), cl::NDRange(256));
    }
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};

        std::weak_ptr<Tensor> weak_out = out;
        bool a_contig = info.a_contig, b_contig = info.b_contig;
        auto gpu_out_strides = info.gpu_out_strides;
        auto gpu_strides_a = info.gpu_strides_a;
        auto gpu_strides_b = info.gpu_strides_b;
        int ndim = info.out_shape.size();
        int out_size = info.out_size;

        out->_backward = [weak_out, a, b, a_contig, b_contig, gpu_out_strides, gpu_strides_a, gpu_strides_b, ndim, out_size]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();

            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();

            if (a_contig && b_contig) {
                cl::Kernel k = ctx.get_kernel("add_backward_contiguous");
                k.setArg(0, out->grad->gpu_data->get());
                k.setArg(1, a->requires_grad ? a->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(2, b->requires_grad ? b->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(3, out_size);
                k.setArg(4, (int)a->requires_grad);
                k.setArg(5, (int)b->requires_grad);
                ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            } else {
                cl::Kernel k = ctx.get_kernel("add_backward_broadcast");
                k.setArg(0, out->grad->gpu_data->get());
                k.setArg(1, a->requires_grad ? a->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(2, b->requires_grad ? b->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(3, gpu_out_strides);
                k.setArg(4, gpu_strides_a);
                k.setArg(5, gpu_strides_b);
                k.setArg(6, ndim);
                k.setArg(7, out_size);
                k.setArg(8, (int)a->requires_grad);
                k.setArg(9, (int)b->requires_grad);
                ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            }
            ctx.finish();
        };
    }
    return out;
}

std::shared_ptr<Tensor> add_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar) {
    auto& ctx = GPUContext::instance();
    int n = a->numel();
    auto out_buf = std::make_shared<GPUBuffer>(n);

    cl::Kernel k = ctx.get_kernel("add_scalar_kernel");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, scalar);
    k.setArg(2, out_buf->get());
    k.setArg(3, n);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, n]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();
            cl::Kernel k = ctx.get_kernel("scalar_backward");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->grad->gpu_data->get());
            k.setArg(2, n);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

// ============================================================
// MUL
// ============================================================

std::shared_ptr<Tensor> mul_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    auto& ctx = GPUContext::instance();
    GPUBroadcastInfo info = setup_gpu_broadcast(a, b);
    auto out_buf = std::make_shared<GPUBuffer>(info.out_size);

    if (info.a_contig && info.b_contig) {
        cl::Kernel k = ctx.get_kernel("mul_contiguous");
        k.setArg(0, a->gpu_data->get());
        k.setArg(1, b->gpu_data->get());
        k.setArg(2, out_buf->get());
        k.setArg(3, info.out_size);
        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(info.out_size, 256)), cl::NDRange(256));
    } else {
        int ndim = info.out_shape.size();
        cl::Kernel k = ctx.get_kernel("mul_broadcast");
        k.setArg(0, a->gpu_data->get());
        k.setArg(1, b->gpu_data->get());
        k.setArg(2, out_buf->get());
        k.setArg(3, info.gpu_out_strides);
        k.setArg(4, info.gpu_strides_a);
        k.setArg(5, info.gpu_strides_b);
        k.setArg(6, ndim);
        k.setArg(7, info.out_size);
        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(info.out_size, 256)), cl::NDRange(256));
    }
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        std::weak_ptr<Tensor> weak_out = out;
        bool a_contig = info.a_contig, b_contig = info.b_contig;
        auto gpu_out_strides = info.gpu_out_strides;
        auto gpu_strides_a = info.gpu_strides_a;
        auto gpu_strides_b = info.gpu_strides_b;
        int ndim = info.out_shape.size();
        int out_size = info.out_size;

        out->_backward = [weak_out, a, b, a_contig, b_contig, gpu_out_strides, gpu_strides_a, gpu_strides_b, ndim, out_size]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();

            if (a_contig && b_contig) {
                cl::Kernel k = ctx.get_kernel("mul_backward_contiguous");
                k.setArg(0, out->grad->gpu_data->get());
                k.setArg(1, a->gpu_data->get());
                k.setArg(2, b->gpu_data->get());
                k.setArg(3, a->requires_grad ? a->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(4, b->requires_grad ? b->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(5, out_size);
                k.setArg(6, (int)a->requires_grad);
                k.setArg(7, (int)b->requires_grad);
                ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            } else {
                cl::Kernel k = ctx.get_kernel("mul_backward_broadcast");
                k.setArg(0, out->grad->gpu_data->get());
                k.setArg(1, a->gpu_data->get());
                k.setArg(2, b->gpu_data->get());
                k.setArg(3, a->requires_grad ? a->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(4, b->requires_grad ? b->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(5, gpu_out_strides);
                k.setArg(6, gpu_strides_a);
                k.setArg(7, gpu_strides_b);
                k.setArg(8, ndim);
                k.setArg(9, out_size);
                k.setArg(10, (int)a->requires_grad);
                k.setArg(11, (int)b->requires_grad);
                ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            }
            ctx.finish();
        };
    }
    return out;
}

std::shared_ptr<Tensor> mul_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar) {
    auto& ctx = GPUContext::instance();
    int n = a->numel();
    auto out_buf = std::make_shared<GPUBuffer>(n);

    cl::Kernel k = ctx.get_kernel("mul_scalar_kernel");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, scalar);
    k.setArg(2, out_buf->get());
    k.setArg(3, n);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, scalar, n]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();
            cl::Kernel k = ctx.get_kernel("mul_scalar_backward");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->grad->gpu_data->get());
            k.setArg(2, scalar);
            k.setArg(3, n);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

// ============================================================
// POW
// ============================================================

std::shared_ptr<Tensor> pow_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    auto& ctx = GPUContext::instance();
    GPUBroadcastInfo info = setup_gpu_broadcast(a, b);
    auto out_buf = std::make_shared<GPUBuffer>(info.out_size);

    if (info.a_contig && info.b_contig) {
        cl::Kernel k = ctx.get_kernel("pow_contiguous");
        k.setArg(0, a->gpu_data->get());
        k.setArg(1, b->gpu_data->get());
        k.setArg(2, out_buf->get());
        k.setArg(3, info.out_size);
        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(info.out_size, 256)), cl::NDRange(256));
    } else {
        int ndim = info.out_shape.size();
        cl::Kernel k = ctx.get_kernel("pow_broadcast");
        k.setArg(0, a->gpu_data->get());
        k.setArg(1, b->gpu_data->get());
        k.setArg(2, out_buf->get());
        k.setArg(3, info.gpu_out_strides);
        k.setArg(4, info.gpu_strides_a);
        k.setArg(5, info.gpu_strides_b);
        k.setArg(6, ndim);
        k.setArg(7, info.out_size);
        ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(info.out_size, 256)), cl::NDRange(256));
    }
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, info.out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};
        std::weak_ptr<Tensor> weak_out = out;
        bool a_contig = info.a_contig, b_contig = info.b_contig;
        auto gpu_out_strides = info.gpu_out_strides;
        auto gpu_strides_a = info.gpu_strides_a;
        auto gpu_strides_b = info.gpu_strides_b;
        int ndim = info.out_shape.size();
        int out_size = info.out_size;

        out->_backward = [weak_out, a, b, a_contig, b_contig, gpu_out_strides, gpu_strides_a, gpu_strides_b, ndim, out_size]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();

            if (a_contig && b_contig) {
                cl::Kernel k = ctx.get_kernel("pow_backward_contiguous");
                k.setArg(0, out->grad->gpu_data->get());
                k.setArg(1, a->gpu_data->get());
                k.setArg(2, b->gpu_data->get());
                k.setArg(3, out->gpu_data->get());
                k.setArg(4, a->requires_grad ? a->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(5, b->requires_grad ? b->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(6, out_size);
                k.setArg(7, (int)a->requires_grad);
                k.setArg(8, (int)b->requires_grad);
                ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            } else {
                cl::Kernel k = ctx.get_kernel("pow_backward_broadcast");
                k.setArg(0, out->grad->gpu_data->get());
                k.setArg(1, a->gpu_data->get());
                k.setArg(2, b->gpu_data->get());
                k.setArg(3, out->gpu_data->get());
                k.setArg(4, a->requires_grad ? a->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(5, b->requires_grad ? b->grad->gpu_data->get() : out->grad->gpu_data->get());
                k.setArg(6, gpu_out_strides);
                k.setArg(7, gpu_strides_a);
                k.setArg(8, gpu_strides_b);
                k.setArg(9, ndim);
                k.setArg(10, out_size);
                k.setArg(11, (int)a->requires_grad);
                k.setArg(12, (int)b->requires_grad);
                ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
            }
            ctx.finish();
        };
    }
    return out;
}

std::shared_ptr<Tensor> pow_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar) {
    auto& ctx = GPUContext::instance();
    int n = a->numel();
    auto out_buf = std::make_shared<GPUBuffer>(n);

    cl::Kernel k = ctx.get_kernel("pow_scalar_kernel");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, scalar);
    k.setArg(2, out_buf->get());
    k.setArg(3, n);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, scalar, n]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();
            cl::Kernel k = ctx.get_kernel("pow_scalar_backward");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, a->gpu_data->get());
            k.setArg(2, a->grad->gpu_data->get());
            k.setArg(3, scalar);
            k.setArg(4, n);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

std::shared_ptr<Tensor> rpow_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar) {
    auto& ctx = GPUContext::instance();
    int n = a->numel();
    auto out_buf = std::make_shared<GPUBuffer>(n);

    cl::Kernel k = ctx.get_kernel("rpow_scalar_kernel");
    k.setArg(0, a->gpu_data->get());
    k.setArg(1, scalar);
    k.setArg(2, out_buf->get());
    k.setArg(3, n);
    ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, a->shape);
    if (a->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a};
        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, scalar, n]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();
            if (!a->grad) a->zero_grad();
            cl::Kernel k = ctx.get_kernel("rpow_scalar_backward");
            k.setArg(0, out->grad->gpu_data->get());
            k.setArg(1, out->gpu_data->get());
            k.setArg(2, a->grad->gpu_data->get());
            k.setArg(3, scalar);
            k.setArg(4, n);
            ctx.get_queue().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(round_up(n, 256)), cl::NDRange(256));
            ctx.finish();
        };
    }
    return out;
}

// ============================================================
// MATMUL via CLBlast
// ============================================================

std::shared_ptr<Tensor> matmul_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    int ndim_a = a->shape.size();
    int ndim_b = b->shape.size();
    if (ndim_a < 2 || ndim_b < 2)
        throw std::invalid_argument("Matmul requires both tensors to have at least 2 dimensions.");

    int M = a->shape[ndim_a - 2];
    int K_a = a->shape[ndim_a - 1];
    int K_b = b->shape[ndim_b - 2];
    int N = b->shape[ndim_b - 1];
    if (K_a != K_b)
        throw std::invalid_argument("Matmul inner dimensions must match.");
    int K = K_a;

    // Compute batch dimensions
    int batch_ndim_a = ndim_a - 2;
    int batch_ndim_b = ndim_b - 2;
    int batch_ndim = std::max(batch_ndim_a, batch_ndim_b);

    std::vector<int> contig_strides_a(ndim_a, 1);
    for (int i = ndim_a - 2; i >= 0; --i) contig_strides_a[i] = contig_strides_a[i+1] * a->shape[i+1];
    std::vector<int> contig_strides_b(ndim_b, 1);
    for (int i = ndim_b - 2; i >= 0; --i) contig_strides_b[i] = contig_strides_b[i+1] * b->shape[i+1];

    std::vector<int> out_batch_shape(batch_ndim);
    std::vector<int> a_batch_strides(batch_ndim, 0);
    std::vector<int> b_batch_strides(batch_ndim, 0);
    std::vector<int> out_batch_strides(batch_ndim, 0);

    for (int i = 0; i < batch_ndim; ++i) {
        int dim_a = (i < batch_ndim - batch_ndim_a) ? 1 : a->shape[i - (batch_ndim - batch_ndim_a)];
        int dim_b = (i < batch_ndim - batch_ndim_b) ? 1 : b->shape[i - (batch_ndim - batch_ndim_b)];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            throw std::invalid_argument("Matmul batch shapes are not broadcastable.");
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

    auto out_buf = std::make_shared<GPUBuffer>(out_size);
    auto& ctx = GPUContext::instance();

    // Fill output with zeros
    cl::Kernel fill = ctx.get_kernel("fill_kernel");
    fill.setArg(0, out_buf->get());
    fill.setArg(1, 0.0f);
    fill.setArg(2, out_size);
    ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange(round_up(out_size, 256)), cl::NDRange(256));
    ctx.finish();

    // Run CLBlast GEMM for each batch
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

        auto status = clblast::Gemm(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo, clblast::Transpose::kNo,
            M, N, K,
            1.0f,
            a->gpu_data->get()(), offset_a, K,
            b->gpu_data->get()(), offset_b, N,
            0.0f,
            out_buf->get()(), offset_out, N,
            &ctx.get_queue()());
        if (status != clblast::StatusCode::kSuccess) {
            throw std::runtime_error("CLBlast GEMM failed with status: " + std::to_string(static_cast<int>(status)));
        }
    }
    ctx.finish();

    auto out = std::make_shared<Tensor>(out_buf, out_shape);
    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->_prev = {a, b};

        std::weak_ptr<Tensor> weak_out = out;
        out->_backward = [weak_out, a, b, num_matrices, M, K, N, batch_ndim, out_batch_strides, a_batch_strides, b_batch_strides]() {
            auto out = weak_out.lock();
            if (!out) throw std::runtime_error("Autograd engine error: Node destroyed prematurely.");
            auto& ctx = GPUContext::instance();

            if (a->requires_grad && !a->grad) a->zero_grad();
            if (b->requires_grad && !b->grad) b->zero_grad();

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
                    clblast::Gemm(
                        clblast::Layout::kRowMajor,
                        clblast::Transpose::kNo, clblast::Transpose::kYes,
                        M, K, N,
                        1.0f,
                        out->grad->gpu_data->get()(), offset_out, N,
                        b->gpu_data->get()(), offset_b, N,
                        1.0f,
                        a->grad->gpu_data->get()(), offset_a, K,
                        &ctx.get_queue()());
                }
                if (b->requires_grad) {
                    // grad_b += a^T @ grad_out
                    clblast::Gemm(
                        clblast::Layout::kRowMajor,
                        clblast::Transpose::kYes, clblast::Transpose::kNo,
                        K, N, M,
                        1.0f,
                        a->gpu_data->get()(), offset_a, K,
                        out->grad->gpu_data->get()(), offset_out, N,
                        1.0f,
                        b->grad->gpu_data->get()(), offset_b, N,
                        &ctx.get_queue()());
                }
            }
            ctx.finish();
        };
    }
    return out;
}

#endif // DLL_GPU_ENABLED
