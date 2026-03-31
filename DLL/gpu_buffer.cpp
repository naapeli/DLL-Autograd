#ifdef DLL_GPU_ENABLED

#include "gpu_buffer.h"
#include <stdexcept>

GPUBuffer::GPUBuffer(size_t n) : count_(n) {
    auto& ctx = GPUContext::instance();
    buffer_ = ctx.alloc(n * sizeof(float));
}

GPUBuffer::GPUBuffer(const float* host_data, size_t n) : count_(n) {
    auto& ctx = GPUContext::instance();
    buffer_ = ctx.alloc(n * sizeof(float));
    ctx.write(buffer_, host_data, n);
}

void GPUBuffer::read_to(float* host, size_t n) const {
    auto& ctx = GPUContext::instance();
    ctx.read(buffer_, host, n);
}

cl::Buffer& GPUBuffer::get() { return buffer_; }
const cl::Buffer& GPUBuffer::get() const { return buffer_; }
size_t GPUBuffer::count() const { return count_; }

#endif // DLL_GPU_ENABLED
