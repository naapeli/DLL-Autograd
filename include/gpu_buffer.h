#pragma once

#ifdef DLL_GPU_ENABLED

#include "gpu_context.h"
#include <memory>

class GPUBuffer {
    cl::Buffer buffer_;
    size_t count_; // number of floats

public:
    // Allocate uninitialized buffer
    GPUBuffer(size_t n);

    // Allocate and upload from host
    GPUBuffer(const float* host_data, size_t n);

    // Download to host
    void read_to(float* host, size_t n) const;

    cl::Buffer& get();
    const cl::Buffer& get() const;
    size_t count() const;
};

#endif // DLL_GPU_ENABLED
