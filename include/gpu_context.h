#pragma once

#ifdef DLL_GPU_ENABLED

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <string>
#include <unordered_map>
#include <mutex>

class GPUContext {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    std::unordered_map<std::string, cl::Kernel> kernels;
    bool available_ = false;
    std::string device_name_;
    std::mutex mtx;

    GPUContext();
    void compile_kernels();

public:
    static GPUContext& instance();

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    bool is_available() const;
    std::string device_name() const;

    cl::Buffer alloc(size_t bytes);
    void write(cl::Buffer& buf, const float* data, size_t n);
    void read(const cl::Buffer& buf, float* data, size_t n);
    void finish();

    cl::Kernel get_kernel(const std::string& name);
    cl::CommandQueue& get_queue();
    cl::Context& get_context();
};

#endif // DLL_GPU_ENABLED
