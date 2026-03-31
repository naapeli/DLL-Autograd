#ifdef DLL_GPU_ENABLED

#include "gpu_context.h"
#include "kernels.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>

GPUContext::GPUContext() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) return;

        // Try to find a GPU device across all platforms
        for (auto& platform : platforms) {
            std::vector<cl::Device> devices;
            try {
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            } catch (...) {
                continue;
            }
            if (!devices.empty()) {
                device = devices[0];
                device_name_ = device.getInfo<CL_DEVICE_NAME>();
                context = cl::Context(device);
                queue = cl::CommandQueue(context, device);
                available_ = true;
                compile_kernels();
                return;
            }
        }
        // No GPU found
    } catch (const cl::Error& e) {
        std::cerr << "[DLL] OpenCL init error: " << e.what() 
                  << " (code " << e.err() << ")" << std::endl;
    }
}

void GPUContext::compile_kernels() {
    try {
        program = cl::Program(context, OPENCL_KERNEL_SOURCE);
        program.build({device}, "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math");
    } catch (const cl::BuildError& e) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "[DLL] OpenCL kernel build error:\n" << log << std::endl;
        available_ = false;
        return;
    }
}

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

bool GPUContext::is_available() const { return available_; }
std::string GPUContext::device_name() const { return device_name_; }

cl::Buffer GPUContext::alloc(size_t bytes) {
    return cl::Buffer(context, CL_MEM_READ_WRITE, bytes);
}

void GPUContext::write(cl::Buffer& buf, const float* data, size_t n) {
    queue.enqueueWriteBuffer(buf, CL_TRUE, 0, n * sizeof(float), data);
}

void GPUContext::read(const cl::Buffer& buf, float* data, size_t n) {
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, n * sizeof(float), data);
}

void GPUContext::finish() {
    queue.finish();
}

cl::Kernel GPUContext::get_kernel(const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = kernels.find(name);
    if (it != kernels.end()) {
        return it->second;
    }
    cl::Kernel k(program, name.c_str());
    kernels[name] = k;
    return k;
}

cl::CommandQueue& GPUContext::get_queue() { return queue; }
cl::Context& GPUContext::get_context() { return context; }

#endif // DLL_GPU_ENABLED
