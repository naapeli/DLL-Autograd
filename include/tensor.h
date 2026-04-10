#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef DLL_GPU_ENABLED
#include "gpu_buffer.h"
#endif

namespace py = pybind11;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::shared_ptr<std::vector<float>> data;
    std::vector<int> shape;

    bool requires_grad = false;
    std::shared_ptr<Tensor> grad = nullptr;
    std::vector<std::shared_ptr<Tensor>> _prev;
    std::function<void()> _backward = []() {};

    // Device tracking
    std::string device = "cpu";
#ifdef DLL_GPU_ENABLED
    std::shared_ptr<GPUBuffer> gpu_data;
#endif

    // Constructors
    Tensor(std::vector<float> d, std::vector<int> s);
    Tensor(std::shared_ptr<std::vector<float>> shared_data, std::vector<int> s);
    Tensor(py::list python_list);

#ifdef DLL_GPU_ENABLED
    // GPU constructor (data lives on GPU)
    Tensor(std::shared_ptr<GPUBuffer> gpu_buf, std::vector<int> s);
#endif

    // Getters
    std::vector<int> get_shape() const;
    std::vector<float> get_data() const;

    // Utilities
    float item() const;
    std::shared_ptr<Tensor> slice(py::object slices);
    std::shared_ptr<Tensor> reshape(const std::vector<int>& new_shape);
    std::shared_ptr<Tensor> squeeze(int dim = -1);
    std::shared_ptr<Tensor> unsqueeze(int dim);
    std::string repr() const;
    void zero_grad();

    // backprop
    void backward(std::shared_ptr<Tensor> initial_grad = nullptr);

    // Device management
    std::shared_ptr<Tensor> to(const std::string& target_device);
    std::shared_ptr<Tensor> cpu();
    bool is_gpu() const;
    int numel() const;

    // Ensure CPU data is available (for Python access)
    void ensure_cpu_data() const;
};

// Helper: warn and auto-transfer for mixed-device ops
#ifdef DLL_GPU_ENABLED
std::shared_ptr<Tensor> ensure_same_device(const std::shared_ptr<Tensor>& a,
                                             const std::shared_ptr<Tensor>& b);
#endif
