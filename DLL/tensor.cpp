#include "tensor.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <string>
#include <set>
#include <vector>

#ifdef DLL_GPU_ENABLED
#include "gpu_context.h"
#endif

Tensor::Tensor(std::vector<float> d, std::vector<int> s) : shape(s) {
    data = std::make_shared<std::vector<float>>(d);
    
    int total_elements = 1;
    for (int dim : shape) total_elements *= dim;
    if (data->size() != total_elements) {
        throw std::invalid_argument("Data size does not match shape.");
    }
}

Tensor::Tensor(std::shared_ptr<std::vector<float>> shared_data, std::vector<int> s) 
    : data(shared_data), shape(s) {}

#ifdef DLL_GPU_ENABLED
Tensor::Tensor(std::shared_ptr<GPUBuffer> gpu_buf, std::vector<int> s) 
    : shape(s), device("gpu"), gpu_data(gpu_buf) {
    // CPU data is empty — will be populated on demand via ensure_cpu_data()
    data = std::make_shared<std::vector<float>>();
}
#endif

std::vector<int> Tensor::get_shape() const { return shape; }

std::vector<float> Tensor::get_data() const { 
    ensure_cpu_data();
    return *data; 
}

float Tensor::item() const {
    ensure_cpu_data();
    size_t n = 1;
    for (int dim : shape) n *= dim;
    if (n != 1) {
        throw std::runtime_error(
            "ValueError: item() can only be called on tensors with 1 element, but this tensor has " + std::to_string(n) + " elements."
        );
    }
    return data->at(0);
}

int Tensor::numel() const {
    int n = 1;
    for (int dim : shape) n *= dim;
    return n;
}

bool Tensor::is_gpu() const {
    return device == "gpu";
}

void Tensor::ensure_cpu_data() const {
#ifdef DLL_GPU_ENABLED
    if (device == "gpu" && gpu_data && (data->empty() || data->size() != (size_t)numel())) {
        // Download GPU data to CPU — cast away const for lazy sync
        auto* mutable_this = const_cast<Tensor*>(this);
        int n = numel();
        mutable_this->data = std::make_shared<std::vector<float>>(n);
        gpu_data->read_to(mutable_this->data->data(), n);
    }
#endif
}

std::shared_ptr<Tensor> Tensor::to(const std::string& target_device) {
    if (target_device == device) {
        return shared_from_this();
    }
#ifdef DLL_GPU_ENABLED
    if (target_device == "gpu") {
        if (!GPUContext::instance().is_available()) {
            throw std::runtime_error("No GPU available. Cannot move tensor to GPU.");
        }
        ensure_cpu_data();
        int n = numel();
        auto gpu_buf = std::make_shared<GPUBuffer>(data->data(), n);
        auto out = std::make_shared<Tensor>(gpu_buf, shape);
        out->requires_grad = requires_grad;
        return out;
    } else if (target_device == "cpu") {
        return cpu();
    }
#endif
    throw std::invalid_argument("Unknown device: " + target_device + ". Expected 'cpu' or 'gpu'.");
}

std::shared_ptr<Tensor> Tensor::cpu() {
    if (device == "cpu") return shared_from_this();
#ifdef DLL_GPU_ENABLED
    ensure_cpu_data();
    auto out = std::make_shared<Tensor>(*data, shape);
    out->requires_grad = requires_grad;
    return out;
#else
    return shared_from_this();
#endif
}

#ifdef DLL_GPU_ENABLED
std::shared_ptr<Tensor> ensure_same_device(const std::shared_ptr<Tensor>& a,
                                             const std::shared_ptr<Tensor>& b) {
    // If both on same device, return b unchanged
    if (a->device == b->device) return b;
    
    // At least one is on GPU — emit Python warning and auto-transfer to GPU
    py::gil_scoped_acquire gil;
    PyErr_WarnEx(PyExc_UserWarning, 
        "DLL: Tensors on different devices. Automatically moving to GPU. "
        "For best performance, use .to('gpu') explicitly.", 1);
    
    if (a->is_gpu() && !b->is_gpu()) {
        // Move b to GPU
        return b->to("gpu");
    }
    // Move b to CPU (shouldn't normally happen, but handle it)
    return b->cpu();
}
#endif

namespace {
    void parse_nested_list(const py::handle& obj, std::vector<float>& flat_data, std::vector<int>& shape, int depth) {
        if (py::isinstance<py::list>(obj)) {
            py::list lst = py::cast<py::list>(obj);
            
            if (depth >= shape.size()) {
                shape.push_back(lst.size());
            } else if (shape[depth] != lst.size()) {
                throw std::invalid_argument("Ragged sequences (lists of varying lengths) are not allowed.");
            }
            
            for (auto item : lst) {
                parse_nested_list(item, flat_data, shape, depth + 1);
            }
        } else {
            flat_data.push_back(py::cast<float>(obj));
        }
    }
}

Tensor::Tensor(py::list python_list) {
    std::vector<float> parsed_data;
    std::vector<int> parsed_shape;
    
    parse_nested_list(python_list, parsed_data, parsed_shape, 0);
    
    this->data = std::make_shared<std::vector<float>>(std::move(parsed_data));
    this->shape = std::move(parsed_shape);
}

void format_tensor(std::stringstream& ss, const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int>& local_strides, int dim, int offset, int indent_level, bool has_negatives) {
    if (dim == shape.size()) {
        float val = data[offset];
        if (has_negatives && val >= 0) {
            ss << " ";
        }
        ss << std::fixed << std::setprecision(4) << val;
        return;
    }

    ss << "[";
    for (int i = 0; i < shape[dim]; ++i) {
        format_tensor(ss, data, shape, local_strides, dim + 1, offset + i * local_strides[dim], indent_level + 1, has_negatives);
        
        if (i < shape[dim] - 1) {
            ss << ", ";
            if (dim < shape.size() - 1) {
                ss << "\n";
                for (int j = 0; j < indent_level + 1; ++j) ss << " ";
            }
        }
    }
    ss << "]";
}

std::string Tensor::repr() const {
    ensure_cpu_data();
    
    bool has_negatives = false;
    for (float v : *data) {
        if (v < 0) {
            has_negatives = true;
            break;
        }
    }

    std::vector<int> local_strides(shape.size());
    int current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        local_strides[i] = current_stride;
        current_stride *= shape[i];
    }

    std::stringstream ss;
    std::string prefix = "Tensor(";
    ss << prefix;
    
    format_tensor(ss, *data, shape, local_strides, 0, 0, (int)prefix.length(), has_negatives);
    
    ss << ", shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ss << "]";
    
    if (device == "gpu") {
        ss << ", device=gpu";
    }
    
    ss << ")";
    return ss.str();
}

void Tensor::zero_grad() {
    int size = numel();
#ifdef DLL_GPU_ENABLED
    if (is_gpu()) {
        auto gpu_buf = std::make_shared<GPUBuffer>(size);
        // Fill with zeros
        auto& ctx = GPUContext::instance();
        cl::Kernel fill = ctx.get_kernel("fill_kernel");
        fill.setArg(0, gpu_buf->get());
        fill.setArg(1, 0.0f);
        fill.setArg(2, size);
        ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange((size + 255) / 256 * 256), cl::NDRange(256));
        ctx.finish();
        grad = std::make_shared<Tensor>(gpu_buf, shape);
        return;
    }
#endif
    grad = std::make_shared<Tensor>(std::vector<float>(size, 0.0f), shape);
}

void Tensor::backward() {
    int size = numel();
    if (size != 1) {
        throw std::runtime_error("grad can be implicitly created only for scalar outputs");
    }

    std::vector<std::shared_ptr<Tensor>> topo;
    std::set<std::shared_ptr<Tensor>> visited;

    std::function<void(std::shared_ptr<Tensor>)> build_topo = 
        [&](std::shared_ptr<Tensor> v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto& parent : v->_prev) {
                    build_topo(parent);
                }
                topo.push_back(v);
            }
        };

    build_topo(shared_from_this());

    if (!this->grad) this->zero_grad();
#ifdef DLL_GPU_ENABLED
    if (is_gpu()) {
        // Fill grad with 1.0
        auto& ctx = GPUContext::instance();
        cl::Kernel fill = ctx.get_kernel("fill_kernel");
        fill.setArg(0, this->grad->gpu_data->get());
        fill.setArg(1, 1.0f);
        fill.setArg(2, 1);
        ctx.get_queue().enqueueNDRangeKernel(fill, cl::NullRange, cl::NDRange(1), cl::NullRange);
        ctx.finish();
    } else
#endif
    {
        std::fill(this->grad->data->begin(), this->grad->data->end(), 1.0f);
    }

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

std::shared_ptr<Tensor> Tensor::slice(py::object slices) {
    ensure_cpu_data();
    
    py::tuple indices;
    if (py::isinstance<py::tuple>(slices)) {
        indices = slices.cast<py::tuple>();
    } else {
        indices = py::make_tuple(slices);
    }

    if (indices.size() > this->shape.size()) {
        throw std::invalid_argument("Too many indices for tensor.");
    }

    std::vector<int> local_strides(this->shape.size());
    int current_stride = 1;
    for (int i = this->shape.size() - 1; i >= 0; --i) {
        local_strides[i] = current_stride;
        current_stride *= this->shape[i];
    }

    std::vector<int> new_shape;
    std::vector<int> new_strides;
    int new_offset = 0;

    for (size_t i = 0; i < this->shape.size(); ++i) {
        if (i >= indices.size()) {
            new_shape.push_back(this->shape[i]);
            new_strides.push_back(local_strides[i]);
            continue;
        }

        py::object item = indices[i];

        if (py::isinstance<py::int_>(item)) {
            int idx = item.cast<int>();
            if (idx < 0) idx += this->shape[i]; 
            if (idx < 0 || idx >= this->shape[i]) throw std::out_of_range("Index out of bounds");
            new_offset += idx * local_strides[i];
        } else if (py::isinstance<py::slice>(item)) {
            py::slice slice_obj = item.cast<py::slice>();
            size_t start, stop, step, slicelength;
            
            if (!slice_obj.compute(this->shape[i], &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
            }

            if (slicelength > 0) {
                new_offset += start * local_strides[i];
                new_shape.push_back(slicelength);
                
                int raw_step = 1; 
                if (!slice_obj.attr("step").is_none()) {
                    raw_step = slice_obj.attr("step").cast<int>();
                }
                new_strides.push_back(local_strides[i] * raw_step);
            } else {
                new_shape.push_back(0);
                new_strides.push_back(local_strides[i]);
            }
        } else {
            throw std::invalid_argument("Invalid index type. Expected int or slice.");
        }
    }

    int total_elements = 1;
    for (int dim : new_shape) {
        total_elements *= dim;
    }

    std::vector<float> copied_data(total_elements);

    if (total_elements > 0) {
        for (int i = 0; i < total_elements; ++i) {
            int old_flat_index = new_offset;
            int temp = i;
            
            for (int j = new_shape.size() - 1; j >= 0; --j) {
                int coord = temp % new_shape[j];
                old_flat_index += coord * new_strides[j];
                temp /= new_shape[j];
            }
            copied_data[i] = this->data->at(old_flat_index);
        }
    }

    auto out = std::make_shared<Tensor>(copied_data, new_shape);
    out->requires_grad = this->requires_grad;

    if (this->requires_grad) {
        out->_prev.push_back(shared_from_this());
        
        auto self = shared_from_this();
        out->_backward = [self, out, new_offset, new_shape, new_strides, total_elements]() {
            if (!self->grad) { self->zero_grad(); }

            if (total_elements > 0 && out->grad) {
                for (int i = 0; i < total_elements; ++i) {
                    int old_flat_index = new_offset;
                    int temp = i;
                    
                    for (int j = new_shape.size() - 1; j >= 0; --j) {
                        int coord = temp % new_shape[j];
                        old_flat_index += coord * new_strides[j];
                        temp /= new_shape[j];
                    }
                    
                    self->grad->data->at(old_flat_index) += out->grad->data->at(i);
                }
            }
        };
    }

    return out;
}
