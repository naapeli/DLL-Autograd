#include "tensor.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <string>
#include <set>


Tensor::Tensor(std::vector<float> d, std::vector<int> s) : shape(s) {
    data = std::make_shared<std::vector<float>>(d);
    
    int total_elements = 1;
    for (int dim : shape) total_elements *= dim;
    if (data->size() != total_elements) {
        throw std::invalid_argument("Data size does not match shape.");
    }

    strides.resize(shape.size());
    int current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
}

Tensor::Tensor(std::shared_ptr<std::vector<float>> shared_data, std::vector<int> s, std::vector<int> st) 
    : data(shared_data), shape(s), strides(st) {}

std::vector<int> Tensor::get_shape() const { return shape; }
std::vector<int> Tensor::get_strides() const { return strides; }    
std::vector<float> Tensor::get_data() const { return *data; } 

float Tensor::item(const std::vector<int>& indices) const {
    int flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_index += indices[i] * strides[i];
    }
    return (*data)[flat_index];
}

void format_tensor(std::stringstream& ss, const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int>& strides, int dim, int offset, int indent_level, bool has_negatives) {
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
        format_tensor(ss, data, shape, strides, dim + 1, offset + i * strides[dim], indent_level + 1, has_negatives);
        
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
    bool has_negatives = false;
    for (float v : *data) {
        if (v < 0) {
            has_negatives = true;
            break;
        }
    }

    std::stringstream ss;
    std::string prefix = "Tensor(";
    ss << prefix;
    
    format_tensor(ss, *data, shape, strides, 0, 0, (int)prefix.length(), has_negatives);
    
    ss << ", shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ss << "])";
    return ss.str();
}

void Tensor::zero_grad() {
    int size = 1;
    for (int dim : shape) size *= dim;
    grad = std::make_shared<Tensor>(std::vector<float>(size, 0.0f), shape);
}

void Tensor::backward() {
    int size = 1;
    for (int dim : shape) size *= dim;
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
    std::fill(this->grad->data->begin(), this->grad->data->end(), 1.0f);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}
