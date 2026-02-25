#pragma once
#include <vector>
#include <memory>
#include <functional>


class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::shared_ptr<std::vector<float>> data;
    std::vector<int> shape;
    std::vector<int> strides;

    bool requires_grad = false;
    std::shared_ptr<Tensor> grad = nullptr;
    std::vector<std::shared_ptr<Tensor>> _prev;
    std::function<void()> _backward = []() {};

    // Constructors
    Tensor(std::vector<float> d, std::vector<int> s);
    Tensor(std::shared_ptr<std::vector<float>> shared_data, std::vector<int> s, std::vector<int> st);

    // Getters
    std::vector<int> get_shape() const;
    std::vector<int> get_strides() const;
    std::vector<float> get_data() const;

    // Utilities
    float item(const std::vector<int>& indices) const;
    std::string repr() const;
    void zero_grad();

    // backprop
    void backward();
    // void backward(const std::shared_ptr<Tensor>& grad);
};
