#include "ops/unary.h"
#include <stdexcept>


std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& tensor) {
    if (tensor->shape.size() != 2) {
        throw std::runtime_error("This simple transpose only supports 2D matrices right now.");
    }
    std::vector<int> new_shape = {tensor->shape[1], tensor->shape[0]};
    std::vector<int> new_strides = {tensor->strides[1], tensor->strides[0]};
    return std::make_shared<Tensor>(tensor->data, new_shape, new_strides);
}
