#pragma once
#include "tensor.h"
#include <memory>


std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& tensor, int dim0=-2, int dim1=-1);

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> cbrt(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> abs(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& tensor);
std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& tensor);
std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& tensor);
std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& tensor, int dim=-1);
