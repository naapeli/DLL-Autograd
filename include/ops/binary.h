#pragma once
#include "tensor.h"
#include <memory>


std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float scalar);

std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float scalar);

std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> sub_scalar(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> rsub_scalar(const std::shared_ptr<Tensor>& a, float scalar);

std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> div_scalar(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> rdiv_scalar(const std::shared_ptr<Tensor>& a, float scalar);

std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> pow_scalar(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> rpow_scalar(const std::shared_ptr<Tensor>& a, float scalar);
