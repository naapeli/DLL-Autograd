#pragma once

#ifdef DLL_GPU_ENABLED

#include "tensor.h"
#include <memory>

std::shared_ptr<Tensor> add_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> add_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> mul_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> mul_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> pow_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> pow_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> rpow_scalar_gpu(const std::shared_ptr<Tensor>& a, float scalar);
std::shared_ptr<Tensor> matmul_gpu(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

#endif // DLL_GPU_ENABLED
