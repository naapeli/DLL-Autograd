#pragma once

#ifdef DLL_GPU_ENABLED

#include "tensor.h"
#include <memory>

std::shared_ptr<Tensor> exp_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> log_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> sqrt_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> cbrt_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> abs_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> sin_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> cos_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> relu_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> tanh_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> sigmoid_gpu(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> transpose_gpu(const std::shared_ptr<Tensor>& a, int dim0, int dim1);

#endif // DLL_GPU_ENABLED
