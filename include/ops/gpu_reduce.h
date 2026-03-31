#pragma once

#ifdef DLL_GPU_ENABLED

#include "tensor.h"
#include <memory>

std::shared_ptr<Tensor> sum_gpu(const std::shared_ptr<Tensor>& a, bool keepdim);
std::shared_ptr<Tensor> sum_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim);
std::shared_ptr<Tensor> max_gpu(const std::shared_ptr<Tensor>& a, bool keepdim);
std::shared_ptr<Tensor> max_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim);
std::shared_ptr<Tensor> min_gpu(const std::shared_ptr<Tensor>& a, bool keepdim);
std::shared_ptr<Tensor> min_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim);
std::shared_ptr<Tensor> prod_gpu(const std::shared_ptr<Tensor>& a, bool keepdim);
std::shared_ptr<Tensor> prod_gpu(const std::shared_ptr<Tensor>& a, int dim, bool keepdim);

#endif // DLL_GPU_ENABLED
