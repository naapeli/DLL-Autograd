#pragma once
#include "tensor.h"
#include <memory>


std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
std::shared_ptr<Tensor> prod(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
std::shared_ptr<Tensor> prod(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
std::shared_ptr<Tensor> min(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
std::shared_ptr<Tensor> min(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
// std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
// std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
// std::shared_ptr<Tensor> std(const std::shared_ptr<Tensor>& tensor, bool keepdim=false);
// std::shared_ptr<Tensor> std(const std::shared_ptr<Tensor>& tensor, int dim, bool keepdim=false);
