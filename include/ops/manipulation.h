#pragma once
#include "tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int dim = 0);
std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int dim = 0);
