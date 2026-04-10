#pragma once
#ifdef DLL_GPU_ENABLED
#include "tensor.h"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> cat_gpu(const std::vector<std::shared_ptr<Tensor>>& tensors, int dim);
#endif
