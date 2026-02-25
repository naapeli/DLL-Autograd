#pragma once
#include "tensor.h"
#include <memory>


std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& tensor);
