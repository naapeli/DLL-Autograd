#pragma once
#include <vector>
#include <memory>
#include "tensor.h"


namespace randomTensor {
    std::shared_ptr<Tensor> rand(std::vector<int> shape, float min = 0.0f, float max = 1.0f);
    std::shared_ptr<Tensor> randn(std::vector<int> shape, float mean = 0.0f, float stddev = 1.0f);
}
