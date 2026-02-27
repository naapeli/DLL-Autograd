#pragma once
#include "tensor.h"
#include <memory>
#include <vector>


namespace randomTensor {
    std::shared_ptr<Tensor> rand(std::vector<int> shape, float min = 0.0, float max = 1.0);
    std::shared_ptr<Tensor> rand(std::vector<int> shape, uint32_t seed, float min = 0.0, float max = 1.0);
    std::shared_ptr<Tensor> randn(std::vector<int> shape, float mean = 0.0, float stddev = 1.0);
    std::shared_ptr<Tensor> randn(std::vector<int> shape, uint32_t seed, float mean = 0.0, float stddev = 1.0);
}
