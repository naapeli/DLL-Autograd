#include <vector>
#include "tensor.h"
#include "Random/random.h"


namespace random {
    std::shared_ptr<Tensor> rand(std::vector<int> shape, float min = 0.0, float max = 1.0) {
        int out_size = 1;
        for (int dim : shape) {
            out_size *= dim;
        }
        std::vector<float> result_data(out_size);
        for (int i = 0; i < out_size; ++i) {
            result_data[i] = random::rand(min, max);
        }
        return std::make_shared<Tensor>(std::move(result_data), shape);
    }

    std::shared_ptr<Tensor> rand(std::vector<int> shape, uint32_t seed, float min = 0.0, float max = 1.0) {
        random::set_seed(seed);
        return rand(shape, min, max);
    }

    std::shared_ptr<Tensor> randn(std::vector<int> shape, float mean = 0.0, float stddev = 1.0) {
        int out_size = 1;
        for (int dim : shape) {
            out_size *= dim;
        }
        std::vector<float> result_data(out_size);
        for (int i = 0; i < out_size; ++i) {
            result_data[i] = random::randn(mean, stddev);
        }
        return std::make_shared<Tensor>(std::move(result_data), shape);
    }

    std::shared_ptr<Tensor> randn(std::vector<int> shape, uint32_t seed, float mean = 0.0, float stddev = 1.0) {
        random::set_seed(seed);
        return randn(shape, mean, stddev);
    }
}
