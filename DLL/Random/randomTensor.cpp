#include "tensor.h"
#include "Random/randomTensor.h"
#include "Random/random.h"


namespace randomTensor {

    static size_t compute_size(const std::vector<int>& shape) {
        size_t size = 1;
        for (int dim : shape) size *= (size_t)dim;
        return size;
    }

    std::shared_ptr<Tensor> rand(std::vector<int> shape, float min, float max) {
        size_t total_size = compute_size(shape);
        
        std::vector<float> data(total_size); 
        
        randomGen::fill_uniform(data.data(), total_size, min, max);
        
        return std::make_shared<Tensor>(std::move(data), shape);
    }

    std::shared_ptr<Tensor> randn(std::vector<int> shape, float mean, float stddev) {
        size_t total_size = compute_size(shape);
        std::vector<float> data(total_size);

        randomGen::fill_normal(data.data(), total_size, mean, stddev);

        return std::make_shared<Tensor>(std::move(data), shape);
    }
}
