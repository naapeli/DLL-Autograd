#pragma once
#include "Xoshiro256.h"
#include <stdint.h>
#include <cstddef>


namespace randomGen {
    extern Xoshiro256 engine;

    void set_seed(uint32_t seed);
    
    void fill_uniform(float* data, size_t n, float min, float max);
    void fill_normal(float* data, size_t n, float mean, float stddev);
}
