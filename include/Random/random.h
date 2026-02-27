#pragma once
#include "MersenneTwister.h"


namespace random {
    float rand(float min = 0.0, float max = 1.0);
    float randn(float mean = 0.0, float stddev = 1.0);
    void set_seed(uint32_t seed);
}
