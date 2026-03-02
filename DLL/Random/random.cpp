#include "Random/random.h"
#include <cmath>
#include <random>

namespace randomGen {
    std::random_device rd;
    Xoshiro256 engine(rd());

    void set_seed(uint32_t seed) {
        engine = Xoshiro256(static_cast<uint64_t>(seed));
    }

    void fill_uniform(float* data, size_t n, float min, float max) {
        const float range = max - min;
        for (size_t i = 0; i < n; ++i) {
            data[i] = min + engine.next_float() * range;
        }
    }

    void fill_normal(float* data, size_t n, float mean, float stddev) {
        const float two_pi = 6.283185307f;
        const float epsilon = 1e-10f;

        size_t i = 0;
        for (; i + 1 < n; i += 2) {
            float u1 = engine.next_float() + epsilon;
            float u2 = engine.next_float();

            float radius = std::sqrt(-2.0f * std::log(u1));
            float theta = two_pi * u2;

            data[i]     = mean + stddev * (radius * std::cos(theta));
            data[i + 1] = mean + stddev * (radius * std::sin(theta));
        }

        if (i < n) {
            float u1 = engine.next_float() + epsilon;
            float u2 = engine.next_float();
            data[i] = mean + stddev * (std::sqrt(-2.0f * std::log(u1)) * std::cos(two_pi * u2));
        }
    }
}
