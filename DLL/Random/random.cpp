#include "Random/random.h"
#include "Random/mersenneTwister.h"
#include <cmath>

namespace randomGen {
    namespace {
        MersenneTwister engine(5489U);
        bool has_spare = false;
        float spare_value = 0.0f;
    }

    float rand(float min, float max) {
        float r = engine.next() * (1.0f / 4294967296.0f); 
        return min + r * (max - min);
    }

    float randn(float mean, float var) {
        float stddev = std::sqrt(var);
        if (has_spare) {
            has_spare = false;
            return mean + stddev * spare_value;
        }

        float u, v, s;
        do {
            u = rand(-1.0f, 1.0f);
            v = rand(-1.0f, 1.0f);
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);

        float multiplier = std::sqrt(-2.0f * std::log(s) / s);
        spare_value = v * multiplier;
        has_spare = true;
        
        return mean + stddev * (u * multiplier);
    }

    void set_seed(uint32_t seed) {
        engine = MersenneTwister(seed);
        has_spare = false; 
    }
}
