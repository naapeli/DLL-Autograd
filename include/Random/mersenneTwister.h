#pragma once
#include <cstdint>

class MersenneTwister {
public:
    explicit MersenneTwister(uint32_t seed = 5489U);
    uint32_t next();

private:
    static const int N = 624;
    static const int M = 397;
    static const uint32_t MATRIX_A = 0x9908b0dfU;
    static const uint32_t UPPER_MASK = 0x80000000U;
    static const uint32_t LOWER_MASK = 0x7fffffffU;

    uint32_t mt[N];
    int mti;

    void twist();
};
