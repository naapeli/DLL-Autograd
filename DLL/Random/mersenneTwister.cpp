#include "Random/mersenneTwister.h"

MersenneTwister::MersenneTwister(uint32_t seed) {
    mt[0] = seed;
    for (mti = 1; mti < N; mti++) {
        mt[mti] = (1812433253U * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    }
}

void MersenneTwister::twist() {
    static const uint32_t mag01[2] = {0x0U, MATRIX_A};
    int kk;
    for (kk = 0; kk < N - M; kk++) {
        uint32_t y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
    }
    for (; kk < N - 1; kk++) {
        uint32_t y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
    }
    uint32_t y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
    mti = 0;
}

uint32_t MersenneTwister::next() {
    if (mti >= N) twist();

    uint32_t y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}
