#pragma once
#include <stdint.h>


class Xoshiro256 {
public:
    explicit Xoshiro256(uint64_t seed) {
        uint64_t z = (seed + 0x9E3779B97F4A7C15);
        for (int i = 0; i < 4; i++) {
            z += 0x9E3779B97F4A7C15;
            uint64_t tmp = z;
            tmp = (tmp ^ (tmp >> 30)) * 0xBF58476D1CE4E5B9;
            tmp = (tmp ^ (tmp >> 27)) * 0x94D049BB133111EB;
            s[i] = tmp ^ (tmp >> 31);
        }
    }

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    inline uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    inline float next_float() {
        return (next() >> 40) * 0x1.0p-24f;
    }

private:
    uint64_t s[4];
};
