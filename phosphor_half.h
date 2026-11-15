/* Half-precision floats by hand, shared by the CUDA kernels and the CPU
 * reference so the two cannot disagree about rounding. Round to nearest, ties
 * to even; inputs are phosphor state, so non-negative and finite. The GPU's own
 * texture unit decodes these bits the same way when the glass reads the state. */
#ifndef PHOSPHOR_HALF_H
#define PHOSPHOR_HALF_H
#ifdef __CUDACC__
#define PH_INLINE __device__ inline
#else
#define PH_INLINE static inline
#endif

PH_INLINE unsigned short ph_f2h(float f) {
    union { float f; unsigned int u; } v;
    v.f = f;
    if (!(f > 0.0f)) return 0;
    unsigned int e = (v.u >> 23) & 0xffu, m = v.u & 0x7fffffu, r, rem, half;
    if (e < 103u) return 0;       /* below the smallest half subnormal */
    if (e > 142u) return 0x7bffu; /* largest finite half */
    if (e < 113u) {               /* half subnormal: units of 2^-24 */
        unsigned int shift = 126u - e;
        m |= 0x800000u;
        r = m >> shift;
        rem = m & ((1u << shift) - 1u);
        half = 1u << (shift - 1u);
    } else {
        r = ((e - 112u) << 10) | (m >> 13);
        rem = m & 0x1fffu;
        half = 0x1000u;
    }
    if (rem > half || (rem == half && (r & 1u))) r++; /* a carry runs into the exponent correctly */
    return (unsigned short)r;
}

PH_INLINE float ph_h2f(unsigned short h) {
    union { float f; unsigned int u; } v;
    unsigned int e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0u) return (float)m * (1.0f / 16777216.0f);
    v.u = ((e + 112u) << 23) | (m << 13);
    return v.f;
}
#endif
