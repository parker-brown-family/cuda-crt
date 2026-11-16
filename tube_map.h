/* The tube's curve, in ONE place.
 *
 * The glass is a gather: for a point on the screen it says which point of the
 * window's picture to show there. A pointer on that glass must be told the
 * same thing: "you are over THIS point of the picture". If the two ever use
 * different formulas, a click lands beside what the eye sees. So both the
 * kernel that draws the picture and the compositor code that maps the pointer
 * call tube_gather() from this file, and nothing else is allowed to compute it.
 *
 * Plain C, so the CUDA kernels, the CPU, and a compositor plugin all compile it.
 */
#ifndef TUBE_MAP_H
#define TUBE_MAP_H
#ifdef __CUDACC__
#define TM_INLINE __device__ inline
#else
#include <math.h>
#define TM_INLINE static inline
#endif

typedef struct TubeMap {
    float k1, k2;  /* the barrel, already scaled by the curve dial (0 = flat glass)         */
    float core;    /* 0..0.95  how much of the middle stays perfectly flat. 0 = the classic */
                   /*          barrel; higher keeps the bend out near the rim and corners   */
    float inset;   /* 0..0.12  draw the picture this much smaller, so the rim, its shadow   */
                   /*          and the rounded corners sit on padding and not on the window */
} TubeMap;

/* Screen position inside the tile (0..1 each way) -> position in the window's
 * picture (0..1 = inside the window; outside that range = bezel or padding). */
TM_INLINE void tube_gather(TubeMap m, float lx, float ly, float *sx, float *sy) {
    float cx = lx - 0.5f, cy = ly - 0.5f;
    float r2 = cx * cx + cy * cy;            /* 0 at the centre, 0.25 mid-edge, 0.5 at a corner */
    float c2 = m.core * 0.45f;               /* where the bend begins                            */
    float e  = r2 > c2 ? (r2 - c2) / (0.5f - c2) : 0.0f;   /* 0 at the core's edge .. 1 at a corner */
    /* With core = 0 this is exactly the classic q = r2. A flat core starts the bend
     * with zero slope (power > 1), so there is no visible crease where it begins. */
    float q  = 0.5f * powf(e, 1.0f + 2.0f * m.core);
    float f  = (1.0f + m.k1 * q + m.k2 * q * q) * (1.0f + m.inset);
    *sx = 0.5f + cx * f;
    *sy = 0.5f + cy * f;
}

/* How far the picture has been moved at a screen position, as a fraction of the
 * tile: the distance between where a thing is drawn and where it "really" is.
 * This is the reach a hand has to make up when nothing maps the pointer. */
TM_INLINE float tube_displacement(TubeMap m, float lx, float ly) {
    float sx, sy;
    tube_gather(m, lx, ly, &sx, &sy);
    return sqrtf((sx - lx) * (sx - lx) + (sy - ly) * (sy - ly));
}
#endif
