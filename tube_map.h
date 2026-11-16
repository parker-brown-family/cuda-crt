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

/* What moves the picture for a moment: the tracking bar's bend and tearing, the
 * jiggle, the wobble at switch-on. Filled once per frame by the host; the pointer
 * keeps the last frame's copy, so it follows exactly the picture on screen. */
typedef struct TubeMotion {
    float degauss, degauss_phase;           /* 0 = none                                  */
    float tracking, band_center, band_h;    /* band_center < -1e8 = no bar this frame     */
    float track_bend, track_noise;
    float jiggle_px;                        /* this frame's vertical hop, tile pixels     */
    float seed;                             /* changes every frame                        */
} TubeMotion;

/* An integer hash: unlike a sine-based one it gives identical bits on the GPU (with
 * fast math) and the CPU, so the pointer can reproduce every torn line exactly. */
TM_INLINE float tm_hash01(int a, int b) {
    unsigned int h = (unsigned int)a * 0x9E3779B1u ^ (unsigned int)b * 0x85EBCA77u;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (float)(h & 0xFFFFFFu) * (1.0f / 16777216.0f);
}
/* sine of a large argument, reduced first so fast and exact sine agree */
TM_INLINE float tm_sin(float a) { return sinf(fmodf(a, 6.2831853f)); }
TM_INLINE float tm_cos(float a) { return cosf(fmodf(a, 6.2831853f)); }

/* Move a screen position inside the tile (0..1 each way) the way this frame's
 * transient effects move the picture under it. fw, fh: the tile in framebuffer
 * pixels. *bar and *tear come back for the static the kernel draws. Returns 1 if
 * anything moved. The row the bar and the tearing act on is the row of the glass
 * the point is on, from tube_gather of the point as given. */
TM_INLINE int tube_disturb(TubeMap m, TubeMotion v, float fw, float fh, float *lx, float *ly, float *bar, float *tear) {
    float gx, gy;
    tube_gather(m, *lx, *ly, &gx, &gy);
    float Ly = gy < 0.0f ? 0.0f : (gy > 1.0f ? 1.0f : gy);
    int moved = 0;
    *bar = 0.0f; *tear = 0.0f;
    if (v.degauss > 0.0f) {
        *lx += v.degauss * 0.012f * tm_sin(v.degauss_phase + *ly * 11.0f);
        *ly += v.degauss * 0.007f * tm_cos(v.degauss_phase * 1.31f + *lx * 9.0f);
        moved = 1;
    }
    if (v.tracking > 0.001f && v.band_center > -1.0e8f) {
        float d = fabsf(Ly * fh - v.band_center) / (v.band_h * 0.5f);
        *bar = 1.0f - (d > 1.0f ? 1.0f : d);
    }
    if (*bar > 0.0f && (v.track_bend > 0.0f || v.track_noise > 0.0f)) {
        *tear = tm_hash01((int)floorf(Ly * fh), (int)v.seed) - 0.5f;   /* one value per scanline per frame */
        float push = v.track_bend * *bar * *bar * tm_sin(Ly * fh * 0.045f + v.seed * 0.9f) + v.track_noise * *bar * *tear * 9.0f;
        *lx += push / fw;
        moved = 1;
    }
    if (v.jiggle_px != 0.0f) { *ly += v.jiggle_px / fh; moved = 1; }
    return moved;
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
