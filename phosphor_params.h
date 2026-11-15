/* Shared by the host (g++), the CPU reference and the CUDA kernels (NVRTC).
 * Plain C, 4-byte fields only, so every compiler agrees on the layout. */
#ifndef PHOSPHOR_PARAMS_H
#define PHOSPHOR_PARAMS_H

/* One temporal step, already reduced to blend weights by the host:
 *   a = 1 - exp(-dt / tau)
 * computed once per frame in double precision and handed to every path (CPU,
 * GLSL, CUDA), so the three cannot disagree about the exponential. */
typedef struct PhosphorStep {
    float a_rise;      /* weight when the target is brighter than the state  */
    float a_decay[3];  /* weight when it is dimmer, per channel R, G, B      */
    float eps;         /* linear light: closer than this snaps to the target */
} PhosphorStep;

/* The glass. Dial names follow crt/crt-glass.frag; values are display-referred
 * exactly as they are there. Per-frame scalars the clock drives (flicker, the
 * tracking band's position) are computed on the host so the kernel and the
 * shader are stateless functions of one struct. */
typedef struct OpticsParams {
    float curv;          /* 0 flat .. 1 house curve                         */
    float k1, k2;        /* TUBE_K1, TUBE_K2                                */
    float aberr;         /* chromatic split at the rim                      */
    float scan;          /* scanline depth                                  */
    float scan_step;     /* px between scanlines                            */
    float glow;          /* 4-tap bloom, taken in linear light              */
    float glow_px;       /* tap distance in tile px                         */
    float bloom;         /* centre phosphor wash                            */
    float vign;          /* corner falloff                                  */
    float specular;      /* upper-left room-light catch                     */
    float glare;         /* hotspot + diagonal streak                       */
    float tracking;      /* band strength                                   */
    float band_h;        /* band height px                                  */
    float band_center;   /* px down the tile this frame; < -1e8 = no band   */
    float flicker_mul;   /* 1.0 except inside a burst                       */
    float phosphor[3];   /* theme accent                                    */
    float glare_tint[3];
} OpticsParams;

#endif
