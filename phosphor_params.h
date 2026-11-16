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

/* Version 2 of the step: the state is HALF precision (rgb = the phosphor,
 * a = a slow "burn" term that charges under light that stays and drains slowly
 * once it goes, drawn as a faint ghost). Half precision is where the speed is:
 * the update is bound by memory traffic, and this halves it. */
typedef struct PhosphorStepV2 {
    float a_rise;
    float a_decay[3];
    float eps;          /* linear light: closer than this snaps to the target */
    float a_burn_rise;  /* how fast the burn charges under steady light       */
    float a_burn_decay; /* how fast it drains                                 */
    float eps_burn;     /* burn closer than this to its target snaps          */
} PhosphorStepV2;

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
    /* CUDA-only extras; 0 switches each off, and with all three off the kernel
     * draws exactly what the matched shader draws. */
    float burn_gain;     /* strength of the burn ghost                      */
    float halation;      /* wide glow: light scattered inside the glass     */
    float mask;          /* aperture grille depth, 0..1                     */
    float beam;          /* bright strokes swell: spill weighted by light   */
    float degauss;       /* 0 = none; 1 fading to 0 over a second at power-on */
    float degauss_phase; /* radians, advances with time                     */
    /* the tracking bar as a disturbance: while it passes, the picture under it
     * bends sideways, tears a little line by line, splits its colours and
     * picks up static. All zero = the plain brightening band of the first pass. */
    float track_bend;    /* px the picture is pushed sideways at the bar's centre */
    float track_noise;   /* 0..1 line tearing and static inside the bar      */
    float track_chroma;  /* extra colour split inside the bar                */
    float jiggle_px;     /* this frame's vertical hop of the picture, in px  */
    float seed;          /* changes every frame; drives the static           */
} OpticsParams;

#endif
