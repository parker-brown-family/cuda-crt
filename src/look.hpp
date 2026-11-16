// The look: one set of dials for every path, and the per-frame scalars the
// clock drives. No OpenGL and no CUDA in here, so a compositor plugin can use it.
#pragma once
#include "phosphor_params.h"
#include "cpu_ref.hpp"
#include <algorithm>
#include <cmath>

// ---- the look: one set of dials for every path ------------------------------
struct Look {
    PhosphorTuning phosphor;
    float curv = 1.0f, k1 = 0.14f, k2 = 0.06f, aberr = 0.55f, scan = 0.22f, scan_step = 4.0f, glow = 0.40f,
          glow_px = 2.0f, bloom = 0.35f, vign = 0.32f, specular = 0.50f, glare = 0.30f, tracking = 0.60f,
          track_rest = 9.0f, track_sweep = 8.0f, band_h = 264.0f, flicker = 0.35f;
    float accent[3] = {0.133f, 0.773f, 0.369f}, glare_tint[3] = {0.72f, 1.00f, 0.78f};
    // version 2 extras (CUDA only)
    BurnTuning burn;
    float halation = 0.30f, glow_knee = 0.02f, mask = 0.0f, beam = 0.5f;
    float degauss_at = -1.0f;  // seconds into the run at which the tube is "switched on"; < 0 = never
    // the tracking bar as a disturbance, and the screen's own unsteadiness
    float track_bend = 6.0f, track_noise = 0.35f, track_chroma = 1.0f, jiggle = 0.15f;
};

inline float hash1(float n) { float v = sinf(n * 127.1f) * 43758.5453f; return v - floorf(v); }

// This frame's vertical hop of the picture, px. A slow breath plus, now and then,
// a two-frame jump: crt-glass.frag's JIGGLE, made continuous so the dial has a feel.
inline float jiggle_px(const Look &k, float t) {
    if (k.jiggle < 0.001f) return 0.0f;
    float breath = 0.35f * sinf(t * 7.3f) + 0.25f * sinf(t * 23.1f + 1.7f);
    float win = floorf(t / 4.0f), start = win * 4.0f + 0.5f + hash1(win + 7.0f) * 3.0f, hop = 0.0f;
    if (t > start && t < start + 0.07f) hop = (hash1(win + 3.0f) - 0.5f) * 9.0f;
    return k.jiggle * (breath + hop);
}

// crt-glass.frag flicker_mul(): a 0.45 s burst of five steps inside every 13 s window
inline float flicker_mul(const Look &k, float t) {
    if (k.flicker < 0.001f) return 1.0f;
    float win = floorf(t / 13.0f);
    float start = win * 13.0f + 2.0f + hash1(win) * 8.0f;
    if (t < start || t > start + 0.45f) return 1.0f;
    static const float ladder[5] = {0.86f, 1.06f, 0.90f, 1.03f, 0.95f};
    int ph = std::min(4, (int)((t - start) / 0.09f));
    return 1.0f + (ladder[ph] - 1.0f) * k.flicker;
}
// crt-glass.frag tracking_add(): where the band's centre is, px down the tile, or none
inline float band_center(const Look &k, float t, float tile_h) {
    if (k.tracking < 0.001f) return -1.0e9f;
    float sweep = k.track_sweep * (1.40f - 0.50f * k.tracking);
    float tc = fmodf(t, sweep + k.track_rest);
    if (tc > sweep) return -1.0e9f;
    return -k.band_h + (tc / sweep) * (tile_h + k.band_h * 2.0f);
}
// extras = false gives the version 1 picture, which is what the matched shader draws
// Seconds until the GLASS next needs a frame for its own reasons (the tracking
// band, a flicker burst), 0 = it is moving now. The surface's `active` flag
// covers the phosphor only; a host that wants to sleep must honour both.
inline float glass_next_wake(const Look &k, float t) {
    float wake = 1.0e9f;
    if (k.tracking >= 0.001f) {
        float sweep = k.track_sweep * (1.40f - 0.50f * k.tracking), period = sweep + k.track_rest;
        float tc = fmodf(t, period);
        wake = tc <= sweep ? 0.0f : period - tc;
    }
    if (k.flicker >= 0.001f) {
        float win = floorf(t / 13.0f), start = win * 13.0f + 2.0f + hash1(win) * 8.0f;
        if (t < start) wake = std::min(wake, start - t);
        else if (t <= start + 0.45f) wake = 0.0f;
        else wake = std::min(wake, (win + 1.0f) * 13.0f + 2.0f + hash1(win + 1.0f) * 8.0f - t);
    }
    if (k.jiggle >= 0.001f) wake = 0.0f;  // the picture never quite holds still
    if (k.degauss_at >= 0.0f && t < k.degauss_at + 1.6f) wake = t >= k.degauss_at ? 0.0f : std::min(wake, k.degauss_at - t);
    return wake;
}

inline OpticsParams make_optics(const Look &k, float t, int tile_h, bool extras = false) {
    OpticsParams o;
    o.burn_gain = extras ? k.burn.gain : 0.0f;
    o.halation = extras ? k.halation : 0.0f;
    o.mask = extras ? k.mask : 0.0f;
    o.beam = extras ? k.beam : 0.0f;
    float since = t - k.degauss_at;
    bool dg = extras && k.degauss_at >= 0.0f && since >= 0.0f && since < 1.6f;
    o.degauss = dg ? expf(-since / 0.32f) : 0.0f;
    o.degauss_phase = dg ? since * 42.0f : 0.0f;
    o.track_bend = extras ? k.track_bend : 0.0f;
    o.track_noise = extras ? k.track_noise : 0.0f;
    o.track_chroma = extras ? k.track_chroma : 0.0f;
    o.jiggle_px = extras ? jiggle_px(k, t) : 0.0f;
    o.seed = fmodf(t * 60.0f, 4096.0f);
    o.curv = k.curv; o.k1 = k.k1; o.k2 = k.k2; o.aberr = k.aberr; o.scan = k.scan; o.scan_step = k.scan_step;
    o.glow = k.glow; o.glow_px = k.glow_px; o.bloom = k.bloom; o.vign = k.vign; o.specular = k.specular;
    o.glare = k.glare; o.tracking = k.tracking; o.band_h = k.band_h;
    o.band_center = band_center(k, t, (float)tile_h);
    o.flicker_mul = flicker_mul(k, t);
    for (int c = 0; c < 3; c++) { o.phosphor[c] = k.accent[c]; o.glare_tint[c] = k.glare_tint[c]; }
    return o;
}
