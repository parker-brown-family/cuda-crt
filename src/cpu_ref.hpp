// The CPU reference: the same arithmetic as kernels/phosphor.cu, one pixel at a
// time, in plain C++. The CUDA result is judged against this.
#pragma once
#include "phosphor_params.h"
#include "phosphor_half.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// The tunables a person turns. Example values, not calibrated CRT physics.
struct PhosphorTuning {
    double tau_rise = 0.001;                   // seconds
    double tau_decay[3] = {0.050, 0.050, 0.050};
    // Half of the smallest step an 8-bit sRGB display can show near black
    // (1/255 sRGB = 0.000304 linear). Below it the trail is invisible, so the
    // state snaps to the target and the tile can go idle.
    float eps = 0.00015f;
};

// a = 1 - exp(-dt / tau), from real elapsed time. fresh = a surface that has no
// history yet (new, resized, shown again): it adopts the frame as it is.
inline PhosphorStep make_step(const PhosphorTuning &t, double dt, bool fresh) {
    PhosphorStep p;
    p.a_rise = fresh ? 1.0f : (float)-std::expm1(-dt / t.tau_rise);
    for (int c = 0; c < 3; c++) p.a_decay[c] = fresh ? 1.0f : (float)-std::expm1(-dt / t.tau_decay[c]);
    p.eps = t.eps;
    return p;
}

inline float cpu_srgb_to_linear(float c) {
    return c <= 0.04045f ? c * (1.0f / 12.92f) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

inline float cpu_phosphor_step(float prev, float target, float a_rise, float a_decay, float eps, int *active) {
    float a = target > prev ? a_rise : a_decay;
    float next = a >= 1.0f ? target : prev + a * (target - prev);
    if (!(fabsf(target - next) >= eps)) return target;
    *active = 1;
    return next;
}

// frame: RGBA8 sRGB, frame_w px per row. state: RGBA32F, w*h. Returns active.
inline int cpu_phosphor_update(const uint8_t *frame, int frame_w, int ox, int oy, float *state, int w, int h,
                               const PhosphorStep &p) {
    int active = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const uint8_t *px = frame + 4 * ((size_t)(oy + y) * frame_w + (ox + x));
            float *s = state + 4 * ((size_t)y * w + x);
            for (int c = 0; c < 3; c++)
                s[c] = cpu_phosphor_step(s[c], cpu_srgb_to_linear(px[c] * (1.0f / 255.0f)), p.a_rise, p.a_decay[c],
                                         p.eps, &active);
            s[3] = 1.0f;
        }
    return active;
}

// ---- version 2: half-precision state with a burn term ----------------------
struct BurnTuning {
    double tau_rise = 0.5;   // seconds of steady light to charge the burn
    double tau_decay = 0.8;  // seconds for the ghost to drain
    float gain = 0.008f;     // how strongly the ghost is drawn; 0 = no burn at all
};

inline PhosphorStepV2 make_step_v2(const PhosphorTuning &t, const BurnTuning &b, double dt, bool fresh) {
    PhosphorStepV2 p;
    p.a_rise = fresh ? 1.0f : (float)-std::expm1(-dt / t.tau_rise);
    for (int c = 0; c < 3; c++) p.a_decay[c] = fresh ? 1.0f : (float)-std::expm1(-dt / t.tau_decay[c]);
    p.eps = t.eps;
    p.a_burn_rise = (float)-std::expm1(-dt / b.tau_rise);
    p.a_burn_decay = (float)-std::expm1(-dt / b.tau_decay);
    // The ghost is drawn at `gain`, so it is invisible once the burn is within eps/gain
    // of where it is going. With no gain the burn is never drawn: let it settle at once.
    p.eps_burn = b.gain > 0.0f ? std::min(t.eps / b.gain, 0.05f) : 4.0f;
    return p;
}

inline uint16_t cpu_phosphor_step_h(uint16_t hprev, float target, float a_rise, float a_decay, float eps, int *active) {
    float prev = ph_h2f(hprev);
    float a = target > prev ? a_rise : a_decay;
    float next = a >= 1.0f ? target : prev + a * (target - prev);
    uint16_t ht = ph_f2h(target), hn = ph_f2h(next);
    if (!(fabsf(target - next) >= eps) || hn == hprev) return ht;
    if (hn != ht) *active = 1;
    return hn;
}

// state: RGBA16F as raw halves, w*h*4. Returns active.
inline int cpu_phosphor_update_v2(const uint8_t *frame, int frame_w, int ox, int oy, uint16_t *state, int w, int h,
                                  const PhosphorStepV2 &p) {
    int active = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const uint8_t *px = frame + 4 * ((size_t)(oy + y) * frame_w + (ox + x));
            uint16_t *s = state + 4 * ((size_t)y * w + x);
            float t[3];
            for (int c = 0; c < 3; c++) {
                t[c] = cpu_srgb_to_linear(px[c] * (1.0f / 255.0f));
                s[c] = cpu_phosphor_step_h(s[c], t[c], p.a_rise, p.a_decay[c], p.eps, &active);
            }
            float lum = 0.2126f * t[0] + 0.7152f * t[1] + 0.0722f * t[2];
            s[3] = cpu_phosphor_step_h(s[3], lum, p.a_burn_rise, p.a_burn_decay, p.eps_burn, &active);
        }
    return active;
}
