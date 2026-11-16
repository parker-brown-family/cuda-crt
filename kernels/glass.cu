// omarchy-crt lab: everything that is drawn FROM the phosphor state. Stateless
// (apart from vector mode's deposit), so there is no CPU reference to match and
// this file is compiled with fast math: its check is the picture, against the
// matched shader. The state itself is kernels/phosphor.cu.
#include "phosphor_params.h"
#include "phosphor_half.h"
#include "tube_map.h"

__device__ inline float linear_to_srgb(float c) {
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

// ---- vector mode: a beam that draws strokes straight into the phosphor -------
// No raster and no frame: the host hands over the line segments the beam swept
// during this frame and each one deposits light where it passed, in a Gaussian
// the width of the spot. The ordinary update then lets it fade, so a moving
// figure leaves a continuous trail. segs: nseg x (x0, y0, x1, y1) in tile pixels.
extern "C" __global__ void beam_deposit(cudaSurfaceObject_t state, int w, int h, const float *segs, int nseg,
                                        float sigma, float energy, float cr, float cg, float cb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float px = x + 0.5f, py = y + 0.5f, reach = 3.5f * sigma, inv = 1.0f / (2.0f * sigma * sigma), e = 0.0f;
    for (int i = 0; i < nseg; i++) {
        float x0 = segs[4 * i], y0 = segs[4 * i + 1], x1 = segs[4 * i + 2], y1 = segs[4 * i + 3];
        if (px < fminf(x0, x1) - reach || px > fmaxf(x0, x1) + reach || py < fminf(y0, y1) - reach || py > fmaxf(y0, y1) + reach)
            continue;
        float dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy;
        float t = len2 > 0.0f ? fminf(fmaxf(((px - x0) * dx + (py - y0) * dy) / len2, 0.0f), 1.0f) : 0.0f;
        float qx = px - (x0 + t * dx), qy = py - (y0 + t * dy);
        e += expf(-(qx * qx + qy * qy) * inv);
    }
    if (e < 1.0e-4f) return;
    e *= energy;
    ushort4 raw = surf2Dread<ushort4>(state, x * (int)sizeof(ushort4), y);
    // phosphor saturates: light past full brightness is kept (it feeds the glow) but capped
    raw.x = ph_f2h(fminf(ph_h2f(raw.x) + e * cr, 4.0f));
    raw.y = ph_f2h(fminf(ph_h2f(raw.y) + e * cg, 4.0f));
    raw.z = ph_f2h(fminf(ph_h2f(raw.z) + e * cb, 4.0f));
    surf2Dwrite(raw, state, x * (int)sizeof(ushort4), y);
}

// ---- halation: light scattered inside the glass ----------------------------
// Three tiny kernels on a 1/8-size image: average each 8x8 block of lit
// phosphor (16 bilinear taps, each the exact mean of 2x2 pixels), then a
// 9-tap Gaussian across and down. 1/64 of the pixels, so it costs almost nothing.
extern "C" __global__ void glow_down8(cudaTextureObject_t state_tex, cudaSurfaceObject_t glow, int gw, int gh,
                                      float knee) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gw || y >= gh) return;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            float4 t = tex2D<float4>(state_tex, 8.0f * x + 1.0f + 2.0f * i, 8.0f * y + 1.0f + 2.0f * j);
            r += fmaxf(t.x - knee, 0.0f); g += fmaxf(t.y - knee, 0.0f); b += fmaxf(t.z - knee, 0.0f);
        }
    surf2Dwrite(make_float4(r * 0.0625f, g * 0.0625f, b * 0.0625f, 1.0f), glow, x * (int)sizeof(float4), y);
}
extern "C" __global__ void glow_blur(cudaTextureObject_t src, cudaSurfaceObject_t dst, int gw, int gh, int dx, int dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gw || y >= gh) return;
    // sigma 2.5 texels = 20 screen pixels
    const float k[5] = {0.16014f, 0.14783f, 0.11626f, 0.07791f, 0.04450f};
    float4 c = tex2D<float4>(src, x + 0.5f, y + 0.5f);
    float r = c.x * k[0], g = c.y * k[0], b = c.z * k[0];
    for (int i = 1; i < 5; i++) {
        float4 p = tex2D<float4>(src, x + 0.5f + dx * i, y + 0.5f + dy * i);
        float4 q = tex2D<float4>(src, x + 0.5f - dx * i, y + 0.5f - dy * i);
        r += (p.x + q.x) * k[i]; g += (p.y + q.y) * k[i]; b += (p.z + q.z) * k[i];
    }
    surf2Dwrite(make_float4(r, g, b, 1.0f), dst, x * (int)sizeof(float4), y);
}

// ---- the glass -------------------------------------------------------------
struct v2 { float x, y; };
__device__ inline v2 V(float x, float y) { v2 r; r.x = x; r.y = y; return r; }
__device__ inline float clamp01(float v) { return fminf(fmaxf(v, 0.0f), 1.0f); }
__device__ inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline float sstep(float e0, float e1, float v) {
    float t = clamp01((v - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}
// the tube's curve: tube_map.h is the only place it is written down
__device__ inline v2 tube_warp(v2 l, TubeMap m) {
    v2 r;
    tube_gather(m, l.x, l.y, &r.x, &r.y);
    return r;
}

// state_tex: the tile's phosphor state, bilinear, clamped, texel coordinates
// out:       the output frame (RGBA8), mapped from OpenGL
// glow_tex:  the tile's 1/8-size halation image (only read when o.halation > 0)
extern "C" __global__ void crt_optics_tex(cudaTextureObject_t state_tex, cudaSurfaceObject_t out,
                                          int ox, int oy, int w, int h, OpticsParams o,
                                          cudaTextureObject_t glow_tex) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fw = (float)w, fh = (float)h;
    v2 l0 = V((x + 0.5f) / fw, (y + 0.5f) / fh);
    TubeMap tm;
    tm.k1 = o.k1 * o.curv; tm.k2 = o.k2 * o.curv; tm.core = o.core; tm.inset = o.inset;
    v2 l2 = tube_warp(l0, tm);
    // With an inset the window is drawn smaller and the glass's edge sits on padding
    // around it. `g2` is the position on the GLASS (0..1 = its edge); l2 stays the
    // position in the WINDOW. With no inset the two are the same thing.
    float pad = o.inset * 0.5f;
    v2 g2 = V((l2.x + pad) / (1.0f + 2.0f * pad), (l2.y + pad) / (1.0f + 2.0f * pad));

    // the glass sits in a rim: a rounded rectangle in SOURCE space, so the warp
    // carries its corners round with the picture (terminal-delight's td-glass-rim)
    float inb = 1.0f, shade = 1.0f, lip = 0.0f;
    if (o.curv > 0.0f) {
        float short_side = fminf(fw, fh);
        float corner = fminf(fmaxf(short_side * 0.055f, 10.0f), 72.0f);
        float qx = fabsf(g2.x * fw - fw * 0.5f) - fw * 0.5f + corner;
        float qy = fabsf(g2.y * fh - fh * 0.5f) - fh * 0.5f + corner;
        float lip_w = fminf(fmaxf(short_side * 0.045f, 8.0f), 48.0f);
        float edge_x = fminf(g2.x, 1.0f - g2.x) * fw, edge_y = fminf(g2.y, 1.0f - g2.y) * fh;
        // Deep inside the glass (most of the picture) the rim contributes exactly
        // nothing: no lip shadow, no corner shade, no edge. Skip its arithmetic there.
        if (fmaxf(qx, qy) < -(lip_w + corner) && (edge_x >= lip_w * 3.0f || edge_y >= lip_w * 3.0f)) goto rim_done;
        float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f);
        float sd = fminf(fmaxf(qx, qy), 0.0f) + sqrtf(mx * mx + my * my) - corner;
        const float aa = 1.5f;
        inb = 1.0f - sstep(-aa, aa, sd);
        float rim = 1.0f - sstep(0.0f, lip_w, fmaxf(-sd, 0.0f));
        float ly = clamp01(g2.y);
        float from_top = mixf(1.10f, 0.85f, ly);
        float ax = fminf(g2.x, 1.0f - g2.x) * fw;
        float ay = fminf(g2.y, 1.0f - g2.y) * fh;
        float cornerness = (1.0f - sstep(0.0f, lip_w * 3.0f, ax)) * (1.0f - sstep(0.0f, lip_w * 3.0f, ay));
        shade = 1.0f - fminf(fmaxf(0.85f * rim * sqrtf(rim) * from_top + 0.30f * cornerness, 0.0f), 0.94f);
        float lip_d = (sd - aa * 1.5f) / fmaxf(aa * 1.2f, 1.0f);
        lip = expf(-lip_d * lip_d) * mixf(0.03f, 0.09f, ly) * (sd >= 0.0f ? 1.0f : 0.0f);
    }
rim_done:;

    v2 L = V(clamp01(l2.x), clamp01(l2.y));
    float r2 = (L.x - 0.5f) * (L.x - 0.5f) + (L.y - 0.5f) * (L.y - 0.5f);
    // The bar's bend and tearing, the jiggle and the switch-on wobble move the picture
    // for a moment. tube_map.h does it, so the pointer can follow the same picture.
    float dg = o.degauss;
    TubeMotion mo;
    mo.degauss = o.degauss; mo.degauss_phase = o.degauss_phase; mo.tracking = o.tracking; mo.band_center = o.band_center;
    mo.band_h = o.band_h; mo.track_bend = o.track_bend; mo.track_noise = o.track_noise; mo.jiggle_px = o.jiggle_px; mo.seed = o.seed;
    float bar = 0.0f, tear = 0.0f;
    bool moved = tube_disturb(tm, mo, fw, fh, &l0.x, &l0.y, &bar, &tear) != 0;
    v2 lp = moved ? tube_warp(l0, tm) : l2;
    // outside the window (the padding an inset leaves, or past the bend) there is no picture
    float in_window = (o.inset > 0.0f && (lp.x < 0.0f || lp.x > 1.0f || lp.y < 0.0f || lp.y > 1.0f)) ? 0.0f : 1.0f;
    float sx = clamp01(lp.x) * fw, sy = clamp01(lp.y) * fh;

    // chromatic aberration: R and B through slightly different warps near the rim
    float a = o.aberr * 0.004f * r2 + dg * 0.010f * sinf(o.degauss_phase * 0.7f) + o.track_chroma * bar * 0.0035f;
    v2 lr = tube_warp(V(l0.x + a, l0.y), tm);
    v2 lb = tube_warp(V(l0.x - a, l0.y), tm);
    float4 cc = tex2D<float4>(state_tex, sx, sy);
    float cr = tex2D<float4>(state_tex, clamp01(lr.x) * fw, clamp01(lr.y) * fh).x;
    float cb = tex2D<float4>(state_tex, clamp01(lb.x) * fw, clamp01(lb.y) * fh).z;
    float lin[3] = {cr * in_window, cc.y * in_window, cb * in_window};

    // 4-tap glow, in linear light: persistence trails bleed the way lit phosphor does
    if (o.glow > 0.001f) {
        float g = o.glow * 0.18f * 0.25f, d = o.glow_px;
        float4 t0 = tex2D<float4>(state_tex, sx + d, sy);
        float4 t1 = tex2D<float4>(state_tex, sx - d, sy);
        float4 t2 = tex2D<float4>(state_tex, sx, sy + d);
        float4 t3 = tex2D<float4>(state_tex, sx, sy - d);
        lin[0] += g * (t0.x + t1.x + t2.x + t3.x);
        lin[1] += g * (t0.y + t1.y + t2.y + t3.y);
        lin[2] += g * (t0.z + t1.z + t2.z + t3.z);
        // beam width grows with brightness: a neighbour spills in proportion to its
        // own light, so bright strokes swell and dim ones stay sharp
        if (o.beam > 0.0f) {
            float b = o.beam * 0.25f;
            float w0 = fmaxf(t0.x, fmaxf(t0.y, t0.z)), w1 = fmaxf(t1.x, fmaxf(t1.y, t1.z));
            float w2 = fmaxf(t2.x, fmaxf(t2.y, t2.z)), w3 = fmaxf(t3.x, fmaxf(t3.y, t3.z));
            lin[0] += b * (t0.x * w0 + t1.x * w1 + t2.x * w2 + t3.x * w3);
            lin[1] += b * (t0.y * w0 + t1.y * w1 + t2.y * w2 + t3.y * w3);
            lin[2] += b * (t0.z * w0 + t1.z * w1 + t2.z * w2 + t3.z * w3);
        }
    }

    // burn: a faint ghost of what sat here long enough to charge it, in the tube's own colour
    if (o.burn_gain > 0.0f) {
        float lum = 0.2126f * cc.x + 0.7152f * cc.y + 0.0722f * cc.z;
        float ghost = o.burn_gain * fmaxf(cc.w - lum, 0.0f);
        for (int c = 0; c < 3; c++) lin[c] += ghost * o.phosphor[c];
    }
    // halation: the wide soft glow of light scattered in the glass
    if (o.halation > 0.0f) {
        float4 gl = tex2D<float4>(glow_tex, sx * 0.125f, sy * 0.125f);
        lin[0] += o.halation * gl.x; lin[1] += o.halation * gl.y; lin[2] += o.halation * gl.z;
    }

    // from here the dials are display-referred, exactly as crt-glass.frag has them
    float col[3] = {linear_to_srgb(lin[0]), linear_to_srgb(lin[1]), linear_to_srgb(lin[2])};
    float fl = o.flicker_mul;

    float scanA = o.scan * fl;
    if (scanA > 0.001f) {
        float row = fmodf(L.y * fh, o.scan_step);
        if (row < 1.0f) {
            for (int c = 0; c < 3; c++) col[c] = mixf(col[c], 0.0f, scanA);
        } else if (row < 2.0f) {
            for (int c = 0; c < 3; c++) col[c] = mixf(col[c], o.phosphor[c], scanA * 0.22f);
        }
    }
    if (o.bloom > 0.001f) {
        float wash = 0.05f * o.bloom * sstep(0.05f, 0.42f, L.y);
        for (int c = 0; c < 3; c++) col[c] = mixf(col[c], o.phosphor[c], wash);
    }
    // the rolling tracking band, plus its band-local scanline pressure
    if (o.tracking > 0.001f && o.band_center > -1.0e8f) {
        float d = 1.0f - clamp01(fabsf(L.y * fh - o.band_center) / (o.band_h * 0.5f));
        if (d > 0.0f) {
            float add = d * d * 0.05f * o.tracking;
            float core = d > 0.92f ? 0.018f * o.tracking : 0.0f;
            float dark = 1.0f - 0.05f * o.tracking * d * 0.5f;
            for (int c = 0; c < 3; c++) col[c] = (col[c] + o.phosphor[c] * add + core) * dark;
        }
    }
    // static inside the bar: per-pixel noise, brighter where the bar is deepest
    if (bar > 0.0f && o.track_noise > 0.0f) {
        float hn = sinf((x + 1.0f) * 91.345f + (y + 1.0f) * 47.853f + o.seed * 311.7f) * 43758.5453f;
        float st = o.track_noise * bar * 0.10f * (hn - floorf(hn)) * (0.6f + 0.8f * fabsf(tear));
        for (int c = 0; c < 3; c++) col[c] += st;
    }
    if (o.glare > 0.001f) {
        float gx = (L.x - 0.18f) / 0.34f, gy = (L.y - 0.10f) / 0.18f;
        float hotspot = expf(-(gx * gx + gy * gy) * 2.2f);
        float band = (1.0f - sstep(0.0f, 0.10f, fabsf(L.y + L.x * 0.22f - 0.17f)))
                   * (1.0f - sstep(0.0f, 0.72f, L.x)) * (1.0f - sstep(0.0f, 0.42f, L.y));
        float amount = clamp01((hotspot * 0.58f + band * 0.34f) * o.glare);
        for (int c = 0; c < 3; c++) col[c] += o.glare_tint[c] * amount;
    }
    if (o.specular > 0.001f) {
        float sxx = (L.x - 0.05f) / 0.13f, syy = (L.y - 0.04f) / 0.10f;
        float sp = 0.05f * o.specular * expf(-(sxx * sxx + syy * syy)) * fl;
        for (int c = 0; c < 3; c++) col[c] += sp;
    }
    // aperture grille: vertical red, green and blue stripes, one screen pixel each,
    // with the lost light given back so the picture does not dim
    if (o.mask > 0.0f) {
        int stripe = (ox + x) % 3;
        float dim = 1.0f - o.mask, gain = 3.0f / (1.0f + 2.0f * dim);
        for (int c = 0; c < 3; c++) col[c] *= (c == stripe ? 1.0f : dim) * gain;
    }
    float vig = 1.0f - o.vign * fl * 1.72f * r2;
    uchar4 px;
    unsigned char *pc = (unsigned char *)&px;
    for (int c = 0; c < 3; c++) {
        float v = clamp01(col[c] * vig * shade * inb + lip);
        pc[c] = (unsigned char)(v * 255.0f + 0.5f);
    }
    px.w = 255;
    surf2Dwrite(px, out, (ox + x) * (int)sizeof(uchar4), oy + y);
}

// passthrough for a tile with no surface (the safe fallback path, on the GPU)
extern "C" __global__ void passthrough_tex(cudaTextureObject_t frame, cudaSurfaceObject_t out,
                                           int ox, int oy, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 in = tex2D<float4>(frame, ox + x + 0.5f, oy + y + 0.5f);
    uchar4 px = make_uchar4((unsigned char)(in.x * 255.0f + 0.5f), (unsigned char)(in.y * 255.0f + 0.5f),
                            (unsigned char)(in.z * 255.0f + 0.5f), 255);
    surf2Dwrite(px, out, (ox + x) * (int)sizeof(uchar4), oy + y);
}
