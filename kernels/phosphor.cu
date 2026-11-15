// omarchy-crt lab: the virtual phosphor surface, in CUDA.
//
// Two kernels per tile, one GPU thread per pixel:
//   phosphor_update_*  the only stateful step. Linear-light RGB per pixel,
//                      pulled toward the incoming frame with a fast rise and a
//                      slow decay. Updated IN PLACE: a thread reads and writes
//                      its own pixel and nobody else's, so no second buffer.
//   crt_optics_tex     the glass, stateless: reads the phosphor state through
//                      the tube's curve and writes display pixels.
// They stay separate because the optics gather from neighbours' state, which
// the update is rewriting.
//
// Compiled at run time by NVRTC (no nvcc on the box), loaded through the
// driver API. `extern "C"` keeps the names findable.
#include "phosphor_params.h"

__device__ inline float srgb_to_linear(float c) {
    return c <= 0.04045f ? c * (1.0f / 12.92f) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}
__device__ inline float linear_to_srgb(float c) {
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

// tau = target > previous ? tau_rise : tau_decay;  next = previous + a (target - previous)
__device__ inline float phosphor_step(float prev, float target, float a_rise, float a_decay,
                                      float eps, int *active) {
    float a = target > prev ? a_rise : a_decay;
    float next = prev + a * (target - prev);
    if (fabsf(target - next) < eps) return target;  // settled: exact, so idle is detectable
    *active = 1;
    return next;
}

// ---- Stage 1: plain device buffers, host copies either side ----------------
// frame:  RGBA8 sRGB, frame_w pixels per row, row 0 on top
// state:  RGBA32F linear light, w*h, this tile only
// active: one int for the whole tile; any thread still moving writes 1
extern "C" __global__ void phosphor_update_buf(const unsigned char *frame, int frame_w,
                                               int ox, int oy, float *state, int w, int h,
                                               PhosphorStep p, int *active) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned char *px = frame + 4 * ((size_t)(oy + y) * frame_w + (ox + x));
    float *s = state + 4 * ((size_t)y * w + x);
    int moving = 0;
    for (int c = 0; c < 3; c++)
        s[c] = phosphor_step(s[c], srgb_to_linear(px[c] * (1.0f / 255.0f)), p.a_rise,
                             p.a_decay[c], p.eps, &moving);
    s[3] = 1.0f;
    if (moving) *active = 1;  // benign race: every writer writes the same value
}

// ---- Stage 2: GPU-resident, no host copies ---------------------------------
// frame: the compositor's texture, mapped from OpenGL (RGBA8, point sampled)
// state: a CUDA array owned by this tile, read and written through a surface
extern "C" __global__ void phosphor_update_tex(cudaTextureObject_t frame, int ox, int oy,
                                               cudaSurfaceObject_t state, int w, int h,
                                               PhosphorStep p, int *active) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 in = tex2D<float4>(frame, ox + x + 0.5f, oy + y + 0.5f);
    float4 s = surf2Dread<float4>(state, x * (int)sizeof(float4), y);
    int moving = 0;
    s.x = phosphor_step(s.x, srgb_to_linear(in.x), p.a_rise, p.a_decay[0], p.eps, &moving);
    s.y = phosphor_step(s.y, srgb_to_linear(in.y), p.a_rise, p.a_decay[1], p.eps, &moving);
    s.z = phosphor_step(s.z, srgb_to_linear(in.z), p.a_rise, p.a_decay[2], p.eps, &moving);
    s.w = 1.0f;
    surf2Dwrite(s, state, x * (int)sizeof(float4), y);
    if (moving) *active = 1;
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
// the tube's barrel: a gather, tile-local 0..1 in, tile-local out
__device__ inline v2 tube_warp(v2 l, float k1, float k2) {
    float cx = l.x - 0.5f, cy = l.y - 0.5f;
    float r2 = cx * cx + cy * cy;
    float f = 1.0f + k1 * r2 + k2 * r2 * r2;
    return V(0.5f + cx * f, 0.5f + cy * f);
}

// state_tex: the tile's phosphor state, bilinear, clamped, texel coordinates
// out:       the output frame (RGBA8), mapped from OpenGL
extern "C" __global__ void crt_optics_tex(cudaTextureObject_t state_tex, cudaSurfaceObject_t out,
                                          int ox, int oy, int w, int h, OpticsParams o) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fw = (float)w, fh = (float)h;
    v2 l0 = V((x + 0.5f) / fw, (y + 0.5f) / fh);
    float k1 = o.k1 * o.curv, k2 = o.k2 * o.curv;
    v2 l2 = tube_warp(l0, k1, k2);

    // the glass sits in a rim: a rounded rectangle in SOURCE space, so the warp
    // carries its corners round with the picture (terminal-delight's td-glass-rim)
    float inb = 1.0f, shade = 1.0f, lip = 0.0f;
    if (o.curv > 0.0f) {
        float short_side = fminf(fw, fh);
        float corner = fminf(fmaxf(short_side * 0.055f, 10.0f), 72.0f);
        float qx = fabsf(l2.x * fw - fw * 0.5f) - fw * 0.5f + corner;
        float qy = fabsf(l2.y * fh - fh * 0.5f) - fh * 0.5f + corner;
        float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f);
        float sd = fminf(fmaxf(qx, qy), 0.0f) + sqrtf(mx * mx + my * my) - corner;
        const float aa = 1.5f;
        inb = 1.0f - sstep(-aa, aa, sd);
        float lip_w = fminf(fmaxf(short_side * 0.045f, 8.0f), 48.0f);
        float rim = 1.0f - sstep(0.0f, lip_w, fmaxf(-sd, 0.0f));
        float ly = clamp01(l2.y);
        float from_top = mixf(1.10f, 0.85f, ly);
        float ax = fminf(l2.x, 1.0f - l2.x) * fw;
        float ay = fminf(l2.y, 1.0f - l2.y) * fh;
        float cornerness = (1.0f - sstep(0.0f, lip_w * 3.0f, ax)) * (1.0f - sstep(0.0f, lip_w * 3.0f, ay));
        shade = 1.0f - fminf(fmaxf(0.85f * rim * sqrtf(rim) * from_top + 0.30f * cornerness, 0.0f), 0.94f);
        float lip_d = (sd - aa * 1.5f) / fmaxf(aa * 1.2f, 1.0f);
        lip = expf(-lip_d * lip_d) * mixf(0.03f, 0.09f, ly) * (sd >= 0.0f ? 1.0f : 0.0f);
    }

    v2 L = V(clamp01(l2.x), clamp01(l2.y));
    float r2 = (L.x - 0.5f) * (L.x - 0.5f) + (L.y - 0.5f) * (L.y - 0.5f);
    float sx = L.x * fw, sy = L.y * fh;

    // chromatic aberration: R and B through slightly different warps near the rim
    float a = o.aberr * 0.004f * r2;
    v2 lr = tube_warp(V(l0.x + a, l0.y), k1, k2);
    v2 lb = tube_warp(V(l0.x - a, l0.y), k1, k2);
    float4 cc = tex2D<float4>(state_tex, sx, sy);
    float cr = tex2D<float4>(state_tex, clamp01(lr.x) * fw, clamp01(lr.y) * fh).x;
    float cb = tex2D<float4>(state_tex, clamp01(lb.x) * fw, clamp01(lb.y) * fh).z;
    float lin[3] = {cr, cc.y, cb};

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
