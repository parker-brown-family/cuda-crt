// omarchy-crt lab: the virtual phosphor surface, in CUDA.
//
// This file is the STATE: the only stateful step, and the part held to the CPU
// reference bit for bit (compiled without fused multiply-adds for that reason).
// The glass, the halation and vector mode live in kernels/glass.cu, compiled
// with fast math because nothing there is compared to a CPU.
//
// Two kinds of kernel per tile, one GPU thread per pixel:
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
#include "phosphor_half.h"

__device__ inline float srgb_to_linear(float c) {
    return c <= 0.04045f ? c * (1.0f / 12.92f) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

// tau = target > previous ? tau_rise : tau_decay;  next = previous + a (target - previous)
__device__ inline float phosphor_step(float prev, float target, float a_rise, float a_decay,
                                      float eps, int *active) {
    float a = target > prev ? a_rise : a_decay;
    // a weight of 1 means "adopt the frame": never read arithmetic off the old state,
    // so a surface that starts with garbage (or a NaN) cannot keep it
    float next = a >= 1.0f ? target : prev + a * (target - prev);
    if (!(fabsf(target - next) >= eps)) return target;  // settled (or not a number): exact
    *active = 1;
    return next;
}

// "Is this tile still moving?" as one int. Every thread writing 1 to the same
// address is a data race even though the values agree, so: threads raise a flag
// in their block's shared memory with an atomic, and one thread per block ORs
// that into the tile's int. All threads of a block must reach both barriers,
// which is why the kernels below guard their work with `inside` rather than
// returning early.
#define ACTIVE_BEGIN()                                              \
    __shared__ int blk_moving;                                      \
    if (threadIdx.x == 0 && threadIdx.y == 0) blk_moving = 0;       \
    __syncthreads();
#define ACTIVE_RAISE() atomicOr(&blk_moving, 1)
#define ACTIVE_END(active)                                          \
    __syncthreads();                                                \
    if (threadIdx.x == 0 && threadIdx.y == 0 && blk_moving) atomicOr(active, 1);

// ---- Stage 1: plain device buffers, host copies either side ----------------
// frame:  RGBA8 sRGB, frame_w pixels per row, row 0 on top
// state:  RGBA32F linear light, w*h, this tile only
// active: one int for the whole tile; any thread still moving writes 1
extern "C" __global__ void phosphor_update_buf(const unsigned char *frame, int frame_w,
                                               int ox, int oy, float *state, int w, int h,
                                               PhosphorStep p, int *active) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    ACTIVE_BEGIN();
    if (x < w && y < h) {
        const unsigned char *px = frame + 4 * ((size_t)(oy + y) * frame_w + (ox + x));
        float *s = state + 4 * ((size_t)y * w + x);
        int moving = 0;
        for (int c = 0; c < 3; c++)
            s[c] = phosphor_step(s[c], srgb_to_linear(px[c] * (1.0f / 255.0f)), p.a_rise,
                                 p.a_decay[c], p.eps, &moving);
        s[3] = 1.0f;
        if (moving) ACTIVE_RAISE();
    }
    ACTIVE_END(active);
}

// ---- Stage 2: GPU-resident, no host copies ---------------------------------
// frame: the compositor's texture, mapped from OpenGL (RGBA8, point sampled)
// state: a CUDA array owned by this tile, read and written through a surface
extern "C" __global__ void phosphor_update_tex(cudaTextureObject_t frame, int ox, int oy,
                                               cudaSurfaceObject_t state, int w, int h,
                                               PhosphorStep p, int *active) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    ACTIVE_BEGIN();
    if (x < w && y < h) {
        float4 in = tex2D<float4>(frame, ox + x + 0.5f, oy + y + 0.5f);
        float4 s = surf2Dread<float4>(state, x * (int)sizeof(float4), y);
        int moving = 0;
        s.x = phosphor_step(s.x, srgb_to_linear(in.x), p.a_rise, p.a_decay[0], p.eps, &moving);
        s.y = phosphor_step(s.y, srgb_to_linear(in.y), p.a_rise, p.a_decay[1], p.eps, &moving);
        s.z = phosphor_step(s.z, srgb_to_linear(in.z), p.a_rise, p.a_decay[2], p.eps, &moving);
        s.w = 1.0f;
        surf2Dwrite(s, state, x * (int)sizeof(float4), y);
        if (moving) ACTIVE_RAISE();
    }
    ACTIVE_END(active);
}

// ---- Version 2: half-precision state, a burn term, hardware sRGB decode -----
// A half has about three decimal digits, so a slow fade can stop one rounding
// step short of its target and sit there forever. Two rules keep idle honest:
// snap when closer than eps, and snap when the stored value did not move (a
// stall is the only way a quantised state fails to make progress).
__device__ inline unsigned short phosphor_step_h(unsigned short hprev, float target, float a_rise,
                                                 float a_decay, float eps, int *active) {
    float prev = ph_h2f(hprev);
    float a = target > prev ? a_rise : a_decay;
    float next = a >= 1.0f ? target : prev + a * (target - prev);
    unsigned short ht = ph_f2h(target), hn = ph_f2h(next);
    if (!(fabsf(target - next) >= eps) || hn == hprev) return ht;
    if (hn != ht) *active = 1;
    return hn;
}
struct h4 { unsigned short x, y, z, w; };
__device__ inline h4 phosphor_pixel_v2(h4 s, float tr, float tg, float tb, const PhosphorStepV2 &p, int *active) {
    s.x = phosphor_step_h(s.x, tr, p.a_rise, p.a_decay[0], p.eps, active);
    s.y = phosphor_step_h(s.y, tg, p.a_rise, p.a_decay[1], p.eps, active);
    s.z = phosphor_step_h(s.z, tb, p.a_rise, p.a_decay[2], p.eps, active);
    // the burn follows the picture's brightness, slowly both ways
    float lum = 0.2126f * tr + 0.7152f * tg + 0.0722f * tb;
    s.w = phosphor_step_h(s.w, lum, p.a_burn_rise, p.a_burn_decay, p.eps_burn, active);
    return s;
}

// Stage 1 flavour: plain buffers. sRGB is decoded through a 256-entry table the
// host fills, so this kernel and the CPU reference start from identical numbers
// and the comparison can demand identical bits.
extern "C" __global__ void phosphor_update_v2_buf(const unsigned char *frame, int frame_w, int ox, int oy,
                                                  unsigned short *state, int w, int h, PhosphorStepV2 p,
                                                  int *active, const float *lut) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    ACTIVE_BEGIN();
    if (x < w && y < h) {
        const unsigned char *px = frame + 4 * ((size_t)(oy + y) * frame_w + (ox + x));
        h4 *s = (h4 *)(state + 4 * ((size_t)y * w + x));
        int moving = 0;
        *s = phosphor_pixel_v2(*s, lut[px[0]], lut[px[1]], lut[px[2]], p, &moving);
        if (moving) ACTIVE_RAISE();
    }
    ACTIVE_END(active);
}

// GPU-resident flavour. frame_lin is the compositor's texture read through a
// texture object that decodes sRGB in hardware, so there is no pow() here.
extern "C" __global__ void phosphor_update_v2_tex(cudaTextureObject_t frame_lin, int ox, int oy,
                                                  cudaSurfaceObject_t state, int w, int h, PhosphorStepV2 p,
                                                  int *active) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    ACTIVE_BEGIN();
    if (x < w && y < h) {
        float4 in = tex2D<float4>(frame_lin, ox + x + 0.5f, oy + y + 0.5f);
        ushort4 raw = surf2Dread<ushort4>(state, x * (int)sizeof(ushort4), y);
        h4 s = {raw.x, raw.y, raw.z, raw.w};
        int moving = 0;
        s = phosphor_pixel_v2(s, in.x, in.y, in.z, p, &moving);
        surf2Dwrite(make_ushort4(s.x, s.y, s.z, s.w), state, x * (int)sizeof(ushort4), y);
        if (moving) ACTIVE_RAISE();
    }
    ACTIVE_END(active);
}
