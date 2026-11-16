// The CUDA side of a frame: per-tile surfaces, textures shared with OpenGL, and
// the kernel launches. It names no OpenGL header on purpose (texture names are
// plain unsigned ints, and the three interop calls are declared here), so it
// compiles the same against desktop GL, GLES, or inside a compositor plugin.
#pragma once
#include "cudart.hpp"
#include "look.hpp"
#include <vector>

extern "C" {
CUresult cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, unsigned int image, unsigned int target, unsigned int Flags);
CUresult cuGLGetDevices_v2(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, int deviceList);
}

// ---- CUDA path --------------------------------------------------------------
// The CUDA device that owns the current OpenGL context. On a machine with two
// NVIDIA GPUs, device 0 is not necessarily the one drawing; sharing a texture
// with the wrong one fails. Call with the GL context current. -1 = could not tell.
inline int cuda_device_for_gl() {
    if (cuInit(0) != CUDA_SUCCESS) return -1;
    unsigned int n = 0;
    CUdevice devs[4];
    if (cuGLGetDevices_v2(&n, devs, 4, 1 /* CU_GL_DEVICE_LIST_ALL */) != CUDA_SUCCESS || n == 0) return -1;
    return (int)devs[0];
}

// One CUDA array with a surface (to write) and a bilinear texture (to read).
struct CuImage {
    CUarray arr = nullptr;
    CUsurfObject surf = 0;
    CUtexObject tex = 0;
    void create(int w, int h, CUarray_format fmt) {
        CUDA_ARRAY3D_DESCRIPTOR d{};
        d.Width = w; d.Height = h; d.Depth = 0; d.Format = fmt; d.NumChannels = 4;
        d.Flags = CUDA_ARRAY3D_SURFACE_LDST;
        CU(cuArray3DCreate(&arr, &d));
        CUDA_RESOURCE_DESC rd{};
        rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = arr;
        CU(cuSurfObjectCreate(&surf, &rd));
        CUDA_TEXTURE_DESC td{};
        td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        td.filterMode = CU_TR_FILTER_MODE_LINEAR;
        CU(cuTexObjectCreate(&tex, &rd, &td, nullptr));
    }
    void zero(int w, int h, int bytes_per_px) {
        std::vector<uint8_t> z((size_t)w * h * bytes_per_px, 0);
        CUDA_MEMCPY2D c{};
        c.srcMemoryType = CU_MEMORYTYPE_HOST; c.srcHost = z.data(); c.srcPitch = (size_t)w * bytes_per_px;
        c.dstMemoryType = CU_MEMORYTYPE_ARRAY; c.dstArray = arr;
        c.WidthInBytes = (size_t)w * bytes_per_px; c.Height = h;
        CU(cuMemcpy2D(&c));
    }
    void destroy() {
        if (tex) cuTexObjectDestroy(tex);
        if (surf) cuSurfObjectDestroy(surf);
        if (arr) cuArrayDestroy(arr);
        tex = 0; surf = 0; arr = nullptr;
    }
};

// One tile's phosphor surface. v2 = half-precision state with the burn term in
// alpha, plus two 1/8-size images for the halation.
struct CudaSurf {
    int w = 0, h = 0, gw = 0, gh = 0;
    bool v2 = false, fresh = true;
    CuImage state, glow[2];
    CUsurfObject surf = 0;  // aliases of state, for the callers
    CUtexObject tex = 0;
    void create(int w_, int h_, bool v2_ = false) {
        w = w_; h = h_; v2 = v2_; fresh = true;
        state.create(w, h, v2 ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT);
        surf = state.surf; tex = state.tex;
        state.zero(w, h, v2 ? 8 : 16);  // an array starts undefined; no light, no burn, until the first frame
        if (v2) {
            gw = (w + 7) / 8; gh = (h + 7) / 8;
            for (auto &g : glow) { g.create(gw, gh, CU_AD_FORMAT_FLOAT); g.zero(gw, gh, 16); }
        }
    }
    void destroy() {
        state.destroy();
        for (auto &g : glow) g.destroy();
        surf = 0; tex = 0;
    }
    size_t bytes() const { return (size_t)w * h * (v2 ? 8 : 16) + (size_t)gw * gh * 32; }
};

// A GL texture registered with CUDA once, and mapped per frame.
struct Shared {
    CUgraphicsResource res = nullptr;
    void reg(unsigned int tex, unsigned flags) { CU(cuGraphicsGLRegisterImage(&res, tex, 0x0DE1 /* GL_TEXTURE_2D */, flags)); }
    void unreg() { if (res) cuGraphicsUnregisterResource(res); res = nullptr; }
};

struct CudaPath {
    Cuda *cu = nullptr;
    CUfunction k_update{}, k_optics{}, k_pass{}, k_update_v2{}, k_down{}, k_blur{}, k_beam{};
    CUtexObject frame_lin = 0;  // the same frame, decoded from sRGB by the texture unit
    CUdeviceptr d_active = 0;
    CUtexObject frame_tex = 0;
    CUsurfObject out_surf = 0;
    CUgraphicsResource mapped[2] = {nullptr, nullptr};
    int n_mapped = 0;
    int bx = 16, by = 16;

    void init(Cuda &c) {
        cu = &c;
        k_update = c.fn("phosphor_update_tex"); k_optics = c.fn("crt_optics_tex"); k_pass = c.fn("passthrough_tex");
        k_update_v2 = c.fn("phosphor_update_v2_tex"); k_down = c.fn("glow_down8"); k_blur = c.fn("glow_blur"); k_beam = c.fn("beam_deposit");
        CU(cuMemAlloc(&d_active, 4));
    }
    // Map the compositor's frame (and, when the glass runs here too, the output).
    void begin(Shared &frame, Shared *out) {
        mapped[0] = frame.res; n_mapped = 1;
        if (out) mapped[n_mapped++] = out->res;
        CU(cuGraphicsMapResources(n_mapped, mapped, 0));
        CUarray a;
        CU(cuGraphicsSubResourceGetMappedArray(&a, frame.res, 0, 0));
        CUDA_RESOURCE_DESC rd{};
        rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = a;
        CUDA_TEXTURE_DESC td{};
        td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        td.filterMode = CU_TR_FILTER_MODE_POINT;
        CU(cuTexObjectCreate(&frame_tex, &rd, &td, nullptr));
        td.flags = CU_TRSF_SRGB;
        CU(cuTexObjectCreate(&frame_lin, &rd, &td, nullptr));
        if (out) {
            CU(cuGraphicsSubResourceGetMappedArray(&a, out->res, 0, 0));
            rd.res.array.hArray = a;
            CU(cuSurfObjectCreate(&out_surf, &rd));
        }
        CU(cuMemsetD32Async(d_active, 0, 1, 0));
    }
    void update(CudaSurf &s, int ox, int oy, const PhosphorStep &p) {
        PhosphorStep pp = p;
        void *args[] = {&frame_tex, &ox, &oy, &s.surf, &s.w, &s.h, &pp, &d_active};
        launch2d(k_update, s.w, s.h, args, bx, by);
        s.fresh = false;
    }
    void update_v2(CudaSurf &s, int ox, int oy, const PhosphorStepV2 &p) {
        PhosphorStepV2 pp = p;
        void *args[] = {&frame_lin, &ox, &oy, &s.surf, &s.w, &s.h, &pp, &d_active};
        launch2d(k_update_v2, s.w, s.h, args, bx, by);
        s.fresh = false;
    }
    // Vector mode: deposit this frame's beam strokes into a v2 surface. d_segs holds nseg x 4 floats.
    void beam(CudaSurf &s, CUdeviceptr d_segs, int nseg, float sigma, float energy, const float rgb[3]) {
        float r = rgb[0], g = rgb[1], b = rgb[2];
        void *args[] = {&s.surf, &s.w, &s.h, &d_segs, &nseg, &sigma, &energy, &r, &g, &b};
        launch2d(k_beam, s.w, s.h, args, bx, by);
    }
    // Rebuild the tile's halation image from its state: down 8x, blur across, blur down.
    void glow(CudaSurf &s, float knee) {
        int one = 1, zero = 0;
        void *a0[] = {&s.tex, &s.glow[0].surf, &s.gw, &s.gh, &knee};
        launch2d(k_down, s.gw, s.gh, a0, 16, 16);
        void *a1[] = {&s.glow[0].tex, &s.glow[1].surf, &s.gw, &s.gh, &one, &zero};
        launch2d(k_blur, s.gw, s.gh, a1, 16, 16);
        void *a2[] = {&s.glow[1].tex, &s.glow[0].surf, &s.gw, &s.gh, &zero, &one};
        launch2d(k_blur, s.gw, s.gh, a2, 16, 16);
    }
    void optics(CudaSurf &s, int ox, int oy, const OpticsParams &k) {
        OpticsParams kk = k;
        CUtexObject glow_tex = s.v2 ? s.glow[0].tex : 0;
        if (!s.v2) kk.burn_gain = kk.halation = 0.0f;  // a version 1 surface has neither
        void *args[] = {&s.tex, &out_surf, &ox, &oy, &s.w, &s.h, &kk, &glow_tex};
        launch2d(k_optics, s.w, s.h, args, bx, by);
    }
    void passthrough(int ox, int oy, int w, int h) {
        void *args[] = {&frame_tex, &out_surf, &ox, &oy, &w, &h};
        launch2d(k_pass, w, h, args, bx, by);
    }
    // Reads the tile-activity int back. This copy waits for the kernels.
    bool active_result() {
        int a = 0;
        CU(cuMemcpyDtoH(&a, d_active, 4));
        return a != 0;
    }
    // After a failure mid-frame: let go of everything without throwing again, so
    // OpenGL is never left drawing into textures CUDA still has mapped.
    void abandon() {
        if (frame_tex) { cuTexObjectDestroy(frame_tex); frame_tex = 0; }
        if (frame_lin) { cuTexObjectDestroy(frame_lin); frame_lin = 0; }
        if (out_surf) { cuSurfObjectDestroy(out_surf); out_surf = 0; }
        if (n_mapped) cuGraphicsUnmapResources(n_mapped, mapped, 0);
        n_mapped = 0;
    }
    void end() {
        if (frame_tex) { cuTexObjectDestroy(frame_tex); frame_tex = 0; }
        if (frame_lin) { cuTexObjectDestroy(frame_lin); frame_lin = 0; }
        if (out_surf) { cuSurfObjectDestroy(out_surf); out_surf = 0; }
        CU(cuGraphicsUnmapResources(n_mapped, mapped, 0));
        n_mapped = 0;
    }
};
