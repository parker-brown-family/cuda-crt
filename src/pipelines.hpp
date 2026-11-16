// The three render paths behind one shape, so the benchmark and the renderer
// drive them identically:
//   Existing  the shipped omarchy-crt glass shader (baseline/crt-glass.frag), as it is: one pass, no memory
//   Glsl      phosphor state + glass in fragment shaders (the matched reference)
//   CudaPath  phosphor state + glass in CUDA kernels, sharing OpenGL's textures
// Convention everywhere: texel row 0 is the TOP of the picture.
#pragma once
#include "glrt.hpp"
#include "cudart.hpp"
#include <cudaGL.h>
#include "cpu_ref.hpp"
#include <cmath>
#include <regex>

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

// ---- GLSL path --------------------------------------------------------------
struct GlslSurf {
    int w = 0, h = 0;
    GLuint tex[2] = {0, 0}, fbo[2] = {0, 0};
    int cur = 0;  // tex[cur] holds the newest state
    bool fresh = true;
    int bpp = 16;
    // format: GL_RGBA32F matches the CUDA state exactly. GL_RGBA16F halves the
    // memory traffic and is TIMED ONLY here: at half precision the settle test
    // can stall one step from a bright target, so idle is not proven for it.
    void create(int w_, int h_, GLenum format = GL_RGBA32F) {
        w = w_; h = h_; fresh = true; cur = 0; bpp = format == GL_RGBA16F ? 8 : 16;
        for (int i = 0; i < 2; i++) {
            tex[i] = gl_tex(format, w, h, GL_LINEAR);
            glClearTexImage(tex[i], 0, GL_RGBA, GL_FLOAT, nullptr);  // storage starts undefined; make it black
            fbo[i] = gl_fbo(tex[i]);
        }
    }
    void destroy() { glDeleteFramebuffers(2, fbo); glDeleteTextures(2, tex); tex[0] = tex[1] = 0; }
    size_t bytes() const { return (size_t)w * h * bpp * 2; }
};

struct Glsl {
    GLuint p_update = 0, p_optics = 0, p_probe = 0, vao = 0, q_any[2] = {0, 0};
    int q_cur = 0, q_issued = 0;
    struct { GLint frame, prev, origin, a_rise, a_decay, eps; } u{};
    struct { GLint state, origin, size, curv, k1, k2, aberr, scan, scan_step, glow, glow_px, bloom, vign, specular,
                   glare, tracking, band_h, band_center, flicker_mul, phosphor, glare_tint; } o{};
    GLint probe_state = -1;

    void init(const std::string &root) {
        p_update = gl_program(slurp(root + "/shaders/update.frag"));
        p_optics = gl_program(slurp(root + "/shaders/optics.frag"));
        p_probe = gl_program(slurp(root + "/shaders/probe.frag"));
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenQueries(2, q_any);
        glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_DITHER);
#define UL(s, prog, name) s.name = glGetUniformLocation(prog, "u_" #name)
        UL(u, p_update, frame); UL(u, p_update, prev); UL(u, p_update, origin); UL(u, p_update, a_rise);
        UL(u, p_update, a_decay); UL(u, p_update, eps);
        UL(o, p_optics, state); UL(o, p_optics, origin); UL(o, p_optics, size); UL(o, p_optics, curv); UL(o, p_optics, k1);
        UL(o, p_optics, k2); UL(o, p_optics, aberr); UL(o, p_optics, scan); UL(o, p_optics, scan_step); UL(o, p_optics, glow);
        UL(o, p_optics, glow_px); UL(o, p_optics, bloom); UL(o, p_optics, vign); UL(o, p_optics, specular); UL(o, p_optics, glare);
        UL(o, p_optics, tracking); UL(o, p_optics, band_h); UL(o, p_optics, band_center); UL(o, p_optics, flicker_mul);
        UL(o, p_optics, phosphor); UL(o, p_optics, glare_tint);
#undef UL
        probe_state = glGetUniformLocation(p_probe, "u_state");
    }

    void update(GlslSurf &s, GLuint frame_tex, int ox, int oy, const PhosphorStep &p) {
        int dst = 1 - s.cur;
        glBindFramebuffer(GL_FRAMEBUFFER, s.fbo[dst]);
        glViewport(0, 0, s.w, s.h);
        glUseProgram(p_update);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, frame_tex);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, s.tex[s.cur]);
        glUniform1i(u.frame, 0); glUniform1i(u.prev, 1);
        glUniform2i(u.origin, ox, oy);
        glUniform1f(u.a_rise, p.a_rise);
        glUniform3f(u.a_decay, p.a_decay[0], p.a_decay[1], p.a_decay[2]);
        glUniform1f(u.eps, p.eps);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        s.cur = dst;
        s.fresh = false;
    }
    // Issue the idle probe. Its answer is read one frame LATE, by active_result():
    // asking for this frame's answer now would stall until the GPU catches up,
    // and no compositor would. The cost is one extra redraw at the end of a fade.
    void probe(GlslSurf &s) {
        q_cur = 1 - q_cur;
        q_issued++;
        glBindFramebuffer(GL_FRAMEBUFFER, s.fbo[1 - s.cur]);  // any target of the right size; nothing is written
        glViewport(0, 0, s.w, s.h);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glUseProgram(p_probe);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, s.tex[s.cur]);
        glUniform1i(probe_state, 0);
        glBeginQuery(GL_ANY_SAMPLES_PASSED, q_any[q_cur]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndQuery(GL_ANY_SAMPLES_PASSED);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }
    // The PREVIOUS frame's answer (already finished, so this does not wait).
    // Before two probes exist there is nothing to read: say "still moving".
    bool active_result() {
        if (q_issued < 2) return true;
        GLuint any = 0;
        glGetQueryObjectuiv(q_any[1 - q_cur], GL_QUERY_RESULT, &any);
        return any != 0;
    }
    void optics(GlslSurf &s, GLuint out_fbo, int ox, int oy, const OpticsParams &k) {
        glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);
        glViewport(ox, oy, s.w, s.h);
        glUseProgram(p_optics);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, s.tex[s.cur]);
        glUniform1i(o.state, 0);
        glUniform2i(o.origin, ox, oy);
        glUniform2f(o.size, (float)s.w, (float)s.h);
        glUniform1f(o.curv, k.curv); glUniform1f(o.k1, k.k1); glUniform1f(o.k2, k.k2); glUniform1f(o.aberr, k.aberr);
        glUniform1f(o.scan, k.scan); glUniform1f(o.scan_step, k.scan_step); glUniform1f(o.glow, k.glow);
        glUniform1f(o.glow_px, k.glow_px); glUniform1f(o.bloom, k.bloom); glUniform1f(o.vign, k.vign);
        glUniform1f(o.specular, k.specular); glUniform1f(o.glare, k.glare); glUniform1f(o.tracking, k.tracking);
        glUniform1f(o.band_h, k.band_h); glUniform1f(o.band_center, k.band_center); glUniform1f(o.flicker_mul, k.flicker_mul);
        glUniform3fv(o.phosphor, 1, k.phosphor); glUniform3fv(o.glare_tint, 1, k.glare_tint);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
};

// ---- the shipped shader, as the baseline ------------------------------------
// Reads baseline/crt-glass.frag (never writes it), splices in one
// tube the way `crt render` does, and sets the same dials the other paths use.
struct Existing {
    GLuint prog = 0;
    GLint u_tex = -1, u_time = -1, u_out = -1;
    void init(const std::string &frag_path, const Look &k, int x, int y, int w, int h) {
        std::string src = slurp(frag_path);
        size_t a = src.find("#define TUBE_COUNT 0"), b = src.find("// ---- END TUBE DATA");
        if (a == std::string::npos || b == std::string::npos) throw std::runtime_error("crt-glass.frag: tube block not found");
        char tubes[512];
        snprintf(tubes, sizeof tubes,
                 "#define TUBE_COUNT 1\nconst vec4 TUBE_RECT[1] = vec4[1](vec4(%d.0, %d.0, %d.0, %d.0));\n"
                 "const int TUBE_MON[1] = int[1](0);\nconst float TUBE_CURV[1] = float[1](%f);\n"
                 "const int TUBE_PASS[1] = int[1](0);\nconst int TUBE_QUIET[1] = int[1](0);\n", x, y, w, h, k.curv);
        src = src.substr(0, a) + tubes + src.substr(b);
        auto set = [&](const char *name, float v) {
            std::regex re(std::string("const float ") + name + R"(\s*=\s*[-0-9.]+;)");
            char rep[96];
            snprintf(rep, sizeof rep, "const float %s = %f;", name, v);
            src = std::regex_replace(src, re, rep, std::regex_constants::format_first_only);
        };
        src = std::regex_replace(src, std::regex("#define ANIMATED 0"), "#define ANIMATED 1", std::regex_constants::format_first_only);
        set("TUBE_K1", k.k1); set("TUBE_K2", k.k2); set("ABERR", k.aberr); set("SCAN", k.scan); set("SCAN_STEP", k.scan_step);
        set("GLOW", k.glow); set("BLOOM", k.bloom); set("VIGN", k.vign); set("SPECULAR", k.specular); set("TRACKING", k.tracking);
        set("TRACK_REST", k.track_rest); set("TRACK_SWEEP", k.track_sweep); set("BAND_H", k.band_h); set("FLICKER", k.flicker);
        set("GLARE", k.glare);
        static const char *vs = "#version 320 es\nout highp vec2 v_texcoord;\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
                                "v_texcoord=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n";
        prog = glCreateProgram();
        GLuint v = gl_shader(GL_VERTEX_SHADER, vs), f = gl_shader(GL_FRAGMENT_SHADER, src);
        glAttachShader(prog, v); glAttachShader(prog, f);
        glLinkProgram(prog);
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) throw std::runtime_error("crt-glass.frag: link failed");
        u_tex = glGetUniformLocation(prog, "tex"); u_time = glGetUniformLocation(prog, "time");
        u_out = glGetUniformLocation(prog, "wl_output");
    }
    // The shipped shader subtracts a per-tube phase offset from nothing: tube 0's phase is 0.
    void draw(GLuint frame_tex, GLuint out_fbo, int fw, int fh, float t) {
        glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);
        glViewport(0, 0, fw, fh);
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, frame_tex);
        glUniform1i(u_tex, 0);
        if (u_time >= 0) glUniform1f(u_time, t);
        if (u_out >= 0) glUniform1i(u_out, 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
};

// ---- CUDA path --------------------------------------------------------------
// The CUDA device that owns the current OpenGL context. On a machine with two
// NVIDIA GPUs, device 0 is not necessarily the one drawing; sharing a texture
// with the wrong one fails. Call with the GL context current. -1 = could not tell.
inline int cuda_device_for_gl() {
    if (cuInit(0) != CUDA_SUCCESS) return -1;
    unsigned int n = 0;
    CUdevice devs[4];
    if (cuGLGetDevices(&n, devs, 4, CU_GL_DEVICE_LIST_ALL) != CUDA_SUCCESS || n == 0) return -1;
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
    void reg(GLuint tex, unsigned flags) { CU(cuGraphicsGLRegisterImage(&res, tex, GL_TEXTURE_2D, flags)); }
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
