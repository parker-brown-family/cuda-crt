// The three render paths behind one shape, so the benchmark and the renderer
// drive them identically:
//   Existing  the shipped omarchy-crt glass shader (baseline/crt-glass.frag), as it is: one pass, no memory
//   Glsl      phosphor state + glass in fragment shaders (the matched reference)
//   CudaPath  phosphor state + glass in CUDA kernels, sharing OpenGL's textures
// Convention everywhere: texel row 0 is the TOP of the picture.
#pragma once
#include "glrt.hpp"
#include "cudart.hpp"
#include "look.hpp"
#include "cuda_path.hpp"
#include <regex>

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
