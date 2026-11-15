// crt_render: frames in, CRT frames out. Reads raw RGBA8 frames on stdin, runs
// each through one of the paths, writes raw RGBA8 frames on stdout.
//
//   ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgba - |
//     bin/crt_render . --size 1280x720 --fps 60 --path cuda |
//     ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - out.mp4
//
// --path cuda | glsl | existing | pass
// --tiles "x,y,w,h;x,y,w,h"   one phosphor surface per tile (default: the whole frame)
// --decay r,g,b   --rise ms   phosphor time constants in milliseconds
// --set name=value,...        any dial in Look (curv, scan, glare, tracking, track_rest, ...)
// If the CUDA path fails at any point the frame goes through untouched and the
// reason is printed once: a broken backend must never cost the picture.
#include "pipelines.hpp"
#include <map>
#include <unistd.h>

struct Rect { int x, y, w, h; };

static bool read_all(void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(0, (char *)buf + got, n - got);
        if (r <= 0) return false;
        got += r;
    }
    return true;
}
static void write_all(const void *buf, size_t n) {
    size_t put = 0;
    while (put < n) {
        ssize_t r = write(1, (const char *)buf + put, n - put);
        if (r <= 0) exit(0);
        put += r;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: crt_render <lab-root> --size WxH [--fps F] [--path cuda|glsl|existing|pass] ...\n"); return 2; }
    std::string root = argv[1], path = "cuda";
    int W = 0, H = 0;
    double fps = 60;
    Look look;
    look.track_rest = 2.0f; look.track_sweep = 5.0f;  // clips are short: let the band be seen
    std::vector<Rect> tiles;
    std::map<std::string, float *> dials = {
        {"curv", &look.curv}, {"k1", &look.k1}, {"k2", &look.k2}, {"aberr", &look.aberr}, {"scan", &look.scan},
        {"scan_step", &look.scan_step}, {"glow", &look.glow}, {"glow_px", &look.glow_px}, {"bloom", &look.bloom},
        {"vign", &look.vign}, {"specular", &look.specular}, {"glare", &look.glare}, {"tracking", &look.tracking},
        {"track_rest", &look.track_rest}, {"track_sweep", &look.track_sweep}, {"band_h", &look.band_h}, {"flicker", &look.flicker}};
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i], v = i + 1 < argc ? argv[i + 1] : "";
        if (a == "--size") { sscanf(v.c_str(), "%dx%d", &W, &H); i++; }
        else if (a == "--fps") { fps = atof(v.c_str()); i++; }
        else if (a == "--path") { path = v; i++; }
        else if (a == "--rise") { look.phosphor.tau_rise = atof(v.c_str()) / 1000; i++; }
        else if (a == "--decay") {
            double r, g, b;
            int n = sscanf(v.c_str(), "%lf,%lf,%lf", &r, &g, &b);
            if (n == 1) g = b = r;
            look.phosphor.tau_decay[0] = r / 1000; look.phosphor.tau_decay[1] = g / 1000; look.phosphor.tau_decay[2] = b / 1000; i++;
        } else if (a == "--tiles") {
            const char *s = v.c_str();
            Rect r;
            int used = 0;
            while (sscanf(s, "%d,%d,%d,%d%n", &r.x, &r.y, &r.w, &r.h, &used) == 4) { tiles.push_back(r); s += used; if (*s == ';') s++; }
            i++;
        } else if (a == "--set") {
            size_t pos = 0;
            while (pos < v.size()) {
                size_t c = v.find(',', pos), e = v.find('=', pos);
                if (c == std::string::npos) c = v.size();
                std::string key = v.substr(pos, e - pos);
                if (!dials.count(key)) { fprintf(stderr, "unknown dial %s\n", key.c_str()); return 2; }
                *dials[key] = (float)atof(v.substr(e + 1, c - e - 1).c_str());
                pos = c + 1;
            }
            i++;
        } else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (W <= 0 || H <= 0) { fprintf(stderr, "--size WxH is required\n"); return 2; }
    if (tiles.empty()) tiles.push_back({0, 0, W, H});

    GLCtx gl;
    Glsl glsl;
    Cuda cu;
    CudaPath cuda;
    Existing existing;
    bool cuda_ok = false;
    try {
        gl.init(false);
        glsl.init(root);
        if (path == "existing") existing.init(root + "/../../crt/crt-glass.frag", look, tiles[0].x, tiles[0].y, tiles[0].w, tiles[0].h);
    } catch (std::exception &e) { fprintf(stderr, "crt_render: %s\n", e.what()); return 2; }

    GLuint frame_tex = gl_tex(GL_RGBA8, W, H, GL_LINEAR), out_tex = gl_tex(GL_RGBA8, W, H, GL_NEAREST);
    GLuint frame_fbo = gl_fbo(frame_tex), out_fbo = gl_fbo(out_tex);
    Shared sh_frame, sh_out;
    std::vector<GlslSurf> gsurf(tiles.size());
    std::vector<CudaSurf> csurf(tiles.size());
    if (path == "cuda") {
        try {
            cu.init();
            cu.load(root + "/kernels/phosphor.cu", root);
            cuda.init(cu);
            sh_frame.reg(frame_tex, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
            sh_out.reg(out_tex, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
            for (size_t i = 0; i < tiles.size(); i++) csurf[i].create(tiles[i].w, tiles[i].h);
            cuda_ok = true;
        } catch (std::exception &e) { fprintf(stderr, "crt_render: CUDA unavailable, passing frames through: %s\n", e.what()); }
    }
    if (path == "glsl")
        for (size_t i = 0; i < tiles.size(); i++) gsurf[i].create(tiles[i].w, tiles[i].h);

    std::vector<uint8_t> in((size_t)W * H * 4), out(in.size());
    const double dt = 1.0 / fps;
    long n = 0, n_active = 0;
    while (read_all(in.data(), in.size())) {
        float t = (float)(n * dt);
        glBindTexture(GL_TEXTURE_2D, frame_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, in.data());
        bool active = false, drawn = false;
        if (path == "cuda" && cuda_ok) {
            try {
                cuda.begin(sh_frame, &sh_out);
                cuda.passthrough(0, 0, W, H);  // the desk between the tiles
                for (size_t i = 0; i < tiles.size(); i++) {
                    cuda.update(csurf[i], tiles[i].x, tiles[i].y, make_step(look.phosphor, dt, csurf[i].fresh));
                    cuda.optics(csurf[i], tiles[i].x, tiles[i].y, make_optics(look, t, tiles[i].h));
                }
                active = cuda.active_result();
                cuda.end();
                drawn = true;
            } catch (std::exception &e) {
                fprintf(stderr, "crt_render: CUDA failed at frame %ld, passing frames through from here: %s\n", n, e.what());
                cuda_ok = false;
            }
        } else if (path == "glsl" || path == "existing") {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, frame_fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, out_fbo);
            glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            if (path == "existing") existing.draw(frame_tex, out_fbo, W, H, t);
            else
                for (size_t i = 0; i < tiles.size(); i++) {
                    glsl.update(gsurf[i], frame_tex, tiles[i].x, tiles[i].y, make_step(look.phosphor, dt, gsurf[i].fresh));
                    glsl.probe(gsurf[i]);
                    active = glsl.active_result() || active;
                    glsl.optics(gsurf[i], out_fbo, tiles[i].x, tiles[i].y, make_optics(look, t, tiles[i].h));
                }
            drawn = true;
        }
        if (drawn) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, out_fbo);
            glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
            write_all(out.data(), out.size());
        } else write_all(in.data(), in.size());  // pass, or a backend that failed
        n++;
        n_active += active;
    }
    fprintf(stderr, "crt_render: %ld frames via %s%s; phosphor still moving on %ld of them\n", n, path.c_str(),
            path == "cuda" && !cuda_ok ? " (FELL BACK TO PASSTHROUGH)" : "", n_active);
    return 0;
}
