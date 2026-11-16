// crt_render: frames in, CRT frames out. Reads raw RGBA8 frames on stdin, runs
// each through one of the paths, writes raw RGBA8 frames on stdout.
//
//   ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgba - |
//     bin/crt_render . --size 1280x720 --fps 60 --path cuda |
//     ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - out.mp4
//
// --path cuda2 | cuda | glsl | existing | pass     (cuda2 = half-precision state,
//                                                   burn ghost, halation, optional mask)
// --burn rise,decay           burn time constants in milliseconds
// --vector lissajous|scope --frames N
//                             vector mode: no input video; a beam draws the figure
//                             straight into the phosphor (reads nothing from stdin)
// --tiles "x,y,w,h;x,y,w,h"   one phosphor surface per tile (default: the whole frame)
// --decay r,g,b   --rise ms   phosphor time constants in milliseconds
// --set name=value,...        any dial (`--dials` lists them: ghost, curv, scan, tracking, jiggle, ...)
// --knobs FILE                re-read FILE whenever it changes and apply its {"values": {...}};
//                             this is how the slider panel (tools/knobs.py) drives a live window
// --dials                     print every dial as JSON (name, range, default) and exit
// If the CUDA path fails at any point the frame goes through untouched and the
// reason is printed once: a broken backend must never cost the picture.
#include "pipelines.hpp"
#include <map>
#include <sys/stat.h>
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
    std::string root = argv[1], path = "cuda", vector_fig;
    long vector_frames = 600;
    int W = 0, H = 0;
    double fps = 60;
    Look look;
    look.track_rest = 2.0f; look.track_sweep = 5.0f;  // clips are short: let the band be seen
    std::vector<Rect> tiles;
    // Time constants are kept here in milliseconds, the unit a person turns, and
    // copied into the look whenever they change (sync_ms).
    float decay_ms[3] = {50, 50, 50}, rise_ms = 1, ghost_rise_ms = 500, ghost_decay_ms = 800;
    auto sync_ms = [&] {
        for (int c = 0; c < 3; c++) look.phosphor.tau_decay[c] = std::max(0.2f, decay_ms[c]) / 1000.0;
        look.phosphor.tau_rise = std::max(0.05f, rise_ms) / 1000.0;
        look.burn.tau_rise = std::max(1.0f, ghost_rise_ms) / 1000.0;
        look.burn.tau_decay = std::max(1.0f, ghost_decay_ms) / 1000.0;
    };
    // Every dial a person can turn: name, where it lives, and how the slider panel should show it.
    struct Dial { const char *name; float *v; const char *label, *group; float lo, hi, step; const char *unit, *help; };
    const std::vector<Dial> table = {
        {"ghost", &look.burn.gain, "Phosphor burn ghost", "Phosphor", 0, 0.03f, 0.0005f, "", "How strongly the last screen lingers. 0 is none."},
        {"ghost_decay", &ghost_decay_ms, "Ghost linger", "Phosphor", 100, 5000, 50, "ms", "How long the ghost takes to drain"},
        {"ghost_rise", &ghost_rise_ms, "Ghost charge time", "Phosphor", 50, 3000, 50, "ms", "How long something must sit before it leaves a ghost"},
        {"decay_g", &decay_ms[1], "Green fade", "Phosphor", 1, 400, 1, "ms", "How long green light takes to fade"},
        {"decay_r", &decay_ms[0], "Red fade", "Phosphor", 1, 400, 1, "ms", "How long red light takes to fade"},
        {"decay_b", &decay_ms[2], "Blue fade", "Phosphor", 1, 400, 1, "ms", "How long blue light takes to fade"},
        {"rise", &rise_ms, "Rise", "Phosphor", 0.2f, 30, 0.2f, "ms", "How fast a new letter reaches full brightness"},
        {"halation", &look.halation, "Halation glow", "Phosphor", 0, 1, 0.01f, "", "Soft glow around lit text"},
        {"beam", &look.beam, "Beam swell", "Phosphor", 0, 2, 0.02f, "", "Bright strokes spread wider than dim ones"},
        {"curv", &look.curv, "Barrel warp", "Tube", 0, 3, 0.02f, "", "How far the glass bows"},
        {"core", &look.core, "Flat middle", "Tube", 0, 0.95f, 0.01f, "", "Keep the middle flat and bend only toward the rim, so things stay where they are"},
        {"inset", &look.inset, "Safe edge", "Tube", 0, 0.12f, 0.005f, "", "Draw the picture a little smaller so the rim never covers the window's edges"},
        {"scan", &look.scan, "Scanline depth", "Tube", 0, 0.6f, 0.01f, "", "How dark the lines between rows are"},
        {"scan_step", &look.scan_step, "Scanline pitch", "Tube", 2, 8, 1, "px", "Pixels between scanlines"},
        {"mask", &look.mask, "Aperture grille", "Tube", 0, 0.6f, 0.01f, "", "Vertical red, green and blue stripes"},
        {"aberr", &look.aberr, "Colour split at the rim", "Tube", 0, 3, 0.05f, "", "Red and blue part company toward the edges"},
        {"glare", &look.glare, "Glass glare", "Tube", 0, 1, 0.01f, "", "The room's light on the glass"},
        {"specular", &look.specular, "Corner highlight", "Tube", 0, 1, 0.01f, "", "A small bright catch, upper left"},
        {"vign", &look.vign, "Corner falloff", "Tube", 0, 1, 0.01f, "", "Darker toward the corners"},
        {"bloom", &look.bloom, "Centre wash", "Tube", 0, 1, 0.01f, "", "A faint tint of the tube's colour"},
        {"glow", &look.glow, "Tight glow", "Tube", 0, 1, 0.01f, "", "A two-pixel glow on every stroke"},
        {"tracking", &look.tracking, "Tracking bar", "Motion", 0, 1, 0.01f, "", "Strength of the bar that rolls down the screen"},
        {"track_bend", &look.track_bend, "Bar bends the picture", "Motion", 0, 40, 0.5f, "px", "How far the bar pushes the picture sideways"},
        {"track_noise", &look.track_noise, "Bar static and tearing", "Motion", 0, 1, 0.01f, "", "Line tearing and static inside the bar"},
        {"track_chroma", &look.track_chroma, "Bar colour split", "Motion", 0, 3, 0.05f, "", "Colours part company inside the bar"},
        {"band_h", &look.band_h, "Bar height", "Motion", 40, 600, 4, "px", "How tall the bar is"},
        {"track_sweep", &look.track_sweep, "Seconds per pass", "Motion", 1, 20, 0.5f, "s", "How long the bar takes to cross"},
        {"track_rest", &look.track_rest, "Seconds between passes", "Motion", 0, 30, 0.5f, "s", "Quiet time before the next bar"},
        {"jiggle", &look.jiggle, "Screen jiggle", "Motion", 0, 1, 0.01f, "", "The picture breathes and now and then hops"},
        {"flicker", &look.flicker, "Flicker", "Motion", 0, 1, 0.01f, "", "Occasional stepped dips in brightness"},
    };
    std::map<std::string, float *> dials = {{"k1", &look.k1}, {"k2", &look.k2}, {"glow_px", &look.glow_px}, {"glow_knee", &look.glow_knee},
                                           {"degauss_at", &look.degauss_at}, {"burn", &look.burn.gain}, {"afterglow", &look.burn.gain}};
    for (const Dial &d : table) dials[d.name] = d.v;
    if (argc > 2 && std::string(argv[2]) == "--dials") {  // the table as JSON, for the slider panel
        printf("[\n");
        for (size_t i = 0; i < table.size(); i++) {
            const Dial &d = table[i];
            printf("  {\"name\": \"%s\", \"label\": \"%s\", \"group\": \"%s\", \"min\": %g, \"max\": %g, \"step\": %g, \"default\": %g, \"unit\": \"%s\", \"help\": \"%s\"}%s\n",
                   d.name, d.label, d.group, d.lo, d.hi, d.step, *d.v, d.unit, d.help, i + 1 < table.size() ? "," : "");
        }
        printf("]\n");
        return 0;
    }
    std::string knobs_path;
    struct timespec knobs_mtime = {0, 0};
    // Re-read the panel's file when it changes: {"values": {"name": number, ...}}. Unknown names
    // are ignored and every value is clamped to its dial's range, whatever the file says.
    auto load_knobs = [&] {
        struct stat sb;
        if (knobs_path.empty() || stat(knobs_path.c_str(), &sb) != 0) return;
        if (sb.st_mtim.tv_sec == knobs_mtime.tv_sec && sb.st_mtim.tv_nsec == knobs_mtime.tv_nsec) return;
        knobs_mtime = sb.st_mtim;
        std::string j;
        try { j = slurp(knobs_path); } catch (...) { return; }
        size_t v = j.find("\"values\"");
        if (v == std::string::npos) return;
        static const std::regex pair("\"([a-z_0-9]+)\"\\s*:\\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)");
        for (std::sregex_iterator it(j.begin() + v, j.end(), pair), end; it != end; ++it)
            for (const Dial &d : table)
                if ((*it)[1] == d.name) *d.v = std::min(d.hi, std::max(d.lo, (float)atof((*it)[2].str().c_str())));
        sync_ms();
    };
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i], v = i + 1 < argc ? argv[i + 1] : "";
        if (a == "--size") { sscanf(v.c_str(), "%dx%d", &W, &H); i++; }
        else if (a == "--fps") { fps = atof(v.c_str()); i++; }
        else if (a == "--path") { path = v; i++; }
        else if (a == "--knobs") { knobs_path = v; i++; }
        else if (a == "--rise") { rise_ms = (float)atof(v.c_str()); i++; }
        else if (a == "--decay") {
            double r, g, b;
            int n = sscanf(v.c_str(), "%lf,%lf,%lf", &r, &g, &b);
            if (n == 1) g = b = r;
            decay_ms[0] = (float)r; decay_ms[1] = (float)g; decay_ms[2] = (float)b; i++;
        } else if (a == "--vector") { vector_fig = v; path = "cuda2"; i++; }
        else if (a == "--frames") { vector_frames = atol(v.c_str()); i++; }
        else if (a == "--burn") {
            double r = 500, d = 800;
            sscanf(v.c_str(), "%lf,%lf", &r, &d);
            ghost_rise_ms = (float)r; ghost_decay_ms = (float)d; i++;
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
    sync_ms();
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
        if (path == "existing") existing.init(root + "/baseline/crt-glass.frag", look, tiles[0].x, tiles[0].y, tiles[0].w, tiles[0].h);
    } catch (std::exception &e) { fprintf(stderr, "crt_render: %s\n", e.what()); return 2; }

    GLuint frame_tex = gl_tex(GL_RGBA8, W, H, GL_LINEAR), out_tex = gl_tex(GL_RGBA8, W, H, GL_NEAREST);
    GLuint frame_fbo = gl_fbo(frame_tex), out_fbo = gl_fbo(out_tex);
    Shared sh_frame, sh_out;
    std::vector<GlslSurf> gsurf(tiles.size());
    std::vector<CudaSurf> csurf(tiles.size());
    const bool v2 = path == "cuda2";
    if (v2) path = "cuda";
    if (path == "cuda") {
        try {
            cu.init(cuda_device_for_gl());
            cu.load_all(root);
            cuda.init(cu);
            sh_frame.reg(frame_tex, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
            sh_out.reg(out_tex, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
            for (size_t i = 0; i < tiles.size(); i++) csurf[i].create(tiles[i].w, tiles[i].h, v2);
            cuda_ok = true;
        } catch (std::exception &e) { fprintf(stderr, "crt_render: CUDA unavailable, passing frames through: %s\n", e.what()); }
    }
    if (path == "glsl")
        for (size_t i = 0; i < tiles.size(); i++) gsurf[i].create(tiles[i].w, tiles[i].h);

    std::vector<uint8_t> in((size_t)W * H * 4), out(in.size());
    const double dt = 1.0 / fps;
    long n = 0, n_active = 0;
    // vector mode: the "frame" is black, and the beam's strokes for each frame are computed here
    const int SEG = 220;
    std::vector<float> segs(SEG * 4);
    CUdeviceptr d_segs = 0;
    if (!vector_fig.empty() && cuda_ok) {
        CU(cuMemAlloc(&d_segs, segs.size() * 4));
        for (size_t i = 0; i < in.size(); i += 4) { in[i] = in[i + 1] = in[i + 2] = 0; in[i + 3] = 255; }
    }
    auto beam_at = [&](double u, float &x, float &y) {  // u = beam time in seconds
        const Rect &r = tiles[0];
        double cx = r.w * 0.5, cy = r.h * 0.5, ax = r.w * 0.36, ay = r.h * 0.36;
        if (vector_fig == "scope") {  // a trace sweeping left to right 60 times a second, the wave drifting
            double sweep = u * 60.0 - floor(u * 60.0);
            double ph = u * 0.9;
            x = (float)(r.w * (0.08 + 0.84 * sweep));
            y = (float)(cy - ay * (0.55 * sin(6.2831853 * (2.0 * sweep + ph)) + 0.30 * sin(6.2831853 * (5.0 * sweep - 1.7 * ph)) * sin(ph * 2.3)));
        } else {  // lissajous: two sines whose ratio and phase drift, so the knot turns and re-ties itself
            double fx = 3.0 + 0.5 * sin(u * 0.21), fy = 2.0, ph = u * 0.6;
            x = (float)(cx + ax * sin(6.2831853 * 41.0 * fx * u / 3.0 + ph));
            y = (float)(cy + ay * sin(6.2831853 * 41.0 * fy * u / 3.0));
        }
    };
    while (vector_fig.empty() ? read_all(in.data(), in.size()) : n < vector_frames) {
        float t = (float)(n * dt);
        load_knobs();  // a slider moved: the very next frame shows it
        glBindTexture(GL_TEXTURE_2D, frame_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, in.data());
        bool active = false, drawn = false;
        if (path == "cuda" && cuda_ok) {
            try {
                cuda.begin(sh_frame, &sh_out);
                cuda.passthrough(0, 0, W, H);  // the desk between the tiles
                for (size_t i = 0; i < tiles.size(); i++) {
                    if (v2) {
                        cuda.update_v2(csurf[i], tiles[i].x, tiles[i].y, make_step_v2(look.phosphor, look.burn, dt, csurf[i].fresh));
                        if (d_segs && i == 0) {  // the strokes the beam swept since the last frame
                            for (int k = 0; k < SEG; k++) {
                                beam_at((n + (double)k / SEG) * dt, segs[4 * k], segs[4 * k + 1]);
                                beam_at((n + (double)(k + 1) / SEG) * dt, segs[4 * k + 2], segs[4 * k + 3]);
                            }
                            CU(cuMemcpyHtoD(d_segs, segs.data(), segs.size() * 4));
                            const float p31[3] = {0.30f, 1.0f, 0.42f};
                            cuda.beam(csurf[i], d_segs, SEG, std::max(1.1f, tiles[i].h / 420.0f), 0.55f, p31);
                        }
                        if (look.halation > 0.0f) cuda.glow(csurf[i], look.glow_knee);
                    } else cuda.update(csurf[i], tiles[i].x, tiles[i].y, make_step(look.phosphor, dt, csurf[i].fresh));
                    cuda.optics(csurf[i], tiles[i].x, tiles[i].y, make_optics(look, t, tiles[i].h, v2));
                }
                active = cuda.active_result();
                cuda.end();
                drawn = true;
            } catch (std::exception &e) {
                fprintf(stderr, "crt_render: CUDA failed at frame %ld, passing frames through from here: %s\n", n, e.what());
                cuda.abandon();  // unmap before OpenGL touches those textures again
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
    fprintf(stderr, "crt_render: %ld frames via %s%s; phosphor still moving on %ld of them\n", n, v2 ? "cuda2" : path.c_str(),
            path == "cuda" && !cuda_ok ? " (FELL BACK TO PASSTHROUGH)" : "", n_active);
    return 0;
}
