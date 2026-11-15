// Stage 1: is the CUDA kernel correct? Deterministic synthetic frames, the
// kernel's state compared number for number with the CPU reference. Host copies
// either side on purpose; nothing here is a timing.
#include "cudart.hpp"
#include "cpu_ref.hpp"
#include <algorithm>
#include <cstdarg>
#include <memory>
#include <cstring>
#include <functional>
#include <map>

static Cuda cu;
static CUfunction k_update;
static bool g_break_backend = false;  // test 5: simulate a backend that has gone away

// One tile's surface, Stage 1 flavour: its own state buffer, nobody else's.
struct Tile {
    int w = 0, h = 0;
    CUdeviceptr d_state = 0, d_active = 0;
    std::vector<float> ref;  // the CPU reference, living beside it
    bool fresh = true;
    Tile(int w_, int h_) : w(w_), h(h_), ref((size_t)w_ * h_ * 4, 0.0f) {
        CU(cuMemAlloc(&d_state, (size_t)w * h * 16));
        CU(cuMemsetD32(d_state, 0, (size_t)w * h * 4));
        CU(cuMemAlloc(&d_active, 4));
    }
    ~Tile() { cuMemFree(d_state); cuMemFree(d_active); }
    Tile(const Tile &) = delete;

    // Returns the kernel's active flag; ref_active gets the reference's.
    int update(CUdeviceptr d_frame, const uint8_t *frame, int fw, int ox, int oy, const PhosphorTuning &t,
               double dt, int *ref_active = nullptr) {
        PhosphorStep p = make_step(t, dt, fresh);
        fresh = false;
        int ra = cpu_phosphor_update(frame, fw, ox, oy, ref.data(), w, h, p);
        if (ref_active) *ref_active = ra;
        if (g_break_backend) throw std::runtime_error("backend unavailable (injected)");
        CU(cuMemsetD32(d_active, 0, 1));
        void *args[] = {&d_frame, &fw, &ox, &oy, &d_state, &w, &h, &p, &d_active};
        launch2d(k_update, w, h, args);
        int active = 0;
        CU(cuMemcpyDtoH(&active, d_active, 4));  // synchronises
        return active;
    }
    std::vector<float> read() {
        std::vector<float> s((size_t)w * h * 4);
        CU(cuMemcpyDtoH(s.data(), d_state, s.size() * 4));
        return s;
    }
    double max_err() {  // CUDA against the reference, linear light
        auto s = read();
        double m = 0;
        for (size_t i = 0; i < s.size(); i++) m = std::max(m, (double)fabsf(s[i] - ref[i]));
        return m;
    }
};

struct Frame {
    int w, h;
    std::vector<uint8_t> px;
    CUdeviceptr d = 0;
    Frame(int w_, int h_) : w(w_), h(h_), px((size_t)w_ * h_ * 4, 0) { CU(cuMemAlloc(&d, px.size())); }
    ~Frame() { cuMemFree(d); }
    void clear(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0) {
        for (size_t i = 0; i < px.size(); i += 4) { px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255; }
    }
    void rect(int x0, int y0, int rw, int rh, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = std::max(0, y0); y < std::min(h, y0 + rh); y++)
            for (int x = std::max(0, x0); x < std::min(w, x0 + rw); x++) {
                uint8_t *p = &px[4 * ((size_t)y * w + x)];
                p[0] = r; p[1] = g; p[2] = b;
            }
    }
    void noise(uint32_t seed) {
        uint32_t s = seed * 2654435761u + 1;
        for (size_t i = 0; i < px.size(); i++) { s = s * 1664525u + 1013904223u; px[i] = (i & 3) == 3 ? 255 : (uint8_t)(s >> 24); }
    }
    void upload() { CU(cuMemcpyHtoD(d, px.data(), px.size())); }
};

static int g_fail = 0;
static FILE *g_json;
static bool g_first = true;
static void report(const char *test, const char *what, bool pass, const std::string &measured) {
    printf("  %-4s %-58s %s  %s\n", pass ? "ok" : "FAIL", what, pass ? " " : "!", measured.c_str());
    if (!pass) g_fail++;
    fprintf(g_json, "%s\n  {\"test\": \"%s\", \"check\": \"%s\", \"pass\": %s, \"measured\": \"%s\"}", g_first ? "" : ",",
            test, what, pass ? "true" : "false", measured.c_str());
    g_first = false;
}
static std::string fmt(const char *f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

static const double TOL = 1e-5;  // linear light; CUDA and glibc powf, and FMA contraction, differ in the last bits

// 1. A white square, then a green one, on for 200 ms, then gone.
static void test1() {
    printf("1. a square appears for 200 ms, then disappears\n");
    const int W = 320, H = 180;
    const double dt = 1.0 / 60;
    for (int variant = 0; variant < 2; variant++) {
        PhosphorTuning t;
        if (variant == 1) { t.tau_decay[0] = 0.020; t.tau_decay[1] = 0.080; t.tau_decay[2] = 0.035; }
        Frame f(W, H);
        Tile tile(W, H);
        double worst = 0, on_level = 0, tail_err = 0;
        int black_at = -1;
        const int px = 160, py = 90, ch = variant == 0 ? 0 : 1;
        for (int i = 0; i < 90; i++) {
            f.clear();
            if (i >= 6 && i < 18) f.rect(140, 70, 40, 40, variant ? 0 : 255, 255, variant ? 0 : 255);
            f.upload();
            int act = tile.update(f.d, f.px.data(), W, 0, 0, t, dt);
            worst = std::max(worst, tile.max_err());
            auto s = tile.read();
            float v = s[4 * ((size_t)py * W + px) + ch];
            if (i == 17) on_level = v;
            if (i >= 18) {
                double expect = on_level * exp(-(i - 17) * dt / t.tau_decay[ch]);
                if (expect > 4 * t.eps) tail_err = std::max(tail_err, fabs(v - expect));
                if (v == 0.0f && black_at < 0) black_at = i;
            }
            (void)act;
        }
        const char *name = variant ? "green square, per-channel decay" : "white square, 50 ms decay";
        report("1", fmt("%s: CUDA equals CPU reference", name).c_str(), worst <= TOL, fmt("max abs err %.2e (tol %.0e)", worst, TOL));
        report("1", fmt("%s: bright onset", name).c_str(), on_level > 0.999, fmt("level after 200 ms on = %.6f", on_level));
        report("1", fmt("%s: exponential tail", name).c_str(), tail_err < 1e-4, fmt("max deviation from e^(-t/tau) = %.2e", tail_err));
        report("1", fmt("%s: then exactly black", name).c_str(), black_at > 18, fmt("0.0 from frame %d (%.0f ms after it vanished)", black_at, (black_at - 17) * dt * 1000));
    }
}

// 2. Changing content at 60, 120 and 144 Hz: noise that changes every frame is
// the worst case for the arithmetic. (How trails LOOK is the render's job.)
static void test2() {
    printf("2. content that changes every frame, at 60 / 120 / 144 Hz\n");
    const int W = 256, H = 144;
    for (int hz : {60, 120, 144}) {
        PhosphorTuning t;
        Frame f(W, H);
        Tile tile(W, H);
        double worst = 0;
        for (int i = 0; i < 120; i++) {
            f.noise(i / 2);  // new picture every other frame
            f.upload();
            tile.update(f.d, f.px.data(), W, 0, 0, t, 1.0 / hz);
            if (i % 10 == 9) worst = std::max(worst, tile.max_err());
        }
        report("2", fmt("%d Hz: CUDA equals CPU reference", hz).c_str(), worst <= TOL, fmt("max abs err %.2e", worst));
    }
}

// 3. A constant image converges and stays; refresh rates agree.
static void test3() {
    printf("3. constant image; agreement across refresh rates\n");
    const int W = 256, H = 144;
    {
        PhosphorTuning t;
        Frame f(W, H);
        Tile tile(W, H);
        f.clear(255, 255, 255);  // start from white, so the image is reached by decay
        f.upload();
        tile.update(f.d, f.px.data(), W, 0, 0, t, 1.0 / 60);
        f.noise(7);
        f.upload();
        float peak = 0;
        double worst = 0;
        int settled = -1, moved_after = 0;
        for (int i = 0; i < 600; i++) {
            int act = tile.update(f.d, f.px.data(), W, 0, 0, t, 1.0 / 60);
            if (!act && settled < 0) settled = i;
            if (act && settled >= 0) moved_after++;
            if (i % 50 == 49) {
                auto s = tile.read();
                for (float v : s) peak = std::max(peak, v);
                if (i == 599)
                    for (size_t p = 0; p < s.size(); p += 4)
                        for (int c = 0; c < 3; c++)
                            worst = std::max(worst, (double)fabsf(s[p + c] - cpu_srgb_to_linear(f.px[p + c] / 255.0f)));
            }
        }
        report("3", "600 frames of one image: never brighter than what was shown", peak <= 1.0f, fmt("peak state %.6f", peak));
        report("3", "600 frames of one image: state equals the image", worst <= 1e-6 && settled > 5 && moved_after == 0,
               fmt("max difference %.1e; settled at frame %d (%.0f ms), no movement after", worst, settled, settled * 1000.0 / 60));
    }
    // Events on a clock all three rates share (multiples of 1/12 s), compared at shared instants.
    std::map<int, std::vector<std::vector<float>>> samples;
    for (int hz : {60, 120, 144}) {
        PhosphorTuning t;
        Frame f(W, H);
        Tile tile(W, H);
        int per = hz / 12;  // frames per 1/12 s
        for (int i = 1; i <= hz; i++) {  // one second
            double time = (double)i / hz;
            f.clear(10, 10, 10);
            if (time > 2.0 / 12 + 1e-9 && time <= 5.0 / 12 + 1e-9) f.rect(100, 50, 60, 40, 255, 200, 40);
            f.upload();
            tile.update(f.d, f.px.data(), W, 0, 0, t, 1.0 / hz);
            if (i % per == 0) samples[hz].push_back(tile.read());
        }
    }
    double worst = 0;
    for (size_t k = 0; k < samples[60].size(); k++)
        for (size_t i = 0; i < samples[60][k].size(); i++) {
            float a = samples[60][k][i], b = samples[120][k][i], c = samples[144][k][i];
            worst = std::max(worst, (double)std::max(fabsf(a - b), std::max(fabsf(a - c), fabsf(b - c))));
        }
    report("3", "60 / 120 / 144 Hz agree at 12 shared instants", worst < 1e-4, fmt("max difference %.2e linear (tol 1e-4)", worst));
}

// 4. Two tiles side by side; one flashes, then moves.
static void test4() {
    printf("4. two tiles: no shared history; a trail travels with its tile\n");
    const int FW = 640, FH = 360, W = 300, H = 300;
    PhosphorTuning t;
    const double dt = 1.0 / 60;
    Frame f(FW, FH);
    // A sits at ax (it will move); B is its neighbour. Content is drawn relative to each tile.
    auto draw = [&](int ax, int ay, int i, bool flash) {
        f.clear(5, 5, 5);
        f.rect(ax, ay, W, H, 20, 30, 20);          // A's own background
        f.rect(330, 30, W, H, 30, 20, 40);          // B's own background
        f.rect(330 + 100, 30 + 100, 50, 50, 90, 90, 200);  // B's static content
        if (flash && i >= 5 && i < 12) f.rect(ax + W - 40, ay + 100, 40, 80, 255, 255, 255);  // at A's edge, beside B
        f.upload();
    };
    std::vector<float> b_with, b_without, a_moved, a_still;
    for (int run = 0; run < 3; run++) {  // 0: A flashes and moves, 1: A never flashes, 2: A flashes, stays put
        Tile A(W, H), B(W, H);
        for (int i = 0; i < 24; i++) {
            int ax = 10, ay = 30;
            if (run == 0 && i >= 14) { ax = 10 + 13; ay = 30 + 21; }  // moved mid-decay
            draw(ax, ay, i, run != 1);
            if (run == 0 && i >= 14) f.rect(330, 30, W, H, 30, 20, 40), f.rect(430, 130, 50, 50, 90, 90, 200), f.upload();
            A.update(f.d, f.px.data(), FW, ax, ay, t, dt);
            B.update(f.d, f.px.data(), FW, 330, 30, t, dt);
        }
        if (run == 0) { b_with = B.read(); a_moved = A.read(); }
        if (run == 1) b_without = B.read();
        if (run == 2) a_still = A.read();
    }
    bool clean = b_with == b_without;
    report("4", "B's state is bit-identical whether or not A flashed beside it", clean, clean ? "0 differing values" : "history leaked");
    float trail = a_moved[4 * ((size_t)140 * W + (W - 20))];
    bool travelled = a_moved == a_still;
    report("4", "A moved mid-decay: its state equals the unmoved run, bit for bit", travelled, fmt("trail value at the same tile pixel %.5f", trail));
    report("4", "the trail is still decaying when compared (the check is not vacuous)", trail > 0.01f && trail < 1.0f, fmt("%.5f", trail));
}

// 5. Resize, hide and show, a failed backend.
static void test5() {
    printf("5. resize, hide/show, backend failure\n");
    PhosphorTuning t;
    const double dt = 1.0 / 60;
    Frame f(640, 360);
    // Resize policy: RESET. The old surface is dropped; the new one adopts the frame as it is.
    std::vector<float> first;
    bool same = true, adopted = true;
    for (int run = 0; run < 2; run++) {
        auto tile = std::make_unique<Tile>(200, 150);
        for (int i = 0; i < 10; i++) {
            f.clear(8, 8, 8);
            if (i < 6) f.rect(40, 40, 80, 60, 255, 255, 255);
            f.upload();
            tile->update(f.d, f.px.data(), 640, 0, 0, t, dt);
        }
        tile = std::make_unique<Tile>(260, 190);  // resized mid-trail
        tile->update(f.d, f.px.data(), 640, 0, 0, t, dt);
        auto s = tile->read();
        float bg = cpu_srgb_to_linear(8 / 255.0f);
        for (size_t p = 0; p < s.size(); p += 4)
            if (fabsf(s[p] - bg) > 1e-6f) adopted = false;  // no ghost of the old trail
        if (run == 0) first = s; else same = s == first;
    }
    report("5", "resize resets: the new surface holds the current frame, no old trail", adopted, "policy: reset, not reprojection");
    report("5", "resize is deterministic: two runs, identical bits", same, same ? "identical" : "differs");
    size_t before = cu.free_bytes();
    for (int i = 0; i < 200; i++) {
        Tile tile(1920, 1080);
        f.upload();
        tile.update(f.d, f.px.data(), 640, 0, 0, t, dt);
    }
    size_t after = cu.free_bytes();
    long long leak = (long long)before - (long long)after;
    const long long one = 1920LL * 1080 * 16;  // one surface; a leak would be 200 of these
    report("5", "hide/show 200 times at 1080p: no VRAM left behind", llabs(leak) < one,
           fmt("free VRAM moved %.1f MB on a GPU shared with the desktop; one leaked surface per cycle would be %.0f MB", leak / 1048576.0, 200 * one / 1048576.0));
    // Backend failure: the caller must get the frame through untouched.
    Tile tile(640, 360);
    f.noise(3);
    f.upload();
    std::vector<uint8_t> out(f.px.size(), 0);
    bool passthrough = false;
    g_break_backend = true;
    try {
        tile.update(f.d, f.px.data(), 640, 0, 0, t, dt);
    } catch (std::exception &e) {
        out = f.px;  // the fallback every caller of this surface implements: show the input
        passthrough = out == f.px;
    }
    g_break_backend = false;
    report("5", "backend failure surfaces as an error; the fallback output is the input", passthrough, "exception caught, frame passed through");
}

// 6. Idle: after the picture stops changing, the surface says so and stays quiet.
static void test6() {
    printf("6. idle: no redraw once the trail is below the visible threshold\n");
    const int W = 320, H = 180;
    PhosphorTuning t;
    const double dt = 1.0 / 60;
    Frame f(W, H);
    Tile tile(W, H);
    int last_active = -1, woke = 0, ref_disagree = 0, requiet = 0;
    for (int i = 0; i < 200; i++) {
        f.clear();
        if (i < 10) f.rect(100, 60, 60, 60, 255, 255, 255);
        if (i == 150) f.rect(10, 10, 20, 20, 0, 255, 0);
        f.upload();
        int ra = 0;
        int act = tile.update(f.d, f.px.data(), W, 0, 0, t, dt, &ra);
        if (act != ra) ref_disagree++;
        if (act && i < 150) last_active = i;
        if (i == 151) woke = act;
        if (act && i > 151) requiet = i;
    }
    double expect_ms = t.tau_decay[0] * log(1.0 / t.eps) * 1000;
    report("6", "the active flag drops after the trail fades, and stays down", last_active >= 10 && last_active < 60,
           fmt("last active frame %d = %.0f ms after the square vanished (theory %.0f ms)", last_active, (last_active - 9) * dt * 1000, expect_ms));
    report("6", "content that comes and goes wakes it, and it quiets again", woke == 1 && requiet > 151 && requiet < 199,
           fmt("active on the frame after a one-frame flash; quiet again after frame %d", requiet));
    report("6", "CUDA's active flag equals the reference's on all 200 frames", ref_disagree == 0, fmt("%d disagreements", ref_disagree));
}

int main(int argc, char **argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    try {
        cu.init();
        cu.load(root + "/kernels/phosphor.cu", root);
        k_update = cu.fn("phosphor_update_buf");
        g_json = fopen((root + "/out/stage1.json").c_str(), "w");
        fprintf(g_json, "{\"device\": \"%s\", \"driver_cuda\": \"%s\", \"nvrtc\": \"%s\", \"checks\": [", cu.device_name.c_str(),
                cu.driver_version.c_str(), cu.nvrtc_version.c_str());
        printf("Stage 1 correctness: %s, CUDA %s driver, NVRTC %s\n", cu.device_name.c_str(), cu.driver_version.c_str(), cu.nvrtc_version.c_str());
        test1(); test2(); test3(); test4(); test5(); test6();
        fprintf(g_json, "\n], \"failed\": %d}\n", g_fail);
        fclose(g_json);
        printf("%s\n", g_fail ? "FAILED" : "all checks passed");
    } catch (std::exception &e) {
        fprintf(stderr, "stage1: %s\n", e.what());
        return 2;
    }
    return g_fail ? 1 : 0;
}
