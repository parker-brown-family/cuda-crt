// cuda-pass: the CUDA CRT running INSIDE Hyprland.
//
// A plugin that, after the windows are drawn and before the top and overlay
// layers, copies part of the finished picture into a texture, hands it to the
// same CUDA kernels the lab measures (phosphor update, halation, glass), and
// copies the result back over that part only.
//
// WHICH part is the `target` file:
//   (absent)         the whole monitor as one tube: the original tracer
//   0x55d0...        one window, by its address (hyprctl clients -j)
//   class:chromium   the first visible window of that class
//   title:still      the first visible window whose title contains that text
// The phosphor belongs to the window: it is keyed to the window's identity and
// sized to it, so moving the window carries its trails along and resizing
// starts it fresh. Pixels outside the window are never read or written.
//
// HOW it looks is the `look` file:
//   tube      (default) the lab's own glass: curve, rim, scanlines, bar, glare
//   phosphor  only what has memory: fade, burn ghost, glow. For a desktop that
//             already wears a glass shader, so the two do not stack.
//
// Everything it reads and writes lives in ONE folder: $CUDA_CRT_STATE, else
// $XDG_STATE_HOME/cuda-crt, else ~/.local/state/cuda-crt. Switches are files
// there, read every frame:
//   root           (required) one line: the path of the cuda-crt checkout
//   set            name=value pairs laid over the look, e.g. curv=1.5,scan=0.3,ghost=0.02,
//                  decay=120 (ms). Read every frame: rewrite it and the window changes
//   off            stand the pass down and release its textures
//   input-flat     keep the picture bent but stop mapping the pointer (a test's control)
//   freeze         a number of seconds: hold the clock there (the bar, the jiggle) for a test
//   ttl            a number of seconds: stand down by itself after that long
//   fail           throw in the middle of a frame, once, to prove the fallback
//   keep-drawing   damage every frame (a capture needs a frame)
//   time-pass      finish the GPU either side of the pass and log its wall time
//   dump-now       write the finished frame to frame.rgba / frame.txt
// It writes: pass.log, frames.now, and perf.csv (one row every five seconds while
// it runs: passes per second, kernel time from CUDA's own timer, the pass's wall
// time, and how often the phosphor was still fading); and, when asked,
// pass-ms.txt, frame.rgba, frame.txt.
//
// It follows the shape of omarchy-crt's native/crt-pass.cpp, which is the pass
// people actually run. This one is a proof: load it by hand, unload it by hand.
#include <GLES3/gl32.h>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include "../src/cuda_path.hpp"
#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/Compositor.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/desktop/Workspace.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/managers/fullscreen/FullscreenController.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/helpers/math/Math.hpp>
#include <hyprland/src/state/MonitorState.hpp>
#define private public     // m_currentlyHeldButtons, read-only here (as omarchy-crt's native pass)
#include <hyprland/src/managers/input/InputManager.hpp>
#undef private
#include <hyprland/src/managers/SessionLockManager.hpp>
#include <hyprland/src/pointer/PointerManager.hpp>
#include <hyprland/src/desktop/state/ViewStateTracker.hpp>
#include <hyprland/src/desktop/state/ViewHitTester.hpp>
#include <hyprland/src/desktop/view/LayerSurface.hpp>
#include <hyprland/protocols/wlr-layer-shell-unstable-v1.hpp>
#include <hyprland/src/protocols/core/DataDevice.hpp>

static HANDLE              PHANDLE = nullptr;
static CHyprSignalListener g_stage;
using Clock = std::chrono::steady_clock;

static const std::string& dir() {
    static const std::string d = [] {
        if (const char* e = getenv("CUDA_CRT_STATE")) return std::string(e);
        const char *x = getenv("XDG_STATE_HOME"), *h = getenv("HOME");
        return (x ? std::string(x) : std::string(h ? h : "/tmp") + "/.local/state") + "/cuda-crt";
    }();
    return d;
}
static bool        flag(const char* name) { return access((dir() + "/" + name).c_str(), F_OK) == 0; }
static std::string line(const char* name) { std::string s; std::getline(std::ifstream(dir() + "/" + name), s); return s; }
static void        note(const std::string& text) { std::ofstream(dir() + "/pass.log", std::ios::app) << text << "\n"; }

struct SRect { int x = 0, y = 0, w = 0, h = 0; };

struct SEngine {
    enum { NEW, READY, DEAD, DONE } state = NEW;  // DONE = stood down on purpose (off, or its time was up)
    Cuda     cu;
    CudaPath path;
    bool     cudaUp = false;
    CudaSurf surf;                 // the target's phosphor
    void*    owner = nullptr;      // whose phosphor it is: a window, or a monitor
    void*    monitor = nullptr;    // the monitor the target was last drawn on
    Shared   shIn, shOut;
    GLuint   inTex = 0, inFb = 0, outTex = 0, outFb = 0;
    int      w = 0, h = 0;
    Look     look;
    Clock::time_point start, last, liveSince;
    bool     active = false, failedOnce = false;
    long     frames = 0, timed = 0;
    double   msSum = 0, msMax = 0;
    // always-on tracking, cheap: no extra waiting on the GPU
    CUevent  ev0 = nullptr, ev1 = nullptr;
    // The "still fading?" answer and the kernels' time arrive one frame late, so the
    // compositor never waits for the GPU: the flag is copied to pinned memory after the
    // kernels, and read at the start of the next frame once ev1 says it has landed.
    int*     activeHost = nullptr;
    bool     pending = false;
    Clock::time_point perfSince;
    long     perfN = 0, perfActive = 0, perfFresh = 0, kN = 0;
    double   kSum = 0, kMax = 0, wallSum = 0, wallMax = 0;
    // where the target was drawn last frame, in the layout's own (logical) pixels,
    // for the pointer: it must be mapped through exactly the box the picture was bent in
    CBox     box;
    void*    boxMon  = nullptr;
    bool     boxLive = false;    // false: nothing drawn yet, or the box was clipped by its monitor
    bool     boxFullscreen = false;   // the target covers its monitor: Hyprland then ignores TOP layers for input
    std::string why;             // the pointer's last decision, written to pointer.why when it changes
    bool     hooksOk = false;    // false: the pointer cannot be mapped, so nothing may be bent
    TubeMotion motion = {};      // the last frame's transient motion, for the pointer
    float    tileW = 0, tileH = 0;   // the last frame's tile, framebuffer pixels
};
static SEngine g;

static void target(GLuint& tex, GLuint& fb, int w, int h) {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) throw std::runtime_error("scratch framebuffer incomplete");
}

// Let go of everything that belongs to a framebuffer size. A texture must be
// unregistered from CUDA before it is deleted. Needs the GL context current.
static void release() {
    g.shIn.unreg(); g.shOut.unreg();
    g.surf.destroy();
    g.surf.w = g.surf.h = 0;
    g.owner = nullptr;
    if (g.inTex) { glDeleteFramebuffers(1, &g.inFb); glDeleteTextures(1, &g.inTex); glDeleteFramebuffers(1, &g.outFb); glDeleteTextures(1, &g.outTex); }
    g.inTex = g.inFb = g.outTex = g.outFb = 0;
    g.w = g.h = 0;
}

static void resize(int w, int h) {
    release();
    target(g.inTex, g.inFb, w, h);
    target(g.outTex, g.outFb, w, h);
    g.shIn.reg(g.inTex, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
    g.shOut.reg(g.outTex, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
    g.surf.create(w, h, true);   // room for the largest window this framebuffer can hold; see CudaSurf::use
    g.w = w; g.h = h;
    note("framebuffer " + std::to_string(w) + "x" + std::to_string(h) + ": two GLES textures registered with CUDA");
}

static void init() {
    const std::string root = line("root");
    if (root.empty()) throw std::runtime_error("no `root` file in " + dir());
    if (!g.cudaUp) {
        const int dev = cuda_device_for_gl();
        note("CUDA device that owns this GL context: " + (dev < 0 ? std::string("could not tell (using device 0)") : std::to_string(dev)));
        g.cu.init(dev);
        g.cu.load_all(root);
        g.path.init(g.cu);
        CU(cuEventCreate(&g.ev0, CU_EVENT_DEFAULT));
        CU(cuEventCreate(&g.ev1, CU_EVENT_DEFAULT));
        CU(cuMemHostAlloc((void**)&g.activeHost, sizeof(int), 0));
        *g.activeHost = 1;
        g.cudaUp = true;
    }
    g.start = g.last = g.liveSince = g.perfSince = Clock::now();
    note("ready: " + g.cu.device_name + ", CUDA driver API " + g.cu.driver_version + ", NVRTC " + g.cu.nvrtc_version + ", GL " + (const char*)glGetString(GL_VERSION) +
         ", kernel files compiled this start: " + std::to_string(g.cu.compiled) + ", target: " + (line("target").empty() ? "the whole monitor" : line("target")) +
         ", look: " + (line("look").empty() ? "tube" : line("look")));
}

static bool matches(const PHLWINDOW& w, const std::string& spec) {
    if (spec.rfind("title:", 0) == 0) return w->m_title.find(spec.substr(6)) != std::string::npos;
    if (spec.rfind("class:", 0) == 0) return w->m_class == spec.substr(6);
    char buf[32];
    snprintf(buf, sizeof buf, "0x%lx", (unsigned long)(uintptr_t)w.get());
    return spec == buf;
}

// Where the target is on this monitor's framebuffer, in pixels, clipped to it.
// False when the target is not being drawn here: then there is nothing to do.
static bool where(PHLMONITOR m, int fbw, int fbh, SRect& r, void*& owner) {
    const std::string spec = line("target");
    if (spec.empty()) {
        r = {0, 0, fbw, fbh}; owner = m.get();
        g.box = m->logicalBox(); g.boxMon = m.get(); g.boxLive = m->m_transform == WL_OUTPUT_TRANSFORM_NORMAL;
        g.boxFullscreen = false;
        return true;
    }
    for (auto const& w : Desktop::windowState()->windows()) {
        if (!w->m_isMapped || w->isHidden() || !g_pHyprRenderer->shouldRenderWindow(w, m) || !matches(w, spec))
            continue;
        CBox logical = {w->positionAnimation()->value(), w->sizeAnimation()->value()};
        // where the renderer really draws it: a sliding workspace carries its windows
        if (w->m_workspace && !w->m_pinned && w->m_workspace->m_renderOffset->isBeingAnimated())
            logical.translate(w->m_workspace->m_renderOffset->value());
        logical.translate(w->m_floatingOffset);
        CBox fb = {(logical.pos() - m->m_position) * m->m_scale, logical.size() * m->m_scale};
        fb.transform(Math::wlTransformToHyprutils(Math::invertTransform(m->m_transform)), m->m_transformedSize.x, m->m_transformedSize.y);
        const int x0 = std::max(0, (int)std::floor(fb.x)), y0 = std::max(0, (int)std::floor(fb.y));
        const int x1 = std::min(fbw, (int)std::ceil(fb.x + fb.w)), y1 = std::min(fbh, (int)std::ceil(fb.y + fb.h));
        if (x1 - x0 < 16 || y1 - y0 < 16) return false;  // off this monitor, or too small to bother
        r = {x0, y0, x1 - x0, y1 - y0};
        owner = w.get();
        // a box cut off by its monitor's edge is drawn as a smaller tile than the window:
        // the pointer would be mapped through the wrong box, so it is not mapped at all
        // Half a pixel of slack: a fullscreen window's box comes out a hair larger than its
        // monitor (1600 x 1.6 is not exactly 2560 in floating point), and an exact test
        // called it "cut off" and switched the pointer mapping off for every fullscreen window.
        const bool whole = fb.x >= -0.5 && fb.y >= -0.5 && fb.x + fb.w <= fbw + 0.5 && fb.y + fb.h <= fbh + 0.5;
        g.box = logical; g.boxMon = m.get(); g.boxLive = whole && m->m_transform == WL_OUTPUT_TRANSFORM_NORMAL;
        g.boxFullscreen = Fullscreen::controller()->isFullscreen(w, Fullscreen::FSMODE_FULLSCREEN);
        return true;
    }
    return false;
}

static void dress(bool phosphorOnly) {
    Look k;  // the lab's defaults
    k.track_rest = 2.0f; k.track_sweep = 5.0f; k.degauss_at = 1.0f;
    // On a desktop the rim must not sit on a window's own edges (its controls live
    // there), so the picture is drawn 6% smaller with the curve across the whole edge.
    // Proven with real clicks in hypr/prove-aim; `set` can still change either.
    k.core = 0.0f; k.inset = 0.06f;
    if (phosphorOnly) {
        // no glass of our own: nothing bends, nothing sweeps, nothing is driven by the
        // clock. What is left is the light's memory, so an idle window owes no frames.
        k.curv = 0; k.aberr = 0; k.scan = 0; k.bloom = 0; k.vign = 0; k.specular = 0; k.glare = 0;
        k.tracking = 0; k.flicker = 0; k.jiggle = 0; k.mask = 0; k.degauss_at = -1.0f;
        k.track_bend = k.track_noise = k.track_chroma = 0;
        // Gentle glow. The lab's defaults were tuned on dark terminals; on a bright page
        // (a browser) they lift black text to grey. These keep text crisp on white.
        k.halation = 0.05f; k.glow = 0.10f; k.beam = 0.15f; k.glow_knee = 0.25f;
    }
    // `set`: name=value pairs (commas or new lines) laid over the look, read every
    // frame, so a dial can be turned on a running window by rewriting one small file.
    std::ifstream f(dir() + "/set");
    std::string   all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    static const std::pair<const char*, float Look::*> dials[] = {
        {"curv", &Look::curv}, {"scan", &Look::scan}, {"scan_step", &Look::scan_step}, {"glow", &Look::glow}, {"bloom", &Look::bloom},
        {"vign", &Look::vign}, {"specular", &Look::specular}, {"glare", &Look::glare}, {"aberr", &Look::aberr}, {"mask", &Look::mask},
        {"tracking", &Look::tracking}, {"track_rest", &Look::track_rest}, {"track_sweep", &Look::track_sweep}, {"band_h", &Look::band_h},
        {"track_bend", &Look::track_bend}, {"track_noise", &Look::track_noise}, {"track_chroma", &Look::track_chroma},
        {"jiggle", &Look::jiggle}, {"flicker", &Look::flicker}, {"halation", &Look::halation}, {"glow_knee", &Look::glow_knee}, {"beam", &Look::beam}, {"core", &Look::core}, {"inset", &Look::inset}};
    size_t pos = 0;
    while (pos < all.size()) {
        size_t end = all.find_first_of(",\n", pos);
        if (end == std::string::npos) end = all.size();
        const std::string pair = all.substr(pos, end - pos);
        const size_t      eq   = pair.find('=');
        if (eq != std::string::npos) {
            std::string name = pair.substr(0, eq);
            name.erase(0, name.find_first_not_of(" \t")); name.erase(name.find_last_not_of(" \t") + 1);
            const float v = (float)atof(pair.c_str() + eq + 1);
            for (auto& d : dials)
                if (name == d.first && std::isfinite(v)) k.*(d.second) = std::min(1000.0f, std::max(0.0f, v));
            if (name == "ghost") k.burn.gain = std::min(0.05f, std::max(0.0f, v));
            if (name == "ghost_decay") k.burn.tau_decay = std::min(10.0, std::max(0.02, (double)v / 1000.0));
            if (name == "decay") for (double& tau : k.phosphor.tau_decay) tau = std::min(2.0, std::max(0.001, (double)v / 1000.0));
        }
        pos = end + 1;
    }
    // Never bend a picture the pointer cannot be mapped through.
    if (!g.hooksOk) k.curv = 0;
    g.look = k;
}

class CCudaPass : public IPassElement {
  public:
    bool             needsLiveBlur() override { return false; }
    bool             needsPrecomputeBlur() override { return false; }
    const char*      passName() override { return "CCudaPass"; }
    ePassElementType type() override { return EK_CUSTOM; }
    bool             undiscardable() override { return true; }
    bool             disableSimplification() override { return true; }  // the glass reads away from each pixel

    std::vector<UP<IPassElement>> draw() override {
        if (g.state == SEngine::DEAD || g.state == SEngine::DONE) return {};
        auto m = g_pHyprRenderer->m_renderData.pMonitor.lock();
        if (!m) return {};
        GLint fb = 0, vp[4] = {}, active = 0, tex = 0;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fb);
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
        const GLboolean blend = glIsEnabled(GL_BLEND), scissor = glIsEnabled(GL_SCISSOR_TEST);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex);
        const int w = vp[2], h = vp[3];
        try {
            const std::string ttl = line("ttl");
            const bool        up  = g.state == SEngine::READY;
            if (flag("off") || (up && !ttl.empty() && std::chrono::duration<double>(Clock::now() - g.liveSince).count() > atof(ttl.c_str()))) {
                if (up) { CU(cuCtxSetCurrent(g.cu.ctx)); release(); note(flag("off") ? "stood down: off" : "stood down: its time was up"); }
                g.state = flag("off") ? SEngine::NEW : SEngine::DONE;
                g.active = false;
            } else {
                SRect r;
                void* owner = nullptr;
                // This monitor is being drawn now: until where() finds the target on it again,
                // the pointer is not mapped here (another workspace, a closed window, a move).
                if (m.get() == g.boxMon) g.boxLive = false;
                if (where(m, w, h, r, owner)) {
                    if (g.state == SEngine::NEW) { init(); g.state = SEngine::READY; }
                    CU(cuCtxSetCurrent(g.cu.ctx));
                    if (w != g.w || h != g.h) resize(w, h);
                    if (owner != g.owner || r.w != g.surf.w || r.h != g.surf.h) {  // a new owner or a new size starts fresh
                        // One line when the target changes hands or settles at a new size, not one per
                        // frame of a resize animation: a fullscreen toggle is a dozen sizes in a row.
                        if (owner != g.owner)
                            note("phosphor " + std::to_string(r.w) + "x" + std::to_string(r.h) + " at " + std::to_string(r.x) + "," + std::to_string(r.y) + " on a " + std::to_string(w) + "x" + std::to_string(h) + " framebuffer");
                        g.surf.use(r.w, r.h);
                        g.owner = owner;
                        g.perfFresh++;
                    }
                    g.monitor = m.get();
                    dress(line("look") == "phosphor");

                    // never read the attachment being drawn to: the lower scene is copied first
                    glDisable(GL_SCISSOR_TEST);
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
                    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g.inFb);
                    glBlitFramebuffer(r.x, r.y, r.x + r.w, r.y + r.h, r.x, r.y, r.x + r.w, r.y + r.h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

                    const bool timed = flag("time-pass");
                    if (timed) glFinish();
                    const auto   t0 = Clock::now();
                    // `freeze`: a fixed clock (seconds) for tests, so the bar holds still between a capture and a click
                    const std::string fz = line("freeze");
                    const float  t  = fz.empty() ? std::chrono::duration<float>(t0 - g.start).count() : (float)atof(fz.c_str());
                    const double dt = std::min(0.25, std::max(0.002, std::chrono::duration<double>(t0 - g.last).count()));
                    g.last = t0;

                    // Last frame's answers, if they have landed. If not (the GPU is behind),
                    // assume still fading: one extra frame is the cost of never waiting.
                    float kms = -1;
                    if (g.pending && cuEventQuery(g.ev1) == CUDA_SUCCESS) {
                        g.active = *g.activeHost != 0;
                        CU(cuEventElapsedTime(&kms, g.ev0, g.ev1));
                        g.pending = false;
                    } else if (g.pending) {
                        g.active = true;
                    }
                    g.path.begin(g.shIn, &g.shOut);
                    if (flag("fail") && !g.failedOnce) { g.failedOnce = true; throw std::runtime_error("injected failure, mid-frame, textures mapped"); }
                    PhosphorStepV2 step = make_step_v2(g.look.phosphor, g.look.burn, dt, g.surf.fresh);
                    // A fresh corner of a reused surface may hold an older window's ghost: on a
                    // fresh frame the ghost adopts the picture too, so nothing stale can show.
                    if (g.surf.fresh) step.a_burn_rise = step.a_burn_decay = 1.0f;
                    if (!g.pending) CU(cuEventRecord(g.ev0, 0));   // never re-record ev0 under an unread ev1
                    g.path.update_v2(g.surf, r.x, r.y, step);
                    g.path.glow(g.surf, g.look.glow_knee);
                    const OpticsParams op = make_optics(g.look, t, r.h, true);
                    g.path.optics(g.surf, r.x, r.y, op);
                    g.motion = make_motion(op); g.tileW = (float)r.w; g.tileH = (float)r.h;
                    if (!g.pending) {
                        CU(cuMemcpyDtoHAsync(g.activeHost, g.path.d_active, sizeof(int), 0));
                        CU(cuEventRecord(g.ev1, 0));
                        g.pending = true;
                    }
                    g.path.end();

                    glBindFramebuffer(GL_READ_FRAMEBUFFER, g.outFb);
                    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb);
                    glBlitFramebuffer(r.x, r.y, r.x + r.w, r.y + r.h, r.x, r.y, r.x + r.w, r.y + r.h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                    if (timed) {
                        glFinish();
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                        g.msSum += ms; g.msMax = std::max(g.msMax, ms); g.timed++;
                        std::ofstream(dir() + "/pass-ms.txt") << g.msSum / g.timed << " " << g.msMax << " " << g.timed << " " << r.w << "x" << r.h << "\n";
                    }
                    g.frames++;
                    // Tracking: kernel time from CUDA's own events, and this function's wall time
                    // from the copy-in to here. Summed, and written out every five seconds.
                    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    g.perfN++; g.perfActive += g.active; g.wallSum += wall; g.wallMax = std::max(g.wallMax, wall);
                    if (kms >= 0) { g.kN++; g.kSum += kms; g.kMax = std::max(g.kMax, (double)kms); }
                    const double span = std::chrono::duration<double>(Clock::now() - g.perfSince).count();
                    if (span >= 5.0) {
                        char row[256];
                        snprintf(row, sizeof row, "%ld,%.1f,%.3f,%.3f,%.3f,%.3f,%.0f,%dx%d,%ld", (long)time(nullptr), g.perfN / span, g.kN ? g.kSum / g.kN : -1.0, g.kMax, g.wallSum / g.perfN, g.wallMax,
                                 100.0 * g.perfActive / g.perfN, r.w, r.h, g.perfFresh);
                        const bool isNew = access((dir() + "/perf.csv").c_str(), F_OK) != 0;
                        std::ofstream f(dir() + "/perf.csv", std::ios::app);
                        if (isNew) f << "unix_time,passes_per_s,kernels_ms_mean,kernels_ms_max,pass_wall_ms_mean,pass_wall_ms_max,percent_frames_still_fading,window_px,fresh_starts\n";
                        f << row << "\n";
                        g.perfSince = Clock::now(); g.perfN = g.perfActive = g.perfFresh = g.kN = 0; g.kSum = g.kMax = g.wallSum = g.wallMax = 0;
                    }
                }
            }
        } catch (std::exception& e) {
            // Let go of anything mapped, stand down for good, and leave this frame as it came.
            g.path.abandon();
            g.state  = SEngine::DEAD;
            g.active = false;
            note(std::string("DEAD, passing frames through untouched: ") + e.what());
        }
        // frame count, still-fading, state: written only when it changes, so a pass that is
        // armed and waiting for its window does not touch the disk on every frame
        static std::string wrote;
        const std::string  now = std::to_string(g.frames) + " " + (g.active ? "1" : "0") + " " + std::to_string((int)g.state);
        if (now != wrote) { std::ofstream(dir() + "/frames.now") << now << "\n"; wrote = now; }
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glBindTexture(GL_TEXTURE_2D, tex);
        glActiveTexture(active);
        if (blend) glEnable(GL_BLEND);
        if (scissor) glEnable(GL_SCISSOR_TEST);
        return {};
    }
};

// The finished frame, to a file, when a test asks (same protocol as omarchy-crt's rig/shot).
class CDump : public IPassElement {
  public:
    bool             needsLiveBlur() override { return false; }
    bool             needsPrecomputeBlur() override { return false; }
    const char*      passName() override { return "CCudaDump"; }
    ePassElementType type() override { return EK_CUSTOM; }
    bool             undiscardable() override { return true; }
    bool             disableSimplification() override { return true; }
    std::vector<UP<IPassElement>> draw() override {
        const std::string want = line("dump-now");   // empty = any monitor, else that monitor's name
        auto m = g_pHyprRenderer->m_renderData.pMonitor.lock();
        if (!want.empty() && (!m || m->m_name != want)) return {};
        GLint vp[4] = {}, fb = 0;
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fb);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
        std::vector<unsigned char> px((size_t)vp[2] * vp[3] * 4);
        glReadPixels(0, 0, vp[2], vp[3], GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        std::ofstream(dir() + "/frame.rgba", std::ios::binary).write((char*)px.data(), px.size());
        std::ofstream(dir() + "/frame.txt") << vp[2] << " " << vp[3] << " frames=" << g.frames << " state=" << (int)g.state << "\n";
        unlink((dir() + "/dump-now").c_str());
        return {};
    }
};

// ---- THE POINTER ------------------------------------------------------------
// The picture bends; the pointer does not. A click must land on what is drawn under
// the cursor, so while Hyprland works out what the pointer is over (its move routine),
// "where is the pointer" answers with the point of the window that is DRAWN at the
// pointer's true position. The drawn cursor and keybinds keep the true position.
// This is omarchy-crt's native pass, ported; the curve comes from tube_map.h, the same
// function the glass kernel calls, so the picture and the pointer cannot disagree.
static CFunctionHook*          g_positionHook = nullptr;
static CFunctionHook*          g_moveHook     = nullptr;
static CFunctionHook*          g_layerHook    = nullptr;
static std::optional<Vector2D> g_mapped;
static std::optional<CBox>     g_pressBox;   // a press belongs to the box it landed in until release

typedef Vector2D (*origPosition)(void*);
typedef SP<CWLSurfaceResource> (*origLayerAt)(void*, const Vector2D&, std::vector<PHLLSREF>*, Vector2D*, PHLLS*, bool);
typedef void (*origMove)(void*, uint32_t, bool, bool, std::optional<Vector2D>);

static Vector2D hkPosition(void* self) { return g_mapped ? *g_mapped : ((origPosition)g_positionHook->m_original)(self); }

// A mapped move has already found no layer under the true position; asked again with
// the mapped one, a pointer in a gap between two cards would hit a card.
static SP<CWLSurfaceResource> hkLayerAt(void* self, const Vector2D& pos, std::vector<PHLLSREF>* layers, Vector2D* coords, PHLLS* found, bool lock) {
    if (g_mapped) return nullptr;
    return ((origLayerAt)g_layerHook->m_original)(self, pos, layers, coords, found, lock);
}

// Screen point -> the window point drawn there. Past the box's edge the correction
// stops growing and carries on at its edge value (a held press far outside a small
// window would otherwise be thrown across the screen).
static Vector2D through(const CBox& box, const Vector2D& p, bool clamp) {
    const Vector2D in = {std::clamp(p.x, box.x, box.x + box.w), std::clamp(p.y, box.y, box.y + box.h)};
    const TubeMap  tm = make_tube(g.look);
    float          lx = (float)((in.x - box.x) / box.w), ly = (float)((in.y - box.y) / box.h), sx = 0, sy = 0, bar = 0, tear = 0;
    // the moment's motion first (the bar, the jiggle, the wobble), exactly as the kernel does
    if (g.tileW > 0) tube_disturb(tm, g.motion, g.tileW, g.tileH, &lx, &ly, &bar, &tear);
    tube_gather(tm, lx, ly, &sx, &sy);
    Vector2D w = {sx, sy};
    if (clamp) w = {std::clamp(w.x, 0.0, 0.999), std::clamp(w.y, 0.0, 0.999)};   // bezel and padding: the nearest point of the window
    return p + box.pos() + Vector2D{w.x * box.w, w.y * box.h} - in;
}

// Why the pointer was or was not mapped, written to pointer.why only when the answer
// changes (a moving mouse asks hundreds of times a second). The record that tells a
// person "it is mapped" from "a layer was in the way" without a debugger.
static void why(const char* reason) {
    if (g.why == reason) return;
    g.why = reason;
    std::ofstream(dir() + "/pointer.why") << reason << "\n";
}

static std::optional<Vector2D> mapped(const Vector2D& p) {
    // Checked at the moment of the move, never cached: the instant the pass stands
    // down (off, its time, a failure) nothing is mapped any more.
    const char* off = g.state != SEngine::READY ? "off: the pass is not running"
                      : !g.boxLive                ? "off: the target is not drawn here (other workspace, cut off, or rotated monitor)"
                      : g.look.curv <= 0          ? "off: the glass is flat"
                      : flag("off")               ? "off: switched off"
                      : flag("input-flat")        ? "off: input-flat"
                      : g_pSessionLockManager->isSessionLocked() ? "off: session locked"
                      : g_pInputManager->isConstrained()         ? "off: pointer constrained by an app"
                                                                 : nullptr;
    if (off) {
        why(off);
        g_pressBox.reset();
        return std::nullopt;
    }
    if (!g_pInputManager->m_currentlyHeldButtons.empty() && !PROTO::data->dndActive()) {
        why(g_pressBox ? "mapped: a held press, through the box it landed in" : "off: a press that began outside the target");
        return g_pressBox ? std::optional{through(*g_pressBox, p, false)} : std::nullopt;
    }
    g_pressBox.reset();
    for (auto const& m : State::monitorState()->monitors()) {
        if (m.get() != g.boxMon || !m->logicalBox().containsPoint(p)) continue;
        Vector2D coords;
        PHLLS    layer;
        // Hyprland's own rule: over a FULLSCREEN window only OVERLAY layers take the
        // pointer (the bar and the dock are behind it). Anything else: OVERLAY, then TOP.
        // Asking about TOP over a fullscreen window found Omarchy's dock along the bottom
        // and switched the mapping off exactly where a video's controls are.
        if (Desktop::viewState()->hitTest().layerSurfaceAt(p, &m->m_layerSurfaceLayers[ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY], &coords, &layer)) {
            why("off: an overlay layer is under the pointer (it is hit where it is drawn)");
            return std::nullopt;
        }
        if (!g.boxFullscreen && Desktop::viewState()->hitTest().layerSurfaceAt(p, &m->m_layerSurfaceLayers[ZWLR_LAYER_SHELL_V1_LAYER_TOP], &coords, &layer)) {
            why("off: a top layer (bar, dock) is under the pointer (it is hit where it is drawn)");
            return std::nullopt;
        }
        if (!g.box.containsPoint(p)) {
            why("off: the pointer is outside the target");
            return std::nullopt;
        }
        why(g.boxFullscreen ? "mapped (fullscreen window)" : "mapped");
        g_pressBox = g.box;
        return through(g.box, p, true);
    }
    why("off: the pointer is on another monitor");
    return std::nullopt;
}

static void hkMove(void* self, uint32_t time, bool refocus, bool mouse, std::optional<Vector2D> override_) {
    if (!override_) g_mapped = mapped(((origPosition)g_positionHook->m_original)(Pointer::mgr().get()));
    ((origMove)g_moveHook->m_original)(self, time, refocus, mouse, override_);
    g_mapped.reset();
}

static CFunctionHook* hookNamed(const std::string& name, const std::string& demangledHas, void* to) {
    for (auto const& f : HyprlandAPI::findFunctionsByName(PHANDLE, name))
        if (f.demangled.contains(demangledHas)) {
            auto* h = HyprlandAPI::createFunctionHook(PHANDLE, f.address, to);
            if (h && h->hook()) return h;
        }
    return nullptr;
}

APICALL EXPORT std::string PLUGIN_API_VERSION() { return HYPRLAND_API_VERSION; }

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;
    if (std::string(__hyprland_api_get_hash()) != std::string(__hyprland_api_get_client_hash()))
        throw std::runtime_error("cuda-pass: built for another Hyprland");
    g_positionHook = hookNamed("position", "CPointerManager::position", (void*)hkPosition);
    g_layerHook    = hookNamed("layerSurfaceAt", "CViewHitTester::layerSurfaceAt", (void*)hkLayerAt);
    g_moveHook     = hookNamed("mouseMoveUnified", "CInputManager::mouseMoveUnified", (void*)hkMove);
    g.hooksOk      = g_positionHook && g_layerHook && g_moveHook;
    note(g.hooksOk ? "pointer: mapped through the glass" : "pointer: could not hook Hyprland's pointer routines; the glass stays flat so clicks stay true");
    g_stage = Event::bus()->m_events.render.stage.listen([](eRenderStage stage) {
        // Ask for another frame only while something is still moving (the phosphor's own
        // flag, or the glass's clock), and only on the monitor the target is on.
        if (stage == RENDER_POST) {
            const float t    = std::chrono::duration<float>(Clock::now() - g.start).count();
            const bool  owed = g.state == SEngine::READY && !flag("off") && (g.active || glass_next_wake(g.look, t) <= 0.02f);
            if (owed || flag("keep-drawing"))
                for (auto& mon : State::monitorState()->monitors())
                    if (flag("keep-drawing") || mon.get() == g.monitor)
                        g_pHyprRenderer->damageMonitor(mon);
        }
        if (!g_pHyprRenderer->m_renderData.pMonitor.lock()) return;
        // While it is up, the pass is added even when `off` is set, once, so it can release what it holds.
        if (stage == RENDER_POST_WINDOWS && (g.state == SEngine::READY || (g.state == SEngine::NEW && !flag("off"))))
            g_pHyprRenderer->m_renderPass.add(makeUnique<CCudaPass>());
        if (stage == RENDER_LAST_MOMENT && flag("dump-now"))
            g_pHyprRenderer->m_renderPass.add(makeUnique<CDump>());
    });
    return {"cuda-pass", "The CUDA CRT inside Hyprland, on one window or one monitor: a proof, loaded by hand", "Parker Brown", "0.0.2"};
}

// Give back everything this plugin holds on the GPU. Hyprland keeps an unloaded
// plugin's code mapped, and a later load of a new build starts with fresh statics:
// without this, every unload leaked two registered textures, a phosphor surface,
// two compiled modules and a reference to CUDA's context, all still live inside
// the compositor. Needs the compositor's GL context current; nothing here throws.
static void teardown() {
    if (!g.cudaUp) return;
    Render::GL::g_pHyprOpenGL->makeEGLCurrent();
    cuCtxSetCurrent(g.cu.ctx);
    cuCtxSynchronize();
    g.path.abandon();          // unmaps anything still mapped
    release();                 // unregisters and deletes the textures, frees the surface
    if (g.activeHost) cuMemFreeHost(g.activeHost);
    g.activeHost = nullptr;
    if (g.ev0) cuEventDestroy(g.ev0);
    if (g.ev1) cuEventDestroy(g.ev1);
    if (g.path.d_active) cuMemFree(g.path.d_active);
    for (CUmodule m : g.cu.mods) cuModuleUnload(m);
    g.cu.mods.clear();
    cuCtxSetCurrent(nullptr);
    cuDevicePrimaryCtxRelease(g.cu.dev);
    g.cudaUp = false;
    note("unloaded: textures unregistered, surface freed, modules unloaded, CUDA context released");
}

APICALL EXPORT void PLUGIN_EXIT() {
    g_stage.reset();
    g_mapped.reset();
    g.state = SEngine::DONE;   // nothing is mapped from here on
    teardown();
    // Hyprland keeps a frame's pass elements until the next frame begins, and would
    // then destroy ours with code that has been unloaded.
    g_pHyprRenderer->m_renderPass.clear();
}
