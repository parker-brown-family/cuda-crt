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
//   off            stand the pass down and release its textures
//   ttl            a number of seconds: stand down by itself after that long
//   fail           throw in the middle of a frame, once, to prove the fallback
//   keep-drawing   damage every frame (a capture needs a frame)
//   time-pass      finish the GPU either side of the pass and log its wall time
//   dump-now       write the finished frame to frame.rgba / frame.txt
// It writes: pass.log, pass-ms.txt, frames.now, frame.rgba, frame.txt.
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
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/helpers/math/Math.hpp>
#include <hyprland/src/state/MonitorState.hpp>

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
        g.cudaUp = true;
    }
    g.start = g.last = g.liveSince = Clock::now();
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
    if (spec.empty()) { r = {0, 0, fbw, fbh}; owner = m.get(); return true; }
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
        return true;
    }
    return false;
}

static void dress(bool phosphorOnly) {
    Look k;  // the lab's defaults
    k.track_rest = 2.0f; k.track_sweep = 5.0f; k.degauss_at = 1.0f;
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
                if (where(m, w, h, r, owner)) {
                    if (g.state == SEngine::NEW) { init(); g.state = SEngine::READY; }
                    CU(cuCtxSetCurrent(g.cu.ctx));
                    if (w != g.w || h != g.h) resize(w, h);
                    if (owner != g.owner || r.w != g.surf.w || r.h != g.surf.h) {  // a new owner or a new size starts fresh
                        g.surf.destroy();
                        g.surf.create(r.w, r.h, true);
                        g.owner = owner;
                        note("phosphor " + std::to_string(r.w) + "x" + std::to_string(r.h) + " at " + std::to_string(r.x) + "," + std::to_string(r.y) + " on a " + std::to_string(w) + "x" + std::to_string(h) + " framebuffer");
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
                    const float  t  = std::chrono::duration<float>(t0 - g.start).count();
                    const double dt = std::min(0.25, std::max(0.002, std::chrono::duration<double>(t0 - g.last).count()));
                    g.last = t0;

                    g.path.begin(g.shIn, &g.shOut);
                    if (flag("fail") && !g.failedOnce) { g.failedOnce = true; throw std::runtime_error("injected failure, mid-frame, textures mapped"); }
                    g.path.update_v2(g.surf, r.x, r.y, make_step_v2(g.look.phosphor, g.look.burn, dt, g.surf.fresh));
                    g.path.glow(g.surf, g.look.glow_knee);
                    g.path.optics(g.surf, r.x, r.y, make_optics(g.look, t, r.h, true));
                    g.active = g.path.active_result();
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

APICALL EXPORT std::string PLUGIN_API_VERSION() { return HYPRLAND_API_VERSION; }

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;
    if (std::string(__hyprland_api_get_hash()) != std::string(__hyprland_api_get_client_hash()))
        throw std::runtime_error("cuda-pass: built for another Hyprland");
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

APICALL EXPORT void PLUGIN_EXIT() {
    g_stage.reset();
    // Hyprland keeps a frame's pass elements until the next frame begins, and would
    // then destroy ours with code that has been unloaded.
    g_pHyprRenderer->m_renderPass.clear();
}
