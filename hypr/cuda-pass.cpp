// cuda-pass: the CUDA CRT running INSIDE Hyprland, as a tracer bullet.
//
// A plugin that, after the windows are drawn and before the top and overlay
// layers, copies the finished picture into a texture, hands it to the same CUDA
// kernels the lab measures (phosphor update, halation, glass), and copies the
// result back. One phosphor surface for the whole monitor: enough to answer the
// questions that could kill the idea (can CUDA share this compositor's
// textures? what does a frame cost in here? does a failure leave the session
// standing?) and nothing more. Per-window surfaces come after this passes.
//
// It is meant for the sealed test seat only (hypr/prove). It follows the shape
// of omarchy-crt's native/crt-pass.cpp, which is the pass people actually run.
//
// Switches are files in $HOME, read every frame, so a test can flip them:
//   cuda-off       stand the pass down (the frame goes through untouched)
//   cuda-fail      throw in the middle of a frame, once, to prove the fallback
//   still-glass    no tracking bar, jiggle, flicker or degauss (for the idle proof)
//   keep-drawing   damage every frame (a capture needs a frame)
//   time-pass      finish the GPU either side of the pass and log its wall time
//   dump-now       write the finished frame to frame.rgba / frame.txt
#include <GLES3/gl32.h>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include "../src/cuda_path.hpp"
#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/Compositor.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/state/MonitorState.hpp>

static HANDLE              PHANDLE = nullptr;
static CHyprSignalListener g_stage;
using Clock = std::chrono::steady_clock;

static std::string home() { const char* h = getenv("HOME"); return h ? h : "/tmp"; }
static bool        flag(const char* name) { return access((home() + "/" + name).c_str(), F_OK) == 0; }
static void        note(const std::string& line) { std::ofstream(home() + "/cuda-pass.log", std::ios::app) << line << "\n"; }

struct SEngine {
    enum { NEW, READY, DEAD } state = NEW;
    Cuda     cu;
    CudaPath path;
    CudaSurf surf;
    Shared   shIn, shOut;
    GLuint   inTex = 0, inFb = 0, outTex = 0, outFb = 0;
    int      w = 0, h = 0;
    Look     look;
    Clock::time_point start, last;
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

// A new size means new textures, and a texture must be unregistered from CUDA
// before it is deleted. The phosphor starts fresh: a resize resets.
static void resize(int w, int h) {
    g.shIn.unreg(); g.shOut.unreg();
    g.surf.destroy();
    if (g.inTex) { glDeleteFramebuffers(1, &g.inFb); glDeleteTextures(1, &g.inTex); glDeleteFramebuffers(1, &g.outFb); glDeleteTextures(1, &g.outTex); }
    target(g.inTex, g.inFb, w, h);
    target(g.outTex, g.outFb, w, h);
    g.shIn.reg(g.inTex, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
    g.shOut.reg(g.outTex, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
    g.surf.create(w, h, true);
    g.w = w; g.h = h;
    note("surface " + std::to_string(w) + "x" + std::to_string(h) + ": two GLES textures registered with CUDA, half-precision state allocated");
}

static void init() {
    std::string root;
    std::getline(std::ifstream(home() + "/cuda-crt-root"), root);
    if (root.empty()) throw std::runtime_error("no cuda-crt-root file beside the plugin's HOME");
    const int dev = cuda_device_for_gl();
    note("CUDA device that owns this GL context: " + (dev < 0 ? std::string("could not tell (using device 0)") : std::to_string(dev)));
    g.cu.init(dev);
    g.cu.load_all(root);
    g.path.init(g.cu);
    g.look.track_rest = 2.0f; g.look.track_sweep = 5.0f;
    g.start = g.last = Clock::now();
    note("ready: " + g.cu.device_name + ", CUDA driver API " + g.cu.driver_version + ", NVRTC " + g.cu.nvrtc_version + ", GL " + (const char*)glGetString(GL_VERSION));
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
        if (g.state == SEngine::DEAD) return {};
        GLint fb = 0, vp[4] = {}, active = 0, tex = 0;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fb);
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
        const GLboolean blend = glIsEnabled(GL_BLEND), scissor = glIsEnabled(GL_SCISSOR_TEST);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex);
        const int w = vp[2], h = vp[3];
        try {
            if (g.state == SEngine::NEW) { init(); g.state = SEngine::READY; }
            CU(cuCtxSetCurrent(g.cu.ctx));
            if (w != g.w || h != g.h) resize(w, h);

            const bool still = flag("still-glass");
            g.look.tracking = still ? 0.0f : 0.6f; g.look.jiggle = still ? 0.0f : 0.15f; g.look.flicker = still ? 0.0f : 0.35f;
            g.look.degauss_at = still ? -1.0f : 1.0f;

            // never read the attachment being drawn to: the lower scene is copied first
            glDisable(GL_SCISSOR_TEST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g.inFb);
            glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

            const bool timed = flag("time-pass");
            if (timed) glFinish();
            const auto  t0  = Clock::now();
            const float t   = std::chrono::duration<float>(t0 - g.start).count();
            const double dt = std::min(0.25, std::max(0.002, std::chrono::duration<double>(t0 - g.last).count()));
            g.last = t0;

            g.path.begin(g.shIn, &g.shOut);
            if (flag("cuda-fail") && !g.failedOnce) { g.failedOnce = true; throw std::runtime_error("injected failure, mid-frame, textures mapped"); }
            g.path.update_v2(g.surf, 0, 0, make_step_v2(g.look.phosphor, g.look.burn, dt, g.surf.fresh));
            g.path.glow(g.surf, g.look.glow_knee);
            g.path.optics(g.surf, 0, 0, make_optics(g.look, t, h, true));
            g.active = g.path.active_result();
            g.path.end();

            glBindFramebuffer(GL_READ_FRAMEBUFFER, g.outFb);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb);
            glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            if (timed) {
                glFinish();
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                g.msSum += ms; g.msMax = std::max(g.msMax, ms); g.timed++;
                std::ofstream(home() + "/cuda-pass-ms.txt") << g.msSum / g.timed << " " << g.msMax << " " << g.timed << " " << w << "x" << h << "\n";
            }
            g.frames++;
        } catch (std::exception& e) {
            // Let go of anything mapped, stand down for good, and leave this frame as it came.
            g.path.abandon();
            g.state = SEngine::DEAD;
            note(std::string("DEAD, passing frames through untouched: ") + e.what());
        }
        std::ofstream(home() + "/cuda-frames.now") << g.frames << " " << (g.active ? 1 : 0) << "\n";
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
        if (!flag("dump-now")) return {};
        GLint vp[4] = {}, fb = 0;
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fb);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
        std::vector<unsigned char> px((size_t)vp[2] * vp[3] * 4);
        glReadPixels(0, 0, vp[2], vp[3], GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        std::ofstream(home() + "/frame.rgba", std::ios::binary).write((char*)px.data(), px.size());
        std::ofstream(home() + "/frame.txt") << vp[2] << " " << vp[3] << " frames=" << g.frames << " state=" << (int)g.state << "\n";
        unlink((home() + "/dump-now").c_str());
        return {};
    }
};

APICALL EXPORT std::string PLUGIN_API_VERSION() { return HYPRLAND_API_VERSION; }

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;
    if (std::string(__hyprland_api_get_hash()) != std::string(__hyprland_api_get_client_hash()))
        throw std::runtime_error("cuda-pass: built for another Hyprland");
    g_stage = Event::bus()->m_events.render.stage.listen([](eRenderStage stage) {
        // Ask for another frame only while something is still moving: the phosphor's
        // own flag, or the glass's clock. (Whole-monitor damage: fine for a tracer.)
        if (stage == RENDER_POST) {
            const float t    = std::chrono::duration<float>(Clock::now() - g.start).count();
            const bool  owed = g.state == SEngine::READY && !flag("cuda-off") && (g.active || glass_next_wake(g.look, t) <= 0.02f);
            if (owed || flag("keep-drawing"))
                for (auto& mon : State::monitorState()->monitors())
                    g_pHyprRenderer->damageMonitor(mon);
        }
        if (!g_pHyprRenderer->m_renderData.pMonitor.lock()) return;
        if (stage == RENDER_POST_WINDOWS && g.state != SEngine::DEAD && !flag("cuda-off"))
            g_pHyprRenderer->m_renderPass.add(makeUnique<CCudaPass>());
        if (stage == RENDER_LAST_MOMENT)
            g_pHyprRenderer->m_renderPass.add(makeUnique<CDump>());
    });
    return {"cuda-pass", "The CUDA CRT inside Hyprland: a tracer bullet for the sealed test seat", "Parker Brown", "0.0.1"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    g_stage.reset();
    // Hyprland keeps a frame's pass elements until the next frame begins, and would
    // then destroy ours with code that has been unloaded.
    g_pHyprRenderer->m_renderPass.clear();
}
