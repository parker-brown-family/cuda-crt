// Stage 2: the same phosphor surface three ways, timed on the GPU with no CPU
// round trip inside the timed path.
//   existing   crt/crt-glass.frag as shipped: one pass, no memory (the baseline)
//   glsl       phosphor update + glass in fragment shaders
//   cuda       phosphor update + glass in CUDA kernels on textures shared with OpenGL
// Two clocks per frame, kept apart on purpose:
//   gpu_ms    time the GPU spent in the pass itself (GL timer query / CUDA events)
//   total_ms  wall clock from an idle GPU to a finished, "presented" frame,
//             including every synchronisation the path forces
#include "pipelines.hpp"
#include "scene.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
static double thread_cpu_ms() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(p * v.size()))];
}

struct Stats { std::vector<double> gpu, total, cpu; };

// The GPU's graphics clock right now, MHz, or -1 if it cannot be read. A laptop
// GPU changes gear with load, and a pass takes several times longer in low gear.
static int gpu_clock_mhz() {
    FILE *p = popen("nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return -1;
    int mhz = -1;
    if (fscanf(p, "%d", &mhz) != 1) mhz = -1;
    pclose(p);
    return mhz;
}

struct Bench {
    std::string root;
    Cuda cu;
    GLCtx gl;
    Glsl glsl;
    CudaPath cuda;
    Look look;
    int W = 0, H = 0;
    static const int K = 4;
    GLuint frames[K]{}, out_tex[3]{}, out_fbo[3]{}, screen_tex = 0, screen_fbo = 0, q_time = 0;
    Shared sh_frames[K], sh_out;
    Existing existing;
    GlslSurf gs, gs16;
    CudaSurf cs;
    CUevent ev0{}, ev1{};

    void setup(int w, int h) {
        W = w; H = h;
        std::vector<uint8_t> px;
        for (int i = 0; i < K; i++) {
            draw_terminal(px, W, H, i, i % 2 == 0);
            frames[i] = gl_tex(GL_RGBA8, W, H, GL_LINEAR);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            sh_frames[i].reg(frames[i], CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
        }
        for (int i = 0; i < 3; i++) { out_tex[i] = gl_tex(GL_RGBA8, W, H, GL_NEAREST); out_fbo[i] = gl_fbo(out_tex[i]); }
        sh_out.reg(out_tex[2], CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
        screen_tex = gl_tex(GL_RGBA8, W, H, GL_NEAREST);
        screen_fbo = gl_fbo(screen_tex);
        existing.init(root + "/baseline/crt-glass.frag", look, 0, 0, W, H);
        gs.create(W, H);
        gs16.create(W, H, GL_RGBA16F);
        cs.create(W, H);
        glFinish();
    }
    void teardown() {
        for (int i = 0; i < K; i++) { sh_frames[i].unreg(); glDeleteTextures(1, &frames[i]); }
        sh_out.unreg();
        glDeleteFramebuffers(3, out_fbo); glDeleteTextures(3, out_tex);
        glDeleteFramebuffers(1, &screen_fbo); glDeleteTextures(1, &screen_tex);
        gs.destroy(); gs16.destroy(); cs.destroy();
        glDeleteProgram(existing.prog);
        glFinish();
    }
    // The stand-in for the compositor using the result: copy it to a "screen" target.
    void present(int which) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, out_fbo[which]);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, screen_fbo);
        glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    // One frame of `path`. Returns gpu ms (pass only); total and cpu are measured by the caller.
    double frame(const std::string &path, int i, double dt, bool *active) {
        float t = (float)(i * dt);
        GLuint ftex = frames[i % K];
        double gpu_ms = 0;
        if (path == "existing") {
            glBeginQuery(GL_TIME_ELAPSED, q_time);
            existing.draw(ftex, out_fbo[0], W, H, t);
            glEndQuery(GL_TIME_ELAPSED);
            present(0);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q_time, GL_QUERY_RESULT, &ns);
            gpu_ms = ns / 1e6;
        } else if (path.rfind("glsl", 0) == 0) {
            GlslSurf &g = path == "glsl_full_16f" ? gs16 : gs;
            PhosphorStep p = make_step(look.phosphor, dt, g.fresh);
            bool full = path != "glsl_update", probe = path != "glsl_full_noprobe";
            glBeginQuery(GL_TIME_ELAPSED, q_time);
            glsl.update(g, ftex, 0, 0, p);
            if (probe) glsl.probe(g);
            if (full) glsl.optics(g, out_fbo[1], 0, 0, make_optics(look, t, H));
            glEndQuery(GL_TIME_ELAPSED);
            present(1);
            if (probe) *active = glsl.active_result();
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q_time, GL_QUERY_RESULT, &ns);
            gpu_ms = ns / 1e6;
        } else {  // cuda_update | cuda_full
            PhosphorStep p = make_step(look.phosphor, dt, cs.fresh);
            bool full = path != "cuda_update";
            cuda.begin(sh_frames[i % K], full ? &sh_out : nullptr);
            CU(cuEventRecord(ev0, 0));
            cuda.update(cs, 0, 0, p);
            if (full) cuda.optics(cs, 0, 0, make_optics(look, t, H));
            CU(cuEventRecord(ev1, 0));
            *active = cuda.active_result();
            cuda.end();
            present(2);
            glFinish();
            float ms = 0;
            CU(cuEventElapsedTime(&ms, ev0, ev1));
            gpu_ms = ms;
        }
        return gpu_ms;
    }

    Stats run(const std::string &path, int n, int hz /*0 = back to back*/, double sim_hz) {
        Stats s;
        gs.fresh = true; gs16.fresh = true; cs.fresh = true;
        const int warm = 40;
        const double dt = 1.0 / sim_hz;
        auto next = Clock::now();
        for (int i = 0; i < n + warm; i++) {
            if (hz) {
                next += std::chrono::nanoseconds((long long)(1e9 / hz));
                std::this_thread::sleep_until(next);
            }
            glFinish();
            bool active = false;
            double c0 = thread_cpu_ms();
            auto t0 = Clock::now();
            double g = frame(path, i, dt, &active);
            double tot = ms_since(t0), cpu = thread_cpu_ms() - c0;
            if (i >= warm) { s.gpu.push_back(g); s.total.push_back(tot); s.cpu.push_back(cpu); }
        }
        return s;
    }

    // Where a CUDA frame's wall time goes, with a full stop after each part.
    void breakdown(FILE *f, int n) {
        std::vector<double> m, k, u;
        cs.fresh = true;
        for (int i = 0; i < n + 20; i++) {
            glFinish();
            auto t0 = Clock::now();
            cuda.begin(sh_frames[i % K], &sh_out);
            CU(cuCtxSynchronize());
            double a = ms_since(t0);
            t0 = Clock::now();
            cuda.update(cs, 0, 0, make_step(look.phosphor, 1.0 / 60, cs.fresh));
            cuda.optics(cs, 0, 0, make_optics(look, i / 60.0f, H));
            CU(cuCtxSynchronize());
            double b = ms_since(t0);
            t0 = Clock::now();
            cuda.end();
            present(2);
            glFinish();
            double c = ms_since(t0);
            if (i >= 20) { m.push_back(a); k.push_back(b); u.push_back(c); }
        }
        fprintf(f, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", W, H, pct(m, .5), pct(m, .95), pct(k, .5), pct(k, .95), pct(u, .5), pct(u, .95));
    }

    // Do the GLSL and CUDA paths draw the same picture? 8-bit output, frame by frame.
    void compare(FILE *f, bool scanlines) {
        Look keep = look;
        if (!scanlines) look.scan = 0.0f;
        gs.destroy(); gs.create(W, H);
        cs.destroy(); cs.create(W, H);
        std::vector<uint8_t> a((size_t)W * H * 4), b(a.size());
        int worst = 0;
        double mean = 0;
        size_t over1 = 0, count = 0;
        for (int i = 0; i < 24; i++) {
            bool act;
            frame("glsl_full", i, 1.0 / 60, &act);
            frame("cuda_full", i, 1.0 / 60, &act);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, out_fbo[1]);
            glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, a.data());
            glBindFramebuffer(GL_READ_FRAMEBUFFER, out_fbo[2]);
            glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
            for (size_t p = 0; p < a.size(); p++) {
                if ((p & 3) == 3) continue;
                int d = abs((int)a[p] - (int)b[p]);
                worst = std::max(worst, d); mean += d; over1 += d > 1; count++;
            }
        }
        fprintf(f, "%d,%d,%s,24,%d,%.5f,%.6f\n", W, H, scanlines ? "full_look" : "scanlines_off", worst, mean / count, 100.0 * over1 / count);
        look = keep;
    }
};

int main(int argc, char **argv) {
    Bench b;
    b.root = argc > 1 ? argv[1] : ".";
    bool quick = argc > 2 && std::string(argv[2]) == "quick";
    try {
        b.gl.init(false);
        b.cu.init();
        b.cu.load(b.root + "/kernels/phosphor.cu", b.root);
        b.glsl.init(b.root);
        b.cuda.init(b.cu);
        glGenQueries(1, &b.q_time);
        CU(cuEventCreate(&b.ev0, CU_EVENT_DEFAULT));
        CU(cuEventCreate(&b.ev1, CU_EVENT_DEFAULT));
        fprintf(stderr, "GL: %s | %s\nCUDA: %s, driver CUDA %s, NVRTC %s\n", b.gl.renderer.c_str(), b.gl.version.c_str(),
                b.cu.device_name.c_str(), b.cu.driver_version.c_str(), b.cu.nvrtc_version.c_str());

        if (argc > 3 && std::string(argv[2]) == "power") {  // hold one path at 60 Hz, 1080p, while something samples the GPU
            b.setup(1920, 1080);
            Stats st = b.run(argv[3], atoi(argc > 4 ? argv[4] : "480"), 60, 60);
            fprintf(stderr, "power run %s: gpu p50 %.3f ms\n", argv[3], pct(st.gpu, .5));
            // one line per second of the run: is the pass time steady, or does the GPU change gear?
            for (size_t i = 0; i + 60 <= st.gpu.size(); i += 60)
                printf("%s,%zu,%.3f,%.3f\n", argv[3], i / 60, pct(std::vector<double>(st.gpu.begin() + i, st.gpu.begin() + i + 60), .5),
                       pct(std::vector<double>(st.total.begin() + i, st.total.begin() + i + 60), .5));
            b.teardown();
            return 0;
        }
        // 1. which block shape suits these kernels (1080p, back to back)
        FILE *fs = fopen((b.root + "/out/block_sweep.csv").c_str(), "w");
        fprintf(fs, "block_x,block_y,kernels_gpu_ms_p50,kernels_gpu_ms_p95\n");
        b.setup(1920, 1080);
        int shapes[][2] = {{8, 8}, {16, 8}, {16, 16}, {32, 8}, {32, 16}, {32, 32}, {64, 4}};
        double best = 1e9;
        int best_x = 16, best_y = 16;
        for (auto &s : shapes) {
            b.cuda.bx = s[0]; b.cuda.by = s[1];
            Stats st = b.run("cuda_full", quick ? 60 : 200, 0, 60);
            double p50 = pct(st.gpu, .5);
            fprintf(fs, "%d,%d,%.4f,%.4f\n", s[0], s[1], p50, pct(st.gpu, .95));
            if (p50 < best) { best = p50; best_x = s[0]; best_y = s[1]; }
        }
        fclose(fs);
        b.cuda.bx = best_x; b.cuda.by = best_y;
        b.teardown();
        fprintf(stderr, "block shape: %dx%d\n", best_x, best_y);

        // 2. the matrix
        FILE *f = fopen((b.root + "/results.csv").c_str(), "w");
        FILE *fb = fopen((b.root + "/out/cuda_breakdown.csv").c_str(), "w");
        FILE *fc = fopen((b.root + "/out/glsl_vs_cuda_image.csv").c_str(), "w");
        fprintf(f, "path,width,height,pacing,frames,gpu_ms_p50,gpu_ms_p95,gpu_ms_p99,total_ms_p50,total_ms_p95,total_ms_p99,"
                   "cpu_ms_p50,state_vram_mb,gpu_mhz_at_end,gpu_timer,total_timer\n");
        fprintf(fb, "width,height,map_ms_p50,map_ms_p95,kernels_synced_ms_p50,kernels_synced_ms_p95,unmap_present_ms_p50,unmap_present_ms_p95\n");
        fprintf(fc, "width,height,look,frames,max_abs_diff_8bit,mean_abs_diff_8bit,percent_values_differing_by_more_than_1\n");
        const char *paths[] = {"existing", "glsl_update", "glsl_full_noprobe", "glsl_full", "glsl_full_16f", "cuda_update", "cuda_full"};
        int sizes[][2] = {{960, 540}, {1920, 1080}, {3840, 2160}};
        for (auto &sz : sizes) {
            b.setup(sz[0], sz[1]);
            for (int hz : {0, 60, 144}) {
                for (const char *p : paths) {
                    std::string path = p;
                    if (hz == 144 && (path == "glsl_update" || path == "cuda_update" || path == "glsl_full_noprobe" || path == "glsl_full_16f")) continue;
                    int n = quick ? 60 : (hz == 0 ? 600 : hz == 60 ? 300 : 432);
                    Stats s = b.run(path, n, hz, hz ? hz : 60);
                    int mhz = gpu_clock_mhz();
                    double vram = path == "existing" ? 0 : path == "glsl_full_16f" ? b.gs16.bytes() / 1048576.0 : path[0] == 'g' ? b.gs.bytes() / 1048576.0 : b.cs.bytes() / 1048576.0;
                    fprintf(f, "%s,%d,%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%s,%s,%s\n", p, sz[0], sz[1],
                            hz == 0 ? "back_to_back" : hz == 60 ? "paced_60hz" : "paced_144hz", n, pct(s.gpu, .5), pct(s.gpu, .95),
                            pct(s.gpu, .99), pct(s.total, .5), pct(s.total, .95), pct(s.total, .99), pct(s.cpu, .5), vram,
                            mhz < 0 ? "unknown" : std::to_string(mhz).c_str(),
                            path[0] == 'c' ? "cuda_events_kernels_only" : "gl_time_elapsed_query", "wall_idle_gpu_to_glFinish_after_present");
                    fflush(f);
                    fprintf(stderr, "%-18s %4dx%-4d %-12s gpu p50 %.3f p95 %.3f | total p50 %.3f p95 %.3f\n", p, sz[0], sz[1],
                            hz == 0 ? "back_to_back" : hz == 60 ? "paced_60hz" : "paced_144hz", pct(s.gpu, .5), pct(s.gpu, .95),
                            pct(s.total, .5), pct(s.total, .95));
                }
            }
            b.breakdown(fb, quick ? 60 : 300);
            b.compare(fc, true);
            b.compare(fc, false);
            b.teardown();
        }
        fclose(f); fclose(fb); fclose(fc);
    } catch (std::exception &e) {
        fprintf(stderr, "stage2: %s\n", e.what());
        return 2;
    }
    return 0;
}
