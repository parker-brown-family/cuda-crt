// CUDA without a toolkit install: NVRTC compiles the .cu at run time, the
// driver API (libcuda.so, which ships with the NVIDIA driver) runs it.
#pragma once
#include <cuda.h>
#include <nvrtc.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::string cu_err(CUresult r) {
    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    return std::string(name ? name : "?") + ": " + (desc ? desc : "?");
}
#define CU(call)                                                                              \
    do {                                                                                      \
        CUresult _r = (call);                                                                 \
        if (_r != CUDA_SUCCESS)                                                               \
            throw std::runtime_error(std::string(#call) + " -> " + cu_err(_r));               \
    } while (0)

inline std::string slurp_or_empty(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct Cuda {
    CUdevice dev{};
    CUcontext ctx{};
    std::vector<CUmodule> mods;
    int compiled = 0;  // how many kernel files had to be compiled this run (0 = all came from the cache)
    std::string device_name, nvrtc_version, driver_version;
    int sm_major = 0, sm_minor = 0;

    // gl_device: a CUDA device handle that owns the current OpenGL context (see
    // cuda_device_for_gl in pipelines.hpp), or -1 to take device 0.
    void init(int gl_device = -1) {
        CU(cuInit(0));
        if (gl_device >= 0) dev = (CUdevice)gl_device;
        else CU(cuDeviceGet(&dev, 0));
        char name[128];
        CU(cuDeviceGetName(name, sizeof name, dev));
        device_name = name;
        CU(cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
        CU(cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
        int v = 0;
        CU(cuDriverGetVersion(&v));
        driver_version = std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
        CU(cuDevicePrimaryCtxRetain(&ctx, dev));
        CU(cuCtxSetCurrent(ctx));
    }

    // Compile `path` (with `incdir` on the include path) and load it. fast = fast
    // math, for kernels whose output is a picture; without it there are no fused
    // multiply-adds, so a kernel rounds exactly as its CPU reference does.
    void load(const std::string &path, const std::string &incdir, bool fast = false) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string src = ss.str();
        int maj = 0, min = 0;
        nvrtcVersion(&maj, &min);
        nvrtc_version = std::to_string(maj) + "." + std::to_string(min);
        std::string arch = "--gpu-architecture=compute_" + std::to_string(sm_major) + std::to_string(sm_minor);
        std::string inc = "-I" + incdir;
        const char *mode = fast ? "--use_fast_math" : "--fmad=false";
        // Compiled kernels are kept beside the lab (out/ptx/), named by a hash of every
        // input that could change them, so a later start loads in milliseconds. A
        // compositor plugin relies on this: it must never stall a frame on a compile.
        std::string key = src + "\n" + slurp_or_empty(incdir + "/phosphor_params.h") + slurp_or_empty(incdir + "/phosphor_half.h") + arch + mode + nvrtc_version;
        unsigned long long h = 1469598103934665603ull;
        for (unsigned char c : key) { h ^= c; h *= 1099511628211ull; }
        char name[64];
        snprintf(name, sizeof name, "/out/ptx/%016llx.ptx", h);
        std::string cache = incdir + name, ptx = slurp_or_empty(cache);
        if (ptx.empty()) {
            nvrtcProgram prog;
            if (nvrtcCreateProgram(&prog, src.c_str(), path.c_str(), 0, nullptr, nullptr) != NVRTC_SUCCESS)
                throw std::runtime_error("nvrtcCreateProgram failed");
            const char *opts[] = {arch.c_str(), inc.c_str(), "--std=c++17", mode};
            nvrtcResult r = nvrtcCompileProgram(prog, 4, opts);
            size_t logn = 0;
            nvrtcGetProgramLogSize(prog, &logn);
            std::string log(logn, '\0');
            nvrtcGetProgramLog(prog, log.data());
            if (r != NVRTC_SUCCESS) throw std::runtime_error("NVRTC compile failed:\n" + log);
            if (logn > 2) fprintf(stderr, "nvrtc: %s\n", log.c_str());
            size_t n = 0;
            nvrtcGetPTXSize(prog, &n);
            ptx.assign(n, '\0');
            nvrtcGetPTX(prog, ptx.data());
            nvrtcDestroyProgram(&prog);
            compiled++;
            // best effort: on a read-only tree this fails quietly and the next run compiles again
            mkdir((incdir + "/out").c_str(), 0755);
            mkdir((incdir + "/out/ptx").c_str(), 0755);
            std::ofstream o(cache + ".tmp", std::ios::binary);
            if (o && o.write(ptx.data(), ptx.size()) && (o.close(), true)) rename((cache + ".tmp").c_str(), cache.c_str());
        }
        CUmodule mod;
        CU(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr));
        mods.push_back(mod);
    }
    // Both kernel files: the state (exact) and the glass (fast math).
    void load_all(const std::string &root) {
        load(root + "/kernels/phosphor.cu", root);
        load(root + "/kernels/glass.cu", root, true);
    }

    CUfunction fn(const char *name) {
        CUfunction f;
        for (CUmodule m : mods)
            if (cuModuleGetFunction(&f, m, name) == CUDA_SUCCESS) return f;
        throw std::runtime_error(std::string("no kernel named ") + name);
    }

    size_t free_bytes() {
        size_t fr = 0, tot = 0;
        CU(cuMemGetInfo(&fr, &tot));
        return fr;
    }
};

// One launch over a w*h tile, bx*by threads per block.
inline void launch2d(CUfunction f, int w, int h, void **args, int bx = 16, int by = 16) {
    CU(cuLaunchKernel(f, (w + bx - 1) / bx, (h + by - 1) / by, 1, bx, by, 1, 0, 0, args, nullptr));
}
