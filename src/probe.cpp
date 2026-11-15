// Stage 0 probe: can this box do what the experiment needs? Prints one fact
// per line. Answers the handoff's question of whether the OpenGL context and
// the CUDA device can share a texture, for desktop GL and for the GLES 3.2
// that Hyprland runs.
#include "glrt.hpp"
#include "cudart.hpp"
#include <cudaGL.h>

static const char *try_register(GLenum internal, unsigned flags) {
    static std::string msg;
    GLuint t = gl_tex(internal, 64, 64, GL_NEAREST);
    CUgraphicsResource res;
    CUresult r = cuGraphicsGLRegisterImage(&res, t, GL_TEXTURE_2D, flags);
    if (r == CUDA_SUCCESS) {
        CUresult m = cuGraphicsMapResources(1, &res, 0);
        if (m == CUDA_SUCCESS) cuGraphicsUnmapResources(1, &res, 0);
        cuGraphicsUnregisterResource(res);
        msg = m == CUDA_SUCCESS ? "yes (registered, mapped, unmapped)" : "registered, map failed: " + cu_err(m);
    } else msg = "no: " + cu_err(r);
    glDeleteTextures(1, &t);
    return msg.c_str();
}

int main(int argc, char **argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    try {
        Cuda cu;
        cu.init();
        printf("cuda_device: %s\n", cu.device_name.c_str());
        printf("cuda_compute_capability: %d.%d\n", cu.sm_major, cu.sm_minor);
        printf("cuda_driver_api: %s\n", cu.driver_version.c_str());
        cu.load(root + "/kernels/phosphor.cu", root);
        printf("nvrtc: %s\n", cu.nvrtc_version.c_str());
        printf("kernel_compile: ok (phosphor_update_buf, phosphor_update_tex, crt_optics_tex, passthrough_tex)\n");
        cu.fn("phosphor_update_buf"); cu.fn("phosphor_update_tex"); cu.fn("crt_optics_tex"); cu.fn("passthrough_tex");
        for (int es = 0; es < 2; es++) {
            GLCtx gl;
            const char *tag = es ? "gles32" : "gl45";
            try {
                gl.init(es);
                printf("%s_context: %s | %s | %s\n", tag, gl.vendor.c_str(), gl.renderer.c_str(), gl.version.c_str());
                printf("%s_interop_rgba8_read: %s\n", tag, try_register(GL_RGBA8, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY));
                printf("%s_interop_rgba8_surface_write: %s\n", tag, try_register(GL_RGBA8, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
                printf("%s_interop_rgba32f: %s\n", tag, try_register(GL_RGBA32F, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
                printf("%s_interop_rgba16f: %s\n", tag, try_register(GL_RGBA16F, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
                printf("%s_interop_srgb8_alpha8: %s\n", tag, try_register(GL_SRGB8_ALPHA8, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY));
            } catch (std::exception &e) {
                printf("%s_context: FAILED %s\n", tag, e.what());
            }
            gl.destroy();
        }
    } catch (std::exception &e) {
        printf("probe_failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
