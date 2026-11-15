// A headless OpenGL context on the NVIDIA GPU, through EGL's device platform.
// Nothing here talks to Wayland, Hyprland or a window: it cannot touch the desk.
#pragma once
#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct GLCtx {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    std::string vendor, renderer, version, egl_device;
    bool gles = false;

    // gles=false: desktop OpenGL 4.5 core. gles=true: OpenGL ES 3.2, which is
    // what Hyprland itself runs.
    void init(bool want_gles = false) {
        gles = want_gles;
        EGLDeviceEXT devs[16];
        EGLint n = 0;
        if (!eglQueryDevicesEXT(16, devs, &n) || n == 0) throw std::runtime_error("no EGL devices");
        EGLDeviceEXT pick = devs[0];
        for (int i = 0; i < n; i++) {
            const char *ext = eglQueryDeviceStringEXT(devs[i], EGL_EXTENSIONS);
            if (ext && strstr(ext, "EGL_NV_device_cuda")) { pick = devs[i]; egl_device = "EGL_NV_device_cuda"; break; }
        }
        dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, pick, nullptr);
        EGLint maj, min;
        if (!eglInitialize(dpy, &maj, &min)) throw std::runtime_error("eglInitialize failed");
        eglBindAPI(gles ? EGL_OPENGL_ES_API : EGL_OPENGL_API);
        EGLint cfg_attr[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
                             gles ? EGL_OPENGL_ES3_BIT : EGL_OPENGL_BIT, EGL_NONE};
        EGLConfig cfg;
        EGLint nc = 0;
        if (!eglChooseConfig(dpy, cfg_attr, &cfg, 1, &nc) || nc == 0) throw std::runtime_error("no EGL config");
        EGLint ctx_gl[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 5,
                           EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
        EGLint ctx_es[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE};
        ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, gles ? ctx_es : ctx_gl);
        if (ctx == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext failed");
        if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) throw std::runtime_error("eglMakeCurrent failed");
        vendor = (const char *)glGetString(GL_VENDOR);
        renderer = (const char *)glGetString(GL_RENDERER);
        version = (const char *)glGetString(GL_VERSION);
    }
    void destroy() {
        if (dpy != EGL_NO_DISPLAY) {
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (ctx != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctx);
            eglTerminate(dpy);
        }
        dpy = EGL_NO_DISPLAY; ctx = EGL_NO_CONTEXT;
    }
};

inline std::string slurp(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline GLuint gl_shader(GLenum type, const std::string &src) {
    GLuint s = glCreateShader(type);
    const char *p = src.c_str();
    glShaderSource(s, 1, &p, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        throw std::runtime_error(std::string("shader compile failed:\n") + log);
    }
    return s;
}

// One oversized triangle covers the viewport; texel coordinates come from gl_FragCoord.
inline GLuint gl_program(const std::string &frag_src) {
    static const char *vs = "#version 450 core\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
                            "gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n";
    GLuint p = glCreateProgram();
    GLuint v = gl_shader(GL_VERTEX_SHADER, vs), f = gl_shader(GL_FRAGMENT_SHADER, frag_src);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        throw std::runtime_error(std::string("program link failed:\n") + log);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

inline GLuint gl_tex(GLenum internal, int w, int h, GLenum filter) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, internal, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

inline GLuint gl_fbo(GLuint tex) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("framebuffer incomplete");
    return f;
}
