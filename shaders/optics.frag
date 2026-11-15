#version 450 core
// The matched GLSL reference for kernels/phosphor.cu : crt_optics_tex.
// The glass: the tube's curve and rim, chromatic split, glow, scanlines, centre
// wash, tracking band, glare, specular, vignette. Dials as in crt/crt-glass.frag;
// the rim is terminal-delight's (gpui_wgpu crt_pass.wgsl, td-glass-rim).
uniform sampler2D u_state;  // RGBA32F linear light, bilinear, clamped
uniform ivec2 u_origin;
uniform vec2  u_size;
uniform float u_curv, u_k1, u_k2, u_aberr, u_scan, u_scan_step, u_glow, u_glow_px, u_bloom, u_vign,
              u_specular, u_glare, u_tracking, u_band_h, u_band_center, u_flicker_mul;
uniform vec3  u_phosphor, u_glare_tint;
out vec4 o;

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
}
vec2 tube_warp(vec2 l, float k1, float k2) {
    vec2 c = l - 0.5;
    float r2 = dot(c, c);
    return 0.5 + c * (1.0 + k1 * r2 + k2 * r2 * r2);
}
vec3 S(vec2 px) { return texture(u_state, px / u_size).rgb; }

void main() {
    vec2 sz = u_size;
    vec2 l0 = (gl_FragCoord.xy - vec2(u_origin)) / sz;
    float k1 = u_k1 * u_curv, k2 = u_k2 * u_curv;
    vec2 l2 = tube_warp(l0, k1, k2);

    float inb = 1.0, shade = 1.0, lip = 0.0;
    if (u_curv > 0.0) {
        float short_side = min(sz.x, sz.y);
        float corner = clamp(short_side * 0.055, 10.0, 72.0);
        vec2 q = abs(l2 * sz - sz * 0.5) - sz * 0.5 + vec2(corner);
        float sd = min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - corner;
        const float aa = 1.5;
        inb = 1.0 - smoothstep(-aa, aa, sd);
        float lip_w = clamp(short_side * 0.045, 8.0, 48.0);
        float rim = 1.0 - smoothstep(0.0, lip_w, max(-sd, 0.0));
        float ly = clamp(l2.y, 0.0, 1.0);
        float from_top = mix(1.10, 0.85, ly);
        float ax = min(l2.x, 1.0 - l2.x) * sz.x;
        float ay = min(l2.y, 1.0 - l2.y) * sz.y;
        float cornerness = (1.0 - smoothstep(0.0, lip_w * 3.0, ax)) * (1.0 - smoothstep(0.0, lip_w * 3.0, ay));
        shade = 1.0 - clamp(0.85 * rim * sqrt(rim) * from_top + 0.30 * cornerness, 0.0, 0.94);
        float lip_d = (sd - aa * 1.5) / max(aa * 1.2, 1.0);
        lip = exp(-lip_d * lip_d) * mix(0.03, 0.09, ly) * (sd >= 0.0 ? 1.0 : 0.0);
    }

    vec2 L = clamp(l2, 0.0, 1.0);
    float r2 = dot(L - 0.5, L - 0.5);
    vec2 s = L * sz;

    float a = u_aberr * 0.004 * r2;
    vec2 lr = clamp(tube_warp(l0 + vec2(a, 0.0), k1, k2), 0.0, 1.0);
    vec2 lb = clamp(tube_warp(l0 - vec2(a, 0.0), k1, k2), 0.0, 1.0);
    vec3 lin = vec3(S(lr * sz).r, S(s).g, S(lb * sz).b);

    if (u_glow > 0.001) {
        float d = u_glow_px;
        lin += u_glow * 0.18 * 0.25 * (S(s + vec2(d, 0.0)) + S(s - vec2(d, 0.0)) + S(s + vec2(0.0, d)) + S(s - vec2(0.0, d)));
    }

    vec3 col = linear_to_srgb(lin);
    float fl = u_flicker_mul;

    float scanA = u_scan * fl;
    if (scanA > 0.001) {
        float row = mod(L.y * sz.y, u_scan_step);
        if (row < 1.0) col = mix(col, vec3(0.0), scanA);
        else if (row < 2.0) col = mix(col, u_phosphor, scanA * 0.22);
    }
    if (u_bloom > 0.001) col = mix(col, u_phosphor, 0.05 * u_bloom * smoothstep(0.05, 0.42, L.y));
    if (u_tracking > 0.001 && u_band_center > -1.0e8) {
        float d = 1.0 - clamp(abs(L.y * sz.y - u_band_center) / (u_band_h * 0.5), 0.0, 1.0);
        if (d > 0.0) {
            float add = d * d * 0.05 * u_tracking;
            float core = d > 0.92 ? 0.018 * u_tracking : 0.0;
            col = (col + u_phosphor * add + core) * (1.0 - 0.05 * u_tracking * d * 0.5);
        }
    }
    if (u_glare > 0.001) {
        vec2 g = (L - vec2(0.18, 0.10)) / vec2(0.34, 0.18);
        float hotspot = exp(-dot(g, g) * 2.2);
        float band = (1.0 - smoothstep(0.0, 0.10, abs(L.y + L.x * 0.22 - 0.17)))
                   * (1.0 - smoothstep(0.0, 0.72, L.x)) * (1.0 - smoothstep(0.0, 0.42, L.y));
        col += u_glare_tint * clamp((hotspot * 0.58 + band * 0.34) * u_glare, 0.0, 1.0);
    }
    if (u_specular > 0.001) {
        vec2 so = (L - vec2(0.05, 0.04)) / vec2(0.13, 0.10);
        col += vec3(0.05 * u_specular) * exp(-dot(so, so)) * fl;
    }
    float vig = 1.0 - u_vign * fl * 1.72 * r2;
    o = vec4(clamp(col * vig * shade * inb + lip, 0.0, 1.0), 1.0);
}
