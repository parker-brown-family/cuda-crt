#version 450 core
// The matched GLSL reference for kernels/phosphor.cu : phosphor_update_tex.
// Same arithmetic, same weights. A fragment shader cannot write the texture it
// reads, so this path needs two state textures and swaps them (ping-pong);
// the CUDA kernel updates one in place.
uniform sampler2D u_frame;  // RGBA8, sRGB-encoded, the compositor's picture
uniform sampler2D u_prev;   // RGBA32F linear-light state, this tile
uniform ivec2 u_origin;     // the tile's top-left in the frame
uniform float u_a_rise;
uniform vec3  u_a_decay;
uniform float u_eps;
out vec4 o;

vec3 srgb_to_linear(vec3 c) {
    return mix(c * (1.0 / 12.92), pow((c + 0.055) * (1.0 / 1.055), vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 target = srgb_to_linear(texelFetch(u_frame, p + u_origin, 0).rgb);
    vec3 prev   = texelFetch(u_prev, p, 0).rgb;
    vec3 a      = mix(u_a_decay, vec3(u_a_rise), greaterThan(target, prev));
    vec3 next   = prev + a * (target - prev);
    bvec3 settled = lessThan(abs(target - next), vec3(u_eps));
    next = mix(next, target, settled);
    // alpha carries "this pixel is still moving", for the idle probe
    o = vec4(next, all(settled) ? 0.0 : 1.0);
}
