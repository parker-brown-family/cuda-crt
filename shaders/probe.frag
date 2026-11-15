#version 450 core
// Idle probe for the GLSL path: drawn inside an any-samples-passed query with
// colour writes off. A fragment survives only where the state is still moving,
// so the query answers "does this tile owe another frame?". The CUDA kernel
// answers the same question with one int written from inside the update.
uniform sampler2D u_state;
out vec4 o;
void main() {
    if (texelFetch(u_state, ivec2(gl_FragCoord.xy), 0).a == 0.0) discard;
    o = vec4(0.0);
}
