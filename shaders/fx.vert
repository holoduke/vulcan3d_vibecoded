#version 460
// FX pass vertex shader (fx.cpp). Shared by:
//   * ribbons  - CPU-built camera-facing strips in WORLD space (model = I):
//                normal = rgb tint, uv = (alpha | billboard s, across t)
//   * volumes  - unit cube [-0.5, 0.5]^3 scaled/translated by pc.model
//                around an explosion volume (fx_volume.frag raymarches it).

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;      // FX: inverse view-projection (volume depth recon)
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vTint;
layout(location = 2) out vec2 vUv;

void main() {
    vWorld = (pc.model * vec4(inPosition, 1.0)).xyz;
    vTint  = inNormal;
    vUv    = inUv;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
