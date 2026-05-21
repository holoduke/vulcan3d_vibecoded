#version 460

// Rasterised water-plane vertex stage. Replaces the old fullscreen
// triangle: a real plane mesh at the water level is drawn in the main
// world pass so the terrain↔water silhouette is pixel-exact hardware
// depth (no analytic-vs-rasterised mismatch → no edge gaps / seam
// lines). water.frag is the verbatim ocean shading from
// terrain_raymarch.frag; it still reconstructs the view ray from
// vWNear/vWFar exactly as before, so all water styles / reflections /
// foam / showthrough stay byte-identical.

layout(location = 0) in vec3 inPos;   // plane XZ in world metres (Y unused)

layout(location = 0) out vec2 vNDC;
layout(location = 1) out vec4 vWNear;
layout(location = 2) out vec4 vWFar;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;       // = inv(view_proj) — frag reconstructs the ray
    mat4 prev_mvp;
    vec4 color;        // .x = water level (world Y)
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

void main() {
    // Plane sits exactly at the water level so its rasterised depth IS
    // the water-surface depth — hardware-tested against the rasterised
    // terrain mesh for an exact shoreline (no compose / no analytic
    // depth write).
    vec3 wp = vec3(inPos.x, pc.color.x, inPos.z);
    vec4 clip = pc.mvp * vec4(wp, 1.0);
    gl_Position = clip;

    vec2 ndc = clip.xy / max(abs(clip.w), 1e-4) * sign(clip.w);
    vNDC   = ndc;
    vWNear = pc.model * vec4(ndc, 0.0, 1.0);
    vWFar  = pc.model * vec4(ndc, 1.0, 1.0);
}
