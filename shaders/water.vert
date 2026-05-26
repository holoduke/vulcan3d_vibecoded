#version 460

// Rasterised water-plane vertex stage. Real plane mesh at the water
// level is drawn in the main world pass so the terrain↔water silhouette
// is pixel-exact hardware depth. Outputs perspective-correct world
// position so the frag does NOT have to reconstruct via inv(view_proj),
// which had per-frame float drift and made the shoreline appear to
// shift/tilt with camera motion.

layout(location = 0) in vec3 inPos;   // plane XZ in world metres (Y unused)

layout(location = 0) out vec2 vNDC;
layout(location = 1) out vec4 vWNear;
layout(location = 2) out vec4 vWFar;
// Real world position of THIS pixel on the plane, perspective-correctly
// interpolated from the actual vertex positions. The plane mesh is at
// fixed y = water_y, so vWPos.y is exactly water_y at every pixel; xz
// is the world point where the screen ray pierces the plane. Replaces
// the frag's inv(view_proj) reconstruction — which depended on
// glm::inverse(vp) recomputed each frame and drifted by ~µm-mm per
// frame as the camera moved, producing the visible shoreline orientation
// wobble.
layout(location = 3) out vec3 vWPos;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;       // = inv(view_proj) — kept for reflection-helper code
    mat4 prev_mvp;
    vec4 color;        // .x = water level (world Y)
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

void main() {
    vec3 wp = vec3(inPos.x, pc.color.x, inPos.z);
    vec4 clip = pc.mvp * vec4(wp, 1.0);
    gl_Position = clip;

    vec2 ndc = clip.xy / max(abs(clip.w), 1e-4) * sign(clip.w);
    vNDC   = ndc;
    vWNear = pc.model * vec4(ndc, 0.0, 1.0);
    vWFar  = pc.model * vec4(ndc, 1.0, 1.0);
    vWPos  = wp;
}
