#version 460

// Minimal depth-prepass vertex shader. Replaces cube.vert in the depth-
// only pipeline so the prepass doesn't pay the cost of:
//   * 1 unused mat4 multiply (prev_mvp; the model mat is needed for the
//     SPOM-wall shell extrusion gate below)
//   * 9 varying outputs (vNormal, vColor, vWorldPos, vEmissive, vUv,
//     vTexParams, vObjectPos, vObjectNormal, vPrevClip)
// All of which are dead in the prepass — cube_depth.frag is empty.
//
// Push-constant layout aliases the first mat4 of the full PushConstants
// struct (mvp). The rest of the 256-byte PC range is implicitly ignored.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // wall-vs-floor gate for SPOM extrusion
layout(location = 2) in vec2 inUv;       // unused, same reason

// Mirror cube.vert's push constants up to and including tex_params. The
// engine writes the same PushConstants struct (mvp, model, prev_mvp,
// color, emissive, tex_params, ...) to BOTH pipelines, so the layouts
// match by-name and SPIRV offset. We declare only the fields we need;
// the trailing bytes are ignored.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
} pc;

// Must match cube.vert's kShellThickness EXACTLY. If the two values
// drift, the depth-prepass writes a different shell-top depth than the
// colour pass, and LESS_OR_EQUAL in the colour pass starts rejecting
// wall fragments at sub-pixel ULP differences (sky patches on walls).
// Today SPOM walls (is_spom_albedo==true) skip the depth pre-pass
// entirely in world.cpp's render path (see "Skip SPOM brushes in the
// depth pre-pass" comment), so the extrusion here is defensive: if that
// skip is ever lifted, the prepass depth still matches the colour pass.
const float kShellThickness = 0.04;

void main() {
    int   albedo_idx = int(pc.tex_params.x);
    bool  is_spom_wall = (albedo_idx == 1 || albedo_idx == 4) &&
                         abs(inNormal.y) < 0.5;
    vec3 inPos_ext = inPosition;
    if (is_spom_wall) {
        inPos_ext += inNormal * kShellThickness;
    }
    gl_Position = pc.mvp * vec4(inPos_ext, 1.0);
}
