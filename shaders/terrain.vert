#version 460

// Terrain vertex shader with CD-LOD morphing. Identical to cube.vert
// for everything but Y, which is lerped between the actual sample
// height (position.y) and the per-vertex parent_y attribute. The
// CPU sets pc.color.w to the morph factor: 0 = full LOD-0 surface,
// 1 = LOD-1 (stride-2 linear interp). Smooth distance-based ramp in
// world.cpp avoids the visible "pop" when a chunk crosses the LOD
// threshold.

layout(location = 0) in vec3  inPosition;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inUv;
// CD-LOD parent height — lives in vertex binding 1 (a separate buffer
// per chunk). Generated at chunk build by gen_chunk_parent_y.
layout(location = 3) in float inParentY;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;       // .rgb = solid colour, .w = morph factor (terrain only)
    vec4 emissive;
    vec4 tex_params;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
layout(location = 6) out vec3 vObjectPos;
layout(location = 7) out vec3 vObjectNormal;
layout(location = 8) out vec4 vPrevClip;

void main() {
    float morph = clamp(pc.color.w, 0.0, 1.0);
    vec3 pos = inPosition;
    pos.y = mix(pos.y, inParentY, morph);

    gl_Position = pc.mvp * vec4(pos, 1.0);
    vec4 wp = pc.model * vec4(pos, 1.0);
    vWorldPos = wp.xyz;
    vNormal = mat3(pc.model) * inNormal;
    // Solid colour for terrain is always white; pc.color.w doubles as the
    // morph slot, so we ignore the alpha component for shading.
    vColor = vec3(1.0);
    vEmissive = pc.emissive;
    vUv = inUv;
    vTexParams = pc.tex_params;
    vObjectPos = pos;
    vObjectNormal = inNormal;
    vPrevClip = pc.prev_mvp * vec4(pos, 1.0);
}
