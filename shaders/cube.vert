#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;     // for world-space position + normal
    mat4 prev_mvp;  // prev_view_proj * prev_model — drives motion vector at fs.location=1
    vec4 color;
    vec4 emissive;  // rgb = emissive, a = skip-lighting flag
    // Texture indices + UV scale + spare. -1 in either index = solid color.
    //   x: albedo index
    //   y: normal index
    //   z: uv scale (multiplier on inUv before sampling, for tiling)
    //   w: spare
    vec4 tex_params;
} pc;

layout(location = 0) out vec3 vNormal;
// vColor/vEmissive/vTexParams come from push-constant — identical at all 3
// verts. Must be `flat` so perspective interpolation precision can't drift
// the value (which corrupted texture-index gating for vTexParams; vColor +
// vEmissive are flagged for consistency / future-proofing).
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
// Local object-space position (pre-model). Lets the fragment shader sample
// triplanar textures in object space for moving objects — without it the
// world-position based sample would "flow" across rotating boxes and the
// texture detail aliasing reads as light/dark flicker.
layout(location = 6) out vec3 vObjectPos;
// Local object-space normal (pre-model). For object-space triplanar we also
// need the unrotated normal to compute the projection blend weights.
layout(location = 7) out vec3 vObjectNormal;
// Previous-frame clip-space position (prev_view_proj * prev_model * local).
// cube.frag turns this into a screen-space motion vector at location=1.
layout(location = 8) out vec4 vPrevClip;

void main() {
    // Conservative-geometry SPOM extrusion for the brick slot. cube.frag
    // ray-marches up to ~12cm of world-space depth into the brick height
    // map; if the silhouette is the original cube edge, that depth is
    // hidden at corners. Push brick-brush vertices outward along their
    // normal by the same world distance — the discarded pixels in the
    // extension zone then have valid depth (depth pre-pass writes the
    // extruded surface too, so depth-EQUAL passes in the color pass).
    // Gated on tex_params.x == 1 (brick) and not emissive (skip lamps),
    // matches the spom_path gate in cube.frag.
    const float kSpomExtWorld = 0.12;
    bool brick_extrude = int(pc.tex_params.x) == 1 && pc.emissive.a < 0.5;
    vec3 world_n_unscaled = mat3(pc.model) * inNormal;
    float ext_factor = brick_extrude
        ? kSpomExtWorld / max(length(world_n_unscaled), 0.01)
        : 0.0;
    vec3 inPos_ext = inPosition + inNormal * ext_factor;

    gl_Position = pc.mvp * vec4(inPos_ext, 1.0);
    vec4 wp = pc.model * vec4(inPos_ext, 1.0);
    vWorldPos = wp.xyz;
    // For uniform-scale or rotation-only models, mat3(model) is fine. We accept
    // small error on non-uniform scale (the static brushes use non-uniform
    // scaling for floor/walls); cubes have axis-aligned faces so it still reads
    // correctly enough for shading.
    vNormal = mat3(pc.model) * inNormal;
    vColor = pc.color.rgb;
    vEmissive = pc.emissive;
    vUv = inUv;
    vTexParams = pc.tex_params;
    vObjectPos = inPos_ext;
    vObjectNormal = inNormal;
    vPrevClip = pc.prev_mvp * vec4(inPos_ext, 1.0);
}
