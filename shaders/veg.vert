#version 460
#extension GL_GOOGLE_include_directive : require
// GPU-driven vegetation / rock instances (vegetation.cpp). Draws come from
// vkCmdDrawIndexedIndirect buckets (archetype x LOD) written by
// veg_cull.comp; gl_InstanceIndex (incl. firstInstance) indexes the
// compacted visible list -> instance record. Output interface matches
// cube.vert exactly so cube.frag / cube_gbuf.frag shade it like any
// static mesh (push constants: grass_params = (0, 1, 1, -1) -> mesh mode,
// world triplanar). Per-vertex material id (uv.y) picks colour / texture.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;      // .x wind weight, .y material

layout(push_constant) uniform PC {
    mat4 mvp;        // view-proj (instances are in world space)
    mat4 model;
    mat4 prev_mvp;   // prev view-proj
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

#include "scene_ubo.glsl"

struct VegInstance { vec4 pos_scale; vec4 rot_arch_phase; };
layout(std430, set = 1, binding = 0) readonly buffer Instances { VegInstance inst[]; };
layout(std430, set = 1, binding = 1) readonly buffer Visible   { uint visible[]; };
layout(std430, set = 1, binding = 3) readonly buffer ArchInfo  { vec4 arch[]; };   // radius, half height, impostor, species
layout(std430, set = 1, binding = 5) readonly buffer Fade      { float fade[]; };

layout(location = 0) out vec3 vNormal;
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
layout(location = 6) out vec3 vObjectPos;
layout(location = 7) out vec3 vObjectNormal;
layout(location = 8) out vec4 vPrevClip;
layout(location = 9) out vec3 vShellBase;
layout(location = 10) flat out vec4 vVegInfo;

void main() {
    uint  ve = visible[gl_InstanceIndex];
    VegInstance I = inst[ve & 0x7FFFFFFFu];
    float fd = fade[gl_InstanceIndex];
    int   mat0 = int(inUv.y + 0.5);
    float s   = I.pos_scale.w;
    float ysq = I.rot_arch_phase.w > 0.0 ? I.rot_arch_phase.w : 1.0;
    float c = cos(I.rot_arch_phase.x), sn = sin(I.rot_arch_phase.x);
    vec3 lp = inPosition * vec3(s, s * ysq, s);
    vec3 wp = vec3(c * lp.x - sn * lp.z, lp.y, sn * lp.x + c * lp.z) + I.pos_scale.xyz;
    vec3 ln = inNormal / vec3(1.0, ysq, 1.0);
    vec3 N  = normalize(vec3(c * ln.x - sn * ln.z, ln.y, sn * ln.x + c * ln.z));
    // Impostor card (LOD 4, material 5): cylindrical billboard facing the
    // camera. inPosition = (u * half_width, v * height, kind).
    vec2 imp_uv = vec2(0.0);
    if (mat0 == 5) {
        vec3 base = I.pos_scale.xyz;
        vec3 toc = scene.camera_pos.xyz - base; toc.y = 0.0;
        toc = length(toc) > 1e-3 ? normalize(toc) : vec3(0.0, 0.0, 1.0);
        vec3 rt = vec3(toc.z, 0.0, -toc.x);
        wp = base + rt * (inPosition.x * s) + vec3(0.0, inPosition.y * s, 0.0);
        N = toc;
        imp_uv = vec2(inNormal.x, inNormal.y);      // (u, v) packed in the normal
    }

    const vec3 rest_wp = wp;   // pre-sway position: anchors the leaf cut-out pattern
    // Crown-level normals (The Witness' vegetation / cloud shading): each
    // leaf clump is its own little ball, so with its own normals every
    // clump gets a lit top and a dark underside and the crown reads as a
    // pile of balls. Bending the normals toward the direction from the
    // crown centre gives one smooth field across the whole crown - it
    // then shades as a single soft volume, with the clumps as detail.
    vec3 card_uvk = vec3(0.0);
    if (mat0 == 7) { card_uvk = inNormal; N = vec3(0.0, 1.0, 0.0); }   // leaf card: (u, v, kind + tone)
    if ((mat0 >= 1 && mat0 <= 3) || mat0 == 6 || mat0 == 7) {
        vec4  ai = arch[uint(I.rot_arch_phase.y + 0.5)];
        float H  = ai.y * 2.0 * s * ysq;
        vec3  base = I.pos_scale.xyz;
        vec3  dc;
        if (mat0 == 1 || (mat0 == 7 && card_uvk.z >= 1.0)) {   // conifer: outward from the trunk, a little up
            vec2 o = wp.xz - base.xz;
            dc = normalize(vec3(o.x, 0.45 * length(o) + 0.05, o.y));
        } else {
            dc = normalize(wp - (base + vec3(0.0, H * (mat0 == 3 ? 0.40 : 0.68), 0.0)));
        }
        N = normalize(mix(N, dc, mat0 == 7 ? 0.8 : (mat0 == 1 ? 0.45 : (mat0 == 6 ? 0.4 : 0.6))));
    }
    // Wind: trunk base still, crown sways; per-instance phase. The previous
    // frame's pose (time - 0.016, the CPU step) feeds the motion vector so
    // swaying crowns reproject correctly in TAA.
    float w = inUv.x;
    vec3 wp_prev = wp;
    if (w > 0.001) {
        float ws = 0.35 + clamp(scene.grass_rm_params.y, 0.0, 3.0);
        vec2  wd = normalize(vec2(0.8, 0.45));
        float ph = I.rot_arch_phase.z;
        vec3  rest = wp;
        for (int k = 0; k < 2; ++k) {
            float t = scene.grass_rm_params.z - (k == 1 ? 0.016 : 0.0);
            float sway = sin(t * 1.3 + ph) * 0.6 + sin(t * 2.9 + ph * 1.7) * 0.25 + 0.35;
            float flutter = sin(t * 9.0 + dot(rest, vec3(1.7, 2.3, 1.9))) * 0.04;
            vec3 p = rest;
            p.xz += wd * (sway * 0.22 * ws * w * w * s);
            p += vec3(flutter * ws * w * w);
            if (k == 0) wp = p; else wp_prev = p;
        }
    }

    int mat = int(inUv.y + 0.5);
    // Materials: 0 bark (rock tex, brown), 1 needles, 2 leaves, 3 bush,
    // 4 rock (rock tex). Foliage untextured -> cube.frag procedural tint.
    vec3  col = vec3(0.075, 0.16, 0.08);
    vec4  tp  = vec4(-1.0, -1.0, 1.0, 0.0);
    if (mat == 0)      { col = vec3(0.85, 0.66, 0.50); tp = vec4(7.0, 7.0, 1.4, 0.0); }
    else if (mat == 2) { col = vec3(0.16, 0.28, 0.07); }
    else if (mat == 3) { col = vec3(0.14, 0.24, 0.08); }
    else if (mat == 7) {            // leaf card: needles darker and bluer than broad leaves
        float tone = fract(card_uvk.z);
        col = (card_uvk.z >= 1.0 ? vec3(0.070, 0.150, 0.075) : vec3(0.150, 0.290, 0.065)) * (0.78 + 0.44 * tone);
    }
    else if (mat == 6) {            // leaf blade (real geometry): each leaf its own tone
        float lv = fract(sin(dot(floor(rest_wp * 2.3), vec3(12.9898, 78.233, 37.719))) * 43758.5453);
        col = vec3(0.17, 0.31, 0.07) * (0.78 + 0.44 * lv);
    }
    else if (mat == 4) { col = vec3(1.4, 1.34, 1.27); tp = vec4(7.0, 7.0, 0.22, 0.0); }
    // Per-instance hue jitter so a forest isn't one green.
    float hj = fract(I.rot_arch_phase.z * 0.6180339);
    if ((mat >= 1 && mat <= 3) || mat == 6 || mat == 7) col *= mix(vec3(0.85, 1.05, 0.8), vec3(1.15, 0.95, 1.1), hj);

    gl_Position = pc.mvp * vec4(wp, 1.0);
    vPrevClip   = pc.prev_mvp * vec4(wp_prev, 1.0);
    vWorldPos   = wp;
    vShellBase  = wp;
    vObjectPos  = rest_wp;
    vObjectNormal = (mat0 == 7) ? card_uvk : N;
    vNormal     = N;
    vColor      = col;
    vEmissive   = vec4(0.0);
    vUv         = inUv;
    vTexParams  = tp;
    if (mat0 == 5) { vUv = imp_uv; vColor = inNormal.z < 1.5 ? vec3(0.075, 0.16, 0.08) : vec3(0.16, 0.28, 0.07); }
    vVegInfo = vec4(fd, (ve & 0x80000000u) != 0u ? 1.0 : 0.0,
                    mat0 == 5 ? inNormal.z : 0.0, I.rot_arch_phase.z);
}
