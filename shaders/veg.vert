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
    float fdp = fade[gl_InstanceIndex];
    float detail = floor(fdp) / 1023.0;          // continuous LOD: fraction of the tree drawn
    float fd = fract(fdp) / 0.99;                // mesh <-> billboard cross-fade
    int   matRaw = int(inUv.y + 0.5);
    int   mat0 = matRaw;
    vec3  Pl = inPosition, Nl = inNormal;
    float windW = inUv.x;
    vec3  card_uvk = vec3(0.0);
    // Continuous-LOD tree elements (eztree.h). Each carries a priority; it
    // is drawn when priority <= detail and grows from nothing just above
    // that, so branches and leaves fade in and out instead of popping.
    if (matRaw == 7 || matRaw >= 9) {            // leaf card
        float q = floor(inUv.x * 0.25);
        float corner = inUv.x - 4.0 * q;
        float prio = q / 1023.0;
        float g = clamp((detail - prio) / (0.03 + 0.25 * prio), 0.0, 1.0);
        // Fewer leaves -> larger ones, so the crown keeps its cover.
        float comp = inversesqrt(max(detail, 0.03));   // leaf count ~ detail -> area kept
        vec3 origin = inPosition - inNormal;
        Pl = origin + inNormal * (g * comp);
        float tone = fract(sin(dot(origin, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
        card_uvk = vec3(mod(corner, 2.0), floor(corner * 0.5 + 0.01), (matRaw == 7 ? 0.0 : float(matRaw - 8)) + 0.05 + 0.9 * tone);   // z = atlas cell + tone
        Nl = vec3(0.0, 1.0, 0.0);
        mat0 = 7;
    } else if (matRaw == 8) {                    // branch tube: thins to nothing
        float q = floor(inUv.x);
        float r = inUv.x - q;
        float prio = q / 1023.0;
        float g = clamp((detail - prio) / (0.03 + 0.25 * prio), 0.0, 1.0);
        Pl = inPosition - inNormal * (r * (1.0 - g));
        mat0 = 0;
    }
    // Rock: the mesh is only a bounding hull for the per-pixel ray march
    // (rock_sdf.glsl) - blow it up into a sphere that encloses the surface.
    float rock_tag = 0.0;
    if (matRaw == 4) {
        vec4 air = arch[uint(I.rot_arch_phase.y + 0.5)];
        if (air.z < -0.5) {
            rock_tag = air.z;
            float rv = -air.z - 1.0;
            float ramp = 0.26 + 0.1 * fract(rv * 0.37);
            const vec3 rc = vec3(0.0, 0.35, 0.0);
            float Rb = (1.0 + ramp + 0.12) * 1.30;
            Pl = rc + normalize(inPosition - rc) * Rb;
            // Camera inside (or touching) the hull: its front faces would be
            // clipped / culled and the rock vanished when you walked up to
            // it. Shrink the hull to a small ball around the camera instead;
            // the fragment shader then marches from the eye (rock_tag < -1000).
            float cr = cos(I.rot_arch_phase.x), sr = sin(I.rot_arch_phase.x);
            float rsc = I.pos_scale.w, rys = I.rot_arch_phase.w > 0.0 ? I.rot_arch_phase.w : 1.0;
            vec3 cw = scene.camera_pos.xyz - I.pos_scale.xyz;
            vec3 co = vec3(cr * cw.x + sr * cw.z, cw.y, -sr * cw.x + cr * cw.z) / vec3(rsc, rsc * rys, rsc);
            if (distance(co, rc) < Rb + 0.35 / max(rsc, 0.05)) {
                Pl = co + normalize(inPosition - rc) * (0.2 / max(rsc, 0.05));
                rock_tag -= 1000.0;
            }
        }
    }
    if (matRaw >= 7) {                           // wind weight from the position in the tree

        float Hl = max(arch[uint(I.rot_arch_phase.y + 0.5)].y * 2.0, 0.5);
        float hh = clamp(Pl.y / Hl, 0.0, 1.0);
        windW = clamp(0.85 * hh * hh + 0.25 * clamp(length(Pl.xz) / (0.35 * Hl), 0.0, 1.0), 0.0, 1.0);
    }
    float s   = I.pos_scale.w;
    float ysq = I.rot_arch_phase.w > 0.0 ? I.rot_arch_phase.w : 1.0;
    float c = cos(I.rot_arch_phase.x), sn = sin(I.rot_arch_phase.x);
    vec3 lp = Pl * vec3(s, s * ysq, s);
    vec3 wp = vec3(c * lp.x - sn * lp.z, lp.y, sn * lp.x + c * lp.z) + I.pos_scale.xyz;
    vec3 ln = Nl / vec3(1.0, ysq, 1.0);
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
    if ((mat0 >= 1 && mat0 <= 3) || mat0 == 6 || mat0 == 7) {
        vec4  ai = arch[uint(I.rot_arch_phase.y + 0.5)];
        float H  = ai.y * 2.0 * s * ysq;
        vec3  base = I.pos_scale.xyz;
        vec3  dc;
        if (mat0 == 1 || (mat0 == 7 && mod(floor(card_uvk.z), 2.0) > 0.5)) {   // conifer: outward from the trunk, a little up
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
    float w = windW;
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

    int mat = mat0;
    // Materials: 0 bark (rock tex, brown), 1 needles, 2 leaves, 3 bush,
    // 4 rock (rock tex). Foliage untextured -> cube.frag procedural tint.
    vec3  col = vec3(0.075, 0.16, 0.08);
    vec4  tp  = vec4(-1.0, -1.0, 1.0, 0.0);
    if (mat == 0)      { col = vec3(0.95, 0.90, 0.84); tp = vec4(16.0, 16.0, 0.55, 0.0); }   // bark texture
    else if (mat == 2) { col = vec3(0.16, 0.28, 0.07); }
    else if (mat == 3) { col = vec3(0.14, 0.24, 0.08); }
    else if (mat == 7) {            // leaf card: needles darker and bluer than broad leaves
        float tone = fract(card_uvk.z);
        col = (mod(floor(card_uvk.z), 2.0) > 0.5 ? vec3(0.72, 0.80, 0.76) : (card_uvk.z >= 2.0 ? vec3(0.46, 0.52, 0.40) : vec3(0.60, 0.64, 0.50))) * (0.80 + 0.40 * tone);   // tint over the atlas
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
    vUv         = vec2(windW, float(mat0));
    vTexParams  = tp;
    if (mat0 == 5) { vUv = imp_uv; vColor = inNormal.z < 1.5 ? vec3(0.075, 0.16, 0.08) : vec3(0.16, 0.28, 0.07); }
    vVegInfo = vec4(fd, (ve & 0x80000000u) != 0u ? 1.0 : 0.0,
                    mat0 == 5 ? inNormal.z : rock_tag, I.rot_arch_phase.z);
    if (rock_tag < -0.5) {      // object-space hull point + the instance transform for the march
        vObjectPos    = Pl;
        vShellBase    = I.pos_scale.xyz;
        vObjectNormal = vec3(s, ysq, I.rot_arch_phase.x);
    }
}
