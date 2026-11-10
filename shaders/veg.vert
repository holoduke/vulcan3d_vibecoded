#version 460
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

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;
    vec4  rt_params;
    ivec4 rt_flags2;
    vec4  rt_params2;
    vec4  camera_pos;
    vec4  rt_lod;
    vec4  viewport;
    vec4  muzzle_pos;
    vec4  muzzle_color;
    vec4  terrain_params;
    vec4  terrain_h_low;
    vec4  terrain_h_high;
    vec4  grass_extra;
    vec4  grass_extra2;
    mat4  light_vp;
    vec4  terrain_extra;
    vec4  _scene_pad[24];
    vec4  restir_params;
    vec4  spom_params;
    vec4  _terrain_local_info;
    vec4  _terrain_max_grid[256];
    vec4  _terrain_disp_params;
    vec4  _terrain_antitile_params;
    vec4  _grass_shadow_on_terrain_params;
    vec4  grass_side_lit_params;
    vec4  _voxel_origin, _voxel_dims, _voxel_grid;
    vec4  _road_info, _road_bbox;
    vec4  _road_pts[64];
    vec4  grass_rm_params;   // .y wind, .z time (s)
} scene;

struct VegInstance { vec4 pos_scale; vec4 rot_arch_phase; };
layout(std430, set = 1, binding = 0) readonly buffer Instances { VegInstance inst[]; };
layout(std430, set = 1, binding = 1) readonly buffer Visible   { uint visible[]; };
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

    // Wind: trunk base still, crown sways; per-instance phase.
    float w = inUv.x;
    if (w > 0.001) {
        float t  = scene.grass_rm_params.z;
        float ws = 0.35 + clamp(scene.grass_rm_params.y, 0.0, 3.0);
        vec2  wd = normalize(vec2(0.8, 0.45));
        float ph = I.rot_arch_phase.z;
        float sway = sin(t * 1.3 + ph) * 0.6 + sin(t * 2.9 + ph * 1.7) * 0.25 + 0.35;
        float flutter = sin(t * 9.0 + dot(wp, vec3(1.7, 2.3, 1.9))) * 0.04;
        wp.xz += wd * (sway * 0.22 * ws * w * w * s);
        wp += vec3(flutter * ws * w * w);
    }

    int mat = int(inUv.y + 0.5);
    // Materials: 0 bark (rock tex, brown), 1 needles, 2 leaves, 3 bush,
    // 4 rock (rock tex). Foliage untextured -> cube.frag procedural tint.
    vec3  col = vec3(0.075, 0.16, 0.08);
    vec4  tp  = vec4(-1.0, -1.0, 1.0, 0.0);
    if (mat == 0)      { col = vec3(0.85, 0.66, 0.50); tp = vec4(7.0, 7.0, 1.4, 0.0); }
    else if (mat == 2) { col = vec3(0.16, 0.28, 0.07); }
    else if (mat == 3) { col = vec3(0.14, 0.24, 0.08); }
    else if (mat == 4) { col = vec3(1.4, 1.34, 1.27); tp = vec4(7.0, 7.0, 0.22, 0.0); }
    // Per-instance hue jitter so a forest isn't one green.
    float hj = fract(I.rot_arch_phase.z * 0.6180339);
    if (mat >= 1 && mat <= 3) col *= mix(vec3(0.85, 1.05, 0.8), vec3(1.15, 0.95, 1.1), hj);

    gl_Position = pc.mvp * vec4(wp, 1.0);
    vPrevClip   = pc.prev_mvp * vec4(wp, 1.0);
    vWorldPos   = wp;
    vShellBase  = wp;
    vObjectPos  = wp;
    vObjectNormal = N;
    vNormal     = N;
    vColor      = col;
    vEmissive   = vec4(0.0);
    vUv         = inUv;
    vTexParams  = tp;
    if (mat0 == 5) { vUv = imp_uv; vColor = inNormal.z < 1.5 ? vec3(0.075, 0.16, 0.08) : vec3(0.16, 0.28, 0.07); }
    vVegInfo = vec4(fd, (ve & 0x80000000u) != 0u ? 1.0 : 0.0,
                    mat0 == 5 ? inNormal.z : 0.0, I.rot_arch_phase.z);
}
