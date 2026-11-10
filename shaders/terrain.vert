#version 460

// Quadtree CDLOD terrain vertex shader (vertex-free: no per-chunk
// geometry). One shared grid patch is drawn instanced; each instance is
// a quadtree node selected on the CPU by camera distance.
//
//   1. Grid coords (cells, [0..K]) -> world XZ via the instance's
//      origin/size.
//   2. Geomorph: inside the node's morph band, odd grid vertices slide
//      toward their even neighbour (grid-snap toward the parent LOD's
//      vertex set). At morph = 1 the edge is bit-identical to the
//      adjacent coarser node, so LOD seams are crack-free without
//      stitching.
//   3. Height + normal are fetched from the absolute heightmap texture
//      (binding 17) — the SAME data physics / BLAS / grass use, so all
//      height consumers agree by construction.
//   4. Near-field detail displacement from the rocky height map
//      (u_height[5]) replaces the old tessellation path. Bumps are
//      up-only (BLAS-safe contract).
//   5. Skirt vertices (flag in the patch mesh) drop by the instance's
//      skirt depth as fp-seam insurance.

layout(location = 0) in vec3 inGrid;    // x = grid_x, y = skirt flag, z = grid_z
layout(location = 1) in vec4 inInstA;   // origin_x, origin_z, size_m, lod
layout(location = 2) in vec4 inInstB;   // morph_start, morph_end, skirt_depth, pad

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;       // identity for terrain — kept for layout parity
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;    // .xy = heightmap origin XZ, .z = 1/cell, .w = 1/(dim+1)
    vec4 tex_params;
    vec4 grass_params; // .z = rocky disp amp, .w = disp mip smooth
} pc;

// Rocky detail height map (slot 5) — same array cube.frag samples.
layout(set = 0, binding = 12) uniform sampler2D u_height[6];
// Absolute terrain heightmap (R32F, (dim+1)^2, bilinear, clamp).
layout(set = 0, binding = 17) uniform sampler2D u_heightmap_full;

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
    vec4  spom_params;          // .z = ground tile metres (UI slider)
} scene;

layout(location = 0) out vec3 vNormal;
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
layout(location = 6) out vec3 vObjectPos;
layout(location = 7) out vec3 vObjectNormal;
layout(location = 8) out vec4 vPrevClip;
layout(location = 9) out vec3 vShellBase;      // cube.frag interface (unused for terrain)
layout(location = 10) flat out vec4 vVegInfo;  // cube.frag interface (0 = not vegetation)

const float kGridCells = 64.0;   // must match kClodGridCells

float hm_height(vec2 world_xz) {
    vec2 uv = ((world_xz - pc.emissive.xy) * pc.emissive.z + 0.5)
              * pc.emissive.w;
    return textureLod(u_heightmap_full, uv, 0.0).r;
}

// Raymarch-matched FBM detail (octaves 9+ of the baked eroded FBM, same
// recipe as terrain_raymarch.frag terrainM / cube.frag
// terrain_fbm_detail). Non-negative by construction (noise >= 0, erosion
// weight > 0) -> up-only displacement, BLAS-safe. `sp` = vertex spacing
// (m): octaves shorter than ~2 vertices are faded out.
float tv_hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 tv_noised(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
    float a = tv_hash21(i), b = tv_hash21(i + vec2(1, 0));
    float c = tv_hash21(i + vec2(0, 1)), d = tv_hash21(i + vec2(1, 1));
    return vec3(a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y,
                du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx));
}
float tv_fbm_detail(vec2 xz, float sp) {
    const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);
    vec2 p = xz * 0.003;
    float a = 0.0, b = 1.0, a_base = 0.0;
    vec2 d = vec2(0.0);
    for (int i = 0; i < 14; ++i) {
        float wl = 333.333 * exp2(-float(i));
        float w = (i < 9) ? 1.0 : smoothstep(sp * 2.0, sp * 4.0, wl);
        if (w <= 0.0) break;
        vec3 n = tv_noised(p);
        d += n.yz;
        a += b * n.x * w / (1.0 + dot(d, d));
        if (i == 8) a_base = a;
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    return (a - a_base) * 120.0;
}

void main() {
    vec2  origin = inInstA.xy;
    float size   = inInstA.z;
    float cell_w = size / kGridCells;          // node cell in world metres

    // Unmorphed world position drives the per-vertex morph factor so it
    // is continuous across node boundaries (shared edge vertices get the
    // same value on both sides).
    vec2 grid = inGrid.xz;
    vec2 world0 = origin + grid * cell_w;
    float dist = length(world0 - scene.camera_pos.xz);
    float morph = clamp((dist - inInstB.x) / max(inInstB.y - inInstB.x, 1e-3),
                        0.0, 1.0);

    // Geomorph: snap odd cells toward the even (parent-LOD) grid.
    vec2 grid_parent = floor(grid * 0.5) * 2.0;
    vec2 grid_m = mix(grid, grid_parent, morph);
    vec2 world_xz = origin + grid_m * cell_w;

    float h = hm_height(world_xz);

    // Gradient normal from the heightmap. The step scales with this
    // node's grid spacing: near nodes sample one source cell apart
    // (crisp), far nodes widen the stencil to half their own cell so
    // distant normals low-pass the heightmap instead of picking up
    // sub-triangle frequencies — fixed-step sampling gave far ridges
    // per-pixel normal noise that read as dark leaf-like speckle.
    float src_cell = 1.0 / max(pc.emissive.z, 1e-4);
    float step_m = max(src_cell, cell_w * 0.5);
    float hx0 = hm_height(world_xz - vec2(step_m, 0.0));
    float hx1 = hm_height(world_xz + vec2(step_m, 0.0));
    float hz0 = hm_height(world_xz - vec2(0.0, step_m));
    float hz1 = hm_height(world_xz + vec2(0.0, step_m));
    vec3 N = normalize(vec3(hx0 - hx1, 2.0 * step_m, hz0 - hz1));

    vec3 pos = vec3(world_xz.x, h, world_xz.y);

    // Rocky-grass detail displacement (replaces tessellation). Same
    // formula and tile size as cube.frag's splat so bumps align with
    // the textured pixels. Up-only: BLAS-safe.
    // Phase 2: amplitude is banded by altitude to roughly track the
    // splat layers — calm sand near the waterline, full relief on the
    // grass band, strongest on the rock band, softened under snow.
    float kRockyAmp = max(0.0, pc.grass_params.z);
    float kDispMip  = clamp(pc.grass_params.w, 0.0, 6.0);
    float fade = 1.0 - smoothstep(60.0, 110.0, dist);
    // Castle-plateau gate: the courtyard floor slab sits only 5 cm
    // above the terrain, and stairs/walls hug it. Up to 0.45 m of
    // detail displacement here pokes terrain THROUGH the courtyard
    // tiles, buries the lower stair steps, and bulges in front of
    // wall bases (read as "transparent walls at grazing angles").
    // Fade the displacement to zero inside the plateau footprint.
    float plat = max(abs(world_xz.x), abs(world_xz.y));
    fade *= smoothstep(11.5, 16.0, plat);
    if (kRockyAmp * fade > 1e-4) {
        float band_sand = smoothstep(scene.terrain_h_low.x,
                                     scene.terrain_h_low.y, h);
        float band_rock = smoothstep(scene.terrain_h_high.x,
                                     scene.terrain_h_high.y, h);
        float band_snow = smoothstep(scene.terrain_h_high.z,
                                     scene.terrain_h_high.w, h);
        float amp_band = mix(0.35, 1.0, band_sand);   // sand → grass
        amp_band = mix(amp_band, 1.6, band_rock);     // rock band
        amp_band = mix(amp_band, 0.8, band_snow);     // snow softens
        float g_tile = max(scene.spom_params.z, 0.25);
        vec2 uvR = pos.xz / g_tile;
        float h0 = textureLod(u_height[5], uvR, kDispMip).r;
        pos.y += h0 * kRockyAmp * amp_band * fade;
    }

    // FBM micro-relief as real geometry near the camera. Spacing estimate
    // from DISTANCE (not this node's cell size) so vertices shared by
    // neighbouring LOD nodes get the identical offset - no cracks.
    {
        float sp = max(0.25, dist * 0.0045);
        float fw = (1.0 - smoothstep(70.0, 120.0, dist)) * smoothstep(11.5, 16.0, plat);
        if (fw > 1e-3 && sp < 0.33) pos.y += tv_fbm_detail(pos.xz, sp) * fw;
    }

    // Skirt drop AFTER height/detail so the ring hangs from the real edge.
    if (inGrid.y > 0.5) pos.y -= inInstB.z;

    gl_Position = pc.mvp * vec4(pos, 1.0);
    vWorldPos = pos;
    vNormal = N;
    vColor = vec3(1.0);
    vEmissive = vec4(0.0);
    vUv = vec2(0.0);
    vTexParams = pc.tex_params;
    vObjectPos = pos;
    vObjectNormal = N;
    vShellBase = pos;
    vVegInfo = vec4(0.0);
    vPrevClip = pc.prev_mvp * vec4(pos, 1.0);
}
