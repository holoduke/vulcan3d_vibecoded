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

const float kGridCells = 64.0;   // must match kClodGridCells

float hm_height(vec2 world_xz) {
    vec2 uv = ((world_xz - pc.emissive.xy) * pc.emissive.z + 0.5)
              * pc.emissive.w;
    return textureLod(u_heightmap_full, uv, 0.0).r;
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

    // Gradient normal from the heightmap (central differences one source
    // cell apart — pc.emissive.z is 1/cell so step = cell).
    float step_m = 1.0 / max(pc.emissive.z, 1e-4);
    float hx0 = hm_height(world_xz - vec2(step_m, 0.0));
    float hx1 = hm_height(world_xz + vec2(step_m, 0.0));
    float hz0 = hm_height(world_xz - vec2(0.0, step_m));
    float hz1 = hm_height(world_xz + vec2(0.0, step_m));
    vec3 N = normalize(vec3(hx0 - hx1, 2.0 * step_m, hz0 - hz1));

    vec3 pos = vec3(world_xz.x, h, world_xz.y);

    // Rocky-grass detail displacement (replaces tessellation). Same
    // formula and tile size as cube.frag's splat so bumps align with
    // the textured pixels. Up-only: BLAS-safe.
    float kRockyAmp = max(0.0, pc.grass_params.z);
    float kDispMip  = clamp(pc.grass_params.w, 0.0, 6.0);
    float fade = 1.0 - smoothstep(60.0, 110.0, dist);
    if (kRockyAmp * fade > 1e-4) {
        float g_tile = max(scene.spom_params.z, 0.25);
        vec2 uvR = pos.xz / g_tile;
        float h0 = textureLod(u_height[5], uvR, kDispMip).r;
        pos.y += h0 * kRockyAmp * fade;
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
    vPrevClip = pc.prev_mvp * vec4(pos, 1.0);
}
