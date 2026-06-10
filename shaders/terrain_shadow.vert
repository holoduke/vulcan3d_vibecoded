#version 460

// Depth-only CLOD terrain for the sun shadow map. Mirrors terrain.vert's
// height fetch + geomorph EXACTLY (same instance data, same camera-based
// morph) so the shadow caster geometry matches what the camera passes
// rasterise. Only gl_Position differs: pc.mvp here is the light VP.
//
// Uses the full PushConstants block + scene set 0 (heightmap fetch needs
// binding 17, morph needs scene.camera_pos), so the terrain shadow draw
// binds the MAIN pipeline layout, not the slim sun_shadow layout.

layout(location = 0) in vec3 inGrid;
layout(location = 1) in vec4 inInstA;
layout(location = 2) in vec4 inInstB;

layout(push_constant) uniform PC {
    mat4 mvp;          // light view-proj
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;     // .xy = heightmap origin XZ, .z = 1/cell, .w = 1/(dim+1)
    vec4 tex_params;
    vec4 grass_params; // .z = rocky disp amp, .w = disp mip smooth
} pc;

layout(set = 0, binding = 12) uniform sampler2D u_height[6];
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
    vec4  spom_params;
} scene;

const float kGridCells = 64.0;

float hm_height(vec2 world_xz) {
    vec2 uv = ((world_xz - pc.emissive.xy) * pc.emissive.z + 0.5)
              * pc.emissive.w;
    return textureLod(u_heightmap_full, uv, 0.0).r;
}

void main() {
    vec2  origin = inInstA.xy;
    float size   = inInstA.z;
    float cell_w = size / kGridCells;

    vec2 grid = inGrid.xz;
    vec2 world0 = origin + grid * cell_w;
    float dist = length(world0 - scene.camera_pos.xz);
    float morph = clamp((dist - inInstB.x) / max(inInstB.y - inInstB.x, 1e-3),
                        0.0, 1.0);
    vec2 grid_m = mix(grid, floor(grid * 0.5) * 2.0, morph);
    vec2 world_xz = origin + grid_m * cell_w;

    vec3 pos = vec3(world_xz.x, hm_height(world_xz), world_xz.y);

    float kRockyAmp = max(0.0, pc.grass_params.z);
    float kDispMip  = clamp(pc.grass_params.w, 0.0, 6.0);
    float fade = 1.0 - smoothstep(60.0, 110.0, dist);
    if (kRockyAmp * fade > 1e-4) {
        float g_tile = max(scene.spom_params.z, 0.25);
        float h0 = textureLod(u_height[5], pos.xz / g_tile, kDispMip).r;
        pos.y += h0 * kRockyAmp * fade;
    }
    if (inGrid.y > 0.5) pos.y -= inInstB.z;

    gl_Position = pc.mvp * vec4(pos, 1.0);
}
