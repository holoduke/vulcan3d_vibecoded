#version 460

// Terrain vertex shader with CD-LOD morphing AND rocky-grass
// displacement (mirror of the tess-eval path). Identical to cube.vert
// for everything but Y, which is:
//   1. Lerped between actual sample height (position.y) and the per-
//      vertex parent_y attribute by `morph` (CD-LOD smooth pop hide).
//   2. Raised by sampling the rocky-grass SPOM height map (u_height[5])
//      at world XZ -- the SAME formula and tile size the tess eval
//      shader uses, so the terrain look stays consistent whether the
//      tess pipeline is active or not. Was producing flat ground when
//      the user disabled tess in the UI; now the non-tess path also
//      gets the rocky displacement (at base mesh vertex density).

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
    // .x = sand ripple, .y = grass line, .z = rocky disp amp,
    // .w = disp mip smooth. Slots .z/.w mirror the tess path so
    // the user's UI sliders drive both pipelines identically.
    vec4 grass_params;
} pc;

// SPOM height map array. Layout matches cube.frag's binding 12
// declaration. Slot 5 = rocky-rugged height map. descriptors.cpp now
// flags this binding for the vertex stage so the lookup is valid.
layout(set = 0, binding = 12) uniform sampler2D u_height[6];

// Scene UBO -- declare only the prefix through spom_params so we can
// read the ground-tile metres slider (matches cube.frag + tese tile
// scale, so displacement aligns with the visible texture tiling).
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

void main() {
    float morph = clamp(pc.color.w, 0.0, 1.0);
    vec3 pos = inPosition;
    pos.y = mix(pos.y, inParentY, morph);

    // Rocky-grass vertex displacement -- mirror of terrain_tess.tese.
    // Sample slot 5 (rocky-rugged height) at the same UV scale as
    // cube.frag's albedo splat and add `h0 * amp` to Y. Same bumps-
    // up-only contract as the tese (BLAS-safe). Amplitude knob lives
    // in pc.grass_params.z, mip bias in .w.
    float kRockyAmp = max(0.0, pc.grass_params.z);
    float kDispMip  = clamp(pc.grass_params.w, 0.0, 6.0);
    if (kRockyAmp > 1e-4) {
        float g_tile = max(scene.spom_params.z, 0.25);
        vec2 uvR = pos.xz / g_tile;
        float h0 = textureLod(u_height[5], uvR, kDispMip).r;
        pos.y += h0 * kRockyAmp;
    }

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
