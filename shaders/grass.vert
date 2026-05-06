#version 460

// Per-vertex blade-mesh attributes (5-vert ribbon).
layout(location = 0) in vec3 inPos;       // local-space (y in [0, blade_h])
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;        // uv.y = height ratio 0..1

// Per-instance blade attributes (one entry per blade in the
// instance buffer; matches GrassBlade in grass.h).
layout(location = 3) in vec4 inWorldPos;  // .xyz = blade base, .w = pad
layout(location = 4) in vec4 inRotHeight; // .x = rotation_y, .y = height_factor, .zw = pad
layout(location = 5) in vec4 inTint;      // .xyz = tint, .w = pad

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
} scene;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    // .x = max grass distance (m). Beyond this, vertex collapses to NaN
    //      so the rasteriser drops the triangle.
    // .y = wind strength
    // .z = time (seconds, used by wind sin)
    // .w = unused
    vec4 grass_params;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec2 vUv;
layout(location = 3) out float vHeightRatio;
layout(location = 4) out float vCullKill;

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
        vec3( c, 0.0,  s),
        vec3(0.0, 1.0, 0.0),
        vec3(-s, 0.0,  c)
    );
}

void main() {
    float rotation     = inRotHeight.x;
    float height_scale = inRotHeight.y;
    vec3  base_world   = inWorldPos.xyz;

    // Local position scaled by per-blade height variation.
    vec3 lp = inPos;
    lp.y *= height_scale;

    // Wind sway: bend the upper portion of the blade by a sin wave
    // that varies per-blade (hash on world XZ). The `inUv.y` (height
    // ratio 0..1) controls how much each vertex bends — the base is
    // anchored, the tip sways most.
    float wind_strength = pc.grass_params.y;
    float t = pc.grass_params.z;
    float phase = base_world.x * 0.13 + base_world.z * 0.21;
    float sway_x = sin(t * 2.4 + phase) * wind_strength;
    float sway_z = cos(t * 1.7 + phase * 1.3) * wind_strength * 0.6;
    // Quadratic falloff so only the top half visibly bends.
    float bend  = inUv.y * inUv.y;
    lp.x += sway_x * bend;
    lp.z += sway_z * bend;

    // Per-blade Y rotation, then translate to world.
    mat3 R = rotY(rotation);

    // Smooth distance fade: shrink blade height to zero over the last
    // 25% of grass_params.x. The hard cull-to-NaN at exactly grass_params.x
    // produced a visible "ring" where blades popped in/out of existence.
    // Shrinking instead of popping makes the boundary invisible.
    float view_dist_base = distance(base_world, scene.camera_pos.xyz);
    float fade_start = pc.grass_params.x * 0.75;
    float fade = 1.0 - smoothstep(fade_start, pc.grass_params.x, view_dist_base);
    lp.y *= fade;

    vec3 world = R * lp + base_world;

    // Hard cull: well beyond max distance, collapse to NaN clip space
    // so the rasteriser drops the triangle entirely.
    vCullKill = step(pc.grass_params.x + 5.0, view_dist_base);
    if (vCullKill > 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vNormal = vec3(0.0, 1.0, 0.0);
        vColor  = vec3(0.0);
        vUv     = vec2(0.0);
        vHeightRatio = 0.0;
        return;
    }

    gl_Position = pc.mvp * vec4(world, 1.0);

    // Normal in world space — the blade billboards roughly toward
    // camera by virtue of its random rotation; for shading we use a
    // mostly-up normal mixed with a slight side bias. Good enough for
    // foliage reads.
    vNormal = normalize(R * vec3(0.0, 0.8, 0.6));
    vColor  = inTint.xyz;
    vUv     = inUv;
    vHeightRatio = inUv.y;
}
