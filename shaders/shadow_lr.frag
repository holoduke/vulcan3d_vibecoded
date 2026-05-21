#version 460
#extension GL_EXT_ray_query : require

// Half-rate sun-shadow producer (roadmap item #4, Phase 2).
//
// Runs after the depth pre-pass and BEFORE the main cube color pass.
// Re-rasterises the same brush geometry at half render-extent, fires
// ONE shadow ray per half-res pixel with per-pixel disk jitter, and
// writes the occlusion bit [0..1] to a single R8 colour attachment
// (shadow_lr_image_). cube.frag's bilateral upsample (Phase 3) will
// sample this with depth/normal edge-stopping in place of its inline
// 32-tap PCSS trace.
//
// Phase 2 keeps the producer simple: 1 ray + fixed jitter cone
// (softness from rt_params.x). The half-res grid plus TAA's frame-
// accumulation reconstructs the soft penumbra. Phase 2.5 may port
// the full blocker-search + adaptive softness if Phase 3 measures
// inadequate penumbra.

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;     // .x:shadow_on, .y:shadow_samples, .w:frame
    vec4  rt_params;    // .x:shadow_softness, .w:shadow_strength
    ivec4 rt_flags2;
    vec4  rt_params2;
    vec4  camera_pos;
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) in  vec3 vNormal;
layout(location = 2) in  vec3 vWorldPos;
layout(location = 0) out float outOcc;

// Cheap hash → [0,1).
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

bool any_hit_shadow(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsOpaqueEXT |
                          gl_RayFlagsTerminateOnFirstHitEXT,
                          0xFFu, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

void main() {
    // Skip entirely when shadows disabled or strength=0 (matches cube.frag).
    if (scene.rt_flags.x == 0 || scene.rt_params.w < 1e-3) {
        outOcc = 0.0;
        return;
    }
    vec3 N = normalize(vNormal);
    vec3 L = normalize(scene.sun_direction.xyz);
    float n_dot_l = dot(N, L);
    if (n_dot_l < 0.04) { outOcc = 0.0; return; }   // self-shadow terminator

    // Per-pixel disk jitter inside a softness-sized cone. Frame-rotated so
    // TAA averages the noise across history into a soft penumbra.
    vec2 px = gl_FragCoord.xy + float(scene.rt_flags.w) * 17.0;
    float r1 = hash12(px);
    float r2 = hash12(px + vec2(13.0, 7.0));
    float ang = r1 * 6.28318530718;
    float rad = sqrt(r2) * scene.rt_params.x;
    vec3  ref = abs(L.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3  tu  = normalize(cross(ref, L));
    vec3  tv  = cross(L, tu);
    vec3  dir = normalize(L + (cos(ang) * tu + sin(ang) * tv) * rad);

    vec3 origin = vWorldPos + N * (0.005 + 0.02 * (1.0 - n_dot_l));
    outOcc = any_hit_shadow(origin, dir, 200.0) ? scene.rt_params.w : 0.0;
}
