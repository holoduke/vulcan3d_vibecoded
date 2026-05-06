#version 460
#extension GL_EXT_ray_query : require

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vUv;
layout(location = 3) in float vHeightRatio;
layout(location = 4) in float vCullKill;
layout(location = 5) in vec3 vWorldPos;
layout(location = 6) in float vSunShadow;
layout(location = 7) in float vDistToCam;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

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
    vec4  grass_extra;   // x: height_scale, y: alpha_cutoff, z: slope_n_min, w: distance_density
    vec4  grass_extra2;  // x: alt_min, y: alt_max
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Single any-hit shadow ray. cull-mask 0x01 matches cube.frag.
// t_min stays near zero — the spatial offset on the origin (caller
// adds 0.5m straight up) is what clears the heightmap BLAS so the
// ray can't self-intersect the ground we're standing on. Earlier
// fix used t_min=1m which skipped PAST close-by occluders (a small
// crate just above the grass would already be behind the ray's
// start point), so castle/crate shadows weren't casting on grass.
bool sun_blocked(vec3 origin, vec3 dir) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x01, origin, 0.001, dir, 200.0);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

void main() {
    if (vCullKill > 0.5) discard;

    // Cheap lighting — the blades are tiny, RT shadows on every pixel
    // would be wasteful. Half-Lambert sun + blue-tinted sky ambient
    // keeps them readable as foliage from any angle.
    vec3 N = normalize(vNormal);
    vec3 L = normalize(scene.sun_direction.xyz);
    float n_dot_l = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);

    // Lighting deliberately heavily under-baked. Grass occupies a
    // large fraction of the lower screen and any per-pixel value above
    // the bloom threshold (≈1.0 by default) blows up into halos when
    // auto-exposure compensates for darker portions of the scene
    // (castle interior). Hard cap below.
    // Light intensity (scalar, preserves blade tint).
    float sun_amt = scene.sun_color.a * 0.10 * n_dot_l;
    float sky_amt = 0.30;

    // Shadow factor:
    //   - Sun shadow comes from a per-vertex ray result (vSunShadow,
    //     interpolated as flat varying — see grass.vert). One ray
    //     per blade vertex, vs the previous per-pixel ray which on
    //     dense close-up grass was hundreds of rays per blade.
    //   - Sky-up "is roofed" ray still runs per fragment for blades
    //     within kSkyShadowDist (only relevant inside the keep).
    const float kSkyShadowDist = 45.0;
    float shadow_factor = 1.0;
    if (vSunShadow > 0.5) shadow_factor *= 0.18;
    if (scene.rt_flags.x != 0 && vDistToCam < kSkyShadowDist) {
        vec3 origin = vWorldPos + vec3(0.0, 0.5, 0.0);
        if (sun_blocked(origin, vec3(0.0, 1.0, 0.0))) {
            shadow_factor *= 0.45;
        }
    }
    float lum = (sun_amt + sky_amt) * shadow_factor;

    vec3 tip_lift = mix(vec3(0.95), vec3(1.05, 1.0, 0.9), vHeightRatio);
    vec3 base = vColor * tip_lift;

    vec3 lit = base * lum;
    // Hard ceiling — guarantees we never feed the bloom mip chain with
    // grass pixels above its threshold even under aggressive
    // auto-exposure boosts.
    lit = min(lit, vec3(0.32));

    // Soft alpha across the blade UV — adjacent blades blend without
    // back-to-front sorting. Discard threshold from grass_extra.y so
    // the user can slide the silhouette between full rectangle and
    // tapered tip.
    float side_taper = smoothstep(0.0, 0.18, vUv.x) *
                       (1.0 - smoothstep(0.82, 1.0, vUv.x));
    if (side_taper < scene.grass_extra.y && vHeightRatio > 0.85) discard;

    outColor = vec4(lit, 1.0);
    outMotion = vec2(0.0);   // wind motion not tracked — TAA tolerates
}
