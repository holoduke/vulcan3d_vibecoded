#version 460
#extension GL_EXT_ray_query : require

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vUv;
layout(location = 3) in float vHeightRatio;
layout(location = 4) in float vCullKill;
layout(location = 5) in vec3 vWorldPos;

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
    vec4  grass_extra;   // x: height_scale, y: alpha_cutoff
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Single shadow ray to the sun. cull-mask 0x01 keeps it consistent
// with cube.frag's shadow tests — sparks and bullets stay out.
// t_min = 1.0 (one metre along the ray) so the origin is clear of
// the full-res terrain BLAS which is co-located with the heightmap
// the blade base sits on. A smaller bias would self-intersect and
// the shadow ray would always report "blocked", but the symptom
// reported was the opposite (shadows not casting at all) which
// suggests this descriptor was reachable but the ray was never
// missing — every shadow test was reading the wrong / no hit.
// Larger t_min makes both failure modes correct: ray clears terrain
// and only registers a hit on something distinct (castle, crate).
bool sun_blocked(vec3 origin, vec3 dir) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x01, origin, 1.0, dir, 200.0);
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
    // Light intensity as a SCALAR — preserves the blade's green tint
    // regardless of the sun direction. Earlier per-channel sky-colour
    // multiplication crushed red/green and pulled shaded blades toward
    // the blue sky tint.
    float sun_amt = scene.sun_color.a * 0.10 * n_dot_l;
    float sky_amt = 0.30;

    // Shadow: one any-hit ray to the sun. Origin is bumped 0.30m up
    // off the heightmap surface so the t_min=1.0 inside sun_blocked
    // can fully escape the BLAS terrain even at glancing sun angles.
    // 200m max-t covers the tallest mountains in the scene.
    if (scene.rt_flags.x != 0) {
        if (sun_blocked(vWorldPos + vec3(0.0, 0.30, 0.0),
                        normalize(scene.sun_direction.xyz))) {
            sun_amt = 0.0;
        }
    }
    float lum = sun_amt + sky_amt;

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
