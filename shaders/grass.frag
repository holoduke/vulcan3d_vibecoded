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
layout(location = 8) in vec4 vPrevClip;

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
    vec4  grass_extra2;  // x: alt_min, y: alt_max, z: shadow_map_world_half
    mat4  light_vp;      // unused in frag, keeps UBO layout identical
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
    // Standard Lambert (no half-Lambert wrap). Blades whose pseudo-
    // billboard normal faces away from the sun get zero direct
    // contribution → the shadow side of a slope reads as visibly
    // darker than the lit side.
    float n_dot_l = max(dot(N, L), 0.0);

    // Light intensity (scalar, preserves blade tint). Sun contribution
    // is the dominant term so blade brightness actually responds to
    // sun direction; constant sky term is small enough to keep
    // back-facing slopes visible without flattening the look. The
    // cap below still bounds total brightness for bloom safety.
    float sun_amt = scene.sun_color.a * 0.35 * n_dot_l;
    float sky_amt = 0.15;

    // Shadow factor:
    //   - Sun shadow comes from the pre-baked heightmap shadow texture
    //     sampled in grass.vert (vSunShadow). Mountain/terrain self-
    //     shadows now apply at any distance with zero per-frame ray
    //     cost.
    //   - Sky-up "is roofed" ray still fires per fragment for blades
    //     within 30m — handles "inside the keep" but not at distance.
    const float kSkyShadowDist = 30.0;
    float shadow_factor = 1.0;
    if (vSunShadow > 0.5) shadow_factor *= 0.18;
    if (scene.rt_flags.x != 0 && vDistToCam < kSkyShadowDist) {
        vec3 origin = vWorldPos + vec3(0.0, 0.5, 0.0);
        if (sun_blocked(origin, vec3(0.0, 1.0, 0.0))) {
            shadow_factor *= 0.45;
        }
    }
    // Translucent back-lit term — when the sun is behind the blade
    // (camera looks roughly toward sun, blade between), foliage
    // glows from forward subsurface scattering. n_dot_l is small in
    // that case but `view_dot_l = dot(view_dir, L)` is large. A 0.45
    // power tightens the lobe to grazing-only. Yellow-green tint
    // matches plant chlorophyll's transmittance peak.
    vec3 V = normalize(scene.camera_pos.xyz - vWorldPos);
    float view_dot_l = max(dot(-V, L), 0.0);
    float backlit = pow(view_dot_l, 4.0) * (1.0 - n_dot_l) * vHeightRatio;
    vec3  trans   = vec3(0.95, 1.05, 0.55) *
                    backlit * 0.30 * scene.sun_color.a;

    // Base AO — darken the bottom 20% of the blade so blades read as
    // anchored to the ground instead of floating. Falls off past 60 m
    // so distant blades don't waste shading on a sub-pixel detail.
    float base_ao = mix(0.55, 1.0, smoothstep(0.0, 0.20, vHeightRatio));
    float ao_fade = 1.0 - smoothstep(40.0, 100.0, vDistToCam);
    base_ao = mix(1.0, base_ao, ao_fade);

    float lum = (sun_amt + sky_amt) * shadow_factor;

    vec3 tip_lift = mix(vec3(0.95), vec3(1.05, 1.0, 0.9), vHeightRatio);
    vec3 base = vColor * tip_lift * base_ao;

    vec3 lit = base * lum + trans * vColor;
    // Hard ceiling — guarantees we never feed the bloom mip chain with
    // grass pixels above its threshold even under aggressive
    // auto-exposure boosts.
    lit = min(lit, vec3(0.50));

    // Soft alpha across the blade UV — adjacent blades blend without
    // back-to-front sorting. Discard threshold from grass_extra.y so
    // the user can slide the silhouette between full rectangle and
    // tapered tip.
    float side_taper = smoothstep(0.0, 0.18, vUv.x) *
                       (1.0 - smoothstep(0.82, 1.0, vUv.x));
    if (side_taper < scene.grass_extra.y && vHeightRatio > 0.85) discard;

    outColor = vec4(lit, 1.0);
    // Screen-space motion vector — current_uv − prev_uv. Same pattern
    // as cube.frag. Without this, walking grass goes black-flickery
    // because TAA reprojects against the same screen pixel from the
    // previous frame (which was a different blade or sky).
    {
        vec2 current_uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            vec2 prev_uv  = prev_ndc * vec2(0.5, 0.5) + vec2(0.5);
            outMotion = current_uv - prev_uv;
        } else {
            outMotion = vec2(0.0);
        }
    }
    // (older comment retained as note for the wind case — wind sway not tracked)
}
