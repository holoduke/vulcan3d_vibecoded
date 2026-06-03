#version 460
#extension GL_EXT_ray_query : require

// Deferred lighting pass (Phase 3 of the deferred-render migration).
//
// Reads G-buffer + depth + the scene UBO + TLAS and computes the same
// direct-sun + ambient + sky-fill shading cube.frag does today. PCSS
// and ReSTIR GI replication is the Phase-3 follow-on; Phase 3 ships
// with sun direct + RT shadow + ambient + sky fill, which already
// covers the dominant visual signal on outdoor surfaces.
//
// Output goes to staging_color_image_ (RGBA16F). compose.frag picks
// scene_color vs staging_color via the deferred_lighting_active_
// runtime toggle — default OFF so the Phase 0 reference gate stays
// green until the lighting code reaches parity.

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

// Scene UBO — std140 lays out the leading vec4 sequence by offset, so
// declaring only the fields we read (matched in order + type to
// cube.frag's SceneUBO) is sufficient. Phase 3 reads through rt_params;
// everything below that we get to via the same-binding-different-set
// route in later phases when GI is ported.
layout(set = 0, binding = 0) uniform Scene {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;     // .x shadow_on
    vec4  rt_params;    // .w shadow_strength
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform sampler2D u_gbuffer0;     // albedo + mat
layout(set = 0, binding = 3) uniform sampler2D u_gbuffer1;     // normal + rough
layout(set = 0, binding = 4) uniform sampler2D u_depth;        // hardware depth
layout(set = 0, binding = 5) uniform sampler2D u_sun_shadow;   // light depth

// Push constant carries the inverse view-projection for world-pos
// reconstruction from the depth buffer. We could derive it from
// scene.light_vp + camera_pos but the engine already has a frame-cached
// inverse; pushing it lets us reuse that exact matrix.
layout(push_constant) uniform PC {
    mat4 inv_vp;
} pc;

// Octahedral normal decode (matches cube.frag's encode in Phase 2).
vec3 octa_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x > 0.0) ? -t : t;
    n.y += (n.y > 0.0) ? -t : t;
    return normalize(n);
}

// Any-hit shadow ray against the TLAS. Same cull-mask 0x01 cube.frag
// uses so shadow casters match exactly.
bool any_hit(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x01, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

void main() {
    vec2 uv = vUv;
    float z = texture(u_depth, uv).r;
    // Sky / depth-cleared pixels — let the existing forward sky path
    // win for those, just write a clear marker and bail. (Phase 4 will
    // bring sky shading into the lighting pass; Phase 3 leaves sky
    // pixels untouched so compose can mix forward sky with deferred
    // surface lighting once we wire that up.)
    if (z >= 0.999999) {
        outColor = vec4(scene.sky_color.rgb, 0.0);
        return;
    }

    // Reconstruct world position from depth + inverse-VP.
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clip = vec4(ndc, z, 1.0);
    vec4 world4 = pc.inv_vp * clip;
    vec3 world_pos = world4.xyz / world4.w;

    // Read G-buffer.
    vec4 g0 = texture(u_gbuffer0, uv);
    vec4 g1 = texture(u_gbuffer1, uv);
    vec3 albedo = g0.rgb;
    vec3 N      = octa_decode(g1.xy);

    // --- Sun direct (Lambert + RT shadow) ---
    vec3 L = normalize(scene.sun_direction.xyz);
    float ndl = max(dot(N, -L), 0.0);
    float sun_vis = 1.0;
    if (scene.rt_flags.x > 0 && ndl > 0.0) {
        vec3 origin = world_pos + N * 0.005;
        if (any_hit(origin, -L, 200.0)) sun_vis = 0.0;
    }
    // sun shadow strength slider attenuates how dark the shadow gets.
    float shadow_strength = clamp(scene.rt_params.w, 0.0, 1.0);
    sun_vis = mix(1.0, sun_vis, shadow_strength);
    vec3 sun_term = albedo * scene.sun_color.rgb * scene.sun_color.a *
                    ndl * sun_vis;

    // --- Ambient + sky fill ---
    // Cube.frag splits ambient into a sky-derived term (fades on
    // enclosed surfaces) and a ground term. Phase 3 uses the full
    // ambient as a single term; Phase 4 adds the sky-vis attenuation
    // from the ReSTIR sky-miss tracker.
    vec3 ambient_term = albedo * scene.ambient.rgb * scene.ambient.a;
    // Sky bounce — wrap-around fill from the upper hemisphere.
    vec3 sky_tint = scene.sky_color.rgb;
    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky_fill = albedo * sky_tint * 0.15 * up;

    vec3 lit = sun_term + ambient_term + sky_fill;
    outColor = vec4(lit, 1.0);
}
