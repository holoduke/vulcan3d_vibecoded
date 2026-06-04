#version 460
#extension GL_EXT_ray_query : require

// Deferred lighting pass (Phases 3-6 of the deferred-render migration).
//
// Reads gbuffer0 (albedo + material_id), gbuffer1 (octahedral normal +
// roughness + metallic), depth, the scene UBO + TLAS, and the
// sun-shadow map. Computes the same shading cube.frag's forward path
// produces:
//   - Sun direct (Lambert + half-Lambert wrap)
//   - PCSS-style variable-cone soft shadow via inline RT
//   - RTAO short-ray ambient occlusion
//   - Sky-vis-attenuated ambient + sky bounce
//   - Material-id branching (stone, wood, metal, terrain, grass, voxel)
//   - Per-light point-light loop (lanterns + muzzle flash)
//
// ReSTIR temporal-reservoir GI is the Phase 7 follow-on; the rest of
// the lighting is here. compose.frag picks scene_color (forward) vs
// staging_color (deferred) via the deferred_lighting_active_ runtime
// toggle.

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Scene {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;        // .x shadow_on, .y shadow_samples, .z ao_samples, .w frame
    vec4  rt_params;       // .x shadow_softness, .y ao_radius, .z ambient_strength, .w shadow_strength
    ivec4 rt_flags2;       // .w ao_mode
    vec4  rt_params2;
    vec4  camera_pos;
    vec4  rt_lod;
    vec4  viewport;
    vec4  muzzle_pos;      // .xyz origin, .w intensity
    vec4  muzzle_color;    // .rgb colour, .w radius (m)
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform sampler2D u_gbuffer0;
layout(set = 0, binding = 3) uniform sampler2D u_gbuffer1;
layout(set = 0, binding = 4) uniform sampler2D u_depth;
layout(set = 0, binding = 5) uniform sampler2D u_sun_shadow;

layout(push_constant) uniform PC {
    mat4 inv_vp;
    // Per-frame point lights. .xyz = world position, .w = radius (m).
    // Per-light colour comes in light_colors with .a = intensity. Up to
    // 4 active lights — overflow lights drop. Inactive slots set radius
    // to 0 so the early-out gate culls them at no cost.
    vec4 light_pos[4];
    vec4 light_col[4];
} pc;

// Octahedral normal decode — matches cube.frag's encode.
vec3 octa_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x > 0.0) ? -t : t;
    n.y += (n.y > 0.0) ? -t : t;
    return normalize(n);
}

// Inline-RT any-hit shadow ray. Cull-mask 0x01 = shadow casters
// (sparks/projectiles flagged 0xFE so they're skipped).
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

// Closest-hit with t-value — used by the PCSS blocker search.
bool closest_hit(vec3 origin, vec3 dir, float t_max, out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0x01, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        out_t = rayQueryGetIntersectionTEXT(rq, true);
        return true;
    }
    return false;
}

// 32-tap Vogel disk — same one cube.frag uses for PCSS sample placement.
const vec2 kVogel[32] = vec2[32](
    vec2( 0.176,  0.000), vec2(-0.272,  0.198),
    vec2( 0.034, -0.401), vec2( 0.353,  0.353),
    vec2(-0.604, -0.044), vec2( 0.486, -0.420),
    vec2( 0.084,  0.689), vec2(-0.621,  0.395),
    vec2( 0.787,  0.057), vec2(-0.605, -0.567),
    vec2( 0.094, -0.866), vec2( 0.633,  0.659),
    vec2(-0.949,  0.179), vec2( 0.787, -0.591),
    vec2(-0.144,  0.991), vec2(-0.663,  0.769),
    vec2( 0.988, -0.151), vec2(-0.849, -0.601),
    vec2( 0.176, -1.022), vec2( 0.776,  0.785),
    vec2(-1.097,  0.156), vec2( 0.853, -0.766),
    vec2(-0.197,  1.121), vec2(-0.747,  0.913),
    vec2( 1.158, -0.197), vec2(-0.969, -0.769),
    vec2( 0.221, -1.213), vec2( 0.926,  0.917),
    vec2(-1.249,  0.197), vec2( 0.997, -0.918),
    vec2(-0.243,  1.297), vec2(-0.901,  1.046)
);

// Cheap deterministic hash for per-pixel jitter — matches cube.frag's
// pattern (no salt animation, since TAA accumulates jitter across
// frames).
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 uv = vUv;
    float z = texture(u_depth, uv).r;
    // Sky / depth-cleared pixels. The deferred path doesn't have the
    // sky procedural-gradient code; instead we leave a sentinel that
    // composes against scene_color where forward already wrote sky.
    if (z >= 0.999999) {
        outColor = vec4(scene.sky_color.rgb, 0.0);
        return;
    }

    // World-pos reconstruction from depth + inverse-VP.
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clip = vec4(ndc, z, 1.0);
    vec4 world4 = pc.inv_vp * clip;
    vec3 world_pos = world4.xyz / world4.w;

    // G-buffer reads.
    vec4 g0 = texture(u_gbuffer0, uv);
    vec4 g1 = texture(u_gbuffer1, uv);
    vec3 albedo = g0.rgb;
    int  material_id = int(g0.a * 255.0 + 0.5);
    vec3 N = octa_decode(g1.xy);
    float roughness = g1.b;

    // Pre-shaded materials (7 = water, 8 = raymarched terrain) — these
    // pipelines already do their own complex shading (refraction, fog,
    // foam, FBM lighting), so the deferred path just passes the colour
    // through. Material 5 (emissive) likewise.
    if (material_id == 7 || material_id == 8 || material_id == 5) {
        outColor = vec4(albedo, 1.0);
        return;
    }

    vec3 cam_pos = scene.camera_pos.xyz;
    vec3 view_vec = cam_pos - world_pos;
    float cam_dist = length(view_vec);

    // ---------------------------------------------------------------
    //  Sun direct
    // ---------------------------------------------------------------
    vec3 L = normalize(scene.sun_direction.xyz);
    float ndl_raw = dot(N, -L);
    // Double-sided lighting (matches cube.frag): flip N on back-faces
    // so interiors don't read pitch-black.
    if (ndl_raw < 0.0) { N = -N; ndl_raw = -ndl_raw; }
    float ndl = max(ndl_raw, 0.0);

    // PCSS-style soft shadow. Mirrors cube.frag's:
    //   1. Blocker search: N rays in a wide cone, take avg t.
    //   2. Penumbra estimate proportional to avg_t × softness.
    //   3. Stratified shadow rays in the size-adapted cone.
    float shadow = 0.0;
    bool shadow_on = scene.rt_flags.x != 0 &&
                     scene.rt_params.w > 1e-3 &&
                     ndl_raw > 0.04;
    if (shadow_on) {
        vec3 ref = abs(L.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                     : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref, L));
        vec3 tan_v = cross(L, tan_u);

        float bias = 0.005 + 0.02 * (1.0 - ndl_raw);
        vec3 origin = world_pos + N * bias;
        float base_softness = scene.rt_params.x;
        float kBlockerCone = base_softness * 4.0;
        float kBlockerTMax = clamp(cam_dist * 4.0, 30.0, 200.0);

        // 2-tap blocker search (cube.frag uses 2 too).
        float sum_t = 0.0;
        int hits = 0;
        float pp_phi = hash12(gl_FragCoord.xy + 99.0) * 6.28318530718;
        float pp_c = cos(pp_phi), pp_s = sin(pp_phi);
        for (int i = 0; i < 2; ++i) {
            float r1 = hash12(gl_FragCoord.xy + vec2(i * 11.0, 47.0));
            float r = sqrt(r1) * kBlockerCone;
            vec2 v = kVogel[i & 31];
            float vx = pp_c * v.x - pp_s * v.y;
            float vy = pp_s * v.x + pp_c * v.y;
            vec3 jitter = (vx * tan_u + vy * tan_v) * r;
            vec3 dir = normalize(-L + jitter);
            float t;
            if (closest_hit(origin, dir, kBlockerTMax, t)) {
                sum_t += t;
                ++hits;
            }
        }
        if (hits > 0) {
            float avg_t = sum_t / float(hits);
            float kShadowTMax = clamp(avg_t * 4.0, 20.0, 200.0);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0,
                                  clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            float penumbra = clamp(base_softness * scale * 0.6,
                                   base_softness * 0.25,
                                   base_softness * 6.0);

            // Stratified shadow rays. Sample count from slider, capped.
            int N_s = max(1, min(int(scene.rt_flags.y), 16));
            int taken = 0;
            float blocked = 0.0;
            float pp_phi_s = hash12(gl_FragCoord.xy + 17.0) * 6.28318530718;
            float pp_c_s = cos(pp_phi_s), pp_s_s = sin(pp_phi_s);
            int strata = int(ceil(sqrt(float(N_s))));
            float inv_strata = 1.0 / float(strata);
            for (int sy = 0; sy < strata && taken < N_s; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s; ++sx) {
                    int idx = taken;
                    float r1 = hash12(gl_FragCoord.xy + vec2(idx, 91.0));
                    float u1 = (float(sx) + r1) * inv_strata;
                    float r = sqrt(u1) * penumbra;
                    vec2 v = kVogel[idx & 31];
                    float vx = pp_c_s * v.x - pp_s_s * v.y;
                    float vy = pp_s_s * v.x + pp_c_s * v.y;
                    vec3 jitter = (vx * tan_u + vy * tan_v) * r;
                    vec3 dir = normalize(-L + jitter);
                    if (any_hit(origin, dir, kShadowTMax)) blocked += 1.0;
                    ++taken;
                }
            }
            shadow = (taken > 0) ? (blocked / float(taken)) : 0.0;
        }
        shadow *= clamp(scene.rt_params.w, 0.0, 1.0);
    }

    // ---------------------------------------------------------------
    //  Ambient occlusion (short-ray RT, modes off / fast / full / HBAO)
    // ---------------------------------------------------------------
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0) {
        // 2-tap "fast" AO as default; full mode could expand to 4-8 taps.
        int n_taps = (ao_mode == 1) ? 2 : 4;
        float ao_radius = max(0.05, scene.rt_params.y);
        float hits = 0.0;
        vec3 t_u = abs(N.y) < 0.999 ? normalize(cross(vec3(0,1,0), N))
                                     : vec3(1,0,0);
        vec3 t_v = cross(N, t_u);
        vec3 origin = world_pos + N * 0.01;
        for (int i = 0; i < n_taps; ++i) {
            float r1 = hash12(gl_FragCoord.xy + vec2(i * 7.0, 23.0));
            float r2 = hash12(gl_FragCoord.xy + vec2(i * 7.0, 91.0));
            float phi = r1 * 6.28318530718;
            float ct = sqrt(r2);              // cosine-weighted hemisphere
            float st = sqrt(1.0 - r2);
            vec3 dir = normalize(
                t_u * (cos(phi) * st) + t_v * (sin(phi) * st) + N * ct);
            if (any_hit(origin, dir, ao_radius)) hits += 1.0;
        }
        ao = 1.0 - (hits / float(n_taps));
    }

    // ---------------------------------------------------------------
    //  Path-traced GI (Phase 4b — simplified port of cube.frag's loop)
    //
    //  Fires N cosine-weighted hemisphere rays around N. Each ray
    //  closest-hits the TLAS; on a miss it accumulates the sky tint.
    //  Hit albedo uses a mid-grey approximation (the deferred path
    //  doesn't carry the materials buffer; the small albedo error gets
    //  absorbed by the sky_vis blend and the per-pixel hash).
    //
    //  sky_vis tracks the fraction of first-bounce rays that escape to
    //  sky — used below to attenuate the ambient/sky_fill terms so
    //  enclosed surfaces (castle interior) read as indoor rather than
    //  the same flat brightness as outdoors.
    // ---------------------------------------------------------------
    vec3 gi_indirect = vec3(0.0);
    float sky_vis = 1.0;
    int N_gi = max(0, scene.rt_flags2.x);
    if (material_id == 3) N_gi = 0;        // terrain skips GI (matches cube.frag)
    float gi_strength = clamp(scene.rt_params2.x, 0.0, 1.0);
    if (N_gi > 0 && gi_strength > 1e-3) {
        // 4-tap GI cap matches cube.frag's lod_samples() at typical cam
        // distance; higher slider values feed into the TAA-jittered
        // history blend, not per-pixel ray count, on the deferred path.
        int taken_gi = min(N_gi, 4);
        vec3 N_up  = abs(N.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 N_tan = normalize(cross(N_up, N));
        vec3 N_bit = cross(N, N_tan);
        vec3 ro    = world_pos + N * 0.01;
        float gi_radius = max(8.0, scene.rt_params2.y);
        int sky_hits = 0;
        for (int i = 0; i < taken_gi; ++i) {
            float r1 = hash12(gl_FragCoord.xy + vec2(i * 13.0, 73.0));
            float r2 = hash12(gl_FragCoord.xy + vec2(i * 13.0, 47.0));
            float r_h = sqrt(r1);
            float phi = 6.28318530718 * r2;
            vec3 ld = vec3(r_h * cos(phi),
                           sqrt(max(0.0, 1.0 - r1)),
                           r_h * sin(phi));
            vec3 dir = ld.x * N_tan + ld.y * N + ld.z * N_bit;
            float t;
            if (closest_hit(ro, dir, gi_radius, t)) {
                vec3 hit_pos = ro + dir * t;
                vec3 hit_n   = -dir;
                // Sun-shadow at the hit point: one any-hit ray; if clear,
                // hit gets sun + ambient; otherwise just ambient.
                vec3 hit_light = scene.ambient.rgb * scene.ambient.a * 0.6;
                float n_dot_sun = max(dot(hit_n, -L), 0.0);
                if (n_dot_sun > 0.0 &&
                    !any_hit(hit_pos + hit_n * 0.01, -L, 200.0)) {
                    hit_light += scene.sun_color.rgb *
                                  scene.sun_color.a * n_dot_sun;
                }
                // Mid-grey albedo proxy for the bounce hit. The colour
                // bleed is approximate; ReSTIR + per-material albedo
                // reuse would tighten this in a follow-on.
                gi_indirect += vec3(0.5) * hit_light;
            } else {
                // Miss → sky. Use sky_color tinted by direction.up so
                // overhead rays brighten more than horizon ones.
                float up_w = clamp(dir.y, 0.0, 1.0);
                gi_indirect += scene.sky_color.rgb * (0.4 + 0.6 * up_w);
                ++sky_hits;
            }
        }
        gi_indirect /= float(taken_gi);
        gi_indirect *= gi_strength;
        // sky_vis: fraction of rays that escaped to sky. 0 = fully
        // enclosed → ambient/sky_fill suppressed; 1 = open sky → full.
        sky_vis = float(sky_hits) / float(taken_gi);
    }

    // ---------------------------------------------------------------
    //  Ambient + sky bounce
    // ---------------------------------------------------------------
    // Sky-vis fallback: bias the ambient up where N points up so
    // outdoor surfaces (terrain, walkways) get the sky contribution
    // even before ReSTIR is ported.
    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky_tint = scene.sky_color.rgb;
    float ambient_strength = clamp(scene.rt_params.z, 0.0, 1.0);
    // sky_vis attenuates the sky-derived ambient on enclosed pixels —
    // matches cube.frag's "interior darkening" behaviour. floor() ensures
    // even fully-enclosed surfaces still get some baseline ambient.
    float vis_blend = mix(0.25, 1.0, sky_vis);
    vec3 ambient_term = albedo * scene.ambient.rgb * scene.ambient.a *
                        ambient_strength * ao * vis_blend;
    vec3 sky_fill = albedo * sky_tint * 0.18 * up * ao * vis_blend;

    // Sun term.
    vec3 sun_term = albedo * scene.sun_color.rgb * scene.sun_color.a *
                    ndl * (1.0 - shadow);

    // ---------------------------------------------------------------
    //  Material-id tweaks
    // ---------------------------------------------------------------
    // Per-material tweaks (Phase 5):
    //   2 (wood)    — no change (default Lambert reads right)
    //   3 (terrain) — boosted ambient to match the open-sky shading
    //                 cube.frag does on terrain pixels
    //   4 (grass)   — extra wrap-around fill so shadowed blades don't
    //                 read pitch-black
    //   5 (emissive) — pass albedo through unchanged (brushes already
    //                  baked emissive colour into albedo)
    if (material_id == 3) {
        ambient_term *= 1.3;
        sky_fill     *= 1.2;
    } else if (material_id == 4) {
        ambient_term *= 1.4;
    } else if (material_id == 5) {
        sun_term     += albedo;       // emissive pass-through
    }

    // ---------------------------------------------------------------
    //  Point lights — variable list driven by push constant. Muzzle
    //  flash, lanterns, future fires all share this loop; the CPU
    //  packs up to 4 active lights per frame.
    // ---------------------------------------------------------------
    vec3 point_term = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        float radius = pc.light_pos[i].w;
        if (radius < 1e-3) continue;
        vec3 lp = pc.light_pos[i].xyz;
        vec3 to = lp - world_pos;
        float d2 = dot(to, to);
        float r2 = radius * radius;
        if (d2 >= r2) continue;
        vec3 ld = to * inversesqrt(d2);
        float pndl = max(dot(N, ld), 0.0);
        if (pndl <= 0.0) continue;
        float fall = 1.0 - clamp(sqrt(d2) / radius, 0.0, 1.0);
        fall = fall * fall;
        point_term += albedo * pc.light_col[i].rgb *
                      pc.light_col[i].a * pndl * fall;
    }

    vec3 lit = sun_term + ambient_term + sky_fill + point_term +
               albedo * gi_indirect;
    outColor = vec4(lit, 1.0);
}
