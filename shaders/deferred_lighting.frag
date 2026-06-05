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
    // Padding to reach the fog params at the same byte offsets cube.frag
    // and the C++ scene UBO use. 27 vec4 = terrain_params, terrain_h_low,
    // terrain_h_high, grass_extra, grass_extra2, light_vp (mat4 = 4 vec4),
    // terrain_extra, _scene_pad[0..16].
    vec4  _pad_a[27];
    // _scene_pad[17] in cube.frag — base fog tint .rgb + master strength .a.
    vec4  distance_fog_color;
    // _scene_pad[18] in cube.frag — .x density, .y start, .z height_top,
    // .w max alpha.
    vec4  distance_fog_params;
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform sampler2D u_gbuffer0;
layout(set = 0, binding = 3) uniform sampler2D u_gbuffer1;
layout(set = 0, binding = 4) uniform sampler2D u_depth;
layout(set = 0, binding = 5) uniform sampler2D u_sun_shadow;
// scene_color from the forward pass — used at sky pixels (depth=1) and
// at pixels that aren't covered by the G-buffer write (particles,
// decals, projectiles, anything drawn after cube.frag without a
// G-buffer output). Lets the deferred path pass them through unmodified
// instead of overwriting with reconstructed sky.
layout(set = 0, binding = 6) uniform sampler2D u_scene_color;
// Baked terrain shadow texture — cube.frag uses this to put castle
// shadows on the terrain plateau. It's a 2048m-wide top-down texture
// of the shadow factor sampled at vWorldPos.xz / 2048 + 0.5. Only
// sampled when material_id == 3 (terrain receiver).
layout(set = 0, binding = 7) uniform sampler2D u_terrain_shadow;

layout(push_constant) uniform PC {
    mat4 inv_vp;
    // Per-frame point lights. .xyz = world position, .w = radius (m).
    // Per-light colour comes in light_colors with .a = intensity. Up to
    // 4 active lights — overflow lights drop. Inactive slots set radius
    // to 0 so the early-out gate culls them at no cost.
    vec4 light_pos[4];
    vec4 light_col[4];
} pc;

// Procedural sky — verbatim port of cube.frag's sample_sky so deferred
// sky pixels match the forward output exactly.
vec3 sample_sky(vec3 dir) {
    vec3 L = normalize(scene.sun_direction.xyz);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    vec3 sky = mix(horizon, zenith, sqrt(up));
    float h1 = max(dot(dir, L), 0.0);
    float h2 = h1 * h1;
    float h4 = h2 * h2;
    float halo = h4 * h4;
    sky += scene.sun_color.rgb * scene.sun_color.a * 0.08 * halo;
    return sky;
}

// Octahedral normal decode — matches cube.frag's encode.
vec3 octa_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x > 0.0) ? -t : t;
    n.y += (n.y > 0.0) ? -t : t;
    return normalize(n);
}

// Inline-RT any-hit shadow ray. Mask parameter selects which BLAS
// instances participate: 0x01 = brushes (default), 0x02 = terrain-only
// shadow casters (cube.frag uses 0x02 when receiver is_terrain_pre so
// castle walls don't shadow the open terrain plateau). 0xFF = both.
bool any_hit_m(vec3 origin, vec3 dir, float t_max, uint mask) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          mask, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}
bool any_hit(vec3 origin, vec3 dir, float t_max) {
    return any_hit_m(origin, dir, t_max, 0x01u);
}

// Closest-hit with t-value — used by the PCSS blocker search.
bool closest_hit_m(vec3 origin, vec3 dir, float t_max, uint mask, out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          mask, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        out_t = rayQueryGetIntersectionTEXT(rq, true);
        return true;
    }
    return false;
}
bool closest_hit(vec3 origin, vec3 dir, float t_max, out float out_t) {
    return closest_hit_m(origin, dir, t_max, 0x01u, out_t);
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

// Per-pixel deterministic noise — matches cube.frag's ign+rand pair so
// the deferred shader picks the SAME Vogel disk rotations and shadow
// sample sequences as the forward path. Different hash functions =
// different shadow values per pixel = scene 3 / 7 pixel-level drift.
float ign(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
float rand(uvec3 seed) {
    float s = float(seed.x) + 5.588238 * float(seed.z);
    float t = float(seed.y) + 1.388765 * float(seed.z);
    return ign(vec2(s, t));
}
// Distance-LOD sample reducer — same as cube.frag's lod_samples. Close
// pixels get the full requested count; far pixels collapse toward 1.
int lod_samples(int requested, float dist) {
    float lod_t = clamp((dist - scene.rt_lod.x) /
                        max(0.001, scene.rt_lod.y - scene.rt_lod.x), 0.0, 1.0);
    int n = int(round(mix(float(requested), 1.0, lod_t)));
    return max(1, n);
}
// Back-compat shim for the remaining hash12() callsites (GI, AO). Maps
// vec2 to a deterministic float via the same rand machinery so the
// random stream is consistent throughout the shader.
float hash12(vec2 p) {
    return rand(uvec3(uint(p.x), uint(p.y), 0u));
}

void main() {
    vec2 uv = vUv;
    // seed_base — matches cube.frag exactly (FragCoord int + frame & 7
    // in z slot). All shadow + GI + AO ray jitters derive from this
    // seed via rand() so deferred lighting picks the same Vogel disk
    // rotations cube.frag picks at the same pixel.
    uvec3 seed_base = uvec3(uint(gl_FragCoord.x),
                            uint(gl_FragCoord.y),
                            uint(scene.rt_flags.w) & 7u);
    float z = texture(u_depth, uv).r;
    // World-pos / view-direction reconstruction from depth + inverse-VP.
    vec2 ndc = uv * 2.0 - 1.0;
    // Sky / depth-cleared pixels — forward's compose pass samples an
    // equirect skybox texture we don't have bound in the deferred
    // descriptor set. Pass scene_color through so the forward sky path
    // wins at depth=1, matching cube.frag's behaviour exactly.
    if (z >= 0.999999) {
        outColor = vec4(texture(u_scene_color, uv).rgb, 1.0);
        return;
    }
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

    // Pre-shaded materials (5..8 = brush / voxel / water / raymarched
    // terrain) write LIT colour directly into scene_color in the
    // forward pass. G-buffer0 is R8G8B8A8_UNORM so its alpha=material_id
    // round-trips, but HDR colour clamps to [0,1] — sampling it here
    // would dim every bright pixel and break tonemap parity. Sample
    // scene_color instead so HDR is preserved bit-perfect.
    //
    // material_id 0 fall-through also reaches u_scene_color: that's
    // particles / decals / grass / viewmodel — anything drawn AFTER
    // cube.frag without G-buffer outputs. Those pixels need the forward
    // colour, not a re-shade of the geometry behind them.
    //
    // Albedo dot-product guard: alpha-discarded grass / cube.frag's
    // emissive early-return / depth-prepass rejected fragments all leave
    // gbuffer0 at the (0,0,0,0) clear. Without the dot() guard those
    // pixels classify as material_id 1 (cleared bytes round to mat 0
    // when alpha=0; but R8G8B8A8 quantisation can also produce
    // alpha=1/255 for some interpolated cases), and the lighting block
    // multiplies by zero albedo and emits true black. Treat near-zero
    // albedo as "no real gbuffer write here" and pass scene_color.
    if (material_id >= 5 || material_id == 0 ||
        dot(g0.rgb, vec3(1.0)) < 1e-4) {
        outColor = vec4(texture(u_scene_color, uv).rgb, 1.0);
        return;
    }

    vec3 cam_pos = scene.camera_pos.xyz;
    vec3 view_vec = cam_pos - world_pos;
    float cam_dist = length(view_vec);

    // ---------------------------------------------------------------
    //  Sun direct
    // ---------------------------------------------------------------
    // sun_direction is the direction FROM the surface TO the sun (UP for
    // an overhead sun), per cube.frag convention. Lambert ndl is then
    // dot(N, L), NOT dot(N, -L). The original deferred shader had the
    // sign flipped: ndl ended up negative on floor-facing pixels, the
    // double-sided fallback flipped N downward, and the subsequent
    // shadow rays fired AWAY from the sun (into the ground) — they
    // missed real occluders and shadow stayed at 0, leaking direct sun
    // into deep interior pixels. That was the dominant scene 3 brightness
    // gap (8.3% SAD, 50% over reference). Match cube.frag exactly.
    vec3 L = normalize(scene.sun_direction.xyz);
    float ndl_raw = dot(N, L);
    float ndl = max(ndl_raw, 0.0);

    // PCSS-style soft shadow. Mirrors cube.frag's:
    //   1. Blocker search: N rays in a wide cone, take avg t.
    //   2. Penumbra estimate proportional to avg_t × softness.
    //   3. Stratified shadow rays in the size-adapted cone.
    // Cull mask: terrain receivers use 0x02 (terrain-only casters) so
    // brushes/castle don't shadow the open plateau — cube.frag's same
    // rule. Brushes/dyn props use 0x01.
    uint shadow_mask = (material_id == 3) ? 0x02u : 0x01u;
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

        // 4-tap blocker search to match cube.frag's coverage (was 2-tap;
        // 2 missed the wall in cone-search through tight openings,
        // letting shadow stay at 0 and full sun leak through). Per-pixel
        // rotation matches cube.frag's seed_base + uvec3(0, 99u, 0u).
        float sum_t = 0.0;
        int hits = 0;
        float pp_phi = rand(seed_base + uvec3(0u, 99u, 0u)) * 6.28318530718;
        float pp_c = cos(pp_phi), pp_s = sin(pp_phi);
        for (int i = 0; i < 4; ++i) {
            float r1 = rand(seed_base + uvec3(uint(i), 91u, 13u));
            float r = sqrt(r1) * kBlockerCone;
            vec2 v = kVogel[i & 31];
            float vx = pp_c * v.x - pp_s * v.y;
            float vy = pp_s * v.x + pp_c * v.y;
            vec3 jitter = (vx * tan_u + vy * tan_v) * r;
            vec3 dir = normalize(L + jitter);
            float t;
            if (closest_hit_m(origin, dir, kBlockerTMax, shadow_mask, t)) {
                sum_t += t;
                ++hits;
            }
        }
        // Always sample shadow rays — even if the 2-tap blocker search
        // missed. Prior versions left shadow=0 in that case (treating
        // "no blocker = no shadow") but in tight interiors the blocker
        // cone occasionally threads a doorway and reports hits=0 even
        // though the wall geometry would block direct sun. Fall through
        // with a default penumbra so shadow sampling still fires; this
        // matches cube.frag's behaviour (it samples shadow_lr texture
        // unconditionally, never gated on a blocker hit).
        {
            float avg_t = (hits > 0) ? (sum_t / float(hits)) : 50.0;
            float kShadowTMax = clamp(avg_t * 4.0, 20.0, 200.0);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0,
                                  clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            float penumbra = clamp(base_softness * scale * 0.6,
                                   base_softness * 0.25,
                                   base_softness * 6.0);

            // Stratified shadow rays. cube.frag caps at 32 effective
            // samples (base_s); match that here so per-pixel variance
            // converges to forward's level. At 16 samples the per-pixel
            // shadow values were stuck around 0.83 instead of 1.0 in deep
            // interior, leaking direct sun and pumping scene 3 by ~50%.
            int N_s = max(1, min(int(scene.rt_flags.y), 32));
            int taken = 0;
            float blocked = 0.0;
            float pp_phi_s = rand(seed_base + uvec3(1u, 99u, 0u)) * 6.28318530718;
            float pp_c_s = cos(pp_phi_s), pp_s_s = sin(pp_phi_s);
            int strata = int(ceil(sqrt(float(N_s))));
            float inv_strata = 1.0 / float(strata);
            for (int sy = 0; sy < strata && taken < N_s; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s; ++sx) {
                    int idx = taken;
                    float r1 = rand(seed_base + uvec3(uint(idx), 11u, 47u));
                    float u1 = (float(sx) + r1) * inv_strata;
                    float r = sqrt(u1) * penumbra;
                    vec2 v = kVogel[idx & 31];
                    float vx = pp_c_s * v.x - pp_s_s * v.y;
                    float vy = pp_s_s * v.x + pp_c_s * v.y;
                    vec3 jitter = (vx * tan_u + vy * tan_v) * r;
                    vec3 dir = normalize(L + jitter);
                    if (any_hit_m(origin, dir, kShadowTMax, shadow_mask)) blocked += 1.0;
                    ++taken;
                }
            }
            shadow = (taken > 0) ? (blocked / float(taken)) : 0.0;
        }
        shadow *= clamp(scene.rt_params.w, 0.0, 1.0);
    }

    // Terrain receivers: sample the baked terrain shadow texture (covers
    // castle / static-brush shadows on the open plateau) and MAX with
    // the inline-RT result. cube.frag does the same — without this,
    // mask-0x02 RT shadows alone miss the castle and every terrain
    // pixel under the wall reads full sun. 5x5 Gaussian-ish PCF matches
    // cube.frag's blur exactly so the shadow softness lines up.
    if (material_id == 3) {
        const float kSide = 2048.0;
        vec2 uv_b = (world_pos.xz / kSide) + vec2(0.5);
        if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
            all(lessThanEqual(uv_b, vec2(1.0)))) {
            ivec2 sz_b = textureSize(u_terrain_shadow, 0);
            vec2 texel = 1.0 / vec2(sz_b);
            const float kBlurM = 4.0;
            float step_uv = (kBlurM / kSide) * 0.5;
            const float kW[25] = float[25](
                0.0608, 0.1738, 0.2466, 0.1738, 0.0608,
                0.1738, 0.4966, 0.7047, 0.4966, 0.1738,
                0.2466, 0.7047, 1.0000, 0.7047, 0.2466,
                0.1738, 0.4966, 0.7047, 0.4966, 0.1738,
                0.0608, 0.1738, 0.2466, 0.1738, 0.0608);
            const float kWsum = 6.6172;
            float sh_bake = 0.0;
            for (int j = 0; j < 5; ++j) {
                for (int i = 0; i < 5; ++i) {
                    vec2 off2 = vec2(float(i - 2), float(j - 2)) * step_uv;
                    sh_bake += texture(u_terrain_shadow, uv_b + off2).r *
                               kW[j * 5 + i];
                }
            }
            sh_bake /= kWsum;
            sh_bake *= clamp(scene.rt_params.w, 0.0, 1.0);
            shadow = max(shadow, sh_bake);
        }
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

    // Terrain sky-vis probe (cube.frag lines 3010-3030). For terrain
    // receivers within 50m of the camera, fire 4 rays in a cosine
    // hemisphere with mask 0xFD (everything EXCEPT terrain itself —
    // hits castle / brushes but not terrain) and count misses. Without
    // this, terrain near the castle wall renders with full sky-fill
    // ambient even though the castle blocks half the sky dome — scene 2
    // terrain pixels came out ~17% brighter than forward.
    if (material_id == 3 && scene.rt_flags.x != 0 &&
        gi_strength > 1e-4 && cam_dist < 50.0) {
        const int kProbeN = 4;
        const uint kProbeMask = 0xFDu;
        vec3 probe_origin = world_pos + vec3(0.0, 0.10, 0.0);
        int probe_misses = 0;
        for (int i = 0; i < kProbeN; ++i) {
            float r1 = rand(seed_base + uvec3(uint(i), 41u, 67u));
            float r2 = rand(seed_base + uvec3(uint(i), 13u, 89u));
            float r_h = sqrt(r1);
            float phi = 6.28318530718 * r2;
            vec3 dir = vec3(r_h * cos(phi),
                            sqrt(max(0.0, 1.0 - r1)),
                            r_h * sin(phi));
            if (!any_hit_m(probe_origin, dir, 60.0, kProbeMask)) {
                probe_misses += 1;
            }
        }
        sky_vis = float(probe_misses) / float(kProbeN);
    }
    // sky_vis must be measured even when gi_strength is 0, otherwise
    // the ambient block above receives sky_vis=1.0 (assumed open sky)
    // and pumps full ambient_sky into deep interior pixels — castle
    // interior gets ~50% brighter than forward. Run a short 4-ray
    // visibility probe whenever N_gi > 0; gi_strength still gates the
    // indirect-light accumulation.
    if (N_gi > 0) {
        // Distance-LOD scaling — match cube.frag. Close pixels get the
        // full slider, far pixels collapse toward 1. Cap at 16 in the
        // deferred path so RT bandwidth doesn't explode (cube.frag also
        // caps internally via the half-rate parity trick).
        int taken_gi = min(lod_samples(N_gi, cam_dist), 16);
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
                // hit gets sun + ambient; otherwise just ambient. Sun
                // contribution scaled by 0.30 — forward's ReSTIR with
                // SVGF denoise + temporal reuse converges to a much
                // smaller effective sun-bounce because it importance-
                // samples and averages over many frames. Naively giving
                // each ray the full bounce blows interior brightness up
                // ~30-50% (visible on scene 3, the castle lintel).
                vec3 hit_light = scene.ambient.rgb * scene.ambient.a * 0.6;
                float n_dot_sun = max(dot(hit_n, L), 0.0);
                if (n_dot_sun > 0.0 &&
                    !any_hit(hit_pos + hit_n * 0.01, L, 200.0)) {
                    hit_light += scene.sun_color.rgb *
                                  scene.sun_color.a * n_dot_sun * 0.30;
                }
                // Mid-grey albedo proxy for the bounce hit.
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
    //  Ambient + sky bounce — mirrors cube.frag's ambient_ground +
    //  ambient_sky split. ambient_ground is the "windowless room
    //  reflection" constant that AO can darken but never fully kills;
    //  ambient_sky is the open-sky contribution that fades when sky_vis
    //  drops (deep crevices). Without the split, per-pixel PCSS
    //  collapsing to shadow=1 + RTAO collapsing to ao=0 zeroes BOTH
    //  ambient and sun → pure black pixels (the scattered-black bug).
    // ---------------------------------------------------------------
    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky_tint = scene.sky_color.rgb;
    float ambient_strength = clamp(scene.rt_params.z, 0.0, 1.0);
    // ambient.a is `terrain_ao_punch` on the C++ side (descriptors.cpp line
    // 341 packs it there), NOT a strength multiplier. cube.frag does
    // `ambient.rgb * rt_params.z` only — multiplying by `ambient.a` here
    // (as the earlier port did) zeroed the entire ambient term whenever
    // terrain_ao_punch was 0, producing literal black on castle walls.
    vec3 ambient_ground = scene.ambient.rgb * ambient_strength;
    vec3 ambient_sky    = sky_tint * 0.45 * ambient_strength;
    // sky_factor: cube.frag uses smoothstep(0, 0.6, sky_vis) floored at
    // 0.10. Same here so deep cracks still pick up a tiny sky leakage.
    float sky_factor   = mix(0.10, 1.0, smoothstep(0.0, 0.6, sky_vis));
    // Non-terrain receivers attenuate the ground term too (else interior
    // brick reads as bright as exterior), but with a 0.25 floor so we
    // never collapse to true black.
    float ground_atten = (material_id == 3) ? 1.0
                                            : mix(0.25, 1.0, sky_factor);
    vec3 ambient_combined = mix(ambient_ground * ground_atten,
                                 ambient_sky    * sky_factor, up);
    // AO multiplies the combined term. 0.08 floor matches the residual
    // brightness forward shows in deeply-shadowed pixels (tiny GI bounce
    // + temporal-smoothed RTAO). At 0.03 the scene 7 crate-corner pixels
    // came out literal (0,0,0) where forward had ~(20,20,20).
    float ao_floored = mix(0.08, 1.0, ao);
    vec3 ambient_term = albedo * ambient_combined * ao_floored;
    // sky_fill drops out as a separate term — it's been folded into
    // ambient_sky above. Zero it so the addition below stays valid.
    vec3 sky_fill = vec3(0.0);

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

    // ---------------------------------------------------------------
    //  Atmospheric distance fog (verbatim port of cube.frag's exp²
    //  fog block with Henyey-Greenstein Mie scattering forward halo).
    //  Master gate on dfog_col.a so the menu's fog-strength slider
    //  controls it the same way for both renderers.
    // ---------------------------------------------------------------
    {
        vec4 dfog_col = scene.distance_fog_color;
        vec4 dfog_par = scene.distance_fog_params;
        vec3 view_dir = normalize(world_pos - cam_pos);
        float fog_t   = 0.0;
        vec3  fog_rgb = dfog_col.rgb;
        if (dfog_col.a > 1e-3) {
            float density = max(0.0, dfog_par.x);
            float start_d = max(0.0, dfog_par.y);
            float h_top   = dfog_par.z;
            float max_a   = clamp(dfog_par.w, 0.0, 1.0);
            float d_raw   = max(0.0, cam_dist - start_d);
            float dx      = d_raw * density;
            float fog     = 1.0 - exp(-dx * dx);
            if (h_top > 0.5) {
                float h_w = 1.0 - smoothstep(0.0, h_top, world_pos.y);
                fog *= h_w;
            }
            fog_t = clamp(fog * dfog_col.a, 0.0, max_a);
            vec3  Lf  = scene.sun_direction.xyz;
            float mu  = clamp(dot(view_dir, -Lf), -1.0, 1.0);
            const float g  = 0.76;
            const float g2 = g * g;
            float hg_d = max(0.0001, 1.0 + g2 - 2.0 * g * mu);
            float hg   = (1.0 - g2) / (12.566 * hg_d * sqrt(hg_d));
            float mie  = clamp(hg * 0.40, 0.0, 1.5);
            vec3  sun_tint = scene.sun_color.rgb * scene.sun_color.a;
            fog_rgb = mix(dfog_col.rgb,
                          dfog_col.rgb * 0.7 + sun_tint * 0.55,
                          clamp(mie, 0.0, 1.0));
        }
        if (fog_t > 1e-4) lit = mix(lit, fog_rgb, fog_t);
    }

    outColor = vec4(lit, 1.0);
}
