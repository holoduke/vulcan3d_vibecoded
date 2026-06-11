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
    // terrain_params.y = wrap_strength (sky-tint lift on back-facing
    // terrain pixels РІР‚вЂќ atmospheric scattering bounce onto shadow side).
    // Other channels: .x fog_strength (unused here, fog has its own
    // params below), .z detail_strength, .w shadow_softness_scale.
    vec4  terrain_params;
    // Remaining padding to reach the fog params at the same byte offsets
    // cube.frag uses. 26 vec4 = terrain_h_low/high, grass_extra(x2),
    // light_vp (4 vec4), terrain_extra, _scene_pad[0..16].
    vec4  _pad_a[26];
    // _scene_pad[17] in cube.frag РІР‚вЂќ base fog tint .rgb + master strength .a.
    vec4  distance_fog_color;
    // _scene_pad[18] in cube.frag РІР‚вЂќ .x density, .y start, .z height_top,
    // .w max alpha.
    vec4  distance_fog_params;
} scene;

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform sampler2D u_gbuffer0;
layout(set = 0, binding = 3) uniform sampler2D u_gbuffer1;
layout(set = 0, binding = 4) uniform sampler2D u_depth;
layout(set = 0, binding = 5) uniform sampler2D u_sun_shadow;
// scene_color from the forward pass РІР‚вЂќ used at sky pixels (depth=1) and
// at pixels that aren't covered by the G-buffer write (particles,
// decals, projectiles, anything drawn after cube.frag without a
// G-buffer output). Lets the deferred path pass them through unmodified
// instead of overwriting with reconstructed sky.
layout(set = 0, binding = 6) uniform sampler2D u_scene_color;
// Baked terrain shadow texture РІР‚вЂќ cube.frag uses this to put castle
// shadows on the terrain plateau. It's a 2048m-wide top-down texture
// of the shadow factor sampled at vWorldPos.xz / 2048 + 0.5. Only
// sampled when material_id == 3 (terrain receiver).
layout(set = 0, binding = 7) uniform sampler2D u_terrain_shadow;
// Raw GI image cube.frag writes (material-aware multi-bounce path-trace
// via the materials buffer + voxel marcher). Sampling it lets the
// deferred shader use a HIGHER-QUALITY GI than its own 4-ray loop can
// produce. The image is in VK_IMAGE_LAYOUT_GENERAL (storage layout)
// when cube.frag runs imageStore; sampler2D reads from GENERAL are
// permitted by the spec for STORAGE+SAMPLED images.
layout(set = 0, binding = 8) uniform sampler2D u_svgf_gi;

// GI temporal-denoise history ping-pong (AAA GI denoiser). .rgb =
// accumulated raw GI irradiance (pre-gi_strength), .a = sample count M.
// Frame parity picks read/write slots; the 2-frame fence overlap makes
// per-frame barriers unnecessary (same contract as cube.frag's pair).
layout(set = 0, binding = 9,  rgba16f) uniform image2D u_gi_hist0;
layout(set = 0, binding = 10, rgba16f) uniform image2D u_gi_hist1;

layout(push_constant) uniform PC {
    mat4 inv_vp;
    // Previous frame's view-proj РІР‚вЂќ reprojects world_pos into last
    // frame's screen for the GI history fetch.
    mat4 prev_vp;
    // Per-frame point lights. .xyz = world position, .w = radius (m).
    // Per-light colour comes in light_colors with .a = intensity. Up to
    // 4 active lights РІР‚вЂќ overflow lights drop. Inactive slots set radius
    // to 0 so the early-out gate culls them at no cost.
    vec4 light_pos[4];
    vec4 light_col[4];
} pc;

// Procedural sky РІР‚вЂќ verbatim port of cube.frag's sample_sky so deferred
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

// Octahedral normal decode РІР‚вЂќ matches cube.frag's encode.
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

// Closest-hit with t-value РІР‚вЂќ used by the PCSS blocker search.
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

// 32-tap Vogel disk РІР‚вЂќ same one cube.frag uses for PCSS sample placement.
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

// Per-pixel deterministic noise РІР‚вЂќ matches cube.frag's ign+rand pair so
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
// Distance-LOD sample reducer РІР‚вЂќ same as cube.frag's lod_samples. Close
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
    // seed_base РІР‚вЂќ matches cube.frag exactly (FragCoord int + frame & 7
    // in z slot). All shadow + GI + AO ray jitters derive from this
    // seed via rand() so deferred lighting picks the same Vogel disk
    // rotations cube.frag picks at the same pixel.
    uvec3 seed_base = uvec3(uint(gl_FragCoord.x),
                            uint(gl_FragCoord.y),
                            uint(scene.rt_flags.w) & 7u);
    // Frame-STABLE seed for shadows + AO (no temporal accumulator backs
    // them, so per-frame seed cycling reads as flicker РІР‚вЂќ same split as
    // cube.frag). GI keeps frame-varying seeds; its accumulator below
    // needs them to converge.
    uvec3 seed_stable = uvec3(uint(gl_FragCoord.x),
                              uint(gl_FragCoord.y), 0u);
    float z = texture(u_depth, uv).r;
    // World-pos / view-direction reconstruction from depth + inverse-VP.
    vec2 ndc = uv * 2.0 - 1.0;
    // Sky / depth-cleared pixels РІР‚вЂќ forward's compose pass samples an
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
    // round-trips, but HDR colour clamps to [0,1] РІР‚вЂќ sampling it here
    // would dim every bright pixel and break tonemap parity. Sample
    // scene_color instead so HDR is preserved bit-perfect.
    //
    // material_id 0 fall-through also reaches u_scene_color: that's
    // particles / decals / grass / viewmodel РІР‚вЂќ anything drawn AFTER
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
    // shadow rays fired AWAY from the sun (into the ground) РІР‚вЂќ they
    // missed real occluders and shadow stayed at 0, leaking direct sun
    // into deep interior pixels. That was the dominant scene 3 brightness
    // gap (8.3% SAD, 50% over reference). Match cube.frag exactly.
    vec3 L = normalize(scene.sun_direction.xyz);
    float ndl_raw = dot(N, L);
    float ndl = max(ndl_raw, 0.0);

    // PCSS-style soft shadow. Mirrors cube.frag's:
    //   1. Blocker search: N rays in a wide cone, take avg t.
    //   2. Penumbra estimate proportional to avg_t Р“вЂ” softness.
    //   3. Stratified shadow rays in the size-adapted cone.
    // Cull mask: terrain receivers use 0x02 (terrain-only casters) so
    // brushes/castle don't shadow the open plateau РІР‚вЂќ cube.frag's same
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

        // Terrain (material 3) needs an aggressive far-distance bias to
        // clear the rasterised-LOD-vs-BLAS height gap on distant ridges.
        // Same formula cube.frag's forward terrain path uses РІР‚вЂќ without
        // it, kilometre-range mountain ridges false-hit their own BLAS
        // detail and the deferred path reproduces the "dark leaves"
        // splatter the forward fix already addressed. Brushes keep the
        // tight base bias (a few cm) so close-shadow precision survives.
        float bias;
        if (material_id == 3) {
            float far_t = max((cam_dist - 60.0) / 200.0, 0.0);
            float far_bias = min(far_t * far_t * 6.0, 32.0);
            bias = 0.05 + 0.10 * (1.0 - ndl_raw) + far_bias;
        } else {
            bias = 0.005 + 0.02 * (1.0 - ndl_raw);
        }
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
        float pp_phi = rand(seed_stable + uvec3(0u, 99u, 0u)) * 6.28318530718;
        float pp_c = cos(pp_phi), pp_s = sin(pp_phi);
        for (int i = 0; i < 4; ++i) {
            float r1 = rand(seed_stable + uvec3(uint(i), 91u, 13u));
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
        // Always sample shadow rays РІР‚вЂќ even if the 2-tap blocker search
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

            // Stratified shadow rays РІР‚вЂќ cube.frag's half-rate PCSS:
            // each frame fires only HALF the ray budget alternating via
            // a per-pixel checkerboard + frame parity, then TAA's history
            // blend accumulates the missing samples over two frames so
            // steady-state quality matches full-rate. Without the
            // checkerboard, my deferred fires double the rays cube.frag
            // does per frame, producing slightly different per-pixel
            // shadow values at edges РІР‚вЂќ same average over 2 frames but
            // different at any single frame readback.
            int N_s = max(1, min(int(scene.rt_flags.y), 32));
            ivec2 ip2     = ivec2(gl_FragCoord.xy);
            int   parity  = ((ip2.x + ip2.y) ^ int(scene.rt_flags.w)) & 1;
            int   N_s_eff = max(1, (N_s + (parity == 0 ? 1 : 0)) / 2);
            N_s = N_s_eff;
            int taken = 0;
            float blocked = 0.0;
            float pp_phi_s = rand(seed_stable + uvec3(1u, 99u, 0u)) * 6.28318530718;
            float pp_c_s = cos(pp_phi_s), pp_s_s = sin(pp_phi_s);
            int strata = int(ceil(sqrt(float(N_s))));
            float inv_strata = 1.0 / float(strata);
            for (int sy = 0; sy < strata && taken < N_s; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s; ++sx) {
                    int idx = taken;
                    float r1 = rand(seed_stable + uvec3(uint(idx), 11u, 47u));
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
    // the inline-RT result. cube.frag does the same РІР‚вЂќ without this,
    // mask-0x02 RT shadows alone miss the castle and every terrain
    // pixel under the wall reads full sun. 5x5 Gaussian-ish PCF matches
    // cube.frag's blur exactly so the shadow softness lines up.
    if (material_id == 3) {
        const float kSide = 2048.0;
        vec2 uv_b = (world_pos.xz / kSide) + vec2(0.5);
        if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
            all(lessThanEqual(uv_b, vec2(1.0)))) {
            // PCSS on the baked map. G stores the blocker distance along
            // the sun ray (t / 400 m). Blocker search first (3x3, 3 m
            // spacing), then a 5x5 PCF whose WORLD radius grows with the
            // average blocker distance: a tower top 60 m up casts a wide
            // soft penumbra (physically: penumbra ~ occluder distance x
            // sun angular size), a low wall stays tight and dark at its
            // base. The old constant-radius blur inverted this - big
            // shadows from tall structures pegged at full dark while
            // thin low-wall bands were blurred lighter.
            float cover = 0.0;
            float dsum  = 0.0;
            const float kSearchUv = 3.0 / kSide;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    vec2 sv = texture(u_terrain_shadow,
                                      uv_b + vec2(float(i), float(j)) *
                                             kSearchUv).rg;
                    cover += sv.r;
                    dsum  += sv.g * sv.r;
                }
            }
            float sh_bake = 0.0;
            if (cover > 1e-4) {
                float avg_d = (dsum / cover) * 400.0;   // metres to blocker
                // ~6 cm of penumbra per metre of blocker distance
                // (exaggerated sun size for readability), clamped so
                // near-base shadow keeps a crisp >=0.6 m edge and far
                // tower shadows top out at an 8 m half-width.
                float half_w = clamp(avg_d * 0.06, 0.6, 8.0);
                float step_uv = (half_w * 0.5) / kSide;  // kernel = +-2 taps
                const float kW[25] = float[25](
                    0.0608, 0.1738, 0.2466, 0.1738, 0.0608,
                    0.1738, 0.4966, 0.7047, 0.4966, 0.1738,
                    0.2466, 0.7047, 1.0000, 0.7047, 0.2466,
                    0.1738, 0.4966, 0.7047, 0.4966, 0.1738,
                    0.0608, 0.1738, 0.2466, 0.1738, 0.0608);
                const float kWsum = 6.6172;
                for (int j = 0; j < 5; ++j) {
                    for (int i = 0; i < 5; ++i) {
                        vec2 off2 = vec2(float(i - 2), float(j - 2)) * step_uv;
                        sh_bake += texture(u_terrain_shadow, uv_b + off2).r *
                                   kW[j * 5 + i];
                    }
                }
                sh_bake /= kWsum;
                // Far-blocker shadows also lose a little peak density:
                // a real sun-disk umbra collapses once the occluder is
                // distant, and open ground under a tower still sees the
                // whole sky. 100 m blocker -> ~18% lighter core.
                sh_bake *= 1.0 - 0.18 * clamp(avg_d / 100.0, 0.0, 1.0);
                sh_bake *= clamp(scene.rt_params.w, 0.0, 1.0);
            }
            shadow = max(shadow, sh_bake);
        }
    }

    // ---------------------------------------------------------------
    //  Ambient occlusion (short-ray RT, modes off / fast / full / HBAO)
    // ---------------------------------------------------------------
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    // cube.frag gates AO on BOTH ao_mode AND rt_flags.z (ao_samples).
    // With ao_samples=0 in the user's settings forward skips the AO ray
    // queries entirely and keeps ao=1.0. My deferred was firing them
    // anyway, producing a 0.4-0.7 ao multiplier that darkened ambient
    // ~50% in indoor pixels РІР‚вЂќ visible as deferred-brighter elsewhere
    // because the LACK of darkening was the issue. Wait, opposite: my
    // AO was darkening pixels forward DOESN'T darken, making deferred
    // dimmer. ...let me just match the gate.
    if (ao_mode > 0 && scene.rt_flags.z > 0) {
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
    //  Path-traced GI (Phase 4b РІР‚вЂќ simplified port of cube.frag's loop)
    //
    //  Fires N cosine-weighted hemisphere rays around N. Each ray
    //  closest-hits the TLAS; on a miss it accumulates the sky tint.
    //  Hit albedo uses a mid-grey approximation (the deferred path
    //  doesn't carry the materials buffer; the small albedo error gets
    //  absorbed by the sky_vis blend and the per-pixel hash).
    //
    //  sky_vis tracks the fraction of first-bounce rays that escape to
    //  sky РІР‚вЂќ used below to attenuate the ambient/sky_fill terms so
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
    // hemisphere with mask 0xFD (everything EXCEPT terrain itself РІР‚вЂќ
    // hits castle / brushes but not terrain) and count misses. Without
    // this, terrain near the castle wall renders with full sky-fill
    // ambient even though the castle blocks half the sky dome РІР‚вЂќ scene 2
    // terrain pixels came out ~17% brighter than forward.
    if (material_id == 3 && scene.rt_flags.x != 0 &&
        gi_strength > 1e-4 && cam_dist < 50.0) {
        const int kProbeN = 4;
        const uint kProbeMask = 0xFDu;
        vec3 probe_origin = world_pos + vec3(0.0, 0.10, 0.0);
        int probe_misses = 0;
        for (int i = 0; i < kProbeN; ++i) {
            float r1 = rand(seed_stable + uvec3(uint(i), 41u, 67u));
            float r2 = rand(seed_stable + uvec3(uint(i), 13u, 89u));
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
    // and pumps full ambient_sky into deep interior pixels РІР‚вЂќ castle
    // interior gets ~50% brighter than forward. Run a short visibility
    // probe whenever N_gi > 0; gi_strength still gates the indirect-
    // light accumulation.
    //
    // The screen-space u_svgf_gi override is gone (it leaked other
    // surfaces' light into walls РІР‚вЂќ see below), so the inner bounce
    // lighting now accumulates for EVERY material.
    bool skip_gi_inner = false;
    if (N_gi > 0) {
        // Distance-LOD scaling РІР‚вЂќ match cube.frag. Close pixels get the
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
        // Frame-cycled seed offset: the temporal accumulator below needs
        // DIFFERENT ray directions every frame to converge to the true
        // irradiance РІР‚вЂќ a frame-stable seed would just freeze the grain.
        vec2 fseed = vec2(float(scene.rt_flags.w & 31)) * vec2(17.13, 9.71);
        for (int i = 0; i < taken_gi; ++i) {
            float r1 = hash12(gl_FragCoord.xy + fseed + vec2(i * 13.0, 73.0));
            float r2 = hash12(gl_FragCoord.xy + fseed + vec2(i * 13.0, 47.0));
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
                if (!skip_gi_inner) {
                    // Sun-shadow at the hit point: one any-hit ray; if
                    // clear, hit gets sun + ambient; otherwise just
                    // ambient. Mid-grey 0.5 albedo proxy for the bounce.
                    vec3 hit_light = scene.ambient.rgb * 0.6;
                    float n_dot_sun = max(dot(hit_n, L), 0.0);
                    if (n_dot_sun > 0.0 &&
                        !any_hit(hit_pos + hit_n * 0.01, L, 200.0)) {
                        hit_light += scene.sun_color.rgb *
                                      scene.sun_color.a * n_dot_sun;
                    }
                    gi_indirect += vec3(0.5) * hit_light;
                }
            } else {
                // Miss РІвЂ вЂ™ sky. Use sky_color tinted by direction.up so
                // overhead rays brighten more than horizon ones. Skip
                // the accumulation when svgf_gi will overwrite below.
                if (!skip_gi_inner) {
                    float up_w = clamp(dir.y, 0.0, 1.0);
                    gi_indirect += scene.sky_color.rgb * (0.4 + 0.6 * up_w);
                }
                ++sky_hits;
            }
        }
        gi_indirect /= float(taken_gi);
        // The u_svgf_gi screen-space override that used to replace the
        // ray-loop result for non-terrain pixels is REMOVED. That image
        // is written by cube.frag with imageStore from fragments running
        // LATE-Z (the wall pipeline writes gl_FragDepth, so occluded
        // fragments still execute their stores), and the deferred read
        // happened at a plain screen UV РІР‚вЂќ geometry behind a wall could
        // stamp its irradiance into the wall's pixel and the wall then
        // shaded with the OTHER surface's light. Users bisected this as
        // "all walls have some degree of transparency, appears when
        // gi_samples >= 1". The in-shader ray loop is per-pixel,
        // image-free, and immune to write-order races.
        //
        // ---- Temporal GI denoiser ----
        // Accumulate the raw irradiance across frames (EMA up to M=28,
        // РІвЂ°в‚¬ effective 28Р“вЂ” the per-frame ray count) with in-shader
        // reprojection through prev_vp and per-channel disocclusion
        // rejection РІР‚вЂќ the AAA pattern: denoise the SIGNAL, never the
        // composed image.
        {
            ivec2 pix = ivec2(gl_FragCoord.xy);
            bool parity0 = (scene.rt_flags.w & 1) == 0;
            vec3  prevRgb = gi_indirect;
            float prevM   = 0.0;
            // Motion factor: 0 = camera static (history fully trusted,
            // no flicker), 1 = fast reprojection (strict rejection, no
            // ghost light "shining through" walls from stale interior
            // history). 0.75 px dead-zone absorbs the TAA jitter.
            float mv = 1.0;
            vec4 pclip = pc.prev_vp * vec4(world_pos, 1.0);
            if (pclip.w > 1e-3) {
                vec2 puv = (pclip.xy / pclip.w) * 0.5 + 0.5;
                if (all(greaterThan(puv, vec2(0.0))) &&
                    all(lessThan(puv, vec2(1.0)))) {
                    vec2 cuv = (vec2(pix) + 0.5) / scene.viewport.xy;
                    float motion_px =
                        length((puv - cuv) * scene.viewport.xy);
                    mv = clamp((motion_px - 0.75) / 4.0, 0.0, 1.0);
                    ivec2 ppx = clamp(ivec2(puv * scene.viewport.xy),
                                      ivec2(0),
                                      ivec2(scene.viewport.xy) - ivec2(1));
                    // ---- Edge-aware spatial gather on the history ----
                    // 5 taps (centre + plus-pattern at stride 2),
                    // weighted by depth + normal similarity from the
                    // CURRENT gbuffer. Because the gathered result is
                    // EMA-d back into history every frame, the effective
                    // kernel grows over time (progressive a-trous
                    // diffusion) while the edge weights keep light from
                    // crossing geometry - the spatial half of an
                    // SVGF-class denoiser without extra passes.
                    ivec2 vmax = ivec2(scene.viewport.xy) - ivec2(1);
                    const ivec2 kTap[5] = ivec2[5](
                        ivec2(0, 0), ivec2( 2, 0), ivec2(-2, 0),
                        ivec2(0, 2), ivec2( 0,-2));
                    vec3  acc = vec3(0.0);
                    float wsum = 0.0;
                    float Mc = 0.0;
                    for (int ti = 0; ti < 5; ++ti) {
                        ivec2 tp = clamp(ppx + kTap[ti], ivec2(0), vmax);
                        vec4 hv = parity0 ? imageLoad(u_gi_hist1, tp)
                                          : imageLoad(u_gi_hist0, tp);
                        // Sanitise: NaN/Inf or out-of-range history must
                        // not enter the EMA (NaN survives comparisons).
                        if (any(isnan(hv.rgb)) || any(isinf(hv.rgb)) ||
                            !(hv.a >= 0.0 && hv.a <= 32.0) ||
                            any(greaterThan(hv.rgb, vec3(64.0)))) continue;
                        vec2 tuv = (vec2(tp) + 0.5) / scene.viewport.xy;
                        float tz = texture(u_depth, tuv).r;
                        vec3  tn = octa_decode(texture(u_gbuffer1, tuv).rg);
                        float wz = exp(-abs(tz - z) * 800.0);
                        float wn = pow(max(dot(tn, N), 0.0), 16.0);
                        float w_t = (ti == 0 ? 1.0 : 0.6) * wz * wn;
                        // Centre-tap floor (static camera only):
                        // normal-mapped surfaces can fail the tight wn
                        // test on ALL taps, zeroing wsum -> silent
                        // history reset -> static flicker. Under motion
                        // the floor is removed so mismatched-surface
                        // history is rejected instead of ghosting.
                        if (ti == 0) w_t = max(w_t, 0.05 * (1.0 - mv));
                        acc  += hv.rgb * w_t;
                        wsum += w_t;
                        if (ti == 0) Mc = hv.a;
                    }
                    if (wsum > 1e-3) {
                        prevRgb = acc / wsum;
                        prevM   = Mc;
                    }
                }
            }
            // Per-channel disocclusion: history that can't plausibly
            // belong to this surface (e.g. sky-blue irradiance on a
            // beige wall РІР‚вЂќ equal LUMINANCE would fool a scalar test)
            // restarts accumulation; moderate mismatch shortens it.
            // Firefly clamp: once history is converged (M >= 8), a single
            // outlier ray bundle must not flash the pixel. Cap the fresh
            // estimate at 3x history + eps BEFORE the ratio test, so noise
            // spikes neither display nor trigger a history reset.
            if (prevM >= 8.0 && mv < 0.5) {
                gi_indirect = min(gi_indirect, prevRgb * 3.0 + vec3(0.05));
            }
            vec3 rcn = (gi_indirect + vec3(0.05)) / (prevRgb + vec3(0.05));
            float ratio = max(max(max(rcn.r, 1.0 / rcn.r),
                                  max(rcn.g, 1.0 / rcn.g)),
                              max(rcn.b, 1.0 / rcn.b));
            // Motion-adaptive rejection. Static camera: deliberately
            // loose tiers (an 8-ray estimate routinely lands 2-3x off
            // the converged mean; clamping M hard on every such frame
            // held the accumulator at ~4 samples = visible flicker).
            // Moving camera: strict tiers + short history, so stale
            // interior light can't ghost onto walls/doors as fake
            // transparency while the view changes.
            float thr_hi = mix(8.0, 3.0, mv);
            float thr_lo = mix(3.5, 2.0, mv);
            float m_lo   = mix(12.0, 3.0, mv);
            if (ratio > thr_hi)      { prevRgb = gi_indirect; prevM = 0.0; }
            else if (ratio > thr_lo) { prevM = min(prevM, m_lo); }
            float Mnew = min(prevM + 1.0, mix(32.0, 14.0, mv));
            gi_indirect = mix(gi_indirect, prevRgb, (Mnew - 1.0) / Mnew);
            vec4 storeVal = vec4(gi_indirect, Mnew);
            if (parity0) imageStore(u_gi_hist0, pix, storeVal);
            else         imageStore(u_gi_hist1, pix, storeVal);
        }
        gi_indirect *= gi_strength;
        // sky_vis: fraction of rays that escaped to sky. 0 = fully
        // enclosed РІвЂ вЂ™ ambient/sky_fill suppressed; 1 = open sky РІвЂ вЂ™ full.
        sky_vis = float(sky_hits) / float(taken_gi);
    }

    // ---------------------------------------------------------------
    //  Ambient + sky bounce РІР‚вЂќ mirrors cube.frag's ambient_ground +
    //  ambient_sky split. ambient_ground is the "windowless room
    //  reflection" constant that AO can darken but never fully kills;
    //  ambient_sky is the open-sky contribution that fades when sky_vis
    //  drops (deep crevices). Without the split, per-pixel PCSS
    //  collapsing to shadow=1 + RTAO collapsing to ao=0 zeroes BOTH
    //  ambient and sun РІвЂ вЂ™ pure black pixels (the scattered-black bug).
    // ---------------------------------------------------------------
    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    // Warm-bias the ambient sky tint (12 % sun colour). See the matching
    // block in cube.frag РІР‚вЂќ the raw saturated Rayleigh blue reads as
    // steel-blue on pure-shadow surfaces; bounce light off warm
    // geometry tonally lifts real shadows toward neutral.
    vec3 sky_tint = mix(scene.sky_color.rgb, scene.sun_color.rgb, 0.12);
    float ambient_strength = clamp(scene.rt_params.z, 0.0, 1.0);
    // ambient.a is `terrain_ao_punch` on the C++ side (descriptors.cpp line
    // 341 packs it there), NOT a strength multiplier. cube.frag does
    // `ambient.rgb * rt_params.z` only РІР‚вЂќ multiplying by `ambient.a` here
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
    // + temporal-smoothed RTAO). Empirical sweep: removing the floor
    // regresses all scenes 0.03-0.05% mean SAD even though cube.frag has
    // no floor of its own РІР‚вЂќ likely because forward's SVGF temporal
    // smoothing averages the per-pixel collapses out where mine can't.
    float ao_floored = mix(0.08, 1.0, ao);
    vec3 ambient_term = albedo * ambient_combined * ao_floored;
    // sky_fill drops out as a separate term РІР‚вЂќ it's been folded into
    // ambient_sky above. Zero it so the addition below stays valid.
    vec3 sky_fill = vec3(0.0);

    // Sun term.
    vec3 sun_term = albedo * scene.sun_color.rgb * scene.sun_color.a *
                    ndl * (1.0 - shadow);

    // ---------------------------------------------------------------
    //  Material-id tweaks
    // ---------------------------------------------------------------
    // Per-material tweaks (Phase 5):
    //   2 (wood)    РІР‚вЂќ no change (default Lambert reads right)
    //   3 (terrain) РІР‚вЂќ boosted ambient to match the open-sky shading
    //                 cube.frag does on terrain pixels
    //   4 (grass)   РІР‚вЂќ extra wrap-around fill so shadowed blades don't
    //                 read pitch-black
    //   5 (emissive) РІР‚вЂќ pass albedo through unchanged (brushes already
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
    //  Point lights РІР‚вЂќ variable list driven by push constant. Muzzle
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

    // Terrain sky-bounce wrap (cube.frag lines 3322-3331). Back-facing
    // terrain pixels (-NР’В·L > 0) pick up a sky-tinted lift to model
    // atmospheric scattering bouncing onto the shadow side of slopes.
    // Without this, the back of distant ridges in scene 5 / 7 reads as
    // a flat dark patch where forward has a 30% sky-tint contribution.
    if (material_id == 3) {
        float wrap_amt = clamp(scene.terrain_params.y, 0.0, 1.0);
        float back     = max(0.0, -dot(N, L));
        lit += albedo * scene.sky_color.rgb * back * 0.30 * wrap_amt;
    }

    // ---------------------------------------------------------------
    //  Atmospheric distance fog (verbatim port of cube.frag's expР’Р†
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

