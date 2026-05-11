#version 460
#extension GL_EXT_ray_query : require

// Procedural FBM heightfield ray-marched terrain вЂ” selectable alternate
// to the chunked rasterised terrain. Implements the technique from
// MiniMax-AI/skills shader-dev terrain-rendering reference: 2D Value
// Noise with analytical derivatives, FBM with derivative-erosion
// suppression, adaptive-step heightfield ray march, soft shadows,
// height/slope material blend, atmospheric extinction.
//
// Runs as a fullscreen triangle AFTER the rasterised cubes / castle /
// dyn-props have written depth. gl_FragDepth at the ray hit lets the
// rasterised geometry occlude correctly. Misses (sky) discard so the
// compose pass's sky still shows through.

layout(location = 0) in vec2 vNDC;
layout(location = 1) in vec4 vWNear;
layout(location = 2) in vec4 vWFar;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

// Same push-constant layout as cube.frag's PushConstants. We use
//   mvp           вЂ” view_proj for gl_FragDepth at the hit point
//   model         вЂ” inverse(view_proj) for world-ray reconstruction
//   color.xy      вЂ” plateau centre (world XZ)
//   color.z       вЂ” plateau half-extent (m)
//   color.w       вЂ” plateau height
//   emissive.x    вЂ” march step factor (0.4..0.8)
//   tex_params.x  вЂ” march step count
//   tex_params.y  вЂ” shadow ray steps
//   tex_params.z  вЂ” march FBM octaves
//   tex_params.w  вЂ” normal FBM octaves
//   grass_params.x вЂ” fog strength multiplier (0 = off)
//   grass_params.y вЂ” relaxation flag (>0.5 = use relaxation step)
//   grass_params.z вЂ” fog god-ray flag (>0.5 = self-shadowed fog)
// The remaining fields are unused by this shader.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

// Heightmap delta texture (R32F). Stores `current_height -
// procedural_baseline_height` per cell, so sculpt edits + plateau
// noise + disk-loaded overlays appear on top of the shader's GLSL
// FBM. Uploaded once at level load and re-uploaded after sculpt /
// noise edits. World XZ в†’ UV maps via the gameplay-plateau push
// constants (pc.color.xy = origin, pc.color.z is plateau-extent
// вЂ” we don't have the heightmap origin/side here, so use the well-
// known engine layout: terrain centred at origin, side = 2048 m).
layout(set = 0, binding = 8) uniform sampler2D u_terrain_height;
// Grass eligibility mask. R8 1024² covering the same 2048 m square as
// u_terrain_height. CPU-baked once at level load (presence + slope from
// the same FBM the GLSL terrain uses, then sample_terrain_height for an
// accurate slope finite-difference). Both this shader and
// grass_raymarch.frag sample binding 13 — single source of truth for
// "is there grass here", and avoids the per-step 9-cell noised() storm
// in the grass map() loop.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
const float kHeightmapSide = 2048.0;
float sampleHeightDelta(vec2 worldXZ) {
    // Was: 4 compare branches before the fetch. Replaced with clamp +
    // textureLod 0 — sampler is CLAMP_TO_EDGE and the heightmap delta
    // is ~0 at the world boundary (no sculpting past the edge), so a
    // clamped read at out-of-world XZ returns the right value too.
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_terrain_height, uv, 0.0).r;
}

// Sun shadow map (single-cascade ortho D32). Rendered each frame
// from the light's POV with castle, dyn-props and terrain chunks
// as casters. Sampled here for water-surface shadow occlusion.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

// TLAS вЂ” castle, dyn-props, terrain BLAS, etc. We fire shadow rays
// against it with cull-mask 0x02 so the terrain BLAS (instance
// mask 0x01) is excluded вЂ” terrain self-shadow is handled by
// calcShadow() against the FBM, and the rasterised terrain mesh sits
// below the BLAS detail anyway. With the mask, each ray only finds
// real dynamic-occluder hits (castle, boxes).
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Materials buffer вЂ” same layout cube.frag reads. Path-traced GI on
// the raymarched terrain looks up the BRDF colour at each bounce hit
// from this buffer using the closest-hit's instance custom index
// (with the kStaticBlasSentinel branch for the merged static BLAS).
struct Material {
    vec4 color;
    vec4 emissive;
    vec4 tex;       // x: albedo idx, y: normal idx, z: uv scale, w: spare
};
layout(set = 0, binding = 2, std430) readonly buffer Materials {
    Material materials[];
};
// Sentinel matching kStaticBlasInstSentinel in src/engine/vk_engine/rt.cpp
// вЂ” the merged-static castle BLAS uses this in instanceCustomIndex; the
// shader recovers the per-brush material via primitive_id / 12.
const int kStaticBlasSentinel = 0xFFFFFF;
const int kCubeTrisPerBox     = 12;

// Mask 0x02 = "anything except the terrain BLAS instance".
bool any_hit_no_terrain(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Closest-hit variant for the PCSS blocker search вЂ” returns the hit
// distance so we can size the penumbra cone.
bool closest_hit_no_terrain(vec3 origin, vec3 dir, float t_max,
                             out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

// Closest-hit + material lookup (mask 0x02 вЂ” terrain BLAS skipped).
// Used by the path-traced GI loop to find castle / dyn-prop bounce
// hits and look up their albedo for the throughput tint.
bool closest_hit_material_no_terrain(vec3 origin, vec3 dir, float t_max,
                                     out float out_t,
                                     out int out_inst,
                                     out int out_prim) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t   = rayQueryGetIntersectionTEXT(rq, true);
    out_inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    out_prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    return true;
}

// Shadow-caster mask 0x01 вЂ” used FROM the GI bounce hit point on a
// castle / dyn-prop surface to test sun visibility. Includes the
// terrain BLAS (which IS marked 0x01) so a bounce hit deep in a
// valley is correctly shadowed by the surrounding terrain. NOT used
// for the primary terrain receiver shadow вЂ” that's the no-terrain
// mask above to avoid procedural-vs-BLAS LOD self-hits.
bool any_hit_shadow_caster(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x01, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Build a cosine-weighted sample direction in the hemisphere around
// `n`. With this distribution the Lambertian estimator simplifies to
// `albedo * Li` per sample (the cos(Оё) and BRDF/ПЂ factors cancel
// against the pdf), which matches cube.frag's GI formulation.
vec3 cos_hemi(float u1, float u2, vec3 n) {
    float r   = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3  d   = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    vec3 up_a = abs(n.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t    = normalize(cross(up_a, n));
    vec3 b    = cross(n, t);
    return d.x * t + d.y * n + d.z * b;
}

// Tiny xorshift32 hash вЂ” deterministic per-pixel jitter for the
// stratified shadow rays.
// Interleaved Gradient Noise вЂ” same dither pattern cube.frag uses
// for PCSS / RTAO. xorshift hashes (the previous rmRand) produce
// row-banding correlations that read as moving "lines in shadows"
// even with TAA. IGN is designed for stochastic rendering: low
// perceptual structure across pixels, well-distributed across the
// sample-index axis when you offset the input position.
float ignBase(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
float rmRand(uvec3 seed) {
    // Same recipe as cube.frag::rand вЂ” shift the input position by
    // the sample index along a magic vector so each sample's jitter
    // is uncorrelated within a pixel but smooth across neighbours.
    float s = float(seed.x) + 5.588238 * float(seed.z);
    float t = float(seed.y) + 1.388765 * float(seed.z);
    return ignBase(vec2(s, t));
}

// Scene UBO вЂ” same layout as cube.frag's binding 0. Only a handful of
// fields are read here (camera, sun, sky); the rest are unused.
layout(set = 0, binding = 0) uniform Scene {
    vec4 sun_direction;
    vec4 sun_color;
    vec4 ambient;
    vec4 sky_color;
    ivec4 rt_flags;
    vec4 rt_params;
    ivec4 rt_flags2;
    vec4 rt_params2;
    vec4 camera_pos;
    vec4 rt_lod;
    vec4 viewport;
    vec4 muzzle_pos;
    vec4 muzzle_color;
    vec4 terrain_params;
    vec4 terrain_h_low;
    vec4 terrain_h_high;
    vec4 grass_extra;
    vec4 grass_extra2;
    mat4 light_vp;
    vec4 terrain_extra;
    // Ocean / water plane:
    //   water_params.x = enabled (0/1)
    //   water_params.y = world Y of calm sea level
    //   water_params.z = wave strength (m)
    //   water_params.w = animation time (seconds)
    //   water_color.rgb = deep-water tint
    vec4 water_params;
    vec4 water_color;
    vec4 water_color_shallow;
    // x: shore blend distance (m), y: shore noise strength,
    // z: TLAS-reflection flag, w: water transparency.
    vec4 water_shore;
    // Fog band: x = y_start, y = y_top, z = noise strength.
    vec4 fog_band;
    // Per-pixel RT caps for the terrain shader (sliders exposed to UI):
    //   x: PCSS sample cap
    //   y: GI sample cap
    //   z: final-colour AO multiplier strength (0..1)
    //   w: GI bounces cap
    vec4 terrain_rt_extra;
    // Grass colour palette + extras (UI-driven).
    //   grass_color_top.rgb    = blade tip colour
    //   grass_color_top.w      = raymarched grass density (0..1)
    //   grass_color_bottom.rgb = blade base colour
    //   grass_color_bottom.w   = blade-base AO floor (0..1)
    //   grass_color_ground.rgb     = CLOSE terrain tint
    //   grass_color_ground.w       = ground tint strength (0..1)
    //   grass_color_ground_far.rgb = FAR terrain tint
    vec4 grass_color_top;
    vec4 grass_color_bottom;
    vec4 grass_color_ground;
    vec4 grass_color_ground_far;
    // Fake grass-cast shadows on raymarched terrain.
    //   x: strength (0..1, 0 = off)
    //   y: sample count (int 0..8 along sun-XZ direction)
    //   z: max reach (m)
    //   w: unused
    vec4 grass_shadow_params;
} scene;

// Cheap directional sky model вЂ” horizon -> zenith ramp around the
// scene's sky colour, with a small sun-tinted lift near the horizon.
// Used as the "no-noise" reference for the GI softener and as the
// sample value for hemisphere rays that miss any occluder.
vec3 sample_sky_atmosphere(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, sqrt(up));   // pow(up, 0.45) approx
}

// Sky sample WITH the sun halo вЂ” used for path-traced GI rays that
// miss into the open. A bounce ray that points near the sun should
// pick up the disk's intensity (that's a sun-direct contribution
// without firing a separate sun ray). The halo is bounded by the
// pow(8) cosine and the 0.08 multiplier so it doesn't spike on
// near-grazing samples; matches cube.frag's path-trace estimator.
vec3 sample_sky(vec3 dir) {
    vec3  L = normalize(scene.sun_direction.xyz);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    vec3 sky = mix(horizon, zenith, sqrt(up));   // pow(up, 0.45) approx
    // pow(x, 8) → 3 squarings instead of exp/log.
    float h1 = max(dot(dir, L), 0.0);
    float h2 = h1 * h1;
    float h4 = h2 * h2;
    float halo = h4 * h4;
    sky += scene.sun_color.rgb * scene.sun_color.a * 0.08 * halo;
    return sky;
}

// === Hash + Value-Noise with analytical derivatives ===
// dot/fract pattern (avoids sin() precision artifacts on some GPUs).
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// Value Noise + в€‚n/в€‚x, в€‚n/в€‚y. Hermite smoothstep `3tВІ-2tВі` for CВ№
// continuity; `du = 6t(1-t)` is the chain-rule derivative used by the
// FBM-erosion accumulator.
vec3 noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    float value = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  deriv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(value, deriv);
}

float noise2(vec2 p) { return noised(p).x; }

// Per-octave ~37В° rotation breaks axis-aligned banding. Unit
// determinant = pure rotation, no scaling.
const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);

const float TERRAIN_SCALE  = 0.003;
const float TERRAIN_HEIGHT = 120.0;

// Plateau blend factor вЂ” 1 inside the gameplay plateau region, 0
// outside, smooth ramp across `kPlateauBlend` metres so the FBM
// surroundings ease into the flat castle pad. Mirrors what
// generate_heightmap() does in the rasterised path so the visible
// terrain shape matches what physics / castle assume.
const float kPlateauBlend = 24.0;
float plateauWeight(vec2 worldXZ) {
    vec2 c = pc.color.xy;
    float ext = pc.color.z;
    vec2 d = abs(worldXZ - c) - vec2(ext);
    float dout = max(max(d.x, 0.0), max(d.y, 0.0));
    return 1.0 - smoothstep(0.0, kPlateauBlend, dout);
}

// Medium-detail FBM вЂ” used by the ray march. Derivative-erosion term
// `1/(1+dot(d,d))` suppresses high-frequency layers on steep slopes,
// producing realistic ridge/valley structure instead of fractal noise.
// Octave count is the per-frame quality knob (pc.tex_params.z, 4..9):
// last 3 of 9 contribute ~1.5 % of total amplitude so dropping them
// barely changes the shape but cuts ~30 % of the inner work.
float terrainM(vec2 p) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    int kOct = int(pc.tex_params.z);
    for (int i = 0; i < kOct; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    // Blend the FBM toward the gameplay plateau height inside the
    // castle's footprint so brushes / dyn-props look seated.
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    // Add sculpt / plateau-noise / disk-overlay deltas. Texture is
    // 0 in unsculpted areas so this is a no-op for stock terrain.
    h += sampleHeightDelta(wp);
    return h;
}

// Distance-LOD variant: drops octaves smoothly as the ray distance
// grows. The full octave count is the user's pc.tex_params.z; we
// fade down to ~2 octaves past 600 m. Same outer wrapping (plateau
// blend + sculpt delta) so close vs far results stay continuous вЂ”
// only the inner FBM converges to a coarser sum past the LOD start.
// Uses a uniform `kMaxOct` cap with an early-break so the loop body
// stays branch-friendly for the GPU.
float terrainM_lod(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, oct_full - 4);
    float lod_f  = smoothstep(120.0, 600.0, ray_t);
    int oct      = oct_full - int(lod_f * float(oct_full - oct_min));
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 16;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float tpl = plateauWeight(wp);
    h = mix(h, pc.color.w, tpl);
    h += sampleHeightDelta(wp);
    return h;
}

// Normal-detail FBM вЂ” finite-difference normals call this 3Г— per hit
// pixel, so it's the single biggest cost in calcNormal. Octave count
// is pc.tex_params.w (4..16). 8 is barely-distinguishable from 16 at
// gameplay distances.
float terrainH(vec2 p) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    int kOct = int(pc.tex_params.w);
    for (int i = 0; i < kOct; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    h += sampleHeightDelta(wp);
    return h;
}

// Generous headroom: sampleHeightDelta() (sculpt strokes + disk
// overlays) can push the FBM-derived terrain well above TERRAIN_HEIGHT.
// Underestimating max height triggers the open-air early-out in
// raymarch() while the camera is still inside the terrain volume,
// producing the "everything broken at high altitude" symptom.
float terrainMaxHeight() { return max(TERRAIN_HEIGHT, pc.color.w) + 80.0; }

// Same FBM as terrainH() but propagates the analytic gradient through
// the per-octave Jacobian: J starts as TERRAIN_SCALE·I (world→local),
// each octave multiplies J by m2·2 (matching p_new = m2 * p_old * 2).
// The world-space derivative of the noise value is transpose(J) * n.yz
// (chain rule on the local-coord gradient noised() returns).
//
// Plateau + sculpt-overlay terms contribute to the height only — their
// gradient is small at the wavelengths normals care about and would
// just slightly smooth the lit normal. Damp factor's gradient is also
// dropped: it's a slowly-varying weight, the contribution to dh/dp is
// noise-floor.
//
// Returns vec3(height, dh/dx_world, dh/dz_world). Used by calcNormal
// to replace 3× terrainH() finite-difference taps with one analytic
// evaluation — biggest single per-pixel terrain saving.
vec3 terrainH_grad(vec2 wp) {
    vec2 p = wp * TERRAIN_SCALE;
    mat2 J = mat2(TERRAIN_SCALE, 0.0, 0.0, TERRAIN_SCALE);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    vec2 grad = vec2(0.0);
    int kOct = int(pc.tex_params.w);
    for (int i = 0; i < kOct; i++) {
        vec3 n = noised(p);
        d += n.yz;
        float damp = 1.0 / (1.0 + dot(d, d));
        a += b * n.x * damp;
        grad += b * damp * (transpose(J) * n.yz);
        b *= 0.5;
        p = m2 * p * 2.0;
        J = m2 * J * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    h += sampleHeightDelta(wp);
    return vec3(h, grad.x * TERRAIN_HEIGHT, grad.y * TERRAIN_HEIGHT);
}

// Coarse 3-octave heightfield вЂ” strictly for AO + GI occlusion
// raymarches where sub-meter detail doesn't matter. Cuts the per-step
// noise cost vs terrainM() (which runs the full march octave count,
// typically 4-6) while staying conservative-ish for occlusion (the
// gross terrain shape is what blocks light).

// (terrain_blocks_ray + terrainCoarse removed — per-pixel terrain self-
// occlusion was replaced by mask-0x02 TLAS-only AO once the sun shadow
// map covered the gross-shadow case. The 3-octave coarse heightfield
// here was unused and just bloating the shader.)

// === Adaptive-step heightfield ray march ===
// h = ray.y - terrain(ray.xz) is a conservative step bound for any
// heightfield. Step factor 0.4 stays safe on sharp ridges; the
// distance-adaptive precision threshold avoids over-iterating distant
// pixels where one screen pixel spans many world meters.
float raymarch(vec3 ro, vec3 rd) {
    float maxH = terrainMaxHeight();
    float t = 0.0;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    if (ro.y > maxH) {
        // Skip the open-air segment by jumping to the upper bound plane.
        t = (ro.y - maxH) / (-rd.y);
    }
    // Phase 5 вЂ” blue-noise jittered start offset. Per-pixel hash-based
    // sub-step shift breaks up the per-fragment march alignment, so
    // contour banding from a too-aggressive step factor disperses into
    // noise (then washes out under TAA). Free; no extra texture taps.
    float jitter = rmRand(uvec3(gl_FragCoord.xy, 0u));
    int kSteps = int(pc.tex_params.x);
    float kStep = pc.emissive.x;
    bool relax = pc.grass_params.y > 0.5;
    // Tiny initial offset; max half-cell so the surface hit shifts
    // sub-pixel only.
    t += jitter * 0.5;
    float maxH_cap = terrainMaxHeight();
    for (int i = 0; i < kSteps; i++) {
        vec3 pos = ro + t * rd;
        // Sky escape: any ray rising above the FBM ceiling with a
        // positive y-component cannot ever hit terrain — bail early
        // instead of walking the budget out to t=1500. Saves the worst-
        // case grazing-ray cost (which used to burn the full kSteps).
        if (rd.y > 0.0 && pos.y > maxH_cap) return -1.0;
        // Distance-LOD on the FBM octave count вЂ” full octaves close
        // to camera, smoothly fading down to ~2 octaves past 600 m.
        // Equivalent visual quality past the LOD onset (sub-pixel
        // detail) but big perf saving for rays that cover hundreds
        // of metres before hitting.
        float h = pos.y - terrainM_lod(pos.xz, t);
        if (abs(h) < 0.0015 * t) break;
        if (t > 1500.0) return -1.0;
        // Phase 6 вЂ” relaxation cone-stepping. Step grows with
        // distance so the same ray covers more ground in fewer
        // iterations. 0.7 attenuation on h prevents penetration
        // when the relaxation factor over-shoots.
        float advance;
        if (relax) {
            float rl = max(t * 0.02, 1.0);
            advance = kStep * h * rl * 0.7;
        } else {
            advance = kStep * h;
        }
        // Once h goes negative we've crossed the surface — the next step
        // would walk us back, and with rl > 1 the back-step grows enough to
        // ping-pong indefinitely until kSteps caps. Break out instead so
        // the caller gets the closest approach.
        if (advance < 0.0) break;
        t += advance;
    }
    return t;
}

// Finite-difference normal with distance-scaled epsilon. Eps grows
// linearly with distance (was quadratic вЂ” that smeared distant
// detail far more than required for anti-aliasing) so distant
// terrain keeps edge crispness while close-up still uses sub-cm
// epsilon. The earlier "micro-noise on lit normal" path was removed
// вЂ” its `noise2(p) - noise2(p + 0.5)` cell-boundary derivative
// produced visible 8вЂ“16 pixel square blocks on the surface past
// 80 m. The FBM normal-octave slider (default 18) gives plenty of
// distance detail without the artefacts.
vec3 calcNormal(vec3 pos, float t) {
    // Reverted (again) from the 1× analytic-gradient version. The damp
    // term `1/(1+dot(d,d))` and its derivative carry real high-frequency
    // detail that the cheap chain-rule loses — the snow/rock mid-
    // distance band reads as washed-out and "missing rockiness" without
    // it. 3× finite-difference is the correct path until the analytic
    // gradient is rewritten with the proper damp + Hessian terms.
    float eps = 0.02 + 0.0008 * t;
    float hC = terrainH(pos.xz);
    float hR = terrainH(pos.xz + vec2(eps, 0.0));
    float hU = terrainH(pos.xz + vec2(0.0, eps));
    vec3 N = normalize(vec3(hC - hR, eps, hC - hU));

    // Closeup micro-bump вЂ” two-octave high-frequency noise gradient
    // tilts the surface normal so near-camera ground reads as rocky
    // / pebbly instead of glass-smooth. Falls off past 60 m so it
    // only costs the always-near pixels and never aliases at depth.
    // Uses noised() which returns (value, dN/dx, dN/dy) in one call.
    // Tightened from 0.001 → 0.05: pixels in the 50–60 m band were
    // paying full noised() cost for sub-1 % blend weight. The lost
    // detail is invisible at the cutoff distance.
    float detail_w = 1.0 - smoothstep(15.0, 60.0, t);
    if (detail_w > 0.05) {
        vec3 d1 = noised(pos.xz * 4.0);
        vec3 d2 = noised(pos.xz * 13.0);
        vec2 grad = (d1.yz * 0.30 + d2.yz * 0.18) * detail_w;
        // Project the noise gradient as a tangent-plane perturbation.
        N = normalize(N + vec3(-grad.x, 0.0, -grad.y) * 0.45);
    }
    return N;
}

// Classic heightfield soft-shadow march вЂ” `min(kВ·h/t)`. k ties to
// the global Shadow softness slider (rt_params.x) so a single
// control softens both terrain self-shadow AND PCSS castle/box
// shadows. Slider 0 в†’ k=64 (razor sharp), slider 0.15 в†’ k=4
// (very soft). Cubic smoothstep at the end softens the linear
// penumbra ramp.
float calcShadow(vec3 pos, vec3 sunDir) {
    // Driven by global Shadow softness slider Г— per-terrain
    // softness scale. Same combination the PCSS below uses so both
    // shadow types respond to the same UI controls in lock-step.
    float soft_g = scene.rt_params.x;
    if (scene.terrain_params.w > 0.0) {
        soft_g *= max(0.05, scene.terrain_params.w);
    }
    float soft = clamp(soft_g / 0.15, 0.0, 1.0);
    float k    = mix(64.0, 4.0, soft);
    float res = 1.0;
    float t   = 0.05;
    int kSteps = int(pc.tex_params.y);
    for (int i = 0; i < kSteps; ++i) {
        vec3 p = pos + t * sunDir;
        // Distance-LOD: deep-valley shadow taps don't need full 8-octave
        // resolution. Same `terrainM_lod(t)` the primary raymarch uses
        // — drops to ~2 octaves past 600 m. Soft shadows are a low-pass
        // signal; the high-octave detail dropped here is sub-pixel.
        float h = p.y - terrainM_lod(p.xz, t);
        if (h < 0.001) { res = 0.0; break; }
        res = min(res, k * h / t);
        t += clamp(h, 0.5, 100.0);
        if (t > 800.0 || res < 0.001) break;
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// Fake grass-cast shadow on the raymarched terrain. Walks the grass
// mask along the sun's XZ direction from the terrain pixel toward
// the sun; cells with grass presence accumulate a soft shadow.
// Cheap (≤8 mask taps), and "good enough" because grass blades are
// sub-pixel at the kind of distance where the player would notice
// shadow inaccuracy. Returns 0..1 (0 = no grass shadow, 1 = fully
// in a grass cell's shadow). Caller multiplies the diffuse term by
// (1 - grass_shadow * strength).
float grassCastShadow(vec3 pos, vec3 sunDir) {
    float strength = scene.grass_shadow_params.x;
    int   N        = int(scene.grass_shadow_params.y);
    float reach    = max(scene.grass_shadow_params.z, 0.1);
    if (strength < 1e-3 || N <= 0) return 0.0;
    // Sun's XZ direction (from terrain toward sun). Skip when the
    // sun is near vertical (sun_xz_len → 0) — no horizontal shadow
    // projection then.
    vec2 sun_xz = sunDir.xz;
    float sun_xz_len = length(sun_xz);
    if (sun_xz_len < 0.05) return 0.0;
    sun_xz /= sun_xz_len;
    float shadow = 0.0;
    for (int i = 1; i <= 8; ++i) {
        if (i > N) break;
        float dist = reach * (float(i) / float(N));
        vec2 sample_xz = pos.xz + sun_xz * dist;
        vec2 uv = (sample_xz / kHeightmapSide) + vec2(0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
        float presence = textureLod(u_grass_mask, uv, 0.0).r;
        // Falloff with distance: nearer grass casts a stronger
        // contribution; far samples are smaller blades / lower sun
        // angle penetration.
        float falloff = 1.0 - smoothstep(0.0, 1.0, dist / reach);
        shadow = max(shadow, presence * falloff);
    }
    return shadow;
}

// Sample the CPU-baked grass mask (binding 13, RG8). Single source of
// truth shared with grass_raymarch.frag.
//   R = presence  (fixed at bake time)
//   G = slope mag (raw |∇h| / 2, un-normalized here)
// Slope cutoff is applied per-frame so the slider (grass_slope_n_min)
// affects both blades and the green underground tint in lock-step.
// Altitude + water gating stay per-pixel since they need terrainY.
float grassEligibility(vec2 worldXZ, float terrainY) {
    vec2 mask_uv = (worldXZ / kHeightmapSide) + vec2(0.5);
    if (mask_uv.x < 0.0 || mask_uv.x > 1.0 ||
        mask_uv.y < 0.0 || mask_uv.y > 1.0) return 0.0;
    vec2 mg = textureLod(u_grass_mask, mask_uv, 0.0).rg;
    float slope_n_min = scene.grass_extra.z;
    float slope_max = sqrt(max(1e-4, 1.0 / max(slope_n_min * slope_n_min, 1e-4) - 1.0));
    float slope_mag = mg.g * 2.0;
    float slope_w   = 1.0 - smoothstep(slope_max * 0.7,
                                        slope_max * 1.2, slope_mag);
    float alt_factor = 1.0 - smoothstep(55.0, 75.0, terrainY);
    float underwater = (scene.water_params.x > 0.5 &&
                         terrainY < scene.water_params.y) ? 0.0 : 1.0;
    return mg.r * slope_w * alt_factor * underwater;
}

// Layered material: rock base, grass on flats, snow up high (with
// noise-broken transition), sand near the floor. Multi-frequency
// noise (`nz`) breaks regularity in the transition lines.
vec3 getMaterial(vec3 pos, vec3 nor) {
    float slope = nor.y;
    float h = pos.y;
    float nz = noise2(pos.xz * 0.04) * noise2(pos.xz * 0.005);
    vec3 rock  = vec3(0.10, 0.09, 0.08);
    vec3 grass = mix(vec3(0.10, 0.08, 0.04),
                     vec3(0.05, 0.09, 0.02), nz);
    vec3 snow  = vec3(0.62, 0.65, 0.70);
    vec3 sand  = vec3(0.50, 0.45, 0.35);
    vec3 col = rock;
    col = mix(col, grass, smoothstep(0.5, 0.8, slope));
    float snowMask = smoothstep(80.0 - 20.0 * nz, 90.0, h)
                   * smoothstep(0.3, 0.7, slope);
    col = mix(col, snow, snowMask);
    float beachMask = smoothstep(2.5, 1.0, h) * smoothstep(0.5, 0.9, slope);
    col = mix(col, sand, beachMask);
    // Grass-density green tint — drives the ground colour where the
    // raymarched grass pass (grass_raymarch.frag) would place blades.
    // Two colours (close + far) blend with camera distance so meadows
    // can read as bright lit-green near the player and a darker
    // muted hue at distance (or whatever the user picks).
    float grass_amt = grassEligibility(pos.xz, pos.y);
    float cam_d     = distance(pos, scene.camera_pos.xyz);
    // Same anchor distances the grass-blade alpha ramp uses, so the
    // blade-fade and the ground-tint transition stay aligned.
    float ground_t  = smoothstep(30.0, 200.0, cam_d);
    vec3 ground_base = mix(scene.grass_color_ground.rgb,
                            scene.grass_color_ground_far.rgb,
                            ground_t);
    vec3 grass_top  = mix(ground_base, ground_base * 0.7, nz);
    float tint_str  = clamp(scene.grass_color_ground.w, 0.0, 1.0);
    col = mix(col, grass_top, grass_amt * tint_str);
    return col;
}

// Cheap raymarch for water-reflection rays. Same conservative-step
// heightfield trace as `raymarch()` but with a fixed reduced step
// count (the reflected rays usually hit within 30-50 steps because
// the surface is close to the camera). Skips jitter to avoid
// per-pixel reflection noise that wouldn't denoise temporally.
float raymarchReflect(vec3 ro, vec3 rd) {
    float maxH = terrainMaxHeight();
    float t = 0.0;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    if (ro.y > maxH) {
        t = (ro.y - maxH) / max(-rd.y, 1e-4);
    }
    const int kReflSteps = 64;
    const float kReflStep = 0.5;     // slightly aggressive
    for (int i = 0; i < kReflSteps; ++i) {
        vec3 pos = ro + t * rd;
        float h = pos.y - terrainM(pos.xz);
        if (abs(h) < 0.005 * t) break;     // looser threshold than primary
        if (t > 800.0) return -1.0;        // shorter range than primary
        t += kReflStep * h;
    }
    return t;
}

// Cheap animated ocean normal вЂ” two scrolling sin/cos directions
// summed at slightly off-axis frequencies. `scale` multiplies the
// base frequencies (1.0 = ~30 m wavelengths; 2.0 = half-wavelength
// finer ripples; 0.5 = bigger sweeping waves).
// Lake-style multi-octave wave normal. Combines a low-frequency
// directional swell (wind blowing along +X) with three cross-direction
// chops, then layers two high-frequency ripples for the close-up
// detail. Each octave is biased toward sharper crests via a sinВІ-ish
// shape (we use cos for the analytical derivative, equivalent up to
// a phase shift). The result reads as proper lake water rather than
// the previous uniform sinusoidal grid.
vec3 waterNormal(vec2 worldXZ, float t, float strength, float scale) {
    // Wind axis biased on +X вЂ” most lakes have a dominant direction.
    vec2 wind  = normalize(vec2(1.0, 0.35));
    vec2 cross1 = normalize(vec2(0.7, -0.7));
    vec2 cross2 = normalize(vec2(-0.5, 0.85));

    // Octaves: low (long swell) в†’ high (ripple) wave numbers.
    // amp[i] decays so high frequencies don't dominate the normal.
    const int kOct = 6;
    float k_base[6]   = float[6](0.06, 0.11, 0.18, 0.27, 0.42, 0.65);
    float spd_base[6] = float[6](0.55, 0.70, 0.95, 1.20, 1.55, 2.10);
    float amp[6]      = float[6](1.00, 0.75, 0.55, 0.40, 0.28, 0.18);

    float dx = 0.0;
    float dz = 0.0;
    // Two octaves go along the wind axis; rest distributed across
    // the cross directions for a non-grid wave field.
    vec2  dir;
    for (int i = 0; i < kOct; ++i) {
        if (i < 2)        dir = wind;
        else if (i < 4)   dir = cross1;
        else              dir = cross2;
        float k = k_base[i] * scale;
        // Phase: dirВ·worldXZ * k - speed * t.
        float ph = (worldXZ.x * dir.x + worldXZ.y * dir.y) * k -
                   spd_base[i] * t;
        // d/dx of sin(ph) wrt worldXZ = cos(ph) * k * dir.x. Sharpen
        // by raising to a 1.4 power on the absolute term вЂ” keeps the
        // sign while pinching crests.
        float c = cos(ph);
        float abs_c = abs(c);
        float sharp = sign(c) * pow(abs_c, 1.4);
        dx += amp[i] * k * dir.x * sharp;
        dz += amp[i] * k * dir.y * sharp;
    }
    return normalize(vec3(-dx * strength, 1.0, -dz * strength));
}

void main() {
    // Ray reconstruction: vWNear / vWFar were computed in the vert
    // shader (pc.model × ndc-near / ndc-far) and interpolated. Per
    // pixel we just need the homogeneous divide + subtract + normalize
    // — the matrix multiplies that used to dominate this stub are gone.
    vec3 ro = scene.camera_pos.xyz;
    vec3 rd = normalize(vWFar.xyz / vWFar.w - vWNear.xyz / vWNear.w);

    float t = raymarch(ro, rd);

    // Water plane intersection. The ray hits the water surface if
    //   - water is enabled AND
    //   - the ray is travelling downward and starts above water
    //     (or upward and starts below вЂ” we render that case as
    //     "underwater" looking up but for now keep it simple)
    bool water_on = scene.water_params.x > 0.5;
    float water_y = scene.water_params.y;
    float t_water = -1.0;
    if (water_on && abs(rd.y) > 1e-4) {
        float tw = (water_y - ro.y) / rd.y;
        if (tw > 0.0 && (t < 0.0 || tw < t)) {
            t_water = tw;
        }
    }

    if (t < 0.0 && t_water < 0.0) {
        // Sky вЂ” let the existing compose-pass sky handle this pixel.
        discard;
    }

    // === Water surface shading ===========================================
    if (t_water > 0.0) {
        vec3 wpos = ro + rd * t_water;
        float wave_str = scene.water_params.z;
        float wave_t   = scene.water_params.w;
        // wave-frequency multiplier вЂ” packed in water_color_shallow.w
        // by update_scene_ubo. Defaults to 1.0; the slider lets the
        // user dial finer ripples or bigger sweeping waves.
        float wave_scale = max(0.05, scene.water_color_shallow.w);
        vec3 wnor = waterNormal(wpos.xz, wave_t, wave_str, wave_scale);
        vec3 sunDirW = scene.sun_direction.xyz;

        // Schlick fresnel вЂ” F0 = 0.02 for water. Looking grazing-on
        // means full reflection; looking straight down в‰€ 2 % reflection.
        // pow(x, 5) → 3 multiplies (x²·x²·x).
        float cosV = clamp(-dot(rd, wnor), 0.0, 1.0);
        const float F0 = 0.02;
        float oneMinusV = 1.0 - cosV;
        float fv2 = oneMinusV * oneMinusV;
        float fv5 = fv2 * fv2 * oneMinusV;
        float fres = F0 + (1.0 - F0) * fv5;

        vec3 refl = reflect(rd, wnor);
        // Sky reflection вЂ” horizonв†’zenith gradient + sun halo. Used
        // directly when the reflection ray escapes the terrain or
        // when RT reflections are off.
        float skyT = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 sky_refl = mix(vec3(0.55, 0.65, 0.78),
                             vec3(0.30, 0.45, 0.75), skyT);
        // pow(x, 60) ≈ pow(x, 64) = 6 squarings; visually identical
        // for a sun-halo lobe at this exponent.
        float sh1 = max(dot(refl, sunDirW), 0.0);
        float sh2 = sh1 * sh1; float sh4 = sh2 * sh2;
        float sh8 = sh4 * sh4; float sh16 = sh8 * sh8;
        float sh32 = sh16 * sh16; float sunHalo = sh32 * sh32;
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sunHalo * 0.6;

        vec3 reflCol = sky_refl;
        // RT-style FBM-march reflection вЂ” picks up the actual mountain
        // / valley silhouettes in the water surface. Cheap because
        // we use a reduced-quality march; if it misses (refl ray
        // exits past the terrain or goes into the sky) fall back to
        // the analytical sky.
        bool do_rt_refl = scene.water_params.x > 0.5 &&
                          pc.grass_params.w > 0.5 && refl.y > 0.02;
        if (do_rt_refl) {
            // Lift the origin slightly along refl so we don't
            // immediately self-intersect the wave bumps below the
            // water plane.
            vec3 r_ro = wpos + refl * 0.5 + vec3(0.0, 0.05, 0.0);
            float t_r = raymarchReflect(r_ro, refl);
            if (t_r > 0.0) {
                vec3 r_pos = r_ro + refl * t_r;
                vec3 r_nor = calcNormal(r_pos, t_r);
                // Lambert + sky ambient. No shadow march in the
                // reflection вЂ” too expensive for a free-running
                // surface effect, and the missing self-shadow is
                // visually fine on a wavy reflection.
                float r_dif = max(dot(r_nor, sunDirW), 0.0);
                float r_amb = 0.5 + 0.5 * r_nor.y;
                vec3  r_mat = getMaterial(r_pos, r_nor);
                vec3  r_lin = r_dif * scene.sun_color.rgb *
                                scene.sun_color.a +
                              r_amb * scene.sky_color.rgb * 0.35;
                vec3  r_col = r_mat * r_lin;
                // Same atmospheric extinction as the primary
                // shading, applied to the reflection-ray distance.
                vec3  r_ext = exp(-t_r * 0.00025 *
                                  vec3(1.0, 1.5, 4.0));
                // pow(x, 8) → 3 squarings.
                float rf1 = max(dot(refl, sunDirW), 0.0);
                float rf2 = rf1 * rf1; float rf4 = rf2 * rf2; float rf8 = rf4 * rf4;
                vec3  r_fog = mix(vec3(0.55, 0.55, 0.58),
                                  vec3(1.0, 0.7, 0.3),
                                  0.3 * rf8);
                r_col = r_col * r_ext + r_fog * (1.0 - r_ext);
                reflCol = r_col;
            }
            // miss в†’ keep sky_refl
        }

        // TLAS reflection вЂ” ray-query against castle / cubes / dyn
        // props (mask 0x02 skips the terrain BLAS). If the TLAS hit
        // is closer than whatever's in `reflCol` (FBM-march or sky),
        // override with a basic shaded stand-in. We don't do per-
        // instance material lookup here to keep the path cheap; a
        // mid-grey lit by sun + sky reads correctly for most stone /
        // wood / box surfaces and reflections in water are too low-
        // detail for the user to notice the missing texture.
        if (scene.water_shore.z > 0.5 && refl.y > 0.02) {
            vec3 r_ro = wpos + wnor * 0.05;
            float t_tlas;
            if (closest_hit_no_terrain(r_ro, refl, 200.0, t_tlas)) {
                // Fake a vaguely-up normal weighted toward the
                // reflection ray (no real RT normal without a hit-
                // attribute pass). Good enough for water mirrors.
                vec3 r_nor = normalize(mix(vec3(0.0, 1.0, 0.0),
                                            -refl, 0.4));
                float r_dif = max(dot(r_nor, sunDirW), 0.0);
                float r_amb = 0.5 + 0.5 * r_nor.y;
                vec3 stone = vec3(0.55, 0.52, 0.48);
                vec3 r_col = stone * (r_dif * scene.sun_color.rgb *
                                       scene.sun_color.a +
                                       r_amb * scene.sky_color.rgb * 0.35);
                // Atmospheric extinction over reflection distance.
                vec3 r_ext = exp(-t_tlas * 0.00025 *
                                  vec3(1.0, 1.5, 4.0));
                vec3 r_fog = mix(vec3(0.55, 0.55, 0.58),
                                  vec3(1.0, 0.7, 0.3),
                                  0.3 * pow(max(dot(refl, sunDirW),
                                                0.0), 8.0));
                r_col = r_col * r_ext + r_fog * (1.0 - r_ext);
                reflCol = r_col;
            }
        }

        // Shore-aware base colour. We compute the actual water depth
        // at this surface point вЂ” water_level minus the FBM terrain
        // height directly below. Shallow water = `water_color_shallow`,
        // deep water = `water_color`, transition over `shore_blend`
        // metres. Mild noise on the depth breaks up the perfectly
        // geometric shoreline so it reads as natural beach.
        vec3 deep    = scene.water_color.rgb;
        vec3 shallow_tint = scene.water_color_shallow.rgb;
        float shore_blend = max(0.1, scene.water_shore.x);
        float shore_noise = scene.water_shore.y;
        float terrain_y = terrainM(wpos.xz);
        float depth_m   = max(0.0, water_y - terrain_y);
        float dn = noise2(wpos.xz * 0.18) +
                    0.5 * noise2(wpos.xz * 0.42);
        depth_m += (dn - 0.6) * shore_blend * shore_noise;
        depth_m  = max(0.0, depth_m);
        float water_blend = smoothstep(0.0, shore_blend, depth_m);
        vec3  shallow = mix(shallow_tint, deep, water_blend);
        // Distance fade to the deep tint so far water doesn't read
        // shallower than near water just because of the noise.
        float distFade = 1.0 - exp(-t_water * 0.012);
        shallow = mix(shallow, deep, distFade * 0.4);

        // Specular highlight on the wave crests when the sun is close
        // to the half-vector. Sharp lobe (~64) so it reads as glints.
        vec3 H = normalize(sunDirW - rd);
        // pow(x, 64) → 6 successive squarings.
        float sp1 = max(dot(wnor, H), 0.0);
        float sp2 = sp1 * sp1; float sp4 = sp2 * sp2;
        float sp8 = sp4 * sp4; float sp16 = sp8 * sp8;
        float sp32 = sp16 * sp16; float spec = sp32 * sp32;

        // Sun shadow on the water surface вЂ” bound-checked sample of
        // the sun shadow map (binding 7), already populated from
        // terrain chunks + castle + dyn-props. Dims the specular
        // (no glints in shadow) and slightly darkens the base tint.
        float water_lit = 1.0;
        if (scene.water_color.w > 0.5) {
            vec4 lc = scene.light_vp * vec4(wpos, 1.0);
            vec3 lndc = lc.xyz / lc.w;
            if (lndc.z >= 0.0 && lndc.z <= 1.0) {
                vec2 luv = lndc.xy * 0.5 + 0.5;
                if (all(greaterThanEqual(luv, vec2(0.0))) &&
                    all(lessThanEqual(luv, vec2(1.0)))) {
                    const float kRecvBias = 0.00005;
                    water_lit = textureLod(u_sun_shadow_map,
                                            vec3(luv, lndc.z - kRecvBias), 0.0);
                }
            }
        }

        // Underwater showthrough вЂ” for shallow water we evaluate the
        // terrain at wpos.xz and pass its surface tint through the
        // water tinted by depth. Beer's-law-style attenuation:
        // T = exp(-depth * absorption) so deep water still hides
        // the bottom while shallow water reads transparent.
        vec3 baseTint = shallow;     // current shore-blended water colour
        float trans_amt = scene.water_shore.w;
        if (trans_amt > 0.001) {
            // Reuse the depth_m we already computed for the shore
            // blend. Absorption coefficient picks ~50 % attenuation
            // at a 2 m depth, ~95 % by 6 m.
            float absorp = 0.5;
            float Tw = exp(-depth_m * absorp);
            // Visibility gate: skip the entire underwater shading
            // (calcNormal = 1× FBM, getMaterial = several `noise2`
            // taps, lambert + ambient) when the contribution is
            // below ~5 % of the surface tint. At 6 m depth Tw drops
            // to 0.05; deep-water pixels read as opaque tint anyway.
            if (Tw * trans_amt > 0.05) {
                // Underwater surface вЂ” terrain material at the under-
                // water point, lambert-shaded by sun (with shadow), no
                // shore noise so the look is calm.
                vec3 u_pos = vec3(wpos.x, terrain_y, wpos.z);
                vec3 u_nor = calcNormal(u_pos, t_water);
                vec3 u_mat = getMaterial(u_pos, u_nor);
                float u_dif = max(dot(u_nor, sunDirW), 0.0) * water_lit;
                float u_amb = 0.5 + 0.5 * u_nor.y;
                vec3  u_col = u_mat * (u_dif * scene.sun_color.rgb *
                                        scene.sun_color.a * 0.8 +
                                        u_amb * scene.sky_color.rgb * 0.35);
                // Tint the underwater colour by the water tint so it
                // reads as "looking through water", not "no water".
                u_col *= mix(vec3(1.0), shallow * 1.5 + vec3(0.05),
                              0.5);
                // Blend: shallow в†’ underwater visible, deep в†’ opaque
                // water_color. trans_amt scales the maximum showthrough.
                baseTint = mix(baseTint, u_col, Tw * trans_amt);
            }
        }

        vec3 col_w = mix(baseTint, reflCol, fres);
        col_w += scene.sun_color.rgb * scene.sun_color.a * spec *
                  0.8 * water_lit;
        // Subtle darkening of the surface tint when in shadow.
        col_w *= mix(0.7, 1.0, water_lit);

        // Wave-peak foam вЂ” wnor.y drops on steep wave faces; fold in
        // a 2-octave drifting noise so the foam reads as a stochastic
        // froth rather than a clean steepness mask. Scrolling along
        // the wind direction gives the field motion. Foam intensity
        // also modulated by `water_lit` so shadowed water doesn't get
        // bright white speckles.
        {
            float steepness = clamp(1.0 - wnor.y * 1.04, 0.0, 1.0);
            vec2 fp = wpos.xz * 0.55 + vec2(wave_t * 0.30, wave_t * 0.18);
            float fn = noise2(fp) * 0.6 + noise2(fp * 2.13) * 0.4;
            float foam = smoothstep(0.45, 0.85, fn * steepness * 1.6);
            col_w = mix(col_w, vec3(0.95, 0.97, 1.0),
                        foam * 0.85 * water_lit);
        }

        // Apply the same volumetric fog as terrain (reuse the
        // existing block by jumping to a tail). To keep the diff
        // tight we re-do the fog inline here using the same code path.
        float sundotW = clamp(dot(rd, sunDirW), 0.0, 1.0);
        vec3 ext_w = exp(-t_water * 0.00025 * vec3(1.0, 1.5, 4.0));
        // pow(x, 8) → 3 squarings.
        float sd2 = sundotW * sundotW;
        float sd4 = sd2 * sd2;
        float sd8 = sd4 * sd4;
        vec3 fogColW = mix(vec3(0.55, 0.55, 0.58),
                           vec3(1.0, 0.7, 0.3),
                           0.3 * sd8);
        col_w = col_w * ext_w + fogColW * (1.0 - ext_w);

        // Write hit depth; rasterised geometry that's actually in
        // front (e.g. a cube on a pier) still occludes us.
        vec4 clipW = pc.mvp * vec4(wpos, 1.0);
        // See main terrain branch вЂ” cap below the compose sky-cutoff
        // threshold so far water doesn't get repainted as sky-below-
        // horizon (black). Guard against w<=0 (point behind camera) so
        // the depth divide doesn't produce NaN.
        gl_FragDepth = min(clipW.z / max(clipW.w, 1e-4), 0.9998);
        outColor = vec4(col_w, 1.0);
        // Motion vector for TAA reprojection (same pattern as
        // terrain branch below). Treats the surface as world-static
        // вЂ” wave-bump animation contributes sub-pixel motion that
        // TAA's spatial filter absorbs.
        vec2 current_uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
        vec4 prev_clipW = pc.prev_mvp * vec4(wpos, 1.0);
        if (prev_clipW.w > 0.0) {
            vec2 prev_ndc = prev_clipW.xy / prev_clipW.w;
            vec2 prev_uv  = prev_ndc * 0.5 + 0.5;
            outMotion = current_uv - prev_uv;
        } else {
            outMotion = vec2(0.0);
        }
        return;
    }

    vec3 pos = ro + t * rd;
    vec3 nor = calcNormal(pos, t);
    vec3 sunDir = scene.sun_direction.xyz;

    float dif = clamp(dot(nor, sunDir), 0.0, 1.0);
    float amb = 0.5 + 0.5 * nor.y;
    float fre_in = clamp(1.0 + dot(rd, nor), 0.0, 1.0);
    float fre    = fre_in * fre_in;   // pow(x, 2) is just a square

    // Terrain self-shadow (FBM march). Bias is small (5 cm along
    // the surface normal) вЂ” large biases (the previous 50 cm) made
    // the shadow visibly detach from the contact line under
    // distant FBM normals.
    // calcShadow returns lit fraction; convert to "shadow amount"
    // (0 = lit, 1 = full shadow) so we can combine it with the RT
    // result the same way cube.frag does for the rasterised path.
    float self_shadow = (dif > 0.0)
        ? (1.0 - calcShadow(pos + nor * 0.05, sunDir))
        : 1.0;

    // Fake grass-cast shadow: probe the grass mask along the sun's
    // XZ direction from this terrain pixel; cells with grass darken
    // the diffuse term. Strength + sample count + reach are all UI
    // sliders (0 strength = entirely off, no taps).
    float gcs_strength = scene.grass_shadow_params.x;
    float grass_shadow = grassCastShadow(pos, sunDir);
    // Combine multiplicatively with self_shadow. Convert grass shadow
    // amount to a transmittance and multiply (higher self_shadow
    // already = more shadow; same convention here).
    self_shadow = max(self_shadow, grass_shadow * gcs_strength);

    // RT PCSS for castle / boxes / dyn-props. Mirrors cube.frag's
    // terrain-receiver shadow path 1:1 вЂ” the cube.frag terrain
    // shadows look clean, so the same algorithm with the same knobs
    // should work here. Origin biased along the sun direction so
    // the per-pixel FBM normal jitter doesn't enter (was the source
    // of the previous dither).
    //   1. 4 blocker rays in a wide cone
    //   2. PCSS penumbra estimate from avg blocker distance
    //   3. N_s stratified shadow rays in the size-adapted cone
    //   4. max-combine with `self_shadow` so where the FBM says we
    //      are in self-shadow we use that clean value (analogue of
    //      cube.frag's `max(shadow, sh_bake)`).
    // Distance-LOD: skip the RT work past 200 m. Far terrain pays
    // the bake/calcShadow penalty only вЂ” keeps the per-frame ray
    // budget bounded and prevents Windows TDR on a heavy view.
    float dyn_shadow = 0.0;
    float cam_dist_pos = distance(pos, scene.camera_pos.xyz);
    // Smooth-fade RT contributions across the LOD band rather than
    // leaving a visible cliff. lod_far comes from the user slider
    // (fog_band.w); lod_near is 60% of that. Effect ramps from 1.0
    // inside lod_near, to 0.0 at lod_far. Far pixels early-out.
    float lod_far  = max(50.0, scene.fog_band.w);
    float lod_near = lod_far * 0.6;
    float rt_lod_t = 1.0 - smoothstep(lod_near, lod_far, cam_dist_pos);
    bool do_rt = rt_lod_t > 0.005;

    if (dif > 0.0 && scene.rt_flags.x != 0 && do_rt) {
        // base_softness from the global Shadow softness slider,
        // scaled by the terrain-specific scale (terrain_params.w).
        float base_softness = scene.rt_params.x;
        if (scene.terrain_params.w > 0.0) {
            base_softness *= max(0.05, scene.terrain_params.w);
        }
        // Sample count from global Shadow samples slider Г— near
        // multiplier, but capped to 6 for the terrain shader. PCSS
        // on terrain at the user's full slider (13+) was the dominant
        // per-frame RT shadow cost and pushed total per-frame work
        // over the GPU TDR threshold once the TLAS grew with streamed
        // dyn-props. cube.frag PCSS still uses the full slider.
        float near_mult = max(1.0, scene.terrain_extra.y);
        int N_s_cap = max(2, int(scene.terrain_rt_extra.x));
        int N_s = clamp(int(ceil(float(scene.rt_flags.y) * near_mult)),
                        2, N_s_cap);

        vec3 ref_l = abs(sunDir.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                            : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref_l, sunDir));
        vec3 tan_v = cross(sunDir, tan_u);
        // Bias along sun direction (NOT surface normal) вЂ” the FBM
        // normal varies high-frequency per pixel; using it offsets
        // each pixel's shadow cone slightly and produces dither.
        vec3 origin = pos + sunDir * 0.1;
        // Per-pixel + per-frame seed вЂ” TAA averages noisy samples.
        uvec3 seed_base = uvec3(gl_FragCoord.xy, scene.rt_flags.w);

        // 1. Blocker search (4 rays, 4Г— softness cone).
        const int kBlocker = 4;
        float sum_t = 0.0;
        int   hits  = 0;
        for (int i = 0; i < kBlocker; ++i) {
            float br1 = rmRand(seed_base + uvec3(i, 91u, 13u));
            float br2 = rmRand(seed_base + uvec3(i, 19u, 71u));
            float r   = sqrt(br1) * base_softness * 4.0;
            float ph  = 6.28318530718 * br2;
            vec3 j    = (cos(ph) * tan_u + sin(ph) * tan_v) * r;
            vec3 d    = normalize(sunDir + j);
            float t_b;
            if (closest_hit_no_terrain(origin, d, 100.0, t_b)) {
                sum_t += t_b;
                ++hits;
            }
        }
        if (hits > 0) {
            // 2. Penumbra estimate (cube.frag formulation).
            float avg_t = sum_t / float(hits);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0,
                                   clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            float penumbra = clamp(base_softness * scale * 0.6,
                                    base_softness * 0.25,
                                    base_softness * 6.0);

            // 3. Stratified shadow rays.
            int strata = int(ceil(sqrt(float(N_s))));
            float inv = 1.0 / float(strata);
            int taken = 0;
            float blocked = 0.0;
            for (int sy = 0; sy < strata && taken < N_s; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s; ++sx) {
                    float r1 = rmRand(seed_base + uvec3(taken, 11u, 47u));
                    float r2 = rmRand(seed_base + uvec3(taken, 53u, 23u));
                    float u1 = (float(sx) + r1) * inv;
                    float u2 = (float(sy) + r2) * inv;
                    float r  = sqrt(u1) * penumbra;
                    float ph = 6.28318530718 * u2;
                    vec3 j   = (cos(ph) * tan_u + sin(ph) * tan_v) * r;
                    vec3 d   = normalize(sunDir + j);
                    if (any_hit_no_terrain(origin, d, 100.0)) blocked += 1.0;
                    ++taken;
                }
            }
            dyn_shadow = (blocked / float(taken)) * scene.rt_params.w;
        }
    }
    // 4. Max-combine: the FBM self-shadow is the analogue of
    //    cube.frag's heightmap bake вЂ” clean, no noise. Where the
    //    surface is in self-shadow, we take that value verbatim.
    //    Where it's lit, the RT cone fills in castle / cube
    //    contributions. Result reads identical to the rasterised
    //    terrain shadow path.
    float shadow = max(dyn_shadow, self_shadow);
    float sha = 1.0 - shadow;

    // RT ambient occlusion against castle / boxes / dyn-props
    // (mask 0x02 вЂ” terrain BLAS skipped). Mirrors cube.frag's RTAO
    // path: cosine-hemisphere samples around the surface normal,
    // sqrt-shaped raw occlusion, ao_floor remap. AO darkens the
    // ambient term вЂ” corners between walls + boxes get visibly
    // shaded, just like the rasterised path.
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0 && scene.rt_flags.z > 0 && do_rt) {
        int requested = (ao_mode == 1) ? min(scene.rt_flags.z, 2)
                                       : scene.rt_flags.z;
        int N_ao = clamp(requested, 1, 32);
        // Don't halve the AO radius in fast mode on the terrain shader
        // — the user's slider is typically dialled for cube.frag's
        // close-quarters AO (~0.7m) and halving puts it under 0.4m
        // which can't reach a 1m wall to find it as an occluder.
        // Lower-bounded at 2m so ground next to walls / boxes still
        // sees them as occluders.
        float ao_radius = max(scene.rt_params.y, 2.0);

        // Origin lifted along the surface normal a few cm so the
        // FBM-derived hit point doesn't self-intersect a BLAS
        // triangle (e.g. a castle pad).
        vec3 origin_ao = pos + nor * 0.05;
        uvec3 ao_seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);
        int taken = 0;
        float occluded = 0.0;
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rmRand(ao_seed + uvec3(i, 7u, 53u));
            float u2 = rmRand(ao_seed + uvec3(i, 41u, 5u));
            vec3 d = cos_hemi(u1, u2, nor);
            // BLAS occluders only (castle / dyn-props). The earlier
            // terrain_blocks_ray addition pushed per-pixel cost over
            // the GPU TDR threshold during long idle frames; analytical
            // ambient + sky_vis fallback carry the soft cavity look
            // without firing N_ao extra heightfield marches per pixel.
            if (any_hit_no_terrain(origin_ao, d, ao_radius)) occluded += 1.0;
            ++taken;
        }
        // Linear curve (no sqrt) вЂ” see commit 12f9ebc rationale.
        float raw = 1.0 - (occluded / float(taken));
        float ao_floor_v = scene.rt_lod.w;
        float ao_local = mix(ao_floor_v, 1.0, raw);
        // Fade AO toward 1.0 (no occlusion) as we approach the LOD
        // cutoff so distant terrain seamlessly drops back to the
        // analytical-only path.
        ao = mix(1.0, ao_local, rt_lod_t);
    }

    vec3 mate = getMaterial(pos, nor);

    // === Path-traced GI on the raymarched terrain =================
    // Mirrors cube.frag's GI loop: stratified cosine-hemisphere
    // samples, N_bounces of throughput tracking, sun-shadow rays at
    // bounce hits up to gi_shadow_max_bounce, sky fallback on miss.
    // The terrain BLAS is skipped (mask 0x02) for the bounce closest
    // hits вЂ” we only want castle/dyn-prop bounce contributions; a
    // procedural-terrain self-bounce would need running getMaterial
    // at the hit point which is much more expensive than the BLAS
    // material lookup. The sun-shadow ray FROM a bounce hit uses
    // mask 0x01 (the standard shadow-caster mask) so the terrain
    // CAN occlude the sun for a castle wall hit deep in a valley.
    //
    // Energy: cosine-weighted sampling makes each hit's contribution
    // = albedo * Li (the BRDF/ПЂ and cos/ПЂ factors cancel). We tag
    // the raw GI with the surface albedo (`mate`) at the end вЂ” same
    // structure as cube.frag вЂ” and lerp toward a smooth sky-at-N
    // estimate by `gi_softener` for low-sample noise control.
    vec3 gi_indirect = vec3(0.0);
    // Terrain GI sample count capped (was scene.rt_flags2.x directly).
    // 17+ samples * (1 BLAS hit + bounce shadow ray) per pixel was the
    // dominant per-frame RT cost and the smoking gun for the GPU TDR
    // observed during long idle frames as the TLAS grew with streamed
    // dyn-props. Slider still controls; we just clamp to 6 for the
    // raymarched terrain path. Cube.frag GI uses the full slider.
    int N_gi_user = scene.rt_flags2.x;
    int N_gi_cap  = max(0, int(scene.terrain_rt_extra.y));
    int N_gi = (N_gi_user > 0 && do_rt) ? min(N_gi_user, N_gi_cap) : 0;
    if (N_gi > 0) {
        int N_bounces     = max(1, int(scene.terrain_rt_extra.w));
        int gi_shadow_max = int(scene.rt_lod.z);
        // GI search radius вЂ” driven directly by the user's slider.
        // Earlier we forced a min of 80m so first-bounce rays could
        // reach distant castle walls, but combined with N_gi samples
        // the long-ray BLAS traversal pushed total per-frame work
        // over the GPU TDR threshold. Crank gi_radius in the UI to
        // 80m+ if you want stronger color bleed.
        float gi_radius   = max(scene.rt_params2.y, 1.0);
        // Soft sky fill for bounce hits that can't see the sun. Same
        // 5%-of-sky constant as cube.frag вЂ” keeps interior bounces
        // believably dark instead of glowing from a too-bright fill.
        vec3 sky_fill = scene.sky_color.rgb * 0.05;

        int strata = int(ceil(sqrt(float(N_gi))));
        float inv_strata = 1.0 / float(strata);
        int  taken = 0;
        int  first_bounce_misses = 0;
        vec3 sum   = vec3(0.0);
        uvec3 gi_seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);

        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rmRand(gi_seed + uvec3(taken, 73u, 11u));
                float r2 = rmRand(gi_seed + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                vec3 ray_dir    = cos_hemi(u1, u2, nor);
                vec3 ray_origin = pos + nor * 0.05;
                vec3 throughput = vec3(1.0);
                vec3 path       = vec3(0.0);

                for (int b = 0; b < N_bounces; ++b) {
                    float t_hit;
                    int   inst_id;
                    int   prim_id;
                    if (!closest_hit_material_no_terrain(ray_origin, ray_dir,
                                                          gi_radius,
                                                          t_hit, inst_id, prim_id)) {
                        // BLAS miss в†’ sky in this direction. Bound by
                        // the halo's pow(8) so no fireflies. Terrain
                        // self-occlusion is intentionally NOT checked
                        // here вЂ” for short gi_radius (~20 m) on flat
                        // terrain the heightfield march false-blocks
                        // sky-bound rays, killing the GI signal. AO
                        // already carries terrain occlusion via the
                        // ambient term; GI keeps its sky-escape
                        // contribution undamped.
                        path += throughput * sample_sky(ray_dir);
                        if (b == 0) ++first_bounce_misses;
                        break;
                    }
                    int mat_idx = (inst_id == kStaticBlasSentinel)
                                    ? (prim_id / kCubeTrisPerBox)
                                    : inst_id;
                    Material m  = materials[mat_idx];
                    vec3 hit_pos = ray_origin + ray_dir * t_hit;
                    // No vertex normal in ray queries; -ray_dir is a
                    // good local approximation for cosine sampling
                    // and the sun-shadow's nВ·L test (matches cube.frag).
                    vec3 hit_n   = -ray_dir;

                    vec3 hit_light = sky_fill;
                    if (b < gi_shadow_max) {
                        float n_dot_sun = dot(hit_n, scene.sun_direction.xyz);
                        if (n_dot_sun > 0.0) {
                            if (!any_hit_shadow_caster(
                                    hit_pos + hit_n * 0.01,
                                    scene.sun_direction.xyz, 200.0)) {
                                hit_light += scene.sun_color.rgb *
                                             scene.sun_color.a * n_dot_sun;
                            }
                        }
                    }
                    path += throughput * (m.emissive.rgb +
                                          m.color.rgb * hit_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint throughput by the surface albedo, kick
                    // off the next bounce in a cosine hemisphere
                    // around the approximate hit normal.
                    throughput *= m.color.rgb;
                    ray_origin  = hit_pos + hit_n * 0.01;
                    float br1 = rmRand(gi_seed + uvec3(taken * 7 + b, 31u, 17u));
                    float br2 = rmRand(gi_seed + uvec3(taken * 7 + b, 7u, 91u));
                    ray_dir = cos_hemi(br1, br2, hit_n);
                }
                sum += path;
                ++taken;
            }
        }
        vec3 gi_raw = (taken > 0) ? (sum / float(taken)) : vec3(0.0);
        // Sky visibility: fraction of FIRST-bounce rays that escaped
        // all occluders. Used to attenuate the smooth fallback so a
        // surface deep in a cavity doesn't get a bright "sky from
        // normal" lift even when the softener slider is high.
        float sky_vis = (taken > 0)
                          ? (float(first_bounce_misses) / float(taken))
                          : 1.0;
        // Softener: lerp toward a noise-free sky-at-normal estimate.
        // 0 = raw path-traced GI (most accurate, noisiest at low N),
        // 1 = smooth fallback (no per-pixel variance).
        // Both terms are AO-modulated AND sky_vis-modulated so corners
        // and cavities go genuinely dark regardless of softener value.
        float ao_p     = max(scene.ambient.w, 0.5);
        float occl = pow(ao, ao_p) * mix(0.4, 1.0, sky_vis);
        vec3 gi_smooth = sample_sky_atmosphere(nor) * occl;
        float soft = clamp(scene.terrain_extra.z, 0.0, 1.0);
        gi_indirect = mate * mix(gi_raw, gi_smooth, soft) *
                      scene.rt_params2.x;
        // Hard ceiling to keep a single bright bounce (e.g. a sun-
        // facing white box wall) from spiking the HDR + bloom.
        gi_indirect = min(gi_indirect, vec3(4.0));
        // Smooth-fade the GI contribution past 150 m so distant
        // terrain doesn't show a hard boundary where the loop stops.
        gi_indirect *= rt_lod_t;
    }

    vec3 lin = vec3(0.0);
    lin += dif * sha * scene.sun_color.rgb * scene.sun_color.a;
    // Ambient + Fresnel rim term darkened by RTAO. ao_punch pow-curves
    // the AO so > 1 makes mid-AO drop fast (corners punch dark) while
    // 1.0 keeps the original linear behaviour. Driven by the
    // 'AO darkness (raymarch terrain)' slider.
    float ao_punch  = max(scene.ambient.w, 0.5);
    float ao_shaped = pow(ao, ao_punch);
    lin += amb * scene.sky_color.rgb * 0.35 * ao_shaped;
    lin += fre * scene.sky_color.rgb * 0.25 * ao_shaped;

    vec3 col = mate * lin + gi_indirect;

    // Final-colour AO multiplier — dampens the WHOLE pixel (sun term
    // included) by the shaped AO, making corner darkening visibly
    // pronounced even when the sun is hitting the ground. Without
    // this, AO only modulated the ambient term which is small
    // relative to the sun, so the corner-vs-open contrast was
    // imperceptible. Mirrors the visible "darker between objects"
    // look on cube.frag — there the higher-contrast albedos plus
    // GI hit-shading make the effect read; on terrain we have to
    // fold AO into the final colour to match.
    //
    // Strength comes from the user-facing 'AO final strength' slider
    // (terrain_rt_extra.z). 0 = no extra darkening, 1 = full punch.
    // ao_shaped drives the AO contour, so the 'AO darkness' slider
    // tunes the curve and 'AO final strength' tunes how much that
    // curve actually shows up on screen.
    float ao_final_str = clamp(scene.terrain_rt_extra.z, 0.0, 1.0);
    float ao_floor     = 1.0 - ao_final_str;
    col *= mix(ao_floor, 1.0, ao_shaped);

    // Debug viz: replace the final colour with the GI contribution
    // only (scaled). 0 = off, 1 = scaled gi_indirect, 2 = scaled raw
    // gi (no albedo modulation, no softener). Toggled from
    // Settings -> Lighting -> "GI debug viz (raymarch terrain)".
    if (scene.terrain_extra.w > 0.5) {
        col = gi_indirect * 4.0;
    }

    // Wavelength-dependent atmospheric extinction (blue attenuates
    // ~4Г— faster than red), with a sun-tinted fog colour near the sun
    // direction so the horizon glows warm where the sun touches it.
    vec3 extinction = exp(-t * 0.00025 * vec3(1.0, 1.5, 4.0));
    float sundot = clamp(dot(rd, sunDir), 0.0, 1.0);
    // pow(x, 8) → 3 squarings.
    float sn2 = sundot * sundot;
    float sn4 = sn2 * sn2;
    float sn8 = sn4 * sn4;
    vec3 fogCol = mix(vec3(0.55, 0.55, 0.58),
                      vec3(1.0, 0.7, 0.3),
                      0.3 * sn8);
    col = col * extinction + fogCol * (1.0 - extinction);

    // === Volumetric ground fog (real ray-march) =================
    // Multi-step march along the camera ray with Beer-Lambert
    // transmittance accumulation + Henyey-Greenstein phase-weighted
    // sun in-scattering. The Frostbite-style energy-conserving
    // integration:
    //   Sint = (S - S * exp(-Пѓe * dt)) / Пѓe    [exact for constant Пѓ
    //                                            over the segment]
    //   trans *= exp(-Пѓe * dt)
    //   col_out += trans_prev * Sint
    // Fog density = exp(-y / scaleHeight) Г— (1 + slow FBM scroll) so
    // the fog has subtle volumetric variation (wisps in the valleys)
    // instead of a uniform sheet. No per-step volumetric shadow ray
    // вЂ” the HG phase already gives the directional sun glow.
    {
        // Slider-driven strength; skip the entire march when 0.
        float fogStrength = pc.grass_params.x;
        bool  fogGodRays  = pc.grass_params.z > 0.5;
        if (fogStrength > 0.001) {
        // Distance-LOD on the fog march: full 12 steps near the
        // camera, drop to 4 past 150 m. Distant terrain pixels with
        // 16-step fog were the dominant per-pixel cost and the
        // direct cause of the GPU TDR вЂ” far fog can't visibly
        // resolve sub-step density anyway.
        // Linear ramp 12→4 across 100..250 m. The previous binary cliff
        // at 150 m caused wave divergence on neighbouring pixels and
        // visible TAA shimmer along the cutoff line.
        float dist_cam = distance(pos, scene.camera_pos.xyz);
        int kVolSteps = max(4, int(mix(12.0, 4.0,
                                        smoothstep(100.0, 250.0, dist_cam))));
        const float kVolDensityBase   = 0.030;  // Пѓe at full density inside the band
        const float kVolPhaseG        = 0.65;   // forward-scattering bias
        // Band parameters from settings (fog_band slot in scene UBO).
        float fog_y_start = scene.fog_band.x;
        float fog_y_top   = scene.fog_band.y;
        float fog_noise   = clamp(scene.fog_band.z, 0.0, 1.0);

        vec3 fogTint   = vec3(0.78, 0.83, 0.90);
        vec3 sunGlow   = scene.sun_color.rgb * scene.sun_color.a;

        // Henyey-Greenstein phase: forward-peaked when looking near
        // the sun. The denominator goes near-zero as cosThв†’1, so
        // phase can spike past 100 and blow up the bloom + auto-
        // exposure (visible flash + sometimes drives a TDR).
        // Clamp the denominator's growth so phase stays bounded.
        float cosTh = clamp(dot(rd, sunDir), -1.0, 0.985);
        float gg    = kVolPhaseG * kVolPhaseG;
        float phase = (1.0 - gg) /
                      (4.0 * 3.14159265 *
                       pow(1.0 + gg - 2.0 * kVolPhaseG * cosTh, 1.5));
        // Hard ceiling matches a typical "sun glow" magnitude.
        phase = min(phase, 4.0);

        // Cap march length at the surface hit. Sky pixels never reach
        // here (we returned -1 above), so t is always finite.
        float t_end = t;
        float t_max = min(t_end, 600.0);   // cap so we don't waste taps in fog-noise pixels far away
        float dt    = t_max / float(kVolSteps);
        // Per-pixel jitter on the start position breaks step banding
        // вЂ” combined with Phase 5's surface jitter this stays dither-stable.
        // Stable per-pixel jitter (no frame number) so the volumetric
        // step pattern is fixed in screen space вЂ” TAA's spatial
        // filter erases it. Per-frame variation made the dither
        // pattern shift between frames and TAA couldn't keep up.
        float jStart = rmRand(uvec3(gl_FragCoord.xy, 7u));
        float t_v   = dt * jStart;

        vec3  scatter = vec3(0.0);
        float trans   = 1.0;
        for (int i = 0; i < kVolSteps; ++i) {
            vec3 p = ro + rd * t_v;
            // Density profile: 0 below y_start, ramps up over 1 m,
            // 1 inside the band, falls off over 4 m above y_top.
            // Compact band that hugs the valleys instead of a global
            // exp falloff.
            float profile = smoothstep(fog_y_start, fog_y_start + 1.0, p.y) *
                            (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, p.y));
            // 3-octave FBM-style wisp pattern. Modulates the density
            // so the layer reads as wispy ground fog instead of a
            // perfect sheet. Slow-scrolling for animated drift.
            vec2 q = p.xz * 0.020 + vec2(scene.water_params.w * 0.05);
            float w = 0.55 * noise2(q)
                    + 0.30 * noise2(q * 2.13)
                    + 0.15 * noise2(q * 4.27);
            // Centred around 0.5 so noise = 1.0 в†’ "no modulation" and
            // noise = 0 в†’ density Г—0.2 (thin patch). Strength slider
            // controls how much the noise can carve gaps.
            float wisp = mix(1.0, 2.0 * w, fog_noise);
            wisp = clamp(wisp, 0.2, 2.0);
            float sigma_e = kVolDensityBase * profile * wisp * fogStrength;

            // Phase 7 вЂ” fog self-shadow / god-rays. Step a few small
            // distances toward the sun and accumulate optical depth
            // through the same density field. Surface (terrain /
            // castle) shadows the fog because the surface RT shadow
            // `sha` already gates the unshadowed mass; the per-step
            // sample additionally darkens fog that's behind a fog-
            // density gradient, giving the godray look.
            float lightTrans = sha;
            if (fogGodRays) {
                // 2 inner steps × 1-octave wisp. Was 4 steps × 3 octaves
                // = 12 noise2 calls per fog tap; with ~12 outer taps
                // that was 144 hash evals per pixel for the godray work
                // alone. Single-octave wisp is the dominant low-freq
                // mode; the higher-freq components alias-tolerate at
                // godray resolution under TAA.
                const int kFogShadowSteps = 2;
                float ds = 9.0;          // bigger base to cover same length in fewer steps
                float ld = ds * 0.5;
                for (int j = 0; j < kFogShadowSteps; ++j) {
                    vec3 sp = p + sunDir * ld;
                    float prof_s = smoothstep(fog_y_start, fog_y_start + 1.0, sp.y) *
                                    (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, sp.y));
                    vec2 qs = sp.xz * 0.020 + vec2(scene.water_params.w * 0.05);
                    float w_s = noise2(qs);
                    float wisp_s = clamp(mix(1.0, 2.0 * w_s, fog_noise), 0.2, 2.0);
                    float sig_s = kVolDensityBase * prof_s * wisp_s * fogStrength;
                    lightTrans *= exp(-sig_s * ds);
                    ds *= 1.7;          // bigger growth so 2 steps still cover far range
                    ld += ds;
                    if (lightTrans < 0.02) break;
                }
            }

            // Single-scattering: Пѓs в‰€ Пѓe (assume cloudy isotropic
            // scattering, no absorption). Phase function gives the
            // sun-aligned directional component; ambient fogTint
            // keeps back-lit fog from going pitch-black.
            vec3  Lin   = sunGlow * phase * lightTrans + fogTint * 0.25;
            float seg  = exp(-sigma_e * dt);
            vec3  Sint = (Lin - Lin * seg) / max(sigma_e, 1e-4);
            scatter += trans * Sint * sigma_e;   // re-multiply Пѓs because we factored Пѓe out
            trans   *= seg;
            t_v     += dt;
            if (trans < 0.02) break;             // early-out вЂ” fully foggy
        }
        // Apply: surface attenuated by trans, scatter added in front.
        col = col * trans + scatter;
        }   // fogStrength > 0
    }

    // Write the hit-point depth so rasterised geometry that wrote
    // depth earlier in the same pass occludes the terrain correctly.
    // Cap below 0.9999 вЂ” the compose pass treats depth >= 0.99999 as
    // a sky pixel and paints `sample_sky(dir)`. For terrain hits past
    // ~800 m the projected depth crosses the threshold and compose
    // overwrites our colour with the skybox sampled below the horizon
    // (which is black for typical HDRIs). Result: distant raymarch
    // terrain renders as pure black. Capping keeps compose's sky-
    // detection from misfiring on far terrain hits.
    vec4 clip = pc.mvp * vec4(pos, 1.0);
    gl_FragDepth = min(clip.z / max(clip.w, 1e-4), 0.9998);

    outColor = vec4(col, 1.0);
    // Screen-space motion vector for TAA. World position is stationary
    // вЂ” only the camera moves вЂ” so prev_uv comes from prev_view_proj
    // applied to the same world hit point. Without this, TAA cannot
    // reproject and per-frame PCSS / fog jitter shows up as moving
    // square dither artefacts.
    {
        vec2 current_uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
        vec4 prev_clip = pc.prev_mvp * vec4(pos, 1.0);
        if (prev_clip.w > 0.0) {
            vec2 prev_ndc = prev_clip.xy / prev_clip.w;
            vec2 prev_uv  = prev_ndc * 0.5 + 0.5;
            outMotion = current_uv - prev_uv;
        } else {
            outMotion = vec2(0.0);
        }
    }
}
