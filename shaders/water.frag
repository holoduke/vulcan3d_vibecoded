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
// Perspective-correct world position of the water surface at this
// pixel, output by water.vert from the actual plane mesh vertices.
// Replaces the per-frame inv(view_proj) reconstruction below — that
// was the source of the "water orientation drifts with camera" wobble.
layout(location = 3) in vec3 vWPos;

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
// FULL baked terrain height (NOT the delta) — the exact array the
// CDLOD mesh + physics are built from. The water path samples this in
// mesh-terrain mode so its shore / showthrough / reflection /
// self-shadow follow the real rasterised mesh surface with zero
// procedural FBM evaluation. Same dim+1 / 2048 m layout as binding 8.
layout(set = 0, binding = 17) uniform sampler2D u_terrain_height_full;
// Grass eligibility mask. R8 1024² covering the same 2048 m square as
// u_terrain_height. CPU-baked once at level load (presence + slope from
// the same FBM the GLSL terrain uses, then sample_terrain_height for an
// accurate slope finite-difference). Both this shader and
// grass_raymarch.frag sample binding 13 — single source of truth for
// "is there grass here", and avoids the per-step 9-cell noised() storm
// in the grass map() loop.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
// Fog wisp pattern (R8 256², REPEAT). One textureLod() replaces the
// 3-octave noise2() weighted sum the fog density + godray probe
// loops used to compute per-tap. UV maps q→texture as `q / 16.0` so
// the texture tiles every 16 q-units (≈800 m of world space).
layout(set = 0, binding = 14) uniform sampler2D u_fog_wisp;
const float kHeightmapSide = 2048.0;
// Forward decl — terrainDetailNoise is defined after `noised` + `m2`
// (further down in the file) so we can reuse them. We use a forward
// declaration here so sampleHeightDelta can call it without moving
// the whole noise toolkit above the declarations of u_terrain_height.
float terrainDetailNoise(vec2 worldXZ);
vec3  noised(in vec2 p);   // single-octave value noise + analytical derivatives

float sampleHeightDelta(vec2 worldXZ) {
    // Was: 4 compare branches before the fetch. Replaced with clamp +
    // textureLod 0 — sampler is CLAMP_TO_EDGE and the heightmap delta
    // is ~0 at the world boundary (no sculpting past the edge), so a
    // clamped read at out-of-world XZ returns the right value too.
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_terrain_height, uv, 0.0).r;
}

// === Mesh-terrain height (water-only mode) ==========================
// Bilinear lookup of the FULL baked heightmap — identical to the
// surface the rasterised CDLOD mesh draws. Used INSTEAD of the FBM
// terrainM* functions throughout the water path when the engine runs
// the water-only pass over mesh terrain, so the water never reflects
// or shows procedural noise that disagrees with the mesh.
float meshHeight(vec2 worldXZ) {
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_terrain_height_full, uv, 0.0).r;
}
vec3 meshNormal(vec2 worldXZ) {
    const float e = 1.5;   // ≈ heightmap cell — smooth mesh-scale normal
    float hL = meshHeight(worldXZ - vec2(e, 0.0));
    float hR = meshHeight(worldXZ + vec2(e, 0.0));
    float hD = meshHeight(worldXZ - vec2(0.0, e));
    float hU = meshHeight(worldXZ + vec2(0.0, e));
    return normalize(vec3(hL - hR, 2.0 * e, hD - hU));
}
// raymarchMesh / raymarchReflectMesh / calcShadowMesh are defined
// after the Scene UBO block (they read scene.terrain_local_info.y for
// the air early-out) — see below, just after raymarchReflect().

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

// Sample-with-precomputed-basis variant. The tangent frame around `n`
// only depends on `n`, which is constant across an N-sample GI/AO
// loop — caller computes the basis once outside the loop and we
// avoid the cross+normalize+cross per sample.
vec3 cos_hemi_basis(float u1, float u2, vec3 n, vec3 t, vec3 b) {
    float r   = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3  d   = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
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
    // Shore-foam highlight band:
    //   water_foam_color.rgb = tint, .a = strength (0 disables)
    //   water_foam_params.x  = band width (m of noise-free depth)
    vec4 water_foam_color;
    vec4 water_foam_params;
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
    // Layout padding: must mirror SceneUBO in internal.h. Grass shore
    // tint slot is consumed by grass_raymarch.frag; declared here so
    // terrain_shore_color (the next field) lands at the right offset.
    vec4 grass_shore_color;
    vec4 grass_shore_params;
    // Shoreline TERRAIN tint — applied to bare ground in getMaterial.
    //   terrain_shore_color.rgb = tint colour
    //   terrain_shore_color.w   = strength (0..1; 0 disables)
    //   terrain_shore_params.x  = fade distance (m above water level)
    vec4 terrain_shore_color;
    vec4 terrain_shore_params;
    // Distance fog — see SceneUBO comment in internal.h.
    vec4 distance_fog_color;   // rgb + strength a (0 = off)
    vec4 distance_fog_params;  // x density, y start_dist, z height_top, w max_alpha
    // GENERAL terrain shore tint — bare-terrain shore band.
    vec4 terrain_shore_general_color;
    vec4 terrain_shore_general_params;
    // Sand base colour for the beach blend — was hardcoded.
    vec4 terrain_sand_color;
    // River water style extras — see SceneUBO comment in internal.h.
    vec4 water_river_extra;     // x: flow angle (rad), y: time speed,
                                // z: detail scale, w: foam amount
    vec4 water_river_extinct;   // rgb: ext colour, w: density
    // .x = ReSTIR enabled (always 0 — quarantined), .y = M_max,
    // .zw = jitter UV delta (cur_jitter − prev_jitter) for FSR3
    // motion-vec output, see descriptors.cpp.
    vec4 restir_params;
    // SPOM tuning — unused by terrain raymarch but present on the C++
    // side; declared so the following offsets match.
    vec4 spom_params;
    vec4 terrain_local_info;          // reserved
    // 32×32 hi-Z max-cell grid (64 m cells, 1024 floats packed 4 per
    // vec4). Each cell = max terrain height in that footprint + 15 m
    // safety. The marcher skips whole cells a rising ray is above.
    vec4 terrain_max_grid[256];
} scene;

// Max terrain height of the 64 m cell containing world XZ. Branch-free
// clamp + packed-vec4 unpack. kHeightmapSide is the 2048 m world span.
float cellMaxHeight(vec2 wp) {
    vec2 uv = (wp / kHeightmapSide) + vec2(0.5);
    ivec2 c = clamp(ivec2(uv * 32.0), ivec2(0), ivec2(31));
    int idx = c.y * 32 + c.x;                  // 0..1023
    return scene.terrain_max_grid[idx >> 2][idx & 3];
}

// Standard exponential² distance fog. Returns 0..1 mix weight toward
// `distance_fog_color.rgb`. Caller does the actual mix so it can
// run AFTER all atmospheric / specular passes.
//   p_world: shaded point in world space
//   cam:     camera position
float distanceFogAmount(vec3 p_world, vec3 cam) {
    float strength = scene.distance_fog_color.a;
    if (strength < 1e-3) return 0.0;
    float density   = scene.distance_fog_params.x;
    float start_d   = scene.distance_fog_params.y;
    float height_top = scene.distance_fog_params.z;
    float max_alpha  = scene.distance_fog_params.w;
    float d_raw = max(0.0, length(p_world - cam) - start_d);
    // Exp² falloff — natural-looking atmospheric fog. Cheaper than
    // Beer-Lambert with no visible difference at typical density.
    float dx = d_raw * density;
    float fog = 1.0 - exp(-dx * dx);
    if (height_top > 0.5) {
        // Optional height fog: thin out above height_top so mountain
        // peaks stay visible above the fog layer.
        float h_w = 1.0 - smoothstep(0.0, height_top, p_world.y);
        fog *= h_w;
    }
    return clamp(fog * strength, 0.0, max_alpha);
}

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

// Value Noise + ∂n/∂x, ∂n/∂y. Hermite smoothstep `3t²-2t³` for C¹
// continuity; `du = 6t(1-t)` is the chain-rule derivative used by the
// FBM-erosion accumulator. (Reverted from a quintic experiment —
// kept the proven cubic that matches the look the project shipped
// with and stays consistent with the CPU rm_noised bake.)
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

// High-frequency FBM-eroded noise used by sampleHeightDelta to paint
// procedural detail onto user-brushed cells. Same n.x / (1 + |∇d|²)
// erosion recipe as terrainM, but with a much higher base scale (0.05
// vs 0.003) so meter-scale ridges actually appear within the brush
// footprint. Centred ≈ 0 so it adds equal positive/negative variation
// — doesn't shift the mean brushed height. Forward-declared above
// sampleHeightDelta; the body lives here so it can use the noise
// helpers + m2 declared above.
float terrainDetailNoise(vec2 worldXZ) {
    vec2 p = worldXZ * 0.05;   // 20 m base; ≤ 30 cm at 6 octaves
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    for (int i = 0; i < 6; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    return a - 0.5;
}

// Plateau blend factor вЂ” 1 inside the gameplay plateau region, 0
// outside, smooth ramp across `kPlateauBlend` metres so the FBM
// surroundings ease into the flat castle pad. Mirrors what
// generate_heightmap() does in the rasterised path so the visible
// terrain shape matches what physics / castle assume.
const float kPlateauBlend = 20.0;  // smooth ramp from castle-flat to FBM mountains
float plateauWeight(vec2 worldXZ) {
    vec2 c = pc.color.xy;
    float ext = pc.color.z;
    vec2 d = abs(worldXZ - c) - vec2(ext);
    float dout = max(max(d.x, 0.0), max(d.y, 0.0));
    return 1.0 - smoothstep(0.0, kPlateauBlend, dout);
}

// Plateau ground-detail noise. The macro terrain FBM uses scales
// 0.003 → 0.096 (333 m → 10 m wavelengths) — its SMALLEST feature is
// ~10 m, so scaling it down for plateau use gives invisibly-small
// "detail" (a few cm). What the player perceives as "rocky ground"
// on the surrounding mountains is those 10 m ridges seen from far
// away; on the flat plateau they're not there at all.
//
// This separate noise fires AT FINER SCALES (2.5 m → 1.25 m
// wavelengths — safely above the ~0.8 m march-step alias floor)
// and adds visible "ground texture" to plateau cells without
// affecting anything outside. 2 octaves of plain noise (no
// derivative erosion damper — we WANT the high freqs to contribute
// fully). m2 (per-octave rotation declared further down) can't be
// forward-referenced — inlined.
float plateauGroundDetail(vec2 wp) {
    // 50/50 mix of plain noise (rolling) + soft ridge (1 - |2n-1|)
    // — gives natural rock character without the glass-flat smoothness
    // of pure FBM or the spike-needle look of multifractal-weighted
    // ridges. No squaring, no weight gating. 2 octaves at safe
    // wavelengths (2.5 m + 1.25 m).
    float a = 0.0, b = 1.0;
    vec2 p = wp * 0.4;
    for (int i = 0; i < 2; ++i) {
        float n = noised(p).x;
        float ridge  = 1.0 - abs(2.0 * n - 1.0);     // ∈ [0, 1]
        float smooth_n = n;                            // ∈ [0, 1]
        a += b * (mix(smooth_n, ridge, 0.5) - 0.5);  // re-centre on 0
        b *= 0.5;
        vec2 q = vec2( 0.8 * p.x - 0.6 * p.y,
                       0.6 * p.x + 0.8 * p.y);
        p = q * 2.0;
    }
    return a;
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
    // Option-2 brush amplifier: the brush's smooth bilinear delta
    // ALSO scales the local FBM amplitude. Brushed cells get more of
    // the SAME procedural FBM (per-pixel detail, same character as
    // the surrounding terrain) instead of a flat smooth bump. The
    // raise/lower offset is preserved at the end via `+= delta` so
    // the user can still sculpt arbitrary shapes; the amplifier is
    // additional, not destructive. AMP = 0.05 → 1 m raise gives +5 %
    // FBM amplitude; capped so a 60 m raise doesn't 4× the local
    // mountains. Inside the plateau the FBM is already mixed out, so
    // the amplifier has no effect there (plateau stays flat under
    // the brush — only the offset fires).
    float h = a * TERRAIN_HEIGHT;
    // Plateau blend — but instead of mixing toward a glass-flat
    // plateau_h, mix toward `plateau_h + small_detail_FBM` so the
    // plateau surface inherits the same procedural character as the
    // surrounding terrain (just smaller amplitude). Uses the SAME
    // accumulated `a` value so it's free, and reads as a continuous
    // scale-down of the surrounding mountains. The castle footprint
    // (|xz| < 11 m) is already discarded for raymarched terrain
    // earlier in the file, so castle siting isn't affected by this
    // detail; only the courtyard / plateau-rim area gains texture.
    // ±~1.5 m of FINE ground texture (2.5 m + 1.25 m wavelengths)
    // instead of scaled-down macro FBM (whose smallest feature is
    // 10 m → invisible at plateau scale).
    // Plateau is now tiny (11.5 m, hidden under castle) — no need
    // for extra ground-detail noise here.
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    h += sampleHeightDelta(wp);
    return h;
}

// Variant of terrainM_lod that ALSO returns the accumulated gradient
// squared norm (dot(d,d)) — the local Lipschitz constant for the FBM
// at this pixel. The marcher uses it as a per-step "1/sqrt(1+slope²)"
// overstep multiplier so flat ground takes bigger strides without
// risking overshoot on cliffs. Idea from Peter Stefek's heightfield
// raymarch writeup; proven safe because we already track d for the
// derivative-erosion damp term, so the data is free.
//
// Returns vec2(height, gradient_sq_in_noise_space). Height is the
// SAME value terrainM_lod would return.
vec2 terrainM_lod_d(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
    return vec2(h, dot(d, d));
}

// Distance-LOD variant: drops octaves smoothly as the ray distance
// grows. The full octave count is the user's pc.tex_params.z; we
// fade down to ~2 octaves past 600 m. Same outer wrapping (plateau
// blend + sculpt delta) so close vs far results stay continuous —
// only the inner FBM converges to a coarser sum past the LOD start.
// Uses a uniform `kMaxOct` cap with an early-break so the loop body
// stays branch-friendly for the GPU.
// LOD ramp endpoints + minimum-octave floor come from push constants:
//   emissive.y = lod_near_m (full octaves below this distance)
//   emissive.z = lod_far_m  (lod_min_oct octaves above this distance)
//   emissive.w = lod_min_oct (floor, typically 2..8)
// Smoothstep ramp between the two endpoints.
float terrainM_lod(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    // Fractional octave count — ceil() to include the partially-faded
    // top octave; `frac` is the [0..1] weight that octave gets so the
    // FBM is C0-continuous across the integer threshold.
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    // Plateau is now tiny (11.5 m, hidden under castle) — no need
    // for extra ground-detail noise here.
    float tpl = plateauWeight(wp);
    h = mix(h, pc.color.w, tpl);
    // Sculpt-delta texture fetch is sub-pixel past ~80 m and the
    // delta itself bilinear-samples to near-zero outside the brush
    // ring. Skip the texture tap on far raymarch steps — biggest
    // single texture-bandwidth win on terrain pixels.
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
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
    // Plateau is now tiny (11.5 m, hidden under castle) — no need
    // for extra ground-detail noise here.
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    h += sampleHeightDelta(wp);
    return h;
}

// Variant of terrainH_lod that takes a precomputed plateau weight,
// so callers like calcNormal — which evaluates the FBM at three
// points within ~2 cm of each other — can compute plateauWeight
// once and pass it in instead of recomputing the same value 3×.
// Saves 2 × (2 abs + smoothstep) per shaded terrain pixel.
float terrainH_lod_pw(vec2 p, float ray_t, float tpl) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.w);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 32;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    h = mix(h, pc.color.w, tpl);
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
    return h;
}

// Distance-LOD variant of terrainH. calcNormal calls this 3× per hit
// pixel — the dominant FBM cost — so dropping octaves at distance is
// the single biggest win for raymarch perf at typical horizon views.
// Same LOD ramp as terrainM_lod (pc.emissive.{y,z,w}) so march hits
// and normal samples stay coherent.
float terrainH_lod(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.w);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    // Fractional-octave fade — same scheme as terrainM_lod so the
    // marcher and normals agree across the LOD threshold.
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 32;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    // Same far-pixel sculpt skip as terrainM_lod above.
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
    return h;
}

// Generous headroom: sampleHeightDelta() (sculpt strokes + disk
// overlays) can push the FBM-derived terrain well above TERRAIN_HEIGHT.
// Underestimating max height triggers the open-air early-out in
// raymarch() while the camera is still inside the terrain volume,
// producing the "everything broken at high altitude" symptom.
//
// Must be the GLOBAL conservative cap. Tried clamping by a per-frame
// camera-local max (terrain_local_info.x) but that's unsafe: a shallow
// rising ray escapes once it's above the LOCAL max, yet a distant peak
// taller than that local max (outside the sample radius) would still
// be on the ray's path → the peak vanishes. A correct local cap needs
// a per-cell max sampled ALONG the ray (proper maximum-mipmap), not a
// single camera-vicinity value. terrain_local_info stays wired on the
// C++ side for when that lands; it is intentionally unused here.
float terrainMaxHeight() { return max(TERRAIN_HEIGHT, pc.color.w) + 80.0; }

// Shadow-march FBM. Identical to terrainM_lod (same user octave
// count + LOD ramp) EXCEPT the sampleHeightDelta texture tap is
// dropped — the sculpt overlay is sub-pixel for shadow boundaries
// and would just be filtered out by TAA anyway. Saves one texture
// fetch per shadow march step (up to 32 per pixel) without changing
// what blocks light. Octave count remains the user's quality
// setting so they keep control over shadow sharpness.
//
// Plateau term is kept (otherwise the castle area's sun shadow
// would mismatch the primary terrain there).
float terrainM_shadow(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    return h;
}

// _d variant exposing dot(d,d) for the shadow-march local-Lipschitz
// overstep — same trick as the primary marcher uses (terrainM_lod_d).
// Bigger shadow-march jumps on flats while keeping conservative
// behaviour on cliffs.
vec2 terrainM_shadow_d(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    return vec2(h, dot(d, d));
}

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

// Distance-LOD variant — same octave-fade scheme as terrainH_lod and
// the same far-pixel sculpt skip. calcNormal uses this past the
// near-detail band so the normal evaluation drops from 3 × kOct
// finite-difference taps down to 1 × oct_lod analytic-gradient call.
// At 30+ m the per-octave damp-derivative term dropped by the chain
// rule is sub-pixel — verified by comparing renders at lod_far and
// against the 3-tap path; no visible difference past the gate.
vec3 terrainH_grad_lod(vec2 wp, float ray_t) {
    vec2 p = wp * TERRAIN_SCALE;
    mat2 J = mat2(TERRAIN_SCALE, 0.0, 0.0, TERRAIN_SCALE);
    int oct_full = int(pc.tex_params.w);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    vec2 grad = vec2(0.0);
    const int kMaxOct = 32;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        float damp = 1.0 / (1.0 + dot(d, d));
        a += b * n.x * w * damp;
        grad += b * w * damp * (transpose(J) * n.yz);
        b *= 0.5;
        p = m2 * p * 2.0;
        J = m2 * J * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float tpl = plateauWeight(wp);
    h = mix(h, pc.color.w, tpl);
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
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
// raymarch with optional max-distance cap. Pass `t_max < 0` for
// unbounded (default 1500 m). Caller can pass a tighter bound — e.g.
// the water-plane intersection distance — so the loop bails as soon
// as any terrain hit past the bound is irrelevant.
float raymarch(vec3 ro, vec3 rd, float t_max) {
    float maxH = terrainMaxHeight();
    float t = 0.0;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    if (ro.y > maxH) {
        // Skip the open-air segment by jumping to the upper bound plane.
        t = (ro.y - maxH) / (-rd.y);
    }
    float t_limit = (t_max > 0.0) ? min(t_max, 1500.0) : 1500.0;
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
        // NOTE: the per-cell hi-Z max-grid skip was REMOVED. It went
        // through four correctness regressions (black dots, FBM-budget
        // exhaustion, and twice skipping real medium-distance low
        // terrain so sky showed through) and never became reliable —
        // the coarse 64 m cell max can't safely bound the fine FBM
        // surface a ray grazes at medium distance regardless of the
        // safety margin. The proven plain marcher below is correct;
        // the local-Lipschitz overstep (terrainM_lod_d's grad term)
        // and the distance-octave LOD are the kept perf wins. The
        // terrain_max_grid UBO data + C++ bake stay wired but unused
        // (zero runtime cost) in case a correct screen-space hi-Z
        // prepass is built later.
        // Distance-LOD on the FBM octave count — full octaves close
        // to camera, smoothly fading down at distance (sub-pixel
        // detail) for a perf saving on long rays. (Reverted to the
        // plain terrainM_lod; the _d grad variant is unused now.)
        float h = pos.y - terrainM_lod(pos.xz, t);
        // P7: loosen the hit threshold with distance. At grazing angle a
        // typical pixel covers 0.3-1 m of world space past 200 m, so
        // demanding sub-millimetre convergence wastes ~30 % of the
        // march steps on far rays. mix 0.0015 → 0.005 over [150, 600] m.
        float hit_eps = mix(0.0015, 0.005, smoothstep(150.0, 600.0, t)) * t;
        if (abs(h) < hit_eps) break;
        if (t > t_limit) return -1.0;
        // Phase 6 вЂ” relaxation cone-stepping. Step grows with
        // distance so the same ray covers more ground in fewer
        // iterations. 0.7 attenuation on h prevents penetration
        // when the relaxation factor over-shoots.
        // Distance-ramped step factor — past 200 m the FBM is LOD'd
        // down to lod_min_octaves so the field is much smoother and
        // big strides won't punch through fine detail. Up to 2.5×
        // close-up step factor by 600 m → roughly halves the worst-
        // case grazing-ray iteration count.
        float kStep_eff = kStep * mix(1.0, 2.5, smoothstep(200.0, 600.0, t));
        // (Reverted the local-Lipschitz overstep experiment — back to
        // the proven conservative kStep_eff * h advance.)
        float advance;
        if (relax) {
            float rl = max(t * 0.02, 1.0);
            advance = kStep_eff * h * rl * 0.7;
        } else {
            advance = kStep_eff * h;
        }
        // Once h goes negative we've crossed the surface — the next step
        // would walk us back, and with rl > 1 the back-step grows enough to
        // ping-pong indefinitely until kSteps caps. Break out instead so
        // the caller gets the closest approach.
        if (advance < 0.0) return t;
        t += advance;
    }
    // Loop exhausted without converging. The fall-through is a real
    // hit when the ray is still close to the surface (just slow to
    // converge under the LOD'd FBM at distance), but NOT a real hit
    // when the ray has clearly gone above any possible terrain — that
    // produces ghost terrain projected into the sky.
    vec3 final_pos = ro + t * rd;
    float final_h  = final_pos.y - terrainM_lod(final_pos.xz, t);
    // Treat as sky if we ended up well above the surface (>5m) AND
    // we were rising. Below that → accept as a hit (the FBM-LOD just
    // didn't converge tightly).
    if (final_h > 5.0 && rd.y > 0.0) return -1.0;
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
    // Far-distance fast path: derive the normal from screen-space
    // derivatives of the hit position itself (`dFdx(pos)` and
    // `dFdy(pos)` are the tangents along screen x/y). Zero terrainH
    // calls vs the 3× finite-difference path. Per-quad-constant — gives
    // a faceted look up close — but at lod_far_m+ the FBM is LOD'd to
    // few octaves anyway, so the surface has no per-pixel detail to lose.
    // Threshold is pc.emissive.z (the existing lod_far_m). Past this we
    // already snap to lod_min_octaves on the FBM, so swapping the normal
    // method costs nothing visible.
    if (t > pc.emissive.z) {
        vec3 dpdx = dFdx(pos);
        vec3 dpdy = dFdy(pos);
        vec3 Nfast = cross(dpdy, dpdx);
        // Cheap sign-flip in case the quad winding lands us pointing
        // into the ground.
        if (Nfast.y < 0.0) Nfast = -Nfast;
        return normalize(Nfast);
    }
    // 3-tap finite-difference is the only path that preserves the
    // per-octave `1/(1+|d|²)` damp variation — that variation IS what
    // makes rocks visible in snow and defines mountain ridges. Two
    // attempts at the analytic gradient (terrainH_grad_lod) were tried
    // (30 m gate and 200 m gate); both visibly lost detail. Until a
    // proper analytic gradient WITH the damp-derivative term is
    // implemented (needs running Hessian alongside the gradient — see
    // IQ's morenoise article), this stays at 3-tap FD. The cheaper
    // sampleHeightDelta gate inside the LOD FBM funcs is kept —
    // that's invisible past 80 m.
    float eps = 0.02 + 0.0008 * t;
    // The 3 taps are within ~2 cm of each other — well below the
    // plateau blend's 20 m smoothstep width — so plateauWeight is
    // identical at all three points. Compute it once, pass it into
    // the FBM via terrainH_lod_pw, save 2 redundant evaluations.
    float tpl = plateauWeight(pos.xz);
    float hC = terrainH_lod_pw(pos.xz, t, tpl);
    float hR = terrainH_lod_pw(pos.xz + vec2(eps, 0.0), t, tpl);
    float hU = terrainH_lod_pw(pos.xz + vec2(0.0, eps), t, tpl);
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
    int kSteps = int(pc.tex_params.y);
    // Receiver-distance shadow budget: shadows on far receivers are
    // sub-pixel at typical FOV — even 4 taps look identical to 96
    // after TAA absorbs the residual noise. Was already capped to
    // 10/6 past 200/500 m by P20; tighten further:
    //   8 past 80 m, 4 past 200 m. ~3× faster on long-shot vistas.
    float recv_dist = length(pos - scene.camera_pos.xyz);
    if (recv_dist > 200.0) kSteps = min(kSteps, 4);
    else if (recv_dist > 80.0)  kSteps = min(kSteps, 8);
    else if (recv_dist > 40.0)  kSteps = min(kSteps, 16);
    // Skip the receiver-adjacent first step. At t = 0.05 we're 5 cm
    // above the surface we just shaded, so terrain(p.xz) ≈ pos.y →
    // h ≈ 0.05·sun.y → res update is bounded near 1.0 (no shadow
    // contribution). Just go straight to min_step (the smallest
    // useful step size we'd take next anyway). Saves 1 FBM call per
    // shadowed pixel — shadow pixels are most of the screen so the
    // sum is meaningful.
    // min_step is a function of recv_dist only — loop-invariant. Hoist out
    // of the per-iteration body (was being recomputed up to kSteps times,
    // each call a mix+smoothstep). Single value used both for the start
    // offset and the clamp inside the loop — the two callsites computed
    // the same number, so collapse to one.
    float min_step = mix(0.5, 4.0, smoothstep(100.0, 500.0, recv_dist));
    float t   = max(0.5, min_step);
    for (int i = 0; i < kSteps; ++i) {
        vec3 p = pos + t * sunDir;
        // Coarse FBM (same user octaves, no sculpt-delta tap).
        // Local-Lipschitz overstep doesn't help here — the shadow
        // march already uses full `h` per step (no global kStep
        // factor to shrink), so the only effect would be SMALLER
        // steps on cliffs (i.e. a regression). Keep the simple
        // h-clamp behaviour.
        float h = p.y - terrainM_shadow(p.xz, t);
        if (h < 0.001) { res = 0.0; break; }
        res = min(res, k * h / t);
        // Bigger min-step and lower hard-cap at distance — same logic
        // as the primary march. Combined with the receiver-distance
        // step budget above, distant shadow contributions converge in
        // 4 jumps instead of crawling 24+ tiny steps.
        t += clamp(h, min_step, 100.0);
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
    vec3 sand  = scene.terrain_sand_color.rgb;
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
    // Far anchor from rt_.grass_ground_tint_far_distance (UI slider,
    // packed into grass_color_ground_far.w). Near anchor is 0 so the
    // close→far ramp is one continuous gradient — no slope-spike
    // inflection band that reads as a hard line. Matches grass.
    float tint_far  = max(50.0, scene.grass_color_ground_far.w);
    float ground_t  = smoothstep(0.0, tint_far, cam_d);
    vec3 ground_base = mix(scene.grass_color_ground.rgb,
                            scene.grass_color_ground_far.rgb,
                            ground_t);
    vec3 grass_top  = mix(ground_base, ground_base * 0.7, nz);
    float tint_str  = clamp(scene.grass_color_ground.w, 0.0, 1.0);
    col = mix(col, grass_top, grass_amt * tint_str);

    // Two-region shore tint — grass-area gets one colour, bare-
    // terrain gets another. Both fade smoothly with height above the
    // water level. Each is gated by grass_amt so they never compete.
    float h_above_water = max(0.0, pos.y - scene.water_params.y);
    {
        // Grass-area shore tint (rt_.terrain_shore_*).
        float ts_strength = scene.terrain_shore_color.a;
        if (ts_strength > 1e-3 && grass_amt > 0.001) {
            float ts_fade = max(0.05, scene.terrain_shore_params.x);
            float ts_w = 1.0 - smoothstep(0.0, ts_fade, h_above_water);
            col = mix(col, scene.terrain_shore_color.rgb,
                      ts_w * ts_strength * grass_amt);
        }
    }
    {
        // Bare-terrain shore tint (rt_.terrain_shore_general_*).
        float gt_strength = scene.terrain_shore_general_color.a;
        if (gt_strength > 1e-3 && grass_amt < 0.999) {
            float gt_fade = max(0.05, scene.terrain_shore_general_params.x);
            float gt_w = 1.0 - smoothstep(0.0, gt_fade, h_above_water);
            col = mix(col, scene.terrain_shore_general_color.rgb,
                      gt_w * gt_strength * (1.0 - grass_amt));
        }
    }
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
        float h = pos.y - terrainM_lod(pos.xz, t);
        if (abs(h) < 0.005 * t) break;     // looser threshold than primary
        if (t > 800.0) return -1.0;        // shorter range than primary
        t += kReflStep * h;
    }
    return t;
}

// === Mesh-terrain heightmap marches (water-only mode) ===============
// Defined here (after the Scene UBO) so they can read
// scene.terrain_local_info.y = mesh max height + pad for the empty-air
// early-out. The baked mesh is smooth (bilinear texels), so an
// aggressive height-proportional step + the air skip keep these cheap
// — the same tricks the FBM marcher uses to stay off the TDR watchdog.
float raymarchMesh(vec3 ro, vec3 rd, float t_max) {
    float maxH = scene.terrain_local_info.y;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    float t = 0.0;
    if (ro.y > maxH) t = (ro.y - maxH) / max(-rd.y, 1e-4);
    for (int i = 0; i < 96; i++) {
        vec3 p = ro + rd * t;
        if (p.y > maxH && rd.y >= 0.0) return -1.0;
        float h = p.y - meshHeight(p.xz);
        if (h < 0.012 * t + 0.05) return t;
        t += max(0.5 + t * 0.012, h * 0.85);
        if (t > t_max) break;
    }
    return -1.0;
}
float raymarchReflectMesh(vec3 ro, vec3 rd) {
    float maxH = scene.terrain_local_info.y;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    float t = 0.0;
    if (ro.y > maxH) t = (ro.y - maxH) / max(-rd.y, 1e-4);
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        if (p.y > maxH && rd.y >= 0.0) return -1.0;
        float h = p.y - meshHeight(p.xz);
        if (h < 0.04 * t + 0.1) return t;
        t += max(0.8 + t * 0.02, h * 0.9);
        if (t > 700.0) break;
    }
    return -1.0;
}
float calcShadowMesh(vec3 pos, vec3 sunDir) {
    float t = 1.5;
    for (int i = 0; i < 18; i++) {
        vec3 p = pos + sunDir * t;
        if (p.y > scene.terrain_local_info.y) break;
        if (p.y - meshHeight(p.xz) < 0.0) return 0.0;
        t += 3.0 + t * 0.15;
        if (t > 140.0) break;
    }
    return 1.0;
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
// ---- Alternative water style: P_Malin "Where the River Goes" port -----
// Advected FBM-with-derivative normal, blended over two time slices for
// stable directional flow. Flow direction is sampled from the terrain
// gradient at the water plane (water flows downhill). Used when
// rt_.water_style >= 1.
//
// Original Shadertoy: https://www.shadertoy.com/view/4dlGW2 by P_Malin
float pm_hash(float p) {
    vec2 p2 = fract(vec2(p) * vec2(4.438975, 3.972973));
    p2 += dot(p2.yx, p2.xy + 19.19);
    return fract(p2.x * p2.y);
}
vec2 pm_hash2(float p) {
    vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xx + p3.yz) * p3.zy);
}
// Value noise + analytic XY derivative in one call (P_Malin's
// SmoothNoise_DXY). Returns (dx, dy, val).
vec3 pm_noise_dxy(vec2 o) {
    vec2 p = floor(o);
    vec2 f = fract(o);
    float n = p.x + p.y * 57.0;
    float a = pm_hash(n);
    float b = pm_hash(n + 1.0);
    float c = pm_hash(n + 57.0);
    float d = pm_hash(n + 58.0);
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 t  = 3.0 * f2 - 2.0 * f3;
    vec2 dt = 6.0 * f  - 6.0 * f2;
    float u = t.x, v = t.y;
    float res = a + (b - a) * u + (c - a) * v + (a - b + d - c) * u * v;
    float dx  = (b - a) * dt.x + (a - b + d - c) * dt.x * v;
    float dy  = (c - a) * dt.y + (a - b + d - c) * u * dt.y;
    return vec3(dx, dy, res);
}
// 4-octave FBM with per-octave flow advection (P_Malin's FBM_DXY).
vec3 pm_fbm_dxy(vec2 p, vec2 flow, float ps, float df) {
    vec3 f = vec3(0.0);
    float tot = 0.0;
    float a = 1.0;
    for (int i = 0; i < 4; ++i) {
        p += flow;
        flow *= -0.75;
        vec3 v = pm_noise_dxy(p);
        f += v * a;
        p += v.xy * df;
        p *= 2.0;
        tot += a;
        a *= ps;
    }
    return f / tot;
}
// Two-slice time blend of an advected normal (P_Malin's
// SampleFlowingNormal). Returns vec4(normal_xyz, foam_amount).
vec4 pm_flowing_normal(vec2 uv, vec2 flow_rate, float foam, float time) {
    float mag = 2.5 / (1.0 + dot(flow_rate, flow_rate) * 5.0);
    float t0 = fract(time);
    float t1 = fract(time + 0.5);
    float i0 = floor(time);
    float i1 = floor(time + 0.5);
    float o0 = t0 - 0.5;
    float o1 = t1 - 0.5;
    vec2 uv0 = uv + pm_hash2(i0);
    vec2 uv1 = uv + pm_hash2(i1);
    float ga = 0.25 + (foam * -1.5);   // gradient ascent factor
    vec3 d0 = pm_fbm_dxy(uv0 * 20.0, flow_rate * 20.0 * o0, 0.75 + foam * 0.25, ga);
    vec3 d1 = pm_fbm_dxy(uv1 * 20.0, flow_rate * 20.0 * o1, 0.75 + foam * 0.25, ga);
    float fade = max(0.25, 1.0 - foam * 5.0);
    vec3 n0 = mix(vec3(0.0, 1.0, 0.0),
                   normalize(vec3(d0.x, mag, d0.y)), fade);
    vec3 n1 = mix(vec3(0.0, 1.0, 0.0),
                   normalize(vec3(d1.x, mag, d1.y)), fade);
    float w = abs(t0 - 0.5) * 2.0;
    vec3 n = normalize(mix(n0, n1, w));
    float foam_tex = mix(d0.z, d1.z, w) * fade;
    return vec4(n, foam_tex);
}
// Underwater depth at world XZ (P_Malin's GetFlowDistance, adapted).
// Returns 0 outside the water plane; positive metres below the surface.
float pm_flow_depth(vec2 wp, float water_y, float ray_t) {
    // Mesh-terrain mode: river flow follows the real mesh bed (texture
    // lookup) instead of the FBM, so flow/foam agree with the ground.
    if (scene.terrain_local_info.x > 0.5)
        return max(0.0, water_y - meshHeight(wp));
    return max(0.0, water_y - terrainM_lod(wp, ray_t));
}
// Full P_Malin GetFlowRate port. Returns vec3(flow.x, flow.y, foam).
// Flow is the local water velocity in XZ; foam is a 0..1 amount that
// rises in shallow / turbulent zones. The engine's water plane is
// flat so we treat +X as the nominal downstream and let the terrain
// gradient distort the flow from there.
vec3 pm_get_flow_rate(vec2 wp, float water_y, float ray_t) {
    // Sample depth at centre + 4 taps for the gradient.
    const float eps_g = 0.5;
    float d_c  = pm_flow_depth(wp, water_y, ray_t);
    float d_xp = pm_flow_depth(wp + vec2(eps_g, 0.0), water_y, ray_t);
    float d_xn = pm_flow_depth(wp - vec2(eps_g, 0.0), water_y, ray_t);
    float d_zp = pm_flow_depth(wp + vec2(0.0, eps_g), water_y, ray_t);
    float d_zn = pm_flow_depth(wp - vec2(0.0, eps_g), water_y, ray_t);
    // P_Malin's GetGradient is the derivative of (-terrainH) so it
    // points toward deeper water. Use that sign convention.
    vec2 grad = vec2(d_xp - d_xn, d_zp - d_zn) / (2.0 * eps_g);

    // Nominal downstream + gradient-driven push toward deeper water.
    // Gradient term falls off where it's already deep so eddies form
    // in shallow regions, not in the middle of the lake.
    vec2 base = vec2(1.0, 0.0);
    vec2 flow = base + grad * 40.0 / (1.0 + d_c * 1.5);
    // Slow down with depth — surface velocity is highest in shallows.
    flow *= 1.0 / (1.0 + d_c * 0.5);

    // Slow down BEHIND obstacles. dot(normalised_grad, -normalised_flow)
    // is +1 when flow runs into a rising bed → big slowdown; -1 when
    // the bed slopes away ahead → no slowdown. behind ∈ [0..1].
    float grad_len = length(grad);
    float flow_len = length(flow);
    float behind = 0.0;
    if (grad_len > 1e-3 && flow_len > 1e-3) {
        behind = 0.5 - dot(grad / grad_len,
                            -flow / flow_len) * 0.5;
    }
    float slow = clamp(d_c * 5.0, 0.0, 1.0);
    slow = mix(slow * 0.9 + 0.1, 1.0, behind * 0.9);
    slow = 0.5 + slow * 0.5;
    flow *= slow;

    // Foam amount — concentrates where flow is fast over shallow
    // water (rapids/eddies). P_Malin's exact mapping.
    // flow_len was already computed above; after `flow *= slow` the new
    // length is flow_len * slow. Skips a second length() (sqrt+dot) per
    // pixel. abs() of a non-negative length was a no-op — dropped.
    float foam = flow_len * slow * 0.5;
    foam += clamp(foam - 0.4, 0.0, 1.0);
    foam = 1.0 - pow(max(0.0, d_c), foam * 0.35);

    return vec3(flow * 0.6, foam);
}
// Foam pattern (P_Malin's SampleWaterFoam). Returns 0..1 mask.
float pm_foam_pattern(vec2 uv, vec2 flow_rate, float time) {
    float t0 = fract(time);
    float t1 = fract(time + 0.5);
    float o0 = t0 - 0.5;
    float o1 = t1 - 0.5;
    vec2 uv0 = uv + pm_hash2(floor(time));
    vec2 uv1 = uv + pm_hash2(floor(time + 0.5));
    float f0 = pm_fbm_dxy(uv0 * 30.0, flow_rate * 50.0 * 0.25 * o0, 0.8, -0.5).z;
    float f1 = pm_fbm_dxy(uv1 * 30.0, flow_rate * 50.0 * 0.25 * o1, 0.8, -0.5).z;
    float w  = abs(t0 - 0.5) * 2.0;
    return mix(f0, f1, w);
}

// ---- Alternative water style: misty-lake-style with bump fade -------
// Inspired by the distance-based bump falloff trick from Reinder
// Nijhoff's "Misty Lake" (Shadertoy, CC BY-NC-SA 4.0). The defining
// visual element is that the water surface goes glassy at distance
// instead of staying chaotic — far water becomes a pure mirror, near
// water keeps its ripples. Implementation uses the engine's noise2
// directly; animation comes from time-shifted FBM samples.
//
// Returns a world-space normal at world XZ, distance-faded.
//   uv_scale  scales sample frequency
//   strength  bump amplitude
//   t         seconds for animation
//   cam_dist  distance from camera to the surface point
//   fade_dist metres at which bumps fade to flat (mirror)
vec3 lakeWaterNormal(vec2 worldXZ, float uv_scale, float strength,
                     float t, float cam_dist, float fade_dist) {
    // Distance falloff: 1 close to camera, 0 past fade_dist.
    float bump_w = 1.0 - smoothstep(0.0, max(0.5, fade_dist), cam_dist);
    if (bump_w < 1e-3) return vec3(0.0, 1.0, 0.0);

    vec2 p = worldXZ * uv_scale;
    // 3-octave FBM with time as a phase offset across octaves so the
    // surface evolves smoothly over time without UV advection.
    // Centred-FD normal extraction — sample 4 neighbours of p.
    const float dx_step = 0.05;
    const float inv_2_dx = 1.0 / (2.0 * 0.05);   // 1/(2*dx_step), folded
    // Octave-1 + octave-2 time offsets are loop-invariants — compute once
    // (was: 4 + 4 redundant vec2(t*a, t*b) builds in the noise calls).
    vec2 t_off_1 = vec2(t * 0.07,  t * 0.04);
    vec2 t_off_2 = vec2(t * 0.13, -t * 0.09);
    vec2 p_o1 = p + t_off_1;
    vec2 p_o2 = p * 2.7 + t_off_2;
    float n_xp = noise2(p_o1 + vec2(dx_step, 0.0));
    float n_xn = noise2(p_o1 - vec2(dx_step, 0.0));
    float n_zp = noise2(p_o1 + vec2(0.0, dx_step));
    float n_zn = noise2(p_o1 - vec2(0.0, dx_step));
    // Add a second octave (higher freq) for ripple texture.
    n_xp += 0.5 * noise2(p_o2 + vec2(dx_step, 0.0));
    n_xn += 0.5 * noise2(p_o2 - vec2(dx_step, 0.0));
    n_zp += 0.5 * noise2(p_o2 + vec2(0.0, dx_step));
    n_zn += 0.5 * noise2(p_o2 - vec2(0.0, dx_step));
    float dx = (n_xp - n_xn) * inv_2_dx;
    float dz = (n_zp - n_zn) * inv_2_dx;
    return normalize(vec3(-dx * strength * bump_w,
                           1.0,
                          -dz * strength * bump_w));
}

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
    // World position of the water surface at this pixel comes directly
    // from the rasterised plane's vertex interpolation (vWPos). The old
    // inv(view_proj) reconstruction (`pc.model * ndc...`) drifted by
    // µm-mm per frame as camera moved — the user saw this as the
    // shoreline orientation shifting with every camera move. Perspective-
    // correct interp of fixed vertex positions is bit-stable per camera
    // pose and matches the rasterised depth exactly. Ray dir & ray-t
    // are derived from (vWPos - camera) so reflection helpers below see
    // the same world geometry.
    vec3 ro = scene.camera_pos.xyz;
    vec3 _to_surf = vWPos - ro;
    vec3 rd = normalize(_to_surf);

    // Water plane intersection FIRST — its `t` gives us a max distance
    // for the heightfield march: terrain past the water plane can't be
    // visible above water anyway (water is opaque-ish). Used as the
    // raymarch's t_max so far rays aren't wasted on terrain hidden
    // beneath water.
    bool water_on = scene.water_params.x > 0.5;
    float water_y = scene.water_params.y;
    // Distance to the water plane is just |vWPos - camera|. Replaces the
    // analytic ray-vs-plane intersection (which was algebraically equal
    // but numerically drifted with inv_vp, same root cause as wpos drift).
    float t_water_plane = water_on ? length(_to_surf) : -1.0;

    // Water-only mode: the mesh terrain path already rasterised the
    // ground (with depth) this frame, so we must NOT raymarch terrain
    // again — just produce the water surface. scene.terrain_local_info.x
    // is 1.0 when the engine runs this pass purely for water on top of
    // mesh terrain. t = -1 (no terrain hit) makes the resolve below
    // treat the water plane as the surface; non-water pixels discard,
    // letting the rasterised mesh show, and the water fragment's
    // gl_FragDepth (written at the water surface) is hardware depth-
    // tested against the mesh depth so hills correctly occlude water.
    // Skips the expensive terrain march entirely — only the water
    // plane + the existing full water shading (all styles, foam,
    // showthrough, reflection) run. Identical water to the FBM path.
    // This shader is ONLY ever bound by the rasterised water-plane
    // pipeline, so water-only is unconditional — the mesh terrain
    // already owns the ground this frame; we just shade water.
    bool water_only = true;
    // The FBM heightfield IS the mesh terrain (the CDLOD mesh is baked
    // from this exact noise), so the water's FBM-driven shore depth,
    // showthrough and reflection all agree with the rasterised ground.
    // We still run the real terrain march here so water shades ONLY
    // where it is genuinely the closest surface — identical cost and
    // behaviour to the proven-stable FBM path (no t=-1 overdraw TDR).
    // In water-only mode the FBM terrain itself is never SHADED (the
    // mesh owns the ground); terrain-hit pixels are discarded below so
    // the rasterised mesh shows through.
    // RASTERISED WATER PLANE: this shader is only ever bound by the
    // water-plane pipeline. The terrain mesh already wrote its depth
    // this frame, so occlusion (terrain in front of water) is a
    // pixel-exact HARDWARE depth test on real geometry. We must NOT
    // re-decide water-vs-land from the heightmap here: raymarchMesh()
    // disagrees with the rasterised+displaced terrain silhouette by a
    // few px, so discarding "land" punched a thin shore/far band that
    // neither water nor terrain filled → cleared sky bled through and
    // crawled as the camera moved. Always shade the water plane; the
    // depth test hides it wherever terrain is nearer. (Also drops the
    // expensive per-pixel terrain march from the water pass.)
    float t = -1.0;
    float t_water = t_water_plane;
    if (t_water <= 0.0) {
        // Ray never crosses the water plane (degenerate / looking away)
        // — nothing to shade. Plane geometry rarely produces these.
        discard;
    }

    // === Water surface shading ===========================================
    if (t_water > 0.0) {
        // Hardware depth-test (depth_test ON in setup.cpp) handles
        // "is terrain in front?" occlusion. wpos comes straight from
        // vWPos — perspective-correct world position from the plane
        // mesh vertices, no per-frame inv_vp drift, no need to snap.
        vec3 wpos = vWPos;
        float wave_str = scene.water_params.z;
        float wave_t   = scene.water_params.w;
        // wave-frequency multiplier вЂ” packed in water_color_shallow.w
        // by update_scene_ubo. Defaults to 1.0; the slider lets the
        // user dial finer ripples or bigger sweeping waves.
        float wave_scale = max(0.05, scene.water_color_shallow.w);
        vec3 wnor;
        float pm_foam_tex = 1.0;  // 1 = no extra foam; <1 darkens
        // Style 0 = engine default (directional sum-of-sines); 1 = P_Malin
        // river style (advected FBM + flow-driven foam). Packed in
        // water_foam_params.y as an int.
        int water_style = int(scene.water_foam_params.y + 0.5);
        if (water_style >= 1) {
            // Full P_Malin GetFlowRate — flow vector + foam amount in
            // one pass over the local depth field.
            vec3 flow_and_foam = pm_get_flow_rate(wpos.xz,
                                                   scene.water_params.y,
                                                   t_water);
            vec2 flow_rate = flow_and_foam.xy;
            float pm_foam_amount = flow_and_foam.z;

            // User knobs:
            //   water_foam_params.z = river flow speed (legacy slot)
            //   water_foam_params.w = river normal strength
            //   water_river_extra.x = flow direction angle (radians)
            //   water_river_extra.y = animation time speed
            //   water_river_extra.z = detail scale (UV multiplier)
            //   water_river_extra.w = foam amount multiplier
            float user_speed      = max(0.05, scene.water_foam_params.z);
            float user_normal_str = max(0.05, scene.water_foam_params.w);
            float flow_angle      = scene.water_river_extra.x;
            float time_speed      = scene.water_river_extra.y;
            float detail_scale    = scene.water_river_extra.z;
            float foam_amount     = scene.water_river_extra.w;

            // Rotate the flow vector by the user's flow_angle so the
            // user can pick any downstream direction. Default 0 keeps
            // P_Malin's +X bias.
            float ca = cos(flow_angle);
            float sa = sin(flow_angle);
            flow_rate = mat2(ca, sa, -sa, ca) * flow_rate * user_speed;
            pm_foam_amount *= foam_amount;

            // Time + UV scale are user-driven so the user can pick the
            // animation speed and the ripple density independently.
            float wave_time = wave_t * 0.4 * time_speed;
            vec2 uv_n = wpos.xz * wave_scale * detail_scale;
            // Pre-foam scaling for the FBM advection's gradient ascent.
            float foam_bias = clamp((pm_foam_amount - 0.2) * 1.5, 0.0, 1.0);
            foam_bias = foam_bias * foam_bias * 0.5;
            vec4 nf = pm_flowing_normal(uv_n, flow_rate, foam_bias, wave_time);
            float n_str = wave_str * user_normal_str;
            wnor = normalize(vec3(nf.x * n_str, 1.0, nf.z * n_str));
            // P_Malin's foam combines the per-pixel turbulence pattern
            // with the global foam amount: 1 - pow(foam_tex, foam_amt*5).
            float foam_tex = pm_foam_pattern(uv_n, flow_rate, wave_time);
            float foam_t = max(0.0, (foam_tex - 0.2) / 0.2);
            foam_tex = pow(0.5, foam_t);
            pm_foam_tex = pow(max(0.001, foam_tex), foam_bias * 5.0);
        } else if (water_style == 2) {
            // Lake style — distance-faded bumps, glassy at distance.
            // water_river_extra reused for tuning when style == 2:
            //   .x = (unused, kept compatible with river)
            //   .y = time speed
            //   .z = uv scale multiplier
            //   .w = (unused)
            // water_foam_params.w = bump strength (legacy slot, shared)
            // water_river_extinct.w * 100 = bump-fade distance (metres)
            float time_speed   = max(0.0, scene.water_river_extra.y);
            float uv_scale     = max(0.05, scene.water_river_extra.z);
            float bump_strength = max(0.0, scene.water_foam_params.w);
            float fade_dist    = max(5.0, scene.water_river_extinct.w * 100.0);
            wnor = lakeWaterNormal(wpos.xz, uv_scale * wave_scale,
                                    bump_strength * wave_str,
                                    wave_t * time_speed,
                                    t_water, fade_dist);
        } else {
            wnor = waterNormal(wpos.xz, wave_t, wave_str, wave_scale);
        }
        vec3 sunDirW = scene.sun_direction.xyz;

        // Schlick fresnel вЂ” F0 = 0.02 for water. Looking grazing-on
        // means full reflection; looking straight down в‰€ 2 % reflection.
        // pow(x, 5) → 3 multiplies (x²·x²·x). Capped at 0.88 so even
        // grazing-angle water keeps a sliver of water colour and
        // doesn't read as a pure-sky-blue line where the surface meets
        // the shore at distance.
        float cosV = clamp(-dot(rd, wnor), 0.0, 1.0);
        const float F0 = 0.02;
        float oneMinusV = 1.0 - cosV;
        float fv2 = oneMinusV * oneMinusV;
        float fv5 = fv2 * fv2 * oneMinusV;
        float fres = min(0.88, F0 + (1.0 - F0) * fv5);

        vec3 refl = reflect(rd, wnor);
        // Sky reflection вЂ” horizonв†’zenith gradient + sun halo. Used
        // directly when the reflection ray escapes the terrain or
        // when RT reflections are off.
        float skyT = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
        // Richer 3-stop sky: deep zenith → mid blue → bright hazy
        // horizon, then tinted toward the scene's actual sky colour so
        // the mirror matches the sky overhead. horizonGlow concentrates
        // brightness right at the waterline where real water reflections
        // are strongest — this is what brings the "nice" look back
        // without any FBM terrain in the reflection.
        float hb = 1.0 - abs(refl.y);
        hb *= hb; hb *= hb;                 // ≈ pow(1-|y|, 4) — horizon band
        vec3 cZenith = vec3(0.20, 0.34, 0.62);
        vec3 cMid    = vec3(0.42, 0.56, 0.78);
        vec3 cHaze   = vec3(0.80, 0.84, 0.88);
        vec3 sky_refl = mix(cMid, cZenith, skyT);
        sky_refl = mix(sky_refl, cHaze, hb * 0.65);
        sky_refl = mix(sky_refl, scene.sky_color.rgb, 0.22);
        // pow(x, 60) ≈ pow(x, 64) = 6 squarings; visually identical
        // for a sun-halo lobe at this exponent. Reuses sh2/sh4 below for
        // the broad-glow term (was redundantly recomputing sg2 = sh1*sh1
        // → sg4 = sg2*sg2 right after this block — same values).
        float sh1 = max(dot(refl, sunDirW), 0.0);
        float sh2 = sh1 * sh1; float sh4 = sh2 * sh2;
        float sh8 = sh4 * sh4; float sh16 = sh8 * sh8;
        float sh32 = sh16 * sh16; float sunHalo = sh32 * sh32;
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sunHalo * 0.6;
        // Broad, soft sun glow (≈pow(.,4)) on top of the tight halo —
        // gives the wide specular sheet that made the old reflection
        // read as "lit water" rather than a flat gradient.
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sh4 * 0.30;

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
            // Mesh-terrain mode reflects the REAL mesh surface (height-
            // map march + heightmap normal); FBM mode keeps the
            // procedural march. All the shading below (material, grass
            // tint, extinction, fog) is FBM-free and shared.
            float t_r = water_only ? raymarchReflectMesh(r_ro, refl)
                                   : raymarchReflect(r_ro, refl);
            if (t_r > 0.0) {
                vec3 r_pos = r_ro + refl * t_r;
                vec3 r_nor = water_only ? meshNormal(r_pos.xz)
                                        : calcNormal(r_pos, t_r);
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
                // Grass tint boost in reflections — getMaterial already
                // includes a grass-eligibility tint, but reflections
                // see slopes from a low angle where the underlying rock
                // shading dominates. Re-blend the configured grass
                // ground colour at full strength when the hit point is
                // grass-eligible so the user sees their painted grass
                // mirrored in the water.
                float r_grass = grassEligibility(r_pos.xz, r_pos.y);
                if (r_grass > 0.05) {
                    // Distance-blend close↔far ground tint (mirrors
                    // getMaterial's primary-path ramp at line ~1077)
                    // so reflected far-distance terrain matches the
                    // surrounding land's tint instead of staying in
                    // the bright close-tint band.
                    //
                    // Distance must be the LIGHT-PATH length the
                    // viewer's eye actually traverses — camera → water
                    // surface → reflected hit. Straight-line camera→
                    // r_pos can be much shorter (think looking down
                    // at water from a hill: the bounce angle stretches
                    // the perceived distance ~2×). That mismatch is
                    // why the reflected mountains were reading in the
                    // close tint while the same mountains seen
                    // directly were already in the far band.
                    float r_path_d = distance(scene.camera_pos.xyz, wpos) +
                                     distance(wpos, r_pos);
                    float tint_far_r = max(50.0, scene.grass_color_ground_far.w);
                    float r_tint_t = smoothstep(0.0, tint_far_r, r_path_d);
                    vec3 r_ground_base = mix(scene.grass_color_ground.rgb,
                                              scene.grass_color_ground_far.rgb,
                                              r_tint_t);
                    // Same noise-darkening the primary path applies in
                    // getMaterial (line ~1081):
                    //   grass_top = mix(ground_base, ground_base*0.7, nz)
                    // — averages ~15 % darker than raw ground_base.
                    // Without it the reflection reads visibly brighter
                    // than the surrounding land at matching distance.
                    float r_nz = noise2(r_pos.xz * 0.04) *
                                  noise2(r_pos.xz * 0.005);
                    vec3 r_grass_top = mix(r_ground_base,
                                            r_ground_base * 0.7,
                                            r_nz);
                    vec3 g_col = r_grass_top * r_lin;
                    r_col = mix(r_col, g_col, r_grass * 0.7);
                }
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
        // In mesh-terrain (water-only) mode the FBM reflection march is
        // gone, so the TLAS reflection is the only geometry reflected.
        // Drive it from the RT-reflections toggle the user already has
        // on (pc.grass_params.w) so enabling reflections "just works"
        // there, in addition to the dedicated water-TLAS toggle.
        bool tlas_refl_on = scene.water_shore.z > 0.5 ||
                            (water_only && pc.grass_params.w > 0.5);
        if (tlas_refl_on && refl.y > 0.02) {
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
                // pow(x, 8) → 3 squarings — same trick as the rest
                // of the file; this water-TLAS-reflection branch was
                // the last stray pow.
                float rfL1 = max(dot(refl, sunDirW), 0.0);
                float rfL2 = rfL1 * rfL1;
                float rfL4 = rfL2 * rfL2;
                float rfL8 = rfL4 * rfL4;
                vec3 r_fog = mix(vec3(0.55, 0.55, 0.58),
                                  vec3(1.0, 0.7, 0.3),
                                  0.3 * rfL8);
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
        // wpos.xz was already 1 cm-snapped at the source (line ~1692)
        // so everything in this block — terrain_y, foam, shore noise,
        // wave normals — sees TAA-jitter-stable coordinates.
        float terrain_y = water_only ? meshHeight(wpos.xz)
                                      : terrainM_lod(wpos.xz, t_water);
        // Noise-free depth — the foam band uses this so the highlight
        // follows the actual shoreline geometry instead of wobbling
        // with the shore-noise. depth_m below adds the noise on top
        // for the wider shallow→deep tint transition only.
        float depth_clean = max(0.0, water_y - terrain_y);
        float depth_m   = depth_clean;
        // Lower-frequency shore noise: the old metre-scale octaves made
        // the water/land boundary visibly crawl when the camera moved
        // along the shore. A slow meander reads as a natural coastline
        // without the swim.
        float dn = noise2(wpos.xz * 0.10) +
                    0.35 * noise2(wpos.xz * 0.26);
        depth_m += (dn - 0.6) * shore_blend * shore_noise;
        depth_m  = max(0.0, depth_m);
        // Screen-space anti-alias the shore transition. A fixed
        // metre-width smoothstep collapses to <1 px at grazing angles
        // / distance, so the shoreline aliases and jumps frame-to-
        // frame as the camera moves. Never let the edge be narrower
        // than ~2.5 px of depth gradient.
        float sb_aa = max(shore_blend, fwidth(depth_m) * 2.5);
        float water_blend = smoothstep(0.0, sb_aa, depth_m);
        // P_Malin-style underwater extinction when river style is on.
        // exp2(-depth * ext_col) gives the characteristic green-river
        // look (red attenuates fastest, then green, blue lasts). Mixed
        // against the user's deep tint at 60% so the slider still has
        // colour authority.
        if (water_style >= 1) {
            // User-tunable extinction colour + density (defaults to
            // P_Malin's (0.5, 0.4, 0.1) × 3.0).
            vec3  ext_col = scene.water_river_extinct.rgb;
            float ext_den = scene.water_river_extinct.w * 3.0;
            vec3 ext = exp2(-depth_m * ext_col * ext_den);
            deep = mix(deep, deep * ext + vec3(0.05, 0.08, 0.06) * (1.0 - ext.r), 0.6);
        }
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
        // Combined with calcShadow against the FBM so distant terrain
        // (mountains beyond the shadow-map cascade) also casts on
        // water — without it, water within sight of a tall ridge stayed
        // fully lit while the land beside it was correctly shadowed.
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
            // FBM self-shadow — terrain casts on the water surface.
            // Heavy: calcShadow does up to 32 FBM steps per call. Three
            // gates needed to keep this cheap from a high vantage where
            // water can fill 50 %+ of the screen (a 1080p shot then
            // fires 1M+ extra ray-style marches per frame and the
            // single submit can blow past Windows' 2 s TDR watchdog —
            // a confirmed device-lost repro from Y≈47 over the bay).
            //   1. Skip back-facing / grazing water (n·L > 0.05).
            //   2. Skip if the cascade shadow map already darkened the
            //      pixel below 0.95 — adding the FBM term won't visibly
            //      change a pixel that's already in shadow.
            //   3. Skip past 80 m view distance — the shadow-map
            //      cascade already covers everything within its frustum
            //      and beyond that the FBM detail is sub-pixel.
            float w_dist = length(wpos - scene.camera_pos.xyz);
            if (water_lit > 0.95 && w_dist < 80.0 &&
                dot(wnor, sunDirW) > 0.05) {
                float self_lit = water_only
                    ? calcShadowMesh(wpos + wnor * 0.05, sunDirW)
                    : calcShadow(wpos + wnor * 0.05, sunDirW);
                water_lit = min(water_lit, self_lit);
            }
        }

        // Underwater showthrough вЂ” for shallow water we evaluate the
        // terrain at wpos.xz and pass its surface tint through the
        // water tinted by depth. Beer's-law-style attenuation:
        // T = exp(-depth * absorption) so deep water still hides
        // the bottom while shallow water reads transparent.
        vec3 baseTint = shallow;     // current shore-blended water colour
        // Transparency / water clarity. The slider now controls how far
        // light penetrates: 0 = murky (bottom hidden within ~0.5 m),
        // 1 = very clear (seabed visible many metres down). Crucially
        // SHALLOW water is strongly see-through whenever clarity > ~0.2,
        // which is what reads as real shallow water over a sandy bottom.
        float trans_amt = scene.water_shore.w;
        if (trans_amt > 0.001) {
            // Two decoupled controls:
            //   trans_amt (Transparency slider) = master see-through
            //     strength: 0 = always opaque, 1 = full clarity.
            //   pc.color.y (Water clarity depth, m) = the depth at
            //     which water goes ~opaque, INDEPENDENT of strength.
            //     Small → only very shallow water is clear, deep stays
            //     opaque (what real shallow water looks like). Large →
            //     see far down.
            // absorp chosen so Tw = exp(-3) ≈ 0.05 (≈opaque) exactly at
            // `clarity_depth` metres; shallower water is clear.
            float clarity_depth = max(0.1, pc.color.y);
            float absorp = 3.0 / clarity_depth;
            float Tw = exp(-depth_m * absorp);
            // Visibility gate: skip the entire underwater shading
            // (calcNormal = 1× FBM, getMaterial = several `noise2`
            // taps, lambert + ambient) when the contribution is
            // below ~5 % of the surface tint. At 6 m depth Tw drops
            // to 0.05; deep-water pixels read as opaque tint anyway.
            if (Tw > 0.04) {
                // Underwater surface вЂ” terrain material at the under-
                // water point, lambert-shaded by sun (with shadow), no
                // shore noise so the look is calm.
                vec3 u_pos = vec3(wpos.x, terrain_y, wpos.z);
                vec3 u_nor = water_only ? meshNormal(u_pos.xz)
                                        : calcNormal(u_pos, t_water);
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
                // Beer's-law transmittance (depth fade governed by the
                // clarity-depth knob) × trans_amt (master strength). At
                // Transparency=1 + small clarity depth: shallow reads
                // crystal-clear, deep water stays fully opaque.
                baseTint = mix(baseTint, u_col, Tw * trans_amt);
            }
        }

        // Damp Fresnel toward zero in shallow shore water so the
        // very-grazing-angle band where water meets land doesn't read
        // as a thin sky-blue line. water_blend is 0 right at the
        // shore (depth = 0) → 1 once the water is `shore_blend`
        // metres deep; multiplying fres by water_blend means deep
        // water keeps its full reflection while shore water shows
        // its actual tint.
        float fres_eff = fres * water_blend;
        vec3 col_w = mix(baseTint, reflCol, fres_eff);
        // Shore-foam band: a thin lighter highlight at the very edge
        // of the water/land boundary that fades out within
        // `water_foam_params.x` metres of noise-free depth. Reads as
        // the natural foam / surf-line you see at real shorelines.
        // Tunable via UI: water_foam_color (rgb tint + a strength)
        // and water_foam_params.x (band width). Strength = 0 disables.
        float foam_amt = 0.0;   // hoisted so the final alpha can see it
        // Foam opacity (pc.color.w) is a MASTER override: it scales
        // both the foam colour AND its alpha, so 0 = no foam at all
        // (the shallow water shows through exactly like non-foam),
        // 1 = full opaque surf. Computed here so the colour blend can
        // use it too (previously only the alpha used it, leaving the
        // white foam tint visible at 0).
        float foam_op = clamp(pc.color.w, 0.0, 1.0);
        if (scene.water_foam_color.a > 0.001 && foam_op > 0.001) {
            float foam_w = max(0.05, scene.water_foam_params.x);
            float band   = exp(-depth_clean * (4.0 / foam_w));
            // River style: blend in the flow-driven foam pattern on top
            // of the shore band. pm_foam_tex was set in the river branch
            // above (1.0 when river style is off, so no effect).
            float flow_foam = clamp(1.0 - pm_foam_tex, 0.0, 1.0);
            float foam_mix = max(band, flow_foam);
            foam_amt = foam_mix;
            col_w = mix(col_w, scene.water_foam_color.rgb,
                        foam_mix * scene.water_foam_color.a * foam_op);
        }
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
        // Gated by the SAME foam-opacity master (foam_op) as the shore
        // band, and folded into foam_amt so the final alpha drops here
        // too — this whitecap foam was previously independent of the
        // slider, which is why low foam opacity still looked opaque.
        if (foam_op > 0.001) {
            float steepness = clamp(1.0 - wnor.y * 1.04, 0.0, 1.0);
            vec2 fp = wpos.xz * 0.55 + vec2(wave_t * 0.30, wave_t * 0.18);
            float fn = noise2(fp) * 0.6 + noise2(fp * 2.13) * 0.4;
            float foam = smoothstep(0.45, 0.85, fn * steepness * 1.6);
            foam_amt = max(foam_amt, foam);
            col_w = mix(col_w, vec3(0.95, 0.97, 1.0),
                        foam * 0.85 * water_lit * foam_op);
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

        // Depth is owned by the RASTERISED water plane (this is real
        // geometry now, not a fullscreen analytic pass) — do NOT write
        // gl_FragDepth. That makes the terrain↔water silhouette a
        // pixel-exact hardware depth test (the whole point of the
        // rasterised-plane port). clipW is still needed for the motion
        // vector below.
        vec4 clipW = pc.mvp * vec4(wpos, 1.0);
        // Distance fog on the water surface too.
        {
            float fa = distanceFogAmount(wpos, scene.camera_pos.xyz);
            col_w = mix(col_w, scene.distance_fog_color.rgb, fa);
        }
        // Soft shore depth-feather. The flat water plane and the
        // terrain mesh INTERSECT at the shoreline, where their depths
        // are ~equal — the razor-thin coplanar edge z-fights and
        // jitters as the camera moves. Fade the water alpha out over
        // the last ~0.35 m of depth so it dissolves into the wet shore
        // (alpha-blended over the terrain already drawn this pass)
        // instead of fighting it. Deeper water keeps alpha = 1, so the
        // open-water look is unchanged.
        // See-through via the HARDWARE alpha blend. The real splatted
        // terrain is already in the framebuffer (drawn before the water
        // plane), so making the water partially transparent in
        // clear/shallow water reveals the actual ground — far better
        // than the shader's internal FBM showthrough, which doesn't
        // match the mesh. trans_amt (Transparency) = master strength;
        // pc.color.y (Clarity depth, m) sets how deep the see-through
        // reaches. Deep water → clarT→0 → fully opaque.
        float clar_d = max(0.1, pc.color.y);
        float clarT  = exp(-depth_m * (3.0 / clar_d));   // 1 shallow→0 deep
        // Grazing-angle opacity: at a low view angle the light path
        // through the water is long and Fresnel turns the surface
        // reflective, so you should see the SKY, not the bottom. cosV
        // is 1 looking straight down, →0 at grazing. Without this you
        // could see through deep water when glancing across it.
        float graze = smoothstep(0.10, 0.45, cosV);
        float see   = clamp(trans_amt, 0.0, 1.0) * clarT * graze;
        // Soft, screen-space-stable waterline. The fade width is the
        // shoreline-softness knob (pc.color.z, metres of depth), but
        // never narrower than ~2.5 px of the depth gradient — that
        // fwidth floor is what kills the frame-to-frame jitter from the
        // point-sampled heightmap vs the rasterised terrain edge.
        // Shore fade band, widened from the previous (0.02 floor, 2.5×
        // fwidth) to (2.0 metre floor, 8× fwidth). The narrow band
        // was producing visible boundary jitter as the user walked:
        // wpos.xz shifts a few cm per frame, terrain_y shifts the
        // same, depth_clean swings ~5 cm — within a narrow shore_w
        // that's a big alpha jump, but inside a 2 m soft band it's
        // a tiny gradient shift TAA can fully absorb.
        // 6 m floor (was 2 m). The visible shore line is the smoothstep
        // of depth_clean across this width. Anything narrower exposes
        // the cm-scale mismatch between the rasterised CDLOD/tessellated
        // terrain mesh (per-frame sub-pixel TAA jitter on triangle
        // edges) and the smooth heightmap meshHeight() this shader
        // samples — the user reads that mismatch as "shore jumps up
        // and down a few cm" as the camera moves. A 6 m fade band
        // makes those few cm < 1% of alpha change → below visual
        // threshold. Pc.color.z (UI slider) can still override upward.
        // Floor dropped 6 m -> 0.8 m: the 64x64 subdivided water plane
        // killed the shore wobble that the wide fade was masking, so
        // the see-through band can shrink to "wet shore" widths. UI
        // slider (pc.color.z) can widen if the user wants soft shores.
        float shore_soft = max(0.8, pc.color.z);
        float shore_w    = max(shore_soft, fwidth(depth_clean) * 8.0);
        float shore_alpha = smoothstep(0.0, shore_w, depth_clean);
        // Aggressive grazing-angle clamp: at shallow viewing angles
        // (cosV < ~0.35) Fresnel makes water near-fully-reflective and
        // the "you can see the terrain UNDER the lake from the shore"
        // bug is the see-through still being nonzero. Multiply shore
        // alpha back up by (1 - graze_terms) so the water plane is
        // hard-opaque at low view angles regardless of clarity.
        float opaque_kick = 1.0 - smoothstep(0.05, 0.40, cosV);
        shore_alpha = max(shore_alpha, opaque_kick * 0.95);
        // Foam forced opaque where present so it reads as a bright
        // white band at the shoreline regardless of underlying water
        // clarity. Safe — the software ray-vs-heightmap occlusion
        // test at the top of this branch already discarded any pixel
        // where terrain is in front of the water plane, so foam can
        // never bleed onto hill-occluded pixels.
        float water_a = shore_alpha * (1.0 - 0.92 * see) *
                        mix(1.0, foam_op, foam_amt);
        float a       = mix(water_a, 1.0, foam_amt);
        outColor = vec4(col_w, a);
        // Motion vector for TAA reprojection. clip-space derivation
        // (resolution-independent) so the LR raymarch path doesn't
        // pick up a constant ~0.5 offset from scene.viewport being
        // full-res — that would kill TAA history every frame.
        vec4 prev_clipW = pc.prev_mvp * vec4(wpos, 1.0);
        if (clipW.w > 0.0 && prev_clipW.w > 0.0) {
            vec2 cur_uv  = clipW.xy      / clipW.w      * 0.5 + 0.5;
            vec2 prev_uv = prev_clipW.xy / prev_clipW.w * 0.5 + 0.5;
            outMotion = (cur_uv - prev_uv) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
        return;
    }

    // Water-only (mesh-terrain) mode: reaching here means water was not
    // the visible surface — the rasterised CDLOD mesh already owns this
    // pixel (sky pixels were discarded above). Never shade the FBM
    // terrain in this mode; let the mesh show through.
    if (water_only) discard;

    vec3 pos = ro + t * rd;
    // EARLY discard for castle-interior pixels. The original discard
    // at the end of the shader (~line 2524) fired AFTER calcNormal +
    // calcShadow + AO + GI + fog — all of that work was wasted on
    // pixels the rasterised castle floor brushes are about to own
    // anyway. Same kCastleHalfExtent constant as the late discard so
    // they can never disagree. Saves the entire shading payload for
    // the ~castle-footprint slice of the frame.
    {
        const float kCastleHalfExtentEarly = 11.05;
        if (abs(pos.x) < kCastleHalfExtentEarly &&
            abs(pos.z) < kCastleHalfExtentEarly) {
            discard;
        }
    }
    vec3 nor = calcNormal(pos, t);
    vec3 sunDir = scene.sun_direction.xyz;

    float dif = clamp(dot(nor, sunDir), 0.0, 1.0);
    // Match cube.frag's terrain contrast power on n_dot_l. Without
    // this, the 'Terrain shading contrast' slider only affected the
    // chunked rasterised mesh path; raymarched terrain stayed pure
    // linear Lambert and looked flatter / lower-contrast than castle
    // floors with the same shadow underneath. Fast paths for the
    // common slider values mirror cube.frag so the two paths match
    // pixel-for-pixel where they meet.
    {
        float contrast = max(0.25, scene.terrain_extra.x);
        if      (contrast == 1.0) { /* linear — no-op */ }
        else if (contrast == 2.0) { dif = dif * dif; }
        else if (contrast == 3.0) { dif = dif * dif * dif; }
        else if (contrast == 0.5) { dif = dif * sqrt(dif); }
        else if (contrast == 4.0) { float n2 = dif * dif; dif = n2 * n2; }
        else                      { dif = pow(dif, contrast); }
    }
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
    // Combine multiplicatively with self_shadow as transmittances.
    // shadow_amount=1 → light_through=0, so combined-blocked
    // = 1 - (1-self)*(1-grass) (union of independent occluders).
    // Was `max(...)` which under-darkens partially-lit slopes that
    // also have grass nearby — the comment was right, the code wasn't.
    self_shadow = 1.0 - (1.0 - self_shadow) *
                        (1.0 - grass_shadow * gcs_strength);

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

    // Gate on shadow_strength too — if the user dialled it to 0,
    // dyn_shadow gets multiplied by 0 at line ~2082 anyway, so the
    // 4 blocker rays + N_s shadow rays were burning BVH for nothing.
    if (dif > 0.0 && scene.rt_flags.x != 0 && do_rt &&
        scene.rt_params.w > 1e-3) {
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
        int N_ao_full = clamp(requested, 1, 32);
        // Same distance-LOD ramp as GI: distant pixels drop toward
        // 1 sample as rt_lod_t fades to 0 at the cliff. Smooth instead
        // of full-or-nothing.
        int N_ao = max(1, int(ceil(float(N_ao_full) * rt_lod_t)));
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
        // Tangent basis around `nor` is constant for all AO samples —
        // hoist outside the loop to skip cross+normalize+cross per sample.
        vec3 ao_up_a = abs(nor.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                          : vec3(1.0, 0.0, 0.0);
        vec3 ao_t    = normalize(cross(ao_up_a, nor));
        vec3 ao_b    = cross(nor, ao_t);
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rmRand(ao_seed + uvec3(i, 7u, 53u));
            float u2 = rmRand(ao_seed + uvec3(i, 41u, 5u));
            vec3 d = cos_hemi_basis(u1, u2, nor, ao_t, ao_b);
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
    // Distance-LOD on sample count: rt_lod_t goes 1→0 across the
    // RT-LOD distance band. Scale the GI sample count smoothly so
    // mid-distance pixels pay 4 rays instead of the full 16, and
    // pixels just inside the LOD cliff pay 1. Avoids the binary
    // "full N or skip" cliff that produced visible noise rings.
    int N_gi_full = (N_gi_user > 0) ? min(N_gi_user, N_gi_cap) : 0;
    int N_gi = (N_gi_full > 0 && do_rt)
                ? max(1, int(ceil(float(N_gi_full) * rt_lod_t)))
                : 0;
    // Same pattern as the PCSS gate: gi_indirect is multiplied by
    // scene.rt_params2.x (the GI strength slider) at line ~2313 so
    // strength=0 makes the whole loop's output zero. Bail before the
    // ray fire instead of burning N_gi × N_bounces ray-queries to
    // produce a value that's about to be multiplied away.
    if (N_gi > 0 && scene.rt_params2.x > 1e-3) {
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

        // Tangent basis around `nor` is constant across all GI samples
        // — precompute once, pass to cos_hemi_basis() to skip the
        // per-sample cross+normalize+cross.
        vec3 gi_up_a = abs(nor.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                          : vec3(1.0, 0.0, 0.0);
        vec3 gi_t    = normalize(cross(gi_up_a, nor));
        vec3 gi_b    = cross(nor, gi_t);

        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rmRand(gi_seed + uvec3(taken, 73u, 11u));
                float r2 = rmRand(gi_seed + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                vec3 ray_dir    = cos_hemi_basis(u1, u2, nor, gi_t, gi_b);
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
                       // pow(x, 1.5) = x * sqrt(x) — one mul + one
                       // sqrt, way cheaper than the exp/log path pow
                       // takes for fractional exponents.
                       (1.0 + gg - 2.0 * kVolPhaseG * cosTh) *
                       sqrt(1.0 + gg - 2.0 * kVolPhaseG * cosTh));
        // Hard ceiling matches a typical "sun glow" magnitude.
        phase = min(phase, 4.0);
        // P16: hoist sunGlow * phase out of the per-step inner loop.
        // Both are loop-invariants — the scattering line at the hot
        // spot below was multiplying them every iteration. ~3 ALU/step
        // back × kVolSteps + early-out cap.
        vec3 sunPhaseTerm = sunGlow * phase;

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
            // 3-octave wisp pattern, pre-baked into the 256² R8 fog
            // wisp texture (binding 14). One textureLod replaces the
            // 3 noise2 calls the inline FBM used to do (~12 hashes
            // per tap). REPEAT addressing handles time scrolling and
            // large camera moves seamlessly.
            vec2 q = p.xz * 0.020 + vec2(scene.water_params.w * 0.05);
            float w = textureLod(u_fog_wisp, q * (1.0 / 16.0), 0.0).r;
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
                    float w_s = textureLod(u_fog_wisp, qs * (1.0 / 16.0), 0.0).r;
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
            vec3  Lin   = sunPhaseTerm * lightTrans + fogTint * 0.25;
            // P16: exp2(-x * 1.4427) ≈ exp(-x), and exp2 is a single
            // hardware instruction on most archs vs exp's exp/log
            // pair. Hoisted constant 1.4427 = 1/ln(2).
            float seg  = exp2(-sigma_e * dt * 1.44269504);
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

    // The castle floor brushes are 0.5 m thin slabs sitting ~5 cm above
    // plateau height. terrain_raymarch traces the FBM (which is mixed
    // to a flat plateau_height inside the plateau), and the marched hit
    // depth at a floor pixel is essentially equal to the floor brush's
    // rasterised depth — sub-pixel TAA jitter + float precision tips
    // the LESS_OR_EQUAL test one way or the other per frame, so the
    // raymarched ground colour bleeds through as "gray flicker on the
    // castle floor" when the camera moves. Discard inside the castle
    // footprint and let the cube floor brushes own those pixels. The
    // castle is centred at origin with half-extent cr = 11 m
    // (level.cpp); a small +0.05 margin matches the floor brush's
    // in_ext = cr*2 - 0.05 extent so the discard region exactly covers
    // the floor brush footprint.
    const float kCastleHalfExtent = 11.05;
    if (abs(pos.x) < kCastleHalfExtent && abs(pos.z) < kCastleHalfExtent) {
        discard;
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
    // Terrain branch is DEAD in water.frag (water_only is forced true →
    // discarded long before here). `clip` is still needed by the motion
    // block below, but we deliberately write NO gl_FragDepth anywhere in
    // this shader so it is not depth-replacing — the rasterised water
    // plane's hardware depth is used as-is (pixel-exact shoreline).
    vec4 clip = pc.mvp * vec4(pos, 1.0);

    // Distance fog — final atmospheric mix, after all the other
    // shading, GI, water, volumetric fog and grass-shadow work.
    {
        float fa = distanceFogAmount(pos, scene.camera_pos.xyz);
        col = mix(col, scene.distance_fog_color.rgb, fa);
    }
    outColor = vec4(col, 1.0);
    // Screen-space motion vector for TAA. World position is stationary
    // вЂ” only the camera moves вЂ” so prev_uv comes from prev_view_proj
    // applied to the same world hit point. Without this, TAA cannot
    // reproject and per-frame PCSS / fog jitter shows up as moving
    // square dither artefacts.
    {
        // clip-space derivation (resolution-independent) — see the
        // matching block in the water branch above.
        vec4 prev_clip = pc.prev_mvp * vec4(pos, 1.0);
        if (clip.w > 0.0 && prev_clip.w > 0.0) {
            vec2 cur_uv  = clip.xy      / clip.w      * 0.5 + 0.5;
            vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
            outMotion = (cur_uv - prev_uv) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
    }
}
