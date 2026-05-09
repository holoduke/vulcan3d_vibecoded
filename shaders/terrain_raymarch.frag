#version 460
#extension GL_EXT_ray_query : require

// Procedural FBM heightfield ray-marched terrain — selectable alternate
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

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

// Same push-constant layout as cube.frag's PushConstants. We use
//   mvp           — view_proj for gl_FragDepth at the hit point
//   model         — inverse(view_proj) for world-ray reconstruction
//   color.xy      — plateau centre (world XZ)
//   color.z       — plateau half-extent (m)
//   color.w       — plateau height
//   emissive.x    — march step factor (0.4..0.8)
//   tex_params.x  — march step count
//   tex_params.y  — shadow ray steps
//   tex_params.z  — march FBM octaves
//   tex_params.w  — normal FBM octaves
//   grass_params.x — fog strength multiplier (0 = off)
//   grass_params.y — relaxation flag (>0.5 = use relaxation step)
//   grass_params.z — fog god-ray flag (>0.5 = self-shadowed fog)
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
// noise edits. World XZ → UV maps via the gameplay-plateau push
// constants (pc.color.xy = origin, pc.color.z is plateau-extent
// — we don't have the heightmap origin/side here, so use the well-
// known engine layout: terrain centred at origin, side = 2048 m).
layout(set = 0, binding = 8) uniform sampler2D u_terrain_height;
const float kHeightmapSide = 2048.0;
float sampleHeightDelta(vec2 worldXZ) {
    vec2 uv = (worldXZ / kHeightmapSide) + vec2(0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
    return texture(u_terrain_height, uv).r;
}

// Sun shadow map (single-cascade ortho D32). Rendered each frame
// from the light's POV with castle, dyn-props and terrain chunks
// as casters. Sampled here for water-surface shadow occlusion.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

// TLAS — castle, dyn-props, terrain BLAS, etc. We fire shadow rays
// against it with cull-mask 0x02 so the terrain BLAS (instance
// mask 0x01) is excluded — terrain self-shadow is handled by
// calcShadow() against the FBM, and the rasterised terrain mesh sits
// below the BLAS detail anyway. With the mask, each ray only finds
// real dynamic-occluder hits (castle, boxes).
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

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

// Closest-hit variant for the PCSS blocker search — returns the hit
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

// Tiny xorshift32 hash — deterministic per-pixel jitter for the
// stratified shadow rays.
// Interleaved Gradient Noise — same dither pattern cube.frag uses
// for PCSS / RTAO. xorshift hashes (the previous rmRand) produce
// row-banding correlations that read as moving "lines in shadows"
// even with TAA. IGN is designed for stochastic rendering: low
// perceptual structure across pixels, well-distributed across the
// sample-index axis when you offset the input position.
float ignBase(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
float rmRand(uvec3 seed) {
    // Same recipe as cube.frag::rand — shift the input position by
    // the sample index along a magic vector so each sample's jitter
    // is uncorrelated within a pixel but smooth across neighbours.
    float s = float(seed.x) + 5.588238 * float(seed.z);
    float t = float(seed.y) + 1.388765 * float(seed.z);
    return ignBase(vec2(s, t));
}

// Scene UBO — same layout as cube.frag's binding 0. Only a handful of
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
} scene;

// Cheap directional sky model — horizon -> zenith ramp around the
// scene's sky colour, with a small sun-tinted lift near the horizon.
// Used as the "no-noise" reference for the GI softener and as the
// sample value for hemisphere rays that miss any occluder.
vec3 sample_sky_atmosphere(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, pow(up, 0.45));
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

// Per-octave ~37° rotation breaks axis-aligned banding. Unit
// determinant = pure rotation, no scaling.
const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);

const float TERRAIN_SCALE  = 0.003;
const float TERRAIN_HEIGHT = 120.0;

// Plateau blend factor — 1 inside the gameplay plateau region, 0
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

// Medium-detail FBM — used by the ray march. Derivative-erosion term
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

// Normal-detail FBM — finite-difference normals call this 3× per hit
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

float terrainMaxHeight() { return max(TERRAIN_HEIGHT, pc.color.w); }

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
    // Phase 5 — blue-noise jittered start offset. Per-pixel hash-based
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
    for (int i = 0; i < kSteps; i++) {
        vec3 pos = ro + t * rd;
        float h = pos.y - terrainM(pos.xz);
        if (abs(h) < 0.0015 * t) break;
        if (t > 1500.0) return -1.0;
        // Phase 6 — relaxation cone-stepping. Step grows with
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
        t += advance;
    }
    return t;
}

// Finite-difference normal with distance-scaled epsilon. Eps grows
// linearly with distance (was quadratic — that smeared distant
// detail far more than required for anti-aliasing) so distant
// terrain keeps edge crispness while close-up still uses sub-cm
// epsilon. The earlier "micro-noise on lit normal" path was removed
// — its `noise2(p) - noise2(p + 0.5)` cell-boundary derivative
// produced visible 8–16 pixel square blocks on the surface past
// 80 m. The FBM normal-octave slider (default 18) gives plenty of
// distance detail without the artefacts.
vec3 calcNormal(vec3 pos, float t) {
    float eps = 0.02 + 0.0008 * t;
    float hC = terrainH(pos.xz);
    float hR = terrainH(pos.xz + vec2(eps, 0.0));
    float hU = terrainH(pos.xz + vec2(0.0, eps));
    return normalize(vec3(hC - hR, eps, hC - hU));
}

// Classic heightfield soft-shadow march — `min(k·h/t)`. k ties to
// the global Shadow softness slider (rt_params.x) so a single
// control softens both terrain self-shadow AND PCSS castle/box
// shadows. Slider 0 → k=64 (razor sharp), slider 0.15 → k=4
// (very soft). Cubic smoothstep at the end softens the linear
// penumbra ramp.
float calcShadow(vec3 pos, vec3 sunDir) {
    // Driven by global Shadow softness slider × per-terrain
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
        float h = p.y - terrainM(p.xz);
        if (h < 0.001) { res = 0.0; break; }
        res = min(res, k * h / t);
        t += clamp(h, 0.5, 100.0);
        if (t > 800.0 || res < 0.001) break;
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
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

// Cheap animated ocean normal — two scrolling sin/cos directions
// summed at slightly off-axis frequencies. `scale` multiplies the
// base frequencies (1.0 = ~30 m wavelengths; 2.0 = half-wavelength
// finer ripples; 0.5 = bigger sweeping waves).
vec3 waterNormal(vec2 worldXZ, float t, float strength, float scale) {
    float k1 = 0.13 * scale;
    float k2 = 0.17 * scale;
    float k3 = 0.21 * scale;
    float k4 = 0.27 * scale;
    // Analytical partials (chain rule on the wave sums).
    float dx = k1 * cos(worldXZ.x * k1 + t * 0.7) +
               k3 * cos((worldXZ.x + worldXZ.y) * k3 + t * 1.1) +
               k4 * cos((worldXZ.x - worldXZ.y) * k4 - t * 0.9);
    float dz = k2 * cos(worldXZ.y * k2 - t * 0.6) +
               k3 * cos((worldXZ.x + worldXZ.y) * k3 + t * 1.1) -
               k4 * cos((worldXZ.x - worldXZ.y) * k4 - t * 0.9);
    return normalize(vec3(-dx * strength, 1.0, -dz * strength));
}

void main() {
    // Reconstruct the world-space ray from this fragment's NDC.xy.
    // ndcNear (z=0) and ndcFar (z=1) → world points; their delta is
    // the unnormalised direction.
    vec4 wNear = pc.model * vec4(vNDC, 0.0, 1.0);
    vec4 wFar  = pc.model * vec4(vNDC, 1.0, 1.0);
    vec3 ro = scene.camera_pos.xyz;
    vec3 rd = normalize(wFar.xyz / wFar.w - wNear.xyz / wNear.w);

    float t = raymarch(ro, rd);

    // Water plane intersection. The ray hits the water surface if
    //   - water is enabled AND
    //   - the ray is travelling downward and starts above water
    //     (or upward and starts below — we render that case as
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
        // Sky — let the existing compose-pass sky handle this pixel.
        discard;
    }

    // === Water surface shading ===========================================
    if (t_water > 0.0) {
        vec3 wpos = ro + rd * t_water;
        float wave_str = scene.water_params.z;
        float wave_t   = scene.water_params.w;
        // wave-frequency multiplier — packed in water_color_shallow.w
        // by update_scene_ubo. Defaults to 1.0; the slider lets the
        // user dial finer ripples or bigger sweeping waves.
        float wave_scale = max(0.05, scene.water_color_shallow.w);
        vec3 wnor = waterNormal(wpos.xz, wave_t, wave_str, wave_scale);
        vec3 sunDirW = scene.sun_direction.xyz;

        // Schlick fresnel — F0 = 0.02 for water. Looking grazing-on
        // means full reflection; looking straight down ≈ 2 % reflection.
        float cosV = clamp(-dot(rd, wnor), 0.0, 1.0);
        const float F0 = 0.02;
        float fres = F0 + (1.0 - F0) * pow(1.0 - cosV, 5.0);

        vec3 refl = reflect(rd, wnor);
        // Sky reflection — horizon→zenith gradient + sun halo. Used
        // directly when the reflection ray escapes the terrain or
        // when RT reflections are off.
        float skyT = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 sky_refl = mix(vec3(0.55, 0.65, 0.78),
                             vec3(0.30, 0.45, 0.75), skyT);
        float sunHalo = pow(max(dot(refl, sunDirW), 0.0), 60.0);
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sunHalo * 0.6;

        vec3 reflCol = sky_refl;
        // RT-style FBM-march reflection — picks up the actual mountain
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
                // reflection — too expensive for a free-running
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
                vec3  r_fog = mix(vec3(0.55, 0.55, 0.58),
                                  vec3(1.0, 0.7, 0.3),
                                  0.3 * pow(max(dot(refl, sunDirW), 0.0), 8.0));
                r_col = r_col * r_ext + r_fog * (1.0 - r_ext);
                reflCol = r_col;
            }
            // miss → keep sky_refl
        }

        // TLAS reflection — ray-query against castle / cubes / dyn
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
        // at this surface point — water_level minus the FBM terrain
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
        float spec = pow(max(dot(wnor, H), 0.0), 64.0);

        // Sun shadow on the water surface — bound-checked sample of
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

        // Underwater showthrough — for shallow water we evaluate the
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
            // Underwater surface — terrain material at the under-
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
            // Blend: shallow → underwater visible, deep → opaque
            // water_color. trans_amt scales the maximum showthrough.
            baseTint = mix(baseTint, u_col, Tw * trans_amt);
        }

        vec3 col_w = mix(baseTint, reflCol, fres);
        col_w += scene.sun_color.rgb * scene.sun_color.a * spec *
                  0.8 * water_lit;
        // Subtle darkening of the surface tint when in shadow.
        col_w *= mix(0.7, 1.0, water_lit);

        // Apply the same volumetric fog as terrain (reuse the
        // existing block by jumping to a tail). To keep the diff
        // tight we re-do the fog inline here using the same code path.
        float sundotW = clamp(dot(rd, sunDirW), 0.0, 1.0);
        vec3 ext_w = exp(-t_water * 0.00025 * vec3(1.0, 1.5, 4.0));
        vec3 fogColW = mix(vec3(0.55, 0.55, 0.58),
                           vec3(1.0, 0.7, 0.3),
                           0.3 * pow(sundotW, 8.0));
        col_w = col_w * ext_w + fogColW * (1.0 - ext_w);

        // Write hit depth; rasterised geometry that's actually in
        // front (e.g. a cube on a pier) still occludes us.
        vec4 clipW = pc.mvp * vec4(wpos, 1.0);
        // See main terrain branch — cap below the compose sky-cutoff
        // threshold so far water doesn't get repainted as sky-below-
        // horizon (black).
        gl_FragDepth = min(clipW.z / clipW.w, 0.9998);
        outColor = vec4(col_w, 1.0);
        // Motion vector for TAA reprojection (same pattern as
        // terrain branch below). Treats the surface as world-static
        // — wave-bump animation contributes sub-pixel motion that
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
    float fre = pow(clamp(1.0 + dot(rd, nor), 0.0, 1.0), 2.0);

    // Terrain self-shadow (FBM march). Bias is small (5 cm along
    // the surface normal) — large biases (the previous 50 cm) made
    // the shadow visibly detach from the contact line under
    // distant FBM normals.
    // calcShadow returns lit fraction; convert to "shadow amount"
    // (0 = lit, 1 = full shadow) so we can combine it with the RT
    // result the same way cube.frag does for the rasterised path.
    float self_shadow = (dif > 0.0)
        ? (1.0 - calcShadow(pos + nor * 0.05, sunDir))
        : 1.0;

    // RT PCSS for castle / boxes / dyn-props. Mirrors cube.frag's
    // terrain-receiver shadow path 1:1 — the cube.frag terrain
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
    // the bake/calcShadow penalty only — keeps the per-frame ray
    // budget bounded and prevents Windows TDR on a heavy view.
    float dyn_shadow = 0.0;
    float cam_dist_pos = distance(pos, scene.camera_pos.xyz);
    bool do_rt = cam_dist_pos < 200.0;

    if (dif > 0.0 && scene.rt_flags.x != 0 && do_rt) {
        // base_softness from the global Shadow softness slider,
        // scaled by the terrain-specific scale (terrain_params.w).
        float base_softness = scene.rt_params.x;
        if (scene.terrain_params.w > 0.0) {
            base_softness *= max(0.05, scene.terrain_params.w);
        }
        // Sample count from global Shadow samples slider × near
        // multiplier. Same source cube.frag uses for terrain.
        float near_mult = max(1.0, scene.terrain_extra.y);
        int N_s = clamp(int(ceil(float(scene.rt_flags.y) * near_mult)),
                        4, 64);

        vec3 ref_l = abs(sunDir.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                            : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref_l, sunDir));
        vec3 tan_v = cross(sunDir, tan_u);
        // Bias along sun direction (NOT surface normal) — the FBM
        // normal varies high-frequency per pixel; using it offsets
        // each pixel's shadow cone slightly and produces dither.
        vec3 origin = pos + sunDir * 0.1;
        // Per-pixel + per-frame seed — TAA averages noisy samples.
        uvec3 seed_base = uvec3(gl_FragCoord.xy, scene.rt_flags.w);

        // 1. Blocker search (4 rays, 4× softness cone).
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
            if (closest_hit_no_terrain(origin, d, 200.0, t_b)) {
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
                    if (any_hit_no_terrain(origin, d, 200.0)) blocked += 1.0;
                    ++taken;
                }
            }
            dyn_shadow = (blocked / float(taken)) * scene.rt_params.w;
        }
    }
    // 4. Max-combine: the FBM self-shadow is the analogue of
    //    cube.frag's heightmap bake — clean, no noise. Where the
    //    surface is in self-shadow, we take that value verbatim.
    //    Where it's lit, the RT cone fills in castle / cube
    //    contributions. Result reads identical to the rasterised
    //    terrain shadow path.
    float shadow = max(dyn_shadow, self_shadow);
    float sha = 1.0 - shadow;

    // RT ambient occlusion against castle / boxes / dyn-props
    // (mask 0x02 — terrain BLAS skipped). Mirrors cube.frag's RTAO
    // path: cosine-hemisphere samples around the surface normal,
    // sqrt-shaped raw occlusion, ao_floor remap. AO darkens the
    // ambient term — corners between walls + boxes get visibly
    // shaded, just like the rasterised path.
    float ao = 1.0;
    vec3  gi_sky = vec3(0.0);
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0 && scene.rt_flags.z > 0 && do_rt) {
        int requested = (ao_mode == 1) ? min(scene.rt_flags.z, 2)
                                       : scene.rt_flags.z;
        // No lod_samples helper here — clamp manually so distant
        // terrain doesn't waste per-pixel rays on AO that doesn't
        // visibly contribute.
        int N_ao = clamp(requested, 1, 32);
        float ao_radius = scene.rt_params.y * (ao_mode == 1 ? 0.5 : 1.0);

        // Origin lifted along the surface normal a few cm so the
        // FBM-derived hit point doesn't self-intersect a BLAS
        // triangle (e.g. a castle pad).
        vec3 origin_ao = pos + nor * 0.05;
        uvec3 ao_seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);
        int taken = 0;
        float occluded = 0.0;
        vec3 sky_gi = vec3(0.0);
        int  gi_misses = 0;
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rmRand(ao_seed + uvec3(i, 7u, 53u));
            float u2 = rmRand(ao_seed + uvec3(i, 41u, 5u));
            // cosine-hemisphere distribution around the surface normal.
            float r = sqrt(u1);
            float ph = 6.28318530718 * u2;
            vec3 d_local = vec3(r * cos(ph), sqrt(max(0.0, 1.0 - u1)),
                                  r * sin(ph));
            vec3 ref_n = abs(nor.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                              : vec3(1.0, 0.0, 0.0);
            vec3 t1 = normalize(cross(ref_n, nor));
            vec3 t2 = cross(nor, t1);
            vec3 d = d_local.x * t1 + d_local.y * nor + d_local.z * t2;
            if (any_hit_no_terrain(origin_ao, d, ao_radius)) {
                occluded += 1.0;
            } else {
                sky_gi += sample_sky_atmosphere(d);
                ++gi_misses;
            }
            ++taken;
        }
        // Sky-bounce GI: average of the sky colour along the rays
        // that *missed* any occluder. Diving by misses (not by total)
        // keeps the GI brightness independent of how much of the
        // hemisphere is occluded — AO alone handles the darkening
        // there. Then soften toward the noise-free sky-at-normal
        // reference so users can trade per-pixel variance for a
        // smoother estimate.
        vec3 sky_at_n = sample_sky_atmosphere(nor);
        vec3 gi_raw   = (gi_misses > 0)
                          ? (sky_gi / float(gi_misses))
                          : sky_at_n;
        float soft = clamp(scene.terrain_extra.z, 0.0, 1.0);
        gi_sky = mix(gi_raw, sky_at_n, soft);
        // Linear curve (no sqrt) — sqrt compresses the dark side
        // of the [0,1] range, so a 25 % hit ratio reads as
        // sqrt(0.75) ≈ 0.87 and corners barely darken. With more
        // samples the AO converges to its true low ratio and the
        // sqrt output reads brighter than the noisy 1-sample case
        // — exactly the "more samples = less shadow" surprise
        // reported. Linear curve makes AO scale monotonically with
        // hit ratio so increasing samples reduces noise without
        // washing out the darkening.
        float raw = 1.0 - (occluded / float(taken));
        float ao_floor_v = scene.rt_lod.w;
        ao = mix(ao_floor_v, 1.0, raw);
    }

    vec3 mate = getMaterial(pos, nor);

    vec3 lin = vec3(0.0);
    lin += dif * sha * scene.sun_color.rgb * scene.sun_color.a;
    // Ambient + Fresnel rim term darkened by RTAO so corners between
    // walls / boxes / castle visibly shade the terrain.
    lin += amb * scene.sky_color.rgb * 0.35 * ao;
    lin += fre * scene.sky_color.rgb * 0.25 * ao;
    // Sky-bounce GI — additive on top of the analytical ambient so
    // disabling GI never goes darker than the no-RT path. Modulated
    // by the global GI strength slider; the gi_sky term has already
    // been blended toward the noise-free sky-at-normal sample by the
    // GI-softener slider so cranking softener → smooth, low-noise GI.
    if (scene.rt_flags2.x > 0) {
        lin += gi_sky * 0.40 * scene.rt_params2.x * ao;
    }

    vec3 col = mate * lin;

    // Wavelength-dependent atmospheric extinction (blue attenuates
    // ~4× faster than red), with a sun-tinted fog colour near the sun
    // direction so the horizon glows warm where the sun touches it.
    vec3 extinction = exp(-t * 0.00025 * vec3(1.0, 1.5, 4.0));
    float sundot = clamp(dot(rd, sunDir), 0.0, 1.0);
    vec3 fogCol = mix(vec3(0.55, 0.55, 0.58),
                      vec3(1.0, 0.7, 0.3),
                      0.3 * pow(sundot, 8.0));
    col = col * extinction + fogCol * (1.0 - extinction);

    // === Volumetric ground fog (real ray-march) =================
    // Multi-step march along the camera ray with Beer-Lambert
    // transmittance accumulation + Henyey-Greenstein phase-weighted
    // sun in-scattering. The Frostbite-style energy-conserving
    // integration:
    //   Sint = (S - S * exp(-σe * dt)) / σe    [exact for constant σ
    //                                            over the segment]
    //   trans *= exp(-σe * dt)
    //   col_out += trans_prev * Sint
    // Fog density = exp(-y / scaleHeight) × (1 + slow FBM scroll) so
    // the fog has subtle volumetric variation (wisps in the valleys)
    // instead of a uniform sheet. No per-step volumetric shadow ray
    // — the HG phase already gives the directional sun glow.
    {
        // Slider-driven strength; skip the entire march when 0.
        float fogStrength = pc.grass_params.x;
        bool  fogGodRays  = pc.grass_params.z > 0.5;
        if (fogStrength > 0.001) {
        // Distance-LOD on the fog march: full 12 steps near the
        // camera, drop to 4 past 150 m. Distant terrain pixels with
        // 16-step fog were the dominant per-pixel cost and the
        // direct cause of the GPU TDR — far fog can't visibly
        // resolve sub-step density anyway.
        int kVolSteps = (distance(pos, scene.camera_pos.xyz) < 150.0) ? 12 : 4;
        const float kVolDensityBase   = 0.030;  // σe at full density inside the band
        const float kVolPhaseG        = 0.65;   // forward-scattering bias
        // Band parameters from settings (fog_band slot in scene UBO).
        float fog_y_start = scene.fog_band.x;
        float fog_y_top   = scene.fog_band.y;
        float fog_noise   = clamp(scene.fog_band.z, 0.0, 1.0);

        vec3 fogTint   = vec3(0.78, 0.83, 0.90);
        vec3 sunGlow   = scene.sun_color.rgb * scene.sun_color.a;

        // Henyey-Greenstein phase: forward-peaked when looking near
        // the sun. The denominator goes near-zero as cosTh→1, so
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
        // — combined with Phase 5's surface jitter this stays dither-stable.
        // Stable per-pixel jitter (no frame number) so the volumetric
        // step pattern is fixed in screen space — TAA's spatial
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
            // Centred around 0.5 so noise = 1.0 → "no modulation" and
            // noise = 0 → density ×0.2 (thin patch). Strength slider
            // controls how much the noise can carve gaps.
            float wisp = mix(1.0, 2.0 * w, fog_noise);
            wisp = clamp(wisp, 0.2, 2.0);
            float sigma_e = kVolDensityBase * profile * wisp * fogStrength;

            // Phase 7 — fog self-shadow / god-rays. Step a few small
            // distances toward the sun and accumulate optical depth
            // through the same density field. Surface (terrain /
            // castle) shadows the fog because the surface RT shadow
            // `sha` already gates the unshadowed mass; the per-step
            // sample additionally darkens fog that's behind a fog-
            // density gradient, giving the godray look.
            float lightTrans = sha;
            if (fogGodRays) {
                const int kFogShadowSteps = 4;
                float ds = 6.0;          // base step in metres
                float ld = ds * 0.5;
                for (int j = 0; j < kFogShadowSteps; ++j) {
                    vec3 sp = p + sunDir * ld;
                    float prof_s = smoothstep(fog_y_start, fog_y_start + 1.0, sp.y) *
                                    (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, sp.y));
                    vec2 qs = sp.xz * 0.020 + vec2(scene.water_params.w * 0.05);
                    float w_s = 0.55 * noise2(qs) + 0.30 * noise2(qs * 2.13)
                              + 0.15 * noise2(qs * 4.27);
                    float wisp_s = clamp(mix(1.0, 2.0 * w_s, fog_noise), 0.2, 2.0);
                    float sig_s = kVolDensityBase * prof_s * wisp_s * fogStrength;
                    lightTrans *= exp(-sig_s * ds);
                    ds *= 1.5;          // exponential growth — covers far w/o more steps
                    ld += ds;
                    if (lightTrans < 0.02) break;
                }
            }

            // Single-scattering: σs ≈ σe (assume cloudy isotropic
            // scattering, no absorption). Phase function gives the
            // sun-aligned directional component; ambient fogTint
            // keeps back-lit fog from going pitch-black.
            vec3  Lin   = sunGlow * phase * lightTrans + fogTint * 0.25;
            float seg  = exp(-sigma_e * dt);
            vec3  Sint = (Lin - Lin * seg) / max(sigma_e, 1e-4);
            scatter += trans * Sint * sigma_e;   // re-multiply σs because we factored σe out
            trans   *= seg;
            t_v     += dt;
            if (trans < 0.02) break;             // early-out — fully foggy
        }
        // Apply: surface attenuated by trans, scatter added in front.
        col = col * trans + scatter;
        }   // fogStrength > 0
    }

    // Write the hit-point depth so rasterised geometry that wrote
    // depth earlier in the same pass occludes the terrain correctly.
    // Cap below 0.9999 — the compose pass treats depth >= 0.99999 as
    // a sky pixel and paints `sample_sky(dir)`. For terrain hits past
    // ~800 m the projected depth crosses the threshold and compose
    // overwrites our colour with the skybox sampled below the horizon
    // (which is black for typical HDRIs). Result: distant raymarch
    // terrain renders as pure black. Capping keeps compose's sky-
    // detection from misfiring on far terrain hits.
    vec4 clip = pc.mvp * vec4(pos, 1.0);
    gl_FragDepth = min(clip.z / clip.w, 0.9998);

    outColor = vec4(col, 1.0);
    // Screen-space motion vector for TAA. World position is stationary
    // — only the camera moves — so prev_uv comes from prev_view_proj
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
