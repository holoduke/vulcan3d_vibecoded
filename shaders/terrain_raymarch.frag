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

// Heightmap binding kept for compatibility with the descriptor layout
// — not sampled by this shader after the revert to pure-FBM terrain.
layout(set = 0, binding = 8) uniform sampler2D u_terrain_height;

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
float rmRand(uvec3 s) {
    s.x ^= s.x << 13u;
    s.x ^= s.x >> 17u;
    s.x ^= s.x << 5u;
    s.x += s.y * 19349663u;
    s.x += s.z * 83492791u;
    return float(s.x & 0x00FFFFFFu) / 16777216.0;
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
    // z: TLAS-reflection flag, w: unused.
    vec4 water_shore;
} scene;

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
    return mix(h, pc.color.w, t);
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
    return mix(h, pc.color.w, t);
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
    float jitter = rmRand(uvec3(gl_FragCoord.xy, scene.rt_flags.w));
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

// Finite-difference normal with distance-scaled epsilon — coarser
// epsilon at distance integrates over noise frequencies the camera
// can't resolve, so distant terrain doesn't alias.
vec3 calcNormal(vec3 pos, float t) {
    float eps = 0.02 + 0.00005 * t * t;
    float hC = terrainH(pos.xz);
    float hR = terrainH(pos.xz + vec2(eps, 0.0));
    float hU = terrainH(pos.xz + vec2(0.0, eps));
    return normalize(vec3(hC - hR, eps, hC - hU));
}

// PCSS-style soft shadow ray. min(k*h/t) tracks the closest grazing
// ratio, giving a penumbra that widens with distance from the
// occluder. k = 16 is moderately soft; 64 would be near-hard.
float calcShadow(vec3 pos, vec3 sunDir) {
    float res = 1.0;
    float t = 1.0;
    int kSteps = int(pc.tex_params.y);
    for (int i = 0; i < kSteps; i++) {
        vec3 p = pos + t * sunDir;
        float h = p.y - terrainM(p.xz);
        if (h < 0.001) return 0.0;
        res = min(res, 16.0 * h / t);
        t += clamp(h, 2.0, 100.0);
        if (t > 800.0) break;
    }
    return clamp(res, 0.0, 1.0);
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
// summed at slightly off-axis frequencies. Looks like the doc's
// "Sea of Greek" demo but without the FBM cost. Returns a
// world-space normal perturbed off (0,1,0).
vec3 waterNormal(vec2 worldXZ, float t, float strength) {
    // Two wave directions, slow-scrolling. Frequencies pick prime-
    // ish ratios so the pattern doesn't repeat on a small grid.
    float f1 = sin(worldXZ.x * 0.13 + t * 0.7) +
               sin(worldXZ.y * 0.17 - t * 0.6);
    float f2 = sin((worldXZ.x + worldXZ.y) * 0.21 + t * 1.1) +
               sin((worldXZ.x - worldXZ.y) * 0.27 - t * 0.9);
    // Analytical partials (chain rule on the sums above).
    float dx = 0.13 * cos(worldXZ.x * 0.13 + t * 0.7) +
               0.21 * cos((worldXZ.x + worldXZ.y) * 0.21 + t * 1.1) +
               0.27 * cos((worldXZ.x - worldXZ.y) * 0.27 - t * 0.9);
    float dz = 0.17 * cos(worldXZ.y * 0.17 - t * 0.6) +
               0.21 * cos((worldXZ.x + worldXZ.y) * 0.21 + t * 1.1) -
               0.27 * cos((worldXZ.x - worldXZ.y) * 0.27 - t * 0.9);
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
        vec3 wnor = waterNormal(wpos.xz, wave_t, wave_str);
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
        gl_FragDepth = clipW.z / clipW.w;
        outColor  = vec4(col_w, 1.0);
        outMotion = vec2(0.0);
        return;
    }

    vec3 pos = ro + t * rd;
    vec3 nor = calcNormal(pos, t);
    vec3 sunDir = scene.sun_direction.xyz;

    float dif = clamp(dot(nor, sunDir), 0.0, 1.0);
    float amb = 0.5 + 0.5 * nor.y;
    float fre = pow(clamp(1.0 + dot(rd, nor), 0.0, 1.0), 2.0);

    float sha = (dif > 0.0) ? calcShadow(pos + nor * 0.5, sunDir) : 1.0;

    // Castle / dyn-prop / box shadows via RT (PCSS-style). Mask 0x02
    // skips the terrain BLAS so we don't double-count terrain self-
    // shadow (handled by calcShadow above against the FBM).
    //   1. blocker search — wide cone, find avg occluder distance
    //   2. penumbra ∝ blocker distance × softness (PCSS)
    //   3. stratified shadow rays in the size-adapted cone
    if (dif > 0.0 && scene.rt_flags.x != 0) {
        const float kBaseSoftness = max(0.02, scene.rt_params.x);
        // Tangent basis perpendicular to L for cone jitter.
        vec3 ref   = abs(sunDir.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                            : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref, sunDir));
        vec3 tan_v = cross(sunDir, tan_u);
        // Ray origin lifted along the surface normal so the FBM-derived
        // hit point doesn't self-intersect a BLAS triangle that happens
        // to overlap (e.g. the castle pad).
        vec3 origin = pos + nor * 0.25;
        // Per-pixel deterministic seed so the dither pattern is stable
        // across frames where this pixel is the same surface.
        uvec3 seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);

        // 1. Blocker search — 4 rays, wide cone.
        float sum_t = 0.0;
        int   hits  = 0;
        const int kBlocker = 4;
        for (int i = 0; i < kBlocker; ++i) {
            float r1 = rmRand(seed + uvec3(i, 91u, 13u));
            float r2 = rmRand(seed + uvec3(i, 19u, 71u));
            float r  = sqrt(r1) * kBaseSoftness * 4.0;
            float ph = 6.28318530718 * r2;
            vec3 j   = (cos(ph) * tan_u + sin(ph) * tan_v) * r;
            vec3 d   = normalize(sunDir + j);
            float t;
            if (closest_hit_no_terrain(origin, d, 200.0, t)) {
                sum_t += t;
                ++hits;
            }
        }

        if (hits > 0) {
            // 2. Penumbra estimate (PCSS): cone width grows with avg
            // blocker distance. Clamped both ways so razor-sharp
            // contacts and far-soft skirts don't blow out.
            float avg_t = sum_t / float(hits);
            float scale = clamp(avg_t * 0.1, 0.05, 6.0);
            float penum = clamp(kBaseSoftness * scale * 0.6,
                                kBaseSoftness * 0.25,
                                kBaseSoftness * 6.0);

            // 3. Stratified shadow rays. 8 rays = decent penumbra at
            // an acceptable per-pixel cost.
            const int kShadow = 8;
            int strata = int(ceil(sqrt(float(kShadow))));
            float inv  = 1.0 / float(strata);
            int taken  = 0;
            float blocked = 0.0;
            for (int sy = 0; sy < strata && taken < kShadow; ++sy) {
                for (int sx = 0; sx < strata && taken < kShadow; ++sx) {
                    float r1 = rmRand(seed + uvec3(taken, 11u, 47u));
                    float r2 = rmRand(seed + uvec3(taken, 53u, 23u));
                    float u1 = (float(sx) + r1) * inv;
                    float u2 = (float(sy) + r2) * inv;
                    float r  = sqrt(u1) * penum;
                    float ph = 6.28318530718 * u2;
                    vec3 j   = (cos(ph) * tan_u + sin(ph) * tan_v) * r;
                    vec3 d   = normalize(sunDir + j);
                    if (any_hit_no_terrain(origin, d, 200.0)) blocked += 1.0;
                    ++taken;
                }
            }
            float dyn_shadow = (blocked / float(taken)) * scene.rt_params.w;
            sha *= (1.0 - dyn_shadow);   // multiply lit-fraction
        }
    }

    vec3 mate = getMaterial(pos, nor);

    vec3 lin = vec3(0.0);
    lin += dif * sha * scene.sun_color.rgb * scene.sun_color.a;
    lin += amb * scene.sky_color.rgb * 0.35;
    lin += fre * scene.sky_color.rgb * 0.25;

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
        const int   kVolSteps         = 16;     // dense enough to hide stepping w/ jitter
        const float kVolScaleHeight   = 22.0;   // density halves every 22 m up
        const float kVolDensityBase   = 0.018;  // σe at y = 0
        const float kVolPhaseG        = 0.65;   // forward-scattering bias
        const float kVolNoiseStrength = 0.6;    // 0 = uniform, 1 = strong wisps

        vec3 fogTint   = vec3(0.78, 0.83, 0.90);
        vec3 sunGlow   = scene.sun_color.rgb * scene.sun_color.a;

        // Henyey-Greenstein phase: forward-peaked when looking near the sun.
        float cosTh = dot(rd, sunDir);
        float gg    = kVolPhaseG * kVolPhaseG;
        float phase = (1.0 - gg) /
                      (4.0 * 3.14159265 *
                       pow(1.0 + gg - 2.0 * kVolPhaseG * cosTh, 1.5));

        // Cap march length at the surface hit. Sky pixels never reach
        // here (we returned -1 above), so t is always finite.
        float t_end = t;
        float t_max = min(t_end, 600.0);   // cap so we don't waste taps in fog-noise pixels far away
        float dt    = t_max / float(kVolSteps);
        // Per-pixel jitter on the start position breaks step banding
        // — combined with Phase 5's surface jitter this stays dither-stable.
        float jStart = rmRand(uvec3(gl_FragCoord.xy, scene.rt_flags.w + 7u));
        float t_v   = dt * jStart;

        vec3  scatter = vec3(0.0);
        float trans   = 1.0;
        for (int i = 0; i < kVolSteps; ++i) {
            vec3 p = ro + rd * t_v;
            float h_norm = exp(-max(0.0, p.y) / kVolScaleHeight);
            // Slow-scrolling FBM gives the wisps. Cheap 2-octave —
            // anything richer would dominate the per-pixel cost.
            float wisp = 1.0;
            {
                vec2 q = p.xz * 0.012;
                float w = noise2(q) * 0.6 + noise2(q * 2.7) * 0.3;
                wisp = mix(1.0, w * 1.6, kVolNoiseStrength);
                wisp = max(wisp, 0.2);
            }
            float sigma_e = kVolDensityBase * h_norm * wisp * fogStrength;

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
                    float h_s   = exp(-max(0.0, sp.y) / kVolScaleHeight);
                    float w_s   = mix(1.0, noise2(sp.xz * 0.012) * 1.6,
                                       kVolNoiseStrength);
                    float sig_s = kVolDensityBase * h_s * max(w_s, 0.2) * fogStrength;
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
    vec4 clip = pc.mvp * vec4(pos, 1.0);
    gl_FragDepth = clip.z / clip.w;

    outColor  = vec4(col, 1.0);
    outMotion = vec2(0.0);  // assume stationary terrain — TAA stable
}
