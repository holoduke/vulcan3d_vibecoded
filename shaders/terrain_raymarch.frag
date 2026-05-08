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
//   mvp        — view_proj for gl_FragDepth at the hit point
//   model      — inverse(view_proj) for world-ray reconstruction
//   color.xy   — gameplay plateau centre (world XZ)
//   color.z    — plateau half-extent (metres) — half-width of the
//                rectangular flat region the castle sits on
//   color.w    — plateau height (target Y the FBM blends toward in
//                the plateau region)
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
    int kSteps = int(pc.tex_params.x);
    float kStep = pc.emissive.x;
    for (int i = 0; i < kSteps; i++) {
        vec3 pos = ro + t * rd;
        float h = pos.y - terrainM(pos.xz);
        if (abs(h) < 0.0015 * t) break;
        if (t > 1500.0) return -1.0;
        t += kStep * h;
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

void main() {
    // Reconstruct the world-space ray from this fragment's NDC.xy.
    // ndcNear (z=0) and ndcFar (z=1) → world points; their delta is
    // the unnormalised direction.
    vec4 wNear = pc.model * vec4(vNDC, 0.0, 1.0);
    vec4 wFar  = pc.model * vec4(vNDC, 1.0, 1.0);
    vec3 ro = scene.camera_pos.xyz;
    vec3 rd = normalize(wFar.xyz / wFar.w - wNear.xyz / wNear.w);

    float t = raymarch(ro, rd);
    if (t < 0.0) {
        // Sky — let the existing compose-pass sky handle this pixel.
        // Discarding leaves scene_color and depth untouched here so
        // the compose stage's atmospheric sky remains visible.
        discard;
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

    // === Low-altitude volumetric fog ============================
    // Real fog hugs valley floors. Density drops off exponentially
    // with altitude. The "trapezoid rule" between camera and hit
    // altitude is a cheap single-sample approximation of the integral
    // along the ray. Tuned so:
    //   ground ↔ ground at ~300 m → noticeable haze
    //   high altitude or short range → almost no effect
    //   distant valley floors → never fully whites out (silhouette
    //                            remains readable)
    {
        const float kFogScaleHeight = 18.0;   // density halves every 18 m up
        const float kFogBaseDensity = 0.003;  // density at y = 0
        vec3 lowFogCol = mix(vec3(0.72, 0.76, 0.82),
                             vec3(1.0, 0.85, 0.65),
                             0.20 * pow(sundot, 8.0));
        float dCam = exp(-max(0.0, ro.y)  / kFogScaleHeight);
        float dHit = exp(-max(0.0, pos.y) / kFogScaleHeight);
        float fog_amt = kFogBaseDensity * 0.5 * (dCam + dHit) * t;
        float trans = exp(-fog_amt);
        col = mix(lowFogCol, col, trans);
    }

    // Write the hit-point depth so rasterised geometry that wrote
    // depth earlier in the same pass occludes the terrain correctly.
    vec4 clip = pc.mvp * vec4(pos, 1.0);
    gl_FragDepth = clip.z / clip.w;

    outColor  = vec4(col, 1.0);
    outMotion = vec2(0.0);  // assume stationary terrain — TAA stable
}
