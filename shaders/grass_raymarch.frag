#version 460

// Raymarched grass — fullscreen pass that walks an SDF blade field.
// Adapted from MonterMan / iq's "grass field with blades" shadertoy
// (https://shadertoy.com/view/dd2cWh) for a Vulkan rasterized engine:
//
//   * Heightmap source is our scene_desc binding 8 (R32_SFLOAT) instead
//     of an inline FBM — blades sit on the actual terrain the rest of
//     the engine sees.
//   * Distance cap kills grass beyond `pc.color.x` so the per-pixel cost
//     stays bounded; the view fades to sky over the last 30% so the cap
//     doesn't read as a hard ring.
//   * gl_FragDepth from the marched hit lets cube/castle/dyn-prop depth
//     occlude grass via the LESS_OR_EQUAL test (depth pre-pass already
//     wrote those before this draw).
//   * Wind animation modulates per-cell blade rotation by `pc.color.z`
//     (time, seconds) × `pc.color.y` (wind strength).

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
    vec4  grass_extra;
    vec4  grass_extra2;
    mat4  light_vp;
    vec4  terrain_extra;
    vec4  water_params;     // x: enabled, y: water level, z..w: unused here
    vec4  water_color;
    vec4  water_color_shallow;
    vec4  water_shore;
    vec4  fog_band;
    vec4  terrain_rt_extra;
    // Grass colour palette + extras (UI-driven).
    //   grass_color_top.rgb    = blade tip colour
    //   grass_color_top.w      = raymarched grass density (0..1, per-cell skip)
    //   grass_color_bottom.rgb = blade base colour
    //   grass_color_bottom.w   = blade-base AO floor (0..1)
    //   grass_color_ground.rgb     = CLOSE ground tint (terrain shader +
    //                                 grass distance fade target)
    //   grass_color_ground.w       = ground tint strength (terrain shader only)
    //   grass_color_ground_far.rgb = FAR ground tint
    vec4  grass_color_top;
    vec4  grass_color_bottom;
    vec4  grass_color_ground;
    vec4  grass_color_ground_far;
} scene;

// Heightmap (R32_SFLOAT) — shared with terrain raymarch + cube shaders.
// kTerrainExtent matches terrain.cpp's world span (2048 m square).
layout(set = 0, binding = 8) uniform sampler2D u_heightmap;

// Sun shadow map (binding 7). sampler2DShadow returns 0..1 PCF result.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

// Grass eligibility mask. R8 1024² — single source of truth shared
// with terrain_raymarch.frag. CPU-baked from the same FBM (presence)
// and a sample_terrain_height finite-difference (slope) the engine
// already runs at level load. Sampling this once per cell replaces a
// 9-call noised() storm in the inner loop of map() — that was the
// dominant cost in this pass per the latest perf review.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;

layout(push_constant) uniform PC {
    mat4 mvp;          // view_proj — for gl_FragDepth at the hit
    mat4 model;        // inverse(view_proj) — for world-ray reconstruction
    mat4 prev_mvp;     // unused here, kept for layout match
    vec4 color;        // x: max distance (m), y: wind strength,
                       // z: time (s), w: density bias
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

// Heightmap texture mapping — matches terrain_raymarch.frag's
// kHeightmapSide so my grass sits on the same surface the terrain
// raymarch draws (otherwise grass appears UNDER the terrain and gets
// occluded by the LESS_OR_EQUAL depth test → no grass visible).
const float kHeightmapSide = 2048.0;

float sampleHeightDelta(vec2 worldXZ) {
    vec2 uv = (worldXZ / kHeightmapSide) + vec2(0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
    return textureLod(u_heightmap, uv, 0.0).r;
}

// === Procedural FBM terrain (ported from terrain_raymarch.frag) ===
// Match its terrainM() output so grass stands on the same y the
// terrain renderer draws. Heightmap texture stores DELTA from baseline
// (sculpt edits); the absolute height is the procedural FBM + delta.

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

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

const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);
const float TERRAIN_SCALE  = 0.003;
const float TERRAIN_HEIGHT = 120.0;
// Plateau matches the gameplay default (engine init: castle area).
const vec2  kPlateauCenter = vec2(0.0, 0.0);
const float kPlateauExtent = 28.0;
const float kPlateauHeight = 22.0;
const float kPlateauBlend  = 24.0;

float plateauWeight(vec2 worldXZ) {
    vec2 d = abs(worldXZ - kPlateauCenter) - vec2(kPlateauExtent);
    float dout = max(max(d.x, 0.0), max(d.y, 0.0));
    return 1.0 - smoothstep(0.0, kPlateauBlend, dout);
}

// Same FBM as terrain_raymarch.frag::terrainM with the per-frame
// octave count fixed at 8 (middle of the engine's 4..9 quality slider).
// Returns absolute terrain height = procedural FBM + plateau blend +
// sculpt delta. MUST stay at 8 octaves so grass blades sit on the
// same surface the terrain raymarch renders — reducing octaves here
// produced a smoother grass-positioning shape that drifted above /
// below the actual terrain green tint, leaving visible gaps.
float terrainH(vec2 wp) {
    vec2 p = wp * TERRAIN_SCALE;
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kOct = 8;
    for (int i = 0; i < kOct; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    h = mix(h, kPlateauHeight, plateauWeight(wp));
    h += sampleHeightDelta(wp);
    return h;
}

// Backwards-compat: rest of the shader still calls sampleHeight.
float sampleHeight(vec2 wp_xz) { return terrainH(wp_xz); }

// Hash helper still used by per-cell blade variation.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec4 hash42(vec2 p) {
    vec4 p4 = fract(vec4(p.xyxy) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

mat2 rotate2d(float t) {
    float c = cos(t), s = sin(t);
    return mat2(c, -s, s, c);
}

float opSubtraction(float d1, float d2) { return max(d1, -d2); }

vec2 opRepeat(vec2 p, vec2 period, out vec2 outId) {
    outId = floor((p + 0.5 * period) / period);
    return mod(p + 0.5 * period, period) - 0.5 * period;
}

float sdCircle(vec2 p, float r) { return length(p) - r; }

// 2D blade silhouette (per shadertoy): two circle subtractions clip the
// crescent shape, then the bottom half-plane truncates the base.
float sdGrassBlade2d(vec2 p) {
    float d = sdCircle(p - vec2(1.7, -1.3), 2.0);
    d = opSubtraction(d, sdCircle(p - vec2(1.7, -1.0), 1.8));
    d = opSubtraction(d, p.y + 1.0);
    d = opSubtraction(d, -p.x + 1.7);
    return d;
}

// Extrude the 2D shape to 3D with a thickness — sqrt(d² + z²) - thickness
// gives a smooth round bevel out of the slab.
float sdGrassBlade(vec3 p, float thickness) {
    p -= vec3(0, 1.0, 0);
    float d2 = max(0.0, sdGrassBlade2d(p.xy));
    return sqrt(d2 * d2 + p.z * p.z) - thickness;
}

// Distance to the grass field at a world-space point. terrainY is sampled
// once per call and reused for the cell-neighbor loop (the original SDF
// would call calcTerrainHeight per neighbor — ~9× more samples).
//
// out_blade_hit flags whether the closest feature is a grass blade
// (true) or just the terrain plane (false). Caller uses this to
// `discard` terrain-plane hits — those would z-fight with the actual
// rasterised / raymarched terrain renderer's output and read as a
// flickering green tint over bare ground.
float map(vec3 p, out bool out_blade_hit) {
    out_blade_hit = false;
    float terrainY = sampleHeight(p.xz);
    float distToTerrain = 0.5 * (p.y - terrainY);

    // Coarse guard: if we're far above terrain, skip the per-cell SDF
    // evaluation (blades top out at ~1 m). Saves the inner 9-cell loop
    // for the long descent through air at ray start.
    const float kGuard = 1.6;
    if (distToTerrain > kGuard) return distToTerrain - (kGuard - 1.0);

    vec2 cellId;
    const float kPeriod = 0.20;        // ~25 blades per m²
    vec3 lp = p;
    lp.y -= terrainY;                  // ground-relative
    lp.xz = opRepeat(lp.xz, vec2(kPeriod), cellId);

    // Altitude + water gates — match terrain_raymarch.frag's
    // grassEligibility so the grass shader and the green terrain tint
    // agree about where grass exists. Snow band fades grass out
    // 55→75m; below water level kills it outright. Skipping early here
    // means the inner 9-cell loop never runs above the snow line or
    // underwater (big savings on mountain peaks).
    float alt_factor = 1.0 - smoothstep(55.0, 75.0, terrainY);
    bool underwater  = scene.water_params.x > 0.5 &&
                       terrainY < scene.water_params.y;
    if (alt_factor <= 0.001 || underwater) return distToTerrain;

    float d = distToTerrain;
    // 3×3 neighbor sweep — blades placed by per-cell hash can lean into
    // adjacent cells, so we have to query the immediate neighbors for the
    // closest blade.
    const float kPi2 = 6.28318530718;
    // Per-frame slope cutoff: convert grass_slope_n_min (max normal Y)
    // into the equivalent max gradient magnitude. n_y = 1/√(1+|∇h|²)
    // so |∇h|_max = √(1/n²−1).
    float slope_n_min = scene.grass_extra.z;
    float slope_max = sqrt(max(1e-4, 1.0 / max(slope_n_min * slope_n_min, 1e-4) - 1.0));
    float density   = clamp(scene.grass_color_top.w, 0.0, 1.0);
    // Distance-density falloff: far cells get extra-thinned on top
    // of the global density slider. dist_density = 0 → no falloff,
    // dist_density = 1 → density drops to ~15% at the max raymarch
    // range. Soft-distance anchor (pc.color.w) and max distance
    // (pc.color.x) match the alpha-cutoff distance ramp so the two
    // visual transitions stay aligned.
    float t_cam       = length(p - scene.camera_pos.xyz);
    float kMaxD       = max(pc.color.x, 4.0);
    float soft_d      = max(pc.color.w, 1.0);
    float dist_w      = smoothstep(soft_d, kMaxD, t_cam);
    float dist_dens   = clamp(scene.grass_extra.w, 0.0, 1.0);
    float density_eff = density * mix(1.0, max(1.0 - dist_dens * 0.85, 0.0), dist_w);
    const float kEligThresh = 0.10;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 nId = cellId + vec2(float(dx), float(dy));
            // Density-driven per-cell skip. Each cell gets a stable
            // hash; cells whose hash exceeds the (distance-modulated)
            // density are skipped. density_eff = global density ×
            // distance-density falloff (computed once outside the
            // sweep above).
            vec4 r = hash42(nId);
            if (r.x > density_eff) continue;
            // Single-source-of-truth eligibility — sample the RG8 mask
            // shared with terrain_raymarch.frag's getMaterial.
            //   .r = presence  (0..1, fixed at bake time)
            //   .g = slope mag (raw |∇h| / 2, un-normalized in shader)
            // Combine with the slider-driven slope cutoff. Drawn-mask
            // editor will paint into .r at runtime.
            vec2 cellW = nId * kPeriod;
            vec2 mask_uv = (cellW / kHeightmapSide) + vec2(0.5);
            vec2 mg = textureLod(u_grass_mask, mask_uv, 0.0).rg;
            float slope_mag = mg.g * 2.0;
            float slope_w = 1.0 - smoothstep(slope_max * 0.7,
                                              slope_max * 1.2, slope_mag);
            float elig = mg.r * slope_w;
            if (elig < kEligThresh) continue;
            vec3 np = lp - vec3(float(dx), 0.0, float(dy)) * kPeriod;
            // Per-blade rotation + sub-cell xy offset.
            np.xz = rotate2d(r.z * kPi2) * np.xz;
            np.xz += (r.xy - 0.5) * kPeriod;
            // Wind sway — bend the blade as a function of height (np.y).
            // Per-blade phase + global time. Magnitude scales with wind
            // strength (pc.color.y) and the blade's own height factor.
            float wind_t = pc.color.z + r.w * kPi2;
            float bend   = sin(wind_t * 1.4) * pc.color.y * 0.025;
            np.x += bend * smoothstep(0.0, 1.0, np.y);
            // sqrt(r.w) gives a more uniform-looking height distribution
            // than the raw [0,1] hash.
            float scale = max(0.65, sqrt(r.w));
            float bladeDist = sdGrassBlade(np / scale, 0.002);
            if (bladeDist < d) {
                d = bladeDist;
                out_blade_hit = true;
            }
        }
    }
    return d;
}

vec3 sample_sky_grad(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, pow(up, 0.45));
}

void main() {
    // Reconstruct world ray from screen UV via inverse view-proj.
    vec2 uv  = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
    vec4 ndc = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 wp4 = pc.model * ndc;        // model = inv(view_proj)
    vec3 ro  = scene.camera_pos.xyz;
    vec3 rd  = normalize(wp4.xyz / wp4.w - ro);

    // (Removed early-out on rd.y > 0. Mountains rise above eye height,
    // so an upward-pointing ray can still hit terrain on a peak — the
    // cut-out used to draw a hard horizon line at exactly the screen's
    // mid-height which read as a cliff in the grass coverage.)

    const float kMaxDist = max(pc.color.x, 4.0);
    // Cut the absolute step cap — 192 was overkill for most pixels and
    // burned the worst-case budget on distant grazing rays that never
    // converge. With the distance-ramped step factor below the loop
    // covers the same ground with fewer iterations at distance.
    const int   kMaxSteps = 128;
    float t = 0.0;
    bool hit = false;
    bool blade_hit = false;
    float hit_d = 0.0;            // SDF distance at the converged hit
    float hit_thresh = 0.0;       // hit threshold at that t — alpha denom
    // Pixel size projected at unit distance — used to early-out when the
    // SDF distance is below sub-pixel detail at this t.
    float pixSize = 2.0 * scene.viewport.z;
    for (int i = 0; i < kMaxSteps; ++i) {
        if (t >= kMaxDist) break;
        vec3 p = ro + t * rd;
        bool step_blade;
        float d = map(p, step_blade);
        // Hit threshold loosens with distance so the loop terminates on
        // sub-pixel features instead of trying to converge on a blade
        // edge from 50 m away.
        float thresh = 0.5 * pixSize * t + 0.0008;
        if (d < thresh) {
            hit = true;
            blade_hit = step_blade;
            hit_d = max(d, 0.0);
            hit_thresh = thresh;
            break;
        }
        // Distance-ramped step factor. Close-up needs the 0.7 cone-safety
        // factor for grazing rays through dense blade fields; past ~15 m
        // blades shrink toward sub-pixel and the conservative step is
        // pure cost. By 50 m we're at 0.95 — half as many iterations
        // cover the same metres of ray.
        float step_f = mix(0.70, 0.95, smoothstep(15.0, 50.0, t));
        t += d * step_f;
    }

    // Discard pure terrain-plane hits (no grass blade closer) so the
    // existing terrain renderer's pixel shows through unaltered. Without
    // this, no-grass cells (presence/slope rejected) painted the bare
    // ground green at z-fight depth → flickering green tint.
    if (!hit || !blade_hit) { discard; }

    vec3 p = ro + t * rd;
    vec3 L = normalize(scene.sun_direction.xyz);

    // Lighting: terrain normal at hit point.
    // Reverted from cross(dFdx(p), dFdy(p)): derivatives across grass
    // silhouettes (where one pixel of a 2×2 quad discards) are spec-
    // undefined and produced NaN normals → propagated through bloom +
    // auto-exposure → fps regression.
    // Use the cheap finite-difference normal — terrainH already runs at
    // reduced 4 octaves (down from 8) so the per-pixel cost is bounded.
    float terrainY = sampleHeight(p.xz);
    const float e = 0.05;
    float yR = sampleHeight(p.xz + vec2(e, 0.0));
    float yU = sampleHeight(p.xz + vec2(0.0, e));
    vec3 N = normalize(vec3(terrainY - yR, e, terrainY - yU));
    float n_dot_l = max(0.0, dot(N, L));

    // Sun shadow — sample the rasterised shadow map. PCF gives soft edges.
    float sun_lit = 1.0;
    {
        vec4 lc  = scene.light_vp * vec4(p, 1.0);
        if (abs(lc.w) > 1e-5) {
            vec3 lndc = lc.xyz / lc.w;
            vec2 luv  = lndc.xy * 0.5 + 0.5;
            if (luv.x >= 0.0 && luv.x <= 1.0 &&
                luv.y >= 0.0 && luv.y <= 1.0 &&
                lndc.z >= 0.0 && lndc.z <= 1.0) {
                sun_lit = textureLod(u_sun_shadow_map,
                                      vec3(luv, lndc.z - 0.0005), 0.0);
            }
        }
    }

    // Per-blade colour: lerp between bottom (base) and top (tip) by
    // height above ground. Both ends are user-driven via UI sliders
    // (scene.grass_color_bottom / grass_color_top). Slight low-freq
    // hash modulation breaks regularity without needing two separate
    // "young/old" colour vars.
    float h_above = max(0.0, p.y - terrainY);
    float h_t     = clamp(h_above / 1.2, 0.0, 1.0);
    float clump   = hash12(floor(p.xz * 0.5)) * 0.15;   // ±15% per-clump tint shift
    vec3 grassCol = mix(scene.grass_color_bottom.rgb,
                        scene.grass_color_top.rgb, h_t);
    grassCol *= 1.0 - clump;

    // Height-from-ground AO. Slider-controlled floor (0 = darkest base,
    // 1 = no AO darkening); the lerp above already shifts colour with
    // height, so the AO term mostly adds extinction at the very base.
    float ao_floor = clamp(scene.grass_color_bottom.w, 0.0, 1.0);
    float ao = ao_floor + (1.0 - ao_floor) * pow(h_t, 0.6);

    vec3 sun_term = scene.sun_color.rgb * scene.sun_color.a *
                    (n_dot_l * sun_lit);
    vec3 sky_term = scene.sky_color.rgb * 0.18;
    vec3 col = grassCol * (sun_term + sky_term) * ao;

    // Fresnel-edge highlight (iq's suggestion in the original).
    vec3 fres = vec3(0.18, 0.18, 0.10) * ao *
                pow(clamp(1.0 + dot(N, rd), 0.0, 1.0), 3.0);
    col += fres;

    // Distance fade colour target — eases blade colour INTO the
    // underlying ground tint so the silhouette transition is invisible.
    // Two ground colours blend by camera distance using the same
    // anchors the terrain shader uses (30..200m), keeping the two
    // passes in lock-step.
    float ground_t   = smoothstep(30.0, 200.0, t);
    vec3 fade_target = mix(scene.grass_color_ground.rgb,
                            scene.grass_color_ground_far.rgb,
                            ground_t);
    float fade = smoothstep(kMaxDist * 0.55, kMaxDist, t);
    col = mix(col, fade_target, fade);

    // Distance-driven opacity. Close-up blades are fully opaque
    // (alpha = 1). Past pc.color.w (the soft-distance setting) the
    // alpha ramps down so far blades become genuinely translucent
    // and the underlying terrain green tint shows through. The
    // user-cutoff slider controls HOW MUCH transparency the far end
    // reaches: cutoff = 0 → far blades stay opaque (alpha = 1);
    // cutoff = 0.6 → far blades drop to alpha ≈ 0.4 (heavy blend
    // with terrain).
    //
    // The earlier "edge alpha from SDF distance" approach was wrong:
    // the raymarch loop converges right AT the blade surface, so the
    // hit_d sample lands near hit_thresh for every hit (interior or
    // silhouette), producing uniformly low coverage and the "always
    // transparent close-up" symptom.
    float user_cut    = scene.grass_extra.y;
    float soft_dist   = max(pc.color.w, 1.0);
    float dist_w      = smoothstep(soft_dist, kMaxDist, t);
    float final_alpha = mix(1.0, 1.0 - user_cut, dist_w);

    outColor = vec4(col, final_alpha);
    // Treat grass as world-static for TAA (per-blade wind sway is sub-
    // pixel and TAA's 5x5 spatial filter absorbs it). Motion attachment
    // is OPAQUE (no blending) so this writes the static-zero value
    // verbatim regardless of edge_alpha.
    outMotion = vec2(0.0);

    // Honour LESS_OR_EQUAL depth test so a closer wall already in
    // depth_image_ correctly occludes the marched grass.
    vec4 clip = pc.mvp * vec4(p, 1.0);
    gl_FragDepth = clamp(clip.z / max(clip.w, 1e-4), 0.0001, 0.9998);
}
