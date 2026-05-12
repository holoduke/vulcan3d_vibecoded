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
} scene;

// Heightmap (R32_SFLOAT) — shared with terrain raymarch + cube shaders.
// kTerrainExtent matches terrain.cpp's world span (2048 m square).
layout(set = 0, binding = 8) uniform sampler2D u_heightmap;

// Sun shadow map (binding 7). sampler2DShadow returns 0..1 PCF result.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

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
// sculpt delta.
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

// Distance-LOD variant — fades the per-march octave count from a fixed
// 8 at ray_t < lod_near_m down to lod_min_octaves past lod_far_m. Same
// LOD ramp endpoints + fractional-octave morph as terrain_raymarch.frag,
// so grass blades sit on the same surface the terrain raymarch draws
// even at distance. Params come from pc.emissive.{y,z,w} (mirrors the
// terrain shader's push-constant layout).
float terrainH_lod(vec2 wp, float ray_t) {
    vec2 p = wp * TERRAIN_SCALE;
    const int oct_full = 8;
    int   oct_min = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f   = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f   = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct     = int(ceil(oct_f));
    float frac    = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    for (int i = 0; i < oct_full; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
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
float sampleHeight_lod(vec2 wp_xz, float ray_t) {
    return terrainH_lod(wp_xz, ray_t);
}

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
float map(vec3 p, float ray_t, out bool out_blade_hit) {
    out_blade_hit = false;
    float terrainY = sampleHeight_lod(p.xz, ray_t);
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
    // Presence + slope thresholds. Tuneable; drawn-mask binding will
    // replace the noise-based presence later.
    const float kPresenceThresh = 0.45;     // value ≥ thresh → grass cell
    const float kSlopeMaxGrad   = 0.6;      // |∇noise2| over the cell_n
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 nId = cellId + vec2(float(dx), float(dy));
            // Per-cell eligibility — one noised() lookup at low frequency
            // gives both a presence mask (clumps grass into ~50 m cell_nes
            // with bare gaps) AND a slope proxy (the gradient magnitude
            // of the same low-freq field correlates with macro terrain
            // slope; flat zones → low gradient → grass, ridges/cliffs →
            // high gradient → no grass).
            vec2 cellW = nId * kPeriod;
            vec3 cell_n = noised(cellW * 0.02);   // ~50 m feature size
            if (cell_n.x < kPresenceThresh) continue;
            if (length(cell_n.yz) > kSlopeMaxGrad) continue;
            vec4 r = hash42(nId);
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

vec3 calcTerrainNormalApprox(vec2 p, float yC, float ray_t) {
    const float e = 0.05;
    float yR = sampleHeight_lod(p + vec2(e, 0.0), ray_t);
    float yU = sampleHeight_lod(p + vec2(0.0, e), ray_t);
    return normalize(vec3(yC - yR, e, yC - yU));
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
    const int   kMaxSteps = 192;
    float t = 0.0;
    bool hit = false;
    bool blade_hit = false;
    // Pixel size projected at unit distance — used to early-out when the
    // SDF distance is below sub-pixel detail at this t.
    float pixSize = 2.0 * scene.viewport.z;
    for (int i = 0; i < kMaxSteps; ++i) {
        if (t >= kMaxDist) break;
        vec3 p = ro + t * rd;
        bool step_blade;
        float d = map(p, t, step_blade);
        // Hit threshold loosens with distance so the loop terminates on
        // sub-pixel features instead of trying to converge on a blade
        // edge from 50 m away.
        if (d < 0.5 * pixSize * t + 0.0008) {
            hit = true;
            blade_hit = step_blade;
            break;
        }
        // 0.7 step factor for cone safety on near-tangent grazing rays.
        t += d * 0.7;
    }

    // Discard pure terrain-plane hits (no grass blade closer) so the
    // existing terrain renderer's pixel shows through unaltered. Without
    // this, no-grass cells (presence/slope rejected) painted the bare
    // ground green at z-fight depth → flickering green tint.
    if (!hit || !blade_hit) { discard; }

    vec3 p = ro + t * rd;
    vec3 L = normalize(scene.sun_direction.xyz);

    // Lighting: terrain normal at hit point + half-Lambert + sky fill.
    float terrainY = sampleHeight_lod(p.xz, t);
    vec3 N = calcTerrainNormalApprox(p.xz, terrainY, t);
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

    // Per-blade colour variation — low-freq world hash gives clumps.
    float clump = hash12(floor(p.xz * 0.5));
    vec3 youngCol = vec3(0.32, 0.42, 0.18);
    vec3 oldCol   = vec3(0.36, 0.39, 0.16);
    vec3 grassCol = mix(youngCol, oldCol, clump);

    // Height-from-ground AO: bottom of blade darker, tip lighter.
    float h_above = max(0.0, p.y - terrainY);
    float ao = 0.35 + 0.65 * pow(clamp(h_above / 1.2, 0.0, 1.0), 0.6);

    vec3 sun_term = scene.sun_color.rgb * scene.sun_color.a *
                    (n_dot_l * sun_lit);
    vec3 sky_term = scene.sky_color.rgb * 0.18;
    vec3 col = grassCol * (sun_term + sky_term) * ao;

    // Fresnel-edge highlight (iq's suggestion in the original).
    vec3 fres = vec3(0.18, 0.18, 0.10) * ao *
                pow(clamp(1.0 + dot(N, rd), 0.0, 1.0), 3.0);
    col += fres;

    // Distance fade target = the same green tint terrain_raymarch's
    // getMaterial blends in for grass-eligible cells. Blade colour eases
    // INTO the underlying ground colour instead of into the sky, so the
    // cutoff reads as "thinning out" rather than a hard sky-coloured
    // ring. Dither the last 8% with a per-pixel hash discard so the
    // very edge thins to bare terrain instead of a uniform tinted band.
    vec3 fade_target = vec3(0.18, 0.30, 0.09);
    float fade = smoothstep(kMaxDist * 0.55, kMaxDist, t);
    col = mix(col, fade_target, fade);
    if (fade > 0.92) {
        float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)))
                              * 43758.5453);
        if (dither < (fade - 0.92) * 12.5) { discard; }
    }

    outColor = vec4(col, 1.0);
    // Treat grass as world-static for TAA (per-blade wind sway is sub-
    // pixel and TAA's 5x5 spatial filter absorbs it).
    outMotion = vec2(0.0);

    // Honour LESS_OR_EQUAL depth test so a closer wall already in
    // depth_image_ correctly occludes the marched grass.
    vec4 clip = pc.mvp * vec4(p, 1.0);
    gl_FragDepth = clamp(clip.z / max(clip.w, 1e-4), 0.0001, 0.9998);
}
