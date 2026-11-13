#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require

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
// G-buffer pass-through outputs — see grass.frag for the design rationale.
// Material_id=5 makes deferred sample scene_color at this pixel instead
// of recomputing lighting, preserving the raymarched grass shading.
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

#include "scene_ubo.glsl"

// Heightmap (R32_SFLOAT) — shared with terrain raymarch + cube shaders.
// kTerrainExtent matches terrain.cpp's world span (2048 m square).
layout(set = 0, binding = 8) uniform sampler2D u_heightmap;

// Sun shadow map (binding 7). sampler2DShadow returns 0..1 PCF result.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;
// Scene TLAS - RT sun shadow on blades (trees / castle / props are not
// all in the raster shadow map).
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Grass eligibility mask. R8 1024² — single source of truth shared
// with terrain_raymarch.frag. CPU-baked from the same FBM (presence)
// and a sample_terrain_height finite-difference (slope) the engine
// already runs at level load. Sampling this once per cell replaces a
// 9-call noised() storm in the inner loop of map() — that was the
// dominant cost in this pass per the latest perf review.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
// Depth PRE-PASS copy (binding 27). Bounds the march: nothing behind the
// nearest opaque surface can be visible, so stop there.
layout(set = 0, binding = 27) uniform sampler2D u_prepass_depth;

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
    // Branchless boundary handling — same trick as terrain_raymarch.frag.
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
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

// Hermite cubic — MUST match terrain_raymarch.frag's noised() and
// the CPU rm_noised bake exactly so grass blades sit on the same
// FBM surface the terrain pass renders. (Reverted quintic experiment.)
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
const float kPlateauExtent = 11.5;   // matches world.cpp HeightmapParams
const float kPlateauHeight = 22.0;
const float kPlateauBlend  = 20.0;

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

// Distance-LOD variant — fades the per-march octave count from a fixed
// 8 at ray_t < lod_near_m down to lod_min_octaves past lod_far_m. Same
// LOD ramp endpoints + fractional-octave morph as terrain_raymarch.frag,
// so grass blades sit on the same surface the terrain raymarch draws
// even at distance. Params come from pc.emissive.{y,z,w} (mirrors the
// terrain shader's push-constant layout).
// Absolute heightmap (dim+1 texels over kHeightmapSide, any cell size,
// centred on the origin) - the surface the
// MESH terrain is built from, including sculpts, the road and craters.
layout(set = 0, binding = 17) uniform sampler2D u_heightmap_full;
float terrainH_mesh(vec2 wp) {
    float W = float(textureSize(u_heightmap_full, 0).x);
    vec2 uv = ((wp / kHeightmapSide + 0.5) * (W - 1.0) + 0.5) / W;
    return textureLod(u_heightmap_full, uv, 0.0).r;
}

float terrainH_lod(vec2 wp, float ray_t) {
    // Mesh-terrain mode: one bilinear fetch instead of 8 octaves of FBM
    // per march step (the main cost of this pass), and the blades stand on
    // the surface actually drawn.
    if (pc.emissive.x > 0.5) return terrainH_mesh(wp);
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
float sampleHeight(vec2 wp_xz) { return pc.emissive.x > 0.5 ? terrainH_mesh(wp_xz) : terrainH(wp_xz); }
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
// Per-pixel SDF invariants — slider-driven, identical for every map()
// call, so the outer raymarch loop hoists them once and passes via
// this struct. Cuts the per-step setup cost from 128× per pixel to 1×.
struct MapCtx {
    float slope_max;
    float density_eff_near;   // density at t = soft_d
    float density_eff_far;    // density at t = kMaxD
    float soft_d;
    float kMaxD;
    bool  underwater;         // global toggle (water_params.x > 0.5)
    float water_y;
};

// Rotation of the blade that won the last map() call (= the hit blade
// after the march): gives the blade face normal for lighting.
float g_blade_rot = 0.0;
float map(vec3 p, float ray_t, MapCtx ctx, out bool out_blade_hit) {
    out_blade_hit = false;
    float terrainY = sampleHeight_lod(p.xz, ray_t);
    float distToTerrain = 0.5 * (p.y - terrainY);

    // Coarse guard: if we're far above terrain, skip the per-cell SDF
    // evaluation (blades top out at ~0.4 m + sway at the default height
    // slider; guard leaves headroom for slider values up to ~2).
    const float kGuard = 1.0;
    if (distToTerrain > kGuard) return distToTerrain - (kGuard - 0.6);

    vec2 cellId;
    const float kPeriod = 0.17;        // ~35 blades per m²
    vec3 lp = p;
    lp.y -= terrainY;                  // ground-relative
    lp.xz = opRepeat(lp.xz, vec2(kPeriod), cellId);

    // Altitude + water gates — match terrain_raymarch.frag's
    // grassEligibility so the grass shader and the green terrain tint
    // agree about where grass exists.
    float alt_factor = 1.0 - smoothstep(55.0, 75.0, terrainY);
    bool underwater  = ctx.underwater && terrainY < ctx.water_y;
    if (alt_factor <= 0.001 || underwater) return distToTerrain;
    // Castle keep-out: the courtyard floor slabs sit 5 cm above the
    // terrain, and blades rooted on the terrain poke up THROUGH them —
    // the courtyard read as a swampy green weave over the tiles. The
    // castle footprint is the plateau extent (kPlateauExtent) centred
    // on origin; no blades inside it.
    if (abs(p.x) < kPlateauExtent + 0.5 &&
        abs(p.z) < kPlateauExtent + 0.5) return distToTerrain;

    float d = distToTerrain;
    const float kPi2 = 6.28318530718;
    // Density at this pixel's distance — interp between the two
    // pre-baked endpoints. Loop-`t` is the distance from camera (ro
    // is camera_pos and rd is normalised), so length(p - cam) was
    // wasted sqrt+dot per map() call.
    float dist_w      = smoothstep(ctx.soft_d, ctx.kMaxD, ray_t);
    float density_eff = mix(ctx.density_eff_near, ctx.density_eff_far, dist_w);
    // Dynamic close-range density: extra blades per cell near the camera,
    // fading out smoothly (each extra blade appears by the per-cell hash,
    // so the lawn thickens gradually as you approach - no ring).
    // (Applied per blade below from the blade cell's own distance - it used
    // to depend on the march sample, so blades shimmered in and out.)
    float near_boost = scene.grass_shore_params.y * step(0.05, ctx.density_eff_near);
    float density_max = density_eff + near_boost * (1.0 - smoothstep(3.5, 10.0, ray_t - 1.0));
    // Aggressive zero-density early-out. When distance has faded
    // density to near zero, the inner 9-cell loop almost always finds
    // every cell rejected by `r.x > density_eff` — cost without
    // benefit. 0.03 cutoff keeps a few stragglers visible past the
    // soft fade so it doesn't read as a hard ring.
    if (density_max < 0.03) return distToTerrain;
    const float kEligThresh = 0.10;
    // Eligibility (presence mask x slope) ONCE per map() call, at the
    // centre cell. The mask is 2 m per texel and bilinear; the 3x3 cell
    // block spans 0.5 m, so the nine per-cell fetches returned the same
    // value to within a few percent - at nine texture reads per march
    // step. Bare ground / rock / road now also skips the whole blade loop.
    float elig;
    {
        vec2 mg = textureLod(u_grass_mask, (cellId * kPeriod / kHeightmapSide) + vec2(0.5), 0.0).rg;
        float slope_w = 1.0 - smoothstep(ctx.slope_max * 0.7, ctx.slope_max * 1.2, mg.g * 2.0);
        elig = mg.r * slope_w;
    }
    // Soft patch border ('grass edge softness'): the mask alone gave a
    // wall of full-height grass next to bare ground. A low-frequency noise
    // frays the outline, and toward it the grass thins out and shortens.
    float edge_soft = scene.grass_side_lit_params.w;
    float elig_gate = smoothstep(0.04, 0.30, elig);     // no noise on cleared ground (road, rock)
    if (elig + 0.175 * edge_soft * elig_gate < kEligThresh) return distToTerrain;
    // Drop the 8 neighbour cells past ~35 % of kMaxDist — at that
    // distance blades are sub-pixel and the neighbour offsets
    // contribute < 1 pixel of silhouette difference. Keeps centre
    // cell only → 9× cheaper per map() call on long-range pixels.
    // Scales with kMaxDist (was hardcoded 40 m which capped well
    // before kMaxDist when the user's grass-distance slider was set
    // above 100 m).
    int rad = (ray_t > ctx.kMaxD * 0.35) ? 0 : 1;
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            vec2 nId = cellId + vec2(float(dx), float(dy));
            // Density-driven per-cell skip — only hash the cells we
            // didn't already reject by mask.
            // Up to two blades per cell: density 0..1 fills cells, 1..2 adds
            // a second blade (near rings only - the far single-cell path
            // keeps one). Each blade gets its own lean and width variation.
            // Per-BLADE-CELL edge noise and near boost (stable: they depend on
            // the cell, not on where along the ray this sample is).
            vec2  cw = nId * kPeriod;
            float en = hash12(floor(cw / 0.9)) * 0.6 + hash12(floor(cw / 2.7) + 17.0) * 0.4;
            float elig_n = elig + (en - 0.5) * 0.35 * edge_soft * elig_gate;
            if (elig_n < kEligThresh) continue;
            float edge_k = mix(1.0, smoothstep(kEligThresh, kEligThresh + 0.55, elig_n), edge_soft);
            float cdist = distance(vec3(cw.x, p.y, cw.y), scene.camera_pos.xyz);
            float dens_n = density_eff + near_boost * (1.0 - smoothstep(3.5, 10.0, cdist));
            const int nb = (rad > 0) ? clamp(int(ceil(dens_n)), 1, 3) : 1;
            for (int bi = 0; bi < nb; ++bi) {
            vec4 r = hash42(nId + vec2(31.7, 47.3) * float(bi));
            if (r.x > (dens_n - float(bi)) * mix(0.25, 1.0, edge_k)) continue;
            vec3 np = lp - vec3(float(dx), 0.0, float(dy)) * kPeriod;
            // Per-blade rotation + sub-cell xy offset.
            np.xz = rotate2d(r.z * kPi2) * np.xz;
            np.xz += (r.xy - 0.5) * kPeriod;
            float bh = clamp(pc.grass_params.w, 0.05, 3.0) * 0.36 * mix(0.45, 1.0, edge_k);
            // Wind sway + a per-blade static lean (some blades stand, some
            // splay out), bending with height.
            float wind_t = pc.color.z + r.w * kPi2;
            float bend   = sin(wind_t * 1.4) * pc.color.y * 0.025;
            float lean   = (fract(r.w * 13.7 + r.z * 5.3) - 0.5) * 0.30;
            np.x += (bend + lean * bh) * smoothstep(0.0, 1.7 * bh, np.y);
            float scale = max(0.65, sqrt(r.w)) * bh;
            float thick = 0.002 * (0.7 + 0.8 * fract(r.z * 7.1));
            float bladeDist = sdGrassBlade(np / scale, thick / scale) * scale;
            if (bladeDist < d) {
                d = bladeDist;
                out_blade_hit = true;
                g_blade_rot = r.z * kPi2;
            }
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
    vec2 uv  = gl_FragCoord.xy * scene.viewport.zw;
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
    // GPU-TDR safety cap: when the player stands IN dense grass and
    // looks horizontally across it, every pixel can burn the worst-
    // case budget × 9 cells × hash+sin+sqrt per step. Past a certain
    // budget the cost outruns the Windows TDR (~2 s on most drivers)
    // and the device is lost. 96 still gives plenty of convergence
    // headroom for normal play; the distance-aware breaks below
    // already collapse far rays to 16-32. Was 128.
    const int   kMaxSteps = 96;
    float t = 0.0;
    bool hit = false;
    bool blade_hit = false;
    float hit_d = 0.0;            // SDF distance at the converged hit
    float hit_thresh = 0.0;       // hit threshold at that t — alpha denom
    // Pixel size projected at unit distance — used to early-out when the
    // SDF distance is below sub-pixel detail at this t.
    float pixSize = 2.0 * scene.viewport.z;

    // Per-pixel map() invariants — slider-driven values that don't
    // change per step. Computing once here saves the per-step setup
    // cost (sqrt, smoothstep, several muls + UBO loads) × 128 steps.
    MapCtx ctx;
    {
        float slope_n_min = scene.grass_extra.z;
        ctx.slope_max = sqrt(max(1e-4,
                                  1.0 / max(slope_n_min * slope_n_min, 1e-4) - 1.0));
        float density   = clamp(scene.grass_color_top.w, 0.0, 2.0);   // >1: 2nd blade per cell
        float dist_dens = clamp(scene.grass_extra.w, 0.0, 1.0);
        float far_mul   = max(1.0 - dist_dens * 0.85, 0.0);
        ctx.density_eff_near = density;
        ctx.density_eff_far  = density * far_mul;
        ctx.kMaxD = max(pc.color.x, 4.0);
        ctx.soft_d = max(pc.color.w, 1.0);
        ctx.underwater = scene.water_params.x > 0.5;
        ctx.water_y    = scene.water_params.y;
    }

    // Distance-aware step budget. Past ~55 % / ~85 % of the user's
    // kMaxDist the worst-case grazing rays burn the full 128 budget
    // converging on sub-pixel blade edges that won't matter — TAA
    // absorbs the residual noise. Scale with kMaxDist so the user's
    // grass-distance slider keeps working at any value (was hardcoded
    // 60 m / 100 m which capped well before kMaxDist when the slider
    // pushed past 100 m).
    // Lossless march bound from the prepass depth: past the nearest
    // opaque surface every hit would fail the depth test anyway. Turns
    // castle walls / floors / props from a full 96-step march into ~0.
    float t_limit = kMaxDist;
    int   i_bias  = 0;
    {
        float zpre = texelFetch(u_prepass_depth, ivec2(gl_FragCoord.xy), 0).r;
        if (zpre < 0.99999) {
            vec4 sp = pc.model * vec4(ndc.xy, zpre, 1.0);
            float t_surf = distance(sp.xyz / sp.w, ro);
            t_limit = min(t_limit, t_surf + 0.05);
            // Skip the empty air: when the prepass surface IS the terrain,
            // blades (< ~0.7 m tall) can only touch this ray within a
            // short distance before it reaches the ground. Start the march
            // there instead of at the camera. At grazing angles that
            // stretch is ~0.9 m / cos(angle to the ground normal), capped
            // at 14 m (blade tips on ridges further back may be skipped -
            // sub-pixel at those angles). Was the bulk of the march cost:
            // ~all steps were spent sphere-tracing through air.
            vec3  ps = sp.xyz / sp.w;
            float yS = sampleHeight_lod(ps.xz, t_surf);
            // NOT when the prepass surface is under water: the depth there
            // is the lake BED (water isn't in the prepass), far behind the
            // shore blades that silhouette against the water - starting
            // the march near the bed skipped them all (no grass in front
            // of water).
            // ONLY with the mesh terrain (pc.emissive.x): the raymarched terrain
            // is not in the prepass, so there the prepass surface is always an
            // object. Its low parts (trunk bases, rocks, small bushes) pass the
            // 'near the ground' test below even when they stand BEHIND a hill -
            // the march then started out at the object and skipped every blade
            // on the hill in front, punching the object's outline through the
            // grass.
            if (pc.emissive.x > 0.5 &&
                abs(ps.y - yS) < 0.6 && ps.y > scene.water_params.y + 0.05) {
                vec3  Ns   = calcTerrainNormalApprox(ps.xz, yS, t_surf);
                float cosg = abs(dot(rd, Ns));
                float back = min(14.0, 0.9 / max(cosg, 0.06));
                t = max(t, t_surf - back);
                // Keep the old far-ray budget: a camera-start march spent
                // ~24 steps in air before reaching far ground, so the
                // distance caps below broke those rays early; without this
                // they converge on sub-pixel far blades (distant sparkle).
                i_bias = (t > kMaxDist * 0.55) ? 24 : 0;
            }
        }
    }
    float step_cap_a = kMaxDist * 0.55;
    float step_cap_b = kMaxDist * 0.85;
    for (int i = 0; i < kMaxSteps; ++i) {
        if (t >= t_limit) break;
        if (i + i_bias >= 32 && t > step_cap_a) break;
        if (i + i_bias >= 16 && t > step_cap_b) break;
        vec3 p = ro + t * rd;
        bool step_blade;
        float d = map(p, t, ctx, step_blade);
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
        // Distance-ramped step factor. Close-up needs the 0.7 cone-
        // safety factor for grazing rays through dense blade fields.
        // Past 50 m blades are sub-pixel and the cone-safety overhead
        // is pure cost — extend the ramp to 2.0 by 120 m so distant
        // rays cover 3× the ground per iteration.
        float step_f = mix(0.70, 0.95, smoothstep(15.0, 50.0, t));
        step_f = mix(step_f, 2.0, smoothstep(50.0, 120.0, t));
        t += d * step_f;
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
    // RT sun shadow: trees, castle and props cast onto the grass (the
    // raster shadow map misses most of them). One ray per grass pixel,
    // started a little up the sun direction so the blade's own ground
    // (terrain BLAS) can't self-hit at the base.
    if (sun_lit > 0.0 && L.y > 0.0) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS,
                              gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                              0x01u, p + vec3(0.0, 0.04, 0.0), 0.3, L, 300.0);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
            sun_lit = 0.0;
    }

    // Per-blade colour: lerp between bottom (base) and top (tip) by
    // height above ground. Both ends are user-driven via UI sliders
    // (scene.grass_color_bottom / grass_color_top). Slight low-freq
    // hash modulation breaks regularity without needing two separate
    // "young/old" colour vars.
    float h_above = max(0.0, p.y - terrainY);
    // Normalise by the actual max blade height (1.7 SDF units × height
    // factor) so the base→tip gradient spans the whole blade. The old
    // /1.2 assumed the unscaled 1.2+ m blades — with lawn-scale blades
    // it left every pixel at the dark base colour.
    float bh_max  = 1.7 * clamp(pc.grass_params.w, 0.05, 3.0) * 0.36;
    float h_t     = min(h_above / bh_max, 1.0);
    float clump   = hash12(floor(p.xz * 0.5)) * 0.15;   // ±15% per-clump tint shift
    vec3 grassCol = mix(scene.grass_color_bottom.rgb,
                        scene.grass_color_top.rgb, h_t);
    grassCol *= 1.0 - clump;

    // Shoreline tint — UI sliders (rt_.grass_shore_color/strength/
    // distance). Cheap: 1 sub + 1 smoothstep + 1 mix per visible
    // grass pixel. Strength 0 disables the entire branch via the
    // mix weight collapsing to 0. terrainY is already in scope.
    {
        float strength = scene.grass_shore_color.a;
        if (strength > 1e-3) {
            float fade = max(0.05, scene.grass_shore_params.x);
            float h_above_water = max(0.0, terrainY - scene.water_params.y);
            float shore_w = 1.0 - smoothstep(0.0, fade, h_above_water);
            grassCol = mix(grassCol, scene.grass_shore_color.rgb,
                            shore_w * strength);
        }
    }

    // Height-from-ground AO. Slider-controlled floor (0 = darkest base,
    // 1 = no AO darkening); the lerp above already shifts colour with
    // height, so the AO term mostly adds extinction at the very base.
    float ao_floor = clamp(scene.grass_color_bottom.w, 0.0, 1.0);
    float ao = ao_floor + (1.0 - ao_floor) * pow(h_t, 0.6);

    // Per-blade "lean" variation — hash the cell ID and tilt the
    // effective n_dot_l so adjacent blades on the same flat ground
    // don't all shade identically. ±0.25 around base, then half-
    // Lambert wrap so the shadow side isn't pitch-black.
    float blade_var = hash12(floor(p.xz / 0.20)) - 0.5;  // [-0.5, 0.5]
    // Blade face normal from the hit blade's rotation (the blade is a
    // flat strip with its face along local z), turned toward the viewer
    // and leaned toward the ground normal for the blade's curl. The face
    // turned to the sun is lit; the far side only gets light transmitted
    // through the leaf - a real dark side instead of every blade shaded
    // like the flat ground beneath it.
    vec3 Nf = vec3(-sin(g_blade_rot), 0.0, cos(g_blade_rot));
    if (dot(Nf, rd) > 0.0) Nf = -Nf;
    vec3 Nb = normalize(Nf + N * 0.45);
    float ndl_b = dot(Nb, L) + blade_var * 0.3;
    // Same scattering model as the tree leaves: wrap diffuse on the lit
    // face, and on the far face the sun transmitted through the blade
    // (yellow-green, stronger toward the thin tip).
    float n_dot_l_wrap = max(0.0, (ndl_b + 0.35) / 1.35);
    float blade_back   = clamp(-ndl_b, 0.0, 1.0);

    vec3 sun_term = scene.sun_color.rgb * scene.sun_color.a *
                    (n_dot_l_wrap * sun_lit) +
                    scene.sun_color.rgb * scene.sun_color.a * vec3(1.25, 1.40, 0.45) *
                    (blade_back * 0.42 * sun_lit * mix(0.45, 1.0, h_t));
    // Sky fill. 0.18 left shaded grass about four times darker than the
    // shaded ground beside it (terrain uses 0.75 * hemisphere), so grass in
    // a tree or hill shadow went near black. A little ground bounce too.
    vec3 sky_term = scene.sky_color.rgb * 0.46 + scene.sun_color.rgb * scene.sun_color.a * 0.02;
    // Side-lit blade shading: sun-side brighter, away-side darker.
    // Uses RAW dot(N, L) so the sign carries direction (n_dot_l above
    // was max'd to 0 for Lambert, which would always tint upward).
    float side_tint = 1.0;
    {
        float sl_on  = scene.grass_side_lit_params.y;
        float sl_str = scene.grass_side_lit_params.x;
        if (sl_on > 0.5 && sl_str > 0.001) {
            float raw_ndl = dot(N, L);
            side_tint = clamp(1.0 + sl_str * raw_ndl,
                              1.0 - sl_str * 0.9,
                              1.0 + sl_str * 0.9);
        }
    }
    // The root darkening (ao) is gentler on the sky fill than on the sun.
    vec3 col = grassCol * (sun_term * side_tint * ao + sky_term * mix(1.0, ao, 0.6) * mix(0.55, 1.0, sun_lit));

    // View-dependent backlit / translucency — sun shining THROUGH the
    // blade from behind makes the leaves glow warm. Strongest when:
    //   * looking toward the sun (v_dot_l > 0)
    //   * the surface isn't already directly lit (1 - n_dot_l)
    //   * we're on the upper part of the blade (h_t)
    // Same trick the rasterised grass.frag uses — was missing here so
    // raymarched grass had no view-dependent tone shift, every blade
    // read identically from front vs back.
    float v_dot_l  = max(0.0, dot(rd, L));   // rd points away from camera
    float v_dot_l2 = v_dot_l * v_dot_l;
    float v_dot_l4 = v_dot_l2 * v_dot_l2;
    vec3 backlit = scene.sun_color.rgb * scene.sun_color.a *
                   (v_dot_l4 * (1.0 - n_dot_l) * h_t) *
                   vec3(0.65, 0.95, 0.35) * sun_lit;
    col += backlit * grassCol;

    // Fresnel-edge rim using the (per-blade-jittered) terrain normal.
    // pow(x, 3) → x*x*x (no exp/log).
    float fr_in = clamp(1.0 + dot(N, rd), 0.0, 1.0);
    vec3 fres = vec3(0.18, 0.18, 0.10) * ao *
                (fr_in * fr_in * fr_in);
    col += fres;

    // Distance fade colour target — eases blade colour INTO the
    // underlying ground tint so the silhouette transition is invisible.
    // Two ground colours blend by camera distance using the same
    // anchors the terrain shader uses (30..200m), keeping the two
    // passes in lock-step.
    // Far anchor comes from rt_.grass_ground_tint_far_distance (UI
    // slider, packed into grass_color_ground_far.w). Near anchor is
    // 0 so the close→far ramp is one continuous gradient over the
    // full 0..far range — no inflection band where the smoothstep
    // slope jumps. Reads as a true merge instead of a visible knee.
    float tint_far  = max(50.0, scene.grass_color_ground_far.w);
    float ground_t  = smoothstep(0.0, tint_far, t);
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

    // Distance fog — final atmospheric mix. Same exp² as terrain.
    {
        float strength = scene.distance_fog_color.a;
        if (strength > 1e-3) {
            float density   = clamp(scene.distance_fog_params.x, 0.0, 0.1);
            float start_d   = scene.distance_fog_params.y;
            float height_top = scene.distance_fog_params.z;
            float max_alpha  = scene.distance_fog_params.w;
            float d_raw = max(0.0, length(p - scene.camera_pos.xyz) - start_d);
            float dx = d_raw * density;
            float fog = 1.0 - exp(-dx * dx);
            if (height_top > 0.5) {
                fog *= 1.0 - smoothstep(0.0, height_top, p.y);
            }
            float fa = clamp(fog * strength, 0.0, max_alpha);
            col = mix(col, scene.distance_fog_color.rgb, fa);
        }
    }
    outColor = vec4(col, final_alpha);
    outGBuffer0 = vec4(clamp(col, vec3(0.0), vec3(1.0)),
                       5.0 / 255.0);   // mat 5 = pass-through
    // Best-effort normal: use the cumulative marched hit normal if a
    // local var carries one; otherwise zero — the downstream deferred
    // pass-through reads scene_color, not gbuffer1, so this is just for
    // any future normal-aware consumer (SSR, etc).
    outGBuffer1 = vec4(0.5, 0.5, 0.5, 0.0);

    // Honour LESS_OR_EQUAL depth test so a closer wall already in
    // depth_image_ correctly occludes the marched grass.
    vec4 clip = pc.mvp * vec4(p, 1.0);
    gl_FragDepth = clamp(clip.z / max(clip.w, 1e-4), 0.0001, 0.9998);

    // Real screen-space motion vector for TAA reprojection. Grass is
    // world-static (per-blade wind sway is sub-pixel), so prev_uv
    // comes from prev_view_proj applied to the SAME world hit point.
    // Was hardcoded to vec2(0) — that meant TAA sampled the previous
    // frame at the SAME screen position, so when the camera strafed
    // grass blended in stale terrain colour from where it HAD been,
    // appearing to lag the rest of the scene by ~history-blend frames.
    // Using clip-space directly (not gl_FragCoord/viewport.zw) makes
    // it resolution-independent.
    vec4 prev_clip = pc.prev_mvp * vec4(p, 1.0);
    if (clip.w > 0.0001 && prev_clip.w > 0.0001) {
        vec2 cur_uv  = clip.xy      / clip.w      * 0.5 + 0.5;
        vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
        vec2 m = cur_uv - prev_uv - scene.restir_params.zw;   // cube.frag convention (unjittered)
        // NaN/inf guard + clamp to a sane range. A blade hit very
        // close to the camera with a glancing prev-frame projection
        // can produce |motion| > 1.0 which poisons TAA history and
        // (under enough motion) lands the next frame's reprojection
        // in undefined memory → device lost. Clamp ±0.5 = ±half-
        // screen, which is wider than any real per-frame motion.
        if (!any(isnan(m)) && !any(isinf(m))) {
            outMotion = clamp(m, vec2(-0.5), vec2(0.5));
        } else {
            outMotion = vec2(0.0);
        }
    } else {
        outMotion = vec2(0.0);
    }
}
