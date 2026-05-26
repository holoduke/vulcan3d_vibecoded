#version 460

// Tessellation evaluation: barycentric-interpolate the patch, then add
// fine detail displacement to the ground close to the camera and
// recompute the normal from the displacement gradient so lighting
// matches the new micro-relief. Emits exactly the varyings cube.frag
// consumes (same set/locations as terrain.vert).
//
// Terrain mesh draw uses an identity model matrix → object space ==
// world space (same assumption terrain.vert / .vert pass-through use).

// cw: the chunk index buffer's triangles are CCW-front (the plain
// terrain pipeline renders them with cull BACK + CCW front face). Fed
// through the tessellator the generated primitives come out with the
// opposite domain winding, so `cw` here makes the final triangles
// CCW-front again → single-sided, no z-fighting double-draw.
layout(triangles, fractional_odd_spacing, cw) in;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    // Terrain draws stuff the rocky displacement knobs into
    // grass_params.z (amplitude, metres) and .w (mip smoothing
    // bias, 0 = crisp). Slots .x and .y are sand-ripple / grass-
    // line scales consumed by cube.frag (terrain frag path).
    vec4 grass_params;
} pc;

// Scene SPOM/material height-map array. Layout MUST match cube.frag's
// binding 12. We only read slot [5] (the rocky-rugged-terrain height
// for the grass-band splat material; see vk_engine.h's
// kSpomMaterialCount comment for the full slot map). The descriptor
// set layout in descriptors.cpp adds the .tese stage flag for this
// binding so the lookup is valid.
layout(set = 0, binding = 12) uniform sampler2D u_height[6];

// Scene UBO -- declared only as a PREFIX through spom_params (the only
// field we need here). The std140 layout of every vec4 / ivec4 puts the
// next field at offset+16, so this prefix is byte-identical with
// cube.frag's full SceneUBO declaration through spom_params. Used to
// match the ground-tile metres knob (scene.spom_params.z) so vertex
// displacement aligns with the visible texture tiling at any slider
// value. descriptors.cpp adds VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
// to binding 0's stage flags so this read is valid.
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
    vec4  _scene_pad[24];
    vec4  restir_params;
    vec4  spom_params;          // .z = ground tile metres (UI slider)
} scene;

layout(location = 0) in vec3 tcPos[];
layout(location = 1) in vec3 tcNormal[];
layout(location = 2) in vec2 tcUv[];

layout(location = 0) out vec3 vNormal;
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
layout(location = 6) out vec3 vObjectPos;
layout(location = 7) flat out vec3 vObjectNormal;
layout(location = 8) out vec4 vPrevClip;

// --- Cheap value noise (hash-based) for the detail displacement -----
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float vnoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);            // smooth (Hermite)
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y) * 2.0 - 1.0; // [-1,1]
}
// 4-octave fBm in [-1,1]. `sharp` blends from smooth rolling noise
// (0) to ridged/sharp crests (1) — rocky ground wants ridges, sand
// and snow want smooth swells.
float fbmH(vec2 q, float sharp) {
    float a = 0.0, amp = 1.0, f = 1.0, norm = 0.0;
    for (int i = 0; i < 4; ++i) {
        float n = vnoise(q * f);              // [-1,1]
        float r = 1.0 - abs(n);               // ridged: peak at n=0
        r = r * r * 2.0 - 1.0;                // sharpen → [-1,1]
        n = mix(n, r, sharp);
        a += n * amp; norm += amp;
        amp *= 0.5; f *= 2.0;
    }
    return a / max(norm, 1e-4);
}
// Per-chunk border taper. pc.color = (chunkMinX, chunkMinZ,
// 1/sizeX, 1/sizeZ). Displacement MUST be zero at every chunk edge:
// neighbouring chunks (and the flat non-tessellated far chunks, and
// the skirts) all keep the base mesh height there, so a tapered
// border guarantees crack-free seams regardless of chunk size, LOD
// or which pipeline drew the neighbour.
float borderMask(vec2 xz) {
    vec2 lp = (xz - pc.color.xy) * pc.color.zw;   // → [0,1] within chunk
    const float e = 0.035;
    return smoothstep(0.0, e, lp.x) * smoothstep(0.0, e, 1.0 - lp.x) *
           smoothstep(0.0, e, lp.y) * smoothstep(0.0, e, 1.0 - lp.y);
}
// Geometry smoothing pass (0 = raw per-type detail, 1 = fully smooth
// low-frequency swell). Low-passes the displacement BEFORE the albedo
// splat is applied in cube.frag so the materials sit on a clean rolling
// surface instead of fighting noisy micro-relief. The smooth version is
// the SAME field evaluated at a lower frequency with no ridge
// sharpening, so it stays seamless and bumps-up-only.
// Runtime-controlled via the terrain draw's pc.tex_params.z (UI
// slider). The terrain fragment path ignores that slot, and the
// depth-prime + colour passes push the SAME value, so the displaced
// surface stays bit-identical between them (no sky-patch z-fight).
float kTessSmoothV() { return clamp(pc.tex_params.z, 0.0, 1.0); }
#define kTessSmooth kTessSmoothV()
// Rock VERTEX relief strength (the coarse/silhouette half of the
// vertex+pixel displacement combo; cube.frag's parallax adds the fine
// sub-triangle cracks on top). Same pc.tex_params.y on the depth-prime
// AND colour tess passes, so the displaced geometry stays identical.
float kRockReliefV() { return clamp(pc.tex_params.y, 0.0, 1.0); }

// Masked + distance-faded per-type detail height at a world XZ.
// ampM/frq/shp are the ground-type-blended profile computed once per
// vertex in main(); evaluating this at xz and xz±e gives a normal
// that matches the exact displaced surface. Amplitudes are kept
// modest (rock peaks ~0.55 m) so it reads as relief, never spiky.
float detailH(vec2 xz, float fade, float ampM, float frq, float shp) {
    // NON-NEGATIVE (0..ampM): the displaced surface must never sink
    // below the un-displaced ray-traced terrain BLAS / bake — a sunk
    // fragment's RT self-shadow ray starts inside the terrain and
    // reports fully-occluded → pure-black "stains" in the noise
    // valleys. Bumps-up-only keeps the render surface ≥ the RT surface.
    // Detail field + a low-pass (half-frequency, un-sharpened) version;
    // kTessSmooth blends toward the low-pass to kill spiky relief.
    float nd = fbmH(xz * frq, shp);
    float ns = fbmH(xz * frq * 0.5, 0.0);
    float n  = mix(nd, ns, kTessSmooth) * 0.5 + 0.5;   // [0,1]
    return ampM * n * fade * borderMask(xz);
}

void main() {
    vec3 b = gl_TessCoord;   // barycentric weights

    vec3 p  = b.x * tcPos[0]    + b.y * tcPos[1]    + b.z * tcPos[2];
    vec3 nb = b.x * tcNormal[0] + b.y * tcNormal[1] + b.z * tcNormal[2];
    vec2 uv = b.x * tcUv[0]     + b.y * tcUv[1]     + b.z * tcUv[2];
    nb = normalize(nb);

    // ---- Phong (point-normal) tessellation ----
    // Plain tessellation just subdivides a triangle and LINEARLY
    // interpolates its three spiky corners, so the faceted sculpt stays
    // faceted no matter how dense. Phong tessellation projects the flat
    // sample onto each corner's tangent plane (from that corner's
    // normal) and blends — curving the subdivided surface so angular
    // hills round off. This is the part that actually makes the
    // tessellated mesh *geometrically* smoother. kTessSmooth drives the
    // shape factor; borderMask tapers it to 0 at chunk edges so the
    // curved patch still meets the flat non-tessellated neighbours with
    // no crack.
    {
        vec3 pn0 = normalize(tcNormal[0]);
        vec3 pn1 = normalize(tcNormal[1]);
        vec3 pn2 = normalize(tcNormal[2]);
        vec3 q0 = p - dot(p - tcPos[0], pn0) * pn0;
        vec3 q1 = p - dot(p - tcPos[1], pn1) * pn1;
        vec3 q2 = p - dot(p - tcPos[2], pn2) * pn2;
        vec3 pPhong = b.x * q0 + b.y * q1 + b.z * q2;
        float alpha = kTessSmooth * borderMask(p.xz);
        vec3 pSmooth = mix(p, pPhong, alpha);
        // BUMPS-UP-ONLY (same rule as the detail displacement): the sun
        // shadow is a ray-query against the un-tessellated terrain BLAS.
        // In concave areas Phong pulls the surface BELOW the base mesh →
        // the shadow ray starts inside the BLAS → fully occluded → solid
        // black stains (worse the higher the smoothing). Keep the curved
        // X/Z but never let the smoothed height sink below the base.
        pSmooth.y = max(pSmooth.y, p.y);
        p = pSmooth;
    }

    // Distance fade: full detail up close, zero by ~95 m so the far
    // edge of a tessellated chunk matches the flat non-tess chunks
    // beyond it (no seam / pop).
    float dist = max((pc.mvp * vec4(p, 1.0)).w, 0.05);
    float fade = 1.0 - smoothstep(60.0, 110.0, dist);

    // ---- Per-ground-type displacement profile ----
    // Classify the ground from world height + slope (approx; this is
    // relief detail, it doesn't need to match the albedo bands exactly)
    // and blend a (amplitude, frequency, sharpness) profile per type:
    //   sand  : low-freq smooth dunes
    //   grass : gentle small bumps
    //   dirt  : moderate, mildly broken
    //   rock  : sharp high-freq ridges (biggest amplitude)
    //   snow  : very smooth wind-blown swells
    float h     = p.y;
    float slope = 1.0 - clamp(nb.y, 0.0, 1.0);
    float w_sand  = 1.0 - smoothstep(5.0, 13.0, h);
    float w_dirt  = smoothstep(28.0, 55.0, h);
    float w_grass = clamp(1.0 - w_sand - w_dirt, 0.0, 1.0);
    float w_rock  = max(smoothstep(0.55, 0.80, slope),
                        smoothstep(78.0, 112.0, h));
    float w_snow  = smoothstep(120.0, 150.0, h);
    float base    = (1.0 - w_rock) * (1.0 - w_snow);
    w_sand *= base; w_dirt *= base; w_grass *= base;
    w_rock *= (1.0 - w_snow);
    float wsum = w_sand + w_dirt + w_grass + w_rock + w_snow + 1e-4;
    // Amplitude×frequency sets the max slope of the displacement; if
    // it approaches ~1 the recomputed normal can tilt past horizontal
    // and shade black. Kept well under that (rock is the steepest at
    // 0.35 m × 0.22 /m ≈ 0.5 slope) and the normal is clamped below.
    // Amplitudes large enough to actually read as relief on the 1 m
    // base mesh (bumps-up-only, so effective peak ≈ these values).
    // amp×freq kept ≲0.25 so it stays rolling, not spiky.
    // Sane dm-scale relief: rolling decimetre bumps, not mountains.
    float wRockN = w_rock / wsum;            // 0..1: how rocky this vert is
    float ampM = (w_sand*0.30 + w_grass*0.16 + w_dirt*0.35 +
                  w_rock*0.60 + w_snow*0.14) / wsum;   // metres
    // Real VERTEX relief — the coarse, silhouette-bearing half of the
    // vertex+pixel displacement combo (cube.frag's parallax adds the
    // fine cracks). A base boost on ALL near terrain so the slider is
    // obviously responsive (it used to gate purely on w_rock, which is
    // ~0 on gentle/low ground → looked like it did nothing), plus a big
    // extra push on actual rock. Bumps-up-only; amp×freq stays bounded
    // so the recomputed normal can't flip to black.
    ampM += kRockReliefV() * (0.40 + wRockN * 1.10);   // up to +1.5 m
    // Smoothing trims gentle terrain so spiky hills round off — but
    // rock KEEPS its amplitude (it should stay chunky even when the
    // smoothing slider is high for the grass/dirt spike fix).
    ampM *= mix(1.0, mix(0.55, 1.0, wRockN), kTessSmooth);
    float frq  = (w_sand*0.05 + w_grass*0.13 + w_dirt*0.15 +
                  w_rock*0.18 + w_snow*0.07) / wsum;   // 1/m
    float shp  = (w_grass*0.08 + w_dirt*0.20 + w_rock*0.45) / wsum;

    // Per-type displacement ON. The depth-prepass now uses the SAME
    // tessellation pipeline + push constants, so the displaced surface
    // is bit-identical between prime and color passes → no z-fight /
    // sky-colour rejects, and being primed means no black stains.
    float disp = detailH(p.xz, fade, ampM, frq, shp);

    // ---- Rocky-rugged-terrain height-map displacement (grass band) ----
    // Sample u_height[5] at the SAME world-XZ UV scale the cube.frag
    // splat uses (kGroundTile metres per repeat -- driven by the same
    // scene.spom_params.z UI slider cube.frag reads as g_tile, so the
    // displaced geometry stays aligned with the visible texture tiling
    // no matter where the user drags it). textureLod(.., 0) -- no mip
    // averaging so the high-frequency detail stays crisp (NO SMOOTHING
    // was an explicit goal). Bumps-up-only (the +rkBias offset is
    // subtracted out by sampling at uvR and using the raw value as a
    // positive height) so the displaced surface never sinks below the
    // un-tessellated base mesh -- otherwise the RT-vs-render mismatch
    // produces black self-shadow stains (project history -- see
    // MEMORY note "quake-like terrain black stains").
    //
    // The amplitude (kRockyAmp = 0.30 m) is the user-requested default
    // and is gated by w_grass so the bump-relief only fires on the
    // grass band material, fades with distance (`fade`), and tapers to
    // zero at chunk edges (`borderMask`) so neighbour chunks stitch
    // seamlessly. Same e-step central-difference normal perturbation
    // as the per-type detail above so lighting follows the new relief.
    // Same min-clamp as cube.frag's g_tile (avoid divide-by-near-zero
    // and pin the slider's lower bound). Both depth-prime and colour
    // tess passes read the SAME UBO so the displaced surface stays
    // bit-identical between them (no sky-patch z-fight).
    float kGroundTile = max(scene.spom_params.z, 0.25);
    // Rocky-grass vertex displacement. Amplitude (metres) AND
    // displacement mip-blur are runtime-driven from the terrain draw's
    // pc.grass_params.zw (UI sliders Disp amp / Disp smooth).
    // Mip 0 = crisp high-freq detail. Higher mip = blur the height
    // sample so adjacent dense-tess verts agree, killing per-vert
    // spikes when the user pushes amplitude high or pulls tess down.
    float kRockyAmp = max(0.0, pc.grass_params.z);
    float kDispMip  = clamp(pc.grass_params.w, 0.0, 6.0);
    float rk_w = w_grass * fade * borderMask(p.xz) * kRockyAmp;
    if (rk_w > 1e-4) {
        vec2 uvR = p.xz / kGroundTile;
        // Linear displacement: peak of the height texture (h=1) gets
        // the full amplitude, valley (h=0) stays at the base mesh.
        // Was max(h-0.5, 0)*2 -- that discarded the bottom 50% of the
        // height map entirely, so only the brightest stones bumped
        // while everything in [0, 0.5] read as flat. Now every pixel
        // in the height texture contributes proportionally, so the
        // vertex relief behaves "just like the heightmap." Stays
        // bumps-up-only (h >= 0) to honour the BLAS-safe constraint
        // documented in the memory note.
        float h0 = textureLod(u_height[5], uvR, kDispMip).r;
        disp += h0 * rk_w;
    }
    vec3 pd = p + vec3(0.0, disp, 0.0);

    // Surface normal = the interpolated base normal, TILTED by the
    // displacement gradient. The old version rebuilt the normal from a
    // -nb.x/abs(nb.y) gauge — fragile on slopes (sign flips) and it
    // produced the green/pitch-black shading. This form is exact at
    // zero displacement (gxD=gzD=0 → n == normalize(nb), so the
    // tessellated near terrain shades IDENTICALLY to the plain far
    // mesh) and a stable cheap perturbation otherwise.
    const float e = 0.75;     // ~heightmap-cell finite-difference step
    float hxp = detailH(p.xz + vec2(e, 0.0), fade, ampM, frq, shp);
    float hxn = detailH(p.xz - vec2(e, 0.0), fade, ampM, frq, shp);
    float hzp = detailH(p.xz + vec2(0.0, e), fade, ampM, frq, shp);
    float hzn = detailH(p.xz - vec2(0.0, e), fade, ampM, frq, shp);
    // Add the rocky-grass height contribution to the central-difference
    // gradient so the SHADING normal follows the new vertex bumps
    // (otherwise the geometry would displace but lighting would still
    // come from the smooth base normal -> bumps look flat).
    if (rk_w > 1e-4) {
        vec2 uvR = p.xz / kGroundTile;
        vec2 duvR = vec2(e / kGroundTile, 0.0);
        // Same mip-bias as the displacement so shading-normal gradient
        // tracks the actual (possibly mip-blurred) geometry.
        // Linear (no -0.5 / max) to match the displacement formula
        // above, so the central-difference gradient -> shading normal
        // tracks the actual geometry.
        float rxp = textureLod(u_height[5], uvR + vec2(duvR.x, 0.0), kDispMip).r;
        float rxn = textureLod(u_height[5], uvR - vec2(duvR.x, 0.0), kDispMip).r;
        float rzp = textureLod(u_height[5], uvR + vec2(0.0, duvR.x), kDispMip).r;
        float rzn = textureLod(u_height[5], uvR - vec2(0.0, duvR.x), kDispMip).r;
        // rk_w is amp*w*fade*borderMask (in metres at the field's full
        // [0..0.5] range -> peak is rk_w when h=1). Add to the gradient
        // accumulators so the same clamp/strength rules apply below.
        // No *2: the displacement formula above is `h0 * rk_w` (was
        // `(h0-0.5)*rk_w*2`, where *2 cancelled the 0.5 bias). Keep
        // the gradient weights identical to the displacement weight.
        hxp += rxp * rk_w; hxn += rxn * rk_w;
        hzp += rzp * rk_w; hzn += rzn * rk_w;
    }
    // Clamp the displacement gradient so the perturbed normal can
    // never tilt past horizontal (which read as black patches).
    // Heavily damped: the GEOMETRY still displaces (relief is visible
    // as real bumps/silhouette), but the SHADING normal stays close to
    // the smooth base normal. Strong perturbation made micro-faces that
    // point away from the sun → n·L≤0 → the terrain-contrast pow()
    // crushed them to pure-black "stains". 0.18 strength + a high n.y
    // floor keeps every fragment lit.
    float gxD = clamp((hxp - hxn) / (2.0 * e), -0.7, 0.7);
    float gzD = clamp((hzp - hzn) / (2.0 * e), -0.7, 0.7);
    // Enough perturbation that the bumps catch light/shade (otherwise
    // the geometry moves but the surface looks flat). n.y floor still
    // keeps it from flipping fully away from the sun → no black.
    // Smoothing also relaxes the shading normal back toward the base
    // surface normal so the lit/shade pattern is a soft gradient, not
    // crisp micro-facets.
    float nStr = mix(0.65, 0.30, kTessSmooth);
    vec3 n = nb + vec3(-gxD, 0.0, -gzD) * nStr;
    n.y = max(n.y, mix(0.25, 0.45, kTessSmooth));
    n = normalize(n);

    gl_Position = pc.mvp * vec4(pd, 1.0);
    vWorldPos     = (pc.model * vec4(pd, 1.0)).xyz;
    vNormal       = mat3(pc.model) * n;
    vColor        = vec3(1.0);
    vEmissive     = vec4(0.0);   // terrain has no emissive; pc.emissive
                                 // is repurposed as the .tesc tess knobs
    vUv           = uv;
    vTexParams    = pc.tex_params;
    vObjectPos    = pd;
    vObjectNormal = n;
    vPrevClip     = pc.prev_mvp * vec4(pd, 1.0);  // terrain static
}
