#version 460

// Compose pass:
//   1. Sample HDR history (post-TAA, post-à-trous).
//   2. If this pixel had no geometry (depth ≈ 1.0), replace its color with a
//      procedural sky: gradient horizon→zenith, plus a sun halo + sharp sun
//      disc when the view ray points near the sun direction. Bloom natur-
//      ally glares the bright disc into a lens-flare-feel halo.
//   3. Spiral-tap bloom on the (possibly sky-augmented) HDR.
//   4. ACES Fitted tonemap → sRGB encode → swapchain.

layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D history_color;
layout(set = 0, binding = 1) uniform sampler2D history_depth;
// Equirectangular LDR sky panorama (sRGB sampler → linear in shader).
layout(set = 0, binding = 2) uniform sampler2D u_skybox;
// Mip 0 of the bloom chain (Karis-prefiltered downsample → upsample). The
// engine runs run_bloom_chain between TAA and compose; we sample it here at
// the same resolution as history_color and add it back additively. Replaces
// the per-pixel spiral_bloom that used to live in this shader.
layout(set = 0, binding = 3) uniform sampler2D u_bloom;

layout(push_constant) uniform ComposePC {
    vec4 viewport;       // x: w, y: h, z: 1/w, w: 1/h
    vec4 bloom_params;   // x: strength, y: threshold, z: radius_px, w: enabled
    vec4 sun_dir;        // xyz: normalized toward-sun
    vec4 sun_color;      // rgb + intensity in .a
    vec4 sky_color;      // rgb
    // Lens flare (Chapman-style screen-space):
    //   x: strength      (0..2, multiplier on the additive contribution)
    //   y: threshold     (HDR luminance below which a pixel doesn't seed flare)
    //   z: ghost dispersal (0..1, spacing between ghosts along the light axis)
    //   w: halo width    (0..0.5, distance from center of the halo ring)
    vec4 flare_params;
    // x: ghost count (int packed in float), y: chromatic aberration strength,
    // z: enabled (>0.5), w: spare
    vec4 flare_params2;
    // Screen-space sun position (xy: uv if z > 0; z: visibility 0/1 — 0 means
    // sun is behind the camera; w: spare). Used to gate lens flare so only
    // the sun seeds it — emissive lanterns and any other bright pixel won't.
    vec4 sun_screen;
    // Post-TAA unsharp mask. x: strength (0=off, ~0.5 typical, >1 punchy);
    // y..w: spare. The compose pass runs a 5-tap (center + 4 cardinal)
    // unsharp filter on history_color to recover detail that taa.frag's
    // 5x5 cross-bilateral à-trous blurred away. Cheap (5 extra texelFetches).
    vec4 sharpen_params;
    // Brush size indicator. xy = brush world XZ, z = radius (m),
    // w = enabled (>0.5 draws the ring). compose reconstructs the
    // pixel's world XZ from depth + inv_view_proj and tints near the
    // ring band so the radius shows on whatever surface the brush hit.
    vec4 brush_indicator;
    mat4 inv_view_proj;  // for view-ray reconstruction
} pc;

vec3 aces_fitted(vec3 x) {
    const mat3 in_m = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777);
    const mat3 out_m = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602);
    vec3 v = in_m * x;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(out_m * (a / b), 0.0, 1.0);
}

vec3 to_srgb(vec3 x) { return pow(max(x, vec3(0.0)), vec3(1.0 / 2.2)); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Synthetic "feature buffer" for the sun. The bloom mip chain doesn't see the
// sun: bloom_chain runs before compose, and the procedural sky (which paints
// the sun's bright disc onto background pixels) only happens inside compose.
// So sampling u_bloom returns zero at the sun's location — flare ended up
// only catching emissive geometry like lanterns.
//
// Building a synthetic sun spot from sun_screen gives the algorithm exactly
// what it needs (a bright source at the sun's screen position) AND makes
// lens flare sun-only by construction: nothing else writes into this
// function's output, so lanterns / muzzle flash / sparks can never seed
// ghosts.
// Per-ghost color tint LUT. Real lenses produce an alternating warm/cool
// palette as different anti-reflective coatings reflect different
// wavelengths preferentially. The Chapman article uses a 1D texture for
// this; we just hardcode a plausible 8-entry sequence.
vec3 ghost_tint(int idx) {
    if (idx == 0) return vec3(1.00, 0.85, 0.55);  // warm orange
    if (idx == 1) return vec3(0.55, 0.80, 1.10);  // cool blue
    if (idx == 2) return vec3(1.05, 0.70, 0.80);  // pink
    if (idx == 3) return vec3(0.70, 1.05, 0.85);  // green
    if (idx == 4) return vec3(1.10, 0.95, 0.65);  // yellow
    if (idx == 5) return vec3(0.65, 0.75, 1.05);  // pale blue
    if (idx == 6) return vec3(0.95, 0.85, 1.05);  // lavender
    return vec3(1.00, 0.90, 0.75);                // soft white
}

// Soft circular ghost. Uses anisotropic-along-axis radii (bigger across
// than along the optical axis would be more accurate, but for this scene
// a circular Gaussian reads well). Per-channel offset gives chromatic
// fringes. Returns the brightness per RGB channel at this output pixel.
vec3 ghost_blob(vec2 uv, vec2 center, vec2 axis_norm,
                float radius, float aberration) {
    float inv_r2 = 1.0 / max(radius * radius, 1e-6);
    vec2 d = uv - center;
    // Squared-distance reject — most pixels are far from any ghost
    // center; their exp() is below noise (~1e-4). Skip the 3 transcendentals
    // when the green channel's normalised distance² > 8 (exp(-8) ≈ 3e-4).
    if (dot(d, d) * inv_r2 > 8.0) return vec3(0.0);
    vec2 ab = axis_norm * aberration;
    float dr2 = dot(d + ab, d + ab);
    float dg2 = dot(d,      d);
    float db2 = dot(d - ab, d - ab);
    return vec3(exp(-dr2 * inv_r2),
                exp(-dg2 * inv_r2),
                exp(-db2 * inv_r2));
}

// Pseudo-lens-flare (John Chapman, 2017). Screen-space, no optical model:
//
//   1. Reflect uv around screen center → bright sources end up on the
//      opposite side of the image.
//   2. March along that reflected vector toward (and past) the center,
//      sampling the (luminance-thresholded) HDR history at each step. Each
//      step is a "ghost" — a faded, scaled, possibly inverted-color copy
//      of every bright source on the matching axis.
//   3. Add a halo: one extra sample at a fixed offset along the same axis,
//      gives the bright ring around the center.
//   4. Chromatic aberration on every sample, biased toward the center —
//      that's what makes flares look prismatic.
// 0..1 visibility of the sun: 1 = fully visible, 0 = fully occluded by
// geometry. Samples history_depth in a 5-tap cross around sun_screen — if
// any of the taps reads near the far plane (depth ≈ 1), at least part of the
// sun's disc is unblocked. Without this the flare keeps showing when the
// player walks behind a wall / pillar with the sun on the far side.
float sun_visibility() {
    if (pc.sun_screen.z < 0.5) return 0.0;
    vec2 c = pc.sun_screen.xy;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) return 0.0;
    const float kFar = 0.99999;
    // 5 taps in a small cross — covers the sun's apparent disc on screen
    // for typical FOVs. Each tap counts as "sun visible there" if its
    // depth is beyond kFar (no geometry occluding).
    float vis = 0.0;
    const float r = 0.012;
    // Clamp UVs so the cross taps near screen edges don't sample past the
    // [0,1] range and pick up CLAMP_TO_EDGE results that read as "sky"
    // (depth ≈ 1.0) → false-positive flare visibility at corners.
    vis += float(texture(history_depth, c).r                                            >= kFar);
    vis += float(texture(history_depth, clamp(c + vec2( r, 0), vec2(0.0), vec2(1.0))).r >= kFar);
    vis += float(texture(history_depth, clamp(c + vec2(-r, 0), vec2(0.0), vec2(1.0))).r >= kFar);
    vis += float(texture(history_depth, clamp(c + vec2(0,  r), vec2(0.0), vec2(1.0))).r >= kFar);
    vis += float(texture(history_depth, clamp(c + vec2(0, -r), vec2(0.0), vec2(1.0))).r >= kFar);
    return vis * 0.2;  // 0/5 .. 5/5
}

// Direct-render lens flare: ghosts are placed at known positions on screen
// (along the line from the sun's mirror through the screen center) and
// each is drawn as a per-channel Gaussian blob with its own size and color
// tint. No feature-buffer sampling — eliminates the "merging-into-one-blob"
// and "tiny-dot" failure modes of the indirect approach by giving each
// ghost direct control over its on-screen radius.
vec3 lens_flare(vec2 uv) {
    if (pc.flare_params2.z < 0.5) return vec3(0.0);
    if (pc.sun_screen.z < 0.5)    return vec3(0.0);
    float vis = sun_visibility();
    if (vis <= 0.0) return vec3(0.0);

    float strength  = pc.flare_params.x;
    float dispersal = pc.flare_params.z;
    float halo_w    = pc.flare_params.w;
    int   ghost_n   = clamp(int(pc.flare_params2.x), 1, 8);
    float aberr     = pc.flare_params2.y;

    vec2 sun_uv = pc.sun_screen.xy;
    // The optical axis on screen runs through the sun and the image center.
    // Ghosts appear OPPOSITE the sun (flipped around center), then fall in
    // toward and past center. axis_dir points from the sun's mirror toward
    // center.
    vec2 mirror   = vec2(1.0) - sun_uv;
    vec2 to_centr = vec2(0.5) - mirror;
    float axis_len = max(length(to_centr), 1e-4);
    vec2 axis_dir = to_centr / axis_len;

    // Bright color we'll modulate per ghost. Sun's HDR color works directly.
    vec3 sun_hot = pc.sun_color.rgb * pc.sun_color.a;

    vec3 ghosts = vec3(0.0);
    for (int i = 0; i < ghost_n; ++i) {
        // Ghost i sits along the axis from `mirror` toward center+beyond.
        // Negative dispersal would put it behind the mirror; we step
        // forward toward center.
        vec2 g_pos = mirror + to_centr * dispersal * float(i);
        // Skip ghosts that walk off-screen — they look broken when half-
        // clipped at an edge.
        if (g_pos.x < -0.05 || g_pos.x > 1.05 ||
            g_pos.y < -0.05 || g_pos.y > 1.05) continue;

        // Radius grows modestly per ghost, tracking the article's reference.
        // 0.030 ≈ 38 px on a 1280×720 screen — a clear soft disc.
        float radius = 0.030 + 0.014 * float(i);
        // Intensity falls off both with ghost index (later ghosts are
        // dimmer in real lenses) and with on-screen distance from center
        // (corner ghosts are dim, central ghosts are punchy).
        float index_fade = 1.0 / (1.0 + 0.5 * float(i));
        float edge_fade  = 1.0 - smoothstep(0.0, 0.8,
                                             length(g_pos - 0.5) * 2.0);
        // Chromatic fringe: R/G/B sampled at small offsets along the axis.
        // The amount of aberration grows slightly per ghost so the outer
        // ghosts read as more "rainbowed" than the punchy ones near the sun.
        vec3 rgb_w = ghost_blob(uv, g_pos, axis_dir, radius,
                                aberr * (1.0 + 0.5 * float(i)));

        ghosts += sun_hot * ghost_tint(i) * rgb_w * index_fade * edge_fade;
    }

    // Halo ring: a single soft disc at fixed distance from `mirror` along
    // axis_dir. Per-channel chromatic offset gives the rainbow-edged ring.
    vec3 halo_rgb = vec3(0.0);
    if (halo_w > 0.001) {
        vec2 halo_pos = mirror + axis_dir * halo_w;
        if (halo_pos.x > -0.05 && halo_pos.x < 1.05 &&
            halo_pos.y > -0.05 && halo_pos.y < 1.05) {
            float halo_r = 0.06;
            vec3 hw = ghost_blob(uv, halo_pos, axis_dir, halo_r, aberr * 1.5);
            float halo_edge = 1.0 - smoothstep(0.0, 0.9,
                                                length(halo_pos - 0.5) * 2.0);
            halo_rgb = sun_hot * vec3(1.0, 0.9, 0.85) * hw * 0.6 * halo_edge;
        }
    }

    return (ghosts + halo_rgb) * strength * vis;
}

// Equirect → UV. Inverse of the loader's UV → direction mapping:
//   longitude (yaw) = atan2(dir.x, dir.z) ∈ [-π, π]   → u ∈ [0, 1]
//   latitude  (pitch) = asin(dir.y)        ∈ [-π/2, π/2] → v ∈ [0, 1] (flipped)
vec2 equirect_uv(vec3 dir) {
    float lon = atan(dir.x, dir.z);
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(lon / (2.0 * 3.14159265) + 0.5,
                0.5 - lat / 3.14159265);
}

// Sky from a loaded equirect panorama. The skybox texture format determines
// the path the engine took on the C++ side: an HDR EXR keeps real sun
// radiance, an LDR JPG clamps it. We detect at runtime by sampling — if any
// channel exceeds 1 the source must be HDR (sRGB JPGs always decode ≤1).
//
// On HDR, the sun pixel is bright enough to feed the bloom mip chain on
// its own and the procedural disc/halo overlay would double up. On LDR we
// keep the overlay so the sun reads as bright after ACES squishes it.
vec3 sample_sky(vec3 dir) {
    vec2 uv = equirect_uv(normalize(dir));
    vec3 sky = texture(u_skybox, uv).rgb;
    bool is_hdr = max(max(sky.r, sky.g), sky.b) > 1.5;

    if (!is_hdr) {
        // LDR lift so the sky doesn't look flat once ACES squashes the scene.
        sky *= 1.4;
    }

    vec3 L = normalize(pc.sun_dir.xyz);
    float sd = max(dot(normalize(dir), L), 0.0);
    // Procedural overlay scales down hard when the source is already HDR —
    // a faint shimmer at the matched sun direction, not a full disc.
    float overlay = is_hdr ? 0.10 : 1.0;
    float halo = pow(sd, 32.0) * 0.5
               + pow(sd, 8.0)  * 0.10;
    sky += pc.sun_color.rgb * pc.sun_color.a * halo * overlay;
    float disc = smoothstep(0.99975, 0.99995, sd);
    sky += pc.sun_color.rgb * pc.sun_color.a * disc * (is_hdr ? 1.0 : 40.0);

    return sky;
}

// Reconstruct the world-space view ray for this pixel.
//
// Important: NO `ndc.y *= -1` here. The engine's projection matrix already has
// `proj[1][1] *= -1` (Vulkan Y-flip), and inv_view_proj inverts that. An extra
// flip would double-invert and mirror the sky vertically — which is the bug
// you saw where the sun "moved down" when you looked up.
//
// (TAA's reprojection has a flip on both sides — world is "wrong" but the
// round-trip cancels — so it works there. Sky doesn't round-trip; it uses
// the raw world direction, so the flip must be omitted here.)
// =====================================================================
// FSR1 — Edge-Adaptive Spatial Upsampling (EASU) + Robust Contrast-
// Adaptive Sharpening (RCAS). Fresh GLSL implementation following the
// algorithm described in AMD's FidelityFX-FSR1 reference (MIT, 2021).
// AMD's reference shader: https://github.com/GPUOpen-Effects/FidelityFX-FSR
// Original GLSL port for mpv (also MIT, by agyild) was used as an
// algorithm cross-check. Both upstream sources retain their copyright
// notices; this port is a clean reimplementation of the same
// well-documented algorithm. © 2021 Advanced Micro Devices, Inc., MIT.
// =====================================================================

// EASU helper: weight = (1.25 * dist^2 - dist^2) * (clamp(...))
// computes the FSR1 base lobe profile around a tap distance from
// the output sample location. This is the "lanczos-2 like" core
// before edge-direction biasing.
float easu_lobe(float dist2) {
    float d = clamp(dist2, 0.0, 4.0);
    float wA = 0.4 * d - 1.0;
    float wB = (25.0 / 16.0) * d - (25.0 / 16.0);
    return (wA * wA - 1.0) * (wB * wB);
}

// EASU directional weight — given the surface gradient direction
// (dir) and the per-tap offset, returns how strongly this tap
// contributes along the edge. Edges along the gradient get a long
// elliptical lobe; perpendicular gets a short one. This is the
// edge-aware part that makes EASU sharper than Lanczos.
float easu_direction(vec2 dir, vec2 off, float len) {
    // Project off onto dir → scalar tap distance along the edge.
    // dist2 is the "stretched" distance used by easu_lobe; len
    // controls how much the edge stretches the lobe.
    vec2 v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x)));
    v *= vec2(1.0 + (len - 1.0) * 0.5,    // stretch along edge
              1.0 / max(0.5, 1.0 - (len - 1.0) * 0.5));  // squash across
    return easu_lobe(dot(v, v));
}

// EASU 12-tap upscale — the heart of FSR1's spatial reconstruction.
// Samples a 4×4 neighbourhood of source texels (corners are zero-
// weighted, leaving 12 effective taps), computes the local gradient
// from luma differences in the centre 2×2, and weights each tap by
// (lobe × direction). Output is clamped to the local 2×2 luma min/
// max to suppress ringing.
vec3 fsr1_easu(sampler2D tex, vec2 uv, vec2 src_size) {
    vec2 inv_src = 1.0 / src_size;
    // Source position centred between pixels (the "phase").
    vec2 pp     = uv * src_size - 0.5;
    vec2 pp_f   = floor(pp);
    vec2 pp_fract = pp - pp_f;

    // Sample the 4×4 neighbourhood. Corner taps (b,c,r,s) get zero
    // direct weight — they only feed the gradient estimate. The
    // centre 2×2 (h,i,k,l) is the "primary" tap group.
    //   a b
    //   c d e f
    //   g h i j   <- centre row at pp_f.y..pp_f.y+1
    //   k l m n
    //     o p
    // (read order is row-major across the 4×4 minus corners)
    vec2 base = (pp_f + 0.5) * inv_src;
    vec3 b = texture(tex, base + vec2( 0.0, -1.0) * inv_src).rgb;
    vec3 c = texture(tex, base + vec2( 1.0, -1.0) * inv_src).rgb;
    vec3 d = texture(tex, base + vec2(-1.0,  0.0) * inv_src).rgb;
    vec3 e = texture(tex, base + vec2( 0.0,  0.0) * inv_src).rgb;
    vec3 f = texture(tex, base + vec2( 1.0,  0.0) * inv_src).rgb;
    vec3 g = texture(tex, base + vec2( 2.0,  0.0) * inv_src).rgb;
    vec3 h = texture(tex, base + vec2(-1.0,  1.0) * inv_src).rgb;
    vec3 i = texture(tex, base + vec2( 0.0,  1.0) * inv_src).rgb;
    vec3 j = texture(tex, base + vec2( 1.0,  1.0) * inv_src).rgb;
    vec3 k = texture(tex, base + vec2( 2.0,  1.0) * inv_src).rgb;
    vec3 o = texture(tex, base + vec2( 0.0,  2.0) * inv_src).rgb;
    vec3 p = texture(tex, base + vec2( 1.0,  2.0) * inv_src).rgb;

    // Luma proxy — perceptual greyscale.
    #define _L(c) dot((c), vec3(0.5, 1.0, 0.5))
    float lE = _L(e), lF = _L(f), lI = _L(i), lJ = _L(j);

    // Gradient from the centre 2×2: horizontal & vertical luma
    // differences. The dominant axis becomes the edge direction.
    vec2 dir = vec2((lF + lJ) - (lE + lI),
                     (lI + lJ) - (lE + lF));
    float dir_len = length(dir);
    // Normalise; fall back to (1,0) on flat regions.
    if (dir_len < 1e-4) dir = vec2(1.0, 0.0);
    else                dir /= dir_len;
    // "Length" feature — strength of the edge, drives lobe stretch.
    float len = clamp(dir_len * 4.0, 0.0, 1.0);
    len = 1.0 + len * 0.5;       // 1.0 (no stretch) → 1.5 (long lobe)

    // Per-tap weight = lobe(dist²) × direction(off).
    // Tap offsets are relative to the output sample position pp_fract.
    vec3 sum = vec3(0.0);
    float w_tot = 0.0;
    #define TAP(c, ox, oy) { \
        vec2 off = vec2(ox, oy) - pp_fract; \
        float w = easu_direction(dir, off, len); \
        sum += (c) * w; w_tot += w; \
    }
    TAP(b,  0.0, -1.0)
    TAP(c,  1.0, -1.0)
    TAP(d, -1.0,  0.0)
    TAP(e,  0.0,  0.0)
    TAP(f,  1.0,  0.0)
    TAP(g,  2.0,  0.0)
    TAP(h, -1.0,  1.0)
    TAP(i,  0.0,  1.0)
    TAP(j,  1.0,  1.0)
    TAP(k,  2.0,  1.0)
    TAP(o,  0.0,  2.0)
    TAP(p,  1.0,  2.0)
    #undef TAP

    vec3 result = sum / max(w_tot, 1e-4);

    // Local 2×2 luma min/max box — clamp output to suppress any
    // overshoot/undershoot ringing from the directional weights.
    vec3 mn = min(min(e, f), min(i, j));
    vec3 mx = max(max(e, f), max(i, j));
    return clamp(result, mn, mx);

    #undef _L
}

// RCAS — Robust Contrast-Adaptive Sharpening. Operates on the EASU-
// upscaled image (or any image) and adds adaptive sharpness without
// ringing. Algorithm: sample centre + 4 cardinals, compute min/max,
// derive a "lobe" weight from local contrast (smaller for noisy
// regions), apply (lobe × blur + center) / (1 + 4×lobe).
//
// `sharpness` is in [0, 1]. 0 = no sharpening, 1 = max strength.
vec3 fsr1_rcas(vec3 c, vec3 n, vec3 s, vec3 e, vec3 w, float sharpness) {
    // Per-channel min/max box.
    vec3 mn = min(min(n, s), min(min(e, w), c));
    vec3 mx = max(max(n, s), max(max(e, w), c));
    // Per-channel lobe: (mn / mx - 1) on the dim end, clamped.
    vec3 mx_safe = max(mx, vec3(0.001));
    vec3 lobe_rgb = max(-vec3(0.18745f), (mn - 1.0) / mx_safe);
    // Use the green channel as the dominant for a single weight (the
    // standard RCAS uses luma-ish greyscale; green is a reasonable
    // proxy that avoids per-channel halos).
    float lobe = lobe_rgb.g * sharpness;
    // Final weighted sum: (lobe * (n+s+e+w) + c) / (1 + 4*lobe).
    vec3 blur = (n + s + e + w);
    return (lobe * blur + c) / (1.0 + 4.0 * lobe);
}

// 9-tap Catmull-Rom bicubic upscale (Jorge Jimenez / Filmic AA — the
// standard "sharper than bilinear, no halos" filter used by most TAA
// implementations for history reconstruction). Collapses 16 corner
// taps to 9 bilinear-filtered taps via the offset-into-tap-group
// trick. Significantly crisper than plain bilinear at low render
// scales without the ringing artefacts you get from a naive bicubic.
//
// Final result is clamped to the local 2×2 min/max box to suppress
// any negative-weight overshoots near hard edges.
vec3 sample_catmull_rom(sampler2D tex, vec2 uv, vec2 src_size) {
    vec2 sample_pos = uv * src_size;
    vec2 tex_pos_1  = floor(sample_pos - 0.5) + 0.5;
    vec2 f          = sample_pos - tex_pos_1;

    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    vec2 w12     = w1 + w2;
    vec2 off12   = w2 / w12;

    vec2 tp0  = (tex_pos_1 - 1.0) / src_size;
    vec2 tp3  = (tex_pos_1 + 2.0) / src_size;
    vec2 tp12 = (tex_pos_1 + off12) / src_size;

    vec3 result = vec3(0.0);
    result += texture(tex, vec2(tp0.x,  tp0.y)).rgb  * (w0.x  * w0.y);
    result += texture(tex, vec2(tp12.x, tp0.y)).rgb  * (w12.x * w0.y);
    result += texture(tex, vec2(tp3.x,  tp0.y)).rgb  * (w3.x  * w0.y);
    result += texture(tex, vec2(tp0.x,  tp12.y)).rgb * (w0.x  * w12.y);
    result += texture(tex, vec2(tp12.x, tp12.y)).rgb * (w12.x * w12.y);
    result += texture(tex, vec2(tp3.x,  tp12.y)).rgb * (w3.x  * w12.y);
    result += texture(tex, vec2(tp0.x,  tp3.y)).rgb  * (w0.x  * w3.y);
    result += texture(tex, vec2(tp12.x, tp3.y)).rgb  * (w12.x * w3.y);
    result += texture(tex, vec2(tp3.x,  tp3.y)).rgb  * (w3.x  * w3.y);

    // Local 2×2 box for overshoot clamp (kills the bicubic ringing
    // halos that show up near hard edges).
    vec2 inv = 1.0 / src_size;
    vec3 mn = vec3(1e10), mx = vec3(0.0);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            vec3 c = texture(tex, (tex_pos_1 + vec2(i, j)) * inv).rgb;
            mn = min(mn, c);
            mx = max(mx, c);
        }
    }
    return clamp(result, mn, mx);
}

vec3 view_ray(vec2 uv) {
    vec4 ndc_far  = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 ndc_near = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    vec4 world_far_h  = pc.inv_view_proj * ndc_far;
    vec4 world_near_h = pc.inv_view_proj * ndc_near;
    vec3 world_far    = world_far_h.xyz  / world_far_h.w;
    vec3 world_near   = world_near_h.xyz / world_near_h.w;
    return normalize(world_far - world_near);
}

void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    // texelFetch indexes by integer texel coords, not UVs. When the
    // history image is at render_extent (LR) and compose runs at
    // swapchain (HR), texelFetch(history_color, ip) reads beyond the
    // LR texture for HR pixels past the LR size — producing the
    // "image is LR-pixel-sized in the top-left of the HR window"
    // artifact at non-100% render scale. Use bilinear sampling at
    // the proper HR UV instead — this also upscales the LR data to
    // fill the full window.
    vec2 sample_uv = (gl_FragCoord.xy + 0.5) * pc.viewport.zw;
    // FSR1 EASU upscale for the colour history — edge-adaptive 12-tap
    // reconstruction that's much sharper than bilinear or Catmull-Rom
    // at non-100 % render scale, with no ringing halos.
    vec2 src_size = vec2(textureSize(history_color, 0));
    vec3 hdr   = fsr1_easu(history_color, sample_uv, src_size);
    // Edge chromatic aberration. Re-samples R and B from history_color
    // at a small radial offset away from the centre — green keeps the
    // EASU-sharpened reconstruction, R/B fringes outward at the corners.
    // ramp t ≈ 0 through centre 30% of frame → ≈1.5 px split at corners.
    // Branchless: the if guard around this previously triggered for ~84%
    // of pixels (the inner cut-off was at length(ndc)=0.4), so the
    // divergence cost more than the saved fetches. Two extra texture
    // fetches per pixel; offsets are ≈zero near the centre so R/B end up
    // visually identical to the EASU values there anyway.
    {
        vec2 inv_res = pc.viewport.zw;
        vec2 ndc     = (gl_FragCoord.xy + 0.5) * inv_res * 2.0 - 1.0;
        float ca_t   = smoothstep(0.55, 1.05, length(ndc));
        vec2 ca_off  = ndc * ca_t * inv_res * 1.5;
        hdr.r = texture(history_color, sample_uv + ca_off).r;
        hdr.b = texture(history_color, sample_uv - ca_off).b;
    }
    float depth = texture(history_depth, sample_uv).r;

    // Background pixel (no geometry hit) — paint procedural sky.
    if (depth >= 0.99999) {
        vec2 uv = (gl_FragCoord.xy + 0.5) * pc.viewport.zw;
        vec3 dir = view_ray(uv);
        hdr = sample_sky(dir);
    }

    // FSR1 RCAS — robust contrast-adaptive sharpening on the EASU-
    // upscaled colour. Sharpens texture detail without ringing on
    // edges. Sharpness slider [0..2] is clamped to RCAS's [0..1]
    // range with a 0.5 mid-point so the existing UI defaults still
    // make sense.
    if (depth < 0.99999 && pc.sharpen_params.x > 0.0) {
        // 4 cardinals at one HR pixel offset (post-upscale image is
        // at HR; we sample neighbouring HR pixels via the upscale
        // function so the sharpener sees full-res signal).
        vec2 hr_step = pc.viewport.zw;
        vec3 sN = fsr1_easu(history_color, sample_uv + vec2(0.0,  hr_step.y), src_size);
        vec3 sS = fsr1_easu(history_color, sample_uv + vec2(0.0, -hr_step.y), src_size);
        vec3 sE = fsr1_easu(history_color, sample_uv + vec2( hr_step.x, 0.0), src_size);
        vec3 sW = fsr1_easu(history_color, sample_uv + vec2(-hr_step.x, 0.0), src_size);
        float sharpness = clamp(pc.sharpen_params.x * 0.5, 0.0, 1.0);
        hdr = fsr1_rcas(hdr, sN, sS, sE, sW, sharpness);
        hdr = max(hdr, vec3(0.0));   // RCAS guard against negative
    }

    // Screen-space sun shafts (crepuscular rays). 24 fixed taps along
    // the line from this pixel toward the sun's screen position; each
    // tap contributes the sun colour iff the tap is a sky pixel (depth
    // sentinel ≥ 0.99999). Geometry occluders along the line silhouette
    // the rays — the classic god-ray look. Cheap: 24 sky-gated depth
    // fetches, no extra colour samples. Gated on the user's sun_shaft
    // intensity slider (pc.sun_dir.w) and the sun being in front of the
    // camera (pc.sun_screen.z > 0.5 — same flag the lens-flare uses).
    // Runs BEFORE bloom so the brightened shaft pixels pick up bloom's
    // soft halo for free.
    // Gate on current pixel NOT being sky — without this the shaft
    // accumulation piles full sun-color on top of the sun itself when
    // the user looks into it (every tap is sky, occ saturates to 1).
    // Shafts are a "light streaks through silhouettes onto geometry"
    // effect; the sky already has its own sun disc + bloom + halo.
    if (pc.sun_dir.w > 0.001 && pc.sun_screen.z > 0.5 && depth < 0.99999) {
        vec2  uv_here = (gl_FragCoord.xy + 0.5) * pc.viewport.zw;
        vec2  uv_sun  = pc.sun_screen.xy;
        vec2  d_total = uv_here - uv_sun;
        const int   kN     = 24;
        const float kStep  = 1.0 / float(kN);
        const float kDecay = 0.94;
        vec2 d_step = d_total * kStep;
        // Per-pixel sub-step jitter. Without it the 24 fixed tap
        // positions visibly band into squares/stripes whenever the line
        // crosses small sky openings (castle windows): every pixel's
        // taps land in the same relative slots so the discretisation
        // shows. The hash offsets the starting position by ∈ [0, kStep),
        // turning the band pattern into per-pixel noise that TAA's
        // history blend integrates into smooth continuous rays.
        float jitter = fract(sin(dot(gl_FragCoord.xy,
                                      vec2(12.9898, 78.233))) * 43758.5453);
        vec2 uv_t   = uv_here - d_step * jitter;
        float w = 1.0;
        float occ = 0.0;
        for (int i = 0; i < kN; ++i) {
            uv_t -= d_step;
            float dz = texture(history_depth, uv_t).r;
            if (dz >= 0.99999) occ += w;
            w *= kDecay;
        }
        // Σ w_i = (1 − decay^N) / (1 − decay); divide so shaft ∈ [0,1].
        const float kNorm  = (1.0 - 0.2272278) / (1.0 - kDecay); // pow(0.94,24)≈0.227
        float shaft        = occ / kNorm;
        // Radial falloff away from the sun. Without it shafts wash the
        // whole frame uniformly whenever the sun is visible at all.
        float r            = length(d_total);
        float radial_fade  = exp(-r * 1.8);
        vec3 shaft_col     = pc.sun_color.rgb * pc.sun_color.a *
                             shaft * radial_fade * pc.sun_dir.w;
        hdr += shaft_col;
    }

    // Bloom and lens flare apply on every pixel — including sky. Bloom is
    // one texture() fetch (no more 24-tap spiral); lens-flare early-outs on
    // "sun behind camera" so most non-flare frames exit cheap. Gating on
    // `!is_sky` here used to save fetches on the old shader, but it also
    // hard-cut the lantern's bloom at sky edges — looked like bloom was
    // drawn on the wall and stopped at the sky.
    if (pc.bloom_params.w > 0.5) {
        // No `+ 0.5`: gl_FragCoord.xy is already the pixel center (X.5, Y.5),
        // so `gl_FragCoord.xy * inv_size` is the pixel-center UV. The earlier
        // `+0.5` introduced a half-pixel offset that made bloom appear shifted
        // to the bottom-right of bright sources (= "objects' bloom looks
        // off-set top-left"). Linear sampling on the half-res bloom mip
        // upsamples cleanly with the correct UV.
        vec2 bloom_uv = gl_FragCoord.xy * pc.viewport.zw;
        vec3 bloom = texture(u_bloom, bloom_uv).rgb;
        // Bloom uses additive blending across 7 mip-up passes — each
        // pass adds another mip's contribution to the destination, so
        // the largest mip reads can sum to ~6 × (mip_count) per channel
        // (with the bloom-down cap at 6). Compose's `hdr += bloom *
        // strength` would then add ~8 per channel screen-wide on top
        // of the (TAA-bounded ≤16) hdr — ACES rolls that off to a
        // sky-tinted near-white wash that alternates with the normal
        // frame as RT GI samples land bright vs dim. Cap the bloom
        // contribution per pixel so the additive chain has a hard
        // ceiling regardless of how the chain accumulated.
        bloom = min(bloom, vec3(4.0));
        hdr += bloom * pc.bloom_params.x;
    }
    // Lens flare is gated at call-site too: even though lens_flare's
    // first line returns early when disabled, hoisting the test out of
    // the function avoids any driver inlining quirks and makes the
    // disabled-path absolutely free (no function-call setup, no register
    // pressure for the unused locals).
    if (pc.flare_params2.z >= 0.5 && pc.sun_screen.z >= 0.5) {
        vec2 flare_uv = gl_FragCoord.xy * pc.viewport.zw;
        hdr += lens_flare(flare_uv);
    }

    // Auto-exposure (eye adaptation). The smallest bloom mip is
    // textureLod-sampled at uv=(0.5, 0.5) — that pixel is roughly the
    // average of the bright-source signal across the screen, which
    // correlates with overall scene brightness (interiors low, open
    // sky high). Compute exposure = pow(target / avg, k) and scale hdr
    // before tonemap. Bloom and lens flare ride along since they're
    // already added; bright-outdoor bloom thus blooms HARDER once the
    // exposure has lifted to compensate for indoor darkness.
    //
    // sharpen_params.y carries auto_exposure_strength (0 = off, 1 =
    // full adaptation). 0.5 default is enough to get the "step out
    // into sun and squint" effect without flattening contrast.
    if (pc.sharpen_params.y > 0.0) {
        // Sample the smallest mip for a coarse scene-luminance estimate.
        // kBloomMips = 5 in the engine, so the smallest mip is index 4.
        // The previous 6.0 value was stale (from when kBloomMips = 7);
        // it relied on the sampler's max-LOD clamp, which works but
        // requires the all-mips view bound at this binding.
        float scene_avg = max(0.005,
            lum(textureLod(u_bloom, vec2(0.5), 4.0).rgb));
        const float kTarget = 0.18;        // mid-grey target
        float ratio = kTarget / scene_avg;
        ratio = clamp(ratio, 0.4, 4.0);    // bound the auto-correction
        // strength = 0 -> ratio collapses to 1 (no scaling)
        // strength = 1 -> full ratio applied
        float exposure = pow(ratio, pc.sharpen_params.y * 0.5);
        // Halved strength + tighter clamp [0.7, 1.6] above keeps the
        // exposure from oscillating frame-to-frame when a bright cube
        // briefly enters the bloom-mip's average — that swing was the
        // visible "surface flicker" between bright and dark frames.
        exposure = clamp(exposure, 0.7, 1.6);
        hdr *= exposure;
    }

    // NaN / inf guard — any single corrupt pixel before tonemap (e.g. a
    // path-traced bounce that happened to land on a sun-disk halo and
    // compounded to inf) propagates through bloom, eventually painting
    // the whole frame and triggering surface flicker. Clamp finite
    // values defensively.
    hdr = clamp(hdr, vec3(0.0), vec3(64.0));

    vec3 mapped = aces_fitted(hdr);
    // Tone controls applied post-tonemap, pre-sRGB. Order: brightness
    // (linear scale) -> contrast (pivot around 0.5) -> gamma curve.
    float brightness = max(pc.sharpen_params.w, 0.001);
    float contrast   = max(pc.sharpen_params.z, 0.001);
    float gamma_v    = max(pc.sun_screen.w,    0.001);
    mapped *= brightness;
    mapped = clamp((mapped - 0.5) * contrast + 0.5, 0.0, 1.0);
    mapped = pow(mapped, vec3(1.0 / gamma_v));

    // Brush size indicator. Drawn in tonemapped LDR (post-aces) so the
    // ring colour reads the same in any lighting. Reconstruct this
    // pixel's world position from its depth value, compare its world
    // XZ distance to the brush radius, and tint pixels within a thin
    // band on either side of the radius. Skipped for background
    // pixels (depth ≈ 1) so the ring doesn't draw on the sky.
    if (pc.brush_indicator.w > 0.5 && depth < 0.99999) {
        vec2 uv = (gl_FragCoord.xy + 0.5) * pc.viewport.zw;
        // NDC for this pixel. Vulkan: gl_FragCoord.y is top-down, so
        // ndc.y = 1 - 2*uv.y. ndc.z uses depth directly (Vulkan z range
        // is [0,1], not [-1,1] like GL).
        vec4 ndc = vec4(uv.x * 2.0 - 1.0,
                        1.0 - uv.y * 2.0,
                        depth, 1.0);
        vec4 wp_h = pc.inv_view_proj * ndc;
        vec3 world = wp_h.xyz / wp_h.w;
        float d = length(world.xz - pc.brush_indicator.xy);
        float radius   = pc.brush_indicator.z;
        float ring_w   = max(0.04, radius * 0.025);
        // Smooth triangular band around the radius — fully tinted at
        // |d - radius| = 0, falling off to 0 at |d - radius| = ring_w.
        float band = 1.0 - clamp(abs(d - radius) / ring_w, 0.0, 1.0);
        // Soft inner fill so the brush footprint reads at a glance —
        // 8% strength inside the ring; ring itself is the bright edge.
        float fill = (d < radius) ? 0.08 : 0.0;
        vec3 ring_color = vec3(0.2, 0.95, 1.0);   // cyan, reads on any terrain
        mapped = mix(mapped, ring_color, clamp(band * 0.85 + fill, 0.0, 1.0));
    }

    // Subtle cinematic vignette. Radial darkening from screen centre —
    // ~8% drop at the corners, zero through the centre 60% of the frame.
    // Cheap (1 length, 1 smoothstep) and always on; deliberately mild so
    // it reads as atmosphere not as a black border.
    {
        vec2 ndc = (gl_FragCoord.xy + 0.5) * pc.viewport.zw * 2.0 - 1.0;
        float r  = length(ndc);
        float vig = 1.0 - 0.18 * smoothstep(0.6, 1.4, r);
        mapped *= vig;
    }

    outColor = vec4(to_srgb(mapped), 1.0);
}
