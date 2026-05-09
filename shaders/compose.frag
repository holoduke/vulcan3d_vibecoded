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
    vec2 ab = axis_norm * aberration;
    float dr2 = dot(uv - center + ab, uv - center + ab);
    float dg2 = dot(uv - center,      uv - center);
    float db2 = dot(uv - center - ab, uv - center - ab);
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
    vis += float(texture(history_depth, c).r              >= kFar);
    vis += float(texture(history_depth, c + vec2( r, 0)).r >= kFar);
    vis += float(texture(history_depth, c + vec2(-r, 0)).r >= kFar);
    vis += float(texture(history_depth, c + vec2(0,  r)).r >= kFar);
    vis += float(texture(history_depth, c + vec2(0, -r)).r >= kFar);
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
    vec3 hdr = texelFetch(history_color, ip, 0).rgb;
    float depth = texelFetch(history_depth, ip, 0).r;

    // Background pixel (no geometry hit) — paint procedural sky.
    if (depth >= 0.99999) {
        vec2 uv = (gl_FragCoord.xy + 0.5) * pc.viewport.zw;
        vec3 dir = view_ray(uv);
        hdr = sample_sky(dir);
    }

    // Unsharp mask — counteracts taa.frag's 5×5 cross-bilateral à-trous so
    // textures, edges, and viewmodel detail come back crisp. Applied only on
    // surface pixels (sky's procedural gradient has no detail to recover).
    // 5 texelFetches: center + 4 cardinals → simple high-pass = (c - blur),
    // amplified by `sharpen_params.x`. Strength clamped to keep negative
    // overshoots from wrapping HDR negative.
    if (depth < 0.99999 && pc.sharpen_params.x > 0.0) {
        vec3 n = texelFetch(history_color, ip + ivec2( 0,  1), 0).rgb;
        vec3 s = texelFetch(history_color, ip + ivec2( 0, -1), 0).rgb;
        vec3 e = texelFetch(history_color, ip + ivec2( 1,  0), 0).rgb;
        vec3 w = texelFetch(history_color, ip + ivec2(-1,  0), 0).rgb;
        vec3 blur = (n + s + e + w) * 0.25;
        hdr = max(vec3(0.0), hdr + (hdr - blur) * pc.sharpen_params.x);
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
        hdr += bloom * pc.bloom_params.x;
    }
    vec2 flare_uv = gl_FragCoord.xy * pc.viewport.zw;
    hdr += lens_flare(flare_uv);

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
        // 6.0 = mip-6 (with kBloomMips=7 that's the smallest one).
        float scene_avg = max(0.005,
            lum(textureLod(u_bloom, vec2(0.5), 6.0).rgb));
        const float kTarget = 0.18;        // mid-grey target
        float ratio = kTarget / scene_avg;
        ratio = clamp(ratio, 0.4, 4.0);    // bound the auto-correction
        // strength = 0 -> ratio collapses to 1 (no scaling)
        // strength = 1 -> full ratio applied
        float exposure = pow(ratio, pc.sharpen_params.y);
        hdr *= exposure;
    }

    vec3 mapped = aces_fitted(hdr);
    // Tone controls applied post-tonemap, pre-sRGB. Order: brightness
    // (linear scale) -> contrast (pivot around 0.5) -> gamma curve.
    float brightness = max(pc.sharpen_params.w, 0.001);
    float contrast   = max(pc.sharpen_params.z, 0.001);
    float gamma_v    = max(pc.sun_screen.w,    0.001);
    mapped *= brightness;
    mapped = clamp((mapped - 0.5) * contrast + 0.5, 0.0, 1.0);
    mapped = pow(mapped, vec3(1.0 / gamma_v));
    outColor = vec4(to_srgb(mapped), 1.0);
}
