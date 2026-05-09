#version 460

// Temporal AA + spatial à-trous denoise.
//
//   spatial_filtered = a-trous(current, depth)
//                       — 5x5 cross-bilateral, weighted by depth + luminance
//                         distance to the center pixel.
//   final            = spatial_filtered  ×  (1 - alpha)  +
//                      variance_clip(reprojected_history)  ×  alpha
//
// Variance clipping replaces the original min/max neighborhood clamp: history
// is clamped to mean ± k·stddev of the 3x3 current-frame box. Less aggressive
// than the min/max box, dramatically reduces edge flicker on sub-pixel-
// jittered static views, and tolerates the natural variance of HDR pixels
// without false rejection.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D current_color;
layout(set = 0, binding = 1) uniform sampler2D history_color;
layout(set = 0, binding = 2) uniform sampler2D current_depth;

layout(set = 0, binding = 3) uniform TaaUBO {
    vec4 viewport;          // x: w, y: h, z: 1/w, w: 1/h
    vec4 params;            // x: history_blend, y: depth_reject_threshold,
                            // z: history_valid (0/1), w: spatial_strength
} taa;

// Per-pixel screen-space motion vector written by cube.frag at location=1
// of the world pass. Encodes current_uv - prev_uv: dynamic surfaces get
// correct reprojection because cube.vert applied per-instance prev_model.
// Sky pixels (no fragment runs) keep the cleared (0, 0), which means
// "history at the same pixel" — fine because the sky is procedural and
// view-direction-driven by compose anyway.
layout(set = 0, binding = 4) uniform sampler2D motion_color;

const float kAtrousKernel[5] = float[](0.0625, 0.25, 0.375, 0.25, 0.0625);

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Edge-aware 5x5 spatial filter on the current frame.
vec3 atrous_filter(ivec2 ip) {
    vec3 center = texelFetch(current_color, ip, 0).rgb;
    float center_d = texelFetch(current_depth, ip, 0).r;
    float center_l = lum(center);

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            ivec2 p = ip + ivec2(i, j);
            vec3 c = texelFetch(current_color, p, 0).rgb;
            float d = texelFetch(current_depth, p, 0).r;
            float l = lum(c);

            float wk = kAtrousKernel[i + 2] * kAtrousKernel[j + 2];
            float wd = exp(-abs(d - center_d) * 200.0);
            float wl = exp(-abs(l - center_l) * 3.5);
            float w = wk * wd * wl;
            sum += c * w;
            wsum += w;
        }
    }
    return sum / max(wsum, 1e-4);
}

// Variance-based history clipping. Returns center color and the clamping
// box (mean ± gamma·stddev) of the 5x5 current-frame neighborhood.
//
// 5x5 (vs 3x3) gives a much more stable stddev estimate — fewer false
// rejections of valid history values that fall just outside the tight 3x3
// box but well inside the broader spatial luminance distribution. Cost is
// 25 texelFetches, which is nothing at 1080p on modern HW.
vec3 variance_clip(ivec2 ip, out vec3 mn_out, out vec3 mx_out) {
    vec3 mean = vec3(0.0);
    vec3 mean_sq = vec3(0.0);
    vec3 center = vec3(0.0);
    const int N = 25;
    // Per-tap NaN/inf guard. A single rogue inf in current_color (e.g.
    // a path-traced GI bounce that landed on a sun-disk halo and
    // compounded) would otherwise sail through mean/mean_sq, blow the
    // stddev to NaN, leave mn/mx undefined, and the variance clamp
    // would do nothing — so the inf/NaN locks into history. When the
    // camera stops moving, the corrupt history reprojects to itself
    // and the screen reads as a flat silhouette. Clamping each tap
    // before accumulating bounds the entire TAA pipeline.
    const vec3 kHdrCap = vec3(16.0);
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            vec3 c = texelFetch(current_color, ip + ivec2(i, j), 0).rgb;
            c = clamp(c, vec3(0.0), kHdrCap);
            mean += c;
            mean_sq += c * c;
            if (i == 0 && j == 0) center = c;
        }
    }
    mean /= float(N);
    mean_sq /= float(N);
    vec3 stddev = sqrt(max(vec3(0.0), mean_sq - mean * mean));
    // Tighter gamma (1.25 vs the 1.5σ Salvi/Karis default) — works because
    // the 5x5 window already gives a wider, more representative
    // distribution. Less false rejection ⇒ less edge flicker.
    const float gamma = 1.25;
    mn_out = mean - stddev * gamma;
    mx_out = mean + stddev * gamma;
    return center;
}

void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec2 uv = (gl_FragCoord.xy + 0.5) * taa.viewport.zw;

    vec3 mn, mx;
    vec3 cur_raw = variance_clip(ip, mn, mx);

    // Read motion early so spatial strength can be modulated by it.
    vec2 motion = texelFetch(motion_color, ip, 0).rg;
    float motion_pix = length(motion * taa.viewport.xy);

    // Spatial à-trous adds a constant low-pass blur to the input. When
    // still, that blur is masked by TAA's super-sampling accumulation;
    // when moving, the spatial filter is the largest remaining blur
    // contributor. Fade it out for moving pixels — at 8+ px/frame
    // (a typical mouse turn) we drop spatial to ~25% of slider strength.
    float spatial_falloff = mix(1.0, 0.25,
                                smoothstep(0.5, 8.0, motion_pix));
    float strength = clamp(taa.params.w, 0.0, 1.0) * spatial_falloff;
    vec3 cur = (strength > 0.0)
        ? mix(cur_raw, atrous_filter(ip), strength)
        : cur_raw;

    if (taa.params.z < 0.5) {
        outColor = vec4(cur, 1.0);
        return;
    }
    vec2 prev_uv = uv - motion;
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 ||
        prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        outColor = vec4(cur, 1.0);
        return;
    }

    vec3 hist = texture(history_color, prev_uv).rgb;
    // Match the variance_clip input cap so a corrupt history value can
    // never re-poison the next frame (variance_clip already caps the
    // current frame; if hist is also bounded the loop is safe).
    hist = clamp(hist, vec3(0.0), vec3(16.0));
    vec3 hist_clamped = clamp(hist, mn, mx);

    // Velocity-adaptive history blend. Constant high-feedback (0.95)
    // is great when the camera is still — TAA reduces RT noise. But
    // during motion the history must be bilinearly reprojected, which
    // adds a sub-pixel blur every frame; with 95% feedback that blur
    // locks in for ~20 frames. The "blurry while moving, sharp when
    // still" complaint.
    //
    // Solution: drop alpha smoothly toward ZERO as motion grows.
    // < 0.5 px/frame keeps full feedback (still pixels denoise
    // normally); 8+ px/frame uses NO history so bilinear reprojection
    // can't blur the current frame. The cost is a tiny pop of RT noise
    // when you start moving — much less objectionable than smear.
    float base_alpha = clamp(taa.params.x, 0.0, 0.98);
    float alpha = mix(base_alpha, 0.0, smoothstep(0.5, 8.0, motion_pix));
    vec3 blended = mix(cur, hist_clamped, alpha);

    outColor = vec4(blended, 1.0);
}
