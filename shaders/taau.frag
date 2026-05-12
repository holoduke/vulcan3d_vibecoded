#version 460

// Temporal Anti-Aliasing Upsample (TAAU). Runs AFTER the standard TAA
// pass; consumes its LR history output and a previous-frame native-res
// history image, produces a new native-res history with motion-vector
// reprojection + neighborhood clamp. Equivalent in spirit to FSR2's
// reconstruction pass but ~200 LOC instead of an SDK.
//
// Inputs (LR = render_extent):
//   binding 0: cur_lr     — current frame's TAA output (history)
//   binding 1: prev_full  — previous frame's TAAU output (native)
//   binding 2: depth_lr   — current depth (for reproject sanity / disocclusion)
//   binding 3: ubo        — viewport sizes (LR + native) + params
//   binding 4: motion_lr  — current motion vectors (uv-space, current - prev)
//
// Algorithm:
//   1. For each native output pixel, compute the matching LR UV.
//   2. Bilinear-sample cur_lr at that UV — gives sub-pixel-anti-aliased input.
//   3. Compute neighborhood (mean, stddev) over a 3×3 LR window — defines
//      a luminance/color clamp box for the reprojected history.
//   4. Sample motion_lr at the LR UV; reproject prev_full by `uv - motion`
//      at NATIVE resolution.
//   5. Clamp prev to the LR neighborhood box → variance clip (rejects ghosts).
//   6. Blend: out = mix(cur_lr_bilinear, prev_clamped, alpha) with alpha
//      reduced by motion magnitude (no smear during fast camera turns).

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D cur_lr;
layout(set = 0, binding = 1) uniform sampler2D prev_full;
layout(set = 0, binding = 2) uniform sampler2D depth_lr;

layout(set = 0, binding = 3) uniform TaauUBO {
    vec4 lr_viewport;       // x:lrW, y:lrH, z:1/lrW, w:1/lrH
    vec4 full_viewport;     // x:fullW, y:fullH, z:1/fullW, w:1/fullH
    vec4 params;            // x:history_blend, y:history_valid (0/1), z:_, w:_
} taau;

layout(set = 0, binding = 4) uniform sampler2D motion_lr;

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// 3×3 neighborhood at LR — returns mean ± gamma·stddev for the clamp box.
// Same scheme as taa.frag's variance_clip; smaller window (9 taps vs 25)
// because we're already doing a bilinear sample for the center.
void neighborhood_box(ivec2 lr_ip, out vec3 mn, out vec3 mx, out vec3 center) {
    vec3 mean = vec3(0.0);
    vec3 mean_sq = vec3(0.0);
    const vec3 kHdrCap = vec3(16.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec3 c = texelFetch(cur_lr, lr_ip + ivec2(i, j), 0).rgb;
            c = clamp(c, vec3(0.0), kHdrCap);
            mean += c;
            mean_sq += c * c;
            if (i == 0 && j == 0) center = c;
        }
    }
    mean /= 9.0;
    mean_sq /= 9.0;
    vec3 stddev = sqrt(max(vec3(0.0), mean_sq - mean * mean));
    const float gamma = 1.25;
    mn = mean - stddev * gamma;
    mx = mean + stddev * gamma;
}

void main() {
    // native pixel → native UV in [0,1]
    vec2 full_uv = (gl_FragCoord.xy + 0.5) * taau.full_viewport.zw;
    // Same UV maps to LR coords (bilinear sample handles sub-pixel)
    vec2 lr_uv = full_uv;
    ivec2 lr_ip = ivec2(lr_uv * taau.lr_viewport.xy);
    lr_ip = clamp(lr_ip, ivec2(0),
                  ivec2(taau.lr_viewport.xy) - ivec2(1));

    // Bilinear current — this is the spatial upscale component. cur_lr
    // was already temporally-supersampled by the upstream TAA pass.
    vec3 cur = texture(cur_lr, lr_uv).rgb;
    cur = clamp(cur, vec3(0.0), vec3(16.0));

    // Build the variance-clip box from the LR neighborhood. We sample
    // texelFetch (not bilinear) here so the stddev reflects actual
    // signal variance, not interpolated values.
    vec3 mn, mx, nb_center;
    neighborhood_box(lr_ip, mn, mx, nb_center);

    if (taau.params.y < 0.5) {
        // History invalid (first frame, resize) → output current only.
        outColor = vec4(cur, 1.0);
        return;
    }

    // Reproject native history. motion is in LR uv-space (cur - prev),
    // but uv-space is resolution-independent so subtracting from full_uv
    // gives the same world location at native res.
    vec2 motion = texelFetch(motion_lr, lr_ip, 0).rg;
    vec2 prev_uv = full_uv - motion;
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 ||
        prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        outColor = vec4(cur, 1.0);
        return;
    }

    vec3 prev = texture(prev_full, prev_uv).rgb;
    prev = clamp(prev, vec3(0.0), vec3(16.0));
    vec3 prev_clamped = clamp(prev, mn, mx);

    // Velocity-adaptive blend, same shape as taa.frag.
    float motion_pix = length(motion * taau.lr_viewport.xy);
    float base_alpha = clamp(taau.params.x, 0.0, 0.98);
    float alpha = mix(base_alpha, 0.0, smoothstep(0.5, 8.0, motion_pix));

    outColor = vec4(mix(cur, prev_clamped, alpha), 1.0);
}
