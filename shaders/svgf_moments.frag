#version 460

// SVGF variance-moments temporal accumulator (4a-deep).
//
// At each pixel:
//   1. Compute current scene luminance from the post-world scene_color.
//   2. Reproject the previous frame's moments via motion_vec.
//   3. EMA-blend the previous moments into the current sample so the
//      per-pixel running estimates of (mean luminance, mean luminance^2)
//      track ~10 frames of history. Disocclusion or off-screen reprojection
//      resets to the fresh sample (alpha=1).
//   4. Write the new (mu, mu2) pair into the moments storage image.
//
// The 1st and 2nd luminance moments give:
//      variance = max(0, mu2 - mu*mu)
// which svgf_atrous.frag reads to size its luminance edge-stop weight
// per pixel -- stable surfaces (low variance) clamp tight, noisy GI
// pixels (high variance) accept bigger spatial neighbours.

// Dummy colour output -- the real work goes through imageStore on the
// storage image at binding 4. Vulkan dynamic-rendering requires a
// colour attachment at vkCmdBeginRendering, so we emit black into
// svgf_atrous_image_[0] which the first a-trous pass overwrites
// before TAA could ever read it.
layout(location = 0) out vec4 outDummy;

layout(set = 0, binding = 0) uniform sampler2D u_scene_color;
layout(set = 0, binding = 1) uniform sampler2D u_depth;          // unused this pass
layout(set = 0, binding = 2) uniform sampler2D u_motion;
layout(set = 0, binding = 3) uniform sampler2D u_prev_moments;
layout(set = 0, binding = 4, rg32f) uniform image2D u_cur_moments;

layout(set = 0, binding = 5) uniform MomentsUBO {
    vec4 viewport;   // x: w, y: h, z: 1/w, w: 1/h
    vec4 params;     // x: alpha_min, y: alpha_disocclusion,
                     //  z: history_valid (0/1), w: unused
} mom;

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec2  uv = (gl_FragCoord.xy + 0.5) * mom.viewport.zw;

    // Current frame luminance moments at this pixel.
    vec3  cur_rgb = texelFetch(u_scene_color, ip, 0).rgb;
    // Per-tap HDR cap matches taa.frag's variance_clip: a single rogue
    // sun-disk-halo path-traced bounce can produce inf/NaN, which a
    // single squared term would lock into the EMA forever and never
    // recover from. Clamp to 16 per channel.
    cur_rgb = clamp(cur_rgb, vec3(0.0), vec3(16.0));
    float cur_l  = lum(cur_rgb);
    vec2  cur_m  = vec2(cur_l, cur_l * cur_l);

    // Default: fresh sample (no history blend). This fires on the very
    // first frame (mom.params.z = 0 because prev_view_proj_valid_ is
    // false) AND on disocclusion (off-screen reprojection or no prev
    // sample).
    float alpha = mom.params.y;       // alpha_disocclusion (default 1.0)
    vec2  prev_m = vec2(0.0);

    if (mom.params.z > 0.5) {
        // Motion-vec reproject. The cube.frag-side motion vector is
        // cur_uv - prev_uv -- subtract to land at the prev_uv. Pixels
        // outside [0,1] fall through to the disocclusion path.
        vec2 motion = texelFetch(u_motion, ip, 0).rg;
        vec2 prev_uv = uv - motion;
        if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
            prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
            // Sample with linear filtering -- the prev moments are a
            // smooth running estimate, bilinear interpolation is fine.
            prev_m = texture(u_prev_moments, prev_uv).rg;
            // History is valid; pick the small alpha for steady-state
            // pixels and keep alpha_disocclusion as a ceiling when the
            // reprojected mean is wildly different from the new sample
            // (a missed motion vector, e.g. dynamic-light flicker).
            // Difference threshold: 4x the current sqrt(variance)
            // estimate (a fairly loose 4-sigma gate -- tighter would
            // false-reject high-frequency lighting).
            float prev_mean = prev_m.x;
            float prev_var  = max(0.0, prev_m.y - prev_mean * prev_mean);
            float prev_sd   = sqrt(prev_var);
            float delta     = abs(cur_l - prev_mean);
            float reject    = step(prev_sd * 4.0 + 0.05, delta);
            alpha = mix(mom.params.x, mom.params.y, reject);
        }
    }

    vec2 new_m = mix(prev_m, cur_m, alpha);
    imageStore(u_cur_moments, ip, vec4(new_m, 0.0, 0.0));

    // Dummy colour output. The pipeline has CLEAR loadOp on this
    // attachment so the prior contents don't matter; we emit black so
    // any debug visualisation of svgf_atrous_image_[0] before the
    // next pass overwrites it shows obvious wipe.
    outDummy = vec4(0.0);
}
