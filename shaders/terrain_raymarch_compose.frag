#version 460

// Bilinear upscale + depth-aware composite for the half-res procedural
// raymarched terrain. Runs as a fullscreen-tri AT THE START of the main
// world pass (before any cube/castle/dyn-prop draw) so the rasterised
// foreground geometry can layer on top via the standard depth test.
//
//   1. sample u_lr_color, u_lr_motion, u_lr_depth at the current
//      fragment's UV
//   2. write color and motion as outputs
//   3. write gl_FragDepth = upscaled depth so the existing
//      depth_compare = LESS_OR_EQUAL on this pipeline only commits the
//      raymarch where it's closer than the cube/castle depth that the
//      depth pre-pass already wrote into scene_depth
//   4. discard sky pixels (lr_depth ≈ 1.0 = far) so the cleared sky
//      colour shows through unchanged

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

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
} scene;

// Push constants — same layout as cube.frag's PushConstants. We use
//   pc.color.x = sharpen strength (0 = off, 1 = aggressive)
//   pc.color.y = LR width  (texels)
//   pc.color.z = LR height (texels)
// The remaining fields are ignored.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

// Half-res raymarch outputs from the previous pass.
layout(set = 0, binding = 9)  uniform sampler2D u_lr_color;
layout(set = 0, binding = 10) uniform sampler2D u_lr_motion;
layout(set = 0, binding = 11) uniform sampler2D u_lr_depth;

// CAS-style adaptive 4-neighbour sharpen.
//   1. sample center + N/E/S/W LR texels
//   2. compute local min/max — `mn` and `mx` per channel
//   3. derive sharp amount from how close `center` is to the local
//      midpoint vs the extrema (CAS contrast adaptive coefficient)
//   4. unsharp mask: `out = c + amount * (c - blur)` with blur =
//      0.25 * (n + e + s + w)
//
// Cheap (5 texture taps + a handful of ALU) and visually close to
// AMD's real CAS for a single-pass sharpen. Skips the negative-lobe
// kernel CAS uses because we run AFTER bilinear upscale, not as a
// general-purpose filter.
vec3 casSharpen(vec3 c, vec3 n, vec3 e, vec3 s, vec3 w, float strength) {
    vec3 mn = min(min(min(n, e), min(s, w)), c);
    vec3 mx = max(max(max(n, e), max(s, w)), c);
    // Distance from the local extrema as a 0..1 ratio. clamps the
    // boost in flat regions (ratio→1) and amplifies it on real edges
    // (ratio→0).
    vec3 ratio = (min(mn, 1.0 - mx)) / max(mx, vec3(0.001));
    float amount = strength * 0.5 * clamp(min(ratio.r, min(ratio.g, ratio.b)), 0.0, 1.0);
    vec3 blur = 0.25 * (n + e + s + w);
    return c + amount * (c - blur);
}

void main() {
    // Map full-res fragment to a UV in [0, 1] and sample the LR
    // textures. The LR sampler is LINEAR, so this is a free bilinear
    // upscale — the sky-sentinel discard below avoids dragging sky
    // colour into terrain edges.
    vec2 uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
    float lr_depth = texture(u_lr_depth, uv).r;

    // The raymarch shader leaves the LR depth attachment cleared to 1.0
    // for sky pixels (raymarch missed). Discarding those preserves the
    // existing sky clear in scene_color and the cube prepass depth.
    if (lr_depth >= 0.9999) discard;

    // Phase 4 — depth-aware bilateral 4-tap upsample. Standard
    // bilinear leaks across depth discontinuities (e.g. cube/castle
    // silhouettes against distant terrain) producing visible halos
    // at low render scales. Weighting the neighbour samples by depth
    // similarity preserves silhouettes while staying free in flat
    // regions where all weights ≈ 1.
    vec2 lr_texel = vec2(1.0) / max(pc.color.yz, vec2(1.0));
    vec2 uv_n = uv + vec2(0.0, -lr_texel.y);
    vec2 uv_s = uv + vec2(0.0,  lr_texel.y);
    vec2 uv_w = uv + vec2(-lr_texel.x, 0.0);
    vec2 uv_e = uv + vec2( lr_texel.x, 0.0);
    vec3 c0 = texture(u_lr_color, uv).rgb;
    vec3 cn = texture(u_lr_color, uv_n).rgb;
    vec3 cs = texture(u_lr_color, uv_s).rgb;
    vec3 cw = texture(u_lr_color, uv_w).rgb;
    vec3 ce = texture(u_lr_color, uv_e).rgb;
    float dn = texture(u_lr_depth, uv_n).r;
    float ds = texture(u_lr_depth, uv_s).r;
    float dw = texture(u_lr_depth, uv_w).r;
    float de = texture(u_lr_depth, uv_e).r;
    // Sharpness 8: kernel falls off when |Δdepth| > ~1/8. Tuned so
    // sub-pixel depth jitter doesn't kill the bilinear blend.
    const float kSharp = 8.0;
    float wn = exp(-kSharp * abs(dn - lr_depth));
    float ws = exp(-kSharp * abs(ds - lr_depth));
    float ww = exp(-kSharp * abs(dw - lr_depth));
    float we = exp(-kSharp * abs(de - lr_depth));
    float wsum = 1.0 + wn + ws + ww + we;
    vec3 c_blur = (c0 + wn*cn + ws*cs + ww*cw + we*ce) / wsum;
    vec3 c = c_blur;

    float strength = pc.color.x;
    if (strength > 0.0 && pc.color.y > 0.5 && pc.color.z > 0.5) {
        // Sharpen uses the same 4 weighted neighbours we already have.
        c = casSharpen(c0, cn, ce, cs, cw, strength);
    }
    outColor     = vec4(c, 1.0);
    outMotion    = texture(u_lr_motion, uv).rg;
    gl_FragDepth = lr_depth;
}
