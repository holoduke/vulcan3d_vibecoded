#version 460

// Bloom downsample, Karis 13-tap pattern from Call of Duty Advanced Warfare:
// 4 corner box-2x2 averages + 1 center box-2x2, weighted to remove fireflies
// when sampling an HDR source. The first mip uses Karis-average prefiltering
// (clamp to inverse-luminance) on each tap to suppress aliased single-pixel
// hot spots; subsequent mips use straight downsample.

layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec4 src_extent;   // x: w, y: h, z: 1/w, w: 1/h  (source size)
    vec4 dst_extent;   // dest size (we render at this resolution)
    vec4 params;       // x: 1.0 if first-pass (apply Karis prefilter), else 0
                       // y: bloom threshold (only used on first pass)
} pc;

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Soft luminance threshold — sub-threshold pixels return ~0, above-threshold
// fade smoothly. Same shape as the earlier inline-spiral prefilter() so the
// bloom mip chain reproduces the user's bloom_threshold setting.
vec3 prefilter(vec3 c, float threshold) {
    float l = lum(c);
    float soft = max(0.0, l - threshold + 0.5);
    soft = soft * soft / (1.0 + 2.0 * 0.5);
    float w = max(soft, l - threshold);
    return c * (w / max(l, 1e-4));
}

// Karis "partial average" — replaces simple average with one weighted by
// inverse-luminance to clip fireflies before they spread through the chain.
vec3 karis_avg(vec3 a, vec3 b, vec3 c, vec3 d) {
    float wa = 1.0 / (1.0 + lum(a));
    float wb = 1.0 / (1.0 + lum(b));
    float wc = 1.0 / (1.0 + lum(c));
    float wd = 1.0 / (1.0 + lum(d));
    return (a * wa + b * wb + c * wc + d * wd) /
           max(wa + wb + wc + wd, 1e-4);
}

void main() {
    vec2 uv = (gl_FragCoord.xy + 0.5) * pc.dst_extent.zw;
    vec2 t  = pc.src_extent.zw;  // texel size in source

    // 13-tap "box of boxes" pattern (CoD: AW).
    //  o . o . o
    //  . X . X .
    //  o . o . o
    //  . X . X .
    //  o . o . o
    //
    // NaN/inf guard: clamp each tap at read. lum() of NaN is NaN and
    // would propagate through Karis (1/(1+NaN) = NaN) into the entire
    // bloom mip chain, then auto-exposure (which samples the smallest
    // mip) would oscillate frame-to-frame producing the visible "two
    // colours alternating" surface flicker.
    const vec3 kHdrCap = vec3(32.0);
    vec3 a = clamp(texture(src, uv + t * vec2(-2,  2)).rgb, vec3(0.0), kHdrCap);
    vec3 b = clamp(texture(src, uv + t * vec2( 0,  2)).rgb, vec3(0.0), kHdrCap);
    vec3 c = clamp(texture(src, uv + t * vec2( 2,  2)).rgb, vec3(0.0), kHdrCap);
    vec3 d = clamp(texture(src, uv + t * vec2(-2,  0)).rgb, vec3(0.0), kHdrCap);
    vec3 e = clamp(texture(src, uv + t * vec2( 0,  0)).rgb, vec3(0.0), kHdrCap);
    vec3 f = clamp(texture(src, uv + t * vec2( 2,  0)).rgb, vec3(0.0), kHdrCap);
    vec3 g = clamp(texture(src, uv + t * vec2(-2, -2)).rgb, vec3(0.0), kHdrCap);
    vec3 h = clamp(texture(src, uv + t * vec2( 0, -2)).rgb, vec3(0.0), kHdrCap);
    vec3 i = clamp(texture(src, uv + t * vec2( 2, -2)).rgb, vec3(0.0), kHdrCap);
    vec3 j = clamp(texture(src, uv + t * vec2(-1,  1)).rgb, vec3(0.0), kHdrCap);
    vec3 k = clamp(texture(src, uv + t * vec2( 1,  1)).rgb, vec3(0.0), kHdrCap);
    vec3 l = clamp(texture(src, uv + t * vec2(-1, -1)).rgb, vec3(0.0), kHdrCap);
    vec3 m = clamp(texture(src, uv + t * vec2( 1, -1)).rgb, vec3(0.0), kHdrCap);

    vec3 result;
    if (pc.params.x > 0.5) {
        // First downsample from the HDR scene image — apply soft threshold
        // prefilter to each tap so sub-threshold pixels don't seed the chain,
        // then Karis-average to suppress single-pixel fireflies.
        float th = pc.params.y;
        a = prefilter(a, th); b = prefilter(b, th); c = prefilter(c, th);
        d = prefilter(d, th); e = prefilter(e, th); f = prefilter(f, th);
        g = prefilter(g, th); h = prefilter(h, th); i = prefilter(i, th);
        j = prefilter(j, th); k = prefilter(k, th); l = prefilter(l, th);
        m = prefilter(m, th);
        // Combine the 13 taps into 5 partial 2x2 averages, each Karis-clipped.
        vec3 c0 = karis_avg(a, b, d, e);
        vec3 c1 = karis_avg(b, c, e, f);
        vec3 c2 = karis_avg(d, e, g, h);
        vec3 c3 = karis_avg(e, f, h, i);
        vec3 c4 = karis_avg(j, k, l, m);
        // Weights from the original CoD slides:
        result = c0 * 0.125 + c1 * 0.125 + c2 * 0.125 + c3 * 0.125 + c4 * 0.5;
    } else {
        result = e * 0.125
               + (a + c + g + i) * 0.03125
               + (b + d + f + h) * 0.0625
               + (j + k + l + m) * 0.125;
    }
    outColor = vec4(result, 1.0);
}
