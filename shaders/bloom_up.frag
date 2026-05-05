#version 460

// Bloom upsample, 9-tap "tent" filter, additively blended into the larger
// mip via a fragment-shader output and pipeline blend (additive).

layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D src;  // smaller mip

layout(push_constant) uniform PC {
    vec4 src_extent;
    vec4 dst_extent;
    vec4 params;       // x: filter_radius_in_dst_texels (default 1.0)
} pc;

void main() {
    vec2 uv = (gl_FragCoord.xy + 0.5) * pc.dst_extent.zw;
    vec2 t  = pc.src_extent.zw * pc.params.x;

    // 3x3 tent kernel (1, 2, 1 / 2, 4, 2 / 1, 2, 1) / 16:
    vec3 sum = vec3(0.0);
    sum += texture(src, uv + t * vec2(-1,  1)).rgb * 1.0;
    sum += texture(src, uv + t * vec2( 0,  1)).rgb * 2.0;
    sum += texture(src, uv + t * vec2( 1,  1)).rgb * 1.0;
    sum += texture(src, uv + t * vec2(-1,  0)).rgb * 2.0;
    sum += texture(src, uv + t * vec2( 0,  0)).rgb * 4.0;
    sum += texture(src, uv + t * vec2( 1,  0)).rgb * 2.0;
    sum += texture(src, uv + t * vec2(-1, -1)).rgb * 1.0;
    sum += texture(src, uv + t * vec2( 0, -1)).rgb * 2.0;
    sum += texture(src, uv + t * vec2( 1, -1)).rgb * 1.0;
    outColor = vec4(sum * (1.0 / 16.0), 1.0);
}
