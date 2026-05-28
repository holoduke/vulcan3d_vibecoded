#version 460

// SSDM Phase 4: compose remap.
//
// For each output pixel: source UV = raster UV + pyrB[0] / base_extent.
// Sample scene_color at that source UV to produce the silhouette-extended
// displaced image. Phase 1-3 ensured pyrB[0] contains zero for pixels
// outside SPOM walls (Phase 1's shader writes 0 for non-SPOM, pyrA mip
// averaging propagates 0 in non-wall regions, refinement converges to
// 0 there); those pixels sample scene_color at their own UV and pass
// through unchanged.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneColor;
layout(set = 0, binding = 1) uniform sampler2D pyrB;       // all mips, sample level 0

layout(push_constant) uniform PC {
    uvec2 extent;       // render_extent_
    uvec2 _pad;
} pc;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(pc.extent);
    vec2 disp_px = textureLod(pyrB, uv, 0.0).xy;
    vec2 source_uv = uv + disp_px / vec2(pc.extent);
    source_uv = clamp(source_uv, vec2(0.0), vec2(1.0));
    outColor = textureLod(sceneColor, source_uv, 0.0);
}
