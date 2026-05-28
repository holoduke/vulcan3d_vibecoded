#version 460

// SSDM Phase 3: barycenter refinement (Lobel 2008, sec 2.3).
//
// Fullscreen-triangle fragment shader, one pass per pyramid mip level.
// Viewport / render-target size is the current mip's extent. For each
// output pixel:
//   1. If top level: previous barycenter = 0 (= raster position).
//      Else:         previous barycenter = pyrB_prev sampled at this UV.
//   2. Sample pyramid A at 4 corners ±0.5 texels around (uv + prev_bary_uv)
//      at the current mip level.
//   3. Output = average of the 4 sample values (the refined barycenter).
//
// Stored convention: displacement vector is in level-0 PIXEL units
// (matches what ssdm_disp.frag emits; mip averaging preserves the unit).

layout(location = 0) out vec2 outBary;

layout(set = 0, binding = 0) uniform sampler2D pyrA;       // all mips
layout(set = 0, binding = 1) uniform sampler2D pyrB_prev;  // single coarser mip

layout(push_constant) uniform PC {
    uvec2 base_extent;       // pyramid level-0 extent (pixels)
    uvec2 level_extent;      // current level's extent (pixels)
    uint  level;             // current mip level being written
    uint  is_top_level;      // 1 = no prev barycenter
    uint  _pad0;
    uint  _pad1;
} pc;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(pc.level_extent);

    // Previous barycenter sampled at the coarser pyrB mip (level+1).
    // Both samplers are all-mip views; the per-iteration mip selection
    // is done via textureLod() rather than a per-mip view binding.
    vec2 prev_bary_px = (pc.is_top_level != 0u)
                           ? vec2(0.0)
                           : textureLod(pyrB_prev, uv,
                                        float(pc.level + 1u)).xy;
    vec2 prev_bary_uv = prev_bary_px / vec2(pc.base_extent);

    vec2 center_uv = uv + prev_bary_uv;
    vec2 texel_uv  = 1.0 / vec2(pc.level_extent);
    float L = float(pc.level);
    vec2 smp00 = textureLod(pyrA, center_uv + vec2(-0.5, -0.5) * texel_uv, L).xy;
    vec2 smp10 = textureLod(pyrA, center_uv + vec2(+0.5, -0.5) * texel_uv, L).xy;
    vec2 smp01 = textureLod(pyrA, center_uv + vec2(-0.5, +0.5) * texel_uv, L).xy;
    vec2 smp11 = textureLod(pyrA, center_uv + vec2(+0.5, +0.5) * texel_uv, L).xy;
    outBary = 0.25 * (smp00 + smp10 + smp01 + smp11);
}
