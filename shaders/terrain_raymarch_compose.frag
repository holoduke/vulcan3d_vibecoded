#version 460
#extension GL_GOOGLE_include_directive : require

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

#include "scene_ubo.glsl"

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

// CAS-style adaptive sharpen — same idea, takes 4 corner neighbours
// (from textureGather) instead of cardinals.
vec3 casSharpen(vec3 c, vec3 a, vec3 b, vec3 d, vec3 e, float strength) {
    vec3 mn = min(min(min(a, b), min(d, e)), c);
    vec3 mx = max(max(max(a, b), max(d, e)), c);
    vec3 ratio = (min(mn, 1.0 - mx)) / max(mx, vec3(0.001));
    float amount = strength * 0.5 * clamp(min(ratio.r, min(ratio.g, ratio.b)), 0.0, 1.0);
    vec3 blur = 0.25 * (a + b + d + e);
    return c + amount * (c - blur);
}

void main() {
    // Map full-res fragment to a UV in [0, 1] and sample the LR
    // textures. The LR sampler is LINEAR, so the centre tap is a
    // free bilinear upscale — the sky-sentinel discard below
    // avoids dragging sky colour into terrain edges.
    vec2 uv = gl_FragCoord.xy * scene.viewport.zw;
    float lr_depth = texture(u_lr_depth, uv).r;

    // Sky pixels: raymarch leaves depth = 1.0 on miss. Discard preserves
    // the existing sky clear in scene_color and the cube prepass depth.
    if (lr_depth >= 0.99999) discard;

    // Depth-aware bilateral 4-tap upsample, using textureGather: one
    // instruction returns the 4 LR-depth corners surrounding `uv`,
    // replacing the 4 separate `texture()` calls of the previous
    // implementation. Three more gathers fetch the same corners' RGB.
    // Total texture ops: 1 (centre depth) + 1 (depth gather) + 3 (RGB
    // gathers) + 1 (centre color) + 1 (motion) = 7, down from 11.
    //
    // Bilateral weight kernel: 1/(1 + α·d²) — a rational kernel with
    // visually identical edge-preserving falloff to the previous
    // exp(-k·|d|) at a fraction of the ALU. α=64 gives the same kink
    // depth as the old kSharp=8.
    vec4 d4 = textureGather(u_lr_depth, uv);                   // .x TL, .y TR, .z BR, .w BL
    vec4 r4 = textureGather(u_lr_color, uv, 0);
    vec4 g4 = textureGather(u_lr_color, uv, 1);
    vec4 b4 = textureGather(u_lr_color, uv, 2);
    vec3 c0  = texture(u_lr_color, uv).rgb;
    vec3 cTL = vec3(r4.x, g4.x, b4.x);
    vec3 cTR = vec3(r4.y, g4.y, b4.y);
    vec3 cBR = vec3(r4.z, g4.z, b4.z);
    vec3 cBL = vec3(r4.w, g4.w, b4.w);

    const float kAlpha = 64.0;
    vec4 dd = d4 - vec4(lr_depth);
    vec4 w  = 1.0 / (1.0 + kAlpha * dd * dd);
    float wsum = 1.0 + w.x + w.y + w.z + w.w;
    vec3 c_blur = (c0 + w.x*cTL + w.y*cTR + w.z*cBR + w.w*cBL) / wsum;
    vec3 c = c_blur;

    float strength = pc.color.x;
    if (strength > 0.0 && pc.color.y > 0.5 && pc.color.z > 0.5) {
        c = casSharpen(c0, cTL, cTR, cBR, cBL, strength);
    }

    // Depth edges. Where the four low-res texels straddle a silhouette
    // (a near bump in front of far ground), the bilinear depth above is a
    // mix of the two: just inside the outline it lands far BEHIND the near
    // surface, so rocks, trunks and bushes standing behind the bump were
    // drawn over it in a fringe one low-res texel wide. Instead pick the
    // side that owns the pixel (bilinear weight >= 1/2, the same outline
    // the blend would give) and take depth AND colour from that side only.
    {
        vec4 z4 = 0.05 / max(vec4(1.0000333) - d4, vec4(1e-7));   // view distance (near 0.05, far 1500)
        float zmin = min(min(z4.x, z4.y), min(z4.z, z4.w));
        float zmax = max(max(z4.x, z4.y), max(z4.z, z4.w));
        if (zmax > zmin * 1.2 + 0.05) {
            vec2 fr = fract(uv * vec2(textureSize(u_lr_depth, 0)) - 0.5);
            // textureGather order: .x (0,1)  .y (1,1)  .z (1,0)  .w (0,0)
            vec4 bw = vec4((1.0 - fr.x) * fr.y, fr.x * fr.y, fr.x * (1.0 - fr.y), (1.0 - fr.x) * (1.0 - fr.y));
            vec4 nearm = vec4(lessThan(z4, vec4(zmin * 1.2 + 0.05)));
            vec4 skym  = vec4(greaterThanEqual(d4, vec4(0.99999)));
            vec4 farm  = (1.0 - nearm) * (1.0 - skym);
            float wn = dot(bw, nearm), ws = dot(bw, skym), wf = dot(bw, farm);
            vec4 pick;
            if (wn >= 0.5)      pick = bw * nearm;
            else if (ws > wf)   discard;
            else                pick = bw * farm;
            float ps = pick.x + pick.y + pick.z + pick.w;
            if (ps < 1e-5) discard;
            pick /= ps;
            c = cTL * pick.x + cTR * pick.y + cBR * pick.z + cBL * pick.w;
            lr_depth = dot(pick, d4);
        }
    }
    outColor     = vec4(c, 1.0);
    outMotion    = texture(u_lr_motion, uv).rg;
    gl_FragDepth = lr_depth;
}
