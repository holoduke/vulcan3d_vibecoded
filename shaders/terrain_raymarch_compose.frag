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

// Catmull-Rom upsample in 9 bilinear taps (the 4x4 footprint folded into
// 3x3 fetches). Continuous across low-res texel borders and it keeps
// edges crisp, where the old "average of the bilinear tap and the four
// gathered corners" changed in texel-sized steps: distant mountains came
// out as a grid of blocks.
vec3 lrCatmullRom(vec2 uv) {
    vec2 ts  = vec2(textureSize(u_lr_color, 0));
    vec2 sp  = uv * ts;
    vec2 tc1 = floor(sp - 0.5) + 0.5;
    vec2 f   = sp - tc1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0  = (tc1 - 1.0) / ts;
    vec2 t3  = (tc1 + 2.0) / ts;
    vec2 t12 = (tc1 + w2 / w12) / ts;
    vec3 r = vec3(0.0);
    r += textureLod(u_lr_color, vec2(t0.x,  t0.y),  0.0).rgb * (w0.x  * w0.y);
    r += textureLod(u_lr_color, vec2(t12.x, t0.y),  0.0).rgb * (w12.x * w0.y);
    r += textureLod(u_lr_color, vec2(t3.x,  t0.y),  0.0).rgb * (w3.x  * w0.y);
    r += textureLod(u_lr_color, vec2(t0.x,  t12.y), 0.0).rgb * (w0.x  * w12.y);
    r += textureLod(u_lr_color, vec2(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
    r += textureLod(u_lr_color, vec2(t3.x,  t12.y), 0.0).rgb * (w3.x  * w12.y);
    r += textureLod(u_lr_color, vec2(t0.x,  t3.y),  0.0).rgb * (w0.x  * w3.y);
    r += textureLod(u_lr_color, vec2(t12.x, t3.y),  0.0).rgb * (w12.x * w3.y);
    r += textureLod(u_lr_color, vec2(t3.x,  t3.y),  0.0).rgb * (w3.x  * w3.y);
    return max(r, vec3(0.0));
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
    vec3 edge_motion = vec3(0.0);
    vec4 d4 = textureGather(u_lr_depth, uv);                   // .x TL, .y TR, .z BR, .w BL
    vec4 r4 = textureGather(u_lr_color, uv, 0);
    vec4 g4 = textureGather(u_lr_color, uv, 1);
    vec4 b4 = textureGather(u_lr_color, uv, 2);
    vec3 c0  = texture(u_lr_color, uv).rgb;
    vec3 cTL = vec3(r4.x, g4.x, b4.x);
    vec3 cTR = vec3(r4.y, g4.y, b4.y);
    vec3 cBR = vec3(r4.z, g4.z, b4.z);
    vec3 cBL = vec3(r4.w, g4.w, b4.w);

    // Smooth interior: Catmull-Rom, then an optional contrast-adaptive
    // sharpen whose neighbours are bilinear taps one low-res texel away
    // (continuous, unlike the gathered corners).
    vec3 c = lrCatmullRom(uv);
    float strength = pc.color.x;
    if (strength > 0.0 && pc.color.y > 0.5 && pc.color.z > 0.5) {
        vec2 lt = 1.0 / vec2(textureSize(u_lr_color, 0));
        vec3 nA = textureLod(u_lr_color, uv + vec2(-lt.x, 0.0), 0.0).rgb;
        vec3 nB = textureLod(u_lr_color, uv + vec2( lt.x, 0.0), 0.0).rgb;
        vec3 nD = textureLod(u_lr_color, uv + vec2(0.0, -lt.y), 0.0).rgb;
        vec3 nE = textureLod(u_lr_color, uv + vec2(0.0,  lt.y), 0.0).rgb;
        c = max(casSharpen(c, nA, nB, nD, nE, strength), vec3(0.0));
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
        // A depth step, and not just a plane seen at a grazing angle: NDC
        // depth is linear across the screen on a plane, so its second
        // difference is ~0 there (flat ground far away also spans > 20 %
        // in depth per texel and used to take this path needlessly).
        vec2  lt_d = 1.0 / vec2(textureSize(u_lr_depth, 0));
        float dL = textureLod(u_lr_depth, uv - vec2(lt_d.x, 0.0), 0.0).r;
        float dR = textureLod(u_lr_depth, uv + vec2(lt_d.x, 0.0), 0.0).r;
        float dU = textureLod(u_lr_depth, uv - vec2(0.0, lt_d.y), 0.0).r;
        float dD = textureLod(u_lr_depth, uv + vec2(0.0, lt_d.y), 0.0).r;
        float curv = max(abs(dL + dR - 2.0 * lr_depth), abs(dU + dD - 2.0 * lr_depth));
        bool  stepped = curv > 0.12 * (1.0000333 - lr_depth);
        if (stepped && zmax > zmin * 1.2 + 0.05) {
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
            // Motion from the same side (a bilinear mix dragged in the sky's
            // zero motion / the far side's).
            vec4 mx4 = textureGather(u_lr_motion, uv, 0), my4 = textureGather(u_lr_motion, uv, 1);
            edge_motion = vec3(dot(pick, mx4), dot(pick, my4), 1.0);
        }
    }
    outColor     = vec4(c, 1.0);
    outMotion    = edge_motion.z > 0.5 ? edge_motion.xy : texture(u_lr_motion, uv).rg;
    gl_FragDepth = lr_depth;
}
