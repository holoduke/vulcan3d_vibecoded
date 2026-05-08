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

// Half-res raymarch outputs from the previous pass.
layout(set = 0, binding = 9)  uniform sampler2D u_lr_color;
layout(set = 0, binding = 10) uniform sampler2D u_lr_motion;
layout(set = 0, binding = 11) uniform sampler2D u_lr_depth;

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

    outColor  = texture(u_lr_color,  uv);
    outMotion = texture(u_lr_motion, uv).rg;
    gl_FragDepth = lr_depth;
}
