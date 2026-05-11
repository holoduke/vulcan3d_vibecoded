#version 460

// Fullscreen-triangle vertex stage for the procedural FBM ray-marched
// terrain. Three indices (0,1,2) cover the entire NDC quad without
// allocating a vertex buffer.
//
// We pre-multiply the near + far world-space corner positions per
// vertex (3 mat4×vec4 ops per frame) and interpolate them; the frag
// stage just does the homogeneous divide + subtract + normalize for
// the per-pixel ray direction. Saves the per-pixel mat4×vec4 (was
// done at 1080p × 2 = ~4M times per frame).

layout(location = 0) out vec2 vNDC;
layout(location = 1) out vec4 vWNear;
layout(location = 2) out vec4 vWFar;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;       // = inv(view_proj)
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

void main() {
    vec2 ndc = vec2(
        float((gl_VertexIndex << 1) & 2),
        float(gl_VertexIndex & 2)) * 2.0 - 1.0;
    // z = 1 = far plane; the frag writes its own gl_FragDepth at the
    // ray-march hit, so the position emitted here only fixes raster
    // coverage. depth_compare LESS_OR_EQUAL lets gl_FragDepth do the
    // actual occlusion against earlier rasterised draws.
    gl_Position = vec4(ndc, 1.0, 1.0);
    vNDC = ndc;
    vWNear = pc.model * vec4(ndc, 0.0, 1.0);
    vWFar  = pc.model * vec4(ndc, 1.0, 1.0);
}
