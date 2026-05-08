#version 460

// Fullscreen-triangle vertex stage for the procedural FBM ray-marched
// terrain. Three indices (0,1,2) cover the entire NDC quad without
// allocating a vertex buffer. Output is just the NDC.xy passed through;
// the fragment stage reconstructs the world ray from inverse(view_proj).

layout(location = 0) out vec2 vNDC;

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
}
