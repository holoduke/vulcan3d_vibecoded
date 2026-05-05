#version 460

// Vertexless fullscreen triangle: 3 vertices that cover the entire viewport.
// Standard trick — no vertex buffer required.
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
