#version 460

// Fullscreen-triangle vertex shader for the deferred lighting pass.
// 3 verts cover the entire viewport. Standard half-screen-triangle
// trick — gl_VertexIndex 0..2 maps to NDC corners (-1,-1), (-1,3),
// (3,-1) so a single triangle covers [-1,1]² with no scissor work.

layout(location = 0) out vec2 vUv;

void main() {
    vUv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUv * 2.0 - 1.0, 0.0, 1.0);
}
