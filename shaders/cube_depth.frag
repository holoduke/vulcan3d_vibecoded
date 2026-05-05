#version 460

// Depth-only fragment shader. The depth pre-pass writes z but doesn't need a
// color output — the rasterizer fills the depth buffer using the vertex
// shader's gl_Position alone, and this stub keeps the pipeline valid.
//
// Subsequent color pass binds a pipeline with depth_compare=LESS_OR_EQUAL +
// depth_write=false, so the GPU's hierarchical early-Z kicks in and only
// runs the expensive cube.frag for fragments that match the pre-pass z.
void main() {}
