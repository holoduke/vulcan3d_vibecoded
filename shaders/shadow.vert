#version 460

// Minimal depth-only shader for the sun shadow map. Reads only the
// vertex position from the standard Vertex layout (location 0); the
// caller pre-multiplies the light view-proj into pc.mvp so this is
// just a single matrix transform. No fragment shader is bound — the
// rasteriser writes only depth.

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PC {
    mat4 mvp;        // light_view_proj * model
    mat4 model;      // unused here, kept for layout compat with cube.vert
    mat4 prev_mvp;   // unused
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
