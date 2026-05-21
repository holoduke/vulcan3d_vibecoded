#version 460

// Minimal depth-only shader for the sun shadow map. Reads only the
// vertex position from the standard Vertex layout (location 0); the
// caller pre-multiplies the light view-proj into pc.mvp so this is
// just a single matrix transform. No fragment shader is bound — the
// rasteriser writes only depth.

layout(location = 0) in vec3 inPosition;

// Declare ONLY the bytes this pipeline's VkPushConstantRange backs.
// sun_shadow.cpp reserves exactly sizeof(mat4) = 64 bytes; the old
// declaration mirrored cube.vert's full ~240-byte PushConstants
// block, so the shader advertised a [0,240] push range against a
// [0,64] layout — a Vulkan spec violation (VUID-...-10069) and UB
// to reference the unbacked tail. Only `mvp` is ever used.
layout(push_constant) uniform PC {
    mat4 mvp;        // light_view_proj * model
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
