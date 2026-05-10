#version 460

// Minimal depth-prepass vertex shader. Replaces cube.vert in the depth-
// only pipeline so the prepass doesn't pay the cost of:
//   * 2 unused mat4 multiplies (model + prev_mvp)
//   * 9 varying outputs (vNormal, vColor, vWorldPos, vEmissive, vUv,
//     vTexParams, vObjectPos, vObjectNormal, vPrevClip)
// All of which are dead in the prepass — cube_depth.frag is empty.
//
// Push-constant layout aliases the first mat4 of the full PushConstants
// struct (mvp). The rest of the 256-byte PC range is implicitly ignored.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // unused, kept for VkVertexInputAttribute compatibility
layout(location = 2) in vec2 inUv;       // unused, same reason

layout(push_constant) uniform PC {
    mat4 mvp;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
