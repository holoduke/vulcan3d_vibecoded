#version 460
// Small-object shadow map (flowers, ferns, scatter rocks). Same instance
// layout and placement as scatter.vert, depth-only from the sun. Only
// instances within `radius` of the camera are kept; the rest collapse.
layout(location = 0) in vec3 inPos;
layout(location = 3) in vec4 inPosScale;
layout(location = 4) in vec4 inRotTint;
layout(push_constant) uniform PC {
    mat4 light_vp;
    vec4 cam;        // .xyz camera, .w radius
} pc;
void main() {
    vec3 base = inPosScale.xyz;
    float s = inPosScale.w;
    vec2 d = base.xz - pc.cam.xz;
    if (dot(d, d) > pc.cam.w * pc.cam.w || s < 0.05) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    float c = cos(inRotTint.x), sn = sin(inRotTint.x);
    mat3 R = mat3(vec3(c, 0, sn), vec3(0, 1, 0), vec3(-sn, 0, c));
    vec3 world = R * (inPos * s) + base;
    gl_Position = pc.light_vp * vec4(world, 1.0);
}
