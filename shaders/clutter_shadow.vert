#version 460
// Small-object shadow map (flowers, ferns, scatter rocks). Same instance
// layout, placement, distance fade and wind sway as scatter.vert, depth-only
// from the sun, so the shadows match exactly what the colour pass draws.
// Only instances within `radius` of the camera are kept; the rest collapse.
layout(location = 0) in vec3 inPos;
layout(location = 2) in vec2 inUv;
layout(location = 3) in vec4 inPosScale;
layout(location = 4) in vec4 inRotTint;
layout(push_constant) uniform PC {
    mat4 light_vp;
    vec4 cam;        // .xyz camera, .w radius
    vec4 p;          // .x wind time, .y plant (sway) flag, .z colour-pass draw distance
} pc;
void main() {
    vec3 base = inPosScale.xyz;
    float s = inPosScale.w;
    vec2 d = base.xz - pc.cam.xz;
    if (dot(d, d) > pc.cam.w * pc.cam.w || s < 0.05) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    // Distance fade (scatter.vert): shrink over the last 25 % of reach.
    float dist = distance(base, pc.cam.xyz);
    float max_d = max(pc.p.z, 10.0);
    float reach = max_d * clamp(s * 2.2 + 0.22, 0.22, 0.85);
    float fade = 1.0 - smoothstep(reach * 0.75, reach, dist);
    if (fade < 0.03) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    float c = cos(inRotTint.x), sn = sin(inRotTint.x);
    mat3 R = mat3(vec3(c, 0, sn), vec3(0, 1, 0), vec3(-sn, 0, c));
    vec3 world = R * (inPos * s * fade) + base;
    if (pc.p.y > 0.5 && inUv.y > 0.0) {   // plant wind sway (scatter.vert)
        float t = pc.p.x;
        float ph = dot(base.xz, vec2(0.37, 0.23));
        vec2 wd = normalize(vec2(0.8, 0.45));
        float w = sin(t * 2.1 + ph) * 0.7 + sin(t * 4.7 + ph * 1.9) * 0.3 + 0.25;
        world.xz += wd * (w * 0.06 * s * inUv.y * inUv.y);
        world.y  -= abs(w) * 0.015 * s * inUv.y;
    }
    gl_Position = pc.light_vp * vec4(world, 1.0);
}
