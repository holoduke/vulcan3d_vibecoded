#version 460
#extension GL_GOOGLE_include_directive : require
// Depth-prepass fragment for GPU vegetation: same LOD-fade dither and
// impostor silhouette cut as cube.frag so the prepass depth matches the
// colour pass exactly (LESS_OR_EQUAL).
layout(location = 0) in vec3 vNormal;
layout(location = 2) in vec3 vWorldPos;
layout(location = 4) in vec2 vUv;
layout(location = 6) in vec3 vObjectPos;
layout(location = 10) flat in vec4 vVegInfo;
#include "scene_ubo.glsl"
#include "veg_leaf.glsl"
void main() {
    if (vVegInfo.x > 0.0) {
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 5.588238 * float(uint(scene.rt_flags.w) & 7u),
                                                  vec2(0.06711056, 0.00583715))));
        if (vVegInfo.y > 0.5 ? (n >= vVegInfo.x) : (n < vVegInfo.x)) discard;
    }
    if (vVegInfo.z > 0.5) {
        float u = vUv.x, v = vUv.y;
        float jag = (fract(sin(dot(floor(vec2(u * 9.0, v * 14.0)), vec2(12.9, 78.2))) * 43758.5) - 0.5);
        bool inside;
        if (vVegInfo.z < 1.5) {
            float tier = fract(v * 6.0);
            float w = (1.0 - v) * (0.75 + 0.25 * tier) + jag * 0.06;
            inside = (v > 0.12 && abs(u) < w) || (v <= 0.14 && abs(u) < 0.06);
        } else {
            vec2 q = vec2(u, (v - 0.62) * 1.5);
            float rr = 0.78 + 0.12 * sin(atan(q.y, q.x) * 5.0 + vVegInfo.w) + jag * 0.08;
            inside = length(q) < rr || (v < 0.4 && abs(u) < 0.07);
        }
        if (!inside) discard;
    } else {
        int vm = int(vUv.y + 0.5);
        if ((vm == 2 || vm == 3) && veg_leaf_cut(vObjectPos, vWorldPos, vNormal, scene.camera_pos.xyz)) discard;
    }
}
