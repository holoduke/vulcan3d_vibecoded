#version 460
#extension GL_GOOGLE_include_directive : require

// Ground clutter fragment: rock-textured Lambert with the scene's sun +
// shadow + sky ambient. Ships pre-shaded colour through the gbuffer as
// material 5 (pass-through), same contract as grass.frag, so the
// deferred pass keeps the forward look exactly.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vWorldPos;
layout(location = 3) in float vSunShadow;
layout(location = 4) in float vCullKill;
layout(location = 5) in float vDistToCam;
layout(location = 6) in vec4 vPrevClip;
layout(location = 7) in vec2 vUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;   // .y = 1 for plants
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

const int kTextureCount = 15;
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];

#include "scene_ubo.glsl"

vec2 _octa_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = (n.z >= 0.0)
        ? n.xy
        : ((1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                     n.y >= 0.0 ? 1.0 : -1.0));
    return e * 0.5 + 0.5;
}

void main() {
    if (vCullKill > 0.5) discard;

    vec3 N = normalize(vNormal);
    bool plant = pc.grass_params.y > 0.5;
    if (plant) {
        // Two-sided cards / fronds: face the normal toward the viewer.
        vec3 V = normalize(scene.camera_pos.xyz - vWorldPos);
        if (dot(N, V) < 0.0) N = -N;
    }

    // Rock albedo: 2-axis "biplanar" sample of the terrain rock layer
    // (slot 7) — full triplanar is overkill for fist-sized clutter.
    vec3 bw = abs(N);
    vec2 uvA = vWorldPos.zy * 2.0;
    vec2 uvB = vWorldPos.xz * 2.0;
    float wA = bw.x + bw.z * 0.5;
    float wB = bw.y;
    vec3 tex = (texture(u_albedo[7], uvA).rgb * wA +
                texture(u_albedo[7], uvB).rgb * wB) / max(wA + wB, 1e-3);
    vec3 albedo = tex * vColor;
    if (plant) {
        // Part colours: stems / leaves green (ferns carry their own
        // tint), petals = instance colour, flower centre golden.
        int part = int(vUv.x + 0.5);
        bool fern = vColor.g > vColor.r * 1.5 && vColor.g > vColor.b * 1.5 && vColor.g < 0.35;
        vec3 leaf = fern ? vColor : vec3(0.11, 0.22, 0.05);
        leaf *= 0.8 + 0.4 * vUv.y;                // tips catch more light
        albedo = (part == 0) ? leaf : (part == 1 ? vColor : vec3(0.75, 0.55, 0.04));
    }

    vec3  L = normalize(scene.sun_direction.xyz);
    // Half-Lambert wrap (same reason as grass.frag): hard Lambert on
    // randomly-rotated faceted rocks floors half the faces to black.
    float ndl_raw = dot(N, L);
    float wrap = ndl_raw * 0.5 + 0.5;
    float ndl = wrap * wrap;
    float lit_sun = ndl * (1.0 - vSunShadow * 0.7);
    float sky_up  = N.y * 0.5 + 0.5;
    vec3  amb = mix(scene.ambient.rgb, scene.sky_color.rgb * 0.45, sky_up) *
                scene.rt_params.z;
    vec3 lit = albedo * (scene.sun_color.rgb * scene.sun_color.a * lit_sun
                          + amb);

    outColor = vec4(lit, 1.0);
    // Material 1: the deferred pass re-shades with RT sun shadows, AO and
    // GI like the rest of the world (was pre-lit material 5 with only
    // the coarse shadow map).
    outGBuffer0 = vec4(clamp(albedo, vec3(0.0), vec3(1.0)), 1.0 / 255.0);
    // Plants: flagged as leaves so the deferred pass lights them with the
    // foliage scattering model (wrap + transmission).
    outGBuffer1 = vec4(_octa_encode(N), 0.6, plant ? 1.0 : 0.0);
    {
        vec2 current_uv = gl_FragCoord.xy * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            // Unjittered motion, same convention as cube.frag.
            outMotion = current_uv - (prev_ndc * 0.5 + 0.5) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
    }
}
