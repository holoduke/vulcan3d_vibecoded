#version 460

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

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

const int kTextureCount = 13;
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;
    vec4  rt_params;
    ivec4 rt_flags2;
    vec4  rt_params2;
    vec4  camera_pos;
    vec4  rt_lod;
    vec4  viewport;
    vec4  muzzle_pos;
    vec4  muzzle_color;
    vec4  terrain_params;
    vec4  terrain_h_low;
    vec4  terrain_h_high;
    vec4  grass_extra;
    vec4  grass_extra2;
    mat4  light_vp;
} scene;

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
    outGBuffer0 = vec4(clamp(lit, vec3(0.0), vec3(1.0)), 5.0 / 255.0);
    outGBuffer1 = vec4(_octa_encode(N), 0.6, 0.0);
    {
        vec2 current_uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            outMotion = current_uv - (prev_ndc * 0.5 + 0.5);
        } else {
            outMotion = vec2(0.0);
        }
    }
}
