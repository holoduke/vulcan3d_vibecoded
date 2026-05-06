#version 460

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vUv;
layout(location = 3) in float vHeightRatio;
layout(location = 4) in float vCullKill;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

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
} scene;

void main() {
    if (vCullKill > 0.5) discard;

    // Cheap lighting — the blades are tiny, RT shadows on every pixel
    // would be wasteful. Half-Lambert sun + blue-tinted sky ambient
    // keeps them readable as foliage from any angle.
    vec3 N = normalize(vNormal);
    vec3 L = normalize(scene.sun_direction.xyz);
    float n_dot_l = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);

    // Sun contribution: deliberately under-baked so 100k+ tiny blades
    // can't over-stuff the bloom mip chain. At sun_color.a == 2 a flat
    // green blade peaks around 0.4 luminance — well under the default
    // 1.05 bloom_threshold, so blades don't trigger global bloom +
    // auto-exposure runaway (which earlier dumped the screen into
    // big green halos).
    vec3 sun_term = scene.sun_color.rgb * (scene.sun_color.a * 0.25) * n_dot_l;
    vec3 sky_term = scene.sky_color.rgb * 0.35;

    // Subtle bottom→tip variation; nothing strong enough to push the
    // tip past bloom threshold.
    vec3 tip_lift = mix(vec3(0.9), vec3(1.05, 1.0, 0.9), vHeightRatio);
    vec3 base = vColor * tip_lift;

    vec3 lit = base * (sun_term + sky_term);

    // Soft alpha across the blade UV — slightly tapered sides so
    // adjacent blades blend visually without needing alpha-blend +
    // back-to-front sort. Clipping to 0.4 keeps the shape solid.
    float side_taper = smoothstep(0.0, 0.18, vUv.x) *
                       (1.0 - smoothstep(0.82, 1.0, vUv.x));
    if (side_taper < 0.4 && vHeightRatio > 0.85) discard;

    outColor = vec4(lit, 1.0);
    outMotion = vec2(0.0);   // wind motion not tracked — TAA tolerates
}
