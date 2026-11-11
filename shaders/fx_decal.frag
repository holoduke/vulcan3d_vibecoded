#version 460
#extension GL_GOOGLE_include_directive : require
// Projected laser-burn decal (screen-space / deferred decal technique:
// the decal is a box volume; each covered pixel rebuilds its world position
// from scene depth, moves it into the decal's frame and shades it there, so
// the mark wraps over whatever geometry is inside the box - bumpy walls,
// terrain, crate corners - instead of being a flat disc floating on top).
//
// Layers, centre outward: a small glassy burn pit with a raised rim, a
// charred core, a soot halo with radial blast streaks and a ragged edge,
// and - while fresh - a molten core that cools white -> yellow -> orange ->
// dull red, hairline glowing cracks and a few dying embers.
//
// Drawn in the FX pass over the LIT scene, premultiplied (rgb, a): soot
// darkens what is underneath (dst * (1 - a)), the heat glow adds on top.
//
// Push constants:
//   color        = decal centre.xyz, radius (m)
//   emissive     = surface normal.xyz, life (1 fresh -> 0 expired)
//   tex_params   = tangent X.xyz, seed
//   grass_params = age (s), projection half-depth (m)
//   prev_mvp     = inverse view-projection

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vTint;
layout(location = 2) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 inv_vp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

#include "scene_ubo.glsl"

layout(set = 0, binding = 29) uniform sampler2D u_scene_depth;

float h21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 x) {
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x),
               mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; ++i) { s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; }
    return s;   // ~0..0.94
}
// Periodic noise around the circle (no seam at +-pi).
float ring_noise(float ang, float freq, float seed) {
    float t = ang * freq / 6.2831853;
    float i = floor(t), f = fract(t);
    f = f * f * (3.0 - 2.0 * f);
    return mix(h21(vec2(mod(i, freq), seed)), h21(vec2(mod(i + 1.0, freq), seed)), f);
}
// Blackbody-ish ramp for a cooling surface, t in 0..1.
vec3 heat_color(float t) {
    vec3 c = vec3(1.0, 0.07, 0.0) * smoothstep(0.0, 0.35, t);          // dull red
    c = mix(c, vec3(1.0, 0.42, 0.04), smoothstep(0.25, 0.6, t));        // orange
    c = mix(c, vec3(1.0, 0.85, 0.45), smoothstep(0.55, 0.85, t));       // yellow
    c = mix(c, vec3(1.0, 0.97, 0.9), smoothstep(0.8, 1.0, t));          // white hot
    return c;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    float zs = texelFetch(u_scene_depth, px, 0).r;
    if (zs >= 0.99999) discard;
    vec2 suv = gl_FragCoord.xy * scene.viewport.zw;
    vec4 wp4 = pc.inv_vp * vec4(suv * 2.0 - 1.0, zs, 1.0);
    vec3 P = wp4.xyz / wp4.w;

    vec3  C = pc.color.xyz;
    float R = pc.color.w;
    vec3  N = pc.emissive.xyz;
    vec3  X = pc.tex_params.xyz;
    vec3  Z = cross(N, X);
    float seed = pc.tex_params.w;
    float age  = pc.grass_params.x;
    float life = pc.emissive.w;

    vec3  d = P - C;
    float h = dot(d, N);
    if (abs(h) > pc.grass_params.y) discard;
    vec2  q = vec2(dot(d, X), dot(d, Z)) / R;
    float r = length(q);
    if (r >= 1.0) discard;

    // Surface normal from the depth-reconstructed position: fade the
    // projection out on faces turned away from the decal axis (stops the
    // mark smearing down the side of a corner it wraps around).
    vec3 sn = cross(dFdx(P), dFdy(P));
    float snl = length(sn);
    float facing = 1.0;
    if (snl > 1e-12) {
        sn /= snl;
        if (dot(sn, scene.camera_pos.xyz - P) < 0.0) sn = -sn;
        facing = smoothstep(0.15, 0.55, dot(sn, N));
    }
    // Soft falloff toward the projection depth limit.
    facing *= 1.0 - smoothstep(0.6, 1.0, abs(h) / pc.grass_params.y);
    if (facing <= 0.002) discard;

    float ang = atan(q.y, q.x);
    vec2  qs  = q + seed * 13.7;                       // per-decal noise offset
    float n_lo = fbm(qs * 3.0);
    float n_hi = fbm(qs * 11.0);

    // Radial blast streaks: the soot reaches further along some directions.
    float streak = ring_noise(ang, 9.0, seed) * 0.6 + ring_noise(ang, 23.0, seed + 3.1) * 0.4;
    float reach  = 0.52 + 0.46 * streak * streak;      // 0.52 .. 0.98 of the radius
    float rr = r / reach + (n_lo - 0.47) * 0.38;       // ragged, noise-eaten edge

    // --- soot / char coverage -------------------------------------------
    float halo = 1.0 - smoothstep(0.30, 1.0, rr);      // wide soft soot
    halo *= halo;
    float core = 1.0 - smoothstep(0.16, 0.42, r + (n_hi - 0.47) * 0.10);   // charred centre
    float pit  = 1.0 - smoothstep(0.045, 0.085, r);                          // burn pit
    float rim  = smoothstep(0.07, 0.10, r) * (1.0 - smoothstep(0.10, 0.15, r));
    // Fine speckle so the halo reads as deposited soot, not a gradient.
    float speck = mix(0.72, 1.0, n_hi);
    float a = clamp(max(core * 0.96, halo * 0.80 * speck), 0.0, 1.0);

    // Soot reflects very little; the charred core and pit almost nothing.
    // Lit by the scene's ambient + sun so it sits in the picture day or night.
    vec3 light = scene.sky_color.rgb * 0.35 + scene.sun_color.rgb * scene.sun_color.a * 0.25;
    vec3 soot_alb = mix(vec3(0.045, 0.036, 0.030), vec3(0.012, 0.011, 0.010), core);
    soot_alb = mix(soot_alb, vec3(0.004), pit);
    soot_alb += vec3(0.03, 0.028, 0.026) * rim * (0.6 + 0.8 * n_hi);          // glassy fused rim catches light
    vec3 col = soot_alb * light;

    // --- heat ------------------------------------------------------------
    // The centre is hottest and stays hot longest; the charred area cools
    // from the outside in. Temperatures are 0..1 on the ramp above.
    float cool_core = exp(-age * 0.55);                 // seconds
    float cool_char = exp(-age * 1.6);
    float t_pit  = pit * (0.25 + 0.75 * cool_core);
    float t_char = core * (1.0 - smoothstep(0.0, 0.40, r)) * cool_char * (0.55 + 0.45 * n_hi);
    // Hairline cracks radiating from the pit, glowing while the core is hot.
    float crack_n = ring_noise(ang, 14.0, seed + 7.7);
    float crack = (1.0 - smoothstep(0.0, 0.06, abs(crack_n - 0.5))) *
                  (1.0 - smoothstep(0.08, 0.36, r)) * smoothstep(0.05, 0.09, r);
    float t_crack = crack * cool_core * 0.8;
    float temp = clamp(max(max(t_pit * cool_core, t_char), t_crack), 0.0, 1.0);
    // Embers: sparse cells in the soot that flicker and die within ~4 s.
    vec2  ec = floor(qs * 26.0);
    float eh = h21(ec);
    float ember = step(0.955, eh) * (1.0 - smoothstep(0.15, 0.75, rr)) *
                  exp(-age * (0.5 + 1.2 * h21(ec + 5.0))) *
                  (0.55 + 0.45 * sin(age * (9.0 + 14.0 * eh) + eh * 40.0));
    temp = max(temp, clamp(ember, 0.0, 1.0) * 0.6);

    vec3 glow = heat_color(temp) * temp * temp * 14.0;

    // End of life: the whole mark fades out over its last 20 %.
    float fade = smoothstep(0.0, 0.2, life) * facing;
    a *= fade;
    outColor = vec4(col * a + glow * fade, a);
}
