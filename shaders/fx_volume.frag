#version 460
// Raymarched explosion volume (fireball / smoke column / ground dust).
//
// Technique (volumetric explosions as used in games, cf. Dunn,
// "Realistic Volumetric Explosions in Games", GPU Pro 6): a noise-
// displaced sphere density field, marched front-to-back with Beer-Lambert
// transmittance; fire emission from a blackbody temperature ramp, smoke
// lit by the sun with one self-shadow tap per step. Output is
// premultiplied (rgb, 1 - T) so it composites with ONE / 1-SRC_ALPHA.
//
// Push constants:
//   color      = centre.xyz, radius (m)
//   emissive   = age (s), kind (0 fire, 1 smoke, 2 dust), seed, density
//   tex_params = y-squash, heat (0..1), fade (0..1 opacity mult), scroll
//   prev_mvp   = inverse view-projection (scene depth reconstruction)

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

layout(set = 0, binding = 0) uniform Scene {
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
} scene;

// Prepass depth copy (terrain + castle meshes): bounds the march so the
// fireball is cut by the ground instead of showing through it.
layout(set = 0, binding = 27) uniform sampler2D u_prepass_depth;
// Full scene depth (gun, props, dynamic objects included) - see descriptors.cpp.
layout(set = 0, binding = 29) uniform sampler2D u_scene_depth;

float h3(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x),
                   mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x),
                   mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p, int oct, float tm) {
    float a = 0.5, s = 0.0, norm = 0.0;
    for (int i = 0; i < 5; ++i) {
        if (i >= oct) break;
        // Boiling: every octave drifts through the noise field over time,
        // finer octaves faster and in a different direction (octave
        // offsets below decorrelate them) -> the volume churns from the
        // inside instead of a texture sliding over a fixed blob.
        p += vec3(0.13, 0.31, -0.19) * tm * (0.35 + 0.45 * float(i));
        // Billow noise (|2n-1|): rounded puffs with creased gaps between
        // them - the cauliflower look of real fireball / smoke surfaces.
        s += a * (1.0 - abs(vnoise(p) * 2.0 - 1.0));
        norm += a;
        p = p * 2.07 + vec3(17.1, 3.7, 9.2);
        a *= 0.5;
    }
    return s / max(norm, 1e-4);
}

// Blackbody-ish ramp, t in 0..1 (~800 K .. ~3000 K): deep red -> orange
// -> yellow -> white, with radiance rising steeply (Stefan-Boltzmann-ish).
vec3 blackbody(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c = mix(vec3(0.55, 0.04, 0.0), vec3(1.0, 0.32, 0.03), smoothstep(0.0, 0.4, t));
    c = mix(c, vec3(1.0, 0.72, 0.25), smoothstep(0.35, 0.75, t));
    c = mix(c, vec3(1.0, 0.95, 0.85), smoothstep(0.75, 1.0, t));
    return c * (t * t * t * 38.0 + t * 1.5);
}

int   g_kind;
float g_age, g_seed, g_scroll;

// Density in unit-sphere space u (|u| = 1 at the volume radius).
float density(vec3 u, int oct) {
    float r = length(u);
    if (r > 1.0) return 0.0;
    float freq = (g_kind == 0) ? 3.4 : (g_kind == 1 ? 2.8 : 3.6);
    vec3 q = u * freq + vec3(g_seed, g_seed * 0.7, -g_seed);
    // Coherent expansion: as the puff grows, features grow with it (sample
    // space shrinks slowly) so billows roll outward rather than shimmer.
    q *= 1.0 / (1.0 + 0.04 * g_scroll);
    // Domain warp: bends the puffs so they don't sit on a grid.
    // Time-varying domain warp: the big swirls themselves evolve.
    vec3 wq = q * 0.5 + vec3(0.0, 0.0, g_scroll * 0.22);
    q += (vec3(vnoise(wq + 3.1), vnoise(wq + 7.7), vnoise(wq + 1.3)) - 0.5) * 1.8;
    float n = fbm(q, oct, g_scroll);
    // Puffy edge: the billow field carves the sphere boundary so the
    // silhouette is lumpy, not a ball.
    float edge = (g_kind == 2) ? 1.9 : 2.3;
    // Old smoke frays: the noise eats deeper into the volume with age.
    if (g_kind == 1) edge += 0.6 + clamp(g_age * 0.12, 0.0, 1.2);   // ragged from birth, frays further
    float d = (1.0 - r) * 3.0 - (1.0 - n) * edge + 0.55;
    // Dust is a ring (torus-ish): thin out the middle, it's pushed outward.
    if (g_kind == 2) d -= (1.0 - smoothstep(0.2, 0.7, length(u.xz))) * 3.2;
    return smoothstep(0.0, 0.28, d);
}

void main() {
    vec3  C   = pc.color.xyz;
    float Rad = pc.color.w;
    g_age   = pc.emissive.x;
    g_kind  = int(pc.emissive.y + 0.5);
    g_seed  = pc.emissive.z;
    float dens_scale = pc.emissive.w;
    float squash = pc.tex_params.x;
    float heat   = pc.tex_params.y;
    float fade   = pc.tex_params.z;
    g_scroll     = pc.tex_params.w;

    vec3 ro = scene.camera_pos.xyz;
    vec3 rd = normalize(vWorld - ro);
    vec3 S  = vec3(Rad, Rad * squash, Rad);
    // Ray in unit-sphere space (t stays world distance along rd).
    vec3 o = (ro - C) / S;
    vec3 d = rd / S;
    float a = dot(d, d), b = dot(o, d), c = dot(o, o) - 1.0;
    float disc = b * b - a * c;
    if (disc <= 0.0) discard;
    float sq = sqrt(disc);
    float t0 = max((-b - sq) / a, 0.0);
    float t1 = (-b + sq) / a;
    // Clip against the full opaque scene (the prepass copy lacked the gun,
    // props and dynamic objects: smoke painted over them from inside).
    vec2  suv = gl_FragCoord.xy * scene.viewport.zw;
    float zs  = texelFetch(u_scene_depth, ivec2(gl_FragCoord.xy), 0).r;
    if (zs < 0.99999) {
        vec4 wp = pc.inv_vp * vec4(suv * 2.0 - 1.0, zs, 1.0);
        float ts = distance(wp.xyz / wp.w, ro);
        t1 = min(t1, ts);
    }
    if (t1 <= t0) discard;

    // Lighting terms.
    vec3  L    = normalize(scene.sun_direction.xyz);
    vec3  Lu   = normalize(L / S) ;           // sun dir in unit space (approx)
    vec3  sun  = scene.sun_color.rgb * scene.sun_color.a;
    vec3  sky  = scene.sky_color.rgb;
    vec3  albedo = (g_kind == 2) ? vec3(0.55, 0.47, 0.36)       // dust
                 : (g_kind == 1) ? vec3(0.24, 0.225, 0.21)       // soot smoke
                 :                 vec3(0.22, 0.18, 0.16);      // fireball soot
    // Forward-scattering phase (Henyey-Greenstein g = 0.35): smoke glows
    // when back-lit by the sun.
    float mu = dot(rd, L);
    float g = 0.35;
    float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5) * 0.25;

    const int N = 40;
    float span = t1 - t0;
    float dt = span / float(N);
    // Interleaved-gradient-noise jitter, frame-stable: a per-frame offset
    // made drifting smoke shimmer under TAA history rejection.
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy +
                       vec2(0.0), vec2(0.06711056, 0.00583715))));
    float t = t0 + dt * ign;
    float T = 1.0;
    vec3  acc = vec3(0.0);
    // Extinction per metre at density 1, scaled so a volume reads solid
    // at its core but wispy at the edges regardless of its size.
    float sigma = dens_scale * 6.0 / Rad;
    for (int i = 0; i < N; ++i) {
        if (t > t1 || T < 0.01) break;
        vec3 p = ro + rd * t;
        vec3 u = (p - C) / S;
        float rho = density(u, 5);
        if (rho > 0.002) {
            // Self-shadow toward the sun: one coarse density tap.
            // Two taps toward the sun -> defined billows (lit tops, dark
            // creases); "powder" darkens thin sunward edges (Schneider 2015).
            float rs = density(u + Lu * 0.16, 2) + density(u + Lu * 0.42, 2);
            float sh = exp(-rs * 2.4);
            float powder = 1.0 - exp(-rho * 5.0);
            float skyw = 0.28 + 0.4 * clamp(u.y * 0.5 + 0.5, 0.0, 1.0);
            vec3 lit = albedo * (sun * (sh * mix(0.55, 1.0, powder) * (0.7 + hg * 2.0)) +
                                 sky * skyw + 0.015);
            vec3 emit = vec3(0.0);
            if (heat > 0.001) {
                // Hotter toward the core; billow tops cool first.
                float core = clamp(1.0 - length(u) * 1.05 + rho * 0.6, 0.0, 1.0);
                float temp = heat * core * (0.75 + 0.5 * vnoise(u * 5.0 + g_age * 2.0));
                emit = blackbody(temp);
            }
            float ext = rho * sigma * dt;
            float Ts  = exp(-ext);
            acc += T * (1.0 - Ts) * (lit + emit);
            T *= Ts;
        }
        t += dt;
    }
    float alpha = (1.0 - T) * fade;
    outColor = vec4(acc * fade, alpha);
}
