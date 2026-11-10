#version 460
// FX ribbons / billboards, premultiplied-alpha output (blend ONE,
// ONE_MINUS_SRC_ALPHA): alpha = 0 output is purely additive (glows).
//   pc.color.x = kind: 0 smoke trail, 1 glow beam (laser / afterburner),
//                      2 radial flash billboard
//   pc.color.y = global intensity, pc.color.z = time (s)

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vTint;
layout(location = 2) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
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

void main() {
    int   kind  = int(pc.color.x + 0.5);
    float inten = pc.color.y;
    float time  = pc.color.z;
    if (kind == 0) {
        // Smoke / vapour trail: soft gaussian cross-section, broken up by
        // drifting noise so it reads as billowing vapour, lit by sun + sky.
        float a = vUv.x * exp(-vUv.y * vUv.y * 2.2);
        vec3  q = vWorld * 0.22 + vec3(0.0, time * 0.25, time * 0.1);
        float n = vnoise(q) * 0.65 + vnoise(q * 2.7) * 0.35;
        a *= smoothstep(0.18, 0.75, n + 0.25 * (1.0 - abs(vUv.y)));
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        vec3 lit = vTint * (sun * 0.75 + scene.sky_color.rgb * 0.55 + 0.05);
        a = clamp(a * inten, 0.0, 1.0);
        outColor = vec4(lit * a, a);
    } else if (kind == 1) {
        // Glow beam: white-hot core + coloured halo, additive.
        float v = vUv.y;
        float core = exp(-v * v * 26.0);
        float halo = exp(-v * v * 3.0);
        // Strongly saturated tints (red AA tracers) keep a coloured core.
        vec3  corec = (vTint.g < 0.1) ? vec3(1.0, 0.35, 0.25) : vec3(1.0, 0.95, 1.0);
        vec3  c = vTint * (halo * 1.4) + corec * core * 3.2;
        outColor = vec4(c * vUv.x * inten, 0.0);
    } else if (kind == 3) {
        // Cloth flag: flat normal from screen derivatives, two-sided
        // Lambert + sun shining THROUGH the cloth when back-lit (thin
        // fabric translucency) + sky fill. Procedural heraldry.
        vec3 Nf = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        vec3 V  = normalize(scene.camera_pos.xyz - vWorld);
        if (dot(Nf, V) < 0.0) Nf = -Nf;
        vec3 L  = normalize(scene.sun_direction.xyz);
        float ndl = dot(Nf, L);
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        int design = int(vTint.x + 0.5);
        vec2 uv = vUv;
        vec3 base, accent;
        if (design == 0) { base = vec3(0.42, 0.03, 0.03); accent = vec3(0.85, 0.62, 0.12); }
        else             { base = vec3(0.04, 0.10, 0.36); accent = vec3(0.86, 0.84, 0.78); }
        vec3 alb = base;
        if (design == 0) {
            // Gold cross (Nordic offset) + fly-end swallowtail fringe.
            float cx = step(abs(uv.x - 0.36), 0.07), cy = step(abs(uv.y - 0.5), 0.09);
            alb = mix(alb, accent, max(cx, cy));
        } else {
            // Bend + diamond charge.
            float bend = step(abs((uv.x - uv.y * 1.6) + 0.3), 0.12);
            float dia  = step(abs(uv.x - 0.6) * 1.6 + abs(uv.y - 0.5), 0.22);
            alb = mix(alb, accent, max(bend, dia));
        }
        // Woven-fabric grain + soft dirt toward the fly end.
        float weave = 0.9 + 0.1 * sin(uv.x * 420.0) * sin(uv.y * 260.0);
        alb *= weave * mix(1.0, 0.8, uv.x * uv.x);
        vec3 lit = alb * (sun * max(ndl, 0.0) * 1.05 +
                          sun * max(-ndl, 0.0) * 0.45 * (0.6 + 0.4 * base) +   // translucency
                          scene.sky_color.rgb * 0.45 + 0.03);
        outColor = vec4(lit, 1.0);
    } else {
        // Radial flash billboard (uv in [-1,1]^2), additive.
        float r2 = dot(vUv, vUv);
        float f = (exp(-r2 * 5.0) + 0.35 * exp(-r2 * 1.4)) * (1.0 - smoothstep(0.55, 1.0, sqrt(r2)));
        outColor = vec4(vTint * f * inten, 0.0);
    }
}
