// Ray-marched rocks. A rock instance is rasterised only as a bounding
// hull; the surface itself is an implicit function sphere-traced per pixel
// from the hull's entry point, so its detail is limited by the pixel
// footprint, not by a triangle count.
//
// The base shape is the SAME function world_props.cpp::add_blob uses for
// the rock meshes (which stay in use for ray-traced shadows and
// collision): radius along a direction = 1 + amp * (fbm - 0.5) * 2, with
// optional sedimentary strata. On top of it come finer octaves, cracks
// and pitting that no mesh LOD could carry.
// Object space: unit rock centred at kRockC, y up.
const vec3 kRockC = vec3(0.0, 0.35, 0.0);

float rk_h3(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float rk_vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(rk_h3(i), rk_h3(i + vec3(1, 0, 0)), f.x),
                   mix(rk_h3(i + vec3(0, 1, 0)), rk_h3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(rk_h3(i + vec3(0, 0, 1)), rk_h3(i + vec3(1, 0, 1)), f.x),
                   mix(rk_h3(i + vec3(0, 1, 1)), rk_h3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float rk_fbm(vec3 p, int oct) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < oct; ++i) { s += a * rk_vnoise(p); p = p * 2.03 + vec3(5.1, 1.7, 9.3); a *= 0.5; }
    return s;
}
// Per-archetype parameters from the rock variant index (world_props.cpp
// builds the meshes from the same formulas).
void rock_params(float v, out float seed, out float amp, out float strata) {
    seed = 3.7 + 6.1 * v;
    amp = 0.26 + 0.1 * fract(v * 0.37);
    strata = (mod(v, 2.0) < 0.5) ? 0.55 : 0.0;
}
// Signed "distance" (not exact - march with a step factor). `fine` 0..1
// fades the small-scale detail out with distance; `cav` returns how deep
// in a crack / pit the point is (for shading).
float rock_sdf(vec3 p, float seed, float amp, float strata, float fine, out float cav) {
    vec3 q = p - kRockC;
    float l = length(q);
    vec3 d = q / max(l, 1e-4);
    float n = rk_fbm(d * 1.6 + vec3(seed, seed * 1.3, -seed), 4);
    vec3 s = d * (1.0 + amp * (n - 0.5) * 2.0);
    if (strata > 0.0) {
        float yq = floor(s.y * 3.5 + 0.5) / 3.5;
        s.y += (yq - s.y) * strata;
    }
    float rho = length(s);
    // Mid-scale lumps and facets.
    float m = rk_fbm(p * 4.3 + seed, 3);
    float det = (m - 0.5) * 0.11;
    cav = 0.0;
    if (fine > 0.01) {
        // Cracks: thin ridged valleys. Pitting: fine grain.
        float cr = abs(rk_fbm(p * 7.1 - seed, 3) - 0.5);
        float crack = 1.0 - smoothstep(0.0, 0.035, cr);
        float grain = rk_fbm(p * 31.0 + seed * 2.0, 3) - 0.5;
        det += fine * (-0.035 * crack + 0.020 * grain);
        cav = fine * (crack * 0.8 + max(-grain, 0.0) * 0.8);
    }
    return l - rho - det;
}
// March from `ro` along `rd` (object space) for at most `t_max`. Returns
// the hit distance or -1.
float rock_march(vec3 ro, vec3 rd, float t_max, float seed, float amp, float strata, float fine, float eps) {
    float t = 0.0, t_prev = 0.0;
    float cav;
    for (int i = 0; i < 56; ++i) {
        float f = rock_sdf(ro + rd * t, seed, amp, strata, fine, cav);
        if (f < eps) {
            if (f < -eps) {             // stepped inside: bisect back to the surface
                float a = t_prev, b = t;
                for (int k = 0; k < 4; ++k) {
                    float mid = 0.5 * (a + b);
                    if (rock_sdf(ro + rd * mid, seed, amp, strata, fine, cav) > 0.0) a = mid; else b = mid;
                }
                return 0.5 * (a + b);
            }
            return t;
        }
        t_prev = t;
        t += max(f * 0.55, eps * 0.5);
        if (t > t_max) break;
    }
    return -1.0;
}
vec3 rock_normal(vec3 p, float seed, float amp, float strata, float fine, float e) {
    float c;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(k.xyy * rock_sdf(p + k.xyy * e, seed, amp, strata, fine, c) +
                     k.yyx * rock_sdf(p + k.yyx * e, seed, amp, strata, fine, c) +
                     k.yxy * rock_sdf(p + k.yxy * e, seed, amp, strata, fine, c) +
                     k.xxx * rock_sdf(p + k.xxx * e, seed, amp, strata, fine, c));
}
