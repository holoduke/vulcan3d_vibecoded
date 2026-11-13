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
    // Every third variant is a SLAB formation (strata < 0 selects it).
    if (mod(v, 3.0) > 0.5 && mod(v, 3.0) < 1.5) strata = -1.0;
}
// Mul/add-only hash (Hoskins hash11): the CPU twin in world_props.cpp
// must give the same beds, and sin() of these arguments differs between
// CPU and GPU by enough to move a bed edge (collision vs drawn rock).
float rk_h1(float x) { float p = fract(x * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
// Slab formation: three or four flat limestone beds stacked like worn
// steps - each an irregular polygon in plan, a little smaller and shifted
// against the one below, with flat tops and near-vertical broken edges.
// The bed layout depends only on the seed: it is built once per pixel
// (first call) instead of re-hashing ~100 values every march step.
float g_slab_seed = -1e30;
int   g_slab_n = 0;
vec4  g_slab_bed[4];    // off.xy, yc, th
vec3  g_slab_edge[20];  // edge normal xy, distance
void slab_setup(float seed) {
    g_slab_seed = seed;
    g_slab_n = 4;
    for (int i = 0; i < 4; ++i) {
        float fi = float(i);
        float h = rk_h1(seed + fi * 7.3);
        if (i == 3 && h < 0.45) { g_slab_n = 3; break; }       // some have only three beds
        float th = 0.085 + 0.04 * rk_h1(seed + fi * 2.9);      // half thickness
        float yc = -0.27 + 0.215 * fi;
        float r  = 1.08 - 0.20 * fi - 0.10 * h;
        vec2  off = (vec2(rk_h1(seed + fi * 3.1), rk_h1(seed + fi * 5.7)) - 0.5) * 0.30 * fi;
        g_slab_bed[i] = vec4(off, yc, th);
        float a0 = h * 6.2832;
        for (int k = 0; k < 5; ++k) {
            float a = a0 + float(k) * 1.2566 + (rk_h1(seed + fi + float(k) * 1.7) - 0.5) * 0.5;
            g_slab_edge[i * 5 + k] = vec3(cos(a), sin(a), r * (0.72 + 0.28 * rk_h1(seed + fi * 4.1 + float(k))));
        }
    }
}
float rock_slabs(vec3 q, float seed) {
    if (seed != g_slab_seed) slab_setup(seed);
    float d = 1e9;
    for (int i = 0; i < 4; ++i) {
        if (i >= g_slab_n) break;
        vec4 bd = g_slab_bed[i];
        vec2 xz = q.xz - bd.xy;
        float dp = -1e9;
        for (int k = 0; k < 5; ++k) {
            vec3 e = g_slab_edge[i * 5 + k];
            dp = max(dp, dot(xz, e.xy) - e.z);
        }
        vec2 w = vec2(dp, abs(q.y - bd.z) - bd.w);
        d = min(d, min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - 0.018);
    }
    return d;
}

// Signed "distance" (not exact - march with a step factor). `fine` 0..1
// fades the small-scale detail out with distance; `cav` returns how deep
// in a crack / pit the point is (for shading).
float rock_sdf(vec3 p, float seed, float amp, float strata, float fine, out float cav) {
    vec3 q = p - kRockC;
    if (strata < 0.0) {
        // Slab formation: beds + weathering. Grikes (the straight-ish
        // fissures of a limestone pavement) cut down into the flat tops.
        float d = rock_slabs(q, seed);
        float m = rk_fbm(p * 3.1 + seed, 3);
        float gr = abs(rk_vnoise(vec3(p.xz * 2.3 + seed, 0.0)) - 0.5);
        float grike = 1.0 - smoothstep(0.0, 0.045, gr);
        float det = (m - 0.5) * 0.045 - 0.07 * grike;
        cav = grike * 0.9;
        if (fine > 0.01) {
            float cr = abs(rk_fbm(p * 9.0 - seed, 3) - 0.5);
            float crack = 1.0 - smoothstep(0.0, 0.03, cr);
            float grain = rk_fbm(p * 31.0 + seed * 2.0, 3) - 0.5;
            det += fine * (-0.02 * crack + 0.012 * grain);
            cav += fine * (crack * 0.7 + max(-grain, 0.0) * 0.6);
        }
        return d - det;
    }
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
