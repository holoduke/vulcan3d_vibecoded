#version 460

// SVGF a-trous filter (4a-deep), one of three chained passes.
//
// Cascade: stride 1 (3x3 of 5x5 reach), stride 2 (5x5 of 9x9 reach),
// stride 4 (9x9 of 17x17 reach). Total spatial reach is 17x17 pixels
// after 3 passes -- comparable to a single large Gaussian but with the
// SVGF edge-stop weights gating each tap independently against:
//
//   w_l = exp(-|L_q - L_p| / (sigma_l * sqrt(var_p) + eps))
//         -- luminance weight. variance comes from binding 2 moments
//         (mu, mu2); sqrt(var) scales the falloff so high-variance
//         pixels accept loosely-matching neighbours and low-variance
//         pixels clamp tight.
//
//   w_z = exp(-|z_q - z_p| / (sigma_z * max(0.01, |grad z * delta|)))
//         -- depth weight. grad_z is the screen-space depth gradient
//         estimated from the centre pixel's depth derivatives, so flat
//         surfaces tolerate small absolute z differences while steep
//         surfaces reject more aggressively.
//
//   w_n = max(0, dot(N_p, N_q))^sigma_n
//         -- normal weight. N is geometric, reconstructed from depth
//         derivatives (cheap proxy for the real surface normal; SVGF
//         reference uses a g-buffer normal but we don't have one).
//
// Final tap weight = w_l * w_z * w_n * b3_spline(i, j) where b3_spline
// is the [1, 4, 6, 4, 1] / 16 separable kernel.
//
// Output: filtered colour. The framework wires this so a future read of
// scene_color picks up the denoised result.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_depth;
layout(set = 0, binding = 2) uniform sampler2D u_moments;

layout(set = 0, binding = 3) uniform AtrousUBO {
    vec4 viewport;   // x: w, y: h, z: 1/w, w: 1/h
    vec4 params;     // x: stride (1,2,4),
                     //  y: sigma_l (luminance),
                     //  z: sigma_z (depth),
                     //  w: sigma_n (normal exponent)
} at;

const float kB3[5] = float[](1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0);

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Geometric normal from depth derivatives. Cheap proxy for the real
// surface normal -- accurate on planar surfaces, slightly off at
// silhouettes (where the derivatives blend depth across the edge).
// Since the normal weight gates exactly those silhouette taps, a
// slightly-off normal at the silhouette is fine: dot(N_self, N_other)
// stays well below 1 either way, so the tap gets rejected.
//
// Returns the screen-space normal in view space, but we only use it as
// a unit vector for the dot-product weight, so the basis doesn't
// matter.
vec3 normal_from_depth(ivec2 ip) {
    float z  = texelFetch(u_depth, ip,                    0).r;
    float zx = texelFetch(u_depth, ip + ivec2(1, 0),      0).r;
    float zy = texelFetch(u_depth, ip + ivec2(0, 1),      0).r;
    // Build two tangent vectors in pixel-space (x, y, dz/dxy) and take
    // their cross product. Pixel scale is 1 in xy and (zx - z) in z
    // for the x-tangent (same for y); normalising kills the absolute
    // scale. dFdx-style operation but without using derivatives (which
    // are quad-uniform and can read wrong values at quad edges in
    // sparse-fragment paths).
    vec3 dx = vec3(1.0, 0.0, zx - z);
    vec3 dy = vec3(0.0, 1.0, zy - z);
    vec3 n  = normalize(cross(dx, dy));
    // Camera looks down +z so the surface normal usually has -z (faces
    // camera). normalize handles sign in either direction; the
    // edge-stop only cares about |dot| effectively because we apply
    // max(0, dot)^sigma_n -- but flipping signs across silhouettes
    // gives 0 weight, which is what we want.
    return n;
}

void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    ivec2 isz = ivec2(at.viewport.xy);
    const int stride = int(at.params.x);
    const float sigma_l = at.params.y;
    const float sigma_z = at.params.z;
    const float sigma_n = at.params.w;

    vec3  c0 = texelFetch(u_color,  ip, 0).rgb;
    float z0 = texelFetch(u_depth,  ip, 0).r;
    vec2  m0 = texelFetch(u_moments,ip, 0).rg;
    float var0 = max(0.0, m0.y - m0.x * m0.x);
    // sqrt(variance) with a small floor: the floor stops a perfectly
    // converged stable pixel (var=0) from collapsing the luminance
    // weight's denominator to zero (which would force exp(-inf) for
    // any non-zero delta and reject all neighbours -- exactly the
    // wrong thing). 0.01 is the SVGF reference epsilon.
    float sd0  = sqrt(var0) + 0.01;

    vec3  n0 = normal_from_depth(ip);

    // Screen-space depth gradient used by the depth edge-stop. Pixel-
    // scale; for stride>1 we scale up so the gate matches the actual
    // sample spacing.
    float dzdx = texelFetch(u_depth, ip + ivec2(1, 0), 0).r - z0;
    float dzdy = texelFetch(u_depth, ip + ivec2(0, 1), 0).r - z0;
    // Sky pixels (z = 1.0 with default depth clear) have zero local
    // gradient and self-occlude all neighbours unless we let them
    // accept any tap. We can short-circuit: if z0 >= 0.9999 the pixel
    // is sky -- there's nothing to denoise (sky is procedural and
    // already smooth in compose.frag). Bypass the filter entirely.
    if (z0 >= 0.9999) {
        outColor = vec4(c0, 1.0);
        return;
    }

    vec3  sum  = vec3(0.0);
    float wsum = 0.0;

    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            ivec2 off = ivec2(i, j) * stride;
            ivec2 p   = ip + off;
            if (p.x < 0 || p.y < 0 || p.x >= isz.x || p.y >= isz.y) continue;

            vec3  cq = texelFetch(u_color, p, 0).rgb;
            float zq = texelFetch(u_depth, p, 0).r;
            // Sky neighbours get rejected via the depth gate below
            // (z0 < 1 but zq == 1 -> huge dz, w_z -> 0).

            // --- luminance weight ---
            float l_diff = abs(lum(cq) - m0.x);
            // m0.x is the temporally-smoothed mean luminance; using
            // the smoothed mean (vs c0's instantaneous luminance) is
            // more robust to single-frame fireflies polluting the
            // gate. Standard SVGF formulation.
            float w_l = exp(-l_diff / (sigma_l * sd0 + 1e-4));

            // --- depth weight ---
            // |grad z * delta| in pixel-space. delta is `off` (pixel
            // offset including stride); take the L1 norm via the
            // gradient projection. Add a small constant so flat-z
            // surfaces don't blow up the denominator and snap-cut all
            // taps.
            float grad_proj = abs(dzdx * float(off.x) +
                                  dzdy * float(off.y));
            float z_denom = sigma_z * max(0.01, grad_proj);
            float w_z = exp(-abs(zq - z0) / z_denom);

            // --- normal weight ---
            vec3  nq = normal_from_depth(p);
            float ndot = max(0.0, dot(n0, nq));
            float w_n = pow(ndot, sigma_n);

            // --- kernel weight ---
            float w_k = kB3[i + 2] * kB3[j + 2];

            float w = w_k * w_l * w_z * w_n;
            sum  += cq * w;
            wsum += w;
        }
    }

    // wsum can underflow on truly isolated pixels (e.g. a single
    // foreground pixel surrounded by sky-depth neighbours); fall back
    // to the centre tap rather than dividing by zero.
    vec3 filtered = (wsum > 1e-5) ? sum / wsum : c0;
    outColor = vec4(filtered, 1.0);
}
