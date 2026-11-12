// Leaf cut-out for canopy / bush meshes (vertex material 2 / 3).
// The crowns are built from displaced blobs. A 3D cellular pattern places
// one elongated "leaf" per ~15 cm cell; where the blob surface passes
// between leaves the pixel is discarded, so a crown reads as a mass of
// separate leaves with gaps (deeper foliage or sky showing through)
// instead of a smooth ball. Shared by cube.frag (colour / G-buffer) and
// veg_depth.frag (depth prepass) - both MUST take the same decision or the
// prepass depth would block the pixels behind the holes.
//   rp   rest position (world space BEFORE wind sway, so the pattern is
//        glued to the crown and does not swim while it moves)
//   wp   actual world position, n = interpolated normal, cam = camera pos
const float kVegLeafCells = 6.5;        // cells per metre
float veg_leaf_h(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}
vec3 veg_leaf_h3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}
// Nearest leaf: returns its normalised distance (< 1 = on the leaf) and
// its per-leaf random triple in `id`. 2x2x2 cell search.
float veg_leaf_cell(vec3 rp, out vec3 id) {
    vec3 p = rp * kVegLeafCells;
    vec3 ip = floor(p - 0.5);
    float best = 1e9;
    id = vec3(0.5);
    for (int k = 0; k < 8; ++k) {
        vec3 c = ip + vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
        vec3 h = veg_leaf_h3(c);
        vec3 v = p - (c + 0.5 + (h - 0.5) * 0.7);
        vec3 a = normalize(h.yzx - 0.5 + vec3(0.0, 1e-3, 0.0));   // long axis of the leaf
        float av = dot(v, a);
        float r  = 0.40 + 0.20 * h.z;
        float d  = (dot(v, v) - 0.62 * av * av) / (r * r);
        if (d < best) { best = d; id = h; }
    }
    return best;
}
bool veg_leaf_cut(vec3 rp, vec3 wp, vec3 n, vec3 cam) {
    vec3  to_cam = cam - wp;
    float dist = length(to_cam);
    // Leaves are ~10 cm: beyond ~60 m the gaps are sub-pixel shimmer, so
    // they close up (the leaves grow until they merge).
    float near_k = 1.0 - smoothstep(34.0, 62.0, dist);
    if (near_k <= 0.0) return false;
    // More open toward the silhouette, denser where the surface faces the
    // viewer (there the gaps would only show the inside of the same clump).
    float facing = abs(dot(normalize(n), to_cam / max(dist, 1e-3)));
    float open = mix(1.0, 0.72, smoothstep(0.25, 0.9, facing));
    vec3 id;
    float d = veg_leaf_cell(rp, id);
    return d * open * near_k * near_k > 1.0;
}

// Leaf card (vertex material 7, the EZ-Tree style leaf quads). `uv` runs
// 0..1 across and 0 (attachment) .. 1 (tip) along the card; floor(kind) is
// 0 for a broadleaf spray, 1 for a needle spray. Returns false where the
// card is empty. `shade`: -1 on the twig, else 0 (leaf edge) .. 1 (midrib).
bool veg_card(vec2 uv, float kind, out float shade) {
    vec2 p = vec2(uv.x - 0.5, uv.y);
    shade = 0.5;
    bool twig = abs(p.x) < 0.012 && p.y < 0.84;
    if (kind < 1.0) {
        // Eight leaves alternating along a twig, plus a terminal one.
        float best = 1e9;
        for (int i = 0; i < 8; ++i) {
            float fi = float(i);
            bool  term = (i == 7);
            float side = ((i & 1) == 0) ? 1.0 : -1.0;
            float ang  = term ? 0.0 : side * (0.98 - 0.05 * fi);
            vec2  base = vec2(0.0, term ? 0.78 : 0.08 + 0.10 * fi);
            float len  = term ? 0.22 : 0.42 - 0.022 * fi;
            vec2  d = vec2(sin(ang), cos(ang));
            vec2  q = p - base;
            float l = dot(q, d) / len;
            if (l <= 0.0 || l >= 1.0) continue;
            float w = (q.x * d.y - q.y * d.x) / len;
            float prof = 0.27 * sin(3.14159 * pow(l, 0.72));
            best = min(best, abs(w) / max(prof, 1e-4));
        }
        if (best < 1.0) { shade = 1.0 - best; return true; }
    } else {
        // Needles: two fans of thin blades along the twig. They widen with
        // the pixel footprint so a distant spray closes up instead of
        // dissolving into sub-pixel shimmer.
        const float a = 0.62, P = 0.055;
        float hw = 0.011 + 0.8 * fwidth(uv.y);
        for (int s = 0; s < 2; ++s) {
            float sg = (s == 0) ? 1.0 : -1.0;
            float n = p.x / (sg * sin(a));
            if (n <= 0.0) continue;
            float t = p.y - n * cos(a);
            float len = 0.36 * (1.0 - 0.55 * smoothstep(0.45, 0.86, t));
            if (t < 0.02 || t > 0.86 || n > len) continue;
            float d = abs(fract(t / P) - 0.5) * P * sin(a);
            if (d < hw) { shade = 1.0 - d / hw; return true; }
        }
    }
    if (twig) { shade = -1.0; return true; }
    return false;
}

// Leaf card from the foliage atlas (albedo slot 15: broadleaf twig in the
// left half, conifer spray in the right). `uv` as for veg_card. Returns
// rgb + coverage; `lod_bias` < 0 keeps thin needles from fading with the mips.
vec2 veg_card_uv(vec2 uv, float kind) {
    // 2 x 2 cells: 0 oak, 1 spruce (top row), 2 birch, 3 pine (bottom row).
    float cell = floor(kind);
    return vec2(mod(cell, 2.0) * 0.5 + clamp(uv.x, 0.006, 0.994) * 0.5,
                (cell >= 2.0 ? 0.5 : 0.0) + (1.0 - clamp(uv.y, 0.006, 0.994)) * 0.5);
}
