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
