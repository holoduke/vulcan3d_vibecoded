// Leafy silhouettes for canopy / bush meshes (vertex material 2 / 3).
// The crowns are built from smooth displaced blobs; a leaf-scale 3D noise
// eats holes into each blob where its surface turns edge-on to the viewer,
// so outlines break up into leaf clusters with sky showing through instead
// of reading as smooth balls. Shared by cube.frag (colour / G-buffer) and
// veg_depth.frag (depth prepass) - both MUST take the same decision or the
// prepass depth would block the pixels behind the holes.
//   rp   rest position (world space BEFORE wind sway, so the pattern is
//        glued to the crown and does not swim while it moves)
//   wp   actual world position, n = interpolated normal, cam = camera pos
float veg_leaf_h(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}
float veg_leaf_n(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(veg_leaf_h(i), veg_leaf_h(i + vec3(1, 0, 0)), f.x),
                   mix(veg_leaf_h(i + vec3(0, 1, 0)), veg_leaf_h(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(veg_leaf_h(i + vec3(0, 0, 1)), veg_leaf_h(i + vec3(1, 0, 1)), f.x),
                   mix(veg_leaf_h(i + vec3(0, 1, 1)), veg_leaf_h(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
bool veg_leaf_cut(vec3 rp, vec3 wp, vec3 n, vec3 cam) {
    vec3  to_cam = cam - wp;
    float dist = length(to_cam);
    // Holes are a few cm: beyond ~60 m they are sub-pixel shimmer.
    float near_k = 1.0 - smoothstep(38.0, 62.0, dist);
    if (near_k <= 0.0) return false;
    float facing = abs(dot(normalize(n), to_cam / max(dist, 1e-3)));
    float edge = 1.0 - smoothstep(0.05, 0.62, facing);        // 1 at the silhouette
    if (edge <= 0.0) return false;
    float pat = veg_leaf_n(rp * 4.3) * 0.6 + veg_leaf_n(rp * 9.7 + 5.1) * 0.4;
    return pat < edge * 0.66 * near_k;
}
