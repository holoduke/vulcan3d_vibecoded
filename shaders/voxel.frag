#version 460

// Voxel shape fragment shader — brick DDA (multi-shape, rotation-capable).
//
// The push-constant carries a rigid transform (rotation R, translation T)
// from shape-local to world. We use the analytic inverse of a rigid
// transform — transpose(R) for rotation, -T for translation — to map the
// world-space camera ray into shape-local space without a per-fragment
// mat4 inverse(). Multi-shape: the directory entries for THIS shape live
// at `entries[grid_dir.w + dir_i]` in the global directory buffer;
// entries point to slot indices in the global brick atlas (kEmpty =
// empty brick).

layout(location = 0) in vec3 vWorldPos;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
// G-buffer dual writes — same convention cube.frag uses (Phase 2/5 of
// the deferred-render migration). Voxel-shape pixels get material_id 6.
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

vec2 octa_encode_v(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = (n.z >= 0.0)
        ? n.xy
        : ((1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                     n.y >= 0.0 ? 1.0 : -1.0));
    return e * 0.5 + 0.5;
}

layout(push_constant) uniform PC {
    mat3  R;              // shape-local→world rotation (48 bytes, std430)
    vec4  T;              // .xyz = translation, .w pad
    vec4  dims_vs;        // xyz = shape extent (local), w = voxel size
    ivec4 grid_dir;       // xyz = brick dims, w = directory base offset
} pc;

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 view_proj;
    mat4 prev_view_proj;
    vec4 camera_pos;
    vec4 sun_dir;
    vec4 sun_color;
    vec4 ambient;
    vec4 viewport;
    vec4 pal[16];
} cam;

struct BrickPayload {
    uint occ[128];
    uint pal[1024];
};

layout(set = 0, binding = 1, std430) readonly buffer BrickAtlas {
    BrickPayload bricks[];
};

// Flat global directory — all shapes' directory entries concatenated.
// Each shape starts at its own `dir_base` offset (`pc.grid_dir.w`).
layout(set = 0, binding = 2, std430) readonly buffer ShapeDir {
    uint entries[];
};

// Shared wall texture — the same Bricks078 albedo the castle brushes use,
// triplanar-sampled in world space so the brick layout reads consistently
// across faces regardless of voxel orientation.
layout(set = 0, binding = 3) uniform sampler2D u_wall_tex;

vec3 triplanar_sample(sampler2D tex, vec3 wp, vec3 N) {
    vec3 blend = abs(N); blend *= blend; blend *= blend;   // pow(|N|, 4)
    blend /= max(blend.x + blend.y + blend.z, 1e-3);
    // Voxel faces are axis-aligned — the dominant axis carries ≥0.8 of the
    // blend on >95% of pixels. Drop the threshold from 0.95 to 0.8 so the
    // single-sample fast path fires on the entire tower interior and most
    // of the chunks, saving 2 texture fetches per pixel.
    if (blend.x >= 0.8) return texture(tex, wp.zy).rgb;
    if (blend.y >= 0.8) return texture(tex, wp.xz).rgb;
    if (blend.z >= 0.8) return texture(tex, wp.xy).rgb;
    vec3 cx = texture(tex, wp.zy).rgb;
    vec3 cy = texture(tex, wp.xz).rgb;
    vec3 cz = texture(tex, wp.xy).rgb;
    return cx * blend.x + cy * blend.y + cz * blend.z;
}

const uint kEmpty = 0xFFFFFFFFu;
const int  kBrickDim = 16;

bool ray_aabb(vec3 lo, vec3 ld, vec3 bmin, vec3 bmax, out float t0, out float t1) {
    vec3 invd = 1.0 / ld;
    vec3 a = (bmin - lo) * invd;
    vec3 b = (bmax - lo) * invd;
    vec3 lov = min(a, b);
    vec3 hiv = max(a, b);
    t0 = max(max(lov.x, lov.y), lov.z);
    t1 = min(min(hiv.x, hiv.y), hiv.z);
    return t1 > max(t0, 0.0);
}

// Any-hit brick DDA in SHAPE-LOCAL space. Returns true if any solid
// voxel is reached between `lo` and the AABB exit. Used for the sun
// self-shadow ray so wall pixels under the upper turret roof, under a
// merlon, or in the lee of the base battlement read as proper shadowed
// stone instead of flat ambient.
bool vox_self_any_hit(vec3 lo, vec3 ld, int dir_base, ivec3 brick_dim,
                      ivec3 voxel_dim, vec3 shape_max, float vs, float bs) {
    float t0, t1;
    if (!ray_aabb(lo, ld, vec3(0.0), shape_max, t0, t1)) return false;
    float t = max(t0, 0.0) + 1e-4;
    if (t >= t1) return false;

    vec3  pe = lo + t * ld;
    ivec3 vc = clamp(ivec3(floor(pe / vs)), ivec3(0), voxel_dim - ivec3(1));
    ivec3 stp;
    stp.x = ld.x >= 0.0 ? 1 : -1;
    stp.y = ld.y >= 0.0 ? 1 : -1;
    stp.z = ld.z >= 0.0 ? 1 : -1;
    vec3 abs_inv = vec3(
        (abs(ld.x) > 1e-8) ? 1.0 / abs(ld.x) : 1e30,
        (abs(ld.y) > 1e-8) ? 1.0 / abs(ld.y) : 1e30,
        (abs(ld.z) > 1e-8) ? 1.0 / abs(ld.z) : 1e30);
    vec3 tDelta = vs * abs_inv;
    vec3 face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
    vec3 tMax;
    tMax.x = (ld.x != 0.0) ? (face.x - lo.x) / ld.x : 1e30;
    tMax.y = (ld.y != 0.0) ? (face.y - lo.y) / ld.y : 1e30;
    tMax.z = (ld.z != 0.0) ? (face.z - lo.z) / ld.z : 1e30;
    // Tight cap — shadow rays through a tower at ~1 m wall thickness
    // and ~16 m diagonal need ≤64 brick-skipping steps in the worst
    // case. Shadow misses fail-open (no shadow) on the rare grazing
    // ray that runs out of budget — reads as the pre-shadow look.
    const int kShadowMaxSteps = 64;
    for (int i = 0; i < kShadowMaxSteps; ++i) {
        if (any(lessThan(vc, ivec3(0))) ||
            any(greaterThanEqual(vc, voxel_dim))) return false;
        ivec3 bc = vc >> 4;
        int   dir_i = (bc.z * brick_dim.y + bc.y) * brick_dim.x + bc.x;
        uint  bp = entries[dir_base + dir_i];
        if (bp == kEmpty) {
            vec3 brickFace = (vec3(bc) + max(vec3(stp), 0.0)) * bs;
            vec3 bMax = vec3(
                (ld.x != 0.0) ? (brickFace.x - lo.x) / ld.x : 1e30,
                (ld.y != 0.0) ? (brickFace.y - lo.y) / ld.y : 1e30,
                (ld.z != 0.0) ? (brickFace.z - lo.z) / ld.z : 1e30);
            t = min(min(bMax.x, bMax.y), bMax.z) + 1e-4;
            vec3 pn = lo + t * ld;
            vc = ivec3(floor(pn / vs));
            face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
            tMax.x = (ld.x != 0.0) ? (face.x - lo.x) / ld.x : 1e30;
            tMax.y = (ld.y != 0.0) ? (face.y - lo.y) / ld.y : 1e30;
            tMax.z = (ld.z != 0.0) ? (face.z - lo.z) / ld.z : 1e30;
            continue;
        }
        ivec3 lv = vc & ivec3(15);
        int   li = (lv.z * kBrickDim * kBrickDim) + (lv.y * kBrickDim) + lv.x;
        if ((bricks[bp].occ[li >> 5] & (1u << (li & 31))) != 0u) return true;
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            t = tMax.x; vc.x += stp.x; tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            t = tMax.y; vc.y += stp.y; tMax.y += tDelta.y;
        } else {
            t = tMax.z; vc.z += stp.z; tMax.z += tDelta.z;
        }
    }
    return false;
}

void main() {
    // World-space ray from camera through this fragment.
    vec3 ro_world = cam.camera_pos.xyz;
    vec3 rd_world = normalize(vWorldPos - ro_world);

    // Transform ray into SHAPE-LOCAL space. Analytic rigid-inverse:
    // R is orthonormal → R^-1 = transpose(R); translation flips sign.
    // ~20 flops vs ~60+ for a general mat4 inverse — and avoids the
    // div-by-determinant the generic inverse compiles to.
    mat3 Rt = transpose(pc.R);
    vec3 lo = Rt * (ro_world - pc.T.xyz);
    vec3 ld = Rt * rd_world;
    // Pure rotation preserves length → no renormalize needed.

    const float vs = pc.dims_vs.w;
    const float bs = vs * 16.0;
    const ivec3 brick_dim = pc.grid_dir.xyz;
    const int   dir_base  = pc.grid_dir.w;
    const ivec3 voxel_dim = brick_dim * kBrickDim;
    const vec3  shape_max = pc.dims_vs.xyz;

    float t0, t1;
    if (!ray_aabb(lo, ld, vec3(0.0), shape_max, t0, t1)) discard;
    float t = max(t0, 0.0) + 1e-4;

    // ---- Voxel-DDA setup (in shape-local coords) ----
    vec3  pe = lo + t * ld;
    ivec3 vc = clamp(ivec3(floor(pe / vs)), ivec3(0), voxel_dim - ivec3(1));
    ivec3 stp;
    stp.x = ld.x >= 0.0 ? 1 : -1;
    stp.y = ld.y >= 0.0 ? 1 : -1;
    stp.z = ld.z >= 0.0 ? 1 : -1;
    vec3 abs_inv = vec3(
        (abs(ld.x) > 1e-8) ? 1.0 / abs(ld.x) : 1e30,
        (abs(ld.y) > 1e-8) ? 1.0 / abs(ld.y) : 1e30,
        (abs(ld.z) > 1e-8) ? 1.0 / abs(ld.z) : 1e30
    );
    vec3 tDelta = vs * abs_inv;
    vec3 face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
    vec3 tMax;
    tMax.x = (ld.x != 0.0) ? t + (face.x - pe.x) / ld.x : 1e30;
    tMax.y = (ld.y != 0.0) ? t + (face.y - pe.y) / ld.y : 1e30;
    tMax.z = (ld.z != 0.0) ? t + (face.z - pe.z) / ld.z : 1e30;

    // Initial face axis = the AABB-entry axis, so an immediate hit gets a
    // sensible face normal.
    vec3 tEntry;
    tEntry.x = (ld.x != 0.0) ? ((stp.x > 0 ? 0.0 : shape_max.x) - lo.x) / ld.x : -1e30;
    tEntry.y = (ld.y != 0.0) ? ((stp.y > 0 ? 0.0 : shape_max.y) - lo.y) / ld.y : -1e30;
    tEntry.z = (ld.z != 0.0) ? ((stp.z > 0 ? 0.0 : shape_max.z) - lo.z) / ld.z : -1e30;
    int last_axis = (tEntry.x > tEntry.y && tEntry.x > tEntry.z) ? 0
                   : (tEntry.y > tEntry.z ? 1 : 2);

    // Tight cap. With brick-level fast-skips (each empty-brick step
    // advances ~16 voxel cells), a ray that threads the entire tower
    // needs ~grid_x+grid_y+grid_z brick steps + at most one filled brick
    // traversed (≤48 cells). 96 covers worst-case tower + all chunks
    // and bounds the grazing-ray-stuck case. Was voxel_dim.x+y+z+32 →
    // up to 544 steps; per-pixel ALU dominated by this loop when close.
    const int max_steps = 96;

    bool  hit = false;
    uint  hit_pal_idx = 0u;
    float t_hit = t;
    int   hit_axis = last_axis;

    for (int i = 0; i < max_steps; ++i) {
        if (any(lessThan(vc, ivec3(0))) ||
            any(greaterThanEqual(vc, voxel_dim))) break;

        ivec3 bc = vc >> 4;
        int   dir_i = (bc.z * brick_dim.y + bc.y) * brick_dim.x + bc.x;
        uint  bp    = entries[dir_base + dir_i];

        if (bp == kEmpty) {
            vec3 brickFace = (vec3(bc) + max(vec3(stp), 0.0)) * bs;
            vec3 bMax;
            bMax.x = (ld.x != 0.0) ? (brickFace.x - lo.x) / ld.x : 1e30;
            bMax.y = (ld.y != 0.0) ? (brickFace.y - lo.y) / ld.y : 1e30;
            bMax.z = (ld.z != 0.0) ? (brickFace.z - lo.z) / ld.z : 1e30;
            float t_skip = min(min(bMax.x, bMax.y), bMax.z);
            t = t_skip + 1e-4;
            vec3 pn = lo + t * ld;
            vc = ivec3(floor(pn / vs));
            face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
            tMax.x = (ld.x != 0.0) ? (face.x - lo.x) / ld.x : 1e30;
            tMax.y = (ld.y != 0.0) ? (face.y - lo.y) / ld.y : 1e30;
            tMax.z = (ld.z != 0.0) ? (face.z - lo.z) / ld.z : 1e30;
            last_axis = (bMax.x < bMax.y && bMax.x < bMax.z) ? 0
                      : (bMax.y < bMax.z ? 1 : 2);
            continue;
        }

        ivec3 lv = vc & ivec3(15);
        int   li = (lv.z * 16 * 16) + (lv.y * 16) + lv.x;
        if ((bricks[bp].occ[li >> 5] & (1u << (li & 31))) != 0u) {
            int word  = li >> 2;
            int shift = (li & 3) * 8;
            hit_pal_idx = (bricks[bp].pal[word] >> shift) & 0xFFu;
            hit = true;
            t_hit = t;
            hit_axis = last_axis;
            break;
        }

        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            t = tMax.x; vc.x += stp.x; tMax.x += tDelta.x; last_axis = 0;
        } else if (tMax.y < tMax.z) {
            t = tMax.y; vc.y += stp.y; tMax.y += tDelta.y; last_axis = 1;
        } else {
            t = tMax.z; vc.z += stp.z; tMax.z += tDelta.z; last_axis = 2;
        }
    }

    if (!hit) discard;

    // ---- Shade the hit, transform back to world ----
    vec3 hit_local = lo + t_hit * ld;
    vec3 hit_world = pc.R * hit_local + pc.T.xyz;

    // Shape-local face normal → world (rotation only). Indexed basis lookup
    // replaces a 3-way ternary chain — single mul, no branches.
    const vec3 kAxisBasis[3] = vec3[3](
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0));
    vec3 N_local = kAxisBasis[hit_axis] * -float(stp[hit_axis]);
    vec3 N = normalize(pc.R * N_local);

    // Clamp instead of mask: a corrupt palette index now lands on slot 15
    // deterministically (visible magenta sentinel) instead of silently
    // wrapping into a valid slot — makes data-corruption bugs obvious.
    int pal_idx = clamp(int(hit_pal_idx), 0, 15);
    vec3 base_palette = cam.pal[pal_idx].rgb;
    // Modulate the palette tint by the shared wall texture. Period 1 m
    // (texture tiles every metre of world space) so bricks read at a
    // sensible ~5 cm size against the 10 cm voxel grid — much tighter
    // than cube.frag's 0.0625 scale, which is tuned for the castle's
    // larger continuous brushes and reads as stretched on the tower.
    vec3 wall_rgb = triplanar_sample(u_wall_tex, hit_world * 1.0, N);
    vec3 base = base_palette * mix(vec3(1.0), wall_rgb * 1.6, 0.65);
    // Double-sided lighting: when the camera sees a back-face (e.g. inside
    // the hollow tower looking out), flip N so the wall doesn't read pitch
    // black. ndl computed against the eye-facing side.
    float ndl_raw = dot(N, -cam.sun_dir.xyz);
    if (ndl_raw < 0.0) { N = -N; ndl_raw = -ndl_raw; }
    float ndl = max(ndl_raw, 0.0);

    // Self-shadow ray. From the hit point, cast a brick-DDA toward the
    // sun. If any voxel is reached before the shape AABB exit, this
    // pixel sits under merlons, the upper-turret roof eave, or the lee
    // of a wall and reads as a proper shadow instead of flat ambient.
    // Only worth casting if the surface is sun-facing; a back-facing
    // sliver gets ndl==0 and the shadow result wouldn't change the
    // colour anyway.
    float shadow_amt = 0.0;
    if (ndl > 0.0) {
        // Local-space sun direction. Rt = transpose(pc.R) is the same
        // inverse-rotation the primary DDA used to transform the camera
        // ray; reusing it keeps the math consistent.
        vec3 sun_local = Rt * (-cam.sun_dir.xyz);
        // Lift the origin a fraction of a voxel along the local face
        // normal so the ray doesn't re-hit the originating voxel.
        vec3 shadow_origin = hit_local + N_local * (vs * 0.5);
        if (vox_self_any_hit(shadow_origin, sun_local, dir_base, brick_dim,
                             voxel_dim, shape_max, vs, bs)) {
            shadow_amt = 0.92;     // mostly-dark; a hair of ambient still leaks in
        }
    }

    // Honour the .a brightness scalars. Apply self-shadow to the sun
    // term only — ambient continues to fill shadowed crevices.
    vec3 col = base * (cam.ambient.rgb * cam.ambient.a +
                       cam.sun_color.rgb * cam.sun_color.a *
                           ndl * (1.0 - shadow_amt));
    outColor = vec4(col, 1.0);

    // G-buffer write — voxel pixels appear in deferred mode. Albedo is
    // pre-lighting (palette × wall_rgb), normal is the face normal we
    // already shaded with, material_id 6 = voxel.
    outGBuffer0 = vec4(clamp(base, vec3(0.0), vec3(1.0)), 6.0 / 255.0);
    outGBuffer1 = vec4(octa_encode_v(N), 0.5, 0.0);

    vec4 clip = cam.view_proj * vec4(hit_world, 1.0);
    gl_FragDepth = clip.z / clip.w;

    // Motion vector — chunks move, but approximating with no motion (no
    // prev_model in PC) gives a small TAA ghost on falling debris,
    // acceptable for small chunks at distance.
    vec4 prev_clip = cam.prev_view_proj * vec4(hit_world, 1.0);
    vec2 cur_ndc  = clip.xy      / clip.w;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    outMotion = (cur_ndc - prev_ndc) * 0.5;
}
