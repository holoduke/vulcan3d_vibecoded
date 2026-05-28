#version 460

// Voxel shape fragment shader — brick DDA.
//
// Traces a ray from the camera through the AABB volume using
// Amanatides-Woo at voxel resolution (0.1 m). Every voxel step looks
// up the containing brick in the shape directory:
//   - directory entry = kEmpty → write a brick-sized skip (advance t to
//     the next brick boundary along the smallest axis)
//   - directory entry valid → check the brick's occupancy bitmap; on a
//     set bit, output the palette colour, write gl_FragDepth and
//     motion vector, done.
//
// Outputs match the world MRT colour pass:
//   location 0 = scene_color (HDR linear)
//   location 1 = motion vector (NDC delta)
// Writes gl_FragDepth so subsequent passes see correct depth for voxel
// pixels. Pipeline draws back-faces of the AABB (cull = FRONT) so we
// get exactly one invocation per pixel of the volume.

layout(location = 0) in vec3 vWorldPos;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

layout(push_constant) uniform PC {
    vec4  origin_world;
    vec4  dims_world;
    ivec4 dims_bricks;
    vec4  voxel_size;     // x: voxel size (m), y: brick size (m)
    vec4  shape_idx;
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

layout(set = 0, binding = 2, std430) readonly buffer ShapeDir {
    uvec4 hdr;       // xyz = brick dims (mirrors PC for safety)
    uint  entries[];
} shape;

const uint kEmpty = 0xFFFFFFFFu;
const int  kBrickDim = 16;

// AABB intersection. Returns the entry/exit t values (t0/t1) along ld.
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

void main() {
    // World-space ray from the camera through this AABB-cube fragment.
    vec3 ro = cam.camera_pos.xyz;
    vec3 rd = normalize(vWorldPos - ro);

    // Shape-local coords (translation only — no rotation in Session A).
    vec3 lo = ro - pc.origin_world.xyz;
    vec3 ld = rd;

    const float vs = pc.voxel_size.x;          // 0.1
    const float bs = pc.voxel_size.y;          // 1.6
    const ivec3 brick_dim = pc.dims_bricks.xyz;
    const ivec3 voxel_dim = brick_dim * kBrickDim;
    const vec3  shape_max = pc.dims_world.xyz;

    // Clip ray to the shape AABB.
    float t0, t1;
    if (!ray_aabb(lo, ld, vec3(0.0), shape_max, t0, t1)) discard;
    float t = max(t0, 0.0) + 1e-4;

    // ---- Voxel-DDA setup ----
    // Voxel coords at entry (shape grid, 0 .. voxel_dim-1).
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

    // t-value at which the ray crosses the NEXT voxel boundary, per axis.
    vec3 face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
    vec3 tMax;
    tMax.x = (ld.x != 0.0) ? t + (face.x - pe.x) / ld.x : 1e30;
    tMax.y = (ld.y != 0.0) ? t + (face.y - pe.y) / ld.y : 1e30;
    tMax.z = (ld.z != 0.0) ? t + (face.z - pe.z) / ld.z : 1e30;

    // Brick-level delta for empty-brick skip (skips 16 voxel steps in one).
    vec3 bDelta = bs * abs_inv;

    // Initial "last stepped axis" so an immediate hit has a sensible normal.
    // Use the axis whose entry-tMin is largest (the axis we entered the AABB on).
    vec3  tEntry;
    tEntry.x = (ld.x != 0.0) ? ((stp.x > 0 ? 0.0 : shape_max.x) - lo.x) / ld.x : -1e30;
    tEntry.y = (ld.y != 0.0) ? ((stp.y > 0 ? 0.0 : shape_max.y) - lo.y) / ld.y : -1e30;
    tEntry.z = (ld.z != 0.0) ? ((stp.z > 0 ? 0.0 : shape_max.z) - lo.z) / ld.z : -1e30;
    int last_axis = (tEntry.x > tEntry.y && tEntry.x > tEntry.z) ? 0
                   : (tEntry.y > tEntry.z ? 1 : 2);

    // Safety bound — diagonal voxel walk + slack.
    const int max_steps = (voxel_dim.x + voxel_dim.y + voxel_dim.z) + 32;

    bool  hit = false;
    uint  hit_pal_idx = 0u;
    float t_hit = t;
    int   hit_axis = last_axis;

    for (int i = 0; i < max_steps; ++i) {
        // Bounds break — ray has exited the shape grid.
        if (any(lessThan(vc, ivec3(0))) ||
            any(greaterThanEqual(vc, voxel_dim))) break;

        ivec3 bc = vc >> 4;
        int   dir_i = (bc.z * brick_dim.y + bc.y) * brick_dim.x + bc.x;
        uint  bp    = shape.entries[dir_i];

        if (bp == kEmpty) {
            // Skip the rest of this brick in one DDA step.
            // brickFace = (bc + max(stp,0)) * bs
            vec3 brickFace = (vec3(bc) + max(vec3(stp), 0.0)) * bs;
            vec3 bMax;
            bMax.x = (ld.x != 0.0) ? (brickFace.x - lo.x) / ld.x : 1e30;
            bMax.y = (ld.y != 0.0) ? (brickFace.y - lo.y) / ld.y : 1e30;
            bMax.z = (ld.z != 0.0) ? (brickFace.z - lo.z) / ld.z : 1e30;
            // Move to brick exit + 1 voxel slack so the next iteration
            // lands in the next brick.
            float t_skip = min(min(bMax.x, bMax.y), bMax.z);
            t = t_skip + 1e-4;
            vec3 pn = lo + t * ld;
            vc = ivec3(floor(pn / vs));
            // Re-init tMax to match new vc.
            face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
            tMax.x = (ld.x != 0.0) ? (face.x - lo.x) / ld.x : 1e30;
            tMax.y = (ld.y != 0.0) ? (face.y - lo.y) / ld.y : 1e30;
            tMax.z = (ld.z != 0.0) ? (face.z - lo.z) / ld.z : 1e30;
            // Last-axis is the brick-exit axis.
            last_axis = (bMax.x < bMax.y && bMax.x < bMax.z) ? 0
                      : (bMax.y < bMax.z ? 1 : 2);
            continue;
        }

        // Brick is occupied — test this voxel.
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

        // Step DDA: advance along smallest tMax axis.
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            t = tMax.x; vc.x += stp.x; tMax.x += tDelta.x; last_axis = 0;
        } else if (tMax.y < tMax.z) {
            t = tMax.y; vc.y += stp.y; tMax.y += tDelta.y; last_axis = 1;
        } else {
            t = tMax.z; vc.z += stp.z; tMax.z += tDelta.z; last_axis = 2;
        }
    }

    if (!hit) discard;

    // ---- Shade the hit ----
    vec3 hit_local = lo + t_hit * ld;
    vec3 hit_world = pc.origin_world.xyz + hit_local;

    // Face normal: opposite the axis we stepped to land in the hit voxel.
    vec3 N = vec3(0.0);
    N[hit_axis] = (hit_axis == 0) ? -float(stp.x)
                : (hit_axis == 1) ? -float(stp.y)
                                  : -float(stp.z);

    vec3 base = cam.pal[int(hit_pal_idx & 15u)].rgb;
    float ndl = max(dot(N, -cam.sun_dir.xyz), 0.0);
    vec3 col = base * (cam.ambient.rgb + cam.sun_color.rgb * ndl);
    outColor = vec4(col, 1.0);

    // Depth from world-space hit, matches the depth main pass writes for
    // brushes/terrain (clip.z / clip.w in Vulkan reversed-Z [0,1]).
    vec4 clip = cam.view_proj * vec4(hit_world, 1.0);
    gl_FragDepth = clip.z / clip.w;

    // Motion vector — shape is static for Session A so prev clip == cur clip.
    vec4 prev_clip = cam.prev_view_proj * vec4(hit_world, 1.0);
    vec2 cur_ndc  = clip.xy      / clip.w;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    outMotion = (cur_ndc - prev_ndc) * 0.5;
}
