#version 460
#extension GL_EXT_nonuniform_qualifier : require

// SSDM (Screen-Space Displacement Mapping, Lobel 2008) -- Phase 1.
// Pyramid A root: for each SPOM-wall fragment, output a 2D screen-space
// displacement vector "where this pixel's brick TOP would project to,
// minus where it actually rasterised". The Phase 2/3 mip-pyramid +
// barycenter iteration uses this field; Phase 4's remap pass samples
// scene_color at (gl_FragCoord.xy + pyrB[0].xy) to produce the final
// silhouette-extended image. Floors + non-SPOM materials write (0, 0).
//
// Wall pixels that hit a brick TOP (height_map = 1.0) write (0,0) -- no
// displacement needed, the rasterised pixel IS the source. Pixels that
// hit a deep crevice (height_map = 0) write the max projected offset
// along the wall's view-aligned tangent direction.

layout(location = 0) in vec3 vNormal;
layout(location = 1) flat in vec3 vColor;
layout(location = 2) in vec3 vWorldPos;
layout(location = 3) flat in vec4 vEmissive;
layout(location = 4) in vec2 vUv;
layout(location = 5) flat in vec4 vTexParams;
layout(location = 6) in vec3 vObjectPos;
layout(location = 7) in vec3 vObjectNormal;
layout(location = 8) in vec4 vPrevClip;
layout(location = 9) in vec3 vShellBase;

layout(location = 0) out vec2 outDisp;

const int kTextureCount = 16;
layout(set = 0, binding = 12) uniform sampler2D u_spom_height[kTextureCount];

// Subset of the scene UBO we need -- camera_pos for view dir, spom_params
// for strength. Layout must match cube.vert/cube.frag's UBO declaration
// (same set 0 binding 0), but only the prefix used here is referenced;
// trailing fields are unused. std140 alignment matches by construction.
layout(set = 0, binding = 0) uniform SceneUBO {
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
    vec4  muzzle_pos;
    vec4  muzzle_color;
    vec4  terrain_params;
    vec4  terrain_h_low;
    vec4  terrain_h_high;
    vec4  grass_extra;
    vec4  grass_extra2;
    mat4  light_vp;
    vec4  terrain_extra;
    vec4  _scene_pad[24];
    vec4  restir_params;
    vec4  spom_params;
} scene;

int height_idx_for_albedo(int a) {
    if (a == 1) return 0;
    if (a == 4) return 1;
    return -1;
}

void main() {
    int albedo_idx = int(vTexParams.x);
    int hidx       = height_idx_for_albedo(albedo_idx);
    // Object-space shading or non-SPOM material -> no displacement.
    int shading_mode = int(vTexParams.w + 0.5);
    if (hidx < 0 || shading_mode != 0) {
        outDisp = vec2(0.0);
        return;
    }
    // Floor exclusion: SPOM displacement is for vertical brick faces.
    // Triplanar axis selection by |normal|; if Y dominates, skip.
    vec3 N = normalize(vNormal);
    if (abs(N.y) > 0.5) {
        outDisp = vec2(0.0);
        return;
    }

    float strength = max(0.0, scene.spom_params.x);
    if (strength < 0.01) {
        outDisp = vec2(0.0);
        return;
    }

    // Triplanar UV in object space: the dominant tangent plane is the one
    // perpendicular to the largest normal axis. For vertical walls that's
    // XY (normal=Z) or ZY (normal=X). Use object-space UVs so brick
    // alignment matches cube.frag's spom_uv() sampling exactly.
    vec3 absN = abs(N);
    vec2 uvw;
    if (absN.x > absN.z) {
        uvw = vec2(vWorldPos.z, vWorldPos.y);
    } else {
        uvw = vec2(vWorldPos.x, vWorldPos.y);
    }
    uvw *= vTexParams.z;  // uv_scale per material

    // Sample height map. h ∈ [0,1] -- 1 = brick top (no recession), 0 = deep crevice.
    float h = texture(u_spom_height[hidx], uvw).r;
    float recession = (1.0 - h) * 0.04 * strength;   // matches cube.frag base

    // Project the recession back to screen pixels via the normal's screen
    // derivative at this fragment. dFdx/dFdy of screen position is by
    // definition (1,0) and (0,1), so the screen delta for a 1-unit world
    // shift along N is just dFdx of (N projected) -- but more directly:
    // a world-space offset of recession*N produces a screen offset of
    // approximately (dScreen/dN) * recession. We approximate via the
    // chain rule using vWorldPos derivatives.
    //
    // Pixel-space dWorld_dPix^-1 maps pixel delta to world delta:
    //   dW/dx = dFdx(vWorldPos), dW/dy = dFdy(vWorldPos)
    // A world shift of (recession * N) corresponds to a pixel shift
    // (dx, dy) = J^-1 * (recession * N), where J = [dW/dx | dW/dy]
    // (3x2 matrix). Use the pseudo-inverse: pix = (J^T J)^-1 J^T * shift.
    vec3 dWdx = dFdx(vWorldPos);
    vec3 dWdy = dFdy(vWorldPos);
    vec3 shift = recession * N;
    // 2x2 Gram matrix J^T J and its inverse.
    float a = dot(dWdx, dWdx);
    float b = dot(dWdx, dWdy);
    float c = dot(dWdy, dWdy);
    float det = a * c - b * b;
    if (abs(det) < 1e-8) {
        outDisp = vec2(0.0);
        return;
    }
    float inv_det = 1.0 / det;
    float jx = dot(dWdx, shift);
    float jy = dot(dWdy, shift);
    vec2 pix = vec2(c * jx - b * jy, a * jy - b * jx) * inv_det;
    // Pyramid A stores the per-pixel projected-displacement vector. Sign:
    // a recessed brick crevice sits BEHIND the raster surface; the
    // source pixel (where the brick TOP would project) is in the
    // OPPOSITE direction of the recession's screen projection -- so
    // negate pix. Phase 3's iteration treats this as "subtract A(b)
    // from current to find source", matching the paper's convention.
    outDisp = -pix;
}
