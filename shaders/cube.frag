#version 460
#extension GL_EXT_ray_query : require
#extension GL_ARB_conservative_depth : require

// Conservative depth: promise the driver that any gl_FragDepth write here
// will be ≥ gl_FragCoord.z. The unconditional `gl_FragDepth = gl_FragCoord.z`
// at the top of main() satisfies "equal", and the SPOM silhouette path
// writes 1.0 (sky/far) which is also "greater". With this qualifier in
// place, hardware early-Z stays enabled even though the shader writes
// depth — the depth pre-pass's z-buffer can reject occluded fragments
// before the heavy cube.frag body (RT shadow PCSS + GI bounces + AO +
// SPOM + muzzle flash light) runs. This was the single biggest GPU win
// found by the latest perf review: cube.frag was running on every
// fragment that the depth pre-pass should have culled.
layout(depth_greater) out float gl_FragDepth;

layout(location = 0) in vec3 vNormal;
// flat вЂ” see cube.vert for the rationale (these are push-constant values
// that must not perspective-interpolate; the vTexParams.x precision drift
// was silently enabling texture sampling when textures were "off").
layout(location = 1) flat in vec3 vColor;
layout(location = 2) in vec3 vWorldPos;
layout(location = 3) flat in vec4 vEmissive;
layout(location = 4) in vec2 vUv;
layout(location = 5) flat in vec4 vTexParams;  // x: albedo idx, y: normal idx,
                                               // z: uv scale,
                                               // w: 0 = world-space triplanar
                                               //    1 = object-space triplanar
layout(location = 6) in vec3 vObjectPos;
layout(location = 7) in vec3 vObjectNormal;
layout(location = 8) in vec4 vPrevClip;  // prev_view_proj Г— prev_model Г— local_pos
layout(location = 0) out vec4 outColor;
// Per-pixel screen-space motion vector вЂ” current_uv minus prev_uv. Dynamic
// surfaces get correct reprojection because cube.vert applied prev_model
// (DynRender::prev_world Г— scale for dyn boxes; current model for static
// brushes / camera-attached viewmodel / sub-pixel particles). Future SVGF
// pass reads motion_vec_image_ + uses this delta to walk into history.
layout(location = 1) out vec2 outMotion;

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;     // x:shadow_on, y:shadow_samples, z:ao_samples, w:frame
    vec4  rt_params;    // x:shadow_softness, y:ao_radius, z:ambient_strength, w:shadow_strength
    ivec4 rt_flags2;    // x:gi_samples, y:reflections_on, z:gi_bounces, w:ao_mode
                        //   ao_mode: 0=off, 1=fast (2 short rays), 2=RTAO (full)
    vec4  rt_params2;   // x:gi_strength, y:gi_radius, z:reflection_strength,
                        // w:shadow_curve (0=linear, 1=cubic)
    vec4  camera_pos;   // xyz = world-space eye
    vec4  rt_lod;       // x:lod_near, y:lod_far
    vec4  viewport;     // x:w, y:h, z:1/w, w:1/h вЂ” used by motion-vec output
    // Muzzle flash dynamic light. xyz = world-space origin, .w = intensity
    // (0 disables the contribution). Color/radius live in muzzle_color.
    vec4  muzzle_pos;
    vec4  muzzle_color; // rgb = color (linear), w = falloff radius (m)
    // Terrain shader knobs:
    //   terrain_params.x = fog_strength (atmospheric perspective)
    //   terrain_params.y = wrap_strength (half-Lambert)
    //   terrain_params.z = detail_strength (texture brightness)
    //   terrain_params.w = shadow_softness_scale (PCSS cone Г— this)
    vec4  terrain_params;
    // Per-layer height-blend smoothstep edges:
    //   terrain_h_low.xy  = sandв†’grass start..end
    //   terrain_h_low.zw  = grassв†’dirt start..end
    //   terrain_h_high.xy = dirtв†’rock start..end
    //   terrain_h_high.zw = rockв†’snow start..end
    vec4  terrain_h_low;
    vec4  terrain_h_high;
    vec4  grass_extra;       // unused in cube.frag вЂ” laid out so grass_extra2.w is reachable
    vec4  grass_extra2;      // .w = terrain_debug_mode (0=off, 1=Lambert, 2=normal, 3=face)
    mat4  light_vp;          // unused in cube.frag — keeps UBO layout aligned
    vec4  terrain_extra;     // .x = terrain shading contrast (n_dot_l power)
    // 24 vec4 padding fields between terrain_extra and restir_params on
    // the C++ side. cube.frag doesn't read any of them today (water,
    // shore tints, distance fog, grass colours, etc. are consumed by
    // terrain_raymarch.frag and grass_raymarch.frag). If a future
    // change adds a use, replace the right slot with a named field.
    vec4  _scene_pad[24];
    // ReSTIR GI runtime knobs (see docs/restir_plan.md).
    //   .x = enabled (0/1) — gates the temporal reservoir read below
    //   .y = M_max (sample-count cap)
    //   .z = disocclusion normal-dot threshold
    vec4  restir_params;
    // SPOM tuning. .x scales the per-pixel height-march depth (1 =
    // engine default, 0 = flat / disabled). Other components reserved.
    vec4  spom_params;
    // Terrain local info — most slots used by other shaders. cube.frag
    // only reads .z (half-rate-shadow consumer toggle: 1 means sample
    // the binding-18 u_shadow_lr texture for brush/dyn surfaces instead
    // of running the inline blocker+PCSS block below).
    vec4  terrain_local_info;
} scene;

// Half-rate shadow producer output, sampled when terrain_local_info.z
// is on. linear_sampler_ → bilinear interpolation across the 4 nearest
// half-res texels is the "bilateral" part of this minimum-viable upsample.
layout(set = 0, binding = 18) uniform sampler2D u_shadow_lr;

// SVGF GI denoiser raw-irradiance write target (Session 1 of
// docs/svgf_plan.md). cube.frag stores the post-temporal/spatial
// reservoir reuse irradiance HERE before it gets multiplied by
// albedo, so the future denoiser passes (sessions 2-4) operate on the
// clean lighting signal. Format MUST match the C++ image
// (VK_FORMAT_R16G16B16A16_SFLOAT). No reader in session 1.
layout(set = 0, binding = 19, rgba16f) uniform image2D u_svgf_gi;

// SVGF temporal accumulator history (Session 2). Ping-pong pair: on
// frame F, cube.frag reads from binding (20 + ((F+1) & 1)) and writes
// to binding (20 + (F & 1)). Race-free under kFrameOverlap == 2 — F's
// write target was last touched by F-2, which is retired by the F-1
// fence the CPU waits on before recording F. .rgb = accumulated
// irradiance, .a = M-count [1..kSvgfMmax] for the EMA weight.
layout(set = 0, binding = 20, rgba16f) uniform image2D u_svgf_hist0;
layout(set = 0, binding = 21, rgba16f) uniform image2D u_svgf_hist1;

// Distance-based sample LOD. fragments within lod_near get full samples;
// fragments past lod_far drop to a single ray. Standard "RT becomes a luxury
// at distance" trick used in commercial engines.
int lod_samples(int requested, float dist) {
    float lod_t = clamp((dist - scene.rt_lod.x) /
                        max(0.001, scene.rt_lod.y - scene.rt_lod.x), 0.0, 1.0);
    int n = int(round(mix(float(requested), 1.0, lod_t)));
    return max(1, n);
}

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct Material {
    vec4 color;
    vec4 emissive;
    vec4 tex;       // x: albedo idx, y: normal idx, z: uv scale, w: spare
};
layout(set = 0, binding = 2, std430) readonly buffer Materials {
    Material materials[];
};

// Bound texture arrays (must match kTextureCount on the C++ side).
const int kTextureCount = 12;
// Procedurally-baked terrain material slots (see bake_terrain_materials
// in world.cpp). Must match kTexRock/.. in vk_engine.h.
const int kMatRock  = 7;
const int kMatGrass = 8;
const int kMatDirt  = 9;
const int kMatSand  = 10;
const int kMatSnow  = 11;
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];
layout(set = 0, binding = 4) uniform sampler2D u_normal[kTextureCount];
// Pre-baked heightmap sun-shadow texture (R8, 0 = lit, 255 = shadowed).
// Sampled here as a distance fallback for terrain self-shadow вЂ” the
// BLAS holds the full heightmap detail, so RT shadow rays from a low-
// LOD rasterised surface false-hit BLAS peaks the raster missed. The
// bake was traced against the heightmap directly so it matches the
// rasterised LOD surface at distance with no false hits.
layout(set = 0, binding = 6) uniform sampler2D u_terrain_shadow;
// Sun shadow map (single-cascade ortho, rendered each frame from the
// sun's POV with castle / dyn-props / terrain as casters). 1024² over
// ~120m world half = ~12cm per texel close to the camera. We sample
// this for terrain receivers within the shadow map's frustum — much
// sharper than the bake's 1m texels.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

// Brick (slot 1) parallax-occlusion displacement. Sampled by the SPOM
// path below for albedo_idx == 1 (castle walls). Other materials don't
// have a height map and skip the parallax march entirely.
// SPOM (parallax-occlusion) displacement map array. cube.frag picks the
// right entry per fragment via height_idx_for_albedo() below; non-SPOM
// materials skip the parallax march entirely. Order MUST match the
// spom_height_textures_ array on the host (vk_engine.h).
layout(set = 0, binding = 12) uniform sampler2D u_height[4];

// ReSTIR GI reservoir buffers (session 2 — see docs/restir_plan.md).
// Layout MUST match the C++ Reservoir struct in restir.cpp (std430,
// 48 B / pixel). Indexed as [pix.y * render_w + pix.x]. Session 2
// only ships the write — temporal read of u_reservoir_prev lands in
// session 3.
struct Reservoir {
    vec3  sample_dir;   // surface normal N of the writing pixel
    vec3  radiance;     // temporally-accumulated GI radiance
    float W;
    float w_sum;
    uint  M;            // sample count (0 = invalid / never written)
    uint  pad;          // session 5: floatBitsToUint(cam_dist) of the
                        //   writing surface — used for depth-aware
                        //   disocclusion (reject reprojected/neighbour
                        //   samples whose surface is at a different
                        //   distance, i.e. a different object).
};
// Bindings 15/16 alias the SAME physical buffer, which holds THREE
// per-pixel reservoir regions (a ring). cube.frag selects the region
// by frame parity (see kResRingBase below): write region frame%3,
// read region (frame+2)%3. This makes prev/cur a genuine ping-pong
// with NO descriptor swap and is race-free under kFrameOverlap==2
// (see the restir.cpp file header for the proof). Both declarations
// omit readonly/writeonly so the compiler can't assume non-aliasing.
layout(set = 0, binding = 15, std430) buffer ReservoirPrev {
    Reservoir r[];
} u_reservoir_prev;
layout(set = 0, binding = 16, std430) buffer ReservoirCur {
    Reservoir r[];
} u_reservoir_cur;

// Map an albedo texture index to a SPOM material slot (or -1 for "no
// parallax"). Keep in sync with vk_engine.h's spom_height_textures_
// allocation order.
int height_idx_for_albedo(int a) {
    if (a == 1) return 0;   // Bricks078         — castle outer walls
    if (a == 4) return 1;   // PaintedBricks001  — keep walls
    // Floor tile materials (5 / 6) intentionally omitted: floor brushes are
    // 0.5 m thin slabs, so their side faces are axis 0/2 and bypass the
    // axis==1 silhouette gate. SPOM raymarched past the slab thickness and
    // produced visible parallax depth on the slab edges that didn't match
    // anything in the world. Floors keep normal maps + triplanar shading;
    // they just lose the parallax displacement effect.
    return -1;
}

// Push constants — match cube.vert's PC layout exactly. Fragment shader
// only reads `model` (for the brush's world center + per-axis size used
// by the SPOM silhouette test); the rest are unused here but the layout
// must mirror the vertex shader because both stages share the range.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    // Already inside the shared push range (other shaders use it);
    // cube.frag just hadn't declared it. For terrain colour draws:
    //   .x = sand ripple scale, .y = grass line scale.
    vec4 grass_params;
} pc;

// Triplanar projection sample. Avoids the "what UV does this cube face
// use" problem and works correctly on every brush size вЂ” the texture stays
// world-aligned regardless of brush dimensions. Cost is 3 samples per pass,
// which is fine for the coverage we have.
vec3 triplanar_sample(sampler2D tex, vec3 wp, vec3 N) {
    // pow(abs(N), 4) via two squarings — saves one exp/log/mul per
    // triplanar call vs the pow() intrinsic, on every textured pixel.
    vec3 blend = abs(N); blend *= blend; blend *= blend;
    blend /= max(blend.x + blend.y + blend.z, 1e-3);
    // P15: dominant-axis early-out. When one weight ≥ 0.95, the other
    // two contribute ≤ 5 % blended → invisible. Sampling those two
    // textures wastes ~2 cache fetches on the very common cube/floor
    // pixels (axis-aligned brushes, flat terrain). Fall back to the
    // 3-tap blend when no single axis dominates.
    if (blend.x >= 0.95) return texture(tex, wp.zy).rgb;
    if (blend.y >= 0.95) return texture(tex, wp.xz).rgb;
    if (blend.z >= 0.95) return texture(tex, wp.xy).rgb;
    // Now that mipmaps exist (texture.cpp generates them via blit chain) and
    // vTexParams is `flat` so the sampler-array index is dynamically uniform,
    // standard `texture()` is correct вЂ” derivatives select the right mip and
    // far-distance moirГ© disappears.
    vec3 cx = texture(tex, wp.zy).rgb;
    vec3 cy = texture(tex, wp.xz).rgb;
    vec3 cz = texture(tex, wp.xy).rgb;
    return cx * blend.x + cy * blend.y + cz * blend.z;
}

// Triplanar tangent-space normal. For each projection axis, sample the
// tangent-space normal and transform into world space using a known basis
// for that axis. Then blend by face weight, same as the albedo. Cheap,
// works without explicit per-vertex tangents.
// Dominant-axis parallax-occlusion mapping for axis-aligned faces.
// `wp` is world position scaled to texture-UV space (the same scaled
// position triplanar_sample would consume), `N` is the surface normal.
// Returns the offset 2D UV inside the dominant axis projection plus
// (in `out_axis`) which axis won so the caller can reconstruct the
// tangent-space normal correctly.
//
// The march is 16 linear steps + 1 binary refine — enough to look like
// real depth on castle bricks at typical view angles without showing
// stair-stepping. Grazing-angle pixels (V_t.z < 0.2) skip the march
// because the parallax displacement explodes there and produces
// shimmer + texture swimming.
// `out_overhang_disc` is set true when the parallax march pushes the
// visible point past the brush face's geometric edge — caller should
// `discard` to make the silhouette read as bumpy bricks instead of a
// flat clipped edge (the "S" in SPOM).
vec2 spom_uv(vec3 wp_scaled, vec3 N, vec3 face_size_world, float uv_scale,
             int height_idx,
             out int out_axis,
             out vec3 out_T, out vec3 out_B, out vec3 out_face_n,
             out bool out_overhang_disc,
             out vec3 out_displaced_pos,
             out vec2 out_uv0) {
    out_overhang_disc = false;
    out_displaced_pos = vWorldPos;
    vec3 absN = abs(N);
    int axis;
    vec2 uv0;
    vec3 T, Bb, face_n;
    if (absN.x >= absN.y && absN.x >= absN.z) {
        axis = 0;
        face_n = vec3(sign(N.x), 0.0, 0.0);
        // sign(N.x) on T flips the texture U handedness so the +X and -X
        // faces aren't mirrored versions of each other.
        T  = vec3(0.0, 0.0, sign(N.x));
        Bb = vec3(0.0, 1.0, 0.0);
        uv0 = vec2(wp_scaled.z * sign(N.x), wp_scaled.y);
    } else if (absN.y >= absN.z) {
        axis = 1;
        face_n = vec3(0.0, sign(N.y), 0.0);
        T  = vec3(1.0, 0.0, 0.0);
        Bb = vec3(0.0, 0.0, sign(N.y));
        uv0 = vec2(wp_scaled.x, wp_scaled.z * sign(N.y));
    } else {
        axis = 2;
        face_n = vec3(0.0, 0.0, sign(N.z));
        T  = vec3(sign(N.z), 0.0, 0.0);
        Bb = vec3(0.0, 1.0, 0.0);
        uv0 = vec2(wp_scaled.x * sign(N.z), wp_scaled.y);
    }
    out_axis = axis;
    out_T = T; out_B = Bb; out_face_n = face_n;
    out_uv0 = uv0;

    vec3 V_w = normalize(scene.camera_pos.xyz - vWorldPos);
    vec3 V_t = vec3(dot(V_w, T), dot(V_w, Bb), dot(V_w, face_n));
    if (V_t.z < 0.2) return uv0;   // grazing — skip parallax (shimmer guard)

    // Ramp parallax depth down at distance so far walls stay flat (no
    // visible texture swimming) and only close-up walls show the bumps.
    float d = distance(vWorldPos, scene.camera_pos.xyz);
    float depth_w = 1.0 - smoothstep(8.0, 30.0, d);
    if (depth_w <= 0.001) return uv0;

    // 4cm peak-to-trough at 1m view, scaled by the runtime slider
    // (scene.spom_params.x; 0 = flat, 1 = engine default). When the
    // multiplier is ~0 we'd march flat texture coords for nothing —
    // bail to the un-parallaxed path so SPOM cost goes to zero.
    const float kHeightScaleBase = 0.04;
    float spom_strength = max(0.0, scene.spom_params.x);
    if (spom_strength < 0.01) return uv0;
    float kHeightScale = kHeightScaleBase * spom_strength;
    vec2  P = (V_t.xy / V_t.z) * (kHeightScale * depth_w);

    // Step count ramps with depth_w: full 12 steps for the closest
    // walls (depth_w → 1), down to 4 steps for walls just inside the
    // SPOM fade kick-in. Cap is 12 (was 16) — TAA averages the
    // remaining step-quantisation noise away on close walls, and the
    // binary-search refine below adds sub-step precision. Each saved
    // step is one texture() in the castle-interior hot path.
    int kSteps = max(4, int(round(depth_w * 12.0)));
    float layer_step = 1.0 / float(kSteps);
    vec2  uv_step    = -P * layer_step;
    float current_layer = 0.0;
    vec2  cur_uv = uv0;
    float cur_h  = 1.0 - texture(u_height[height_idx], cur_uv).r;
    for (int i = 0; i < 12; ++i) {
        if (i >= kSteps) break;
        if (current_layer >= cur_h) break;
        cur_uv += uv_step;
        current_layer += layer_step;
        cur_h = 1.0 - texture(u_height[height_idx], cur_uv).r;
    }
    // Refine in two stages so the height steps don't read as visible
    // "stair lines" on the surface:
    //   1. Linear-interp between the last two layer samples (cheap).
    //   2. 2-iter binary search bracketed by (prev_uv, cur_uv) — each
    //      iteration halves the residual error, so 2 iters takes the
    //      step seam from ~6 % of one layer down to ~1.5 %. Visually
    //      fine after TAA averaging; costs 2 extra texture taps per
    //      SPOM-active pixel (was 4 — castle-interior perf tighten).
    vec2  prev_uv = cur_uv - uv_step;
    float prev_h  = 1.0 - texture(u_height[height_idx], prev_uv).r;
    float prev_layer = current_layer - layer_step;
    float a = cur_h  - current_layer;
    float b = prev_h - prev_layer - a;
    float w = clamp(a / (b - a + 1e-4), 0.0, 1.0);
    vec2 final_uv = mix(cur_uv, prev_uv, w);
    float final_layer = mix(current_layer, prev_layer, w);
    {
        vec2 lo_uv = prev_uv;     // shallower (above surface)
        vec2 hi_uv = cur_uv;      // deeper (below surface)
        float lo_l = prev_layer;
        float hi_l = current_layer;
        for (int j = 0; j < 2; ++j) {
            vec2  m_uv = (lo_uv + hi_uv) * 0.5;
            float m_l  = (lo_l  + hi_l)  * 0.5;
            float m_h  = 1.0 - texture(u_height[height_idx], m_uv).r;
            if (m_l < m_h) {       // still above the surface
                lo_uv = m_uv; lo_l = m_l;
            } else {                // below — bracket the upper half
                hi_uv = m_uv; hi_l = m_h;
            }
        }
        final_uv    = (lo_uv + hi_uv) * 0.5;
        final_layer = (lo_l  + hi_l)  * 0.5;
    }

    // SPOM silhouette test: convert the UV offset back to world units
    // along the face's tangent + bitangent. If the visible point lies
    // outside the cube face's geometric extent (face_size_world holds
    // the brush's full XYZ size), the bricks would be jutting past the
    // edge — make those pixels transparent so the silhouette looks
    // bumped at corners instead of flat-clipped.
    vec2 d_uv = final_uv - uv0;
    float d_T = d_uv.x / max(uv_scale, 1e-4);
    float d_B = d_uv.y / max(uv_scale, 1e-4);
    // World-axis index for T and B: 0=X, 1=Y, 2=Z. T's axis depends on
    // the dominant face axis (see basis selection above).
    int t_axis = (axis == 0 || axis == 2) ? (axis == 0 ? 2 : 0) : 0;
    int b_axis = (axis == 1) ? 2 : 1;
    // Center of this brush in world space lives in pc.model[3].
    vec3 center = vec3(pc.model[3].x, pc.model[3].y, pc.model[3].z);
    vec3 rel = vWorldPos - center;
    float pos_T = (axis == 0) ? rel.z * sign(N.x) :
                  (axis == 1) ? rel.x :
                                rel.x * sign(N.z);
    float pos_B = (axis == 1) ? rel.z * sign(N.y) : rel.y;
    float vis_T = pos_T + d_T;
    float vis_B = pos_B + d_B;
    float ext_T = face_size_world[t_axis] * 0.5;
    float ext_B = face_size_world[b_axis] * 0.5;
    // Silhouette overhang: parallax has walked the visible point past
    // the geometric face edge. Caller draws this fragment with alpha
    // = 0 so the underlying scene_color (terrain / other walls drawn
    // earlier) shows through the brick cavity.
    // Silhouette test with tolerance. Pure ext_T/ext_B fired at joints
    // between two abutting brushes (e.g. tower-to-wall, gate-block-to-
    // wall) — the parallax displacement crossed the face edge but the
    // neighbour brush is RIGHT THERE filling the gap, so writing
    // alpha = 0 made the joint show the (often-shadowed) terrain
    // behind both brushes instead of the neighbour wall. Allowing the
    // displacement to extend up to 25 % past the geometric edge means
    // the test only fires when the parallax goes into a REAL gap (an
    // outside corner with no neighbour brush). Trade-off: bricks now
    // visibly poke a bit past the edge before silhouette kicks in;
    // visually fine for the brick depth scale we use.
    // Skip silhouette discard for top/bottom-facing surfaces (axis == 1).
    // Floors don't have visible silhouette edges from above — you're
    // looking AT them, not along them — but at grazing camera angles
    // the parallax displacement can still walk past the floor's XZ
    // extent and trigger the discard. That wrote alpha = 0 + depth =
    // 1.0 → compose substituted procedural sky for the downward ray
    // direction → grey patches that flickered with TAA jitter as the
    // player moved. SPOM displacement still works on floors; only the
    // alpha-cavity effect is gated to walls.
    // Tightened from 0.25 → 0.15: the wider pad let entire wall panes
    // read as "elevated outward" because all corner bricks could poke a
    // noticeable distance past the edge. 15 % is enough to absorb the
    // typical 1-cm parallax displacement without producing a visible
    // shelf at the geometric edge. The improved seam ray below catches
    // the extra silhouette pixels the tighter pad creates.
    float pad = 0.15;
    if (axis != 1 &&
        (abs(vis_T) > ext_T * (1.0 + pad) ||
         abs(vis_B) > ext_B * (1.0 + pad))) out_overhang_disc = true;
    // Displaced world position: lateral tangent offset only. The
    // `-face_n * depth` push (toward inside the cube) was tempting
    // for "true" 3D shadow origins but moved the origin INSIDE the
    // brush — every RT shadow ray then hit the cube's own back face,
    // dark patches everywhere. Keeping just the tangent shift means
    // shadow rays trace from a laterally-offset point on the cube
    // surface; that captures the brick-row alignment for falling sun
    // angles (visible self-shadow at brick edges) without the
    // self-occlusion bug.
    out_displaced_pos = vWorldPos + T * d_T + Bb * d_B;
    return final_uv;
}

vec3 triplanar_normal(sampler2D ntex, vec3 wp, vec3 N) {
    // Same pow(abs(N), 4) → squarings trick as triplanar_sample.
    vec3 blend = abs(N); blend *= blend; blend *= blend;
    blend /= max(blend.x + blend.y + blend.z, 1e-3);

    // P15: dominant-axis early-out (matches triplanar_sample). Skip
    // 2 of 3 normal-map taps + their TBN math when one axis dominates.
    if (blend.x >= 0.95) {
        vec3 nx = texture(ntex, wp.zy).rgb * 2.0 - 1.0;
        return normalize(vec3(N.x * sign(N.x) + nx.z * sign(N.x), nx.y, nx.x));
    }
    if (blend.y >= 0.95) {
        vec3 ny = texture(ntex, wp.xz).rgb * 2.0 - 1.0;
        return normalize(vec3(ny.x, N.y * sign(N.y) + ny.z * sign(N.y), ny.y));
    }
    if (blend.z >= 0.95) {
        vec3 nz = texture(ntex, wp.xy).rgb * 2.0 - 1.0;
        return normalize(vec3(nz.x, nz.y, N.z * sign(N.z) + nz.z * sign(N.z)));
    }

    vec3 nx = texture(ntex, wp.zy).rgb * 2.0 - 1.0;
    vec3 ny = texture(ntex, wp.xz).rgb * 2.0 - 1.0;
    vec3 nz = texture(ntex, wp.xy).rgb * 2.0 - 1.0;

    // For each axis, build a world-space normal: the tangent-space sample's
    // .xy goes into the perpendicular plane, .z reinforces the axis.
    vec3 wnx = vec3(N.x * sign(N.x) + nx.z * sign(N.x), nx.y, nx.x);
    vec3 wny = vec3(ny.x, N.y * sign(N.y) + ny.z * sign(N.y), ny.y);
    vec3 wnz = vec3(nz.x, nz.y, N.z * sign(N.z) + nz.z * sign(N.z));

    return normalize(wnx * blend.x + wny * blend.y + wnz * blend.z);
}

// Interleaved Gradient Noise (Jorge Jimenez, "Next Generation Post-Processing
// in Call of Duty: Advanced Warfare", SIGGRAPH 2014).
// Adjacent pixels see *smoothly varying* random values rather than hash-style
// uncorrelated jumps, so the per-pixel integration error has low spatial
// frequency вЂ” looks like a smooth gradient rather than a dotted dither.
float ign(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Isotropic hashed value-noise for TERRAIN spatial break-up. ign()
// above is Interleaved Gradient Noise — a screen-space DITHER whose
// values are organised along one fixed direction; sampled as a spatial
// noise on world XZ it stamps parallel same-direction bands onto the
// ground (independent of height/geometry). tnoise() is a normal
// bilinear value-noise with no directional structure → no banding.
float thash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float tnoise(vec2 x) {
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float a = thash21(i);
    float b = thash21(i + vec2(1.0, 0.0));
    float c = thash21(i + vec2(0.0, 1.0));
    float d = thash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);   // [0,1], isotropic
}

// === Hash + value-noise with analytical derivatives ===
// Identical recipe to terrain_raymarch.frag::noised — the terrain
// micro-detail pass below uses these so the rasterised mesh inherits
// the same FBM-eroded look as the per-pixel raymarched terrain
// (otherwise sub-cell detail snaps to the 1 m vertex spacing).
float t_hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 t_noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = t_hash21(i + vec2(0.0, 0.0));
    float b = t_hash21(i + vec2(1.0, 0.0));
    float c = t_hash21(i + vec2(0.0, 1.0));
    float d = t_hash21(i + vec2(1.0, 1.0));
    float v = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  dv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(v, dv);
}
const mat2 t_m2 = mat2(0.8, -0.6, 0.6, 0.8);

// FBM with derivative erosion — `n.x / (1 + |∇d|²)` damps high-freq
// octaves on steep slopes, producing the ridge/valley structure
// instead of fractal mush. Returns (value, ∂/∂x, ∂/∂z) where the
// derivatives are with respect to `world_xz` (Jacobian-corrected
// per octave so they stay in world space).
vec3 terrain_detail_fbm(vec2 wp, float scale, int octaves) {
    vec2 p = wp * scale;
    mat2 J = mat2(scale, 0.0, 0.0, scale);
    float a = 0.0, b = 1.0;
    vec2 dacc = vec2(0.0);
    for (int i = 0; i < 9; ++i) {
        if (i >= octaves) break;
        vec3 n = t_noised(p);
        // World-space derivative: J^T · local_deriv
        vec2 wd = J * n.yz;
        dacc += wd;
        float damp = 1.0 / (1.0 + dot(dacc, dacc));
        a += b * n.x * damp;
        b *= 0.5;
        // Per-octave 2× scale + rotation
        p = t_m2 * p * 2.0;
        J = t_m2 * J * 2.0;
    }
    // Approx-centred value (FBM tends toward ~0.5 mean) and
    // accumulated world-space gradient
    return vec3(a - 0.5, dacc.x, dacc.y);
}

// Per-sample IGN: shift the input position by the sample index along a known
// magic vector so each sample's jitter is uncorrelated within a pixel but
// stays smooth across neighbouring pixels.
float rand(uvec3 seed) {
    float s = float(seed.x) + 5.588238  * float(seed.z);
    float t = float(seed.y) + 1.388765  * float(seed.z);
    return ign(vec2(s, t));
}

vec3 cos_hemi(float u1, float u2, vec3 n) {
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3 d = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    vec3 up_a = abs(n.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 t = normalize(cross(up_a, n));
    vec3 b = cross(n, t);
    return d.x * t + d.y * n + d.z * b;
}

// Vogel disk taps (cos, sin of i × golden-angle). 32 entries; PCSS
// loops index `i & 31` so any sample count up to 32 maps cleanly. Per-
// pixel rotation (one sin/cos at the start of the loop) breaks the
// fixed-pattern aliasing while still saving 2N − 2 trig calls on an
// N-tap PCSS shadow.
const vec2 kVogelDisk[32] = vec2[32](
    vec2( 1.000000,  0.000000),
    vec2(-0.737369,  0.675490),
    vec2( 0.087426, -0.996171),
    vec2( 0.608439,  0.793601),
    vec2(-0.984713, -0.174182),
    vec2( 0.843755, -0.536728),
    vec2(-0.259604,  0.965715),
    vec2(-0.460907, -0.887448),
    vec2( 0.939321,  0.343039),
    vec2(-0.924346,  0.381556),
    vec2( 0.423846, -0.905734),
    vec2( 0.299284,  0.954164),
    vec2(-0.865211, -0.501408),
    vec2( 0.976676, -0.214719),
    vec2(-0.575129,  0.818062),
    vec2(-0.128511, -0.991708),
    vec2( 0.764649,  0.644447),
    vec2(-0.999146,  0.041318),
    vec2( 0.708829, -0.705380),
    vec2(-0.046191,  0.998933),
    vec2(-0.640709, -0.767784),
    vec2( 0.991069,  0.133347),
    vec2(-0.820858,  0.571132),
    vec2( 0.219481, -0.975617),
    vec2( 0.497181,  0.867647),
    vec2(-0.952693, -0.303935),
    vec2( 0.907791, -0.419423),
    vec2(-0.386061,  0.922473),
    vec2(-0.338452, -0.940984),
    vec2( 0.885189,  0.465231),
    vec2(-0.966970,  0.254890),
    vec2( 0.540838, -0.841127)
);

// Shadow + AO test. Cull-mask 0x01 picks up only instances marked as
// shadow-casters (bit 0). Sparks and projectiles are flagged 0xFE on the
// host вЂ” they're tiny visual effects whose hard shadow streaks looked wrong
// and whose AO darkening produced visible halos around hits.
bool any_hit(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x01, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Shadow-ray variant for TERRAIN receivers. Uses cull-mask 0x02 which
// the terrain BLAS instance does NOT have set — so the ray only finds
// castle / dyn-prop occluders, not the terrain itself. The terrain BLAS
// would otherwise produce per-triangle false-hits because its full-
// resolution geometry sits above the rasterised LOD surface. Terrain
// self-shadow comes from the heightmap bake instead.
bool any_hit_no_terrain(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Closest-hit variant of the terrain-receivers shadow query — used
// for the PCSS blocker search distance estimate.
bool closest_hit_no_terrain(vec3 origin, vec3 dir, float t_max,
                             out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

// Sky tint without the sun halo вЂ” for atmospheric fog where adding
// the localized halo brightness on distant terrain pixels makes them
// glow blindingly when looking near the sun. Same horizonв†’zenith
// gradient as `sample_sky` but no halo term.
vec3 sample_sky_atmosphere(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, sqrt(up));    // pow(up, 0.45) ≈ sqrt(up); 3% error, no exp/log
}

// Procedural sky: warm low horizon в†’ cool zenith, brighter near sun.
vec3 sample_sky(vec3 dir) {
    vec3 L = normalize(scene.sun_direction.xyz);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    vec3 sky = mix(horizon, zenith, sqrt(up));   // pow(up, 0.45) approx
    // pow(x, 8) → 3 squarings, no exp/log.
    float h1 = max(dot(dir, L), 0.0);
    float h2 = h1 * h1;
    float h4 = h2 * h2;
    float halo = h4 * h4;
    sky += scene.sun_color.rgb * scene.sun_color.a * 0.08 * halo;
    return sky;
}

// Mask-parameterised variants — let the PCSS loop hoist `is_terrain_pre`
// out into a single uniform mask instead of branching on a function call
// every tap (which forced two specialised loop bodies into the kernel).
bool any_hit_m(vec3 origin, vec3 dir, float t_max, uint mask) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          mask, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}
bool closest_hit_m(vec3 origin, vec3 dir, float t_max, uint mask,
                   out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          mask, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

// Closest-hit ray cast. Returns true on hit, fills out_t.
bool closest_hit(vec3 origin, vec3 dir, float t_max,
                 out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

// Closest-hit + material lookup. Returns instance custom index AND
// primitive index. The merged-static BLAS instance carries
// kStaticBlasInstSentinel as its custom index; the shader recovers the
// per-brush material via primitive_id / 12 in that case (12 tris/brush).
// Dynamic instances carry their materials-buffer slot directly.
bool closest_hit_material(vec3 origin, vec3 dir, float t_max,
                          out float out_t, out int out_inst, out int out_prim) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    out_inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    out_prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    return true;
}

// Sentinel for the merged-static BLAS instance вЂ” must match
// `kStaticBlasInstSentinel` in src/engine/vk_engine/rt.cpp. Picked at the
// top of the 24-bit instanceCustomIndex range so it never collides with a
// real materials-buffer slot.
const int kStaticBlasSentinel = 0xFFFFFF;
const int kCubeTrisPerBox     = 12;

void main() {
    // Default depth = rasterized depth; SPOM silhouette path may bump
    // to 1.0 below. The depth_greater qualifier on the gl_FragDepth
    // declaration keeps early-Z enabled regardless.
    gl_FragDepth = gl_FragCoord.z;
    // Screen-space motion vector вЂ” current_uv в€’ prev_uv. prev_uv comes from
    // perspective-divided prev_clip (smooth-interpolated by the rasterizer).
    // Behind-camera prev pixels (prev_clip.w в‰¤ 0) have no valid prev_uv вЂ”
    // fall back to zero motion; the future SVGF pass treats sentinel as
    // "no history available" and rebuilds variance from current-frame
    // neighborhood instead of reprojecting.
    {
        vec2 current_uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            vec2 prev_uv = prev_ndc * vec2(0.5, 0.5) + vec2(0.5);
            // Subtract jitter delta packed into restir_params.zw —
            // produces unjittered geometric motion that FSR3 expects.
            outMotion = (current_uv - prev_uv) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
    }

    vec3 N = normalize(vNormal);

    if (vEmissive.a > 0.5) {
        outColor = vec4(vColor + vEmissive.rgb, 1.0);
        return;
    }

    // Hoist the two scalar derivatives the shader uses repeatedly:
    //   * cam_dist appears 13× in branches; with the conditional code
    //     in between, the compiler often fails to CSE it.
    //   * L (sun direction unit vector) appears 4×.
    // Computing once at the top costs nothing extra on the fast paths
    // (early-out emissive returns above this line) and saves ~12 sqrt
    // and 3 inversesqrt per pixel on the heavy lit-surface path.
    float cam_dist = distance(vWorldPos, scene.camera_pos.xyz);
    vec3  L        = normalize(scene.sun_direction.xyz);
    // shading_pos = the effective surface position lighting sees.
    // For most fragments it equals vWorldPos. The SPOM path below
    // overrides it with the parallax-displaced point so RT shadow,
    // AO, and muzzle-flash rays trace from the actual brick crevice
    // instead of the flat geometric cube face — gives real
    // self-shadowing inside SPOM bricks at almost no cost.
    vec3 shading_pos = vWorldPos;

    bool is_terrain_pre = vTexParams.w > 1.5;

    // ---- Terrain debug visualisation (set in pause-menu Graphics) ----
    // Bypasses everything except a basic Lambert / normal preview so we
    // can isolate which downstream effect (RT shadow, AO, GI, slope
    // blend, cavity, fog, triplanar) is producing visible artifacts.
    if (is_terrain_pre) {
        int dbg = int(scene.grass_extra2.w + 0.5);
        if (dbg == 1) {
            // Pure Lambert with constant grey albedo. No RT, no AO,
            // no slope blend, no fog. If the artifact persists here
            // it's a geometry/normal problem; if it disappears the
            // bug is in one of the disabled effects.
            vec3 L = normalize(scene.sun_direction.xyz);
            float ndl = max(dot(N, L), 0.0);
            vec3 lit = vec3(0.55) * (0.25 + 0.75 * ndl);
            outColor = vec4(lit, 1.0);
            return;
        } else if (dbg == 2) {
            // Per-vertex normal as RGB. Adjacent triangles with
            // mismatched corner normals will show as visible colour
            // jumps вЂ” direct view into the LOD/Gouraud issue.
            outColor = vec4(N * 0.5 + 0.5, 1.0);
            return;
        } else if (dbg == 3) {
            // Geometric face normal (cross of world-pos derivatives,
            // sign-guarded for Vulkan y-down). Each large LOD triangle
            // is a uniform colour вЂ” confirms what flat-shaded geometry
            // looks like at distance.
            vec3 fn = cross(dFdx(vWorldPos), dFdy(vWorldPos));
            if (fn.y < 0.0) fn = -fn;
            outColor = vec4(normalize(fn) * 0.5 + 0.5, 1.0);
            return;
        } else if (dbg >= 4) {
            // Incremental debug modes: each higher number adds one
            // feature on top of "Lambert + hybrid RT/bake shadow",
            // so the user can pinpoint which feature reintroduces
            // distant-terrain artifacts.
            //   4  = base Lambert + hybrid shadow.
            //   5  = + height-band layer albedo (sand/grass/dirt/rock/snow).
            //   6  = + slope-rock blend (faded by distance).
            //   7  = + cavity AO from screen-space N derivatives (faded).
            //   8  = + triplanar texture detail (faded at distance).
            //   9  = + RTAO (faded at distance).
            //   10 = + GI bounce (faded at distance).
            vec3 L = normalize(scene.sun_direction.xyz);
            float ndl = max(dot(N, L), 0.0);
            float dist_to_cam = cam_dist;   // P5: use hoisted distance

            // PURE BAKE with PCF вЂ” 5-tap kernel and no threshold so
            // shadow edges blend smoothly across triangles instead of
            // snapping at the 0.5 step boundary (which looked like
            // adjacent triangles flipping fully-lit / fully-shadowed).
            ivec2 sz = textureSize(u_terrain_shadow, 0);
            const float side_b = 2048.0; // terrain world extent (supersample-invariant)
            vec2 uv_b = (vWorldPos.xz / side_b) + vec2(0.5);
            float sh = 0.0;
            if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
                all(lessThanEqual(uv_b, vec2(1.0)))) {
                vec2 texel = 1.0 / vec2(sz);
                float s = texture(u_terrain_shadow, uv_b).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2( 1.0,  0.0)).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2(-1.0,  0.0)).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2( 0.0,  1.0)).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2( 0.0, -1.0)).r;
                sh = s * 0.2;
            }

            // Albedo: grey at base, layer-blend from mode 5.
            vec3 base_col = vec3(0.55);
            if (dbg >= 5) {
                const vec3 sand   = vec3(0.78, 0.71, 0.50);
                const vec3 grass  = vec3(0.31, 0.45, 0.18);
                const vec3 dirt   = vec3(0.42, 0.30, 0.18);
                const vec3 rock   = vec3(0.40, 0.38, 0.36);
                const vec3 snow   = vec3(0.95, 0.95, 0.97);
                float h = vWorldPos.y;
                float t_sand = smoothstep(scene.terrain_h_low.x,  scene.terrain_h_low.y,  h);
                float t_dirt = smoothstep(scene.terrain_h_low.z,  scene.terrain_h_low.w,  h);
                float t_rock = smoothstep(scene.terrain_h_high.x, scene.terrain_h_high.y, h);
                float t_snow = smoothstep(scene.terrain_h_high.z, scene.terrain_h_high.w, h);
                base_col = mix(sand, grass, t_sand);
                base_col = mix(base_col, dirt, t_dirt);
                base_col = mix(base_col, rock, t_rock);
                base_col = mix(base_col, snow, t_snow);
            }
            // Slope-rock blend at mode 6+.
            if (dbg >= 6) {
                float slope = 1.0 - clamp(N.y, 0.0, 1.0);
                float steep = smoothstep(0.45, 0.75, slope);
                float steep_w = 1.0 - smoothstep(80.0, 200.0, dist_to_cam);
                base_col = mix(base_col, vec3(0.40, 0.38, 0.36), steep * steep_w);
            }
            // Cavity AO at mode 7+.
            if (dbg >= 7) {
                float curvature = -dot(dFdx(N), dFdx(vWorldPos)) -
                                  dot(dFdy(N), dFdy(vWorldPos));
                float cav_w  = 1.0 - smoothstep(80.0, 200.0, dist_to_cam);
                float cavity = clamp(0.5 - curvature * 0.4, 0.45, 1.0);
                base_col *= mix(1.0, cavity, cav_w);
            }
            // Triplanar texture detail at mode 8+.
            if (dbg >= 8) {
                vec3 detail = triplanar_sample(u_albedo[0], vWorldPos * 0.0625, N);
                float det_w = 1.0 - smoothstep(80.0, 200.0, dist_to_cam);
                base_col = mix(base_col, base_col * detail, det_w);
            }
            // AO at mode 9+ (single short ray, faded at distance).
            float ao_dbg = 1.0;
            if (dbg >= 9 && scene.rt_flags2.w > 0) {
                vec3 origin_ao = vWorldPos + N * 0.01;
                vec3 ao_dir = normalize(N + vec3(0.3, 0.0, 0.0));
                if (any_hit(origin_ao, ao_dir, 1.0)) ao_dbg = 0.7;
                float ao_far_t = smoothstep(80.0, 200.0, dist_to_cam);
                ao_dbg = mix(ao_dbg, 1.0, ao_far_t);
            }
            // GI at mode 10+ (single bounce, faded).
            vec3 gi_dbg = vec3(0.0);
            if (dbg >= 10) {
                vec3 origin_gi = vWorldPos + N * 0.01;
                vec3 gi_dir = normalize(N + vec3(0.0, 0.0, 0.5));
                if (!any_hit(origin_gi, gi_dir, 50.0)) {
                    gi_dbg = scene.sky_color.rgb * 0.2;
                }
                float gi_far_t = smoothstep(80.0, 200.0, dist_to_cam);
                gi_dbg = mix(gi_dbg, vec3(0.0), gi_far_t);
            }

            vec3 lit = base_col * (0.25 + 0.75 * ndl * (1.0 - sh)) * ao_dbg + gi_dbg;
            outColor = vec4(lit, 1.0);
            return;
        }
    }

    // (L = sun direction unit vector hoisted at top of main — used for
    // n_dot_l, shadow rays, GI bounces and slope material blends.)

    // --- Albedo + bump mapping (triplanar projection) ---
    // Choose object-space (texture glued to the model) for dynamic objects,
    // world-space (texture aligned to the world grid) for static brushes.
    int   albedo_idx_raw = int(vTexParams.x);
    int   normal_idx_raw = int(vTexParams.y);
    float scale          = max(0.001, vTexParams.z);
    // tex_params.w convention:
    //   0 = world-space triplanar (default вЂ” castle brushes)
    //   1 = object-space triplanar (dynamic crates so the texture
    //       rotates with the body)
    //   2 = terrain shading: sandв†’grassв†’dirtв†’rockв†’snow blended by
    //       world height + slope, with optional triplanar Ground054
    //       detail multiplied on top.
    bool  is_terrain     = vTexParams.w > 1.5;
    bool  obj_space      = !is_terrain && vTexParams.w > 0.5;
    // Terrain uses world-space triplanar like the static brushes вЂ” the
    // mesh model is identity so vObjectPos == vWorldPos already, but we
    // pick world-space explicitly so the height read uses true world Y.
    vec3  sample_pos     = (obj_space ? vObjectPos : vWorldPos) * scale;
    vec3  proj_n         = obj_space ? normalize(vObjectNormal) : N;
    bool  use_albedo     = albedo_idx_raw >= 0 && albedo_idx_raw < kTextureCount;
    bool  use_normal     = normal_idx_raw >= 0 && normal_idx_raw < kTextureCount &&
                           !obj_space && !is_terrain;
    // Texture sampling lives in branches gated on use_albedo / use_normal.
    // Sampler-array indexing on Vulkan requires a known-valid index; clamp
    // before indexing to keep the access well-defined regardless of any
    // hoisting the optimiser might do.
    vec3 albedo = vColor;
    if (is_terrain) {
        // ---- Terrain layer blend (height + slope) ----
        // Layer palette tuned for "natural earth" вЂ” desaturated, not too
        // toy-coloured. Heights match the heightmap params:
        //   plateau ~22, mountain crests ~120-140 (height_scale=140 in
        //   the generator). Sea level isn't real here, but we still
        //   give the bottom band a sandy hue.
        const vec3 sand   = vec3(0.78, 0.71, 0.50);
        const vec3 grass  = vec3(0.31, 0.45, 0.18);
        const vec3 dirt   = vec3(0.42, 0.30, 0.18);
        const vec3 rock   = vec3(0.40, 0.38, 0.36);
        const vec3 snow   = vec3(0.95, 0.95, 0.97);

        float h = vWorldPos.y;
        // Slope: 0 = flat (normal up), 1 = vertical wall.
        float slope = 1.0 - clamp(N.y, 0.0, 1.0);

        // ---- Automatic beach band from the live water level ----
        // scene._scene_pad[0] aliases the C++ water_params
        // (.x = water enabled, .y = level). Sand owns everything from
        // below the waterline up into a band above it; grass/dirt NEVER
        // appear under or right at the water, with zero manual tuning.
        //
        // The band edge is NOT a flat height contour (that read as a
        // straight "line" up slopes): a low-frequency world-XZ meander
        // wobbles the threshold so the sand/grass boundary follows an
        // irregular natural coastline. It's also slope-aware — a broad
        // beach on gentle ground, a thin sand strip on steep coast — so
        // it visually hugs the shore instead of ringing the terrain at
        // one elevation.
        float wtr_lvl = scene._scene_pad[0].y;
        float wtr_on  = step(0.5, scene._scene_pad[0].x);
        // (1) HARD guarantee: everything at or below the water line is
        // sand, ALWAYS — independent of any noise. This is what stops
        // grass ever appearing under water.
        float under = 1.0 - smoothstep(wtr_lvl - 0.20, wtr_lvl + 0.05, h);
        // (2) The sand strip ABOVE water. A flat height threshold on a
        // slope is a straight iso-line; we displace the threshold by a
        // strong MID-frequency multi-octave world-XZ wander whose
        // amplitude is much larger than the local height span across
        // the strip, so the sand→grass edge swings widely in 2D and
        // reads as an irregular coast, not a contour. Soft (±1 m) so
        // it's a sand→grass gradient, not a hard line.
        // Big sweeping bays + mid coves + fine crenellation. Amplitude
        // is deliberately huge vs the slope so the sand→grass edge
        // swings tens of metres in world-XZ — it cannot read as a
        // straight contour even on a uniform ramp.
        float wrp = (tnoise(vWorldPos.xz * 0.008) - 0.5) * 14.0   // bays ~125 m
                  + (tnoise(vWorldPos.xz * 0.028) - 0.5) * 6.0    // coves ~36 m
                  + (tnoise(vWorldPos.xz * 0.090) - 0.5) * 2.0;   // crenel ~11 m
        float topH  = wtr_lvl + 2.0 + wrp;
        float above = 1.0 - smoothstep(topH - 1.5, topH + 1.5, h);
        float beach = wtr_on * max(under, above);

        // ---- Layer-transition break-up ----
        // Without a per-pixel offset, layer boundaries form perfectly
        // horizontal contour lines that read as obvious "stripes" on
        // gentle slopes. We jitter each transition's effective height
        // by a low-frequency hash on world XZ вЂ” same as how natural
        // soils don't change at exactly the same elevation everywhere.
        // The jitter is in metres and matched to ~half the smoothstep
        // width so it breaks up the line without erasing the gradient.
        // Multi-octave, large-amplitude jitter so the layer boundaries
        // dissolve into natural patches instead of horizontal contour
        // "line banding". (The old ±3 m single-ish octave was too weak
        // once the detail/noise layers were stripped.)
        // Only low-frequency octaves: the boundary should *meander*
        // smoothly (a wandering coastline), not flicker per-metre. The
        // old 0.30-freq (~3 m) octave made the threshold jump up/down
        // every few metres, which with real per-texel materials reads
        // as a spiky salt-and-pepper mix of two grounds instead of a
        // gradient. Drop it; keep a broad 40 m + gentle 11 m wander,
        // smaller amplitude so it never overshoots the blend ramp.
        float n0 = tnoise(vWorldPos.xz * 0.012) - 0.5;
        float n1 = tnoise(vWorldPos.xz * 0.045 + vec2(13.7, 41.3)) - 0.5;
        float jitter = n0 + 0.35 * n1;                // ~±0.7, slow
        float jh = h + jitter * 3.0;                  // ±~2 m meander

        // Widen every band transition into a LONG gradient. On steep
        // mountain flanks a narrow height band projects to a thin spiky
        // line on screen; a very wide ramp (≈5.6x the configured span,
        // around the same centre) turns grass→rock→snow into a smooth
        // fade. The +8 m floor guarantees a usable ramp even if a band's
        // start/end are configured close together.
        #define WIDE_BAND(a, b, x) smoothstep(                              \
            0.5 * ((a) + (b)) - (1.4 * ((b) - (a)) + 8.0),                  \
            0.5 * ((a) + (b)) + (1.4 * ((b) - (a)) + 8.0), (x))
        float t_sand = WIDE_BAND(scene.terrain_h_low.x,  scene.terrain_h_low.y,  jh);
        float t_dirt = WIDE_BAND(scene.terrain_h_low.z,  scene.terrain_h_low.w,  jh);
        float t_rock = WIDE_BAND(scene.terrain_h_high.x, scene.terrain_h_high.y, jh);
        float t_snow = WIDE_BAND(scene.terrain_h_high.z, scene.terrain_h_high.w, jh);
        #undef WIDE_BAND

        vec3 base = mix(sand, grass, t_sand);
        base = mix(base, dirt, t_dirt);
        base = mix(base, rock, t_rock);
        base = mix(base, snow, t_snow);

        // ---- Procedural surface detail noise ----
        // Two-octave value noise on world XZ. Fine octave (0.45 m
        // freq) modulates albedo within ±8% so flat patches don't
        // read as a single uniform colour; coarser octave (3 m freq)
        // adds large-scale tonal variation that survives at distance.
        // Cheap (4 ign() calls) and unlike the triplanar detail it
        // doesn't depend on N so it's stable across LOD-mismatched
        // triangle edges.
        // Now that the Ground054 texture tiles at a sane size it
        // carries the surface detail; this procedural modulation only
        // needs to subtly break up flat patches. The old coarse 3 m
        // octave at ±8% read as big blurry blotches OVER the texture —
        // drop it almost entirely (mostly the fine 0.45 m octave) and
        // halve the amplitude.
        float dn_fine = tnoise(vWorldPos.xz * 2.2 + vec2(11.0, 23.0));
        float dn_far  = tnoise(vWorldPos.xz * 0.33 + vec2(91.0, 47.0));
        float dn = mix(dn_far, dn_fine, 0.9);          // 0..1, mostly fine
        float noise_amp = 0.07;                        // ±3.5% albedo swing
        base *= 1.0 + (dn - 0.5) * noise_amp;

        // Steep faces become rocky regardless of altitude. Slope jitter
        // breaks up the cliff/grass border the same way the height
        // jitter breaks up horizontal layer transitions.
        // Fade slope-driven rock with camera distance: at LOD 2/3 the
        // Gouraud-interpolated normals between widely-spaced verts
        // produce per-triangle slope swings, and adjacent large
        // triangles end up classified as rock vs grass at random,
        // reading as patchy "different faces". Past ~120 m we let the
        // height-band layering carry the look вЂ” visually identical at
        // distance, no per-triangle classification noise.
        // Slope→rock is the main grass↔rock spike: `slope` comes from
        // the per-pixel mesh normal, which flickers across the dense
        // sculpt, so a tight slope window sprinkles rock through grass.
        // Use a LOW-frequency meander (not the old ~11 m hash) and a
        // very wide smoothstep so the cliff/grass border is a long soft
        // gradient, and cap the max so slope never fully erases grass.
        float slope_jitter = (tnoise(vWorldPos.xz * 0.02 + vec2(7.0, 19.0)) - 0.5) * 0.12;
        float steep_raw = smoothstep(0.40 + slope_jitter, 1.02 + slope_jitter, slope);
        float steep_dist = cam_dist;    // P5: use hoisted distance
        float steep_w = 1.0 - smoothstep(80.0, 200.0, steep_dist);
        float steep = steep_raw * steep_w * 0.85;
        base = mix(base, rock, steep);

        // ---- Cavity AO from local height curvature ----
        // dFdx(N)В·dFdx(vWorldPos) gives concavity per fragment. Adds
        // "natural shadowing in cracks" for free near the camera, but
        // on distant LOD-2/3 chunks the screen-space derivatives of
        // Gouraud-interpolated normals between widely-spaced verts
        // become erratic (large per-pixel jumps), and the cavity term
        // oscillates 0.45в†”1.0 across adjacent triangles вЂ” visible as
        // patchy dark faces on far ridges. Fade the effect out past
        // ~120 m so distant terrain reads as smooth.
        // DISABLED: this curvature term uses dFdx(N) on the 1 m faceted
        // mesh, so every triangle gets a different darkening — that is
        // the blocky "pixelated dark overlay baked into the heightmap"
        // (a faceted-normal artifact, not real AO). Removed for terrain.
        // float curvature = -dot(dFdx(N), dFdx(vWorldPos)) -
        //                   dot(dFdy(N), dFdy(vWorldPos));
        // float cav_w     = 1.0 - smoothstep(80.0, 200.0, cam_dist);
        // float cavity    = clamp(0.5 - curvature * 0.4, 0.45, 1.0);
        // base *= mix(1.0, cavity, cav_w);

        // ---- Per-pixel FBM-eroded micro-detail ----
        // The rasterised mesh has 1 m vertex spacing, so anything
        // finer than that gets flattened between vertices — sculpted
        // / brushed terrain reads as "faceted" compared with the
        // raymarched terrain (which evaluates noise per pixel). This
        // pass evaluates the same FBM-with-derivative-erosion in the
        // fragment shader and pushes its analytical gradient into N,
        // adding sub-cell ridges/valleys at any zoom level. Geometry
        // stays at mesh resolution; only the SHADING normal moves.
        // Faded with camera distance so distant chunks (where 1-pixel
        // surfaces span many noise cycles) don't shimmer.
        // DISABLED: this FBM micro-detail pushed a per-pixel noise
        // gradient into the shading normal AND ridge-darkened the
        // albedo. On the faceted 1 m mesh it read as a noisy
        // "low-resolution heightmap baked in" overlay. Removed so the
        // terrain is just the clean height-band colour + fine grain.
        // (Re-introduce later as a proper, toggleable detail-normal.)
        // {
        //     float det_w = 1.0 - smoothstep(40.0, 120.0, cam_dist);
        //     if (det_w > 0.001) {
        //         vec3 dn = terrain_detail_fbm(vWorldPos.xz, 0.30, 6);
        //         N = normalize(N + vec3(-dn.y, 0.0, -dn.z) * 0.25 * det_w);
        //         float ridge = clamp(dn.x * 0.6 + 0.5, 0.0, 1.0);
        //         base *= mix(1.0, mix(0.85, 1.10, ridge), det_w);
        //     }
        // }

        // Optional triplanar detail using the engine's Ground054 albedo
        // (texture index 0). When textures are off (use_albedo=false)
        // we just show the layer-blended colour.
        // The Ground054 triplanar "detail" overlay is disabled for
        // terrain: that texture is too low-res to tile per-metre over a
        // 2 km field — at any tiling it read as a big pixelated stamp
        // on top of the ground. The look is now the height-band layer
        // colours + the subtle procedural fine-grain noise above +
        // anisotropic filtering, which stays clean at all distances.
        // ---- Full material splatting ----
        // Real per-texel rock/grass/dirt/sand/snow (procedurally baked,
        // seamless tiling) blended by the SAME height/slope weights that
        // drive the art-directed band colour `base`. `base` is reused as
        // the distance-fade target so far chunks settle to the clean
        // band colour (no tiling shimmer, no LOD-facet normal noise).
        // Runtime knobs (free slots of spom_params — see descriptors.cpp):
        //   .y = material strength (0 = old flat band look, 1 = full),
        //   .z = metres per material repeat,
        //   .w = detail-normal strength.
        float g_str  = clamp(scene.spom_params.y, 0.0, 1.0);
        float g_tile = max(scene.spom_params.z, 0.25);
        vec2  uvm = vWorldPos.xz / g_tile;          // material repeat
        vec2  uvM = vWorldPos.xz / (g_tile * 8.0);  // macro break-up

        // ---- Parallax-occlusion (per-pixel displacement) ----
        // Pixel-displaces the ROCK heightfield (baked into the rock
        // normal map's alpha, with extra eroded gullies) so cliffs /
        // snow read as carved, weathered rock without any geometry.
        // Cost is bounded three ways: (1) only where it matters —
        // rocky/snow pixels (grass/sand have ~0 weight → skipped, the
        // common case); (2) step count ramps with distance to ZERO by
        // ~45 m (a per-pixel LOD); (3) 1 height sample/step + 2 refine.
        // Strength comes from the terrain draw's pc.tex_params.x (the
        // terrain path ignores that slot otherwise); 0 = off.
        vec3 Tt = normalize(vec3(1.0, 0.0, 0.0) - N * N.x);
        vec3 Bt = cross(N, Tt);
        float pom_str = clamp(vTexParams.x, 0.0, 1.0);
        // Parallax everywhere rock/dirt/snow contribute (a small base
        // floor too so even gentle rocky ground gets some), ramped over
        // a longer range so it doesn't vanish the moment you step back.
        float rockiness = max(max(t_rock, steep),
                              max(t_dirt * 0.6, t_snow * 0.7));
        rockiness = max(rockiness, 0.25 * pom_str);
        float pom_w = pom_str * rockiness *
                      (1.0 - smoothstep(35.0, 90.0, cam_dist));
        if (pom_w > 0.01) {
            vec3 Vw = normalize(scene.camera_pos.xyz - vWorldPos);
            vec3 Vt = vec3(dot(Vw, Tt), dot(Vw, Bt), dot(Vw, N));
            Vt.z = max(abs(Vt.z), 0.20);
            int   steps = int(mix(16.0, 40.0, pom_w));
            float layer = 1.0 / float(steps);
            // Slightly shallower so the staircase isn't exaggerated;
            // the binary refine below removes the residual banding.
            vec2  Pmax  = (Vt.xy / Vt.z) * (0.12 * pom_w);  // max uv shift
            vec2  dtex  = Pmax * layer;
            vec2  uvc   = uvm;
            float curD  = 0.0;
            for (int i = 0; i < steps; ++i) {
                float hh = texture(u_normal[kMatRock], uvc).a;
                if (curD >= 1.0 - hh) break;
                uvc -= dtex;
                curD += layer;
            }
            // Binary-search refine between the last layer ABOVE the
            // surface (lo) and the first BELOW it (hi). The previous
            // single linear interpolation left the ray landing on
            // discrete march planes — visible as parallel "extrude"
            // step lines all along the march direction. 6 bisections
            // converge to the true crossing → no banding.
            vec2  lo = uvc + dtex;   float dLo = curD - layer;  // above
            vec2  hi = uvc;          float dHi = curD;          // below
            for (int b = 0; b < 6; ++b) {
                vec2  mid  = (lo + hi) * 0.5;
                float dMid = (dLo + dHi) * 0.5;
                float hM   = texture(u_normal[kMatRock], mid).a;
                if (dMid < 1.0 - hM) { lo = mid; dLo = dMid; }
                else                 { hi = mid; dHi = dMid; }
            }
            vec2  duv = (lo + hi) * 0.5 - uvm;
            uvm += duv;
            uvM += duv * 0.125;
        }

        #define MAT_A(i) (texture(u_albedo[i], uvm).rgb * 0.75 + \
                          texture(u_albedo[i], uvM).rgb * 0.25)
        vec3 m = MAT_A(kMatSand);
        m = mix(m, MAT_A(kMatGrass), t_sand);
        m = mix(m, MAT_A(kMatDirt),  t_dirt);
        m = mix(m, MAT_A(kMatRock),  t_rock);
        m = mix(m, MAT_A(kMatSnow),  t_snow);
        m = mix(m, MAT_A(kMatRock),  steep);
        m = mix(m, MAT_A(kMatSand),  beach);   // auto sand at the waterline
        #undef MAT_A

        // Tint the neutral material toward the art-directed band hue so
        // the overall palette stays controlled, keeping the per-texel
        // detail as a multiplicative term; then fade to the flat band
        // colour with distance.
        // Force the band/flat colour to sand at the waterline too, so
        // the distance-fade target and the g_str=0 look also have no
        // grass under water.
        base = mix(base, sand, beach);
        float m_lum  = max(dot(m, vec3(0.299, 0.587, 0.114)), 1e-3);
        vec3  tinted = mix(m, m * (base / m_lum), 0.55);
        // g_str blends old flat band colour ↔ full per-texel material
        // (g_str = 0 is the exact previous look — an A/B toggle).
        vec3  splat = mix(base, tinted, g_str);
        // Fade to the flat band colour only at long range so mid-
        // distance terrain still shows real material (the prior
        // 110→300 m fade made everything past the plateau look flat).
        float far_fade = smoothstep(280.0, 760.0, cam_dist);
        albedo = mix(splat, base, far_fade);

        // ---- Detail normal mapping ----
        // Splat the matching baked normal maps, rotate into world space
        // via a tangent frame built on the geometric terrain normal.
        // Strength fades out by ~140 m: distant LOD chunks have widely
        // spaced verts whose interpolated N makes per-texel normals
        // shimmer (the old faceted-overlay artifact) — let the band
        // colour carry the distance.
        #define MAT_N(i) (texture(u_normal[i], uvm).xyz * 2.0 - 1.0)
        vec3 nm = MAT_N(kMatSand);
        nm = mix(nm, MAT_N(kMatGrass), t_sand);
        nm = mix(nm, MAT_N(kMatDirt),  t_dirt);
        nm = mix(nm, MAT_N(kMatRock),  t_rock);
        nm = mix(nm, MAT_N(kMatSnow),  t_snow);
        nm = mix(nm, MAT_N(kMatRock),  steep);
        nm = mix(nm, MAT_N(kMatSand),  beach);   // sand normal at waterline
        #undef MAT_N

        // ---- Shore-following beach ripples ----
        // Real sand ripples, but the phase is HEIGHT ABOVE WATER
        // (h - waterLevel), so ripple crests are iso-height contours.
        // The waterline itself is a height contour, so the ripples are
        // automatically parallel to the coast and curve with every bay
        // — never a fixed world-axis stripe. Only on near-flat sand by
        // the water (beach), fading out up the slope; a gentle noise
        // wander keeps them from looking mechanical.
        // Phase = height above water → crests are iso-height contours,
        // so the lines run parallel to the coast and curve with it
        // (never a fixed world axis). Independent scales: sand uses
        // pc.grass_params.x, grass uses pc.grass_params.y (0 = off).
        // Both fade out on slopes; the normal is tilted along the world
        // uphill (height-gradient) direction so the lines catch light
        // as ridges.
        float flatv = clamp((N.y - 0.55) * 2.5, 0.0, 1.0);
        vec2  up2   = -N.xz;
        float ulen  = length(up2);
        if (ulen > 1e-3 && flatv > 0.001) {
            vec3  uw  = vec3(up2.x / ulen, 0.0, up2.y / ulen);
            vec2  ut  = vec2(dot(uw, Tt), dot(uw, Bt));
            float wob = (tnoise(vWorldPos.xz * 0.05) - 0.5) * 3.0;
            // Distance LOD: the ripple is sin() of world height, so far
            // away its spatial frequency outruns the pixel grid and
            // moirés into banding. Fade it out by ~110 m so distant
            // sand is smooth (no bands); near sand keeps the ripple.
            float rip_dist = 1.0 - smoothstep(35.0, 110.0, cam_dist);
            // Sand ripples: phase = height above water → crests are
            // iso-height ≈ parallel to the shore, curving with it.
            // Smooth (no texture-UV rotation → no per-triangle seams);
            // the normal tilt direction is the tangent-space uphill.
            float s_sc = pc.grass_params.x > 0.01 ? pc.grass_params.x : 9.0;
            float s_w  = beach * flatv * rip_dist;
            if (s_w > 0.001) {
                float ph = (h - wtr_lvl) * s_sc + wob;
                albedo  *= 1.0 + sin(ph) * 0.10 * s_w;
                nm.xy   += ut * (cos(ph) * 0.7 * s_w);
            }
            // Grass contour rows — independent scale, 0 disables.
            float g_sc = pc.grass_params.y;
            if (g_sc > 0.01) {
                float g_w = t_sand * (1.0 - t_dirt) * (1.0 - steep)
                            * (1.0 - beach) * flatv * rip_dist;
                if (g_w > 0.001) {
                    float ph = (h - wtr_lvl) * g_sc + wob;
                    albedo *= 1.0 + sin(ph) * 0.05 * g_w;
                    nm.xy  += ut * (cos(ph) * 0.4 * g_w);
                }
            }
            nm = normalize(nm);
        }

        float nf = (1.0 - smoothstep(70.0, 180.0, cam_dist)) *
                   clamp(scene.spom_params.w, 0.0, 1.0) * g_str;
        // Tt/Bt were built above for the parallax march — reuse them.
        vec3  Nt = normalize(Tt * nm.x + Bt * nm.y + N * max(nm.z, 0.2));
        N = normalize(mix(N, Nt, nf));
        // Up-bias floor so terrain never self-shadows to black under the
        // RT sun query (hard-won fix — do not remove).
        N.y = max(N.y, 0.18);
        N = normalize(N);
    } else {
        // SPOM (parallax-occlusion) — for any material that has a height
        // map registered in u_height[]. height_idx_for_albedo() returns
        // -1 for non-SPOM materials so they fall through to the standard
        // triplanar path. obj_space brushes (dyn props) skip SPOM because
        // their normal isn't world-aligned; terrain skips because it has
        // its own raymarched normal path.
        int spom_h_idx = height_idx_for_albedo(albedo_idx_raw);
        bool spom_path = use_albedo && spom_h_idx >= 0 && !obj_space &&
                         !is_terrain;
        if (spom_path) {
            // pc.model is translate * scale (no rotation for static brushes,
            // and obj_space — which would rotate — already failed the gate
            // above). Each column's length is the brush's world-space size
            // along that local axis, which for axis-aligned brushes equals
            // the face dimensions we need for the silhouette discard.
            vec3 face_size = vec3(length(pc.model[0].xyz),
                                  length(pc.model[1].xyz),
                                  length(pc.model[2].xyz));
            int axis;
            vec3 spom_T, spom_B, spom_face_n;
            bool spom_disc;
            vec3 spom_world;
            vec2 spom_uv0;
            vec2 spom_uv_off = spom_uv(sample_pos, proj_n, face_size, scale,
                                        spom_h_idx,
                                        axis, spom_T, spom_B, spom_face_n,
                                        spom_disc, spom_world, spom_uv0);
            // Silhouette overhang resolution.
            //
            // Two competing visual goals:
            //   - At true OUTER corners (sky/scene behind), we want the
            //     "bricks bumped past the geometric edge" trick: write
            //     outColor=0 + depth=1 → compose substitutes sky, the
            //     silhouette reads as bumpy bricks not a flat clipped
            //     edge (the "S" in SPOM).
            //   - At packed INNER seams (wall↔tower, gate↔wall) the
            //     adjacent brush fills the gap. The sky substitution
            //     punched a black/sky pane through the joint. The 25 %
            //     pad in spom_uv() catches most of these but tightly-
            //     packed brushwork still triggers it.
            //
            // Distinguishing the two: cast a SHORT ray query from
            // vWorldPos along the parallax-extension direction
            // (vWorldPos → spom_world). If anything is within ~30 cm
            // we're at an inner seam → flat fallback. Otherwise it's
            // a true outer corner → silhouette extension.
            //
            // 1 ray per silhouette pixel. Silhouette pixels are a small
            // % of the screen (only at brush edges) so the cost is
            // bounded; a few thousand extra rays per frame is well
            // under any GPU-budget concern.
            if (spom_disc) {
                // Reworked seam ray (trade-off, not a clean fix). Old
                // version fired the ray purely laterally along this
                // wall's tangent plane — at coplanar wall segments
                // meeting edge-to-edge it skimmed parallel to both
                // faces and never hit the neighbour, so silhouette
                // discarded → user saw dark sky panes at the seam.
                //
                // New version fires perpendicularly INTO the wall plane
                // (-face_n) FROM the displaced lateral point. This
                // reliably catches a coplanar neighbour: the neighbour's
                // face is in the same plane, our ray crosses it within
                // ~10 mm. BUT the discard fires on a SINGLE-axis
                // overshoot (vis_T OR vis_B past the padded edge), so
                // the displaced point is often still inside the current
                // brush on the other axis. The -face_n probe then hits
                // the current brush's own front face, has_neighbour
                // reports true, and we fall back to flat un-parallaxed
                // brick. The visible net: no more dark sky panes (the
                // user's complaint), at the cost of silhouette
                // extension at true outside corners — those now also
                // fall back to flat instead of showing sky-through
                // bumped bricks. Accepted trade-off; revisit only if
                // outside-corner silhouette becomes desirable again
                // (would need instance-ID rejection or per-axis ray
                // dispatch — see commit notes).
                bool has_neighbour = false;
                {
                    vec3 origin = spom_world + spom_face_n * 0.005;
                    vec3 dir    = -spom_face_n;
                    rayQueryEXT rq_seam;
                    rayQueryInitializeEXT(rq_seam, topLevelAS,
                                          gl_RayFlagsTerminateOnFirstHitEXT |
                                          gl_RayFlagsOpaqueEXT,
                                          0xFFu,
                                          origin, 0.001, dir, 0.12);
                    while (rayQueryProceedEXT(rq_seam)) {}
                    has_neighbour =
                        rayQueryGetIntersectionTypeEXT(rq_seam, true) ==
                        gl_RayQueryCommittedIntersectionTriangleEXT;
                }
                if (!has_neighbour) {
                    // True outer corner — silhouette extension. Sky
                    // shows behind the bumped bricks. depth=1 lets
                    // compose's sky branch (depth >= 0.99999) paint
                    // the sky color over this pixel.
                    outColor     = vec4(0.0);
                    outMotion    = vec2(0.0);
                    gl_FragDepth = 1.0;
                    return;
                }
                // Inner seam — fall back to the un-parallaxed wall so
                // we don't black-hole the joint. Lose silhouette bricks
                // at this specific edge; the neighbour brush conceals
                // the loss.
                spom_uv_off = spom_uv0;
                spom_world  = vWorldPos;
            }
            // RT origins from the brick crevice — but only on WALLS
            // (axis 0 / 2). On floors (axis 1) the lateral tangent
            // offset can push the AO/shadow ray origin past the floor
            // brush's XZ extent, where rays land in adjacent wall
            // colliders and produce noise that reads as gray patches
            // when TAA history is unavailable (camera moving / falling).
            // Floors keep shading_pos = vWorldPos so AO/shadow rays
            // trace from the actual rasterised hit point.
            if (axis != 1) shading_pos = spom_world;
            int  a_idx = clamp(albedo_idx_raw, 0, kTextureCount - 1);
            vec3 tex_albedo = texture(u_albedo[a_idx], spom_uv_off).rgb;
            albedo = tex_albedo * vColor;
            if (use_normal) {
                vec3 n_t = texture(u_normal[a_idx], spom_uv_off).rgb * 2.0 - 1.0;
                N = normalize(spom_T  * n_t.x +
                              spom_B  * n_t.y +
                              spom_face_n * n_t.z);
            }
        } else if (use_albedo) {
            int  albedo_idx = clamp(albedo_idx_raw, 0, kTextureCount - 1);
            vec3 tex_albedo = triplanar_sample(u_albedo[albedo_idx],
                                               sample_pos, proj_n);
            albedo = tex_albedo * vColor;
            if (use_normal) {
                int  normal_idx = clamp(normal_idx_raw, 0, kTextureCount - 1);
                N = triplanar_normal(u_normal[normal_idx], sample_pos, N);
            }
        } else if (use_normal) {
            int  normal_idx = clamp(normal_idx_raw, 0, kTextureCount - 1);
            N = triplanar_normal(u_normal[normal_idx], sample_pos, N);
        }
    }

    // Corner softening for axis-aligned brushes. Cube meshes have one
    // flat normal per face, so 90° outside corners (e.g. tower-to-
    // wall, gate-block-to-wall) read as a hard vertical seam — Lambert
    // n·L jumps abruptly across the edge. Bend the shading normal
    // toward the AVERAGE of (this face, the perpendicular face whose
    // edge we're close to) inside the last `kCornerWidth` of the
    // brush extent. Costs ~10 ALU per fragment; gives a smooth
    // ~25 m m-scale wrap around outside corners.
    if (!is_terrain && !obj_space) {
        vec3 face_size_w = vec3(length(pc.model[0].xyz),
                                length(pc.model[1].xyz),
                                length(pc.model[2].xyz));
        vec3 brush_center = vec3(pc.model[3].x, pc.model[3].y, pc.model[3].z);
        vec3 rel = vWorldPos - brush_center;
        // Normalised position along each axis: -1 (one face) .. +1 (other).
        vec3 norm_pos = rel / max(face_size_w * 0.5, vec3(1e-3));
        vec3 absN = abs(N);
        int dom = (absN.x >= absN.y)
                    ? (absN.x >= absN.z ? 0 : 2)
                    : (absN.y >= absN.z ? 1 : 2);

        const float kCornerWidth = 0.18;
        vec3 corner_target = N;
        float best_t = 0.0;
        // Walk the two non-dominant axes; each is a candidate "edge"
        // we might be approaching. Strongest contributor wins.
        for (int a = 0; a < 3; ++a) {
            if (a == dom) continue;
            float p      = norm_pos[a];
            float edge_t = clamp((abs(p) - (1.0 - kCornerWidth)) / kCornerWidth,
                                  0.0, 1.0);
            if (edge_t > best_t) {
                best_t = edge_t;
                vec3 nbr = vec3(0.0);
                nbr[a] = sign(p);
                corner_target = normalize(N + nbr);
            }
        }
        // 0.5 is "full lerp at the edge". Higher = rounder, lower =
        // sharper. 0.5 reads as a believable subtle bevel without
        // making corners look like a sphere.
        N = normalize(mix(N, corner_target, best_t * 0.5));
    }

    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    // Split ambient into a sky-derived component (fades when the surface
    // is enclosed вЂ” sky_factor below) and a constant ground/reflected
    // component that exists indoors too (a real room has a floor that
    // reflects light, not a black void). Combining them later avoids
    // the "AO * sky_factor" double-darkening that crushed interior
    // edges to black.
    vec3 ambient_ground = scene.ambient.rgb * scene.rt_params.z;
    vec3 ambient_sky    = scene.sky_color.rgb * 0.45 * scene.rt_params.z;

    // Direct lighting uses pure Lambert (max(0, NВ·L)). The "wrap" lift
    // for back-of-mountain pixels is applied LATER as a sky-tinted
    // ambient bounce вЂ” applying it to direct here would produce the
    // "back-face fully lit while front-face shadow-rayed" artifact:
    // back faces have n_dot_l_raw = 0 so no shadow ray fires, but the
    // wrap pushes their effective n_dot_l above zero and the missing
    // shadow term means full sun colour leaks through. Front-facing
    // pixels resolved correctly via the shadow ray. Adjacent pixels
    // looking radically different is what the user reported.
    float n_dot_l_raw = max(dot(N, L), 0.0);
    float n_dot_l = n_dot_l_raw;
    // Terrain shading contrast: pow on n_dot_l. exp=1 → standard
    // Lambert; exp>1 sharpens the lit→shadow transition by darkening
    // grazing slopes faster. Brushes/dyn-props keep linear Lambert.
    if (is_terrain_pre) {
        float contrast = max(0.25, scene.terrain_extra.x);
        // Fast paths for the common slider values:
        //   contrast == 1 → no-op (default; most users)
        //   contrast == 2 → just n_dot_l²
        // Anything else falls through to the (slower) pow().
        // Common slider stops get cheap dedicated paths (avoids the
        // exp+log of pow). Anything not covered falls through to pow.
        if (contrast > 1.99 && contrast < 2.01) {
            n_dot_l = n_dot_l * n_dot_l;
        } else if (contrast < 1.01) {
            // identity — skip pow entirely
        } else if (contrast > 2.99 && contrast < 3.01) {
            // P11: x³ via two muls
            n_dot_l = n_dot_l * n_dot_l * n_dot_l;
        } else if (contrast > 1.49 && contrast < 1.51) {
            // P11: x^1.5 = x·sqrt(x), one mul + one sqrt vs full pow
            n_dot_l = n_dot_l * sqrt(n_dot_l);
        } else if (contrast > 3.99 && contrast < 4.01) {
            // P11: x⁴ via two squarings
            float n2 = n_dot_l * n_dot_l;
            n_dot_l = n2 * n2;
        } else {
            n_dot_l = pow(n_dot_l, contrast);
        }
    }

    // Per-pixel seed WITH the frame counter вЂ” TAA accumulates over ~8
    // frames, so animating the noise lets temporal averaging resolve the
    // residual dither into a clean signal. We tried a frame-stable seed
    // earlier (less moment-to-moment shimmer) but it left a fixed dither
    // pattern visible on flat surfaces; with TAA on (default) the temporal
    // average wins. We mod the frame counter by a small power-of-two so
    // the cycle is short and TAA always reaches steady state quickly.
    uvec3 seed_base = uvec3(uint(gl_FragCoord.x),
                            uint(gl_FragCoord.y),
                            uint(scene.rt_flags.w) & 7u);

    // (cam_dist already hoisted at the top of main — single source of
    // truth for distance-from-camera across LOD, terrain blending, AO
    // distance fade, GI sampling, etc.)

    // --- Sun shadow (RT, contact-hardening / PCSS-style) ---
    // 1. BLOCKER SEARCH: a handful of cheap closest-hit rays in a wide cone
    //    around L; record the average distance to the occluder.
    // 2. PENUMBRA ESTIMATE: cone half-angle в€ќ avg_blocker_distance Г— softness.
    //    Close-by occluders give a small cone (sharp shadow); far ones give
    //    a wide cone (soft shadow). Real-world effect: shadow under a pole
    //    is razor-sharp, the same shadow stretched across the floor is fuzzy.
    // 3. SHADOW RAYS: stratified sampling inside the size-adapted cone.
    float shadow = 0.0;
    // Terrain receivers always fire RT shadows now — but they use
    // mask 0x02 so the ray queries skip the terrain BLAS entirely
    // (which is what used to false-hit at distance from LOD/BLAS
    // mismatch). The bake handles terrain self-shadow; the RT
    // ray-queries handle castle / dyn-props at every distance, so
    // box / castle shadows on terrain no longer fade out past 80 m.
    float terrain_dist = cam_dist;         // P5: use hoisted distance
    // P10: at near-terminator the surface is already at <4% direct
    // light, so PCSS softness is invisible. Skip the entire 4-blocker
    // + N_s shadow-ray block. shadow stays at the default 0; the
    // direct *= (1 - shadow) * n_dot_l math collapses to the Lambert
    // falloff, which is what dominates here anyway.
    // Gate on shadow_strength too — both `shadow` (line ~1501) and
    // `sh_bake` (line ~1529) are multiplied by rt_params.w at the
    // end. With strength=0 the entire blocker + PCSS + bake-PCF block
    // was wasting ~12 RT rays + 9 texture taps per pixel for a final
    // result of zero.
    if (n_dot_l_raw > 0.04 && scene.rt_flags.x != 0 &&
        scene.rt_params.w > 1e-3) {
      if (true) {   // (kept block for diff hygiene — always-on now)
        // Roadmap item #4 Phase 3: when the half-rate shadow toggle is
        // on AND this is a brush/dyn surface (NOT terrain — terrain
        // keeps its hybrid bake/RT path further below), sample the
        // pre-traced shadow_lr image at the matching screen UV. The
        // producer already multiplied by shadow_strength so the value
        // drops straight into `shadow`. Skips the entire blocker+PCSS
        // RT-ray block — the whole point of the half-rate pass.
        if (scene.terrain_local_info.z > 0.5 && !is_terrain_pre) {
            vec2 hr_uv = gl_FragCoord.xy / scene.viewport.xy;
            shadow = textureLod(u_shadow_lr, hr_uv, 0.0).r;
        } else {
        // Base sample count from slider, distance-LOD'd. terrain_extra.y
        // is a multiplier on the BASE count that's applied before LOD
        // reduction, so close fragments get N×multiplier rays while far
        // ones still drop toward 1.
        float near_mult = max(1.0, scene.terrain_extra.y);
        // Cap base sample count: TAA accumulates ~8 frames of jitter, so
        // anything past ~16 effective samples per pixel falls below the
        // perceptual noise floor. Ultra preset (slider=40) × near_mult=4
        // would otherwise hand the shader 160 → ~80 effective rays per
        // pixel, all but a handful of which are wasted work TAA blurs out.
        const int kMaxBase = 32;
        int  base_s = min(int(ceil(float(scene.rt_flags.y) * near_mult)), kMaxBase);
        int  N_s = lod_samples(max(1, base_s), cam_dist);
        // Per-pixel softness вЂ” terrain optionally tightens the cone via
        // the Settings UI to reduce PCSS dither. Tightening trades soft
        // shadow edges for cleaner per-pixel results at the same sample
        // count. Brushes/dyn props always use the global slider value.
        float base_softness = scene.rt_params.x;
        if (is_terrain_pre) {
            base_softness *= max(0.05, scene.terrain_params.w);
        }

        vec3 ref = abs(L.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref, L));
        vec3 tan_v = cross(L, tan_u);

        // Terrain shadow-ray bias has a non-linear ramp with camera
        // distance because the chunked raster mesh uses distance-LOD
        // (stride 1/2/4/8) while the BLAS holds the full heightmap.
        // At LOD 3 (в‰Ґ320 m) the rasterised straight-line interpolation
        // can sit 5вЂ“15 m BELOW true heightmap peaks; shadow rays fired
        // upward then false-hit those missed peaks в†’ dark patches on
        // distant ridges. We hold near bias tight (no peter-panning on
        // close shadows) and ramp hard past LOD 0's range:
        //   в‰¤ 80 m  : 0.04 m (matches the LOD-0 / BLAS gap)
        //   320 m   : ~6 m  (covers typical LOD-3 peak undershoot)
        //   в‰Ґ 600 m : ~12 m (covers worst-case)
        float dist_to_cam = cam_dist;      // P5: use hoisted distance
        float far_t = clamp((dist_to_cam - 80.0) / 320.0, 0.0, 1.0);
        float terrain_far_bias = far_t * far_t * 8.0;   // 0 в†’ 8 across the ramp
        // Bias along the receiver normal so the ray clears the
        // surface. Terrain has near-flat surfaces so a small lift
        // is enough; the BLAS-vs-LOD mismatch that needed the big
        // distance-ramped bias is now sidestepped by the no-terrain
        // mask above (the ray won't hit terrain at all).
        float bias = is_terrain_pre
            ? (0.05 + 0.10 * (1.0 - n_dot_l_raw))
            : (0.005 + 0.02 * (1.0 - n_dot_l_raw));
        // For axis-aligned brushes (non-terrain), the shading normal
        // N has been bent by corner softening (~30° at outer corners)
        // and possibly perturbed by the SPOM/normal-map basis. Using
        // the bent N for the bias offset pushes the ray origin off
        // the face axis and partway TOWARD the adjacent brush body;
        // shadow rays then grazingly self-occlude on the neighbour
        // and produce the thin dark lines/panes the user sees along
        // wall↔tower and wall↔wall seams. Snap the bias direction to
        // the dominant face axis so the offset stays purely outward
        // along the face. Terrain keeps the FBM normal (its surfaces
        // aren't axis-aligned).
        vec3 bias_n = N;
        if (!is_terrain_pre) {
            vec3 absN = abs(N);
            if (absN.x >= absN.y && absN.x >= absN.z) {
                bias_n = vec3(sign(N.x), 0.0, 0.0);
            } else if (absN.y >= absN.z) {
                bias_n = vec3(0.0, sign(N.y), 0.0);
            } else {
                bias_n = vec3(0.0, 0.0, sign(N.z));
            }
        }
        vec3 origin = shading_pos + bias_n * bias;

        // 1. Blocker search вЂ” 2 rays in a wider cone (4Г— base softness) so we
        //    catch occluders even when the surface is close to lit. Was 4
        //    rays before; halving cuts ~25% of the per-pixel shadow ray
        //    budget (blocker:shadow ≈ 4:N_s_eff was 4:8 worst case, now
        //    2:8). Penumbra estimate degrades from 4-tap-mean to 2-tap-mean
        //    but the downstream `clamp(softness * 0.25 .. * 6)` is
        //    forgiving enough that the visible difference is invisible
        //    after TAA averaging.
        // Terrain receivers use the no-terrain variant so we don't get
        // false-hits from the BLAS terrain detail above the rasterised
        // LOD surface (the heightmap bake covers terrain self-shadow).
        // P9: scale RT distances by view distance. Surfaces at ~10 m
        // from the camera have visible occluders within ~30 m; firing
        // 200 m blocker rays into thin air past that just burns BVH
        // traversal. Cap blocker rays at 4× cam_dist (min 30 m for
        // close-up); cap shadow rays at 4× avg blocker hit distance
        // computed below.
        const float kBlockerTMax = clamp(cam_dist * 4.0, 30.0, 200.0);
        const int kBlockerSearch = 2;
        // Hoist the constant cone-width multiplier out of the per-iter
        // expression. base_softness is loop-invariant; kBlockerCone =
        // base_softness * 4.0 was being recomputed every iteration.
        const float kBlockerCone = base_softness * 4.0;
        float sum_t = 0.0;
        int   hits = 0;
        // One loop, mask hoisted to a uniform-per-pixel value. The earlier
        // two-specialised-loops form was a workaround for a per-tap
        // ternary on the ray-query function — closest_hit_m takes the mask
        // as an arg so a single function body inlines and the ray-query
        // call has no divergent control on it.
        const uint blocker_mask = is_terrain_pre ? 0x02u : 0xFFu;
        // Per-pixel rotation: ONE sin/cos covers the disk-rotation
        // dither for all blocker taps, instead of one sin/cos pair
        // per tap. Mixed with the Vogel constant table this drops the
        // per-pixel trig cost from 2·N to 2.
        float pp_phi_b = rand(seed_base + uvec3(0, 99u, 0u)) * 6.28318530718;
        float pp_c_b = cos(pp_phi_b);
        float pp_s_b = sin(pp_phi_b);
        for (int i = 0; i < kBlockerSearch; ++i) {
            float br1 = rand(seed_base + uvec3(i, 91u, 13u));
            float r   = sqrt(br1) * kBlockerCone;
            vec2  v   = kVogelDisk[i & 31];
            float vx  = pp_c_b * v.x - pp_s_b * v.y;
            float vy  = pp_s_b * v.x + pp_c_b * v.y;
            vec3 jitter = (vx * tan_u + vy * tan_v) * r;
            vec3 dir = normalize(L + jitter);
            float t;
            if (closest_hit_m(origin, dir, kBlockerTMax, blocker_mask, t)) { sum_t += t; ++hits; }
        }

        if (hits == 0) {
            // No occluder anywhere in the wide cone в†’ fully lit.
            shadow = 0.0;
        } else {
            // 2. Penumbra estimate. Distance is normalised against a 10m
            //    reference, then raised to a power between 1 (linear, the
            //    classic PCSS) and 3 (cubic вЂ” very contact-sharp close-up,
            //    very soft far away). The exponent comes from the
            //    shadow_curve slider in the menu.
            float avg_t = sum_t / float(hits);
            // P9: shadow rays only need to reach occluders that could
            // contribute to the penumbra estimate — anything past 4×
            // the average blocker hit can't possibly close the cone
            // around this surface point. Capped to existing 200 m so
            // we never extend the budget.
            float kShadowTMax = clamp(avg_t * 4.0, 20.0, 200.0);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0, clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            // Terrain receivers need a much higher minimum penumbra
            // floor than walls. Boxes / dyn props sit ~1 m above the
            // ground (small blocker distance → scale ≈ 0.1), which
            // clamps the cone to ~0.25× base_softness — too tight for
            // ~10 half-rate samples to produce a smooth gradient, so
            // the edge reads as a hard line with dither dots. Walls
            // have far-away casters (long blocker distance) so they
            // don't hit the min clamp. Lift the terrain floor to 2.0×
            // so box-on-ground shadows always have a perceptibly soft
            // penumbra regardless of how low the blocker sits.
            float min_pen_mult = is_terrain_pre ? 2.0 : 0.25;
            float penumbra = clamp(base_softness * scale * 0.6,
                                   base_softness * min_pen_mult,
                                   base_softness * 6.0);

            // 3. Stratified shadow rays in the size-adapted cone.
            // Half-rate PCSS: each frame fire only HALF the ray budget,
            // alternating which half via a per-pixel checkerboard +
            // frame-parity. TAA's history blend (cube fragments use the
            // same TAA pass) accumulates the missing samples over two
            // frames so steady-state quality matches full-rate. Kicks
            // ~50% of the per-pixel PCSS cost off the GPU without a
            // visible quality drop on TAA-on configurations.
            ivec2 ip2     = ivec2(gl_FragCoord.xy);
            int   parity  = ((ip2.x + ip2.y) ^ scene.rt_flags.w) & 1;
            int   N_s_eff = max(1, (N_s + (parity == 0 ? 1 : 0)) / 2);
            int strata = int(ceil(sqrt(float(N_s_eff))));
            float inv_strata = 1.0 / float(strata);
            float blocked = 0.0;
            int taken = 0;
            // One loop, mask hoisted (same trick as the blocker loop).
            const uint shadow_mask = is_terrain_pre ? 0x02u : 0x01u;
            // Same Vogel + per-pixel rotation trick as the blocker loop.
            // Trig drops from 2·N_s_eff to 2 across the whole shadow
            // pass; on heavy slider settings (32 samples) that's ~60
            // sin/cos ops removed per shadowed pixel.
            float pp_phi_s = rand(seed_base + uvec3(1, 99u, 0u)) * 6.28318530718;
            float pp_c_s = cos(pp_phi_s);
            float pp_s_s = sin(pp_phi_s);
            for (int sy = 0; sy < strata && taken < N_s_eff; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s_eff; ++sx) {
                    int idx = taken * 2 + parity;
                    float r1 = rand(seed_base + uvec3(idx, 11u, 47u));
                    float u1 = (float(sx) + r1) * inv_strata;
                    float r   = sqrt(u1) * penumbra;
                    vec2  v   = kVogelDisk[idx & 31];
                    float vx  = pp_c_s * v.x - pp_s_s * v.y;
                    float vy  = pp_s_s * v.x + pp_c_s * v.y;
                    vec3 jitter = (vx * tan_u + vy * tan_v) * r;
                    vec3 dir = normalize(L + jitter);
                    if (any_hit_m(origin, dir, kShadowTMax, shadow_mask)) blocked += 1.0;
                    ++taken;
                }
            }
            shadow = (blocked / float(taken)) * scene.rt_params.w;
        }
        }  // end half-rate else
      }  // end fire_rt_shadow
        // shadow_strength can exceed 1.0 via the slider — without this clamp
        // the (1 - shadow) factor at line ~808 goes negative and produces
        // black-and-then-some pixels (max() in compose can't recover them).
        shadow = clamp(shadow, 0.0, 1.0);

        // Terrain hybrid: RT (castle / dyn-props) + bake (terrain
        // self-shadow). `shadow` holds the RT no-terrain result, the
        // bake holds terrain-on-terrain occlusion. Take the MAX so
        // any blocker (castle or distant ridge) shadows the receiver.
        // Castle shadows on terrain therefore work at every distance.
        if (is_terrain_pre) {
            ivec2 sz_b = textureSize(u_terrain_shadow, 0);
            const float side_b = 2048.0;
            vec2 uv_b = (vWorldPos.xz / side_b) + vec2(0.5);
            float sh_bake = 0.0;
            if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
                all(lessThanEqual(uv_b, vec2(1.0)))) {
                // The bake is ~1 texel / m and near-binary, so a tight
                // 3x3 PCF read as hard pixelated blocks close up. Use a
                // wide 5x5 Gaussian-ish kernel whose spacing is fixed in
                // WORLD space (≈ kBlurM metres total), so the softness is
                // constant regardless of bake resolution / supersample —
                // turns the low-res bake into a smooth shadow gradient.
                vec2 texel = 1.0 / vec2(sz_b);
                const float kBlurM = 4.0;                 // ~4 m penumbra
                float step_uv = (kBlurM / side_b) * 0.5;  // half-extent
                float s = 0.0, wsum = 0.0;
                for (int oy = -2; oy <= 2; ++oy) {
                    for (int ox = -2; ox <= 2; ++ox) {
                        vec2 d = vec2(float(ox), float(oy)) * (step_uv * 0.5);
                        float w = exp(-float(ox*ox + oy*oy) * 0.35);
                        s    += texture(u_terrain_shadow, uv_b + d).r * w;
                        wsum += w;
                    }
                }
                sh_bake = (s / max(wsum, 1e-4)) * scene.rt_params.w;
            }
            shadow = max(shadow, sh_bake);
        }
        // Cubic smoothstep on the combined shadow factor (lit
        // fraction = 1 − shadow): softens the merged terrain-bake +
        // RT-castle visibility ramp into one continuous penumbra.
        float lit_c = clamp(1.0 - shadow, 0.0, 1.0);
        lit_c = lit_c * lit_c * (3.0 - 2.0 * lit_c);
        shadow = 1.0 - lit_c;
    }

    vec3 direct = albedo * scene.sun_color.rgb * scene.sun_color.a *
                  n_dot_l * (1.0 - shadow);

    // --- Muzzle flash: dynamic point light + RT shadow ---
    // Active for kMuzzleFlashDuration on each shot; CPU sets intensity > 0
    // and a position just in front of the eye. One shadow ray per pixel,
    // gated by NВ·L > 0 and within the falloff radius вЂ” keeps the cost
    // negligible when no shot is firing. Inverse-square attenuation with
    // a smooth cutoff at the radius so contributions don't pop.
    // Tighter intensity gate (1e-3 vs > 0.0): the CPU clears intensity
    // to exactly 0 between flashes, but FP comparison with > 0.0 still
    // entered this block on epsilon residuals after physics writes.
    if (scene.muzzle_pos.w > 1e-3) {
        vec3  m_pos       = scene.muzzle_pos.xyz;
        float m_intensity = scene.muzzle_pos.w;
        vec3  m_color     = scene.muzzle_color.rgb;
        float m_radius    = max(0.5, scene.muzzle_color.w);

        vec3  to_light    = m_pos - shading_pos;
        // Squared-distance test first — saves the sqrt when this pixel
        // is outside the falloff radius (the common case for ground-far
        // pixels during a flash that lights only the local area).
        float dist2       = dot(to_light, to_light);
        float r2          = m_radius * m_radius;
        // Inner-radius cutoff: surfaces within ~1 m of the muzzle are
        // ALWAYS the viewmodel (gun barrel sits ~30–80 cm from the
        // muzzle origin). The inverse-square (1/(1+d²)) at d≈0.3 m
        // gives atten ≈ 0.92; even capped to 4.0 the gun visibly
        // glowed in flash colour during firing — the user explicitly
        // does not want that. Skip the muzzle dynamic light for
        // pixels closer than 1 m. World pixels are always farther
        // than the gun so they keep the flash lighting normally.
        const float kSelfLightRadius2 = 1.0;
        if (dist2 < r2 && dist2 > kSelfLightRadius2) {
            float dist     = sqrt(dist2);
            vec3 L_m       = to_light / max(dist, 1e-3);
            float n_dot_lm = max(dot(N, L_m), 0.0);
            if (n_dot_lm > 0.0) {
                // Smooth falloff: 1/(1 + dВІ)В·(1 - d/r)ВІ. The window term zeroes
                // contribution exactly at the radius, no hard edge.
                float window = 1.0 - dist / m_radius;
                float atten  = (window * window) / (1.0 + dist * dist);

                // RT shadow: one any-hit ray to the light. t_max stops short
                // of the light origin so a triangle exactly at m_pos can't
                // self-occlude.
                float bias    = 0.005 + 0.02 * (1.0 - n_dot_lm);
                vec3  origin  = shading_pos + N * bias;
                bool  in_shadow = any_hit(origin, L_m, dist - 0.01);
                if (!in_shadow) {
                    // Cap the contribution per-pixel. Surfaces VERY
                    // close to the muzzle (the viewmodel itself, ~0.5
                    // m from the muzzle origin) hit the inverse-square
                    // singularity — the gun would self-light into the
                    // tens of thousands of nits and bloom the entire
                    // screen during firing. Hard ceiling at 4.0 keeps
                    // the muzzle-glow look without runaway HDR.
                    vec3 contrib = albedo * m_color * m_intensity * n_dot_lm * atten;
                    direct += min(contrib, vec3(4.0));
                }
            }
        }
    }

    // --- AO (stratified hemisphere) ---
    //
    // ao_mode (rt_flags2.w):
    //   0 = off вЂ” no AO, ao = 1.0 (lighting reads as fully lit at concavities)
    //   1 = "fast contact AO" вЂ” short hemisphere rays at HALF the radius and
    //       capped to 2 samples. Reads similar to GTAO at far less cost
    //       than full RTAO, but is still inline ray-traced (true GTAO needs
    //       a separate screen-space pass with depth+normal bound; deferred).
    //   2 = full RTAO вЂ” N hemisphere rays at user-set radius, the original
    //       quality path. Picks up off-screen occluders correctly.
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0 && scene.rt_flags.z > 0) {
        // mode 1 caps to 2 samples; mode 2 honors the slider.
        int requested = (ao_mode == 1) ? min(scene.rt_flags.z, 2)
                                       : scene.rt_flags.z;
        int N_ao = lod_samples(requested, cam_dist);
        // mode 1 halves the search radius вЂ” bias toward contact AO rather
        // than ambient occlusion of the whole hemisphere.
        float ao_radius = scene.rt_params.y * (ao_mode == 1 ? 0.5 : 1.0);
        // Distance-scaled origin lift for terrain вЂ” pushes the AO ray's
        // origin above the BLAS detail that the rasterised LOD-1+ raster
        // sits below. Without this, AO rays from the under-shooting
        // raster surface false-hit BLAS peaks в†’ patchy darkening per
        // triangle. Lift ramps with distance, capped at ~6m which clears
        // typical LOD-2 peak undershoot.
        float ao_lift = 0.01;
        if (is_terrain_pre) {
            float t = clamp((cam_dist - 40.0) / 200.0, 0.0, 1.0);
            ao_lift = mix(0.01, 6.0, t);
        }
        vec3 origin_ao = shading_pos + N * ao_lift;
        // Unstratified random samples driven by IGN + temporal frame seed.
        // The stratified version (ceil(sqrt(N))ВІ-cell grid) was biased
        // when N wasn't a perfect square вЂ” at N=2 strata=2 picked only
        // the cells with sy=0 в†’ all rays in one half of the hemisphere
        // azimuth, which read as systematically darker edges. Random +
        // TAA temporal accumulation converges to true AO over ~8 frames
        // regardless of N.
        int taken = 0;
        float occluded = 0.0;
        // TBN hoist (same trick as the GI loop) — N is constant for all
        // AO rays at this pixel so the per-tap cross+normalize that
        // cos_hemi() does internally was redundant.
        vec3 ao_up  = abs(N.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 ao_tan = normalize(cross(ao_up, N));
        vec3 ao_bit = cross(N, ao_tan);
        // Vogel disk + per-pixel rotation for the hemisphere phi term.
        // The cosine-hemisphere y is still derived from u1 (sqrt-based);
        // only the disk-azimuth trig moves to the table. Saves N_ao
        // sin/cos pairs per pixel.
        float pp_phi_a = rand(seed_base + uvec3(0, 99u, 1u)) * 6.28318530718;
        float pp_c_a = cos(pp_phi_a);
        float pp_s_a = sin(pp_phi_a);
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rand(seed_base + uvec3(i, 7u, 53u));
            float r_h = sqrt(u1);
            float y   = sqrt(max(0.0, 1.0 - u1));
            vec2  v   = kVogelDisk[i & 31];
            float vx  = pp_c_a * v.x - pp_s_a * v.y;
            float vy  = pp_s_a * v.x + pp_c_a * v.y;
            vec3 d = (r_h * vx) * ao_tan + y * N + (r_h * vy) * ao_bit;
            if (any_hit(origin_ao, d, ao_radius)) occluded += 1.0;
            ++taken;
        }
        // Two-step shaping to control corner overlap-darkness:
        //   1. sqrt curve on raw AO вЂ” compresses the [0..1] range so the
        //      delta between 30% occluded (1 edge) and 60% (2 edges
        //      overlapping) reads as a small darkening step instead of
        //      "twice as dark". Linear AO stacked occlusions
        //      multiplicatively, perceptually wrong at inner corners.
        //   2. ao_floor remap so even fully-occluded never crushes to
        //      pure black.
        float raw = 1.0 - (occluded / float(taken));
        raw = sqrt(raw);
        float ao_floor_v = scene.rt_lod.w;
        ao = mix(ao_floor_v, 1.0, raw);
        // Mild AO fade only вЂ” the origin-lift above is the primary
        // false-hit fix; this just softens any residual mid-distance
        // patchiness without flattening the look entirely.
        if (is_terrain_pre) {
            float ao_far_t = smoothstep(150.0, 400.0, cam_dist);  // P5
            ao = mix(ao, 1.0, ao_far_t);
        }
    }

    // --- Path-traced GI (1..N bounces) ---
    // Each sample follows a path of up to N_bounces ray segments.
    //   - On a surface hit: accumulate emissive + albedoГ—scene_light (a cheap
    //     local-light approximation), tint the throughput by surface albedo,
    //     bounce in a cosine-weighted hemisphere oriented around the
    //     approximate hit normal (-ray_dir).
    //   - On a miss (sky): accumulate throughput Г— procedural sky and stop.
    // Samples are stratified on a sqrt(N)Г—sqrt(N) grid for clean low-N look.
    vec3 gi_indirect = vec3(0.0);
    // Sky visibility вЂ” fraction of first-bounce GI rays that escape the
    // scene. Used after the GI loop to attenuate the (sky-derived) ambient
    // term: enclosed surfaces (e.g. inside the keep) get sky_vis в‰€ 0 and
    // their ambient drops to near zero, giving the room a real "interior"
    // feel instead of the same flat brightness as outdoors. Default 1.0
    // when GI is disabled so existing behaviour is unchanged.
    float sky_vis = 1.0;
    int sky_misses = 0;
    int sky_total  = 0;
    int N_gi = scene.rt_flags2.x > 0 ? lod_samples(scene.rt_flags2.x, cam_dist) : 0;
    int N_bounces = max(1, scene.rt_flags2.z);
    // Terrain skips GI entirely — adjacent triangles' N differs slightly
    // and cos_hemi() is N-relative, so per-triangle hit/miss flips on
    // BLAS detail produce patchy bright/dark faces. The earlier code
    // fired the rays then forced gi_indirect=0 / sky_vis=1; gating here
    // saves all those rays.
    //
    // gi_strength gate: when the slider is at 0 the loop's sole output
    // (gi_indirect *= rt_params2.x at line 1308) is multiplied by 0 —
    // every ray is pure waste. Skip the loop and keep sky_vis = 1.0
    // (ambient stays full, matching "GI off" expectation in the UI).
    // Cheap sky-vis probe for TERRAIN pixels (full GI loop is gated off
    // for terrain — adjacent triangle normals diverge enough that
    // cos_hemi sampling produces patchy bright/dark faces). Without
    // this, terrain inside enclosed structures (the little house, the
    // castle keep) stays at sky_vis = 1.0 → full sky ambient → bright
    // green floor while the cube walls/ceiling around it darken to
    // pitch black. Fire 4 hemisphere rays around +Y with the terrain
    // BLAS masked out (so terrain doesn't self-occlude). Distance-LOD'd
    // to near pixels — far terrain doesn't matter for interior look.
    if (is_terrain_pre && scene.rt_flags.x != 0 && cam_dist < 50.0) {
        const int kProbeN = 4;
        const uint kProbeMask = 0xFDu;        // skip terrain BLAS (bit 0x02)
        vec3 probe_origin = vWorldPos + vec3(0.0, 0.10, 0.0);
        int probe_misses = 0;
        for (int i = 0; i < kProbeN; ++i) {
            float r1 = rand(seed_base + uvec3(uint(i), 41u, 67u));
            float r2 = rand(seed_base + uvec3(uint(i), 13u, 89u));
            float r_h = sqrt(r1);
            float phi = 6.28318530718 * r2;
            vec3 dir = vec3(r_h * cos(phi),
                            sqrt(max(0.0, 1.0 - r1)),
                            r_h * sin(phi));
            if (!any_hit_m(probe_origin, dir, 60.0, kProbeMask)) {
                probe_misses += 1;
            }
        }
        sky_misses = probe_misses;
        sky_total  = kProbeN;
        sky_vis    = float(probe_misses) / float(kProbeN);
    }
    if (N_gi > 0 && !is_terrain_pre && scene.rt_params2.x > 1e-4) {
        float gi_radius = scene.rt_params2.y;

        // Sky-only ambient term вЂ” used when a GI bounce hits a surface
        // that CANNOT see the sun. Approximates the soft fill light from
        // the sky dome that reaches indoor surfaces near openings. 0.05
        // (~5% of sky color) keeps deep interiors believably dark; the
        // earlier 0.18 made every shadowed bounce hit accumulate more
        // light than a real enclosed room would have.
        vec3 sky_fill = scene.sky_color.rgb * 0.05;

        int strata = int(ceil(sqrt(float(N_gi))));
        float inv_strata = 1.0 / float(strata);
        int taken = 0;
        vec3 sum = vec3(0.0);
        // Hoist the receiver TBN out of the GI sample loop — N is constant
        // for all samples at this pixel, so the cross+normalize that
        // cos_hemi() does internally was being repeated N_gi times for
        // identical inputs. Bounce hemispheres still rebuild because
        // hit_n changes per bounce.
        vec3 N_up   = abs(N.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 N_tan  = normalize(cross(N_up, N));
        vec3 N_bit  = cross(N, N_tan);
        vec3 ray_origin0 = shading_pos + N * 0.01;
        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rand(seed_base + uvec3(taken, 73u, 11u));
                float r2 = rand(seed_base + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                // Inline cos_hemi using the hoisted TBN.
                float r_h = sqrt(u1);
                float phi = 6.28318530718 * u2;
                vec3 ld = vec3(r_h * cos(phi),
                               sqrt(max(0.0, 1.0 - u1)),
                               r_h * sin(phi));
                vec3 ray_dir = ld.x * N_tan + ld.y * N + ld.z * N_bit;
                vec3 ray_origin = ray_origin0;
                vec3 throughput = vec3(1.0);
                vec3 path = vec3(0.0);

                for (int b = 0; b < N_bounces; ++b) {
                    float t;
                    int idx;
                    int prim;
                    if (!closest_hit_material(ray_origin, ray_dir, gi_radius, t, idx, prim)) {
                        path += throughput * sample_sky(ray_dir);
                        // First-bounce miss = the surface can see sky in
                        // this direction. Tracks "open vs enclosed" for the
                        // ambient attenuation below.
                        if (b == 0) ++sky_misses;
                        break;
                    }
                    // Static-castle hit: 1 BLAS instance covers all brushes
                    // вЂ” recover the per-brush material slot via primitive id.
                    int mat_idx = (idx == kStaticBlasSentinel)
                                    ? (prim / kCubeTrisPerBox)
                                    : idx;
                    Material m = materials[mat_idx];
                    vec3 hit_pos = ray_origin + ray_dir * t;
                    vec3 hit_n = -ray_dir;  // approximate outward normal

                    // *** GI bounce lighting *** вЂ” fires a shadow ray
                    // from the hit point to the sun for bounce levels up
                    // to gi_shadow_max_bounce (rt_lod.z). Bounces beyond
                    // that fall back to sky_fill alone. 0 = no GI sun
                    // shadows (cheap, every shadowed bounce uses fill);
                    // 1 = first bounce only (default, the cheap one
                    // that gives the indoor-vs-outdoor delta we want);
                    // higher = more accurate at proportional cost.
                    int gi_shadow_max = int(scene.rt_lod.z);
                    vec3 hit_light = sky_fill;
                    if (b < gi_shadow_max) {
                        float n_dot_sun = dot(hit_n, scene.sun_direction.xyz);
                        if (n_dot_sun > 0.0) {
                            if (!any_hit(hit_pos + hit_n * 0.01,
                                         scene.sun_direction.xyz, 200.0)) {
                                hit_light += scene.sun_color.rgb *
                                             scene.sun_color.a * n_dot_sun;
                            }
                        }
                    }
                    path += throughput * (m.emissive.rgb +
                                          m.color.rgb * hit_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint the path's throughput by the surface albedo and
                    // continue from the hit point in a new cosine-hemisphere.
                    throughput *= m.color.rgb;
                    ray_origin = hit_pos + hit_n * 0.01;
                    float br1 = rand(seed_base + uvec3(taken * 7 + b, 31u, 17u));
                    float br2 = rand(seed_base + uvec3(taken * 7 + b, 7u, 91u));
                    ray_dir = cos_hemi(br1, br2, hit_n);
                }
                sum += path;
                ++taken;
            }
        }
        vec3 cur_avg = sum / float(taken);

        // Session-3 temporal reservoir reuse. When restir is on, motion-
        // reproject the previous frame's reservoir at this pixel, gate
        // on the surface normal (sample_dir field carries N from the
        // writing pixel), and exponential-blend toward the prev radiance
        // with weight prev_M / (prev_M + 1). Cap M at restir_params.y so
        // a static camera doesn't lock onto a stale sample forever.
        ivec2 ipix = ivec2(gl_FragCoord.xy);
        uint  idx  = uint(ipix.y) * uint(scene.viewport.x) + uint(ipix.x);

        // Session-5 ring addressing. One SSBO, 3 per-pixel regions;
        // frame%3 is the write region, (frame+2)%3 the read region.
        // Reading a strictly older region than we write means a
        // spatial tap never sees this frame's partial writes (kills
        // the session-3 single-buffer aliasing race), and the
        // prev↔cur swap costs no descriptor update.
        uint  resPx   = uint(scene.viewport.x) * uint(scene.viewport.y);
        uint  resFrm  = uint(scene.rt_flags.w);
        uint  resCur  = (resFrm % 3u) * resPx;          // write base
        uint  resPrev = ((resFrm + 2u) % 3u) * resPx;   // read  base

        // Disocclusion thresholds (session 5). The session-3 quarantine
        // root cause #2 was normal-only acceptance pulling GI from a
        // different surface at the same orientation (parallel walls,
        // separate floor patches). We now also require the candidate's
        // recorded camera distance (Reservoir.pad) to match this
        // fragment's within a distance-scaled band. Constants, not UBO
        // slots — restir_params.zw is owned by FSR jitter.
        const float kResNrmCos  = 0.92;   // ~23° max normal deviation
        const float kResDepthRel = 0.04;  // 4% of camera distance …
        const float kResDepthAbs = 0.20;  //  … + 20 cm absolute slack
        float resDepthTol = kResDepthRel * cam_dist + kResDepthAbs;

        vec3 blended = cur_avg;
        uint M_new   = 1u;
        if (scene.restir_params.x > 0.5 && vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            vec2 prev_uv  = prev_ndc * 0.5 + 0.5;
            if (prev_uv.x > 0.0 && prev_uv.x < 1.0 &&
                prev_uv.y > 0.0 && prev_uv.y < 1.0) {
                ivec2 prev_pix = ivec2(prev_uv * scene.viewport.xy);
                prev_pix = clamp(prev_pix,
                                 ivec2(0),
                                 ivec2(scene.viewport.xy) - ivec2(1));
                uint prev_idx = uint(prev_pix.y) * uint(scene.viewport.x)
                              + uint(prev_pix.x);
                Reservoir prev = u_reservoir_prev.r[resPrev + prev_idx];
                float prev_d = uintBitsToFloat(prev.pad);
                if (prev.M > 0u &&
                    dot(prev.sample_dir, N) > kResNrmCos &&
                    abs(prev_d - cam_dist) < resDepthTol) {
                    uint M_max = uint(scene.restir_params.y);
                    M_new = min(prev.M + 1u, M_max);
                    float w_prev = float(M_new - 1u) / float(M_new);
                    blended = mix(cur_avg, prev.radiance, w_prev);
                }
            }
        }

        // Session-4 spatial neighbour merge: REMOVED in session 5.
        // The 3-tap variant added 3 divergent 48-B SSBO loads per
        // GI pixel and measured a 3.7× median / 12× castle GPU-time
        // regression — it cost far more than the ¼-sample ray saving
        // it was meant to converge. Temporal reuse alone recovers
        // most of the convergence at near-zero extra cost (one load +
        // one store), which is the point of ReSTIR here. If spatial
        // is ever revisited it must be a separate cheap compute pass,
        // not an inline per-fragment gather.
        vec3 shade_radiance = blended;

        // SVGF GI denoiser feed (Session 1). Store the unmodulated
        // irradiance signal (pre-albedo, post-temporal-reservoir-blend)
        // for future spatial-filter passes to denoise.
        imageStore(u_svgf_gi, ivec2(gl_FragCoord.xy),
                   vec4(shade_radiance, 1.0));

        // SVGF temporal accumulator (Session 2). When the toggle is on
        // (terrain_local_info.w), EMA-blend with the previous frame's
        // accumulated irradiance at the motion-reprojected pixel. .a
        // tracks M (sample count, 1..kSvgfMmax); EMA weight =
        // (M-1)/M. Frame parity picks read/write halves of the
        // ping-pong pair.
        vec3 svgf_out = shade_radiance;
        if (scene.terrain_local_info.w > 0.5) {
            const float kSvgfMmax = 8.0;
            uint   svFrame = uint(scene.rt_flags.w);
            ivec2  svPix   = ivec2(gl_FragCoord.xy);
            bool   parity0 = ((svFrame & 1u) == 0u);
            // Reprojected previous pixel from the existing vPrevClip
            // varying (already used by the motion-vec output).
            float  prevM   = 0.0;
            vec3   prevRgb = vec3(0.0);
            if (vPrevClip.w > 0.0) {
                vec2 pndc = vPrevClip.xy / vPrevClip.w;
                vec2 puv  = pndc * 0.5 + 0.5;
                if (puv.x > 0.0 && puv.x < 1.0 &&
                    puv.y > 0.0 && puv.y < 1.0) {
                    ivec2 ppx = ivec2(puv * scene.viewport.xy);
                    ppx = clamp(ppx, ivec2(0),
                                ivec2(scene.viewport.xy) - ivec2(1));
                    vec4 prev = parity0
                        ? imageLoad(u_svgf_hist1, ppx)
                        : imageLoad(u_svgf_hist0, ppx);
                    prevRgb = prev.rgb;
                    prevM   = prev.a;
                }
            }
            float Mnew = min(prevM + 1.0, kSvgfMmax);
            float wPrev = (Mnew - 1.0) / Mnew;
            svgf_out = mix(shade_radiance, prevRgb, wPrev);
            // Write to the OPPOSITE slot of what we just read.
            vec4 storeVal = vec4(svgf_out, Mnew);
            if (parity0) imageStore(u_svgf_hist0, svPix, storeVal);
            else         imageStore(u_svgf_hist1, svPix, storeVal);
        }
        gi_indirect = albedo * svgf_out * scene.rt_params2.x;
        // Hard ceiling — a bounce ray happening to land on a sun-lit
        // white wall + multiple bounces can compound to 10×+ sun color
        // for a single sample. Without a cap the bright spike feeds
        // bloom + auto-exposure → next frame's exposure plummets →
        // next frame brightens → oscillation reads as surface flicker.
        // Clamp at 6 per channel: caps the firefly without crushing
        // legitimate strong bounces in the visible range.
        gi_indirect = min(gi_indirect, vec3(6.0));
        sky_total = taken;
        sky_vis = float(sky_misses) / float(max(1, sky_total));

        // Reservoir writeback for next frame's temporal combine. Stores
        // the pre-spatial-merge radiance so the spatial blur stays a
        // per-frame display effect, not a temporally-compounding one.
        Reservoir cur;
        cur.sample_dir = N;
        cur.radiance   = blended;
        cur.W          = 1.0;
        cur.w_sum      = float(M_new);
        cur.M          = M_new;
        cur.pad        = floatBitsToUint(cam_dist);  // depth disoccl. key
        u_reservoir_cur.r[resCur + idx] = cur;
    }

    // --- AO --- (also stratified to spread samples cleanly)
    // [moved below, recomputed from N_ao]

    // Sky-occlusion term. Surfaces that can't see the sky get a much
    // darker ambient вЂ” that's the natural look of an enclosed room. The
    // 0.15 floor stops pure-interior pixels from going completely black
    // (real rooms have some bounced light), and the smoothstep curve
    // rolls in the sky contribution faster than linear so the open arena
    // still reads bright. GI-disabled fallback (sky_vis=1.0) leaves the
    // outdoor look identical to before this change.
    // Combine: sky_factor only attenuates the sky-derived part of
    // ambient, the ground term carries through everywhere. AO multiplies
    // the whole thing so edges still darken, but never to zero (the
    // ground term keeps a baseline). 0.10 floor on sky_factor is a tiny
    // gesture toward "windowless rooms still have some light leakage."
    float sky_factor = mix(0.10, 1.0, smoothstep(0.0, 0.6, sky_vis));
    // Non-terrain receivers (brick walls, keep walls, tile floors) also
    // attenuate the constant ground term by sky_factor so deep cracks /
    // corners actually go dark — without this, the ground baseline keeps
    // walls bright even when GI rays mostly hit other walls and never
    // see the sky. Terrain keeps the original "ground term carries
    // through everywhere" behaviour because its GI is gated off and
    // sky_factor would otherwise stick at 0.10 forever.
    float ground_atten = is_terrain_pre ? 1.0
                                        : mix(0.25, 1.0, sky_factor);
    vec3 ambient_combined = mix(ambient_ground * ground_atten,
                                 ambient_sky * sky_factor, up);
    vec3 indirect = albedo * ambient_combined * ao + gi_indirect;
    // Terrain sky-bounce wrap. Back-facing pixels (-NВ·L > 0) pick up a
    // sky-tinted lift to model atmospheric scattering bouncing onto the
    // shadow side of mountains. Applied as ambient here (not as direct)
    // so it never overrides a real shadow-ray result. wrap_strength
    // tunes the intensity from the Settings UI.
    if (is_terrain_pre) {
        float wrap_amt = clamp(scene.terrain_params.y, 0.0, 1.0);
        float back = max(0.0, -dot(N, L));
        indirect += albedo * scene.sky_color.rgb * back * 0.30 * wrap_amt;
    }
    vec3 final = direct + indirect + vEmissive.rgb;

    // Atmospheric perspective for terrain. Distant ground fades toward
    // the sky tint along the view direction вЂ” the standard "real
    // mountains" look. Onset starts at 200m and reaches 75% blend at
    // 1500m; never fully hides the silhouette so the scene retains
    // depth. Cheaper than volumetric fog and visually indistinguishable
    // for clear-day weather.
    if (is_terrain_pre) {
        // P6: early-out before computing view_dir + sky atmospheric
        // sample (which is ~12 ALU + 1 sqrt + a normalize). For close
        // terrain (cam_dist < 250 m) or when the user has fog disabled
        // (terrain_params.x ≈ 0), the fog contribution rounds to zero
        // and the entire block is wasted. Saves the cost on ~half of
        // terrain pixels (everything indoors / castle-area / nearby
        // hills) plus 100% of pixels when the slider is off.
        float fog_strength = clamp(scene.terrain_params.x, 0.0, 1.0);
        if (cam_dist > 250.0 && fog_strength > 1e-3) {
            vec3 view_dir = normalize(vWorldPos - scene.camera_pos.xyz);
            // sample_sky_atmosphere: no sun halo. The halo is fine for
            // the actual sky pixels (compose pass / GI rays), but
            // applying it to FOG made distant terrain near the sun
            // direction look like glowing white blobs.
            vec3 fog_color = sample_sky_atmosphere(view_dir);
            // 95% sky by ~1.3km — both edges (corner ray reaches ~2km
            // of world at 80° FOV / far=1500m) AND centre (clipped at
            // 1500m hard) are deep into fog before the cut, so the
            // asymmetric visible disc is invisible.
            float fog_t = clamp((cam_dist - 250.0) / 1100.0, 0.0, 0.95);
            fog_t = fog_t * fog_t * (3.0 - 2.0 * fog_t);
            fog_t *= fog_strength;
            final = mix(final, fog_color, fog_t);
        }
    }

    outColor = vec4(final, 1.0);
}

