#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;     // for world-space position + normal
    mat4 prev_mvp;  // prev_view_proj * prev_model — drives motion vector at fs.location=1
    vec4 color;
    vec4 emissive;  // rgb = emissive, a = skip-lighting flag
    // Texture indices + UV scale + spare. -1 in either index = solid color.
    //   x: albedo index
    //   y: normal index
    //   z: uv scale (multiplier on inUv before sampling, for tiling)
    //   w: spare
    vec4 tex_params;
} pc;

// Scene UBO -- declare only the prefix through spom_params so we can
// scale the shell thickness with the runtime SPOM-strength slider.
// std140 ABI matches the C++ SceneUBO header up to .spom_params (~ offset
// 784 / 49 vec4). Binding 0 is already flagged VERTEX|FRAGMENT|TESS_EVAL.
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
    vec4  spom_params;          // .x = SPOM strength (UI slider)
} scene;

layout(location = 0) out vec3 vNormal;
// vColor/vEmissive/vTexParams come from push-constant — identical at all 3
// verts. Must be `flat` so perspective interpolation precision can't drift
// the value (which corrupted texture-index gating for vTexParams; vColor +
// vEmissive are flagged for consistency / future-proofing).
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
// Local object-space position (pre-model). Lets the fragment shader sample
// triplanar textures in object space for moving objects — without it the
// world-position based sample would "flow" across rotating boxes and the
// texture detail aliasing reads as light/dark flicker.
layout(location = 6) out vec3 vObjectPos;
// Local object-space normal (pre-model). For object-space triplanar we also
// need the unrotated normal to compute the projection blend weights.
layout(location = 7) out vec3 vObjectNormal;
// Previous-frame clip-space position (prev_view_proj * prev_model * local).
// cube.frag turns this into a screen-space motion vector at location=1.
layout(location = 8) out vec4 vPrevClip;
// Pre-extrusion (shell-base) world position. For non-extruded geometry
// equals vWorldPos. For SPOM wall brushes (see the extrusion below)
// vShellBase is the world-space position of the un-extruded brush face;
// `dot(vWorldPos - vShellBase, geometric_normal)` then gives the distance
// from this fragment to the original face along the outward normal --
// constant per face under uniform extrusion. cube.frag uses it in the
// SSDM shell-mapping path to decide brick-vs-sky inside the shell band.
layout(location = 9) out vec3 vShellBase;

// Shell-mapping extrusion thickness in world metres (matches the
// kHeightScaleBase in cube.frag's spom_uv). When the SPOM-strength UI
// slider is 1, the wall faces are pushed outward by this much; the
// fragment shader's SSDM shell-mapping path uses the same shell band as
// its parallax march depth so the brick height field lives entirely in
// the visible shell above the original face. The cube mesh has per-face
// vertices (see create_cube_mesh in src/engine/mesh.cpp -- 24 verts /
// 12 tris), so each wall face moves along its OWN flat normal. Outer
// corners between two faces therefore open a tiny ~kShellThickness gap
// where the bricks of each face poke out past the original brush edge
// -- that gap IS the silhouette extension the user wants. (Previous
// attempts that shared verts at corners couldn't get this; with shared
// verts the corner vert can't simultaneously move +X and +Z.)
const float kShellThickness = 0.04;

void main() {
    // SPOM shell-mapping wall extrusion. Gates:
    //  - albedo is a SPOM wall material (1 = Bricks067 castle outer
    //    walls, 4 = PaintedBricks001 keep walls -- matches
    //    height_idx_for_albedo() in cube.frag).
    //  - face is a WALL, not top/bottom (|inNormal.y| < 0.5). Floors
    //    are SPOM-textured but face up; extruding them upward would
    //    poke through wall-top crenellations / walkways and gain
    //    nothing visually (you look at floors, not along them).
    //  - SPOM strength > 0; if the slider is 0 the shell collapses
    //    to zero thickness and we render exactly as un-extruded.
    int   albedo_idx     = int(pc.tex_params.x);
    int   shading_mode   = int(pc.tex_params.w + 0.5);
    bool  is_spom_wall = (albedo_idx == 1 || albedo_idx == 4) &&
                         abs(inNormal.y) < 0.5 &&
                         shading_mode == 0;  // world-space triplanar
                                              // (rules out obj-space dyn
                                              // props + terrain)
    // Shell extrusion with OUTWARD + LATERAL push, decoupled. Each
    // SPOM-wall side-face vert pushes (a) outward along its face normal
    // by kShellThickness (= 4cm wall puff in normal direction), and
    // (b) laterally toward its own corner along the two NON-normal
    // axes by kLateralPush (= 12cm corner extension). The bigger
    // lateral makes the per-brush corner region big enough to actually
    // see the stepped silhouette from typical viewing distances; the
    // smaller outward keeps the wall's main-surface thickness change
    // subtle. cube.frag SSDM discards only at TRUE corners (past
    // brush on 2 lateral axes), so face centers + edges still render
    // as solid wall — the visible enlargement is concentrated at
    // brush corners where bricks form a stepped silhouette.
    float spom_strength  = max(0.0, scene.spom_params.x);
    float shell_world_t  = kShellThickness * spom_strength;
    const float kLateralPush = 0.12;
    float lateral_world_t = kLateralPush * spom_strength;
    bool  do_extrude     = is_spom_wall && spom_strength > 0.01;

    vec3 scale = vec3(length(pc.model[0].xyz),
                      length(pc.model[1].xyz),
                      length(pc.model[2].xyz));

    // Outward push (scale-aware so world-space displacement is exactly
    // shell_world_t regardless of brush dimensions).
    float world_norm_len = length(vec3(pc.model * vec4(inNormal, 0.0)));
    float obj_normal_push = (do_extrude && world_norm_len > 0.001)
                              ? (shell_world_t / world_norm_len) : 0.0;
    vec3 inPos_ext = inPosition + obj_normal_push * inNormal;

    // Lateral push: extend each side face's in-plane footprint outward
    // by lateral_world_t toward the vert's own corner.
    if (do_extrude) {
        if (abs(inNormal.x) < 0.5 && scale.x > 0.001)
            inPos_ext.x += sign(inPosition.x) * lateral_world_t / scale.x;
        if (abs(inNormal.y) < 0.5 && scale.y > 0.001)
            inPos_ext.y += sign(inPosition.y) * lateral_world_t / scale.y;
        if (abs(inNormal.z) < 0.5 && scale.z > 0.001)
            inPos_ext.z += sign(inPosition.z) * lateral_world_t / scale.z;
    }

    gl_Position = pc.mvp * vec4(inPos_ext, 1.0);
    vec4 wp = pc.model * vec4(inPos_ext, 1.0);
    vWorldPos = wp.xyz;
    // Shell base = un-extruded world position. cube.frag computes
    // dot(vWorldPos - vShellBase, geometric_normal) to get the
    // shell offset (= kShellThickness on extruded faces, 0 elsewhere).
    vec4 wp_base = pc.model * vec4(inPosition, 1.0);
    vShellBase = wp_base.xyz;
    // For uniform-scale or rotation-only models, mat3(model) is fine. We accept
    // small error on non-uniform scale (the static brushes use non-uniform
    // scaling for floor/walls); cubes have axis-aligned faces so it still reads
    // correctly enough for shading.
    vNormal = mat3(pc.model) * inNormal;
    vColor = pc.color.rgb;
    vEmissive = pc.emissive;
    vUv = inUv;
    vTexParams = pc.tex_params;
    vObjectPos = inPos_ext;
    vObjectNormal = inNormal;
    vPrevClip = pc.prev_mvp * vec4(inPos_ext, 1.0);
}
