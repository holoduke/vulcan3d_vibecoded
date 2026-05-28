#version 460

// SPOM silhouette shell pass: vertex stage.
// Extrudes wall brush vertices OUTWARD by shellThickness * spom_strength
// along the per-face world-space normal. The fragment shader marches the
// height map inward; pixels in the shell band whose brick top doesn't
// reach up get discarded (= silhouette). Outer corner pixels where the
// brick DOES extend get drawn — that's the silhouette extension.
//
// Same vertex format + push constants as cube.vert so shell pass reuses
// the same per-brush draw call infrastructure.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;     // x=albedo idx, y=normal idx, z=uv_scale, w=mode
} pc;

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
    vec4  spom_params;  // .x = strength
} scene;

// 12cm shell. The previous 4cm matched cube.frag's parallax depth but
// projected to only ~3-5 visible pixels at typical viewing distance --
// invisible to the user. 12cm gives a band wide enough to see bricks
// actually poking past the geometric corner. The lateral push amount
// scales with this too, so corners visibly puff out by 12cm.
const float kShellThickness = 0.12;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vWorldPos;
layout(location = 2) out vec2 vUv;
layout(location = 3) flat out vec4 vTexParams;
layout(location = 4) out vec3 vShellBase;
layout(location = 5) out float vShellT;
// Un-extruded clip-space Z (depth as the main pass would have written it
// for this pixel without the shell push). shell.frag samples the actual
// depth buffer; if the depth at this pixel is at or in front of
// vOrigDepth, the main pass already drew a wall here -- discard.
layout(location = 6) out float vOrigDepth;

void main() {
    float strength      = max(0.0, scene.spom_params.x);
    float shell_world_t = kShellThickness * strength;

    // OUTWARD push along face normal (the main extrusion).
    float world_norm_len = length(vec3(pc.model * vec4(inNormal, 0.0)));
    float obj_normal_push = (strength > 0.01 && world_norm_len > 0.001)
                              ? (shell_world_t / world_norm_len) : 0.0;
    vec3 inPos_ext = inPosition + obj_normal_push * inNormal;

    // LATERAL push along the two NON-NORMAL object-space axes, toward
    // the vertex's existing sign. This expands each face's footprint
    // outward in the in-plane directions so adjacent extruded faces
    // OVERLAP at brush corners instead of leaving a triangular sky
    // gap. The cube mesh has per-face verts (24 total) so each face
    // pushes independently along its own normal; without the lateral
    // push, +X and +Y face corner verts diverge along orthogonal axes
    // and the corner pixel between them rasterises nothing -- that's
    // the "no silhouette" the user reported. With lateral push, both
    // faces extend laterally toward each other; the corner gap is
    // filled by overlapping triangles from both faces.
    vec3 scale = vec3(length(pc.model[0].xyz),
                      length(pc.model[1].xyz),
                      length(pc.model[2].xyz));
    if (strength > 0.01) {
        if (abs(inNormal.x) < 0.5 && scale.x > 0.001)
            inPos_ext.x += sign(inPosition.x) * shell_world_t / scale.x;
        if (abs(inNormal.y) < 0.5 && scale.y > 0.001)
            inPos_ext.y += sign(inPosition.y) * shell_world_t / scale.y;
        if (abs(inNormal.z) < 0.5 && scale.z > 0.001)
            inPos_ext.z += sign(inPosition.z) * shell_world_t / scale.z;
    }

    gl_Position = pc.mvp * vec4(inPos_ext, 1.0);

    vec4 wp_ext  = pc.model * vec4(inPos_ext, 1.0);
    vec4 wp_base = pc.model * vec4(inPosition, 1.0);
    vWorldPos    = wp_ext.xyz;
    vShellBase   = wp_base.xyz;
    vNormal      = mat3(pc.model) * inNormal;
    vUv          = inUv;
    vTexParams   = pc.tex_params;
    vShellT      = shell_world_t;

    // Un-extruded clip-space depth: what the main pass's wall pixel
    // would have written at this vertex. shell.frag compares this
    // against the actual depth buffer to detect "wall already here".
    vec4 orig_clip = pc.mvp * vec4(inPosition, 1.0);
    vOrigDepth     = orig_clip.z / orig_clip.w;
}
