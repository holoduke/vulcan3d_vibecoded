// Per-frame scene UBO (set 0, binding 0) shared by every world shader.
//
// This is the ONE GLSL declaration of the block. It mirrors
// `struct SceneUBO` in src/engine/vk_engine/internal.h member for member
// (std140; every member is a vec4 / ivec4 / mat4 / vec4[N], so C++ and
// GLSL offsets agree without manual padding). Field meanings are
// documented on the C++ side. When adding a field: append it to BOTH
// (at the end), and update the sizeof static_assert in helpers.cpp.
//
// Include after #version with GL_GOOGLE_include_directive enabled.
#ifndef QLIKE_SCENE_UBO_GLSL
#define QLIKE_SCENE_UBO_GLSL

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4   sun_direction;
    vec4   sun_color;
    vec4   ambient;
    vec4   sky_color;
    ivec4  rt_flags;
    vec4   rt_params;
    ivec4  rt_flags2;
    vec4   rt_params2;
    vec4   camera_pos;
    vec4   rt_lod;
    vec4   viewport;
    vec4   muzzle_pos;
    vec4   muzzle_color;
    vec4   terrain_params;
    vec4   terrain_h_low;
    vec4   terrain_h_high;
    vec4   grass_extra;
    vec4   grass_extra2;
    mat4   light_vp;
    vec4   terrain_extra;
    vec4   water_params;
    vec4   water_color;
    vec4   water_color_shallow;
    vec4   water_shore;
    vec4   water_foam_color;
    vec4   water_foam_params;
    vec4   fog_band;
    vec4   terrain_rt_extra;
    vec4   grass_color_top;
    vec4   grass_color_bottom;
    vec4   grass_color_ground;
    vec4   grass_color_ground_far;
    vec4   grass_shadow_params;
    vec4   grass_shore_color;
    vec4   grass_shore_params;
    vec4   terrain_shore_color;
    vec4   terrain_shore_params;
    vec4   distance_fog_color;
    vec4   distance_fog_params;
    vec4   terrain_shore_general_color;
    vec4   terrain_shore_general_params;
    vec4   terrain_sand_color;
    vec4   water_river_extra;
    vec4   water_river_extinct;
    vec4   restir_params;
    vec4   spom_params;
    vec4   terrain_local_info;
    vec4   terrain_max_grid[256];
    vec4   terrain_disp_params;
    vec4   terrain_antitile_params;
    vec4   grass_shadow_on_terrain_params;
    vec4   grass_side_lit_params;
    vec4   voxel_origin;
    vec4   voxel_dims;
    vec4   voxel_grid;
    vec4   road_info;
    vec4   road_bbox;
    vec4   road_pts[64];
    vec4   grass_rm_params;
    vec4   grass_rm_params2;
    mat4   clutter_vp;
    vec4   atmo_params;
} scene;

#endif
