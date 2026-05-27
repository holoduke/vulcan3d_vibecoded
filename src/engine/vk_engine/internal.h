#pragma once

// Implementation-private header for vk_engine's split-up .cpp files. Holds
// the shared statics — push-constant + UBO layouts, RT extension function
// pointers, the validation messenger callback, and small math/RNG helpers
// that several engine sections reach for. Outside of `src/engine/vk_engine/`
// nothing should include this file.

#include "engine/vk_engine.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <cstdlib>

namespace qlike {

// ---- Validation layer toggle ----
// Debug builds: always on. Release builds: OFF by default (validation
// is far too slow for shipping), but force-enable when the env var
// QLIKE_VK_VALIDATION is set to a non-empty value. This lets us run
// the normal Release exe WITH SYNC + GPU-ASSISTED validation to
// diagnose intermittent device-lost crashes that leave no signal in
// a plain Release build (validation was previously compiled out, so
// "vk_err=0" was meaningless — nothing was checking).
inline bool vk_validation_enabled() {
#ifndef NDEBUG
    return true;
#else
    static const bool v = []{
        const char* e = std::getenv("QLIKE_VK_VALIDATION");
        return e != nullptr && e[0] != '\0';
    }();
    return v;
#endif
}

// Path of the autosaved settings file relative to CWD.
inline constexpr const char* kSettingsPath = "qlike_settings.cfg";

// Pi constants used by several engine sections. Float-precision; matches the
// literal `3.14159265f` previously scattered across combat / world / helpers.
inline constexpr float kPi     = 3.14159265358979f;
inline constexpr float kHalfPi = 1.57079632679490f;
inline constexpr float kTwoPi  = 6.28318530717958f;

// Stage flags used for every vkCmdPushConstants call against pipeline_layout_.
// The PushConstants struct is consumed by the vertex, fragment, tessellation-
// control AND tessellation-evaluation stages (see init_pipeline). Same bit
// pattern was previously inlined at 16+ call sites; lifting it removes both
// the line-noise and the per-site mismatch risk.
inline constexpr VkShaderStageFlags kPushConstantStages =
    VK_SHADER_STAGE_VERTEX_BIT |
    VK_SHADER_STAGE_FRAGMENT_BIT |
    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

// ---- Shared push-constant + UBO layouts ----

// Per-draw push constants for the cube/cylinder raster pipeline. The depth
// pre-pass uses the same layout (depth_pipeline_'s shader ignores everything
// past mvp/model); grass.vert reads `grass_params` and ignores the rest.
// 256 bytes total — see static_assert in helpers.cpp.
struct PushConstants {
    glm::mat4 mvp        = glm::mat4(1.0f);
    glm::mat4 model      = glm::mat4(1.0f);
    // prev_view_proj × prev_model. cube.vert emits `vPrevClip` from this
    // and cube.frag turns it into a screen-space motion vector (location=1)
    // for the TAA / SVGF pass.
    glm::mat4 prev_mvp   = glm::mat4(1.0f);
    glm::vec4 color      = glm::vec4(1.0f);
    glm::vec4 emissive   = glm::vec4(0.0f);
    // x: albedo idx (-1 = none), y: normal idx (-1 = none), z: uv scale,
    // w: 0 = world-space triplanar, 1 = object-space triplanar, 2 = terrain.
    glm::vec4 tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
    // Grass-only knobs (read by grass.vert):
    //   x: max draw distance (m). beyond this the vertex shader collapses
    //      the blade to a degenerate triangle so the rasteriser drops it.
    //   y: wind strength (peak side-sway in metres at the blade tip).
    //   z: time (seconds; drives the wind sin).
    //   w: unused.
    glm::vec4 grass_params = glm::vec4(80.0f, 0.04f, 0.0f, 0.0f);
};

// Per-frame scene UBO, bound at scene_desc_set binding 0.
struct SceneUBO {
    glm::vec4  sun_direction;
    glm::vec4  sun_color;
    glm::vec4  ambient;
    glm::vec4  sky_color;
    glm::ivec4 rt_flags;     // x:shadow_on, y:shadow_samples, z:ao_samples, w:frame
    glm::vec4  rt_params;    // x:shadow_softness, y:ao_radius, z:ambient_strength, w:shadow_strength
    glm::ivec4 rt_flags2;    // x:gi_samples, y:reflections_on, z:gi_bounces
    glm::vec4  rt_params2;   // x:gi_strength, y:gi_radius, z:reflection_strength, w:shadow_curve
    glm::vec4  camera_pos;   // xyz = world-space eye
    glm::vec4  rt_lod;       // x:lod_near, y:lod_far
    glm::vec4  viewport;     // x:w, y:h, z:1/w, w:1/h — used by motion-vec output
    // Muzzle flash dynamic light. Pulses for kMuzzleFlashDuration on each
    // shot; cube.frag adds a point-light contribution with an RT shadow ray
    // when intensity > 0. Off-frames have intensity = 0 (the gate).
    //   muzzle_pos.xyz   = world-space light origin
    //   muzzle_pos.w     = intensity (0 = off)
    //   muzzle_color.rgb = light color (linear)
    //   muzzle_color.w   = falloff radius (m); contribution drops to ~0 past it
    glm::vec4  muzzle_pos;
    glm::vec4  muzzle_color;
    // Terrain-only shader knobs (raster path with tex_params.w==2):
    //   x: atmospheric fog strength (0=off, 1=full)
    //   y: half-Lambert wrap amount (0=hard Lambert, 1=full wrap)
    //   z: detail-texture brightness (multiplier on the layer base)
    //   w: shadow softness scale — multiplies global shadow softness
    //      for terrain pixels, lets the user reduce PCSS dither
    //      without affecting brush/dyn shadow softness.
    glm::vec4  terrain_params;
    // Per-layer height transitions (start..end smoothstep widths):
    //   x..y: sand → grass (default 4..12)
    //   z..w: grass → dirt (default 28..42)
    glm::vec4  terrain_h_low;
    //   x..y: dirt → rock (default 58..80)
    //   z..w: rock → snow (default 95..120)
    glm::vec4  terrain_h_high;
    // Grass shader knobs (read by grass.vert / grass.frag):
    //   x: height_scale (multiplies per-blade Y, 0.3..2.0)
    //   y: alpha_cutoff (side-taper discard, 0..0.6)
    //   z: slope_n_min  (blade fades to 0 below this stored n.y)
    //   w: distance_density (0..1, 0 = uniform, 1 = strong far falloff)
    glm::vec4  grass_extra;
    // More grass knobs:
    //   x: alt_min (blade fades to 0 below this world Y)
    //   y: alt_max (blade fades to 0 above this world Y)
    //   z, w: unused
    glm::vec4  grass_extra2;
    // Sun shadow map's view-projection matrix. Grass.vert transforms its
    // world position into light-clip space and samples binding 7 (a
    // sampler2DShadow) for sun visibility. Computed each frame in
    // update_scene_ubo from sun_pitch/sun_yaw + the player position;
    // texel-grid-snapped so distant edges don't crawl as the camera moves.
    glm::mat4  light_vp;
    // Misc terrain knobs:
    //   x: shading contrast — power exponent on n_dot_l for terrain.
    //      1.0 = pure Lambert (default), 2..6 sharpens the lit/shadow
    //      transition by darkening grazing-angle areas.
    //   y: shadow_near_mult, z: gi_softener, w: gi_debug_viz flag.
    glm::vec4  terrain_extra;
    // Ocean / water plane.
    //   water_params.x = enabled (0/1)
    //   water_params.y = water level (world Y)
    //   water_params.z = wave strength (height of bump pattern, m)
    //   water_params.w = animation time (seconds, drives wave scroll)
    //   water_color.rgb = deep-water tint
    glm::vec4  water_params;
    glm::vec4  water_color;
    // Shallow-water colour (used near shores). Linear gradient from
    // this toward water_color over `water_shore.x` metres of depth,
    // with `water_shore.y` strength of FBM noise so the shore line
    // isn't perfectly geometric. .w slot reserved.
    glm::vec4  water_color_shallow;
    //   x: shore blend distance (m of water depth where shallow→deep)
    //   y: shore noise strength (0..1, perturbs the depth in shore band)
    //   z: TLAS reflection flag (0/1) — RT cubes / castle into water
    //   w: water transparency (0..1)
    glm::vec4  water_shore;
    // Shore-foam band overlay (rendered AFTER fresnel mix so it shows
    // through the reflection too):
    //   rgb = foam tint (typically pale blue-white)
    //   a   = strength (0 disables; 1 = full tint at depth=0)
    glm::vec4  water_foam_color;
    //   x   = band width in metres of NOISE-FREE depth (0.6 ≈ surf line)
    //   y/z/w spare
    glm::vec4  water_foam_params;
    // Volumetric ground-fog band:
    //   x: y_start (world Y where fog density starts)
    //   y: y_top   (world Y where fog density falls to zero)
    //   z: noise strength (0..1, FBM modulation on density)
    //   w: unused
    glm::vec4  fog_band;
    // Raymarched-terrain RT tuning, exposed as sliders so the user can
    // trade visual fidelity vs perf without recompiling. Each value is
    // the per-pixel cap applied INSIDE the terrain shader (cube.frag
    // continues to honour the global rt_flags / rt_params sliders for
    // its own RT work).
    //   x: terrain_pcss_samples_cap   — max PCSS shadow rays per terrain pixel
    //   y: terrain_gi_samples_cap     — max GI primary rays per terrain pixel
    //   z: terrain_ao_final_strength  — strength of final-colour AO multiplier
    //                                   (0 = no extra AO darkening; 1 = full)
    //   w: terrain_gi_bounces_cap     — max bounces in terrain GI loop
    glm::vec4  terrain_rt_extra;
    // Grass appearance — exposed via UI sliders.
    //   grass_color_top.rgb       = blade tip colour (lit, sunward)
    //   grass_color_top.w         = raymarched grass density (0..1, per-cell skip)
    //   grass_color_bottom.rgb    = blade base colour (shadowed, near ground)
    //   grass_color_bottom.w      = blade-base AO floor (0..1; 0=full dark base,
    //                               1=no AO darkening between base and tip)
    //   grass_color_ground.rgb    = CLOSE ground tint where grass is eligible
    //   grass_color_ground.w      = ground tint strength (0..1; 0=no tint,
    //                               1=full green-mat replaces rock/sand)
    //   grass_color_ground_far.rgb = FAR ground tint. Blends 0..1 from
    //                               grass_color_ground at grass_cutoff_soft_dist
    //                               to here at grass_raymarch_distance, in both
    //                               the terrain shader and the grass shader's
    //                               distance fade target.
    //   grass_color_ground_far.w  = unused.
    glm::vec4  grass_color_top;
    glm::vec4  grass_color_bottom;
    glm::vec4  grass_color_ground;
    glm::vec4  grass_color_ground_far;
    // Fake grass-cast shadows on the raymarched terrain.
    //   x: strength (0 = off, 1 = full shadow under dense grass cells)
    //   y: sample count (0..8, more = smoother but slower)
    //   z: max sample distance (m, controls shadow reach along sun XZ)
    //   w: unused
    glm::vec4  grass_shadow_params;
    // Shoreline grass tint. Blades within a few metres of water level
    // get drier/browner — reads as dry sandy grass at the beach,
    // fading back to the slider greens further uphill. ~13 ALU/pixel
    // in grass_raymarch.frag, no texture taps.
    //   grass_shore_color.rgb = tint colour
    //   grass_shore_color.w   = strength (0 = disabled, 1 = full mix)
    //   grass_shore_params.x  = fade distance in metres (height above
    //                            water level where the tint hits zero)
    //   grass_shore_params.yzw = unused
    glm::vec4  grass_shore_color;
    glm::vec4  grass_shore_params;
    // Shoreline TERRAIN tint — same shape as grass_shore_*, but
    // applied to the bare terrain `col` in terrain_raymarch.frag's
    // getMaterial. Lets the beach read as sand/wet-rock without
    // depending on slope detection alone.
    //   terrain_shore_color.rgb = tint colour
    //   terrain_shore_color.w   = strength (0 = disabled, 1 = full mix)
    //   terrain_shore_params.x  = fade distance (m above water level)
    //   terrain_shore_params.yzw = unused
    glm::vec4  terrain_shore_color;
    glm::vec4  terrain_shore_params;
    // Distance fog — standard exponential² atmospheric depth fog,
    // applied as final-step mix in cube/terrain/grass shaders.
    //   distance_fog_color.rgb   = fog tint (linear RGB)
    //   distance_fog_color.w     = master strength (0 = disabled)
    //   distance_fog_params.x    = density (per metre, exp² falloff)
    //   distance_fog_params.y    = start distance in metres (no fog before this)
    //   distance_fog_params.z    = height-falloff top (m); 0 disables height fog
    //   distance_fog_params.w    = max fog amount (0..1; clamp on the mix weight)
    glm::vec4  distance_fog_color;
    glm::vec4  distance_fog_params;
    // GENERAL terrain shore tint — applied to BARE terrain near water
    // (where grass eligibility is 0). Companion to terrain_shore_*
    // which now only tints the GRASS-area ground at shore.
    //   terrain_shore_general_color.rgb = tint colour
    //   terrain_shore_general_color.w   = strength (0 disables)
    //   terrain_shore_general_params.x  = fade distance (m above water)
    glm::vec4  terrain_shore_general_color;
    glm::vec4  terrain_shore_general_params;
    // Bare-shore SAND base colour used by the raymarched terrain's
    // beach blend. Was hardcoded (0.50, 0.45, 0.35) inside the
    // shader; now slider-driven so the user can tune the wet-sand /
    // dry-sand / coral hue without rebuilding shaders.
    //   terrain_sand_color.rgb = sand colour
    //   terrain_sand_color.w   = reserved
    glm::vec4  terrain_sand_color;
    // River water style extras (only consumed when water_style >= 1).
    //   .x = flow direction angle (radians; 0 = +X)
    //   .y = animation time speed multiplier (1 = engine default)
    //   .z = detail scale (multiplier on the 20× UV constant inside
    //        pm_flowing_normal; >1 = denser ripples, <1 = bigger swells)
    //   .w = foam contrast (0 = no foam, 1 = engine default, 2 = strong)
    glm::vec4  water_river_extra;
    // Underwater extinction colour for the river style. rgb is the
    // per-channel absorption coefficient (red attenuates fastest by
    // default), .w is density scale (1 = engine default).
    glm::vec4  water_river_extinct;
    // ReSTIR GI runtime knobs (session 3+ in docs/restir_plan.md).
    //   .x = enabled (0/1) — gates the temporal reservoir read in cube.frag
    //   .y = M_max (sample-count cap; default 32). Higher = more lag but
    //        less noise on stable surfaces.
    //   .z = disocclusion normal-dot threshold (default 0.8 ≈ 36° max angle)
    //   .w = reserved
    glm::vec4  restir_params;
    // SPOM (silhouette parallax occlusion mapping) tuning.
    //   .x = strength multiplier on the per-pixel height scale.
    //        1.0 = engine default (~4 cm peak-to-trough), 0 = disabled
    //        (effectively flat textures). Only affects the active SPOM
    //        materials (currently castle wall bricks; floors are
    //        excluded — see #198).
    //   .yzw = reserved
    glm::vec4  spom_params;
    // Reserved (was the unsafe single camera-local max — see
    // terrainMaxHeight() comment). Kept for layout stability.
    glm::vec4  terrain_local_info;
    // Per-cell maximum-height hi-Z grid for the terrain raymarcher.
    // 32×32 cells over the 2048 m world (64 m per cell), each holding
    // max(terrain height in cell) + safety margin. Baked once at
    // level load from terrain_data_.heights. The marcher samples the
    // cell at its current XZ each step: if a rising ray is above the
    // cell's max it skips ~1 cell forward and re-checks (never
    // declares sky — the NEXT cell is re-evaluated, so a distant
    // taller peak is never missed). 1024 floats packed 4-per-vec4.
    glm::vec4  terrain_max_grid[256];
    // Runtime knobs added later. Appended at the very END of the UBO
    // so the (large) terrain_max_grid array indexing in shaders stays
    // unchanged.
    //   .x = rocky-grass vertex displacement amplitude (m)
    //   .y = rocky displacement smoothing mip-bias (0 = crisp, +2 = blurred)
    //   .z = per-pixel POM far-fade distance (m)
    //   .w = reserved
    glm::vec4  terrain_disp_params;
    // Anti-tile sampling on terrain ground (cube.frag is_terrain block).
    //   .x = master toggle (0 / 1)
    //   .y = blend strength of the rotated sample (0..1)
    //   .zw = reserved
    glm::vec4  terrain_antitile_params;
    // Grass-cast shadow on rasterised terrain (cube.frag is_terrain),
    // sampled along the sun XZ direction through the grass mask
    // (binding 13). Separate from grass_shadow_params (raymarched
    // terrain only).
    //   .x = strength (0 = off, 1 = full dark under dense grass)
    //   .y = sample count along sun XZ (1..8)
    //   .z = max walk distance (m)
    //   .w = master toggle (0 / 1) -- shader fast-exits when 0 even if
    //        strength was left non-zero, to keep the toggle deterministic.
    glm::vec4  grass_shadow_on_terrain_params;
    // Side-lit grass shading (grass.frag / grass_raymarch.frag).
    //   .x = strength (0 = engine default flat shading, 1 = full wrap)
    //   .y = master toggle (0 / 1)
    //   .zw = reserved
    glm::vec4  grass_side_lit_params;
};

// ---- KHR ray tracing entry points ----
//
// Not exported by vulkan-1.lib; loaded once at device-create time and
// shared across rt.cpp / world.cpp / pipeline.cpp via this struct. The
// definition lives in helpers.cpp.
struct RtFuncs {
    PFN_vkCreateAccelerationStructureKHR create_as = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy_as = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_as = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR get_as_device_addr = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR get_as_build_sizes = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR cmd_write_as_props = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR cmd_copy_as = nullptr;
};
extern RtFuncs g_rt;
void load_rt_functions(VkDevice device);

// Validation messenger callback — increments g_validation_warning_count /
// g_validation_error_count from vk_engine.h and forwards the message to the
// engine logger.
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*);

// ---- Geometry / animation helpers ----

// Spark "temperature" gradient. t ∈ [0, 1], 0 = cooled / dim ember,
// 1 = freshly emitted yellow-white.
glm::vec3 spark_blackbody(float t);

// Rotate the local +Y axis to point along `dir`, then translate to `pos`.
// Used by projectile + spark trail rendering to align the cylinder mesh
// to its current velocity vector.
glm::mat4 align_local_y_to(glm::vec3 pos, glm::vec3 dir);

// Quaternion-only counterpart of align_local_y_to — used at Jolt body
// spawn time where the engine needs an orientation, not a full transform.
glm::quat align_local_y_to_quat(glm::vec3 dir);

// Compute the world-space AABB of an axis-aligned local box (half-extents
// `he`, centred at the local origin) under a rigid/non-shearing transform
// `m`. Closed-form alternative to the 8-corner loop: extent = |R| * he
// where R is the rotation/scale 3x3 — saves ~120 mul + 24 min/max per call
// and is exact for any rotation-and-scale transform (which is everything
// Jolt produces for rigid bodies).
inline void world_aabb_of_box(const glm::mat4& m, const glm::vec3& he,
                              glm::vec3& out_min, glm::vec3& out_max) {
    const glm::mat3 R(m);
    const glm::vec3 ext = glm::abs(R[0]) * he.x +
                          glm::abs(R[1]) * he.y +
                          glm::abs(R[2]) * he.z;
    const glm::vec3 ctr(m[3]);
    out_min = ctr - ext;
    out_max = ctr + ext;
}

// Smoothstep-ish curve for viewmodel recoil offset given how much of the
// recoil-stroke duration has already elapsed.
float recoil_kick(float t_remaining, float duration, float stroke);

// ---- Deterministic RNG used for box / spark spawns ----
uint32_t xorshift32(uint32_t& s);
float    frand(uint32_t& s);
float    frand_range(uint32_t& s, float a, float b);

// ---- Buffer / acceleration-structure address lookups ----
VkDeviceAddress buffer_device_address(VkDevice device, VkBuffer buffer);
VkDeviceAddress as_device_address(VkDevice device, VkAccelerationStructureKHR as);

// ---- Descriptor wiring ----
//
// Writes the scene desc set's bindings 0–5 (UBO, TLAS, materials, albedo[N],
// normal[N], prev_transforms). Called once after init_rt(); buffer/TLAS/view
// handles are stable for the life of the engine.
void write_scene_descriptors_once(
    VkDevice device, VkDescriptorSet set,
    VkBuffer ubo, VkAccelerationStructureKHR tlas, VkBuffer materials,
    VkBuffer prev_transforms,
    const VkImageView* albedo_views, const VkImageView* normal_views,
    uint32_t tex_count, VkSampler tex_sampler,
    const VkImageView* spom_height_views, uint32_t spom_count,
    VkImageView grass_mask_view,
    VkImageView fog_wisp_view,
    VkBuffer reservoir_prev,
    VkBuffer reservoir_cur);

} // namespace qlike
