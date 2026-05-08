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

namespace qlike {

// ---- Validation layer toggle ----
inline constexpr bool kUseValidationLayers =
#ifdef NDEBUG
    false;
#else
    true;
#endif

// Path of the autosaved settings file relative to CWD.
inline constexpr const char* kSettingsPath = "qlike_settings.cfg";

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
    //   y, z, w: unused.
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
    // Volumetric ground-fog band:
    //   x: y_start (world Y where fog density starts)
    //   y: y_top   (world Y where fog density falls to zero)
    //   z: noise strength (0..1, FBM modulation on density)
    //   w: unused
    glm::vec4  fog_band;
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
    uint32_t tex_count, VkSampler tex_sampler);

} // namespace qlike
