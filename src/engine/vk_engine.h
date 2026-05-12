#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "engine/frustum.h"
#include "engine/mesh.h"
#include "engine/physics_world.h"
#include "engine/skybox.h"
#include "engine/terrain.h"
#include "engine/grass.h"
#include "game/level.h"
#include "game/player.h"

namespace qlike {

// Running totals of validation-layer warnings / errors observed during this
// process. Incremented (atomically — the validation layer is allowed to call
// the messenger callback from worker threads) by the debug_utils messenger
// in vk_engine.cpp; surfaced in the ImGui HUD so new validation regressions
// are visible during dev without grepping logs. Inline so all TUs see the
// same counter instance.
inline std::atomic<uint64_t> g_validation_warning_count{0};
inline std::atomic<uint64_t> g_validation_error_count{0};

class Window;

struct RunOptions {
    int max_frames = -1;
    std::string screenshot_path;
    int screenshot_after_frames = 5;
    // Auto-demo: if > 0, the engine forces cursor-capture + Playing state and
    // synthesises walk+turn+shoot input for this many wall-clock seconds, then
    // requests window close. Lets us reproduce shoot+hit gameplay crashes
    // deterministically without manual play.
    float autodemo_seconds = 0.0f;
};

struct FrameData {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore swapchain_semaphore = VK_NULL_HANDLE;
    VkFence render_fence = VK_NULL_HANDLE;
    // Async-compute queue: separate command pool/buffer + a binary semaphore
    // signaled when the per-frame TLAS rebuild completes. The graphics
    // submit waits on this before the fragment-shader stage so cube.frag's
    // ray queries see a built TLAS. Both pool and buffer are allocated from
    // the compute queue family (which may equal graphics_queue_family_ on
    // hardware without a dedicated async-compute family).
    VkCommandPool compute_pool = VK_NULL_HANDLE;
    VkCommandBuffer compute_buffer = VK_NULL_HANDLE;
    VkSemaphore tlas_build_done = VK_NULL_HANDLE;
};

constexpr uint32_t kFrameOverlap = 2;

class VulkanEngine {
public:
    VulkanEngine();
    ~VulkanEngine();

    VulkanEngine(const VulkanEngine&) = delete;
    VulkanEngine& operator=(const VulkanEngine&) = delete;

    void init();
    void run(const RunOptions& opts = {});
    void shutdown();

private:
    void init_vulkan();
    void init_swapchain();
    void destroy_swapchain();
    void recreate_swapchain();

    // Apply preset levels (0=Low, 1=Med, 2=High, 3=Ultra) to rt_.* fields.
    // Called from the UI when the user picks a preset. Marks rt_ dirty so
    // the autosave debounce re-saves to disk.
    void apply_quality_preset(int level);

    // Recompute render_extent_ from swapchain_extent_ × rt_.render_scale and
    // re-create scene_color / history / motion_vec / bloom / depth so the
    // pipeline picks up the new resolution. Cheap (no shaders rebuild).
    // Triggered by the Apply button in the resolution menu.
    void apply_render_scale();
    void init_depth_image();
    void destroy_depth_image();
    void init_commands();
    void init_sync();

    // Show a blocking loader frame on the swapchain. Cheap progress bar
    // drawn via vkCmdClearAttachments rectangles — no pipelines, no
    // ImGui. Safe to call as soon as init_sync() has run; uses the same
    // FrameData ring as the main render loop so it consumes one frame
    // index per call. `progress` is clamped to [0,1].
    void present_loader_frame(const char* label, float progress);
    void init_readback_buffer();
    void destroy_readback_buffer();
    void init_pipeline();
    void destroy_pipeline();
    void init_world();
    // Lazy-build paths for the rasterised terrain mesh/chunks and the
    // 800k-blade grass buffer. init_world skips these when the matching
    // raymarch flag is on (saves hundreds of MB of VRAM). If the user
    // toggles the flag OFF at runtime, the per-frame settings tick calls
    // these to materialise the buffers on demand. Idempotent — no-op
    // when already built.
    void ensure_terrain_raster_built();
    void ensure_grass_raster_built();
    void init_imgui();
    void destroy_imgui();
    void build_menu_ui();
    void reset_player();
    void init_descriptors();
    void destroy_descriptors();
    void init_rt();
    void destroy_rt();
    void update_scene_ubo();
    void rebuild_tlas(VkCommandBuffer cmd);
    // Rebuild static_brush_instances_ / static_brush_materials_ from
    // world_.brushes for the current rt_.textures_enabled. Cheap; called only
    // on init and when textures-toggled.
    void bake_static_brushes();
    // Build the merged-static BLAS once per level. Concatenates every
    // world_.brush into a single world-space triangle soup, then runs an
    // AS-build over it with PREFER_FAST_TRACE. Called from init_rt() AFTER
    // init_world() has populated world_.brushes.
    void bake_merged_static_blas();
    // Build a BLAS over the terrain heightmap mesh. Called once per level
    // load. The terrain mesh itself is created in init_world().
    void bake_terrain_blas();
    // Depth-only pre-pass over the same brush + dyn-prop geometry that
    // render_world() draws — populates depth_image_ so the subsequent color
    // pass's depth_compare=LESS_OR_EQUAL early-rejects occluded fragments
    // before cube.frag's heavy inline-RT body runs. Particles, projectiles
    // and the viewmodel skip the prepass (sparse / screen-space).
    void render_world_depth_pass(VkCommandBuffer cmd);
    void build_hud_ui();
    void load_settings();
    // One-key peek that runs BEFORE init_world so the heightmap source
    // (FBM vs FastNoiseLite) can match what the user toggled last session.
    void preload_terrain_raymarch_flag();
    void save_settings() const;
    // Diff rt_ against last-saved snapshot; debounce-save 500ms after the
    // last detected change so dragging a slider doesn't churn the disk.
    void maybe_autosave_settings(float frame_dt);
    void init_taa();
    void destroy_taa();
    void recreate_taa_targets();
    void init_bloom();
    void destroy_bloom();
    void recreate_bloom_targets();
    void init_compose();
    void destroy_compose();
    // Rewrite compose_desc_sets_[]'s image bindings (history, depth, sky,
    // bloom) so they point at the current views. Needed after a swapchain
    // resize destroys and recreates depth + history + bloom — without this,
    // compose samples freed image views.
    void rewrite_compose_image_bindings();
    void run_bloom_chain(VkCommandBuffer cmd);
    void init_skybox();
    void destroy_skybox_resources();
    void init_textures();
    void destroy_textures();
    void init_grass_mask_texture();
    void destroy_grass_mask_texture();
    void init_fog_wisp_texture();
    void destroy_fog_wisp_texture();
    void init_pipeline_cache();
    void destroy_pipeline_cache();
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;

    void draw(uint32_t img_index);
    void render_world(VkCommandBuffer cmd);
    void capture_screenshot(VkCommandBuffer cmd, VkImage src,
                            VkExtent2D extent, VkImageLayout pre_layout);
    void write_ppm(const std::string& path, uint32_t w, uint32_t h);

    FrameData& current_frame() { return frames_[frame_number_ % kFrameOverlap]; }

    std::unique_ptr<Window> window_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = 0;

    // Async-compute queue used for the per-frame TLAS rebuild. On NVIDIA this
    // is typically a dedicated compute family, allowing the rebuild to run
    // concurrently with the graphics queue's work (the previous frame's
    // raster). On hardware without a separate compute family (e.g. mobile)
    // vk-bootstrap returns a queue from the graphics family — falls back to
    // simply using the graphics queue (no perf gain, no behavior change).
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = 0;
    bool compute_queue_distinct_ = false;  // true iff family != graphics_queue_family_

    VmaAllocator allocator_ = nullptr;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent_{};
    // Resolution at which the world / TAA / bloom passes render. Equals
    // swapchain_extent_ × rt_.render_scale (clamped to ≥ 1×1). Scene-color,
    // motion-vec, history, depth and bloom mip-0 are all sized off this.
    // Only the final compose pass targets swapchain_extent_; its linear
    // sampler reads `history_color` at render_extent_ to do the (up/down)
    // scale. Updated alongside swapchain_extent_ in recreate_swapchain().
    VkExtent2D render_extent_{};
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    std::vector<VkSemaphore> render_semaphores_;

    VkImage depth_image_ = VK_NULL_HANDLE;
    VkImageView depth_view_ = VK_NULL_HANDLE;
    VmaAllocation depth_alloc_ = nullptr;
    VkFormat depth_format_ = VK_FORMAT_D32_SFLOAT;

    std::array<FrameData, kFrameOverlap> frames_{};
    uint64_t frame_number_ = 0;

    VkBuffer readback_buffer_ = VK_NULL_HANDLE;
    VmaAllocation readback_alloc_ = nullptr;
    VkDeviceSize readback_size_ = 0;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkShaderModule vert_module_ = VK_NULL_HANDLE;
    VkShaderModule frag_module_ = VK_NULL_HANDLE;
    // Depth-only pipeline used for the early-Z pre-pass; shares pipeline
    // layout + vertex shader with `pipeline_`.
    VkPipeline depth_pipeline_ = VK_NULL_HANDLE;
    // Terrain-specific pipelines that bind a second vertex stream (parent_y)
    // and morph between LOD 0 and LOD 1 in the vertex shader. Color and
    // depth-prepass variants must both morph or their depth values
    // diverge and the LESS_OR_EQUAL test would discard fragments.
    VkPipeline      terrain_pipeline_       = VK_NULL_HANDLE;
    VkPipeline      terrain_depth_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule  terrain_vert_module_    = VK_NULL_HANDLE;
    void init_terrain_pipelines();
    void destroy_terrain_pipelines();

    // Procedural FBM heightfield ray-marched terrain — selectable
    // alternate to the chunked rasterised path. Fullscreen triangle
    // (gl_VertexIndex 0..2) writes hit-point depth via gl_FragDepth so
    // it composites with rasterised cubes / castle / dyn-props.
    // Toggled by rt_.terrain_raymarch_enabled.
    VkPipeline      terrain_raymarch_pipeline_   = VK_NULL_HANDLE;
    VkShaderModule  terrain_raymarch_vert_module_ = VK_NULL_HANDLE;
    VkShaderModule  terrain_raymarch_frag_module_ = VK_NULL_HANDLE;
    void init_terrain_raymarch_pipeline();
    void destroy_terrain_raymarch_pipeline();

    // Low-res raymarch render target. When `terrain_raymarch_scale` < 1
    // the FBM ray-march renders into these (cheaper, since each pixel
    // does ~hundreds of FBM evaluations) and a separate compose pass
    // bilinear-upscales them into the full-res scene_color/depth with
    // depth-aware blending so rasterised cubes/castle still occlude
    // correctly. Each LR* image is sized at render_extent_ × scale.
    VkImage         tr_lr_color_image_  = VK_NULL_HANDLE;
    VmaAllocation   tr_lr_color_alloc_  = nullptr;
    VkImageView     tr_lr_color_view_   = VK_NULL_HANDLE;
    VkImage         tr_lr_motion_image_ = VK_NULL_HANDLE;
    VmaAllocation   tr_lr_motion_alloc_ = nullptr;
    VkImageView     tr_lr_motion_view_  = VK_NULL_HANDLE;
    VkImage         tr_lr_depth_image_  = VK_NULL_HANDLE;
    VmaAllocation   tr_lr_depth_alloc_  = nullptr;
    VkImageView     tr_lr_depth_view_   = VK_NULL_HANDLE;
    VkSampler       tr_lr_sampler_      = VK_NULL_HANDLE;
    // NEAREST sampler dedicated to u_lr_depth (binding 11). Bilinear on
    // depth values bleeds terrain↔sky across silhouettes — at low
    // raymarch scales the >=0.9999 sky-discard then either eats valid
    // terrain (gaps) or writes fake mid-depth halos.
    VkSampler       tr_lr_depth_sampler_ = VK_NULL_HANDLE;

    // --- Attachment-based Variable Rate Shading (VK_KHR_fragment_shading_rate) ---
    // Small R8_UINT image sized at (render_extent + tile - 1) / tile. Each
    // texel encodes the shading rate for an N×N pixel block of the
    // framebuffer via `(width_log2 << 2) | height_log2`: 0 = 1×1 (full),
    // 5 = 2×2 (quarter cost), 10 = 4×4 (1/16 cost). Populated once with a
    // center-fine / edge-coarse vignette so distant terrain (which is
    // already LOD'd in the FBM) pays one shader invocation per 4 or 16
    // pixels.
    //
    // Only applied to the LR raymarch render pass — the FBM-heavy
    // fullscreen tri. Compose pass + main world pass stay at native rate.
    bool            vrs_supported_      = false;
    VkExtent2D      vrs_texel_size_{};   // device-reported tile size
    VkExtent2D      vrs_extent_{};       // attachment image extent
    VkImage         vrs_image_          = VK_NULL_HANDLE;
    VmaAllocation   vrs_alloc_          = nullptr;
    VkImageView     vrs_view_           = VK_NULL_HANDLE;
    void init_vrs_attachment();
    void destroy_vrs_attachment();
    void recreate_vrs_attachment();
    VkExtent2D      tr_lr_extent_{};

    // Compose pipeline that samples the low-res raymarch and writes
    // upscaled color/motion + gl_FragDepth into scene_color/depth with
    // depth-test enabled (so closer rasterised geometry survives).
    VkPipeline      tr_compose_pipeline_     = VK_NULL_HANDLE;
    VkShaderModule  tr_compose_frag_module_  = VK_NULL_HANDLE;

    void init_terrain_raymarch_lowres();
    void destroy_terrain_raymarch_lowres();
    // Recreate when render_extent_ or terrain_raymarch_scale changes.
    void recreate_terrain_raymarch_lowres();
    void init_terrain_raymarch_compose_pipeline();
    void destroy_terrain_raymarch_compose_pipeline();
    // Helper: returns true if the LR-upscale path should run this frame.
    bool tr_use_lowres() const;
    // Pass entry-points for the LR raymarch + the upscale composite.
    void render_terrain_raymarch_lr(VkCommandBuffer cmd);
    void render_terrain_raymarch_compose(VkCommandBuffer cmd);
    VkShaderModule depth_frag_module_ = VK_NULL_HANDLE;
    // Minimal vertex shader bound to the depth-prepass pipeline.
    // Replaces cube.vert in that pipeline so the prepass doesn't pay
    // the cost of model + prev_mvp matrix multiplies and 9 unused
    // varyings (cube_depth.frag is empty — no varyings consumed).
    VkShaderModule depth_vert_module_ = VK_NULL_HANDLE;

    Mesh cube_mesh_{};
    Mesh cylinder_mesh_{};
    Skybox skybox_{};
    // Procedural heightmap data — kept around so the streaming Phase 2
    // and the Phase 4 sculpt brush can read/edit it. Phase 1 only needs
    // it to compute the castle anchor and Jolt collision shape.
    struct TerrainData {
        int   dim = 0;
        float cell = 0.0f;
        float origin_x = 0.0f;
        float origin_z = 0.0f;
        std::vector<float> heights;
    };
    TerrainData terrain_data_{};

    // Bilinearly sample the terrain heightmap at world XZ. Returns 0 if
    // the heightmap isn't loaded or (x, z) is outside its grid. Used by
    // the player update to clamp the kinematic capsule to the ground —
    // Jolt's HeightFieldShape collides with crates/projectiles but the
    // player runs on a custom AABB-sliding path that doesn't query Jolt.
    float sample_terrain_height(float x, float z) const;

    // PBR-ish material textures. kTextureCount must match the size of the
    // sampler arrays in cube.frag.
    static constexpr int kTextureCount = 7;
    struct TextureSlot {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        VkImageView view = VK_NULL_HANDLE;
    };
    TextureSlot albedo_textures_[kTextureCount]{};
    TextureSlot normal_textures_[kTextureCount]{};
    // Height-map array for SPOM (parallax occlusion). Only the materials
    // that actually need parallax depth get a slot here; cube.frag picks
    // the right entry by albedo index. Order MUST match cube.frag's
    // height_idx_for_albedo() switch:
    //   [0] Bricks078         (albedo idx 1) — castle outer walls
    //   [1] PaintedBricks001  (albedo idx 4) — keep walls
    //   [2] Tiles130          (albedo idx 5) — courtyard floor
    //   [3] Tiles074          (albedo idx 6) — keep interior floor
    static constexpr int kSpomMaterialCount = 4;
    TextureSlot spom_height_textures_[kSpomMaterialCount]{};

    // Grass-density mask. R8 1024² covering 2048m world (≈2 m/texel).
    // Baked at level load: CPU runs the same presence × slope formula
    // both shaders previously evaluated per-pixel, packs the result
    // into a single value the shaders sample once. Replaces the
    // 9-cell noised() storm in grass_raymarch.frag::map() and the
    // duplicate computation in terrain_raymarch.frag::grassEligibility,
    // and is the texture the (forthcoming) drawn-mask editor will paint
    // into.
    TextureSlot grass_mask_{};
    static constexpr int kGrassMaskDim = 1024;
    // CPU mirror of the RG8 grass mask. Populated alongside the GPU
    // upload in init_grass_mask_texture so the GrassAdd/GrassRemove
    // brush can paint into it without rebaking the noise field.
    // `grass_mask_dirty_` triggers a re-upload at the safe point in
    // the next frame.
    std::vector<uint8_t> grass_mask_data_;
    bool grass_mask_dirty_ = false;
    void reupload_grass_mask();
    bool save_grass_mask();
    bool load_grass_mask();
    // Fog wisp pattern, 256² R8. Pre-baked 3-octave noise so the
    // terrain raymarch fog density + godray probe loops sample one
    // textureLod() per tap instead of running 3× noise2() each — the
    // dominant per-pixel cost on godray-on configs.
    TextureSlot fog_wisp_{};
    static constexpr int kFogWispDim = 256;
    VkSampler   texture_sampler_ = VK_NULL_HANDLE;
    // Texture index assigned to the four spawnable box variants (so dynamic
    // boxes pick from a wood/metal/painted/brick palette at spawn time).
    static constexpr int kBoxTextureChoices = 4;

    game::Level world_;
    // Per-static-brush model matrix cache. Built once in init_world from
    // (translate(b.center) * scale(b.size)). Static brushes never move,
    // so the depth pre-pass, color pass and sun-shadow pass all reuse
    // these instead of rebuilding ~190 matrices × 3 passes/frame.
    std::vector<glm::mat4> static_brush_models_;
    game::Player player_{};
    std::unique_ptr<PhysicsWorld> physics_;
    std::unique_ptr<class AudioEngine> audio_;

    struct DynamicProp {
        uint32_t body_id;
        // Cached PhysicsWorld::BodyHandle (Jolt BodyID's index+seq). All
        // per-frame read paths use this to skip the uint32_t→BodyID hash
        // lookup. 0 = invalid (lookup failed at spawn time).
        uint32_t jolt_handle = 0;
        glm::vec3 full_size;   // for rendering: scale of unit cube mesh
        glm::vec4 color;            // texture-tint
        glm::vec4 fallback_color;   // saturated color used when textures off
        int   tex_albedo = -1;
        int   tex_normal = -1;
        float uv_scale   = 1.0f;
    };
    std::vector<DynamicProp> dyn_props_;

    // Hyper-fast cylinder bullets. Jolt body uses LinearCast motion quality
    // (sweep-cast at the start of each step) so the projectile cannot tunnel
    // through dynamic boxes between physics ticks. Despawn on TTL.
    struct Projectile {
        uint32_t body_id;
        uint32_t jolt_handle = 0;  // cached BodyHandle, see DynamicProp
        float    radius;
        float    half_length;
        float    ttl;            // seconds remaining
        float    initial_speed;  // for impact-detection threshold
        glm::vec3 initial_dir;
        glm::vec4 color;
        glm::vec3 emissive;
    };
    std::vector<Projectile> projectiles_;
    void fire_projectile(glm::vec3 origin, glm::vec3 direction);
    void update_projectiles(float dt);
    static constexpr float kProjectileTtl     = 3.0f;
    static constexpr float kProjectileSpeed   = 220.0f;   // m/s
    static constexpr float kProjectileRad     = 0.025f;   // thin
    static constexpr float kProjectileHalf    = 0.40f;    // long (= 0.8m total)
    static constexpr float kProjectileMass    = 10.0f;    // kg — really shoves
    static constexpr float kProjectileMinSpeed = 100.0f;  // despawn below this

    // Hit-effect sparks. Jolt sphere bodies for collision (so they bounce
    // realistically off every surface in the world), rendered as thin
    // velocity-stretched cylinders for the streak look. Color follows a
    // blackbody-style cooldown curve over the TTL.
    // Trail length: each particle costs (kSparkTrailLen-1) cylinder draw
    // calls per frame in draw_spark_trails. With kMaxParticles = 384 a length
    // of 8 = 2688 draw calls per frame — that's CPU-side command-buffer
    // work and was the dominant cost when shooting at a wall at high RPM.
    // 4 = 1152 draws → fps recovers most of the way to no-fire baseline.
    static constexpr int kSparkTrailLen = 4;
    struct Particle {
        uint32_t  body_id;
        uint32_t  jolt_handle = 0;  // cached BodyHandle, see DynamicProp
        float     ttl;
        float     ttl_max;
        float     vis_radius;    // visual cylinder radius
        float     vis_base_half; // visual cylinder half-length at zero speed
        // Cached world matrix from this frame's first physics query, valid
        // when valid_world is true. update_particles fills it; rebuild_tlas
        // and draw_spark_trails read it instead of re-locking Jolt.
        glm::mat4 world{1.0f};
        bool      valid_world = false;
        // Ring buffer of recent world-space positions, sampled at render rate.
        // Renderer draws cylinders between consecutive samples → real motion-
        // blur trails. trail_count grows up to kSparkTrailLen.
        glm::vec3 trail[kSparkTrailLen]{};
        int       trail_count = 0;
    };
    std::vector<Particle> particles_;
    void spawn_hit_particles(glm::vec3 pos, glm::vec3 reflect_dir,
                             glm::vec3 incoming_dir);
    void update_particles(float dt);
    void draw_spark_trails(VkCommandBuffer cmd, const glm::mat4& vp);
    // Renders shadow-map debug visuals when rt_.shadow_debug_overlay is on:
    // wireframe of the light ortho frustum, sun-direction line at the
    // player, and a circle at the bake/shadow-map cross-fade boundary.
    // Lines are drawn as thin emissive cylinders (re-uses cylinder mesh).
    void draw_shadow_debug(VkCommandBuffer cmd, const glm::mat4& vp);
    static constexpr int   kParticlesPerHit   = 14;
    static constexpr float kParticleTtl       = 0.6f;
    static constexpr float kParticleColRad    = 0.018f;  // collision sphere
    static constexpr float kParticleVisRad    = 0.008f;  // visual cylinder radius
    static constexpr float kParticleVisBase   = 0.025f;  // visual half-length @ 0 m/s
    static constexpr float kParticleStretch   = 0.06f;   // m of streak per (m/s)
    static constexpr float kParticleMaxStretchMul = 4.0f;
    static constexpr int   kMaxParticles      = 384;

    // Wall decals — small dark "scorch mark" quads placed at each bullet
    // impact, oriented along the surface normal. Capped FIFO; fade out
    // over kDecalTtl. Rendered via the existing brush pipeline (cube
    // mesh scaled flat) so no new pipeline is needed.
    struct Decal {
        // Position + normal in WORLD space if parent_body_id == 0
        // (impact on static brush). For dyn-body impacts pos/normal are
        // stored in the body's LOCAL space; render-time multiplies them
        // by the parent's current world matrix so the decal travels
        // with the moving crate.
        glm::vec3 pos;
        glm::vec3 normal;
        uint32_t  parent_body_id = 0;   // 0 = static / world-space
        uint32_t  parent_handle  = 0;   // cached BodyHandle for parent
        float     size;
        float     age = 0.0f;
        float     ttl;
    };
    std::vector<Decal> decals_;
    static constexpr int   kMaxDecals = 64;
    static constexpr float kDecalTtl  = 25.0f;
    void spawn_impact_decal(glm::vec3 hit_pos, glm::vec3 incoming_dir);
    void update_decals(float dt);
    void draw_decals(VkCommandBuffer cmd, const glm::mat4& vp);
    // Speed range / count are scaled by GameSettings::spark_scale at spawn.

    // Viewmodel: a "gun" rendered at a fixed offset from the camera. Two
    // paths:
    //   - If assets/gun.glb exists, it's loaded once into gun_meshes_ + their
    //     own BLAS list and rendered via that path.
    //   - Otherwise, a procedural fallback is drawn from the existing cube +
    //     cylinder primitives (barrel, body, grip).
    struct ViewmodelPart {
        enum class Kind { Cube, Cylinder } kind;
        glm::mat4 local;        // local-to-camera offset matrix
        glm::vec4 color;
    };
    std::vector<ViewmodelPart> viewmodel_proc_parts_;
    void init_viewmodel();
    void destroy_viewmodel();
    void draw_viewmodel(VkCommandBuffer cmd, const glm::mat4& vp,
                        const glm::mat4& view_inv);

    // glTF-loaded viewmodel (optional). Each primitive becomes its own Mesh
    // (no separate BLAS — viewmodel is raster-only, so it doesn't cast world
    // shadows or contribute to GI; that's the conventional FPS look).
    struct ViewmodelMesh {
        Mesh mesh;
        glm::vec4 base_color = glm::vec4(0.6f, 0.6f, 0.65f, 1.0f);
    };
    std::vector<ViewmodelMesh> viewmodel_gltf_;
    glm::mat4 viewmodel_root_offset_{ 1.0f };    // local-to-camera placement

    // Spacejet flyover system — glTF asset loaded once, rendered as
    // an instanced flock that periodically streams across the level.
    // No physics, no BLAS; raster-only. Each `SpacejetFlight` is a
    // straight-line path at constant velocity at high altitude over
    // the castle area. spacejet_spawn_timer_ counts down to the
    // next wave (1-5 jets in tight formation, random direction).
    struct SpacejetMesh {
        Mesh mesh;
        glm::vec4 base_color = glm::vec4(0.7f, 0.72f, 0.78f, 1.0f);
    };
    struct SpacejetFlight {
        glm::vec3 pos{};          // current world position (XZ tracked, Y constant)
        float     heading = 0.0f; // current yaw (radians, 0 = +Z)
        float     turn_rate = 0.0f; // rad/sec (signed; 0 = straight line)
        float     speed   = 90.0f;
        float     scale   = 1.0f;
        float     t       = 0.0f;
        float     ttl     = 30.0f;
        glm::vec3 prev_pos{};     // previous-frame pos for TAA motion vec
        float     prev_heading = 0.0f;
    };
    std::vector<SpacejetMesh>   spacejet_meshes_;
    std::vector<SpacejetFlight> spacejet_flights_;
    glm::vec3 spacejet_centre_offset_{0.0f};      // gltf bbox-centre, applied per draw
    float     spacejet_base_scale_      = 1.0f;   // gltf-bbox-derived scale
    float     spacejet_spawn_timer_     = 2.0f;   // first wave after 2s
    uint32_t  spacejet_rng_state_       = 0xC0DEFEEDu;
    void init_spacejet();
    void destroy_spacejet();
    void tick_spacejet(float dt);                 // move + spawn + despawn

    // 60 dynamic crates: 100 still let the autodemo TDR (~2.4s on a 4080
    // even with the merged-static BLAS), so we drop further. The bottleneck
    // is per-pixel ray budget × in-flight projectiles + active dyn props,
    // not BLAS topology. Static-brush merge (see rt.cpp::bake_merged_static_blas)
    // already shrank static-side load; this caps the dynamic side.
    static constexpr int   kMaxDynProps    = 60;
    static constexpr float kSpawnInterval  = 0.4f;  // seconds between spawns
    float spawn_timer_ = 0.0f;
    uint32_t spawn_rng_state_ = 0xc0ffee42u;
    void spawn_random_box();

    // Reused buffer of "static brush AABBs + per-tick dynamic-box AABBs" so
    // the player can't walk through Jolt-driven boxes.
    std::vector<collision::AABB> tick_aabbs_;
    // Per-dyn-prop AABB cache. Sleeping bodies reuse their last-computed
    // AABB instead of re-running the 8-corner mat-vec each physics tick.
    // At 100 dyn props × 6 ticks/frame × 8 corners that's ~5k mat-vec
    // per frame skipped once the bodies settle. Indexing matches
    // dyn_props_; the cached body_id catches FIFO-eviction renumbering
    // (front-erase shifts every other slot down by 1, so a slot's body
    // identity can change without size changing — without the body_id
    // check the cached AABB would describe a now-removed box and the
    // player would walk through the new occupant).
    struct DynTickAabb {
        collision::AABB aabb{};
        uint32_t body_id = 0;   // 0 = invalid / not populated
    };
    std::vector<DynTickAabb> dyn_tick_aabb_cache_;
    void rebuild_tick_aabbs();

    // Per-frame cache of dynamic-box render state. Computed once at the top of
    // draw() and reused by both rebuild_tlas() (instance writes) and
    // render_world() (frustum culling). Avoids the O(n×3) recomputation that
    // existed when each subsystem queried Jolt independently.
    struct DynRender {
        bool      valid = false;
        // body_id of the dyn_props_ entry this slot was populated from. Lets
        // rebuild_dyn_render_cache() validate the slot still maps to the same
        // body before reusing its cached transform across frames (slot
        // identity changes when boxes despawn / swap-remove reorders dyn_props_).
        uint32_t  body_id = 0;
        glm::mat4 world{1.0f};        // current-frame matrix from Jolt
        // Previous-frame world matrix. Future per-instance motion-vector pass
        // needs prev_world to reproject dynamic surfaces correctly (the
        // current TAA reprojection assumes a static world, which smears
        // trails behind moving boxes). Populated from the slot's previous
        // `world` value before each frame's recompute; equals `world` for
        // sleeping bodies (no motion) and for newly-created slots (no prior).
        glm::mat4 prev_world{1.0f};
        glm::vec3 aabb_min{0.0f};
        glm::vec3 aabb_max{0.0f};
    };
    std::vector<DynRender> dyn_render_cache_;
    void rebuild_dyn_render_cache();

    // Shooting state.
    int score_ = 0;
    float fire_cooldown_ = 0.0f;   // seconds remaining until next auto-fire shot
    float muzzle_flash_timer_ = 0.0f; // seconds of flash remaining
    float recoil_timer_ = 0.0f;       // seconds since last shot, for kick anim
    static constexpr float kMuzzleFlashDuration = 0.07f;   // ~4 frames @ 60fps
    static constexpr float kRecoilDuration      = 0.14f;   // total kick-then-return
    static constexpr float kRecoilStroke        = 0.05f;   // m of pull-back
    void try_fire_hitscan(glm::vec3 origin, glm::vec3 direction);

    // After the player's tick: push any dynamic box the player is in contact
    // with. `pre_velocity` is the player's velocity *before* slide_move
    // clamped it — that's the velocity that would have continued into the box.
    void apply_player_pushes(glm::vec3 pre_velocity);

    enum class State { Playing, Paused };
    State state_ = State::Playing;
    bool prev_menu_key_ = false;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
    bool imgui_initialized_ = false;

    float pos_log_timer_ = 0.0f;

    // Per-frame draw stats (for HUD).
    int last_draw_static_ = 0;
    int last_draw_dyn_    = 0;
    int last_culled_      = 0;

    // Fixed-timestep physics state.
    float physics_accumulator_ = 0.0f;
    glm::vec3 prev_player_position_{};
    int last_physics_ticks_ = 0;
    float last_frame_dt_ = 0.0f;
    float ema_fps_ = 0.0f;
    // Rolling history for the Three.js-style HUD panels. Each frame
    // appends to the head; PlotLines reads the whole array.
    static constexpr int kStatsHistory = 80;
    float fps_history_[kStatsHistory] = {};
    float ms_history_[kStatsHistory]  = {};
    int   stats_history_head_ = 0;

    // --- Lighting / scene UBO ---
    VkBuffer scene_ubo_buffer_ = VK_NULL_HANDLE;
    VmaAllocation scene_ubo_alloc_ = nullptr;

    VkDescriptorPool scene_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout scene_desc_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet scene_desc_set_ = VK_NULL_HANDLE;

    // --- Ray tracing acceleration structures ---
    VkBuffer blas_buffer_ = VK_NULL_HANDLE;
    VmaAllocation blas_alloc_ = nullptr;
    VkAccelerationStructureKHR blas_ = VK_NULL_HANDLE;
    VkDeviceAddress blas_device_address_ = 0;

    // Second BLAS for the projectile cylinder mesh.
    VkBuffer cylinder_blas_buffer_ = VK_NULL_HANDLE;
    VmaAllocation cylinder_blas_alloc_ = nullptr;
    VkAccelerationStructureKHR cylinder_blas_ = VK_NULL_HANDLE;
    VkDeviceAddress cylinder_blas_device_address_ = 0;

    // Merged static-brush BLAS: every world_.brushes box pre-transformed into
    // world space and concatenated into a single triangle soup. One TLAS
    // instance covers the whole castle (vs ~190 small instances), which keeps
    // the TLAS small + reduces per-ray inter-instance traversal overhead.
    // Materials are looked up per-primitive in cube.frag using
    // gl_PrimitiveID / 12 (12 tris per cube) — see kStaticBlasInstSentinel.
    Mesh merged_static_mesh_{};
    VkBuffer merged_static_blas_buffer_ = VK_NULL_HANDLE;
    VmaAllocation merged_static_blas_alloc_ = nullptr;
    VkAccelerationStructureKHR merged_static_blas_ = VK_NULL_HANDLE;
    VkDeviceAddress merged_static_blas_device_address_ = 0;
    uint32_t merged_static_brush_count_ = 0;

    // Phase 1 terrain BLAS: a single full-resolution mesh used by the
    // RT path (shadows / GI / AO see crisp ground regardless of raster
    // LOD). Raster goes through `terrain_chunks_` instead so the visible
    // mesh can swap LOD per chunk and Phase 4 sculpt can rebuild only
    // affected chunks.
    Mesh terrain_mesh_{};
    VkBuffer terrain_blas_buffer_ = VK_NULL_HANDLE;
    VmaAllocation terrain_blas_alloc_ = nullptr;
    VkAccelerationStructureKHR terrain_blas_ = VK_NULL_HANDLE;
    VkDeviceAddress terrain_blas_device_address_ = 0;
    glm::vec3 terrain_color_ = glm::vec3(0.45f, 0.42f, 0.32f);

    // Phase 2 chunked raster terrain.
    TerrainChunkSet terrain_chunks_{};

    // Instanced grass — one shared blade mesh + per-blade transforms.
    // Built once at level load via build_grass(). Pipeline is separate
    // from the cube pipeline because grass uses its own (cheap) shading
    // path — no RT shadows or GI per blade.
    GrassMesh grass_{};
    VkPipeline       grass_pipeline_       = VK_NULL_HANDLE;
    VkShaderModule   grass_vert_module_    = VK_NULL_HANDLE;
    VkShaderModule   grass_frag_module_    = VK_NULL_HANDLE;
    void init_grass_pipeline();
    void destroy_grass_pipeline();

    // Raymarched grass — fullscreen SDF blade field. Drawn into the same
    // color/motion/depth attachments as cube draws after them, so the
    // depth pre-pass (cubes/castle/dyn) correctly occludes grass. Reuses
    // taa.vert as a fullscreen-triangle vertex shader.
    VkPipeline       grass_rm_pipeline_   = VK_NULL_HANDLE;
    VkShaderModule   grass_rm_frag_module_ = VK_NULL_HANDLE;
    void init_grass_raymarch_pipeline();
    void destroy_grass_raymarch_pipeline();
    void render_grass_raymarch(VkCommandBuffer cmd);

    // Heightmap raw heights as an R32_SFLOAT texture. Used by the
    // procedural raymarched terrain shader (binding 8) so the
    // procedural surface follows the gameplay heightmap shape.
    // Uploaded once at init from terrain_data_.heights; sculpt edits
    // don't propagate (re-call init_terrain_height_texture to refresh).
    VkImage           terrain_height_image_   = VK_NULL_HANDLE;
    VmaAllocation     terrain_height_alloc_   = nullptr;
    VkImageView       terrain_height_view_    = VK_NULL_HANDLE;
    VkSampler         terrain_height_sampler_ = VK_NULL_HANDLE;
    float             terrain_height_max_     = 0.0f;   // upper-bound for raymarch early-out
    void init_terrain_height_texture();
    void destroy_terrain_height_texture();
    // Persist / load the live (possibly sculpt-edited) heightmap to
    // assets/level1_heights.bin. Save writes dim/cell/origin header +
    // raw float data; load is automatic in init_world if the file
    // exists and metadata matches.
    bool save_terrain_heights();
    // Re-upload the heightmap delta (current - baseline) into binding
    // 8 so the raymarch shader's terrainM picks up sculpt edits +
    // plateau-noise + saved-overlay changes. Baseline is captured at
    // level load (the procedural FBM result); only deltas are sent.
    // Cheap (one staging copy of dim×dim×4 bytes); use sparingly.
    void refresh_terrain_height_texture();
    // Set after sculpt / plateau-noise / load so the next frame can
    // call refresh_terrain_height_texture once instead of every brush
    // step.
    bool terrain_height_dirty_ = false;
    // Procedural baseline captured at init_world. delta_to_upload =
    // terrain_data_.heights[i] - terrain_baseline_heights_[i].
    std::vector<float> terrain_baseline_heights_;
    // Add stratified noise to the heightmap inside the rectangular
    // plateau region — gives the castle pad some natural relief
    // instead of being a perfect flat. Modifies terrain_data_ in
    // place; caller can save it. Amplitude is in metres.
    void add_plateau_noise(float amplitude_m, float frequency);
    // Re-bake an FBM erosion-style perturbation into every cell that
    // has been raised above the procedural baseline. The result is
    // that smooth dome-shaped sculpt strokes pick up natural-looking
    // ridges + valleys + grain. amplitude_factor controls how strong
    // the perturbation is relative to the local sculpt amount;
    // frequency is the base FBM scale (cycles per metre). Marks all
    // chunks dirty + triggers a delta-texture refresh.
    void add_fbm_erosion_to_sculpted(float amplitude_factor, float frequency);

    // CPU-baked heightmap sun-shadow texture (R8). Sampled by
    // grass.vert per blade so distant grass picks up mountain
    // shadows without firing any RT rays. Re-baked when sun
    // direction changes (sun_pitch/sun_yaw sliders).
    VkImage           terrain_shadow_image_   = VK_NULL_HANDLE;
    VmaAllocation     terrain_shadow_alloc_   = nullptr;
    VkImageView       terrain_shadow_view_    = VK_NULL_HANDLE;
    VkSampler         terrain_shadow_sampler_ = VK_NULL_HANDLE;
    int               terrain_shadow_dim_     = 0;
    glm::vec3         terrain_shadow_sun_dir_ = glm::vec3(0.0f);
    // Async worker-thread bake. The bake CPU work runs entirely off
    // the main thread; main only does (a) sorting + posting the
    // job list when the sun direction changes and (b) draining
    // ready results once per frame and uploading them to the GPU
    // image. Distant tiles arrive with delay (acceptable per spec)
    // but the main FPS is never blocked by bake work.
    struct ShadowBakeTile { int ix, iz, w, h; float dist_sq; };
    struct ShadowBakeJob  { int ix, iz, w, h; int ss; glm::vec3 sun_dir; };
    struct ShadowBakeResult {
        int ix, iz, w, h, ss;
        glm::vec3 sun_dir;
        std::vector<uint8_t> data;
    };
    int terrain_shadow_active_ss_ = 1;  // current texture's supersample factor
    glm::vec3 terrain_shadow_target_sun_dir_ = glm::vec3(0.0f);
    static constexpr int kShadowTileSize           = 64;
    static constexpr int kShadowUploadsPerFrame    = 12;
    // Immutable heightmap snapshot the worker reads — taken once at
    // engine init (after init_world). Sculpt edits don't propagate
    // (acceptable; sculpt is rare and we can rebuild the snapshot
    // explicitly if/when that path matters).
    Heightmap                       terrain_shadow_baker_hm_;
    std::thread                     terrain_shadow_worker_;
    std::mutex                      terrain_shadow_mutex_;
    std::condition_variable         terrain_shadow_cv_;
    std::deque<ShadowBakeJob>       terrain_shadow_jobs_;     // main → worker
    std::deque<ShadowBakeResult>    terrain_shadow_results_;  // worker → main
    std::atomic<bool>               terrain_shadow_stop_{false};
    void rebuild_terrain_shadow_texture();
    void enqueue_terrain_shadow_rebake(glm::vec3 target_sun);
    void tick_terrain_shadow_progressive();
    void destroy_terrain_shadow_texture();
    // Shared upload helper for init_/refresh_terrain_height_texture.
    void upload_terrain_height_payload(VkImageLayout src_layout);
    void start_terrain_shadow_worker();
    void stop_terrain_shadow_worker();
    void terrain_shadow_worker_loop();

    // Sun shadow map (single-cascade, ortho). Replaces the heightmap-bake
    // path for grass and lets dynamic occluders (castle, dyn-props) cast
    // sun shadows on grass. 2048² D32 depth target rendered each frame
    // via shadow_pipeline_ in render_sun_shadow_pass(). Sampled by
    // grass.vert at scene_desc binding 7 as a sampler2DShadow (hardware
    // PCF). Cube.frag continues to use RT shadow rays — the shadow map
    // here is grass-only for now.
    // Allowed resolutions for the rt_.shadow_map_resolution slider. Anything
    // else snaps to the nearest. 4096 is borderline on weak GPUs; the cap
    // keeps a typo from forcing a huge alloc.
    static constexpr int kShadowMapResolutions[] = { 512, 1024, 2048, 4096, 8192 };
    int             sun_shadow_dim_     = 1024;     // current image side
    VkImage         sun_shadow_image_   = VK_NULL_HANDLE;
    VmaAllocation   sun_shadow_alloc_   = nullptr;
    VkImageView     sun_shadow_view_    = VK_NULL_HANDLE;
    VkSampler       sun_shadow_sampler_ = VK_NULL_HANDLE;
    VkPipeline       sun_shadow_pipeline_ = VK_NULL_HANDLE;
    // Dedicated pipeline layout — shadow.vert reads only push constants,
    // not the scene descriptor set. Using the cube pipeline_layout_ here
    // forced render_sun_shadow_pass to bind scene_desc_set (TLAS, every
    // texture array) for nothing.
    VkPipelineLayout sun_shadow_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule   sun_shadow_vert_module_ = VK_NULL_HANDLE;
    glm::mat4       sun_shadow_light_vp_ = glm::mat4(1.0f);
    void init_sun_shadow_resources();
    void destroy_sun_shadow_resources();
    void init_sun_shadow_pipeline();
    void destroy_sun_shadow_pipeline();
    void update_sun_shadow_light_vp();
    void render_sun_shadow_pass(VkCommandBuffer cmd);

    // Phase 4 sculpt state. The brush is driven by the player's center-
    // screen ray; click-and-hold raises/lowers/smooths the heightmap
    // within `terrain_brush_radius` of the hit point. Mutated chunks
    // are rebuilt the next frame; the BLAS + Jolt heightfield are
    // refreshed when the stroke ends (mouse-up) so the per-frame cost
    // stays low.
    // GrassAdd / GrassRemove paint the R channel of grass_mask_ (the
    // eligibility mask sampled by both grass renderers). They reuse the
    // same brush radius / strength / falloff as the height modes; the
    // height array is left alone, so a stroke in either mode never
    // touches collision or geometry.
    // Erode adds the same FBM-with-derivative-erosion detail the
    // raymarched terrain uses, scaled by brush falloff. ErodeSmooth
    // is the inverse: pulls detail back toward the local mean. Both
    // are heightmap modes; the existing one-shot 'Apply FBM erosion
    // to sculpted area' button still exists in the UI.
    enum class TerrainBrushMode {
        Raise, Lower, Smooth, Flatten,
        GrassAdd, GrassRemove,
        Erode, ErodeSmooth
    };
    static bool brush_mode_is_grass(TerrainBrushMode m) {
        return m == TerrainBrushMode::GrassAdd ||
               m == TerrainBrushMode::GrassRemove;
    }
    bool  terrain_edit_mode_     = false;
    bool  terrain_stroke_active_ = false;     // mouse held since stroke begin
    bool  terrain_blas_dirty_    = false;     // rebuild BLAS at next safe point
    bool  terrain_jolt_dirty_    = false;     // rebuild Jolt heightfield
    TerrainBrushMode terrain_brush_mode_ = TerrainBrushMode::Raise;
    float terrain_brush_radius_  = 6.0f;       // metres
    float terrain_brush_strength_ = 8.0f;      // metres/sec at brush centre
    float terrain_brush_flatten_target_ = 22.0f;
    // Multiply per-cell sculpt delta by `1 + noise·strength` so
    // raise/lower areas pick up FBM-like variation instead of
    // looking like flat squares. 0 = clean / artificial, 1 = strong
    // hill-like noise. Frequency is metres-per-cycle of the base
    // octave (≈ 4 m default = visible bumps without aliasing).
    float terrain_brush_noise_strength_ = 0.6f;
    float terrain_brush_noise_freq_     = 0.25f;
    // When true, the per-cell noise multiplier on Raise/Lower uses the
    // same FBM-with-derivative-erosion the raymarched terrain uses
    // (terrain_raymarch.frag::terrainM), so sculpted hills inherit the
    // ridge/valley character of the surrounding procedural terrain.
    // When false, the cheap 3-octave value-noise is used (faster, but
    // looks like generic bumps that don't match the terrain).
    bool  terrain_brush_use_fbm_erosion_ = true;
    int   terrain_brush_fbm_octaves_     = 6;
    glm::vec3 terrain_brush_world_pos_{0.0f}; // last hit point (UI gizmo)
    bool  terrain_brush_has_hit_ = false;
    std::vector<int> terrain_dirty_chunks_;    // indices into terrain_chunks_.chunks
    void apply_terrain_brush(float dt);
    void rebuild_dirty_terrain_chunks();
    void refresh_terrain_blas();
    void refresh_terrain_collision();

    VkBuffer tlas_buffer_ = VK_NULL_HANDLE;
    VmaAllocation tlas_alloc_ = nullptr;
    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    VkDeviceSize tlas_buffer_size_ = 0;
    uint32_t tlas_max_instances_ = 0;

    VkBuffer tlas_scratch_buffer_ = VK_NULL_HANDLE;
    VmaAllocation tlas_scratch_alloc_ = nullptr;
    VkDeviceAddress tlas_scratch_address_ = 0;

    VkBuffer instances_buffer_ = VK_NULL_HANDLE;
    VmaAllocation instances_alloc_ = nullptr;
    void* instances_mapped_ = nullptr;
    VkDeviceAddress instances_buffer_address_ = 0;

    // Per-instance material data (color, emissive). Indexed in the shader by
    // the TLAS instance's custom index. Used by the GI bounce to pick up
    // surface color of the hit point.
    VkBuffer materials_buffer_ = VK_NULL_HANDLE;
    VmaAllocation materials_alloc_ = nullptr;
    void* materials_mapped_ = nullptr;

    // Per-instance previous-frame world matrix (mat4 per slot, indexed by
    // TLAS custom index). Foundation for proper motion vectors / SVGF: a
    // future motion-vector pass reads this to reproject a hit's world point
    // through prev_view_proj and computes screen-space delta vs current uv.
    // Static brushes write current = prev (they don't move; reprojection is
    // camera-only). Dynamic props write DynRender::prev_world. Particles &
    // projectiles write current as prev (zero-motion approximation, fine at
    // their sub-pixel size). Buffer is host-mapped, written every frame.
    VkBuffer prev_transforms_buffer_ = VK_NULL_HANDLE;
    VmaAllocation prev_transforms_alloc_ = nullptr;
    void* prev_transforms_mapped_ = nullptr;

    bool rt_initialized_ = false;
    // TLAS instance count from the previous rebuild. When the new build's
    // count matches, rebuild_tlas() uses MODE_UPDATE instead of MODE_BUILD —
    // big perf win since update is much cheaper than full build. Reset to
    // UINT32_MAX to force a full rebuild next frame (e.g., on resize).
    uint32_t prev_tlas_n_ = UINT32_MAX;
    // Skip the TLAS update entirely when the dynamic instance set hasn't
    // moved since last frame. Hash of dyn-prop transforms + projectile
    // poses; if it matches AND nothing structural changed, the existing
    // tlas_ contents are still valid for cube.frag's reads.
    uint64_t prev_dyn_signature_ = 0;
    bool     tlas_first_build_done_ = false;

    // Per-frame camera state shared by render_terrain_raymarch_lr,
    // render_world_depth_pass and render_world. compute_frame_view() at
    // the top of draw() captures it once; the three render methods read
    // from current_frame_view_ instead of each rebuilding aspect/lerp/
    // view/proj/Halton-jitter independently.
    struct FrameView {
        glm::vec3 render_pos{0.0f};
        // Smoothed eye position (render_pos + eye_offset) — must mirror
        // the lerp used for the view matrix, otherwise raymarched passes
        // (terrain, grass) whose ray origin reads scene.camera_pos drift
        // out of sync with rasterised geometry whose mvp uses fv.vp,
        // showing as one-frame "lag" on stop.
        glm::vec3 eye_pos{0.0f};
        glm::mat4 view{1.0f};
        glm::mat4 proj{1.0f};
        glm::mat4 vp{1.0f};
        glm::mat4 inv_vp{1.0f};
    };
    FrameView current_frame_view_{};
    FrameView compute_frame_view();

    // Static-brush portion of the TLAS instance + material buffer is baked
    // once per (level, textures_enabled) and memcpy'd into slots
    // [0..static_n) every frame instead of being recomputed. Re-baked when
    // textures-toggled (material entries depend on tex_on).
    std::vector<VkAccelerationStructureInstanceKHR> static_brush_instances_;
    std::vector<glm::vec4> static_brush_materials_;  // 3 vec4 per brush
    // Parallel mat4 per brush (the un-transposed model matrix). Used to
    // populate prev_transforms_buffer_'s static slots — static brushes don't
    // move so prev_world == current world; we just memcpy this block.
    std::vector<glm::mat4> static_brush_worlds_;
    bool static_brush_tex_on_ = false;
    bool static_brush_dirty_ = true;
    // Tracks whether the last bake_static_brushes() included the terrain
    // BLAS. Flipped when rt_.terrain_raymarch_enabled toggles → forces a
    // rebake so the TLAS instance set matches the current mode.
    bool terrain_in_tlas_baked_ = false;

    // True when the chunked terrain BLAS should be in the TLAS. Skipped
    // while raymarch terrain is active — the mesh BLAS shape doesn't
    // match the visible FBM surface, so castle/dyn-prop RT shadows would
    // land on the wrong geometry.
    inline bool tlas_includes_terrain_blas() const {
        return terrain_blas_device_address_ != 0 && !rt_.terrain_raymarch_enabled;
    }
    // True when something on screen samples the heightmap-shadow texture
    // (cube.frag's mesh-terrain branch + grass.vert). Raymarch terrain
    // and raymarch grass don't read it.
    inline bool needs_heightmap_shadow() const {
        return !rt_.terrain_raymarch_enabled ||
               (rt_.grass_enabled && !rt_.grass_raymarch_enabled);
    }
    // Set whenever bake_static_brushes() reseeds static_brush_materials_
    // / _worlds_ / _instances_; cleared after the per-frame memcpy
    // copies them into the GPU-visible buffers. Avoids re-uploading
    // ~200 KB of unchanged static data every frame.
    bool static_brush_uploaded_ = false;

    // --- TAA: scene color + history ping-pong ---
    VkImage      scene_color_image_ = VK_NULL_HANDLE;
    VmaAllocation scene_color_alloc_ = nullptr;
    VkImageView  scene_color_view_ = VK_NULL_HANDLE;
    VkFormat     scene_color_format_ = VK_FORMAT_R16G16B16A16_SFLOAT;

    // Per-pixel screen-space motion vector image: prev_uv - current_uv (in
    // [-1, 1] NDC-half-space). Allocated at swapchain res, recreated on
    // resize alongside scene_color. Sentinel value (0, 0) means "no motion
    // info available" — the TAA shader's existing depth-reconstruct path is
    // used as a fallback. Currently allocated but not yet written by
    // cube.frag — see 4a-svgf-finalize task for the remaining shader work.
    VkImage      motion_vec_image_ = VK_NULL_HANDLE;
    VmaAllocation motion_vec_alloc_ = nullptr;
    VkImageView  motion_vec_view_ = VK_NULL_HANDLE;
    VkFormat     motion_vec_format_ = VK_FORMAT_R16G16_SFLOAT;

    static constexpr int kHistorySlots = 2;
    VkImage      history_image_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation history_alloc_[kHistorySlots]{ nullptr, nullptr };
    VkImageView  history_view_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkSampler    linear_sampler_ = VK_NULL_HANDLE;

    VkDescriptorPool       taa_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout  taa_desc_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet        taa_desc_sets_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkBuffer               taa_ubo_buffer_ = VK_NULL_HANDLE;
    VmaAllocation          taa_ubo_alloc_ = nullptr;

    VkPipelineLayout taa_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline       taa_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule   taa_vert_module_ = VK_NULL_HANDLE;
    VkShaderModule   taa_frag_module_ = VK_NULL_HANDLE;

    // --- TAAU (temporal upsample). Optional pass between TAA and compose.
    // Reads LR history + LR motion + LR depth + previous-frame NATIVE
    // upscaled, outputs native upscaled with motion-reproject + variance
    // clamp. Compose then samples taau_view_ instead of history_view_.
    // Sized at swapchain_extent_; ping-pong on history_write_slot_.
    VkImage          taau_image_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation    taau_alloc_[kHistorySlots]{ nullptr, nullptr };
    VkImageView      taau_view_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorPool      taau_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout taau_desc_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet       taau_desc_sets_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkBuffer              taau_ubo_buffer_ = VK_NULL_HANDLE;
    VmaAllocation         taau_ubo_alloc_ = nullptr;
    VkPipelineLayout      taau_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            taau_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule        taau_frag_module_ = VK_NULL_HANDLE;
    bool                  taau_history_valid_ = false;
    // Compose desc set is rewired when this flag toggles — tracks the
    // currently-bound source so we only re-write on actual change.
    bool                  compose_uses_taau_ = false;
    void init_taau();
    void destroy_taau();
    void recreate_taau_targets();
    void rewrite_compose_for_taau();

    // --- Compose / tonemap pass ---
    VkDescriptorPool       compose_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout  compose_desc_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet        compose_desc_sets_[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout       compose_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline             compose_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule         compose_frag_module_ = VK_NULL_HANDLE;

    // --- Bloom mip chain (Karis-style downsample + tent upsample) ---
    static constexpr int kBloomMips = 5;
    VkImage      bloom_image_ = VK_NULL_HANDLE;
    VmaAllocation bloom_alloc_ = nullptr;
    VkImageView  bloom_mip_views_[kBloomMips]{};
    VkExtent2D   bloom_mip_extents_[kBloomMips]{};
    // All-mips view of bloom_image_ used by compose so cube.frag can
    // textureLod() to the smallest mip for an auto-exposure scene-avg
    // proxy. Per-mip views above are still needed by the down/up passes
    // which must rasterise into each mip individually.
    VkImageView  bloom_full_view_ = VK_NULL_HANDLE;

    VkDescriptorPool       bloom_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout  bloom_desc_set_layout_ = VK_NULL_HANDLE;
    // 2 sets per slot per mip: one to read history (for first downsample),
    // (kBloomMips-1) to read mip k-1 for downsample, (kBloomMips-1) for
    // upsample. Pre-built and indexed.
    VkDescriptorSet bloom_down_sets_[kHistorySlots][kBloomMips]{};
    VkDescriptorSet bloom_up_sets_[kBloomMips]{};

    VkPipelineLayout bloom_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline       bloom_down_pipeline_ = VK_NULL_HANDLE;
    VkPipeline       bloom_up_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule   bloom_down_frag_ = VK_NULL_HANDLE;
    VkShaderModule   bloom_up_frag_ = VK_NULL_HANDLE;

    glm::mat4 prev_view_proj_{ 1.0f };
    glm::mat4 last_view_proj_{ 1.0f };
    // Cached glm::inverse(last_view_proj_) — recomputed once per frame
    // when last_view_proj_ is updated and reused by TAA + compose +
    // viewmodel paths. Was being inverted 3-4× per frame on the CPU
    // (~150 cycles each).
    glm::mat4 last_inv_view_proj_{ 1.0f };
    bool      prev_view_proj_valid_ = false;
    int       history_write_slot_ = 0;  // toggles each frame

    // --- GPU stage timing (per-frame timestamp queries) ---
    enum GpuStage {
        kStageTlas = 0,
        kStageScene = 1,
        kStageTaa = 2,
        kStageBlitUi = 3,
        kStageCount = 4,
    };
    static constexpr int kQueriesPerFrame = kStageCount * 2;
    VkQueryPool gpu_query_pool_ = VK_NULL_HANDLE;
    float       gpu_timestamp_period_ns_ = 1.0f;  // ns per tick from device limits
    uint64_t    gpu_query_recorded_mask_ = 0;     // bit i set if frame slot i has data
    struct GpuTimers {
        float tlas_ms = 0.0f;
        float scene_ms = 0.0f;
        float taa_ms = 0.0f;
        float blit_ui_ms = 0.0f;
    } gpu_timers_;
    float gpu_log_timer_ = 0.0f;

    struct RtSettings {
        // Sun position via spherical angles (so a UI slider feels intuitive).
        float sun_pitch_deg = 52.0f;   // elevation above horizon
        float sun_yaw_deg   = 35.0f;   // around Y, 0 = +Z
        float sun_intensity = 2.6f;
        glm::vec3 sun_color  = glm::vec3(1.00f, 0.96f, 0.86f);
        glm::vec3 sky_color  = glm::vec3(0.55f, 0.72f, 0.95f);
        glm::vec3 ground_ambient = glm::vec3(0.18f, 0.20f, 0.24f);
        float ambient_strength = 1.0f;
        float shadow_strength  = 1.0f;
        bool  shadow_enabled = true;
        // 40 stratified samples + slope-scale bias + the TAA/à-trous denoiser
        // gives a cinema-clean soft shadow.
        int   shadow_samples = 40;
        float shadow_softness = 0.016f;
        // PCSS curve: 0 = linear (penumbra ∝ blocker distance, our default),
        //             1 = cubic (sharp on contact, much softer at distance).
        float shadow_curve = 0.5f;
        int   ao_samples = 0;         // 0 = off; 1/4/8/16
        float ao_radius  = 1.5f;

        // Path-traced GI: each "sample" follows a path of up to gi_bounces
        // segments, picking up tinted color along the way.
        int   gi_samples = 64;        // paths per pixel; 0 = off
        int   gi_bounces = 3;         // 1 = single-bounce sky GI; 2-5 = multi-bounce
        // How many bounce levels get a sun shadow ray. 0 = no shadow
        // rays in GI (every shadowed bounce uses sky_fill, fastest);
        // 1 = first bounce only (default, balanced); higher = more
        // accurate but linearly more expensive (each level adds N_gi
        // any_hit rays per pixel).
        int   gi_shadow_max_bounce = 1;
        // AO darkening floor. Raw AO is 0..1 (1 = clear, 0 = fully
        // occluded); we remap to [ao_floor, 1.0] so corners where 2-3
        // edges all block rays don't compound into near-black. 0 = AO
        // can crush surfaces to pure black; 0.5 = AO darkens at most
        // 50%; 1.0 = AO has no effect. 0.30 keeps clear contact AO at
        // simple edges while preventing the "triple-darkness corner"
        // artifact.
        float ao_floor = 0.30f;
        // Auto-exposure: scales pre-tonemap HDR by a target/scene-avg
        // ratio so dark interiors lift while bright outdoor pixels seen
        // through a doorway over-expose into bloom. Strength 0 = off,
        // 1 = full eye-adaptation, ~0.5 = stylised (the default).
        float auto_exposure_strength = 0.5f;
        // Static-castle BLAS merging: true = one BLAS for all 151 brushes
        // (the merged path baked at level load), false = per-brush TLAS
        // instance pointing at the shared cube BLAS (the original path).
        // Toggle exists so we can A/B autodemo to confirm the merge isn't
        // making per-pixel ray cost worse on a given GPU. Leave on by
        // default once verified.
        bool  use_merged_static_blas = true;

        // ---- Terrain shader knobs (cube.frag, raster tex_params.w==2) ----
        // Atmospheric perspective fog: distant terrain blends toward
        // the view-direction sky tint. 0=off, 1=full.
        float terrain_fog_strength = 1.0f;
        // Half-Lambert wrap: 0 = hard Lambert (back-of-ridge can go
        // pure black with shadow), 1 = full wrap (back gets ~half
        // light from the half-Lambert formula).
        float terrain_wrap_strength = 1.0f;
        // Multiplies the triplanar Ground054 detail × layer-base in the
        // terrain shading branch. >1 punches up texture detail; ~1 is
        // neutral; <1 fades the detail and reads more uniform.
        float terrain_detail_strength = 1.6f;
        // Multiplies the global shadow_softness for terrain pixels only.
        // Lower = tighter shadow cone = less PCSS dither at the cost
        // of harder shadow edges. Useful when shadow_samples is modest
        // and the terrain looks noisy.
        float terrain_shadow_softness_scale = 0.5f;
        // Layer transition heights (smoothstep edges). Tweak per-level
        // if height_scale changes; defaults match the 140m mountains.
        float terrain_h_sand_grass_start = 4.0f;
        float terrain_h_sand_grass_end   = 12.0f;
        float terrain_h_grass_dirt_start = 28.0f;
        float terrain_h_grass_dirt_end   = 42.0f;
        float terrain_h_dirt_rock_start  = 58.0f;
        float terrain_h_dirt_rock_end    = 80.0f;
        float terrain_h_rock_snow_start  = 95.0f;
        float terrain_h_rock_snow_end    = 120.0f;

        // ---- Grass ----
        bool  grass_enabled        = true;
        // When true, the rasterised grass pipeline is skipped and the
        // fullscreen raymarched grass pass takes over. Adapted from
        // shadertoy.com/view/dd2cWh — SDF blades on a domain-repeated
        // grid. Heavier per-pixel cost but no instance-buffer rebuild
        // and looks volumetric/3D rather than flat-billboarded.
        bool  grass_raymarch_enabled = false;
        // Max ray distance for the raymarched grass (metres). Cuts the
        // per-pixel step budget; beyond this, grass fades to sky.
        float grass_raymarch_distance = 150.0f;
        // Max blade-render distance in metres. Beyond this, blades
        // collapse to NaN clip space (no fragment work).
        float grass_distance       = 80.0f;
        // Peak per-blade tip wind sway in metres. The sway is per-blade
        // sin/cos so neighbours move out of phase.
        float grass_wind           = 0.04f;
        // Density: fraction of placed blades to actually draw (0..1).
        // Drives instance count in vkCmdDrawIndexed — cheap perf knob.
        float grass_density        = 0.7f;
        // Per-blade Y multiplier (0.3..2.0). 1.0 = built-in 0.55 m blades.
        float grass_height_scale   = 1.0f;
        // Far-distance opacity drop for the raymarched grass.
        //   0.0 = far blades stay fully opaque (alpha = 1)
        //   1.0 = far blades go fully transparent (alpha = 0)
        // Close-up blades (within `grass_cutoff_soft_dist` metres) are
        // always alpha = 1 regardless. Also affects the rasterized
        // grass.frag's side-taper threshold (legacy use of the same
        // slider).
        float grass_alpha_cutoff   = 0.4f;
        // Distance at which the user's `grass_alpha_cutoff` value
        // applies in the raymarched grass shader. Inside this radius
        // the slider value is used as-is (soft-edge dither). Past
        // this, the cutoff ramps linearly to ~0.85 by the time the
        // ray reaches `grass_raymarch_distance`, so far blades
        // discard more aggressively and thin out into the terrain
        // tint instead of reading as a uniform green far-mat.
        float grass_cutoff_soft_dist = 30.0f;
        // Slope cutoff: blades whose stored heightmap-normal Y is
        // below this fade their height to 0 (smoothstep window).
        // 1.0 = only perfectly flat ground; 0.55 = up to ~57° slope.
        // 0.85 (default) ≈ 32° max slope.
        float grass_slope_n_min    = 0.85f;
        // Distance-based density falloff (0 = uniform density, 1 = far
        // grass thinned to ~40% of near density). Implemented per-
        // blade in grass.vert via a deterministic rank vs distance-
        // driven threshold with a wide smoothstep so blades don't
        // flicker as the player walks.
        float grass_distance_density = 0.6f;
        // Altitude band: blades fade to 0 height outside [alt_min,
        // alt_max]. Placement covers a generous superset (-10..200m)
        // so the slider can reshape the band live without rebake.
        float grass_alt_min = -2.0f;
        float grass_alt_max = 35.0f;
        // Grass colour palette (linear RGB). Exposed via UI sliders.
        glm::vec3 grass_color_top         = { 0.40f, 0.55f, 0.18f };  // blade tip
        glm::vec3 grass_color_bottom      = { 0.18f, 0.22f, 0.08f };  // blade base
        glm::vec3 grass_color_ground      = { 0.18f, 0.30f, 0.09f };  // close terrain tint
        glm::vec3 grass_color_ground_far  = { 0.13f, 0.22f, 0.06f };  // far terrain tint
        // Distance (m) at which `grass_color_ground_far` fully takes
        // over from `grass_color_ground`. Near ramp scales as 15 % of
        // this so the transition stays smooth at any far-anchor.
        // Used by both grass_raymarch.frag and terrain_raymarch.frag
        // (delivered via the otherwise-unused grass_color_ground_far.w).
        float grass_ground_tint_far_distance = 200.0f;
        // Fake grass-cast shadows on the raymarched terrain.
        // Fake mask-based grass-cast-shadow on raymarched terrain.
        // Disabled by default: it walks the grass mask along the sun
        // XZ at the terrain pixel and darkens — looks like a flat
        // patch tint, not real per-blade shadow. Slider stays so a
        // user who wants the cheap effect can re-enable it.
        float grass_shadow_strength = 0.0f;    // 0 = off
        int   grass_shadow_samples  = 6;       // 0..8 sample taps along sun XZ
        float grass_shadow_max_dist = 3.0f;    // metres of reach along sun direction
        // Strength of the terrain green tint (0..1).
        float grass_ground_tint_strength = 0.85f;
        // Raymarched-grass base AO floor at the blade base. Lower = darker
        // base / more contrast top-to-bottom; higher = flatter / fuller.
        float grass_base_ao_floor = 0.35f;

        // Sun shadow map (binding 7). Two knobs:
        //   shadow_map_resolution: square texture side. Snapped at apply
        //     time to {512, 1024, 2048, 4096}. Higher = sharper edges,
        //     more VRAM + fragment cost. Image is destroyed and recreated
        //     when changed (requires a vkDeviceWaitIdle).
        //   shadow_map_world_half: half-width of the ortho frustum in
        //     world metres. Bigger = far-off ridges still cast on grass
        //     under your feet, but per-texel coverage gets coarser.
        int   shadow_map_resolution  = 1024;
        float shadow_map_world_half  = 120.0f;
        // Debug overlay: draws a wireframe of the shadow ortho frustum
        // and the bake/shadow-map cross-fade boundary on top of the
        // world. Useful for tuning shadow_map_world_half and seeing
        // exactly which casters land inside the shadow map.
        bool  shadow_debug_overlay = false;

        // Terrain debug visualisation modes (cube.frag branch on
        // scene.grass_extra2.w):
        //   0 = full shading (normal)
        //   1 = simple Lambert only (no RT shadow, AO, GI, slope blend, fog)
        //   2 = visualise per-vertex normal as RGB
        //   3 = visualise screen-space face normal as RGB
        int   terrain_debug_mode = 0;

        // Terrain LOD distance thresholds (metres of camera-XZ-to-chunk
        // centre). A chunk renders at LOD 0 when below `terrain_lod1`,
        // LOD 1 below `terrain_lod2`, LOD 2 below `terrain_lod3`, else
        // LOD 3. Bigger thresholds = sharper distant terrain at higher
        // triangle cost. Defaults preserve the original 80/160/320m
        // behaviour.
        // Default LOD distances bumped much higher so most of the
        // visible mid-far terrain stays at full / half resolution by
        // default — the previous 80/160/320 had everything past 80m
        // looking soft because it was already at stride-2 cells.
        float terrain_lod1 = 200.0f;
        float terrain_lod2 = 500.0f;
        float terrain_lod3 = 1000.0f;
        // Master multiplier on all three LOD distances. 1.0 = defaults
        // above. Crank up to push distant terrain to higher LOD (more
        // triangles, more detail).
        float terrain_lod_scale = 1.0f;
        // Supersample factor for the heightmap shadow bake. 1 = native
        // (1 texel per 1m heightmap cell). 2 = 4x more texels (sub-
        // metre shadow precision). 4 = 16x more texels (sharpest
        // medium/far shadows). Memory: 1 = ~4MB, 2 = ~17MB, 4 = ~67MB.
        // Bake time scales linearly; worker thread handles it.
        int terrain_bake_supersample = 1;
        // Power exponent on n_dot_l for terrain shading. 1 = standard
        // Lambert (gradient is the natural cosine falloff). >1 sharpens
        // the lit→shadow transition (grazing slopes go darker faster);
        // <1 softens. Visible difference between e.g. 2.0 and 4.0 on
        // mountain slopes.
        float terrain_shading_contrast = 1.0f;
        // Heightmap resolution multiplier (per-axis). 1 = baseline
        // 2048×2048 cells at 1m. 2 = 4096×4096 at 0.5m (16× more
        // samples total, ~600MB extra mesh memory). 4 = 8192×8192
        // at 0.25m (256× more samples, NOT recommended — pushes past
        // ~2GB of GPU memory on the chunk meshes alone). Applied at
        // level load — change requires a restart.
        int terrain_heightmap_scale = 1;
        // Selectable terrain renderer. false = chunked rasterised mesh
        // (default, matches the gameplay heightmap). true = fullscreen
        // procedural FBM ray-marched terrain (terrain_raymarch.frag,
        // visual-only — physics still uses the rasterised heightmap so
        // castle / cubes won't sit on the procedural surface). The
        // shader is toggled at draw time, no rebuild needed.
        bool terrain_raymarch_enabled = false;
        // Raymarch FPS knobs. Larger = better quality, slower. Defaults
        // are the values the reference document recommends; on weaker
        // GPUs drop both to ~120 / ~32 for a meaningful FPS bump.
        int  terrain_raymarch_max_steps    = 200;   // 60–300
        int  terrain_raymarch_shadow_steps = 64;    // 16–96
        // FBM octaves used by the per-march sampling (terrainM). 9 is
        // the document default; 5–6 is noticeably faster and visually
        // close because each higher octave adds half the previous
        // amplitude (last 3 of 9 contribute ~1.5 % of total).
        int  terrain_raymarch_octaves        = 9;   // 4–24
        // FBM octaves for normal sampling (terrainH). Called 3× per
        // hit pixel for finite-difference normals — biggest single
        // perf knob. 16 is the doc default; 8 is barely-distinguishable.
        int  terrain_raymarch_normal_octaves = 18;  // 4–32
        // Step-factor: ratio of the height-field gap consumed per
        // march iteration. 0.4 is conservative (no overshoot on sharp
        // ridges). 0.6–0.8 finishes the same march in ~half the
        // iterations at the cost of a tiny chance of skipping over a
        // thin spike.
        float terrain_raymarch_step_factor   = 0.4f;
        // Distance-LOD knobs. At ray distance < lod_near_m the FBM uses
        // the full octave count (terrain_raymarch_octaves for march,
        // terrain_raymarch_normal_octaves for normals). Past lod_far_m
        // it falls to lod_min_octaves. Smoothstep ramp in between. Saves
        // big at typical horizon views — sub-pixel FBM detail past
        // ~500 m is below Nyquist anyway.
        float terrain_raymarch_lod_near_m    = 120.0f;  // 20..400
        float terrain_raymarch_lod_far_m     = 600.0f;  // 200..2000
        int   terrain_raymarch_lod_min_octaves = 3;     // 2..8
        // Render-scale for the procedural raymarch ONLY. 1.0 = native
        // (current behaviour). 0.5 = half-res raymarch + bilinear
        // upscale composite (~4× fewer FBM evaluations per frame).
        // Cubes / castle / dyn-props always rasterise at native res.
        float terrain_raymarch_scale         = 1.0f;
        // Post-upscale sharpen strength applied during the compose
        // pass. Cheap CAS-style 4-neighbour adaptive sharpen — only
        // boosts contrast around real edges, ignores flat / noisy
        // regions. 0 = off, 0.5 = subtle, 1.0 = aggressive.
        float terrain_raymarch_sharpen       = 0.6f;
        // Volumetric ground fog scale (0 = off, 1 = doc baseline,
        // 2 = thick valley fog). Multiplies both density and the
        // sun-scatter contribution so the slider behaves linearly:
        // half = half as visible at every distance.
        float terrain_raymarch_fog_strength  = 0.6f;
        // Phase 6 — relaxation cone-stepping. When true, march step
        // grows with distance (`step *= max(t*0.02, 1.0)`) so we
        // cover the same range in ~half the iterations at the cost
        // of a tiny chance of skipping thin spikes. Doc-recommended
        // for distance-heavy scenes.
        bool  terrain_raymarch_relaxation    = false;
        // Phase 7 — volumetric fog self-shadow. When true, each fog
        // step fires N tiny rays toward the sun + accumulates optical
        // depth so terrain / castle shadow the fog itself, giving a
        // god-ray look. Cheap (4 short rays per fog step), but multi
        // ALU + branches so off by default.
        bool  terrain_raymarch_fog_godrays   = false;
        // Fog band — density is non-zero between [fog_y_start, fog_y_top]
        // and falls off above/below with a soft transition. Combined
        // with a low-frequency noise so the layer isn't a perfect
        // sheet (real ground fog is wispy).
        float terrain_raymarch_fog_y_start   = 0.0f;
        float terrain_raymarch_fog_y_top     = 18.0f;
        float terrain_raymarch_fog_noise     = 0.7f;   // 0 = uniform sheet, 1 = strong wisps

        // Ocean / water plane. Rendered inside the terrain raymarch
        // (fullscreen tri) so it composites with rasterised geometry
        // via gl_FragDepth — same as the terrain itself.
        bool      water_enabled        = false;
        float     water_level          = 2.0f;     // world Y of the calm sea level
        float     water_wave_strength  = 0.06f;    // amplitude of normal perturbation
        // Wave frequency multiplier — multiplies the base scrolling
        // sin/cos frequencies. 1.0 = baseline (~30 m wavelengths).
        // 2.0 = half the wavelength (more, smaller ripples). 0.5 =
        // double the wavelength (bigger sweeping waves).
        float     water_wave_scale     = 1.0f;
        glm::vec3 water_color          = glm::vec3(0.04f, 0.18f, 0.22f);
        // RT reflections on water: when true, the water surface
        // ray-marches the FBM terrain along the reflection vector
        // and uses the shaded hit point (lambert + sky tint) as the
        // reflected colour. When false, a cheap horizon→zenith sky
        // gradient is used instead.
        bool      water_rt_reflections = true;
        // RT TLAS reflection: when true, additionally fire a ray
        // query into the TLAS along the reflect direction so cubes /
        // castle / dyn-props show up in the water surface (not just
        // the FBM terrain). Cheap (1 closest_hit ray per pixel).
        bool      water_tlas_reflections = true;
        // Shore tinting — colour the water lighter where it's shallow.
        // Depth = water_level − terrain_height_at(xz) (in metres).
        glm::vec3 water_color_shallow  = glm::vec3(0.45f, 0.62f, 0.65f);
        // How many metres of water depth the shallow→deep blend
        // covers. 4 m = noticeable beach band; 0.5 m = razor-sharp.
        float     water_shore_blend    = 4.0f;
        // Noise on the shore band so the line isn't a perfect contour.
        // 0 = clean, 1 = strongly broken.
        float     water_shore_noise    = 0.4f;
        // Shore-foam highlight band: a thin lighter tint right at the
        // water/land boundary. Fades exponentially with the noise-FREE
        // depth so the line follows the shoreline geometry exactly.
        // strength 0 = disabled.
        glm::vec3 water_foam_color     = glm::vec3(0.88f, 0.94f, 0.96f);
        float     water_foam_strength  = 0.55f;
        float     water_foam_width     = 0.6f;     // metres of depth
        // Shoreline grass tint — blades within `grass_shore_distance`
        // metres above water level fade toward `grass_shore_color`.
        // Strength 0 disables. Cheap (~13 ALU/pixel in
        // grass_raymarch.frag, no extra texture taps).
        glm::vec3 grass_shore_color    = glm::vec3(0.40f, 0.30f, 0.16f);
        float     grass_shore_strength = 0.65f;
        float     grass_shore_distance = 3.5f;     // metres above water
        // Shoreline TERRAIN tint — same shape, applied to bare ground
        // in terrain_raymarch.frag's getMaterial(). Mixed on top of the
        // existing snow/sand/grass tint pass so the beach reads as a
        // user-tunable colour band instead of relying on slope alone.
        glm::vec3 terrain_shore_color    = glm::vec3(0.55f, 0.46f, 0.30f);
        float     terrain_shore_strength = 0.55f;
        float     terrain_shore_distance = 3.0f;
        // Distance fog — standard exp² atmospheric fog. Disabled by
        // default (strength = 0). Applied at end of cube.frag /
        // terrain_raymarch.frag / grass_raymarch.frag shading.
        glm::vec3 distance_fog_color    = glm::vec3(0.62f, 0.70f, 0.78f);
        float     distance_fog_strength = 0.0f;     // 0 = off
        float     distance_fog_density  = 0.005f;   // per metre (exp² falloff)
        float     distance_fog_start    = 50.0f;    // metres before fog kicks in
        float     distance_fog_height   = 80.0f;    // height-falloff top (0 = uniform fog)
        float     distance_fog_max      = 0.95f;    // clamp on mix weight (1.0 = pure fog at infinity)
        // GENERAL terrain shore tint — applied to BARE terrain near
        // water (no grass). Companion to terrain_shore_* (grass-area).
        glm::vec3 terrain_shore_general_color    = glm::vec3(0.50f, 0.42f, 0.30f);
        float     terrain_shore_general_strength = 0.55f;
        float     terrain_shore_general_distance = 3.0f;
        // Bare-shore SAND base colour — the underlying material that
        // the slope-driven beachMask reveals near water level on the
        // raymarched terrain. Was hardcoded; now slider-tunable.
        glm::vec3 terrain_sand_color = glm::vec3(0.50f, 0.45f, 0.35f);
        // Shadow casting on water: when true, the water surface
        // dims its specular and base tint where the sun shadow map
        // says it's occluded by terrain / castle / dyn-props.
        bool      water_shadows_enabled = true;
        // 0 = fully opaque (water_color is the colour you see), 1 =
        // shallow water shows the underwater terrain through it
        // (linearly attenuated by depth so deep water still hides).
        float     water_transparency    = 0.6f;
        // Multiplier on the RT shadow ray count for fragments within
        // rt_lod.x (close to the camera). 1 = unchanged, higher = more
        // rays close-by for smoother penumbras under boxes / castle.
        // Costs extra rays only on the closest pixels; distance LOD
        // continues to drop samples toward 1 ray past rt_lod.y.
        float shadow_near_mult = 1.0f;

        float gi_strength = 1.0f;
        float gi_radius   = 60.0f;
        // Spatial smoothing factor on the raymarched-terrain sky-bounce GI.
        // 0 = raw per-pixel GI from the sample loop (noisiest, most accurate).
        // 1 = collapse the GI estimate to the smooth sky-at-normal sample
        //     (no per-pixel noise but loses occluder-aware GI variation).
        // Useful when gi_samples is too low to converge on its own.
        float gi_softener = 0.5f;
        // Debug viz: when true, the raymarched-terrain shader replaces its
        // final colour with the gi_indirect term only (scaled up so it's
        // clearly visible). Lets you confirm the path-traced GI loop is
        // firing and producing meaningful contribution at this pixel.
        bool  gi_debug_viz = false;
        // Power curve on the raymarched-terrain AO. 1.0 = linear (current
        // behaviour). > 1 punches corners darker (pow makes mid AO drop
        // faster than linear). < 1 flattens. Modulates BOTH the analytical
        // ambient term AND the GI smooth fallback so cavities go genuinely
        // dark without depending solely on the ao_floor knob.
        float terrain_ao_punch = 1.5f;
        // Per-pixel RT caps for the raymarched terrain shader. Lower
        // them when the GPU TDRs; raise for visual fidelity. Defaults
        // chosen to keep total per-frame RT work under the Windows TDR
        // threshold even with the heaviest scene (kMaxDynProps=60).
        int   terrain_pcss_samples_cap   = 4;     // max PCSS rays per pixel
        int   terrain_gi_samples_cap     = 4;     // max GI primary rays per pixel
        int   terrain_gi_bounces_cap     = 1;     // max GI bounces per ray
        float terrain_ao_final_strength  = 0.45f; // 0=no final-colour AO, 1=full
        // Specular reflection on flagged surfaces (the pedestal).
        bool  reflections_enabled = true;
        float reflection_strength = 0.5f;

        // Temporal + spatial denoiser.
        float taa_history_blend = 0.95f;  // 0 = no temporal, 1 = full history
        // Temporal upsample (TAAU). When true, a second temporal pass runs
        // after TAA: samples LR history bilinearly, reprojects a native-
        // resolution prev frame, neighborhood-clamps to kill ghosts.
        // Compose then samples the native upscaled image. Off by default
        // so the standard render-scale=1 path stays bit-identical.
        bool  taau_enabled       = false;
        float taa_spatial_strength = 0.7f;  // 0 = no à-trous, 1 = fully filtered

        // Bloom (single-pass spiral-tap inside the compose shader).
        bool  bloom_enabled = true;
        float bloom_strength = 0.6f;
        float bloom_threshold = 1.05f;  // luminance above which a pixel "blooms"
        float bloom_radius = 24.0f;     // largest tap offset in pixels

        // Spark-trail emissive multiplier — applied per-trail-segment in
        // draw_spark_trails. The sparks already feed bloom because they're
        // drawn as emissive cylinders; cranking this scales their HDR
        // contribution which directly drives a bigger bloom halo. Direct
        // brightness clips to ~white under ACES, so the visible difference
        // is mostly in the bloom halo width/intensity. 1.0 = current look.
        float spark_bloom = 1.0f;       // 0..5 sane range

        // Screen-space lens flare (Chapman 2017): reflects bright sources
        // around the screen center, samples ghosts along the light axis,
        // adds a halo ring + chromatic aberration. Runs in the compose pass.
        bool  lens_flare_enabled    = true;
        float lens_flare_strength   = 0.5f;
        float lens_flare_threshold  = 1.4f;   // HDR luminance cutoff
        float lens_flare_dispersal  = 0.30f;  // ghost spacing along axis
        float lens_flare_halo_width = 0.40f;  // halo offset toward center
        int   lens_flare_ghosts     = 3;      // 1..8 (3 = cinematic default)
        float lens_flare_aberration = 0.005f; // chromatic-fringe offset

        // Master textures on/off. When disabled, all texture lookups in the
        // raster + RT paths fall back to per-instance solid colors. Useful
        // for comparing texture cost vs. baseline and for the "looks like
        // the prototype" aesthetic.
        bool  textures_enabled = true;

        // Sub-pixel Halton jitter on the projection matrix → TAA accumulates
        // sub-pixel samples → supersampled/SSAA-quality edges over ~16 frames.
        // Default OFF: without per-pixel motion vectors, jitter inherently
        // causes mild edge-flicker that variance clipping can only mostly
        // hide. Toggle on if you want the smoother-edge supersampled look
        // and accept some shimmer in exchange.
        bool  taa_jitter_enabled = false;
        float taa_jitter_strength = 0.5f;

        // Post-TAA unsharp mask in compose.frag. Counteracts the soft look
        // taa.frag's à-trous spatial filter introduces. 0 disables; 0.55 is
        // a reasonable default that recovers most lost detail without
        // ringing on edges. >1 looks deliberately punchy / over-sharpened.
        float compose_sharpen_strength = 0.55f;
        // Final-image contrast applied in compose.frag right after the
        // ACES tonemap, before sRGB encode. 1.0 = neutral, > 1 punches
        // up midtones, < 1 flattens them. Sane range 0.5..2.0.
        float image_contrast = 1.0f;
        // Final-image brightness multiplier (post-tonemap, pre-sRGB).
        // 1.0 = neutral; range 0.3..2.5.
        float image_brightness = 1.0f;
        // Output gamma curve (post-tonemap, pre-sRGB encode). 1.0 =
        // pass-through. > 1 lifts midtones (lighter look); < 1
        // darkens. Range 0.4..2.5.
        float image_gamma = 1.0f;
        // Raymarched-terrain RT (shadows / AO / GI) cutoff distance.
        // Effect fades smoothly from `* 0.6` to this value, then turns
        // off entirely past it. 250 m default keeps gameplay-range
        // pixels fully RT-shaded; lower if you need perf headroom.
        float terrain_rt_lod_distance = 250.0f;

        // Render resolution scale: 0.5 = render at half-res then upscale,
        // 1.0 = native, 2.0 = render at 2× then downscale (SSAA). The scene
        // / TAA / bloom passes all run at this resolution; only the final
        // compose targets the swapchain. Single biggest perf knob in the
        // engine — fragment-shader cost (which dominates here) scales with
        // pixel count, so 0.75 ≈ 1.78× faster, 1.5 ≈ 2.25× slower.
        // Reapplying requires re-creating render targets (recreate_swapchain
        // does that automatically; the UI's Apply button forces it).
        float render_scale = 1.0f;

        // Quality preset selector. -1 = custom (sliders won the last edit);
        // 0..3 = Low / Medium / High / Ultra. Switching to a non-custom
        // preset writes a bundle of values onto the other RtSettings fields
        // (shadow_samples, gi_samples, render_scale, ...) so users don't
        // have to tune individually. See apply_quality_preset().
        int quality_preset = 2;  // High — matches existing defaults

        // AO mode. 0 = off, 1 = GTAO (screen-space, fast), 2 = RTAO (true
        // ray-traced, slower but handles off-screen occluders). GTAO is the
        // production default; RTAO is the "high-quality" preset choice.
        int ao_mode = 1;

        // Member-wise equality, ignoring padding bytes — required because
        // memcmp on this struct caught uninitialised padding as "changes" and
        // triggered spurious auto-saves every frame.
        bool operator==(const RtSettings&) const = default;
    } rt_;

    // Gameplay-tuning settings. Kept outside RtSettings so the menu can put
    // them in a separate "Game settings" section instead of in graphics.
    struct GameSettings {
        // Gravity in m/s² (downward). Default = 25 (~2.5 G, game-feel).
        // Slider in the menu goes up to 50× that for absurd-fall experiments.
        float gravity = 25.0f;
        // Continuous box spawner rate. Engine converts to seconds-per-spawn.
        int   cubes_per_minute = 150;
        // Bullet mass in kg. Higher = more impulse delivered to boxes on hit.
        // 10 kg @ 220 m/s = 2200 kg·m/s, ~17 m/s push on a 125 kg box.
        float bullet_mass = 10.0f;
        // Auto-fire rate while LMB is held, in rounds per second. 0 = single
        // shot per click (legacy hitscan behaviour).
        float fire_rate_rps = 6.0f;
        // Bullet exit velocity in m/s. Higher = more impulse delivered on
        // impact (since projectile mass × velocity = momentum), but Jolt's
        // LinearCast CCD has practical limits — anything above ~600 m/s with
        // 1cm-class targets starts to behave weirdly.
        float bullet_speed = 220.0f;
        // Scales spark count + initial speed at spawn. 1.0 = default. Use
        // smaller values for subtle effects, larger for explosions.
        float spark_scale   = 1.0f;

        bool operator==(const GameSettings&) const = default;
    } game_;
    GameSettings game_last_saved_{};
    GameSettings game_prev_frame_{};

    // Auto-save state.
    //   rt_last_saved_  = snapshot of rt_ at last successful save_settings().
    //   rt_prev_frame_  = snapshot of rt_ at last frame's auto-save check.
    // The debounce uses rt_prev_frame_ to detect "user is *currently* dragging
    // a slider" (timer reset), and rt_last_saved_ to detect "needs to be
    // flushed" (timer counts up). Keeping both stops the timer from resetting
    // forever just because the live struct still differs from the saved one.
    RtSettings rt_last_saved_{};
    RtSettings rt_prev_frame_{};
    float      settings_dirty_timer_ = 0.0f;

    bool initialized_ = false;
    // Set when run() catches a DEVICE_LOST-class exception. shutdown() skips
    // vkDeviceWaitIdle in that case — waiting on a hung device just hangs the
    // shutdown thread and adds 2-3s to process exit.
    bool device_lost_ = false;
    // Most-recent TLAS instance count from rebuild_tlas. Surfaced in the
    // per-second telemetry log so we can correlate device-lost events with
    // RT scene size.
    uint32_t last_tlas_n_ = 0;
    bool resize_requested_ = false;

    RunOptions opts_{};
    bool screenshot_taken_ = false;
    // F12 in-game capture: input handler stores a generated path here,
    // the next render frame writes a PPM to it and clears.
    std::string pending_screenshot_path_;
    int  ingame_screenshot_counter_ = 0;
};

} // namespace qlike
