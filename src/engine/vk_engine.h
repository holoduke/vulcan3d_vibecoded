#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/frustum.h"
#include "engine/mesh.h"
#include "engine/physics_world.h"
#include "engine/skybox.h"
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
    void init_readback_buffer();
    void destroy_readback_buffer();
    void init_pipeline();
    void destroy_pipeline();
    void init_world();
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
    // Depth-only pre-pass over the same brush + dyn-prop geometry that
    // render_world() draws — populates depth_image_ so the subsequent color
    // pass's depth_compare=LESS_OR_EQUAL early-rejects occluded fragments
    // before cube.frag's heavy inline-RT body runs. Particles, projectiles
    // and the viewmodel skip the prepass (sparse / screen-space).
    void render_world_depth_pass(VkCommandBuffer cmd);
    void build_hud_ui();
    void load_settings();
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
    VkShaderModule depth_frag_module_ = VK_NULL_HANDLE;

    Mesh cube_mesh_{};
    Mesh cylinder_mesh_{};
    Skybox skybox_{};

    // PBR-ish material textures. kTextureCount must match the size of the
    // sampler arrays in cube.frag.
    static constexpr int kTextureCount = 5;
    struct TextureSlot {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        VkImageView view = VK_NULL_HANDLE;
    };
    TextureSlot albedo_textures_[kTextureCount]{};
    TextureSlot normal_textures_[kTextureCount]{};
    VkSampler   texture_sampler_ = VK_NULL_HANDLE;
    // Texture index assigned to the four spawnable box variants (so dynamic
    // boxes pick from a wood/metal/painted/brick palette at spawn time).
    static constexpr int kBoxTextureChoices = 4;

    game::Level world_;
    game::Player player_{};
    std::unique_ptr<PhysicsWorld> physics_;

    struct DynamicProp {
        uint32_t body_id;
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
    static constexpr int   kParticlesPerHit   = 14;
    static constexpr float kParticleTtl       = 0.6f;
    static constexpr float kParticleColRad    = 0.018f;  // collision sphere
    static constexpr float kParticleVisRad    = 0.008f;  // visual cylinder radius
    static constexpr float kParticleVisBase   = 0.025f;  // visual half-length @ 0 m/s
    static constexpr float kParticleStretch   = 0.06f;   // m of streak per (m/s)
    static constexpr float kParticleMaxStretchMul = 4.0f;
    static constexpr int   kMaxParticles      = 384;
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

    static constexpr int   kMaxDynProps    = 200;
    static constexpr float kSpawnInterval  = 0.4f;  // seconds between spawns
    float spawn_timer_ = 0.0f;
    uint32_t spawn_rng_state_ = 0xc0ffee42u;
    void spawn_random_box();

    // Reused buffer of "static brush AABBs + per-tick dynamic-box AABBs" so
    // the player can't walk through Jolt-driven boxes.
    std::vector<collision::AABB> tick_aabbs_;
    // Per-dyn-prop AABB cache. Sleeping bodies reuse their last-computed
    // AABB instead of re-running the 8-corner mat-vec each physics tick.
    // At 200 dyn props × 6 ticks/frame × 8 corners that was 9600 mat-vec
    // per frame; now most of those are skipped once the bodies settle.
    // Indexing matches dyn_props_; resized in rebuild_tick_aabbs.
    std::vector<collision::AABB> dyn_tick_aabb_cache_;
    std::vector<bool>            dyn_tick_aabb_valid_;
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
        float gi_strength = 1.0f;
        float gi_radius   = 60.0f;
        // Specular reflection on flagged surfaces (the pedestal).
        bool  reflections_enabled = true;
        float reflection_strength = 0.5f;

        // Temporal + spatial denoiser.
        float taa_history_blend = 0.95f;  // 0 = no temporal, 1 = full history
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
};

} // namespace qlike
