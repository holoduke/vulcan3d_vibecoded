// VulkanEngine implementation.
//
// File map (search "// === Section: " to jump):
//
//   === Section: setup       — instance, device, swapchain, depth, sync
//   === Section: descriptors — scene UBO + materials + texture sampler arrays
//   === Section: rt          — BLAS/TLAS init + per-frame TLAS rebuild
//   === Section: pipeline    — main raster pipeline (cube/cylinder mesh draws)
//   === Section: taa+compose — TAA pingpong, compose pass, bloom mip chain
//   === Section: viewmodel   — procedural / glTF-loaded weapon viewmodel
//   === Section: combat      — projectiles + spark particles + trails
//   === Section: world+ui    — level setup, render_world, ui hooks
//   === Section: settings    — load/save/autosave qlike_settings.cfg
//   === Section: textures    — albedo + normal map upload (calls into
//                              engine/texture.cpp for the per-image work)
//   === Section: skybox      — engine/skybox.cpp wrapper + sun-sync
//   === Section: shutdown    — destroy_* in dependency order
//
// Sub-systems with their own translation units:
//   engine/gltf_loader.cpp — glTF 2.0 → MeshData (used by viewmodel)
//   engine/skybox.cpp      — equirect HDRI/JPG → VkImage + sun extraction
//   engine/texture.cpp     — generic 2D texture upload (used by init_textures)
//   engine/physics_world.cpp — Jolt wrapper (rigid bodies + raycast)
//   engine/mesh.cpp        — cube/cylinder/glTF mesh data → vertex/index VkBuffer
//
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
// Pulled in here (not in vk_engine.h) so the unique_ptr<voxel::VoxelWorld>
// member's destructor can be generated in this TU where the type is
// complete. Forward-declared in the header to keep its include surface tight.
// Must be at file scope (NOT inside namespace qlike) so voxel_world.h's
// `namespace qlike::voxel` resolves to the same fully-qualified name as
// the forward decl in vk_engine.h.
#include "engine/voxel/voxel_world.h"

#include "engine/audio.h"

#include "engine/frustum.h"
#include "engine/gltf_loader.h"
#include "engine/log.h"
#include "engine/texture.h"
#include "engine/window.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <cstdlib>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

// stb_image declarations only — implementation owned by skybox.cpp's TU.
#include <stb_image.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>

#ifndef QLIKE_SHADER_DIR
#define QLIKE_SHADER_DIR "shaders"
#endif

namespace qlike {

// PushConstants, SceneUBO, debug_callback, spark_blackbody, align_local_y_to,
// recoil_kick, xorshift32 / frand / frand_range, buffer_device_address,
// as_device_address, write_scene_descriptors_once, the RT extension function
// pointers (g_rt) and load_rt_functions all live in
// src/engine/vk_engine/helpers.cpp; declared in
// src/engine/vk_engine/internal.h which we include above.

// Frustum + extract_frustum + aabb_visible live in engine/frustum.h.

VulkanEngine::VulkanEngine() = default;
VulkanEngine::~VulkanEngine() { shutdown(); }

void VulkanEngine::init() {
    log::info("VulkanEngine::init()");
    window_ = std::make_unique<Window>(WindowConfig{ .width = 1280, .height = 720,
                                                     .title = "quake-like" });
    // Audio engine starts before Vulkan so a missing audio device just
    // means silence; the renderer keeps working. Clip files are loaded
    // by name from assets/sounds/; missing files are ignored.
    audio_ = std::make_unique<AudioEngine>();
    audio_->load_clip(ClipID::Shot,   "assets/sounds/shot.ogg");
    audio_->load_clip(ClipID::Impact, "assets/sounds/impact.ogg");
    audio_->load_clip(ClipID::Jump,   "assets/sounds/jump.ogg");
    // Single shot cut from laser_gun_#4 (0.045-0.225 s, 100 ms fade, -3.5 dB
    // to sit at the old shot level).
    audio_->load_clip(ClipID::Laser,  "assets/sounds/laser_shot.wav");
    init_vulkan();
    init_pipeline_cache();
    init_swapchain();
    init_depth_image();
    init_commands();
    init_sync();
    init_gpu_query_pool();
    // FSR2 needs the swapchain (displaySize) before its context can be
    // created; the per-frame dispatch path (Phase 4) is still gated on
    // rt_.fsr2_enabled, so the toggle remains the sole runtime switch.
    init_fsr2();
    init_dlss();   // Streamline DLSS output (no-op without SDK / unsupported GPU)
    // FSR3 init is lazy (only fires when the user picks fsr_backend=1
    // in the UI). The ffx-api SDK's session-2 status was crash-on-
    // create at startup; deferring keeps the engine launchable even
    // if the SDK's create path AVs in the user's environment.
    // Loader is presentable from here on. Each subsequent step is the
    // long-running ones (init_world / init_rt / shadow bake) that the
    // user used to stare at a blank window for.
    present_loader_frame("Building world",        0.05f);
    init_readback_buffer();
    // load_settings() runs AFTER init_world (it restores player pose
    // that init_world otherwise overwrites). But init_world needs to
    // know whether to source the heightmap from the shader-matching
    // FBM (when raymarch is enabled) or the default FastNoiseLite path.
    // Peek just that one flag here.
    preload_terrain_raymarch_flag();
    // Kick off JPG decodes for the world texture set BEFORE init_world
    // starts the multi-second terrain mesh + chunk upload. The decodes
    // run on worker threads while the main thread is busy with terrain;
    // by the time init_textures() needs them they're (mostly) already
    // available and only the serial graphics-queue uploads remain.
    start_world_texture_decodes();
    init_world();
    present_loader_frame("Loading skybox",        0.30f);
    texture_upload_batch_begin();
    init_skybox();
    init_textures();
    texture_upload_batch_flush(device_, allocator_, graphics_queue_);
    present_loader_frame("Loading settings",      0.40f);
    load_settings();
    // load_settings reads rt_.render_scale from disk, but init_swapchain
    // already ran with the default scale of 1.0 — render_extent_ is wrong
    // until something forces a recreate. If the saved scale differs from
    // 1.0, apply it now so the user's preferred scale survives a restart
    // without a manual Apply click.
    if (std::abs(rt_.render_scale - 1.0f) > 1e-3f) {
        apply_render_scale();
    }
    // FSR2 mip-bias depends on the loaded fsr2_enabled flag + the actual
    // render_extent_, both of which are only known after the two calls
    // above. The recreated sampler is what write_scene_descriptors_once
    // below captures — without this the FSR2 path runs with bias 0.
    update_texture_sampler_for_fsr2();
    init_descriptors();
    present_loader_frame("Building ray tracing",  0.50f);
    init_rt();
    {
        VkImageView alb[kTextureCount]{}, nrm[kTextureCount]{};
        for (int i = 0; i < kTextureCount; ++i) {
            alb[i] = albedo_textures_[i].view;
            nrm[i] = normal_textures_[i].view;
        }
        VkImageView spom[kSpomMaterialCount]{};
        for (int i = 0; i < kSpomMaterialCount; ++i) {
            spom[i] = spom_height_textures_[i].view;
        }
        // Bake grass mask + fog wisp BEFORE descriptor write so the
        // binding-13/14 image views are real (not the slot-0 albedo
        // fallback). Both are CPU-baked, ~1 MB total, runs once.
        init_grass_mask_texture();
        init_terrain_aux_texture();
        init_fog_wisp_texture();
        // ReSTIR reservoir SSBOs at scene_desc bindings 15/16. Allocated
        // here so the descriptor write below captures live handles.
        init_restir();
        write_scene_descriptors_once(device_, scene_desc_set_,
                                     scene_ubo_buffer_, tlas_, materials_buffer_,
                                     prev_transforms_buffer_,
                                     alb, nrm, kTextureCount, texture_sampler_,
                                     spom, kSpomMaterialCount,
                                     grass_mask_.view,
                                     fog_wisp_.view,
                                     // Session 3 aliases both bindings to buf[0]
                                     // (see restir.cpp::init_restir). When session
                                     // 5 ships proper FIF descriptor sets, restore
                                     // buf[1 - write_slot] / buf[write_slot] here.
                                     reservoir_buf_[0],
                                     reservoir_buf_[0]);
        write_depth_copy_binding();   // binding 27 (prepass depth copy)
    }
    present_loader_frame("Compiling pipelines",   0.70f);
    init_pipeline();
    init_voxel();
    init_doors();
    init_deferred_lighting();
    init_terrain_pipelines();
    init_terrain_raymarch_pipeline();
    init_terrain_water_pipeline();
    init_terrain_raymarch_compose_pipeline();
    init_terrain_raymarch_lowres();
    init_shadow_lr();
    // SVGF GI denoiser storage image (Session 1 — see docs/svgf_plan.md).
    // cube.frag writes raw irradiance to it via imageStore. No
    // consumer in session 1 — just allocates + binds.
    init_svgf_targets();
    // VRS attachment is sized in LR-tile units, so it must come after
    // init_terrain_raymarch_lowres() has set tr_lr_extent_.
    init_vrs_attachment();
    init_sun_shadow_pipeline();
    init_grass_pipeline();
    init_scatter_pipeline();
    present_loader_frame("Baking shadows",        0.85f);
    // Heightmap raw-heights texture (binding 8) for the procedural
    // raymarched terrain shader. Uploaded once, after descriptors and
    // init_world have populated terrain_data_. Cheap (~16 MB at 2048²
    // R32F).
    init_terrain_height_texture();
    init_terrain_height_full_texture();
    // Heightmap shadow texture must be baked AFTER descriptors and the
    // texture itself can be created; depends on terrain_data_ which
    // init_world fills. Done here so the descriptor write at the end
    // of rebuild_terrain_shadow_texture lands on the live scene set.
    rebuild_terrain_shadow_texture(/*defer_bake=*/true);
    // Async worker thread bakes the heightmap shadow off the main
    // thread. Started after the initial full bake so the texture is
    // valid before grass renders.
    start_terrain_shadow_worker();
    enqueue_terrain_shadow_rebake(terrain_shadow_sun_dir_);   // stream the real bake
    // Sun shadow map (binding 7). Created after descriptors so the
    // one-shot binding-7 write at the end of init_sun_shadow_resources
    // lands on the live scene set.
    init_sun_shadow_resources();
    present_loader_frame("Initialising UI",       0.95f);
    // Order: TAA owns linear_sampler_ + history images that compose samples;
    // bloom must run before compose so binding 3's bloom_mip_views_[0] is
    // live when compose's descriptor set is written.
    init_taa();
    init_taau();
    // SSDM Phase 1: dedicated displacement-vector pass. Must run after
    // init_taa() so render_extent_ is finalised (the disp image sizes to it)
    // and before init_compose() in case future phases hook into compose.
    init_ssdm();
    init_bloom();
    init_compose();
    // SVGF moments + 3-pass a-trous pipelines (4a-deep). Must run
    // AFTER init_taa() (reuses linear_sampler_ + taa_vert_module_) and
    // AFTER init_svgf_targets() (binds the moments + atrous storage
    // image views). Toggled per-frame via rt_.svgf_enabled; pipelines
    // stand up regardless so the toggle is a per-frame branch.
    init_svgf_passes();
    // Must run AFTER init_taa() — needs linear_sampler_ for the binding-18
    // descriptor write that lets cube.frag sample the half-res shadow.
    init_shadow_lr_pipeline();
    // Must run AFTER init_taa() — grass raymarch reuses taa_vert_module_
    // as its fullscreen-triangle vertex shader.
    init_grass_raymarch_pipeline();
    init_viewmodel();
    init_spacejet();
    init_fx();
    init_flags();
    init_vegetation_pipelines();
    init_clutter_shadow();
    init_imgui();
    update_scene_ubo();
    present_loader_frame("Ready",                 1.00f);
    // Loader frames advanced frame_number_ — reset so --frames N and
    // screenshot-after counters in run() count from the first real
    // gameplay frame, not from "real frames + N loader frames".
    frame_number_ = 0;
    if (window_) glfwSetWindowTitle(window_->handle(), "quake-like");
    initialized_ = true;
    log::info("VulkanEngine::init() done");
}


// init_imgui / destroy_imgui live in vk_ui.cpp.



// build_hud_ui / build_menu_ui live in vk_ui.cpp.





// Called on a device-lost (device handle still valid per spec). Asks
// the driver — via VK_EXT_device_fault — for the faulting GPU
// address(es) and vendor fault code, and logs them. This is the only
// way to get hard data on an intermittent device-lost that won't
// reproduce under (GPU-slowing) validation.
void VulkanEngine::dump_device_fault() {
    if (!device_fault_supported_ || !device_) return;
    auto pfn = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
        vkGetDeviceProcAddr(device_, "vkGetDeviceFaultInfoEXT"));
    if (!pfn) { log::warn("[devfault] vkGetDeviceFaultInfoEXT not loadable"); return; }

    VkDeviceFaultCountsEXT counts{};
    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    VkResult r = pfn(device_, &counts, nullptr);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        log::errorf("[devfault] count query failed (VkResult=%d)", static_cast<int>(r));
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT>  vend(counts.vendorInfoCount);
    counts.vendorBinarySize = 0;   // don't request the opaque vendor blob
    VkDeviceFaultInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    info.pAddressInfos = addrs.empty() ? nullptr : addrs.data();
    info.pVendorInfos  = vend.empty()  ? nullptr : vend.data();
    r = pfn(device_, &counts, &info);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        log::errorf("[devfault] info query failed (VkResult=%d)", static_cast<int>(r));
        return;
    }
    log::errorf("[devfault] --- GPU DEVICE FAULT ---");
    log::errorf("[devfault] description: %s", info.description);
    for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
        const VkDeviceFaultAddressInfoEXT& a = addrs[i];
        log::errorf("[devfault] addr[%u] type=%d reported=0x%016llX precision=0x%llX",
                    i, static_cast<int>(a.addressType),
                    static_cast<unsigned long long>(a.reportedAddress),
                    static_cast<unsigned long long>(a.addressPrecision));
    }
    for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
        const VkDeviceFaultVendorInfoEXT& v = vend[i];
        log::errorf("[devfault] vendor[%u] '%s' code=0x%016llX data=0x%016llX",
                    i, v.description,
                    static_cast<unsigned long long>(v.vendorFaultCode),
                    static_cast<unsigned long long>(v.vendorFaultData));
    }
    log::errorf("[devfault] --- end GPU DEVICE FAULT ---");
}

void VulkanEngine::run(const RunOptions& opts) {
    opts_ = opts;
    if (opts_.deferred >= 0) deferred_lighting_active_ = (opts_.deferred != 0);
    log::infof("run(): max_frames=%d screenshot='%s' after=%d",
               opts_.max_frames, opts_.screenshot_path.c_str(),
               opts_.screenshot_after_frames);

    bool headless = opts_.max_frames > 0 || !opts_.screenshot_path.empty();
    bool autodemo = opts_.autodemo_seconds > 0.0f;
    if (!headless) window_->set_cursor_captured(true);
    if (autodemo) {
        window_->set_cursor_captured(true);
        state_ = State::Playing;
        log::infof("[autodemo] running for %.2fs (forward+turn+fire)",
                   opts_.autodemo_seconds);
    }

    constexpr float kFixedDt = 1.0f / 120.0f;  // 120 Hz physics
    constexpr int   kMaxTicksPerFrame = 6;

    // FPS cap = primary monitor's refresh rate. Vulkan's FIFO present mode
    // *should* already pace the swap to the display, but we've measured
    // 500+ fps in this loop on some setups (driver fast-path / multi-
    // monitor mismatch). Letting the CPU outrun the GPU saturates the
    // RT pipeline and triggers TDR (DEVICE_LOST). A simple sleep-to-target
    // at the start of each frame caps the loop at the monitor's rate
    // without changing visuals. Falls back to 144 if GLFW can't read the
    // monitor (rare on Windows).
    int monitor_hz = 0;
    if (auto* monitor = glfwGetPrimaryMonitor()) {
        if (auto* mode = glfwGetVideoMode(monitor)) {
            monitor_hz = mode->refreshRate;
        }
    }
    if (monitor_hz <= 0) monitor_hz = 144;
    const auto frame_target = std::chrono::nanoseconds(1'000'000'000 / monitor_hz);
    log::infof("fps cap: %d Hz (monitor refresh)", monitor_hz);

    auto last = std::chrono::steady_clock::now();
    auto demo_start = last;
    prev_player_position_ = player_.position;

    while (!window_->should_close()) {
        window_->poll_events();
        if (resize_requested_) recreate_swapchain();

        // Frame-rate cap: sleep until at least frame_target has elapsed
        // since the previous frame began. std::this_thread::sleep_until
        // is precise enough on Windows ≥ 8.1 (timeBeginPeriod isn't
        // strictly needed at 144 Hz target), and the busy-wait tail in
        // the OS scheduler keeps wake-up jitter under ~1ms. Skipped in
        // headless mode (max_frames / screenshot) so test runs aren't
        // throttled to monitor refresh.
        if (!headless) {
            auto target = last + frame_target;
            auto now0 = std::chrono::steady_clock::now();
            if (now0 < target) std::this_thread::sleep_until(target);
        }

        auto now = std::chrono::steady_clock::now();
        float frame_dt = std::chrono::duration<float>(now - last).count();
        // Hitch diagnostics: section timings, logged for any slow frame.
        auto hz_mark = [](std::chrono::steady_clock::time_point& t) {
            auto n = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(n - t).count();
            t = n;
            return ms;
        };
        auto hz_t = now;
        float hz_ms[6] = {};
        const float hz_frame_ms = frame_dt * 1000.0f;
        if (hz_frame_ms > 28.0f && frame_number_ > 30)
            log::warnf("[hitch] frame %.1f ms | prev: update %.1f, jets+fx %.1f, ui %.1f, fence %.1f, acquire %.1f, draw %.1f "
                       "(draw: tlas-wait %.1f, cpu-build %.1f, record+submit %.1f)",
                       hz_frame_ms, hitch_prev_[0], hitch_prev_[1], hitch_prev_[2], hitch_prev_[3],
                       hitch_prev_[4], hitch_prev_[5], hitch_draw_[0], hitch_draw_[1], hitch_draw_[2]);
        {
            // Rolling average of the draw() split (every ~5 s): how long the
            // CPU waits on the GPU vs builds / records while the GPU idles.
            static double acc[4] = {}; static int acc_n = 0;
            acc[0] += hitch_draw_[0]; acc[1] += hitch_draw_[1]; acc[2] += hitch_draw_[2]; acc[3] += hz_frame_ms;
            if (++acc_n >= 300) {
                log::infof("[cpu] avg frame %.2f ms | prev-fence wait %.2f, cpu-build %.2f, record+submit %.2f",
                           acc[3] / acc_n, acc[0] / acc_n, acc[1] / acc_n, acc[2] / acc_n);
                acc[0] = acc[1] = acc[2] = acc[3] = 0.0; acc_n = 0;
            }
        }
        last = now;
        if (frame_dt > 1.0f / 15.0f) frame_dt = 1.0f / 15.0f;
        if (frame_dt < 1e-6f)        frame_dt = 1.0f / 240.0f;
        // Headless captures (--frames N --screenshot) need bit-stable physics
        // and spawn timing across runs; wallclock-derived dt varies by ~30%
        // run-to-run depending on shader compile, GPU contention, etc. Pin
        // it to a fixed 60 Hz step so crate spawns + Jolt integration land
        // at the same place every time and SAD comparisons are meaningful.
        if (opts_.max_frames > 0 || !opts_.screenshot_path.empty()) {
            frame_dt = 1.0f / 60.0f;
        }
        last_frame_dt_ = frame_dt;
        ema_fps_ = ema_fps_ == 0.0f
                   ? 1.0f / frame_dt
                   : (ema_fps_ * 0.9f + (1.0f / frame_dt) * 0.1f);

        InputFrame in = window_->consume_input();
        // F12 in-game capture: queue a numbered PPM in the working dir.
        // The render frame below picks it up and writes to disk.
        if (in.screenshot && pending_screenshot_path_.empty()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "screenshot_%04d.ppm",
                          ingame_screenshot_counter_++);
            pending_screenshot_path_ = buf;
            log::infof("[screenshot] F12 queued %s", buf);
        }
        if (autodemo) {
            float demo_t = std::chrono::duration<float>(now - demo_start).count();
            if (demo_t >= opts_.autodemo_seconds) {
                log::infof("[autodemo] elapsed %.2fs — requesting close",
                           demo_t);
                window_->request_close();
            }
            // Synthetic input: walk a calm counter-clockwise lap around the
            // castle on the plateau. Position is TELEPORTED each frame to
            // an orbit point (R=15 m, well inside the plateau and clear of
            // both the castle walls and the water below the plateau edge);
            // yaw points along the tangent so the camera looks straight
            // ahead in the direction of motion; pitch locked at 0 (no
            // head-bob / look-down). No fwd input — position override
            // does the moving, so physics never accelerates us off-orbit.
            // No jump, no fire — this demo is for visual + perf inspection,
            // not for crash repro (use a separate autodemo flavour for that
            // when it returns). y locked at 23 m (~1 m above plateau top).
            in = InputFrame{};
            // --scene N override: park the camera at one of the deferred-
            // renderer reference scenes for deterministic before/after
            // captures. Locks position/yaw/pitch and zeroes velocity each
            // frame. Each scene covers a specific lighting/material case
            // the migration must preserve.
            if (opts_.scene > 0) {
                struct ScenePose { glm::vec3 pos; float yaw; float pitch; };
                static const ScenePose kScenes[] = {
                    // 1: castle exterior, north entrance, sun behind us (lit walls).
                    // Yaw 0 faces -Z (south, toward the castle from z=+18);
                    // the old kPi here faced +Z and framed the voxel tower
                    // 100 m north instead of the castle.
                    { glm::vec3(0.0f,  23.5f,  18.0f),  0.0f,       0.10f },
                    // 2: castle exterior, south side (shadow side), looking north
                    { glm::vec3(0.0f,  23.5f, -18.0f),  kPi,        0.10f },
                    // 3: castle interior, looking up at the entrance lintel (PCSS)
                    { glm::vec3(0.0f,  23.5f,   2.0f),  kPi,        0.55f },
                    // 4: voxel tower mid-range (multi-shape, self-shadow)
                    { glm::vec3(8.0f,  27.0f,  35.0f),  kPi,        0.10f },
                    // 5: grass meadow far from castle (GI fed by terrain)
                    { glm::vec3(60.0f, 26.0f,  60.0f), -0.75f,      0.05f },
                    // 6: underwater at the bay (water blend chain)
                    { glm::vec3(40.0f, 19.0f, -40.0f),  0.5f,      -0.15f },
                    // 7: crate spawn zone (TAA + dyn-prop)
                    { glm::vec3(0.0f,  24.0f,   0.0f),  kHalfPi,    0.0f  },
                };
                int idx = std::clamp(opts_.scene, 1, 7) - 1;
                player_.position = kScenes[idx].pos;
                player_.yaw      = kScenes[idx].yaw;
                player_.pitch    = kScenes[idx].pitch;
                player_.velocity = glm::vec3(0.0f);
            } else {
                const float R     = 15.0f;
                const float omega = 0.30f;   // rad/s — ~17°/s
                const float ang   = demo_t * omega;
                player_.position  = glm::vec3(R * std::sin(ang),
                                              23.0f,
                                              R * std::cos(ang));
                player_.yaw       = ang + kHalfPi;          // tangent (CCW)
                player_.pitch     = 0.0f;
            }
            in.mouse_dx = 0.0;
            in.mouse_dy = 0.0;
        }
        // Dev hook for headless captures of weapon visuals: QLIKE_TEST_FIRE=1
        // pulls the trigger every 8th frame (pair with --screenshot).
        {
            static const bool test_fire = std::getenv("QLIKE_TEST_FIRE") != nullptr;
            static int test_fire_n = 0;
            if (test_fire && (++test_fire_n % 8) == 0) {
                in.fire = true;
                in.fire_held = true;
            }
            // QLIKE_TEST_HOLD=1: pin the camera at the restored settings
            // pose (no gravity) for aerial / overview captures.
            static const bool test_hold = std::getenv("QLIKE_TEST_HOLD") != nullptr;
            static bool hold_init = false;
            static glm::vec3 hold_pos{0.0f};
            // QLIKE_TEST_BOOM=<metres>: ground explosion that far ahead of
            // the camera at frame 40 (headless explosion captures).
            static const char* test_boom = std::getenv("QLIKE_TEST_BOOM");
            static int boom_frame = 0;
            if (test_boom && ++boom_frame == 40) {
                glm::vec3 f = player_.forward(); f.y = 0.0f;
                f = glm::normalize(f + glm::vec3(1e-4f));
                glm::vec3 p = player_.eye_position() + f * static_cast<float>(std::atof(test_boom));
                p.y = std::max(sample_terrain_height(p.x, p.z), rt_.water_level);
                spawn_explosion(p, 1.0f, true);
            }
            if (test_hold) {
                if (!hold_init) { hold_pos = player_.position; hold_init = true; }
                player_.position = hold_pos;
                player_.velocity = glm::vec3(0.0f);
                // QLIKE_TEST_PAN=<rad/s>: steady yaw pan (motion artefact captures).
                static const char* test_pan = std::getenv("QLIKE_TEST_PAN");
                if (test_pan) player_.yaw += static_cast<float>(std::atof(test_pan)) * frame_dt;
            }
        }
        // --scene N teleport works WITHOUT --autodemo too. Pin the camera
        // each frame so headless captures (--frames N --screenshot) hit the
        // intended pose; without this the camera kept whatever position the
        // settings file had saved from the previous run, silently producing
        // 17%+ SAD against the reference shots.
        if (opts_.scene > 0 && !autodemo) {
            struct ScenePose { glm::vec3 pos; float yaw; float pitch; };
            static const ScenePose kScenes[] = {
                // Keep in lock-step with the autodemo copy above (scene 1/2
                // yaws face the castle, not the voxel tower north of it).
                { glm::vec3(0.0f,  23.5f,  18.0f),  0.0f,       0.10f },
                { glm::vec3(0.0f,  23.5f, -18.0f),  kPi,        0.10f },
                { glm::vec3(0.0f,  23.5f,   2.0f),  kPi,        0.55f },
                { glm::vec3(8.0f,  27.0f,  35.0f),  kPi,        0.10f },
                { glm::vec3(60.0f, 26.0f,  60.0f), -0.75f,      0.05f },
                { glm::vec3(40.0f, 19.0f, -40.0f),  0.5f,      -0.15f },
                { glm::vec3(0.0f,  24.0f,   0.0f),  kHalfPi,    0.0f  },
            };
            int idx = std::clamp(opts_.scene, 1, 7) - 1;
            player_.position = kScenes[idx].pos;
            player_.yaw      = kScenes[idx].yaw;
            player_.pitch    = kScenes[idx].pitch;
            player_.velocity = glm::vec3(0.0f);
            in.mouse_dx = 0.0;
            in.mouse_dy = 0.0;
        }
        bool menu_edge = in.menu_key && !prev_menu_key_;
        prev_menu_key_ = in.menu_key;
        // E toggles terrain edit mode (task #201). Edge-detected so a held
        // key doesn't oscillate. Only fires while playing — pressing E in
        // the menu shouldn't drop into edit-mode behind the menu.
        bool edit_edge = in.edit_key && !prev_edit_key_;
        prev_edit_key_ = in.edit_key;
        if (edit_edge && state_ == State::Playing) {
            terrain_edit_mode_ = !terrain_edit_mode_;
            log::infof("[terrain] edit mode %s (E)",
                       terrain_edit_mode_ ? "ON" : "OFF");
        }
        // Digit 9 -> toggle terrain wireframe. Same edge-detect pattern
        // as the E key above. Only fires while playing so a 9 in the
        // pause menu / chat input wouldn't flip it.
        bool wire_edge = in.wireframe_key && !prev_wireframe_key_;
        prev_wireframe_key_ = in.wireframe_key;
        if (wire_edge && state_ == State::Playing) {
            rt_.terrain_wireframe = !rt_.terrain_wireframe;
            log::infof("[terrain] wireframe %s (9)",
                       rt_.terrain_wireframe ? "ON" : "OFF");
        }
        // Hold [ / ] in edit mode to shrink/grow the brush radius
        // (mirrors the common sculpt UX in Blender, Houdini, etc.).
        // 4 m/s rate covers the practical 0.5-50 m clamp in ~12 s; the
        // existing slider in the tools panel still works.
        if (terrain_edit_mode_ && state_ == State::Playing) {
            const float kBrushRate = 4.0f;
            if (in.brush_larger)  terrain_brush_radius_ += kBrushRate * frame_dt;
            if (in.brush_smaller) terrain_brush_radius_ -= kBrushRate * frame_dt;
            terrain_brush_radius_ = std::clamp(terrain_brush_radius_, 0.5f, 50.0f);
        }
        // Q / R cycle brush mode (edge-detected so a held key doesn't spin
        // through every mode each frame). Wraps around the 8-entry enum.
        bool bm_prev_edge = in.brush_mode_prev && !prev_brush_mode_prev_;
        bool bm_next_edge = in.brush_mode_next && !prev_brush_mode_next_;
        prev_brush_mode_prev_ = in.brush_mode_prev;
        prev_brush_mode_next_ = in.brush_mode_next;
        if (terrain_edit_mode_ && state_ == State::Playing &&
            (bm_prev_edge || bm_next_edge)) {
            constexpr int kModeCount = 8;   // matches TerrainBrushMode enum
            int m = static_cast<int>(terrain_brush_mode_);
            m = (m + (bm_next_edge ? 1 : kModeCount - 1)) % kModeCount;
            terrain_brush_mode_ = static_cast<TerrainBrushMode>(m);
            static const char* kModeNames[kModeCount] = {
                "Raise", "Lower", "Smooth", "Flatten",
                "GrassAdd", "GrassRemove", "Erode", "ErodeSmooth"
            };
            log::infof("[terrain] brush mode -> %s (%s)",
                       kModeNames[m], bm_next_edge ? "R" : "Q");
        }
        if (menu_edge) {
            if (state_ == State::Playing) {
                state_ = State::Paused;
                window_->set_cursor_captured(false);
            } else {
                state_ = State::Playing;
                window_->set_cursor_captured(true);
            }
        }

        // Mouse-look at render rate (low latency). Only when window has focus
        // AND cursor is captured AND we're in Playing state.
        if (state_ == State::Playing && window_->cursor_captured() &&
            window_->has_focus()) {
            float dx = static_cast<float>(in.mouse_dx);
            float dy = static_cast<float>(in.mouse_dy);
            player_.yaw   += dx * game::Player::kMouseSensitivity;
            player_.pitch -= dy * game::Player::kMouseSensitivity;
            // kHalfPi from engine/vk_engine/internal.h.
            constexpr float kPitchCap = kHalfPi - 0.01f;
            if (player_.pitch >  kPitchCap) player_.pitch =  kPitchCap;
            if (player_.pitch < -kPitchCap) player_.pitch = -kPitchCap;
        }

        // Auto-fire: hold LMB, fires at game_.fire_rate_rps. The cooldown
        // accumulates frame_dt so the rate is independent of frame rate.
        // fire_rate_rps == 0 falls back to one-shot-per-click.
        if (fire_cooldown_ > 0.0f) fire_cooldown_ -= frame_dt;
        if (muzzle_flash_timer_ > 0.0f) muzzle_flash_timer_ -= frame_dt;
        if (recoil_timer_       > 0.0f) recoil_timer_       -= frame_dt;
        if (state_ == State::Playing && (window_->cursor_captured() ||
                                           std::getenv("QLIKE_TEST_FIRE"))) {
            // Raymarch terrain reads FBM directly in the shader — sculpting
            // the heightmap has no visual effect and only triggers wasted
            // chunk rebuilds / BLAS rebuilds / texture uploads.
            if (terrain_edit_mode_ && !rt_.terrain_raymarch_enabled) {
                // ---- Phase 4 sculpt path ----
                // Raycast from the eye along the camera forward against
                // the heightfield (Jolt's static body) — works because
                // the heightfield is in NON_MOVING and add_static_heightfield
                // uses the standard collision layer.
                glm::vec3 origin = player_.eye_position();
                glm::vec3 dir    = player_.forward();
                if (physics_) {
                    auto rh = physics_->raycast(origin, dir, 400.0f);
                    if (rh.hit) {
                        terrain_brush_world_pos_ = rh.position;
                        terrain_brush_has_hit_ = true;
                    } else {
                        terrain_brush_has_hit_ = false;
                    }
                }
                bool sculpt_active = (in.fire_held || in.alt_fire_held) &&
                                      terrain_brush_has_hit_;
                if (sculpt_active) {
                    // Right-click swaps mode to its inverse for the
                    // duration of the stroke: Raise↔Lower for the
                    // height brush, GrassAdd↔GrassRemove for the grass
                    // brush. Smooth / Flatten stay as-is. Restored
                    // after the call so the UI shows the user's
                    // selected mode on the next frame.
                    TerrainBrushMode prev_mode = terrain_brush_mode_;
                    if (in.alt_fire_held && !in.fire_held) {
                        if (terrain_brush_mode_ == TerrainBrushMode::Raise) {
                            terrain_brush_mode_ = TerrainBrushMode::Lower;
                        } else if (terrain_brush_mode_ == TerrainBrushMode::GrassAdd) {
                            terrain_brush_mode_ = TerrainBrushMode::GrassRemove;
                        } else if (terrain_brush_mode_ == TerrainBrushMode::Erode) {
                            terrain_brush_mode_ = TerrainBrushMode::ErodeSmooth;
                        }
                    }
                    apply_terrain_brush(frame_dt);
                    terrain_brush_mode_ = prev_mode;
                    // CLOD raster reads the heightmap texture, so the
                    // stroke shows up via the terrain_height_dirty_
                    // re-upload — no per-chunk mesh rebuild needed.
                    if (!terrain_stroke_active_) terrain_stroke_active_ = true;
                } else if (terrain_stroke_active_ && !in.fire_held && !in.alt_fire_held) {
                    // Mouse-up: heavy refresh (BLAS + Jolt) deferred until
                    // here so the stroke itself stays cheap. Grass-only
                    // strokes have no dirty bits to consume — the
                    // mask reupload already happened the frame the
                    // stroke modified the CPU mirror.
                    if (!brush_mode_is_grass(terrain_brush_mode_)) {
                        refresh_terrain_blas();
                        refresh_terrain_collision();
                        // Re-bake the curvature/sky-vis aux for the new
                        // surface (cheap ~100 ms, only at stroke end).
                        vkDeviceWaitIdle(device_);
                        destroy_terrain_aux_texture();
                        init_terrain_aux_texture();
                    }
                    terrain_stroke_active_ = false;
                }
            } else {
                float rps = std::max(0.0f, game_.fire_rate_rps);
                if (rps <= 0.0f) {
                    if (in.fire) {
                        fire_projectile(player_.eye_position(), player_.forward());
                    }
                } else if (in.fire_held) {
                    float interval = 1.0f / rps;
                    if (fire_cooldown_ <= 0.0f) {
                        fire_projectile(player_.eye_position(), player_.forward());
                        fire_cooldown_ += interval;
                    }
                } else {
                    fire_cooldown_ = 0.0f;  // reset between bursts
                }
            }
        }

        // Fixed-timestep physics: accumulate frame dt, drain in fixed ticks.
        last_physics_ticks_ = 0;
        if (state_ == State::Playing) {
            physics_accumulator_ += frame_dt;
            if (physics_accumulator_ > kFixedDt * (kMaxTicksPerFrame + 1)) {
                physics_accumulator_ = kFixedDt * kMaxTicksPerFrame;  // anti-spiral
            }
            while (physics_accumulator_ >= kFixedDt &&
                   last_physics_ticks_ < kMaxTicksPerFrame) {
                prev_player_position_ = player_.position;
                game::PlayerInput pin{};
                pin.fwd = in.fwd; pin.back = in.back;
                pin.left = in.left; pin.right = in.right;
                pin.jump = in.jump;
                pin.sprint = in.sprint;
                pin.crawl = in.crawl;
                pin.gravity = game_.gravity;
                pin.mouse_dx = 0.0f;  // mouse already applied above
                pin.mouse_dy = 0.0f;
                rebuild_tick_aabbs();
                glm::vec3 pre_vel = player_.velocity;
                bool was_on_ground = player_.on_ground;
                game::update_player(player_, pin, tick_aabbs_, kFixedDt,
                                     world_.aabbs.size() +
                                     voxel_collision_aabbs_.size());
                // Terrain ground clamp. update_player collides only
                // against AABBs; the heightfield isn't an AABB so the
                // player would free-fall through it. We:
                //   1. Sample terrain at player XZ.
                //   2. Estimate slope normal from finite differences;
                //      anything steeper than ~50° is a "wall" — no
                //      ground-clamp, the player slides off.
                //   3. If feet sank below ground, snap up.
                //   4. STICKY GROUNDING: if the player WAS grounded
                //      last tick and is still moving downward (or zero)
                //      vertically and their feet sit within kStepDown
                //      above the new ground sample, snap them DOWN.
                //      Without this, walking down a slope alternates
                //      airborne/grounded each frame — gravity accumulates
                //      between snaps and the player slides faster than
                //      they intended (the user's "weird sliding").
                {
                    constexpr float kHalfHeight   = 0.9f;
                    constexpr float kStepDown     = 0.45f;
                    constexpr float kMinGroundN   = 0.55f;  // ≈56° max walkable
                    constexpr float kSampleDelta  = 1.0f;
                    float gx = player_.position.x;
                    float gz = player_.position.z;
                    float h0  = sample_terrain_height(gx, gz);
                    float hpx = sample_terrain_height(gx + kSampleDelta, gz);
                    float hnx = sample_terrain_height(gx - kSampleDelta, gz);
                    float hpz = sample_terrain_height(gx, gz + kSampleDelta);
                    float hnz = sample_terrain_height(gx, gz - kSampleDelta);
                    glm::vec3 n(-(hpx - hnx) / (2.0f * kSampleDelta),
                                 1.0f,
                                 -(hpz - hnz) / (2.0f * kSampleDelta));
                    n = glm::normalize(n);
                    bool walkable = n.y > kMinGroundN;

                    float feet = player_.position.y - kHalfHeight;
                    if (feet < h0) {
                        // Snapped up onto terrain.
                        player_.position.y = h0 + kHalfHeight;
                        if (player_.velocity.y < 0.0f) player_.velocity.y = 0.0f;
                        if (walkable) player_.on_ground = true;
                    } else if (walkable && was_on_ground &&
                               !player_.on_ground &&
                               player_.velocity.y <= 0.5f &&
                               (feet - h0) < kStepDown) {
                        // Sticky-ground: pull the player down onto the
                        // surface they were just on. Keeps slopes and
                        // stairs feeling "stuck to the ground" instead of
                        // hopping on every step.
                        //
                        // Skip if slide_move already grounded us on a brush
                        // (player_.on_ground TRUE this tick). Otherwise the
                        // castle floor brush sits 5 cm above the heightmap
                        // plateau and sticky-ground yanks the player DOWN
                        // through the floor every tick — depenetration
                        // pushes them back up next frame, oscillating
                        // between 22.0 and 22.05. That oscillation:
                        //   - reads as "invisible terrain" (heightmap
                        //     under the rendered castle floor),
                        //   - disrupts stair step-up (player feet keep
                        //     reverting to terrain Y between steps).
                        player_.position.y = h0 + kHalfHeight;
                        player_.velocity.y = 0.0f;
                        player_.on_ground = true;
                    }
                }
                // Audio triggers from the player tick. Land sound used to
                // fire on every air→ground transition, but on_ground flickers
                // each tick while walking over uneven terrain → it doubled as
                // an unwanted footstep sound. Removed; only jump remains.
                if (audio_ && state_ == State::Playing) {
                    if (was_on_ground && !player_.on_ground && pin.jump) {
                        audio_->play_local(ClipID::Jump, 0.7f, 0.06f, 0.10f);
                    }
                }
                apply_player_pushes(pre_vel);
                if (physics_) physics_->step(kFixedDt);
                update_projectiles(kFixedDt);
                update_particles(kFixedDt);
                update_decals(kFixedDt);
                physics_accumulator_ -= kFixedDt;
                ++last_physics_ticks_;
            }

            // Continuous spawner: drop a new tilted box on a timer derived
            // from the user-configurable cubes-per-minute setting.
            float interval = 60.0f / std::max(1.0f, float(game_.cubes_per_minute));
            spawn_timer_ += frame_dt;
            while (spawn_timer_ >= interval) {
                spawn_timer_ -= interval;
                spawn_random_box();
            }

            pos_log_timer_ += frame_dt;
            if (pos_log_timer_ >= 1.0f) {
                pos_log_timer_ = 0.0f;
                glm::vec2 hv(player_.velocity.x, player_.velocity.z);
                log::infof("player pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f) "
                           "speed_h=%.3f grounded=%d feet_y=%.3f fps=%.1f ticks=%d",
                           player_.position.x, player_.position.y, player_.position.z,
                           player_.velocity.x, player_.velocity.y, player_.velocity.z,
                           glm::length(hv),
                           player_.on_ground ? 1 : 0,
                           player_.position.y - game::Player::kHalfExtents.y,
                           ema_fps_, last_physics_ticks_);
                log::infof("state  proj=%zu particles=%zu dyn_props=%zu "
                           "tlas_n=%u frame=%llu vk_warn=%llu vk_err=%llu",
                           projectiles_.size(), particles_.size(),
                           dyn_props_.size(), last_tlas_n_,
                           static_cast<unsigned long long>(frame_number_),
                           static_cast<unsigned long long>(g_validation_warning_count.load()),
                           static_cast<unsigned long long>(g_validation_error_count.load()));
                if (gpu_query_pool_) {
                    log::infof("gpu    frame=%.2f ms (TDR limit ~2000 ms)",
                               gpu_timers_.scene_ms);
                    prof_log();
                }
            }
        }

        // Voxel destruction: debounced heavy work (collapse + collision
        // rebuild) once per frame, off the per-bullet impact path.
        process_voxel_updates(frame_dt);

        // Castle gate animation — proximity trigger + lerp toward target
        // angles. Runs once per frame (the lerp is dt-relative so it's
        // not tied to the physics fixed-step).
        update_doors(frame_dt);

        // Recompute the sun shadow-map light view-proj before update_scene_ubo
        // so the matrix copied into the UBO this frame matches the geometry
        // we'll draw into the shadow target.
        update_sun_shadow_light_vp();
        // If the user changed the shadow-map resolution slider, recreate the
        // image at the new size. Cheap (one vkDeviceWaitIdle + a 4-16MB
        // alloc); only fires when the slider actually moves.
        if (rt_.shadow_map_resolution != sun_shadow_dim_) {
            vkDeviceWaitIdle(device_);
            destroy_sun_shadow_resources();
            init_sun_shadow_resources();
        }
        // (The chunk-LOD stride rebake + near-densify systems are gone —
        // the quadtree CLOD terrain has no per-chunk geometry to rebake.
        // Density follows the camera continuously via per-frame node
        // selection in update_terrain_clod_instances.)
        // Heightmap-bake supersample changed → recreate texture and
        // restart the worker so subsequent jobs run at the new ss.
        // The user can flip 1x/2x/4x freely; texture realloc is cheap.
        if (rt_.terrain_bake_supersample != terrain_shadow_active_ss_) {
            vkDeviceWaitIdle(device_);
            stop_terrain_shadow_worker();
            destroy_terrain_shadow_texture();
            rebuild_terrain_shadow_texture();
            start_terrain_shadow_worker();
            // Trigger an immediate re-bake at the new ss so the worker
            // refines beyond the SS=1 fallback the sync-bake replicated.
            float p_rad = glm::radians(rt_.sun_pitch_deg);
            float y_rad = glm::radians(rt_.sun_yaw_deg);
            glm::vec3 cur(std::sin(y_rad) * std::cos(p_rad),
                          std::sin(p_rad),
                          std::cos(y_rad) * std::cos(p_rad));
            enqueue_terrain_shadow_rebake(cur);
        }
        // update_scene_ubo() moved into draw(), after the prev-frame fence
        // wait: writing it here raced the GPU still reading it for the
        // previous frame (single host-visible UBO shared by both slots).
        // Heightmap sun-shadow is sun-direction-dependent. Instead of
        // re-baking the whole 1024² texture in one go (~100 ms hitch),
        // we tile the bake (kShadowTileSize per side), sort tiles by
        // distance to the camera, and drain a small budget per frame.
        // Near-camera tiles update first so grass shadows under the
        // player respond promptly to the sun slider; distant tiles
        // catch up over the next handful of frames. Threshold ~1.4°
        // — enough to absorb continuous slider drag without thrashing
        // the queue.
        // Lazy-build for the rasterised mesh/chunks/grass — only fires
        // the first time the user toggles a raymarch flag off (init
        // skips the build when raymarch is on to save VRAM).
        if (!rt_.terrain_raymarch_enabled) ensure_terrain_raster_built();
        if (rt_.grass_enabled && !rt_.grass_raymarch_enabled) {
            ensure_grass_raster_built();
        }
        // TAAU toggle: re-point compose's history sampler at taau_view_ or
        // history_view_. Cheap when state matches.
        rewrite_compose_for_taau();
        // FSR2 toggle: takes priority over TAAU/TAA in compose's history
        // binding when the context is valid. wait_idle is paid only on the
        // actual switch.
        rewrite_compose_for_fsr2();
        // FSR3 toggle: only meaningful when fsr2_enabled is on AND the
        // backend is 1 (FSR3). Same wait-idle-only-on-actual-switch semantics.
        rewrite_compose_for_fsr3();

        // Heightmap-shadow texture is sampled by cube.frag's mesh-terrain
        // branch and grass.vert (raster grass). When raymarch terrain is on
        // AND raster grass is off, nothing reads it — skip the rebake.
        if (!terrain_data_.heights.empty() && needs_heightmap_shadow()) {
            float p_rad = glm::radians(rt_.sun_pitch_deg);
            float y_rad = glm::radians(rt_.sun_yaw_deg);
            glm::vec3 cur_sun(std::sin(y_rad) * std::cos(p_rad),
                              std::sin(p_rad),
                              std::cos(y_rad) * std::cos(p_rad));
            if (glm::distance(cur_sun, terrain_shadow_target_sun_dir_) > 0.025f) {
                enqueue_terrain_shadow_rebake(cur_sun);
            }
            tick_terrain_shadow_progressive();
        }
        // Re-upload the heightmap delta texture if the user sculpted /
        // added plateau noise this frame. Cheap one-shot upload of
        // dim*dim*4 bytes; happens only on the frame the dirty bit
        // is set, not every frame the brush is held.
        if (terrain_height_dirty_) {
            refresh_terrain_height_texture();
            refresh_terrain_height_full_texture();
        }
        // Reupload grass eligibility mask if a GrassAdd/GrassRemove
        // stroke modified the CPU mirror this frame. Same one-shot
        // upload pattern as the heightmap delta texture above.
        if (grass_mask_dirty_) {
            reupload_grass_mask();
        }
        // Audio listener follows the camera. Reaping drained one-shots
        // also runs here so the active-voice list stays small.
        if (audio_) {
            audio_->set_listener(player_.eye_position(), player_.forward());
        }
        hz_ms[0] = hz_mark(hz_t);
        // Spacejet flyovers: advance active flights, spawn next wave.
        tick_spacejet(frame_dt);
        tick_fx(frame_dt);
        tick_aa(frame_dt);
        maybe_autosave_settings(frame_dt);
        hz_ms[1] = hz_mark(hz_t);

        // ImGui frame begins regardless; we just don't fill it when playing.
        if (imgui_initialized_) {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            build_hud_ui();
            build_menu_ui();
            ImGui::Render();
        }

        // Frame body wrapped in a try/catch so any vk_check throw from a
        // sync/submit call is caught HERE — at the throw site's stack frame,
        // before the runtime_error's heap message can be invalidated by
        // unwinding through the engine's deeper destructors. We log the type
        // and break the loop; shutdown() then sees device_lost_ and skips the
        // vkDeviceWaitIdle that would otherwise stall on the dead GPU.
        try {
            // SwapChain-proxy teardown gate — runs BEFORE acquire so we
            // don't leak a swapchain semaphore. When the user toggles
            // FSR off (or FG turns off), the SDK's proxied present
            // doesn't gracefully pass-through — symptom is world
            // geometry stops reaching the screen, only sky-clear
            // survives. Tear the proxy down + rebuild plain swapchain
            // here, then `continue` the loop without touching the
            // half-prepared frame state.
            {
                const bool fg_should_run = rt_.fsr2_enabled && rt_.fg_enabled &&
                                           !fsr3_fg_fatal_ && !fg_runtime_disabled_ &&
                                           rt_.fsr_backend == 1;
                if (!fg_should_run && fsr3_swapchain_active_) {
                    log::info("[fsr3 sc] FG/FSR turned off — tearing proxy down + restoring plain swapchain");
                    vkDeviceWaitIdle(device_);
                    destroy_fsr3_swapchain();
                    for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
                    swapchain_views_.clear();
                    swapchain_images_.clear();
                    for (auto s : render_semaphores_) {
                        if (s) vkDestroySemaphore(device_, s, nullptr);
                    }
                    render_semaphores_.clear();
                    init_swapchain();
                    continue;   // re-enter loop with fresh swapchain
                }
            }
            hz_ms[2] = hz_mark(hz_t);
            auto& frame = current_frame();
            QLIKE_VK_CHECK(vkWaitForFences(device_, 1, &frame.render_fence,
                                           VK_TRUE, UINT64_MAX),
                           "vkWaitForFences");
            hz_ms[3] = hz_mark(hz_t);
            // GPU watchdog: this slot's previous-iteration GPU work is
            // now finished (we just waited on its fence) so its
            // timestamps are safe to read non-blocking. Internally
            // logs a WARN at >1500 ms (Windows TDR is ~2 s).
            read_gpu_timestamps_for_slot(frame_number_ % kFrameOverlap);
            prof_read_slot(frame_number_ % kFrameOverlap);

            uint32_t img_index = 0;
            VkResult acq = acquire_next_image(UINT64_MAX,
                                              frame.swapchain_semaphore,
                                              VK_NULL_HANDLE, &img_index);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) { resize_requested_ = true; continue; }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                QLIKE_VK_CHECK(acq, "vkAcquireNextImageKHR");
            }

            QLIKE_VK_CHECK(vkResetFences(device_, 1, &frame.render_fence),
                           "vkResetFences");
            QLIKE_VK_CHECK(vkResetCommandBuffer(frame.command_buffer, 0),
                           "vkResetCommandBuffer");

            hz_ms[4] = hz_mark(hz_t);
            draw(img_index);
            hz_ms[5] = hz_mark(hz_t);
            for (int k = 0; k < 6; ++k) hitch_prev_[k] = hz_ms[k];

            ++frame_number_;
            {
                static bool first_logged = false;
                if (!first_logged) {
                    first_logged = true;
                    log::infof("[init] first game frame submitted %.0f ms after launch",
                               glfwGetTime() * 1000.0);
                }
            }
            if (opts_.max_frames >= 0 &&
                static_cast<int>(frame_number_) >= opts_.max_frames) {
                log::infof("max_frames reached (%d), exiting loop",
                           opts_.max_frames);
                break;
            }
            if (!opts_.screenshot_path.empty() && screenshot_taken_) {
                log::info("screenshot captured, exiting loop");
                break;
            }
        } catch (const std::exception& e) {
            // typeid().name() is rdata-stored — safe even if e.what()'s heap
            // string has been freed by unwind.
            log::errorf("[run] frame aborted via std::exception (type=%s) — "
                        "marking device_lost and exiting loop",
                        typeid(e).name());
            device_lost_ = true;
            // The device handle is still valid here (spec allows
            // vkGetDeviceFaultInfoEXT after device-lost) — grab the GPU
            // fault info NOW, before shutdown tears the device down.
            dump_device_fault();
            break;
        }
    }
    // Skip the wait if we already know the device is dead — otherwise we
    // hang in vkDeviceWaitIdle for several seconds before shutdown can run.
    if (device_ && !device_lost_) vkDeviceWaitIdle(device_);
}

void VulkanEngine::shutdown() {
    if (!initialized_) return;
    // If run() flagged the device as lost, skip vkDeviceWaitIdle — waiting on
    // a hung GPU just stalls shutdown for seconds before returning DEVICE_LOST
    // anyway. Subsequent vkDestroy* calls accept a "device_lost" device.
    if (device_ && !device_lost_) vkDeviceWaitIdle(device_);
    log::infof("VulkanEngine::shutdown() (device_lost=%d)",
               device_lost_ ? 1 : 0);

    // Each cleanup step is wrapped so a single failure can't propagate out
    // and abort the whole shutdown — leaving Vulkan / GLFW in a half-torn-
    // down state on the way to process exit. Any exception is logged
    // defensively (exception_ptr inspected via type_info — what() can be
    // unsafe if the exception's internal storage has already been freed by
    // an earlier unwinding step). Cleanup continues regardless.
    auto guarded = [](const char* label, auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            // typeid().name() lives in rdata — safe to log even if the
            // exception's heap-allocated what() string is freed/poisoned
            // (we've seen 0xDD). e.what() itself is intentionally untouched.
            log::errorf("[shutdown] %s threw std::exception (type=%s)",
                        label, typeid(e).name());
        } catch (...) {
            log::errorf("[shutdown] %s threw non-std exception", label);
        }
    };

    guarded("save_settings",  [&]{ save_settings(); });
    // FSR2 owns Vulkan resources via its scratch + context — destroy
    // first so the device + descriptor pool tear-down below isn't
    // racing with the SDK's internal allocator.
    guarded("destroy_fsr2",   [&]{ destroy_fsr2(); });
    guarded("destroy_dlss",   [&]{ destroy_dlss(); });   // while the allocator still exists
    // Destroy SwapChain proxy first — it holds Vulkan resources tied
    // to the FG context's pacing state.
    guarded("destroy_fsr3_sc",[&]{ destroy_fsr3_swapchain(); });
    guarded("destroy_fsr3_fg",[&]{ destroy_fsr3_fg(); });
    guarded("destroy_fsr3",   [&]{ destroy_fsr3(); });
    guarded("destroy_restir", [&]{ destroy_restir(); });
    guarded("destroy_imgui",  [&]{ destroy_imgui(); });
    guarded("destroy_loader_bg", [&]{ destroy_loader_bg_(); });
    guarded("destroy_viewmodel", [&]{ destroy_viewmodel(); });
    guarded("destroy_spacejet",  [&]{ destroy_spacejet(); });
    guarded("destroy_fx",        [&]{ destroy_fx(); });
    // Reverse of init order: compose first (samples bloom + TAA history),
    // then bloom (samples scene), then TAA (owns linear_sampler_ + history).
    guarded("destroy_compose", [&]{ destroy_compose(); });
    guarded("destroy_bloom",   [&]{ destroy_bloom(); });
    guarded("destroy_taau",    [&]{ destroy_taau(); });
    // SSDM owns its own pipeline + image; tear down before destroy_taa
    // which removes the cube vert_module_/pipeline_layout_ it borrows.
    // Actually those live on init_pipeline (destroyed in destroy_pipeline),
    // not destroy_taa, but the ordering doesn't matter for SSDM's own
    // resources — they're self-contained.
    guarded("destroy_ssdm",    [&]{ destroy_ssdm(); });
    // SVGF passes own pipelines + descriptors that reference
    // linear_sampler_ (owned by TAA) and the SVGF storage image views
    // (owned by init_svgf_targets). Tear them down BEFORE destroy_taa
    // (sampler) and destroy_svgf_targets (image views).
    guarded("destroy_svgf_passes", [&]{ destroy_svgf_passes(); });
    guarded("destroy_taa",     [&]{ destroy_taa(); });
    // Order: tear down the renderer's GPU dependencies on dynamic-prop world
    // matrices (TLAS instance buffer) BEFORE destroying the physics world that
    // owns those bodies. Otherwise the renderer's deferred destruction could
    // deref freed Jolt body data.
    guarded("destroy_veg",    [&]{ destroy_vegetation(); });
    guarded("destroy_aa",     [&]{ destroy_aa(); });
    guarded("destroy_clutter_shadow", [&]{ destroy_clutter_shadow(); });
    guarded("destroy_castle", [&]{ destroy_castle_meshes(); });
    guarded("destroy_rt",     [&]{ destroy_rt(); });
    guarded("physics_.reset", [&]{ physics_.reset(); });
    guarded("destroy_descriptors", [&]{ destroy_descriptors(); });
    guarded("destroy_meshes", [&]{
        destroy_mesh(allocator_, cube_mesh_);
        destroy_mesh(allocator_, cylinder_mesh_);
    });
    guarded("destroy_gpu_query_pool", [&]{ destroy_gpu_query_pool(); });
    guarded("destroy_skybox", [&]{ destroy_skybox_resources(); });
    guarded("destroy_grass_mask_texture", [&]{ destroy_grass_mask_texture(); });
    guarded("destroy_terrain_aux_texture", [&]{ destroy_terrain_aux_texture(); });
    guarded("destroy_fog_wisp_texture",   [&]{ destroy_fog_wisp_texture(); });
    guarded("destroy_textures", [&]{ destroy_textures(); });
    guarded("destroy_grass_raymarch_pipeline", [&]{ destroy_grass_raymarch_pipeline(); });
    guarded("destroy_grass_pipeline", [&]{ destroy_grass_pipeline(); });
    guarded("destroy_scatter_pipeline", [&]{ destroy_scatter_pipeline(); });
    guarded("destroy_terrain_height_texture", [&]{ destroy_terrain_height_texture(); });
    guarded("destroy_vrs_attachment", [&]{ destroy_vrs_attachment(); });
    guarded("destroy_terrain_raymarch_lowres", [&]{ destroy_terrain_raymarch_lowres(); });
    guarded("destroy_shadow_lr", [&]{ destroy_shadow_lr(); });
    guarded("destroy_svgf_targets", [&]{ destroy_svgf_targets(); });
    guarded("destroy_terrain_raymarch_compose_pipeline", [&]{ destroy_terrain_raymarch_compose_pipeline(); });
    guarded("destroy_terrain_raymarch_pipeline", [&]{ destroy_terrain_raymarch_pipeline(); });
    guarded("destroy_terrain_water_pipeline", [&]{ destroy_terrain_water_pipeline(); });
    guarded("destroy_terrain_pipelines", [&]{ destroy_terrain_pipelines(); });
    guarded("destroy_sun_shadow_pipeline", [&]{ destroy_sun_shadow_pipeline(); });
    guarded("destroy_voxel", [&]{ destroy_voxel(); });
    guarded("destroy_deferred_lighting", [&]{ destroy_deferred_lighting(); });
    guarded("destroy_pipeline", [&]{ destroy_pipeline(); });
    guarded("destroy_grass", [&]{ destroy_grass(allocator_, grass_); });
    guarded("destroy_scatter", [&]{ destroy_scatter(allocator_, scatter_); });
    guarded("stop_terrain_shadow_worker", [&]{ stop_terrain_shadow_worker(); });
    guarded("destroy_terrain_shadow_texture", [&]{ destroy_terrain_shadow_texture(); });
    guarded("destroy_sun_shadow_resources", [&]{ destroy_sun_shadow_resources(); });
    guarded("destroy_pipeline_cache", [&]{ destroy_pipeline_cache(); });
    guarded("destroy_readback_buffer", [&]{ destroy_readback_buffer(); });

    guarded("destroy_frames", [&]{
        for (auto& f : frames_) {
            if (f.render_fence) vkDestroyFence(device_, f.render_fence, nullptr);
            if (f.swapchain_semaphore) vkDestroySemaphore(device_, f.swapchain_semaphore, nullptr);
            if (f.tlas_build_done) vkDestroySemaphore(device_, f.tlas_build_done, nullptr);
            if (f.command_pool) vkDestroyCommandPool(device_, f.command_pool, nullptr);
            if (f.compute_pool) vkDestroyCommandPool(device_, f.compute_pool, nullptr);
            f = {};
        }
    });

    guarded("destroy_depth_image", [&]{ destroy_depth_image(); });
    guarded("depth_copy_sampler", [&]{
        if (depth_copy_sampler_) vkDestroySampler(device_, depth_copy_sampler_, nullptr);
        depth_copy_sampler_ = VK_NULL_HANDLE;
    });
    guarded("destroy_swapchain",   [&]{ destroy_swapchain(); });

    guarded("vmaDestroyAllocator", [&]{
        if (allocator_) vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
    });

    guarded("sl_shutdown", [&]{ sl_shutdown(); });   // Streamline: before the device
    guarded("destroy_surface_device_instance", [&]{
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
        if (device_) vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        if (debug_messenger_) vkb::destroy_debug_utils_messenger(instance_, debug_messenger_);
        debug_messenger_ = VK_NULL_HANDLE;
        if (instance_) vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    });

    guarded("window.reset",   [&]{ window_.reset(); });
    initialized_ = false;
}

} // namespace qlike
