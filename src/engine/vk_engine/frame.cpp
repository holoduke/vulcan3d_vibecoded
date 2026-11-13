// Per-frame render-graph orchestration: VulkanEngine::draw() walks the
// whole frame -- depth pre-pass, shadow bake, world color pass, deferred
// lighting, TAA/FSR, bloom, compose, screenshot readback, present.
// Split out of vk_engine.cpp (maintainability pass).

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
// Compose push constants (compose.frag + its -DSKY_FILL variant).
struct ComposePC {
    glm::vec4 viewport;
    glm::vec4 bloom_params;
    glm::vec4 sun_dir;
    glm::vec4 sun_color;
    glm::vec4 sky_color;
    glm::vec4 flare_params;   // strength, threshold, dispersal, halo_w
    glm::vec4 flare_params2;  // ghost_count, aberration, enabled, _
    glm::vec4 sun_screen;     // xy = sun screen-uv, z = visibility (0/1), w = spare
    glm::vec4 sharpen_params; // x = post-TAA unsharp strength, yzw spare
    // Brush indicator ring drawn on the terrain in compose.frag.
    // xy = brush world XZ, z = brush radius (m), w = enabled (0/1).
    // Reconstructs each pixel's world position from depth +
    // inv_view_proj so the ring projects onto whatever surface
    // is under the brush hit (no separate gizmo geometry).
    glm::vec4 brush_indicator;
    glm::mat4 inv_view_proj;
    // Physical sky: x = sky_mode (1 = atmosphere), y = intensity,
    // z = camera radius in planet space (km), w = AP max distance (km).
    glm::vec4 atmo;
    glm::vec4 tone;   // x = tonemap mode (0 ACES, 1 AgX, 2 PBR Neutral)
};

void VulkanEngine::draw(uint32_t img_index) {
    auto& frame = current_frame();
    auto dz_t = std::chrono::steady_clock::now();
    auto dz_mark = [&]() {
        auto n = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(n - dz_t).count();
        dz_t = n;
        return ms;
    };

    // Snapshot every dynamic body's world matrix + AABB ONCE for this frame.
    // rebuild_tlas (below) and render_world (in the rendering pass) both read
    // from this cache, so we avoid repeating O(n) Jolt queries.
    rebuild_dyn_render_cache();

    // Two frames in flight: no CPU wait for the previous frame here. Its
    // GPU work may still be running, so everything written below goes to
    // per-slot memory (upload ring, TLAS instance slot, CLOD / FX rings)
    // and the TLAS rebuild waits on the GPU for the previous graphics
    // submit (gfx_timeline_). See UploadRing in vk_engine.h.
    dz_mark();
    // QLIKE_SERIAL_FRAMES=1: old behaviour (CPU waits for the previous frame)
    // - A/B switch for the frames-in-flight path.
    static const bool serial_frames = std::getenv("QLIKE_SERIAL_FRAMES") != nullptr;
    if (serial_frames) wait_gpu_frames_idle();
    begin_frame_uploads();
    hitch_draw_[0] = dz_mark();

    // ---------- Compute queue: per-frame TLAS rebuild ----------
    submit_tlas_build(frame);
    hitch_draw_[1] = dz_mark();

    // ---------- Graphics command buffer ----------
    const uint32_t gpu_query_slot = frame_number_ % kFrameOverlap;
    begin_frame_commands(frame, gpu_query_slot);   // camera, uploads, culling, CLOD
    record_shadow_passes(frame);                   // sun shadow, depth pre-pass, half-rate shadow
    record_terrain_and_world(frame);               // low-res raymarch terrain + world G-buffer
    record_lighting_and_upscale(frame);            // deferred, SVGF, TAA, sky fill, DLSS/FSR/TAAU
    record_compose_and_ui(frame, img_index);       // bloom, compose, ImGui -> swapchain


    // Two screenshot triggers: the --screenshot CLI one-shot AND F12 in-game.
    std::string screenshot_path_this_frame;
    bool screenshot_is_oneshot = false;
    if (!pending_screenshot_path_.empty()) {
        screenshot_path_this_frame = pending_screenshot_path_;
        pending_screenshot_path_.clear();
    } else if (!opts_.screenshot_path.empty() && !screenshot_taken_ &&
               static_cast<int>(frame_number_) >= opts_.screenshot_after_frames) {
        if (opts_.screenshot_burst <= 1) {
            screenshot_path_this_frame = opts_.screenshot_path;
            screenshot_is_oneshot = true;
        } else {
            // Burst: PATH.NN.ppm for N consecutive frames, exit after the
            // last one (screenshot_is_oneshot latches screenshot_taken_).
            int idx = static_cast<int>(frame_number_) -
                      opts_.screenshot_after_frames;
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s.%02d.ppm",
                          opts_.screenshot_path.c_str(), idx);
            screenshot_path_this_frame = buf;
            screenshot_is_oneshot = (idx + 1 >= opts_.screenshot_burst);
        }
    }
    bool want_screenshot = !screenshot_path_this_frame.empty();
    if (want_screenshot) {
        capture_screenshot(frame.command_buffer, swapchain_images_[img_index],
                           swapchain_extent_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    vkinit::transition_image(frame.command_buffer, swapchain_images_[img_index],
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    // history image was just sampled by compose; leave it in SHADER_READ_ONLY
    // for next frame's TAA-pass read of this slot.

    if (gpu_query_pool_) {
        const uint32_t base = gpu_query_slot *
                              static_cast<uint32_t>(kQueriesPerFrame);
        vkCmdWriteTimestamp(frame.command_buffer,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            gpu_query_pool_,
                            base + static_cast<uint32_t>(kStageScene) * 2u + 1u);
        gpu_query_recorded_mask_ |= uint64_t{1} << gpu_query_slot;
    }
    vk_check(vkEndCommandBuffer(frame.command_buffer), "vkEndCommandBuffer");

    // Snapshot for next frame's reprojection.
    prev_view_proj_ = last_view_proj_;
    prev_view_proj_valid_ = true;
    // Snapshot the jitter we used this frame so the next frame can
    // compute (cur_jitter − prev_jitter) and emit unjittered motion.
    prev_jitter_ = current_frame_view_.jitter;
    history_write_slot_ = 1 - history_write_slot_;

    // Two wait semaphores on the graphics submit:
    //   1. swapchain image acquired (wait at COLOR_ATTACHMENT_OUTPUT — the
    //      compose pass writes to the swapchain image).
    //   2. TLAS build completed on the compute queue (wait at FRAGMENT_SHADER —
    //      cube.frag's ray queries dereference the TLAS during the world pass).
    VkSemaphoreSubmitInfo waits[2] = {
        vkinit::semaphore_submit_info(
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame.swapchain_semaphore),
        vkinit::semaphore_submit_info(
            // + COMPUTE: the froxel fog pass ray-queries the TLAS from compute.
            // + AS_BUILD: record_tlas_publish clones the stage TLAS at frame start.
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, frame.tlas_build_done),
    };
    // Signals: present semaphore + the frames-in-flight timeline (next
    // frame's TLAS rebuild waits for this value).
    VkSemaphoreSubmitInfo signals[2] = {
        vkinit::semaphore_submit_info(
            VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, render_semaphores_[img_index]),
        VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = gfx_timeline_,
            .value = gfx_timeline_value_ + 1,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        },
    };
    auto cmd_info = vkinit::command_buffer_submit_info(frame.command_buffer);
    VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr, .flags = 0,
        .waitSemaphoreInfoCount = 2,
        .pWaitSemaphoreInfos = waits,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = 2,
        .pSignalSemaphoreInfos = signals,
    };

    sl_marker(2);   // PCL render submit start
    vk_check(vkQueueSubmit2(graphics_queue_, 1, &submit, frame.render_fence),
             "vkQueueSubmit2");
    sl_marker(3);   // PCL render submit end
    ++gfx_timeline_value_;
    upload_ring_.in_frame = false;

    if (want_screenshot) {
        vk_check(vkWaitForFences(device_, 1, &frame.render_fence, VK_TRUE, UINT64_MAX),
                 "screenshot vkWaitForFences");
        write_ppm(screenshot_path_this_frame, swapchain_extent_.width, swapchain_extent_.height);
        if (screenshot_is_oneshot) screenshot_taken_ = true;
        log::infof("[screenshot] saved %s", screenshot_path_this_frame.c_str());
    }

    // Session 5 — push the latest HUDLessColor into the FG context's
    // pacer state right before present. The SDK uses this for the next
    // pair of (real, generated) frames.
    update_fsr3_fg_per_frame_config();

    // Route through FSR3 SwapChain proxy fn when active — that path
    // internally paces real + generated frames into the present chain.
    sl_marker(4);   // PCL present start
    anti_lag_mark(true);   // AMD Anti-Lag 2: present stage
    VkResult pres = queue_present_image(img_index, render_semaphores_[img_index]);
    sl_marker(5);   // PCL present end
    hitch_draw_[2] = dz_mark();
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        resize_requested_ = true;
    } else if (pres != VK_SUCCESS) {
        vk_check(pres, "vkQueuePresentKHR");
    }
}

// ---------------------------------------------------------------------
// draw() stages. Each records one group of passes into frame.command_buffer
// (submit_tlas_build: the compute-queue TLAS rebuild) in frame order.
// ---------------------------------------------------------------------

void VulkanEngine::submit_tlas_build(FrameData& frame) {
    // Recorded onto a separate compute command buffer and submitted on the
    // compute queue (or graphics queue if hardware lacks a dedicated compute
    // family). Signals frame.tlas_build_done; the graphics submit below
    // waits on it at fragment-shader stage so cube.frag's ray queries see
    // a built TLAS. On NVIDIA with a separate compute family this overlaps
    // with the previous frame's raster on the graphics queue.
    {
        auto begin_c = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        vk_check(vkBeginCommandBuffer(frame.compute_buffer, &begin_c),
                 "vkBeginCommandBuffer compute");
        rebuild_tlas(frame.compute_buffer);
        vk_check(vkEndCommandBuffer(frame.compute_buffer),
                 "vkEndCommandBuffer compute");

        VkCommandBufferSubmitInfo cb_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = frame.compute_buffer,
            .deviceMask = 0,
        };
        VkSemaphoreSubmitInfo signal_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = frame.tlas_build_done,
            .value = 0,
            // Ensure the AS-build write has completed before the graphics
            // queue's fragment shader reads the TLAS. The semaphore is
            // signaled at the AS-build stage; the graphics queue waits at
            // FRAGMENT_SHADER_BIT.
            .stageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .deviceIndex = 0,
        };
        VkSubmitInfo2 csubmit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr, .flags = 0,
            // No wait: the build targets tlas_stage_[slot], which only frame
            // N-2 (fence already waited) touched - it overlaps frame N-1.
            .waitSemaphoreInfoCount = 0, .pWaitSemaphoreInfos = nullptr,
            .commandBufferInfoCount = 1, .pCommandBufferInfos = &cb_info,
            .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal_info,
        };
        vk_check(vkQueueSubmit2(compute_queue_, 1, &csubmit, VK_NULL_HANDLE),
                 "vkQueueSubmit2 compute (TLAS build)");
    }
}

void VulkanEngine::begin_frame_commands(FrameData& frame, uint32_t gpu_query_slot) {
    auto begin = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vk_check(vkBeginCommandBuffer(frame.command_buffer, &begin), "vkBeginCommandBuffer");
    // GPU watchdog: reset this slot's queries and write the BEGIN
    // timestamp at TOP_OF_PIPE. The matching END timestamp goes in
    // before vkEndCommandBuffer below, into kStageScene's slot — the
    // pair brackets the entire frame so total GPU time can be diff'd
    // out post-fence-wait. (Other GpuStage slots stay zero — wire
    // per-stage breakdown later if we need to localise costs.)
    if (gpu_query_pool_) {
        const uint32_t base = gpu_query_slot *
                              static_cast<uint32_t>(kQueriesPerFrame);
        vkCmdResetQueryPool(frame.command_buffer, gpu_query_pool_, base,
                            static_cast<uint32_t>(kQueriesPerFrame));
        vkCmdWriteTimestamp(frame.command_buffer,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            gpu_query_pool_,
                            base + static_cast<uint32_t>(kStageScene) * 2u);
    }

    // Capture per-frame camera state once for all three render passes
    // (terrain raymarch LR, depth pre-pass, color pass). Replaces three
    // identical aspect/lerp/view/proj/Halton-jitter rebuilds per frame.
    prof_begin_frame(frame.command_buffer, gpu_query_slot);
    current_frame_view_ = compute_frame_view();
    // Scene UBO + per-pass UBOs: staged after this frame's camera so
    // camera_pos / jitter are current, then copied in one batch before any
    // pass reads them (frames in flight - see frame_upload.cpp).
    // Ground clutter follows the active terrain surface (raymarch FBM is
    // finer than the CPU heightmap) - no-op unless the mode was toggled.
    if (scatter_.rm_surface != rt_.terrain_raymarch_enabled) {
        wait_gpu_frames_idle();   // rewrites the live (single-buffered) clutter instances
        scatter_set_raymarch_surface(scatter_, rt_.terrain_raymarch_enabled);
    }
    update_scene_ubo();
    write_pass_ubos();
    flush_frame_uploads(frame.command_buffer);
    record_tlas_publish(frame.command_buffer);   // this frame's TLAS -> tlas_
    // GPU vegetation: cull + LOD-select every instance, fill indirect draws.
    cull_vegetation(frame.command_buffer);

    // Quadtree CLOD terrain: select nodes + fill this frame's instance
    // ranges (camera list + light list). Must run after
    // compute_frame_view (camera frustum) and update_sun_shadow_light_vp
    // (light frustum), before any pass records terrain draws — depth,
    // colour and shadow all consume the same selection.
    update_terrain_clod_instances();

}

void VulkanEngine::record_shadow_passes(FrameData& frame) {
    // Sun shadow map pass — runs first so its depth target is in
    // SHADER_READ_ONLY_OPTIMAL by the time the scene color pass samples
    // it via grass.vert (binding 7). Skip the entire pass when no
    // consumer wants the result this frame: grass paths off + water-
    // surface shadows off / water itself off. Image stays in
    // SHADER_READ_ONLY from the previous pass (the init transition
    // covers the very first frame), so the descriptor read returns
    // stale-but-valid depth that nothing samples anyway.
    bool sun_shadow_needed =
        rt_.grass_enabled || rt_.grass_raymarch_enabled ||
        (rt_.water_enabled && rt_.water_shadows_enabled) ||
        scatter_.instance_count > 0;   // scatter.vert samples it every frame
    if (sun_shadow_needed) {
        render_sun_shadow_pass(frame.command_buffer);
    }
    prof_mark(frame.command_buffer, "sun_shadow_map");
    render_clutter_shadow(frame.command_buffer);
    prof_mark(frame.command_buffer, "clutter_shadow");

    // Batch the world-pass attachment transitions into a single
    // vkCmdPipelineBarrier2 — was 3 separate barrier calls each
    // forcing a driver cache flush. (Perf P1.)
    {
        const vkinit::ImageTransition kWorldStart[] = {
            { scene_color_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            { motion_vec_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            { gbuffer0_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            { gbuffer1_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            { depth_image_, VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL },
        };
        vkinit::transition_images_batch(frame.command_buffer,
                                         kWorldStart, 5);
    }

    VkClearValue clear_color{};
    clear_color.color = { { 0.55f, 0.72f, 0.95f, 1.0f } };  // sky color matches UBO
    VkClearValue clear_depth{};
    clear_depth.depthStencil.depth = 1.0f;

    // ---------- Pass 0: depth pre-pass (no color attachment) ----------
    // Populates depth_image_ with the front-most surface so the color pass's
    // depth_compare=LESS_OR_EQUAL early-rejects occluded pixels before the
    // expensive cube.frag (inline RT, sun-shadow PCSS, GI bounce) runs. The
    // color pass below uses depth loadOp=LOAD instead of CLEAR.
    {
        VkRenderingAttachmentInfo prepass_depth{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = depth_view_,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clear_depth,
        };
        VkRenderingInfo prepass_ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            // Depth pre-pass renders into depth_image_ at render_extent_;
            // the user's render_scale controls how many fragments it covers.
            .renderArea = { {0, 0}, render_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 0, .pColorAttachments = nullptr,
            .pDepthAttachment = &prepass_depth, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &prepass_ri);
        render_world_depth_pass(frame.command_buffer);
        vkCmdEndRendering(frame.command_buffer);
    }
    prof_mark(frame.command_buffer, "depth_prepass");

    // Snapshot the prepass depth into a sampled image (binding 27). The
    // grass raymarcher draws INSIDE the colour pass, where the live depth
    // attachment cannot be sampled; with this copy it stops marching at
    // the nearest opaque surface instead of walking 96 steps through the
    // blade field behind castle walls. ~5 MB copy, negligible.
    if (depth_copy_image_) {
        VkCommandBuffer c = frame.command_buffer;
        const vkinit::ImageTransition pre[] = {
            { depth_image_, VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
            { depth_copy_image_, VK_IMAGE_ASPECT_DEPTH_BIT,
              depth_copy_layout_valid_ ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                       : VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL },
        };
        vkinit::transition_images_batch(c, pre, 2);
        VkImageCopy region{};
        region.srcSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
        region.dstSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
        region.extent = { render_extent_.width, render_extent_.height, 1 };
        vkCmdCopyImage(c, depth_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       depth_copy_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region);
        const vkinit::ImageTransition post[] = {
            { depth_image_, VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL },
            { depth_copy_image_, VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        };
        vkinit::transition_images_batch(c, post, 2);
        depth_copy_layout_valid_ = true;
    }

    // ---------- Pass 0.25: half-rate sun shadow producer ----------
    // Roadmap item #4 Phase 2 — when rt_.half_rate_shadows is on, fire
    // one RT shadow ray per half-res pixel into shadow_lr_image_ for
    // brushes + dyn props. cube.frag's Phase-3 bilateral consumer will
    // sample this instead of doing the inline 24/40-tap PCSS trace. The
    // entire pass is skipped (zero cost) when the toggle is off or the
    // shadow_strength slider is at 0.
    if (rt_.half_rate_shadows && shadow_lr_pipeline_ &&
        rt_.shadow_strength > 0.0f) {
        vkinit::transition_image(frame.command_buffer, shadow_lr_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkClearValue lr_clear{};
        lr_clear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkRenderingAttachmentInfo sh_att = vkinit::color_attachment_info(
            shadow_lr_view_, &lr_clear,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo sh_ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0, 0}, shadow_lr_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &sh_att,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &sh_ri);
        render_world_shadow_lr_pass(frame.command_buffer);
        vkCmdEndRendering(frame.command_buffer);
        vkinit::transition_image(frame.command_buffer, shadow_lr_image_,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    prof_mark(frame.command_buffer, "shadow_lr");
}

void VulkanEngine::record_terrain_and_world(FrameData& frame) {
    VkClearValue clear_color{};
    clear_color.color = { { 0.55f, 0.72f, 0.95f, 1.0f } };  // sky color matches UBO
    VkClearValue clear_depth{};
    clear_depth.depthStencil.depth = 1.0f;
    // ---------- Pass 0.5: low-res raymarch terrain (when scale < 1) ----------
    // Renders the FBM raymarch into tr_lr_* attachments at a fraction of
    // the main resolution. Compose pass at the start of the world color
    // pass below upscales it into scene_color/depth with depth-aware
    // composite so cube/castle/dyn-props (drawn afterwards) layer on top.
    if (tr_use_lowres() && terrain_raymarch_pipeline_) {
        vkinit::transition_image(frame.command_buffer, tr_lr_color_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        vkinit::transition_image(frame.command_buffer, tr_lr_motion_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        vkinit::transition_image_aspect(frame.command_buffer, tr_lr_depth_image_,
                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_ASPECT_DEPTH_BIT);

        VkClearValue lr_clear_c{};
        lr_clear_c.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkClearValue lr_clear_m{};
        lr_clear_m.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkClearValue lr_clear_d{};
        lr_clear_d.depthStencil.depth = 1.0f;

        VkRenderingAttachmentInfo lr_color_att =
            vkinit::color_attachment_info(tr_lr_color_view_, &lr_clear_c,
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo lr_motion_att =
            vkinit::color_attachment_info(tr_lr_motion_view_, &lr_clear_m,
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo lr_color_atts[2] = { lr_color_att, lr_motion_att };
        VkRenderingAttachmentInfo lr_depth_att{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = tr_lr_depth_view_,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = lr_clear_d,
        };
        // Optional attachment-based VRS — chained via pNext. The texel
        // size + shading-rate combiner are pipeline state; the attachment
        // image just supplies the per-tile rate.
        VkRenderingFragmentShadingRateAttachmentInfoKHR vrs_info{};
        if (vrs_supported_ && vrs_view_) {
            vrs_info.sType = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR;
            vrs_info.imageView = vrs_view_;
            vrs_info.imageLayout = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
            vrs_info.shadingRateAttachmentTexelSize = vrs_texel_size_;
        }
        VkRenderingInfo lr_ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            // No VRS here: the rate image maps full-res pixels, so on the
            // low-res target it put coarse shading in the wrong places.
            .pNext = nullptr,
            .flags = 0,
            .renderArea = { {0, 0}, tr_lr_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 2, .pColorAttachments = lr_color_atts,
            .pDepthAttachment = &lr_depth_att, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &lr_ri);

        VkViewport lr_vp{};
        lr_vp.x = 0.0f; lr_vp.y = 0.0f;
        lr_vp.width  = static_cast<float>(tr_lr_extent_.width);
        lr_vp.height = static_cast<float>(tr_lr_extent_.height);
        lr_vp.minDepth = 0.0f; lr_vp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.command_buffer, 0, 1, &lr_vp);
        VkRect2D lr_sc{ {0, 0}, tr_lr_extent_ };
        vkCmdSetScissor(frame.command_buffer, 0, 1, &lr_sc);

        render_terrain_raymarch_lr(frame.command_buffer);

        vkCmdEndRendering(frame.command_buffer);

        // Transition all three to SHADER_READ_ONLY for the compose pass.
        vkinit::transition_image(frame.command_buffer, tr_lr_color_image_,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkinit::transition_image(frame.command_buffer, tr_lr_motion_image_,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkinit::transition_image_aspect(frame.command_buffer, tr_lr_depth_image_,
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    prof_mark(frame.command_buffer, "terrain_raymarch_lr");

    auto color_att = vkinit::color_attachment_info(scene_color_view_,
                                                   &clear_color,
                                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    // Motion-vector attachment: cleared to (0, 0). cube.frag writes its
    // location=1 output for every brush/dyn fragment that survives the
    // depth test. Sky pixels (no fragment runs) keep the cleared 0; TAA's
    // existing "outside [0,1]" guard handles the resulting prev_uv branch.
    VkClearValue clear_motion{};
    clear_motion.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    auto motion_att = vkinit::color_attachment_info(motion_vec_view_,
                                                    &clear_motion,
                                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    // G-buffer attachments (Phase 2 of deferred migration). Cleared to
    // (0, 0, 0, 0). cube.frag dual-writes albedo+material to gbuffer0 and
    // octa-normal+roughness+metallic to gbuffer1; other pipelines drawing
    // in this scope (water, voxel, terrain, etc) don't write G-buffer
    // and the attachments keep cleared / previously-written values.
    VkClearValue clear_gbuf{};
    clear_gbuf.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    auto gbuffer0_att = vkinit::color_attachment_info(gbuffer0_view_,
                                                     &clear_gbuf,
                                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    auto gbuffer1_att = vkinit::color_attachment_info(gbuffer1_view_,
                                                     &clear_gbuf,
                                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo color_atts[4] = { color_att, motion_att,
                                                gbuffer0_att, gbuffer1_att };

    VkRenderingAttachmentInfo depth_att{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = depth_view_,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        // LOAD (not CLEAR): depth was just written by the prepass above. The
        // color pipeline uses LESS_OR_EQUAL, so brushes/dyn props will pass
        // exactly at the depth value the prepass wrote, while particles,
        // projectiles and the viewmodel (which skipped the prepass) still
        // depth-test against existing world depth correctly.
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear_depth,
    };
    VkRenderingInfo rendering{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        // Pass 1 outputs scene_color + motion_vec at render_extent_.
        .renderArea = { {0, 0}, render_extent_ },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 4, .pColorAttachments = color_atts,
        .pDepthAttachment = &depth_att, .pStencilAttachment = nullptr,
    };

    // ---------- Pass 1: scene → scene_color_image_ + motion_vec_image_ ----------
    // (svgf_gi_image_ is in GENERAL — transitioned once at init via
    // one_time_submit in init_svgf_targets, survives recreate cleanly.)
    vkCmdBeginRendering(frame.command_buffer, &rendering);
    // When LR upscaling is active, run the compose first so the
    // upscaled raymarch terrain populates scene_color/motion/depth
    // before the rasterised geometry draws on top.
    if (tr_use_lowres() && tr_compose_pipeline_ != VK_NULL_HANDLE) {
        VkViewport vp{};
        vp.x = 0.0f; vp.y = 0.0f;
        vp.width  = static_cast<float>(render_extent_.width);
        vp.height = static_cast<float>(render_extent_.height);
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.command_buffer, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, render_extent_ };
        vkCmdSetScissor(frame.command_buffer, 0, 1, &sc);
        render_terrain_raymarch_compose(frame.command_buffer);
    }
    render_world(frame.command_buffer);
    vkCmdEndRendering(frame.command_buffer);
    prof_mark(frame.command_buffer, "world_tail");
}

void VulkanEngine::record_lighting_and_upscale(FrameData& frame) {
    // ---------- Pass 2: TAA into history_image_[history_write_slot_] ----------
    // Flip scene_color + depth into shader-read for TAA in one batched
    // barrier (was 2 calls + a dead motion_vec one). The motion_vec
    // image isn't sampled by anyone right now — dropping its transition
    // saves a real cross-pass cache flush on the 16 MB RG16F surface.
    // Perf P1+P2.
    {
        const vkinit::ImageTransition kPostWorld[] = {
            { scene_color_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { motion_vec_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { gbuffer0_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { gbuffer1_image_, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { depth_image_, VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        };
        vkinit::transition_images_batch(frame.command_buffer, kPostWorld, 5);
        // cube.frag imageStore()s the raw GI (svgf_gi_image_, GENERAL) that
        // deferred_lighting samples next: make those writes visible.
        {
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT };
            VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(frame.command_buffer, &di);
        }
    }

    // ---------- Pass 1d: Deferred lighting (Phase 3 of migration) ----------
    // Reads gbuffer0/1 + depth + sun_shadow and writes lit colour to
    // staging_color_image_. Skipped when deferred_lighting_active_ is
    // false (default until Phase 4 promotes this to the canonical scene
    // colour writer).
    render_deferred_lighting(frame.command_buffer);
    prof_mark(frame.command_buffer, "deferred_lighting");
    dispatch_radiance_cache(frame.command_buffer);
    prof_mark(frame.command_buffer, "rcache");
    // When the toggle is on, blit the deferred output back into
    // scene_color so the downstream chain (SVGF, TAA, FSR3, compose)
    // sees the deferred result without descriptor surgery. scene_color
    // is currently in SHADER_READ_ONLY (post-world transition); the
    // blit needs it in TRANSFER_DST, then back to SHADER_READ_ONLY.
    if (deferred_lighting_active_) {
        vkinit::transition_image(frame.command_buffer, scene_color_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.srcOffsets[1] = { (int32_t)render_extent_.width,
                                (int32_t)render_extent_.height, 1 };
        blit.dstOffsets[1] = { (int32_t)render_extent_.width,
                                (int32_t)render_extent_.height, 1 };
        vkCmdBlitImage(frame.command_buffer,
                        staging_color_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        scene_color_image_,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_NEAREST);
        vkinit::transition_image(frame.command_buffer, scene_color_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // ---------- Pass 1.5: SVGF deep denoise (4a-deep) ----------
    // Updates the per-pixel luminance moments (binding 22/23) by
    // reprojecting last frame's moments through motion_vec, then runs
    // a 3-pass a-trous filter with edge-stop weights derived from the
    // resulting per-pixel variance plus depth + normal-from-depth. The
    // final pass writes its filtered colour BACK into scene_color,
    // so TAA (below) sees a denoised input and its own single-pass
    // a-trous compounds rather than competes. No-op when
    // rt_.svgf_enabled is false (per-frame branch -- pipelines stand
    // up unconditionally so the toggle is allocation-free).
    prof_mark(frame.command_buffer, "deferred_blit");
    // FX: explosions (volumetric), contrails, laser beams - blended into
    // the lit HDR scene before SVGF/TAA/bloom.
    render_fx(frame.command_buffer);
    prof_mark(frame.command_buffer, "fx");
    render_svgf(frame.command_buffer, static_cast<uint32_t>(frame_number_));
    prof_mark(frame.command_buffer, "svgf");
    // Both history slots end the previous frame in SHADER_READ_ONLY (write slot
    // exited at frame end; read slot wasn't touched). Read slot needs no
    // transition; write slot moves into COLOR_ATTACHMENT.
    vkinit::transition_image(frame.command_buffer, history_image_[history_write_slot_],
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // (TAA UBO: staged in write_pass_ubos() before the frame upload flush.)

    // Render TAA pass to the chosen history slot.
    {
        auto taa_color = vkinit::color_attachment_info(
            history_view_[history_write_slot_], nullptr,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            // History images live at render_extent_, so TAA renders at that.
            .renderArea = { {0, 0}, render_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &taa_color,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &ri);

        VkViewport vp{};
        vp.width = static_cast<float>(render_extent_.width);
        vp.height = static_cast<float>(render_extent_.height);
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.command_buffer, 0, 1, &vp);
        VkRect2D sc{ {0,0}, render_extent_ };
        vkCmdSetScissor(frame.command_buffer, 0, 1, &sc);

        vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, taa_pipeline_);
        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                taa_pipeline_layout_, 0, 1,
                                &taa_desc_sets_[history_write_slot_], 0, nullptr);
        vkCmdDraw(frame.command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(frame.command_buffer);
    }

    vkinit::transition_image(frame.command_buffer, history_image_[history_write_slot_],
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    prof_mark(frame.command_buffer, "taa");

    // ---------- Pass 2.15: sky fill (DLSS path) ----------
    // Put the real sky (atmosphere + clouds) into the render-res scene
    // colour BEFORE DLSS. Sky coverage lives in the colour's alpha, written
    // at the same jittered positions as the colour, so the blend
    // (colour += (sky - placeholder) * dst.alpha) is exact per pixel. DLSS
    // then upscales a finished image; compose no longer rebuilds the sky
    // at silhouettes from a mismatched depth / upscaled alpha (dark rims).
    atmo_dispatched_this_frame_ = false;
    if (rt_.fsr2_enabled && rt_.fsr_backend == 2 && dlss_active() &&
        deferred_lighting_active_ && sky_fill_pipeline_ != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = frame.command_buffer;
        if (rt_.sky_mode == 1) {
            dispatch_atmosphere(cmd);
            atmo_dispatched_this_frame_ = true;
        }
        vkinit::transition_image(cmd, scene_color_image_,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        auto sc_att = vkinit::color_attachment_info(scene_color_view_, nullptr,
                                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0, 0}, render_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &sc_att,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{ 0.0f, 0.0f, float(render_extent_.width), float(render_extent_.height), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, render_extent_ };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_fill_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compose_pipeline_layout_,
                                0, 1, &compose_desc_sets_[history_write_slot_], 0, nullptr);
        ComposePC sp{};
        const float rw = float(render_extent_.width), rh = float(render_extent_.height);
        sp.viewport = glm::vec4(rw, rh, 1.0f / rw, 1.0f / rh);
        {
            float pitch = glm::radians(rt_.sun_pitch_deg), yaw = glm::radians(rt_.sun_yaw_deg);
            glm::vec3 sun = glm::normalize(glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch),
                                                     std::cos(pitch) * std::cos(yaw)));
            sp.sun_dir = glm::vec4(sun, rt_.sun_shaft_intensity);
        }
        sp.sun_color = glm::vec4(rt_.sun_color, rt_.sun_intensity * rt_.sun_glare_strength);
        sp.sky_color = glm::vec4(last_sky_rgb_, 1.0f);
        sp.flare_params2 = glm::vec4(0.0f, 0.0f, 0.0f, skybox_.hdr ? 1.0f : 0.0f);
        sp.inv_view_proj = last_inv_view_proj_;
        sp.atmo = glm::vec4((rt_.sky_mode == 1 && atmo_pipeline_) ? 1.0f : 0.0f,
                            std::max(0.0f, rt_.atmosphere_intensity),
                            6360.0f + std::max(current_frame_view_.eye_pos.y, 1.0f) * 0.001f,
                            std::max(0.5f, rt_.aerial_max_km));
        vkCmdPushConstants(cmd, compose_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(sp), &sp);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        vkinit::transition_image(cmd, scene_color_image_,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        prof_mark(cmd, "sky_fill");
    }

    // ---------- Pass 2.2: FSR2 dispatch (optional, replaces TAAU) ----------
    // When FSR2 is on, the SDK does its own temporal accumulation +
    // upscale from raw scene_color/depth/motion (NOT from the TAA history,
    // which is already temporally blended). TAA still runs each frame
    // because compose falls back to it when fsr2 toggles off; the per-
    // frame cost is negligible vs the world pass and avoids a one-frame
    // black flash on toggle.
    if (rt_.fsr2_enabled) {
        if (rt_.fsr_backend == 2 && dlss_active()) {   // else: FSR2 fallback below
            dispatch_dlss(frame.command_buffer);   // NVIDIA DLSS (Streamline)
        } else
        if (rt_.fsr_backend == 1 && !fsr3_fatal_) {
            // Lazy init on first frame the FSR3 backend is selected.
            // SEH-guarded: if init AVs in the SDK, fsr3_fatal_ flips
            // and the next branch falls back to FSR2.
            if (!fsr3_context_valid_) init_fsr3();
            if (fsr3_context_valid_) {
                dispatch_fsr3(frame.command_buffer);
                // Session 6 — rolling render-dt avg for low-FPS
                // auto-disable. last_frame_dt_ excludes FG-paced
                // present cadence (it's the engine's render-loop dt).
                fg_dt_avg_ms_ = fg_dt_avg_ms_ * 0.95f +
                                last_frame_dt_ * 1000.0f * 0.05f;
                const float kDisableMs = 20.0f;   // 50 fps base
                const float kEnableMs  = 16.6f;   // 60 fps base (hysteresis)
                if (!fg_runtime_disabled_ && fg_dt_avg_ms_ > kDisableMs) {
                    if (++fg_dt_below_count_ > 60) {
                        fg_runtime_disabled_ = true;
                        log::infof("[fsr3 fg] auto-disabled — render dt %.1fms "
                                   "(<50 fps); FG quality degrades below this",
                                   fg_dt_avg_ms_);
                    }
                } else if (fg_runtime_disabled_ && fg_dt_avg_ms_ < kEnableMs) {
                    fg_runtime_disabled_ = false;
                    fg_dt_below_count_   = 0;
                    log::info("[fsr3 fg] auto-re-enabled — render dt back above 60 fps");
                } else if (fg_dt_avg_ms_ < kDisableMs) {
                    fg_dt_below_count_ = 0;
                }
                // FG is on iff user opted in AND we're not auto-disabled.
                const bool fg_active = rt_.fg_enabled && !fsr3_fg_fatal_ &&
                                       !fg_runtime_disabled_;
                if (fg_active) {
                    if (!fsr3_fg_context_valid_) init_fsr3_fg();
                    if (fsr3_fg_context_valid_) {
                        // Lazy-init the SwapChain proxy so generated frames
                        // actually reach the display via the SDK's pacer.
                        // NOT here: replacing the swapchain mid-draw left this
                        // frame recording into / presenting an image of the old
                        // one. run() does it before the next acquire.
                        if (!fsr3_swapchain_active_ && !fsr3_swapchain_fatal_) {
                            fsr3_swapchain_init_pending_ = true;
                        }
                    }
                }
                // (FG-off teardown handled by the universal gate above.)
            } else if (fsr3_fatal_ && fsr2_context_valid_) {
                rt_.fsr_backend = 0;     // permanently revert UI to FSR2
                dispatch_fsr2(frame.command_buffer);
            }
        } else if (fsr2_context_valid_) {
            dispatch_fsr2(frame.command_buffer);
        }
    }

    // (DLSS on/off + output (re)creation: ensure_dlss_bindings(), before recording.)

    // ---------- Pass 2.25: TAAU temporal upsample (optional) ----------
    // Reads LR history (just written) + LR motion + LR depth + previous-
    // frame native upscaled. Outputs new native upscaled. Compose then
    // samples taau_view_ instead of history_view_ (set by
    // rewrite_compose_image_bindings via the toggle handler).
    if (rt_.taau_enabled && taau_pipeline_ != VK_NULL_HANDLE) {

        vkinit::transition_image(frame.command_buffer,
                                  taau_image_[history_write_slot_],
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        auto taau_color = vkinit::color_attachment_info(
            taau_view_[history_write_slot_], nullptr,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo taau_ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            // Output at NATIVE — this is the upscale step.
            .renderArea = { {0, 0}, swapchain_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &taau_color,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &taau_ri);
        VkViewport tvp{};
        tvp.width = static_cast<float>(swapchain_extent_.width);
        tvp.height = static_cast<float>(swapchain_extent_.height);
        tvp.minDepth = 0.0f; tvp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.command_buffer, 0, 1, &tvp);
        VkRect2D tsc{ {0, 0}, swapchain_extent_ };
        vkCmdSetScissor(frame.command_buffer, 0, 1, &tsc);
        vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          taau_pipeline_);
        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                taau_pipeline_layout_, 0, 1,
                                &taau_desc_sets_[history_write_slot_], 0, nullptr);
        vkCmdDraw(frame.command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(frame.command_buffer);

        vkinit::transition_image(frame.command_buffer,
                                  taau_image_[history_write_slot_],
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        taau_history_valid_ = true;
    } else {
        // Mode flipped off — invalidate history so toggling back on
        // doesn't reproject stale frames.
        taau_history_valid_ = false;
    }

    prof_mark(frame.command_buffer, "fsr_taau");
}

void VulkanEngine::record_compose_and_ui(FrameData& frame, uint32_t img_index) {
    // ---------- Copy denoised history → swapchain ----------
    // ---------- Pass 2.5: Compose history (HDR) into swapchain (LDR) with ACES tonemap ----------

    // Bloom mip chain reads the now-SHADER_READ history image (mip 0 input)
    // and writes mips 0..N down then back up additively. Final state leaves
    // mip 0 in SHADER_READ_ONLY which is what compose's binding 3 expects.
    // Skipped internally if rt_.bloom_enabled is false, but compose still
    // samples the texture — the previous frame's bloom mip 0 stays bound,
    // which is fine because compose multiplies by bloom_params.x and the
    // strength slider is the user-facing toggle.
    run_bloom_chain(frame.command_buffer);
    prof_mark(frame.command_buffer, "bloom");
    // src stage = COLOR_ATTACHMENT_OUTPUT (NOT the default TOP_OF_PIPE
    // for UNDEFINED) so this layout write chains with the acquire
    // semaphore wait (also COLOR_ATTACHMENT_OUTPUT, see the submit's
    // waits[] below). Without this the presentation engine's read of
    // the swapchain image isn't ordered before this write —
    // sync-validation flagged it as a WRITE_AFTER_READ hazard and it
    // is a real intermittent device-lost risk under load.
    // Auto-exposure: reduce the final HDR image to an adapted log-luminance
    // (compose binding 4) before the compose pass reads it.
    // Physical sky LUTs + aerial perspective for compose (sky_mode 1).
    if (rt_.sky_mode == 1 && !atmo_dispatched_this_frame_) dispatch_atmosphere(frame.command_buffer);
    if (rt_.fog_volumetric && fog_pipeline_) dispatch_fog(frame.command_buffer);
    prof_mark(frame.command_buffer, "sky_clouds");
    dispatch_exposure(frame.command_buffer);
    vkinit::transition_image_src_stage(
        frame.command_buffer, swapchain_images_[img_index],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    {
        auto sw_color = vkinit::color_attachment_info(swapchain_views_[img_index], nullptr,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0, 0}, swapchain_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &sw_color,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &ri);
        VkViewport vp{};
        vp.width = static_cast<float>(swapchain_extent_.width);
        vp.height = static_cast<float>(swapchain_extent_.height);
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.command_buffer, 0, 1, &vp);
        VkRect2D sc{ {0,0}, swapchain_extent_ };
        vkCmdSetScissor(frame.command_buffer, 0, 1, &sc);
        vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, compose_pipeline_);
        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                compose_pipeline_layout_, 0, 1,
                                &compose_desc_sets_[history_write_slot_], 0, nullptr);
        ComposePC pc_data{};
        float w = static_cast<float>(swapchain_extent_.width);
        float h = static_cast<float>(swapchain_extent_.height);
        pc_data.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
        {
            // The push-constant block is full (256 B), so the colour-grading
            // controls ride in viewport.xy (unused by the shader): four 6-bit
            // values per float.
            auto q6 = [](float v01) { return std::floor(glm::clamp(v01, 0.0f, 1.0f) * 62.0f + 0.5f); };   // 31 = exact middle
            auto pack4 = [](float a, float b, float c, float d) { return a + 64.0f * (b + 64.0f * (c + 64.0f * d)); };
            pc_data.viewport.x = pack4(q6(rt_.grade_temperature * 0.5f + 0.5f), q6(rt_.grade_saturation * 0.5f),
                                       q6(rt_.grade_vibrance), float(frame_number_ % 64u));
            pc_data.viewport.y = pack4(q6(rt_.grade_split_tone), q6(rt_.film_grain), q6(rt_.vignette_strength), 0.0f);
        }
        pc_data.bloom_params = glm::vec4(rt_.bloom_strength,
                                         rt_.bloom_threshold,
                                         rt_.bloom_radius,
                                         rt_.bloom_enabled ? 1.0f : 0.0f);
        float pitch = glm::radians(rt_.sun_pitch_deg);
        float yaw   = glm::radians(rt_.sun_yaw_deg);
        glm::vec3 sun(std::cos(pitch) * std::sin(yaw),
                      std::sin(pitch),
                      std::cos(pitch) * std::cos(yaw));
        sun = glm::normalize(sun);
        // .w carries sun_shaft_intensity for compose.frag's screen-space
        // crepuscular-ray accumulation. Free slot — the shader treats
        // sun_dir as a normalized direction (xyz) regardless.
        pc_data.sun_dir   = glm::vec4(sun, rt_.sun_shaft_intensity);
        // .a scaled by sun_glare_strength so the user can dial down
        // every sun-derived compose-pass effect together: procedural
        // disc + halo, lens flare, AND sun shafts (compose.frag reads
        // pc.sun_color.a for the shaft tint too). Scene direct
        // lighting comes from scene.sun_color in the UBO and is
        // unaffected — the slider only changes the "look-at-the-sun"
        // brightness, not how much the sun illuminates the world.
        pc_data.sun_color = glm::vec4(rt_.sun_color,
                                      rt_.sun_intensity * rt_.sun_glare_strength);
        // Sky colour deferred wrote into clear sky pixels (w = 1 when the
        // deferred pass ran); compose subtracts it before adding the sky.
        pc_data.sky_color = glm::vec4(last_sky_rgb_, deferred_lighting_active_ ? 1.0f : 0.0f);
        pc_data.flare_params = glm::vec4(rt_.lens_flare_strength,
                                          rt_.lens_flare_threshold,
                                          rt_.lens_flare_dispersal,
                                          rt_.lens_flare_halo_width);
        pc_data.flare_params2 = glm::vec4(static_cast<float>(rt_.lens_flare_ghosts),
                                           rt_.lens_flare_aberration,
                                           rt_.lens_flare_enabled ? 1.0f : 0.0f,
                                           skybox_.hdr ? 1.0f : 0.0f);   // .w: sky is HDR
        // Project the sun direction (treated as a directional light at infinity
        // — w = 0 in homogeneous coords) to NDC to find its screen-space UV.
        // The lens-flare shader uses this to gate ghost contributions to the
        // sun only — emissive lanterns and other bright pixels are rejected
        // by angular distance regardless of their HDR luminance. sun_screen.z
        // = 1 when sun is in front of camera (clip.w > 0), else 0.
        {
            glm::vec4 sun_clip = last_view_proj_ * glm::vec4(sun, 0.0f);
            if (sun_clip.w > 0.0f) {
                glm::vec2 sun_ndc(sun_clip.x / sun_clip.w, sun_clip.y / sun_clip.w);
                glm::vec2 sun_uv = sun_ndc * 0.5f + 0.5f;
                pc_data.sun_screen = glm::vec4(sun_uv.x, sun_uv.y, 1.0f,
                                                rt_.image_gamma);
            } else {
                pc_data.sun_screen = glm::vec4(0.0f, 0.0f, 0.0f,
                                                rt_.image_gamma);
            }
        }
        // Post-TAA unsharp strength. 0.55 is a reasonable default that
        // recovers most of the detail taa.frag's à-trous spatial filter
        // blurs away (default taa_spatial_strength can be 0.4–0.5). User
        // can override via `compose_sharpen_strength` in qlike_settings.cfg.
        // sharpen_params.y carries auto_exposure_strength so compose.frag
        // can scale pre-tonemap HDR by (target / scene_avg). 0 = off.
        pc_data.sharpen_params = glm::vec4(rt_.compose_sharpen_strength,
                                            rt_.auto_exposure_strength,
                                            rt_.image_contrast,
                                            rt_.image_brightness);
        // Brush ring: enabled whenever the user is in edit mode AND the
        // crosshair raycast hit the terrain this frame.
        if (terrain_edit_mode_ && terrain_brush_has_hit_) {
            pc_data.brush_indicator = glm::vec4(terrain_brush_world_pos_.x,
                                                 terrain_brush_world_pos_.z,
                                                 terrain_brush_radius_,
                                                 1.0f);
        } else {
            pc_data.brush_indicator = glm::vec4(0.0f);
        }
        pc_data.inv_view_proj = last_inv_view_proj_;   // cached
        // tone.x: tonemap mode, +8 = colour is DLSS output whose alpha
        // carries exact sky coverage (deferred writes geometry alpha 0).
        pc_data.tone = glm::vec4(float(rt_.tonemap_mode) +
                                     ((compose_uses_dlss_ && deferred_lighting_active_) ? 8.0f : 0.0f),
                                 (rt_.fog_volumetric && fog_pipeline_) ? std::max(20.0f, rt_.fog_max_dist) : 0.0f,
                                 hdr_active_ ? std::max(80.0f, rt_.hdr_paper_nits) : 0.0f,
                                 std::max(rt_.hdr_peak_nits, rt_.hdr_paper_nits));
        pc_data.atmo = glm::vec4((rt_.sky_mode == 1 && atmo_pipeline_) ? 1.0f : 0.0f,
                                 std::max(0.0f, rt_.atmosphere_intensity),
                                 6360.0f + std::max(current_frame_view_.eye_pos.y, 1.0f) * 0.001f,
                                 std::max(0.5f, rt_.aerial_max_km));
        vkCmdPushConstants(frame.command_buffer, compose_pipeline_layout_,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc_data), &pc_data);
        vkCmdDraw(frame.command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(frame.command_buffer);
    }

    // ---------- Session 5: snapshot hudless backbuffer for FG ----------
    // Captures the post-compose, pre-ImGui swapchain to a private image
    // the SDK uses for HUD extraction. No-op when FSR3 SwapChain proxy
    // isn't active.
    prof_mark(frame.command_buffer, "compose");
    capture_hudless_snapshot(frame.command_buffer, img_index);
    dlssg_capture_hudless(frame.command_buffer, img_index);

    // ---------- Pass 3: ImGui directly on swapchain (still in COLOR_ATTACHMENT) ----------

    if (imgui_initialized_) {
        auto sw_color = vkinit::color_attachment_info(swapchain_views_[img_index], nullptr,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0, 0}, swapchain_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &sw_color,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.command_buffer, &ri);
        ImDrawData* dd = ImGui::GetDrawData();
        if (dd && hdr_active_) {
            // HDR10 swapchain: ImGui vertex colours are sRGB code values;
            // re-encode them as PQ at paper-white so the UI keeps its SDR
            // brightness instead of being read as up-to-10000-nit PQ.
            static uint8_t lut[256];
            static float lut_paper = -1.0f;
            const float paper = std::max(80.0f, rt_.hdr_paper_nits);
            if (lut_paper != paper) {
                lut_paper = paper;
                for (int i = 0; i < 256; ++i) {
                    float c = i / 255.0f;
                    float lin = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
                    float lp = std::pow(lin * paper / 10000.0f, 0.1593017578125f);
                    float pq = std::pow((0.8359375f + 18.8515625f * lp) / (1.0f + 18.6875f * lp), 78.84375f);
                    lut[i] = static_cast<uint8_t>(std::clamp(pq * 255.0f + 0.5f, 0.0f, 255.0f));
                }
            }
            for (int l = 0; l < dd->CmdListsCount; ++l) {
                for (ImDrawVert& v : dd->CmdLists[l]->VtxBuffer) {
                    ImU32 c = v.col;
                    v.col = (c & 0xFF000000u) |
                            (ImU32(lut[(c >> 16) & 0xFF]) << 16) |
                            (ImU32(lut[(c >> 8) & 0xFF]) << 8) |
                             ImU32(lut[c & 0xFF]);
                }
            }
        }
        if (dd) ImGui_ImplVulkan_RenderDrawData(dd, frame.command_buffer);
        vkCmdEndRendering(frame.command_buffer);
    }

    prof_mark(frame.command_buffer, "imgui");
}


// Per-pass UBOs (TAA, TAAU, SVGF, voxel camera): their inputs are all known
// at frame start, so they are staged here, before flush_frame_uploads().
void VulkanEngine::write_pass_ubos() {
    // Update TAA UBO. inv_vp/prev_vp dropped after svgf-tao-rewrite —
    // the motion-vec image (binding 4) provides the reprojection now,
    // so TAA only needs viewport metrics + the history-blend params.
    {
        struct TaaUBO {
            glm::vec4 viewport;
            glm::vec4 params;
            glm::mat4 inv_vp;    // current frame
            glm::mat4 prev_vp;   // previous frame (sky reprojection)
            glm::vec4 eye;
        } u{};
        u.inv_vp  = current_frame_view_.inv_vp;
        u.prev_vp = prev_view_proj_valid_ ? prev_view_proj_ : current_frame_view_.vp;
        u.eye     = glm::vec4(current_frame_view_.eye_pos, 1.0f);
        // TAA reads + writes at render_extent_, so the viewport vec it uses
        // for `gl_FragCoord → uv` conversion must reflect that.
        float w = static_cast<float>(render_extent_.width);
        float h = static_cast<float>(render_extent_.height);
        u.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
        // When SVGF is on, its 3-pass a-trous already filtered the
        // scene_color we're about to TAA-blend; running TAA's own
        // single-pass a-trous on top of that is just double-blurring.
        // Zero out the spatial strength so TAA degrades to temporal-
        // only (history blend + variance clip). When SVGF is off,
        // taa_spatial_strength keeps its slider value -- the existing
        // single-pass a-trous covers the spatial reconstruction.
        const float spatial = rt_.svgf_enabled ? 0.0f
                                               : rt_.taa_spatial_strength;
        u.params = glm::vec4(rt_.taa_history_blend,
                             0.05f,
                             prev_view_proj_valid_ ? 1.0f : 0.0f,
                             spatial);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, taa_ubo_alloc_, &ai);
        upload(taa_ubo_buffer_, 0, &u, sizeof(u), ai.pMappedData);
    }
    if (rt_.taau_enabled && taau_pipeline_ != VK_NULL_HANDLE) {
        // Update UBO each frame — LR + native viewport sizes, params.
        struct TaauUBO { glm::vec4 lr_vp; glm::vec4 full_vp; glm::vec4 params; };
        TaauUBO u{};
        u.lr_vp = glm::vec4(static_cast<float>(render_extent_.width),
                            static_cast<float>(render_extent_.height),
                            1.0f / render_extent_.width,
                            1.0f / render_extent_.height);
        u.full_vp = glm::vec4(static_cast<float>(swapchain_extent_.width),
                              static_cast<float>(swapchain_extent_.height),
                              1.0f / swapchain_extent_.width,
                              1.0f / swapchain_extent_.height);
        u.params = glm::vec4(rt_.taa_history_blend,
                             taau_history_valid_ ? 1.0f : 0.0f,
                             0.0f, 0.0f);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, taau_ubo_alloc_, &ai);
        upload(taau_ubo_buffer_, 0, &u, sizeof(u), ai.pMappedData);
    }
    write_svgf_ubos();
    update_voxel_camera_ubo();
}
} // namespace qlike
