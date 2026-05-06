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
    audio_->load_clip("shot",     "assets/sounds/shot.ogg");
    audio_->load_clip("impact",   "assets/sounds/impact.ogg");
    audio_->load_clip("jump",     "assets/sounds/jump.ogg");
    audio_->load_clip("land",     "assets/sounds/land.ogg");
    init_vulkan();
    init_pipeline_cache();
    init_swapchain();
    init_depth_image();
    init_commands();
    init_sync();
    init_readback_buffer();
    init_world();
    init_skybox();
    init_textures();
    load_settings();
    init_descriptors();
    init_rt();
    {
        VkImageView alb[kTextureCount]{}, nrm[kTextureCount]{};
        for (int i = 0; i < kTextureCount; ++i) {
            alb[i] = albedo_textures_[i].view;
            nrm[i] = normal_textures_[i].view;
        }
        write_scene_descriptors_once(device_, scene_desc_set_,
                                     scene_ubo_buffer_, tlas_, materials_buffer_,
                                     prev_transforms_buffer_,
                                     alb, nrm, kTextureCount, texture_sampler_);
    }
    init_pipeline();
    init_grass_pipeline();
    // Heightmap shadow texture must be baked AFTER descriptors and the
    // texture itself can be created; depends on terrain_data_ which
    // init_world fills. Done here so the descriptor write at the end
    // of rebuild_terrain_shadow_texture lands on the live scene set.
    rebuild_terrain_shadow_texture();
    init_taa();
    init_viewmodel();
    init_imgui();
    update_scene_ubo();
    initialized_ = true;
    log::info("VulkanEngine::init() done");
}


// init_imgui / destroy_imgui live in vk_ui.cpp.



// build_hud_ui / build_menu_ui live in vk_ui.cpp.




void VulkanEngine::draw(uint32_t img_index) {
    auto& frame = current_frame();

    // Snapshot every dynamic body's world matrix + AABB ONCE for this frame.
    // rebuild_tlas (below) and render_world (in the rendering pass) both read
    // from this cache, so we avoid repeating O(n) Jolt queries.
    rebuild_dyn_render_cache();

    // ---------- Compute queue: per-frame TLAS rebuild ----------
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
            .waitSemaphoreInfoCount = 0, .pWaitSemaphoreInfos = nullptr,
            .commandBufferInfoCount = 1, .pCommandBufferInfos = &cb_info,
            .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal_info,
        };
        vk_check(vkQueueSubmit2(compute_queue_, 1, &csubmit, VK_NULL_HANDLE),
                 "vkQueueSubmit2 compute (TLAS build)");
    }

    auto begin = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vk_check(vkBeginCommandBuffer(frame.command_buffer, &begin), "vkBeginCommandBuffer");

    // Scene-color image: undefined -> color attachment optimal for rendering.
    vkinit::transition_image(frame.command_buffer, scene_color_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    // Motion-vector image: 2nd color attachment of the world color pass.
    vkinit::transition_image(frame.command_buffer, motion_vec_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkinit::transition_image(frame.command_buffer, depth_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

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
    VkRenderingAttachmentInfo color_atts[2] = { color_att, motion_att };

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
        .colorAttachmentCount = 2, .pColorAttachments = color_atts,
        .pDepthAttachment = &depth_att, .pStencilAttachment = nullptr,
    };

    // ---------- Pass 1: scene → scene_color_image_ + motion_vec_image_ ----------
    vkCmdBeginRendering(frame.command_buffer, &rendering);
    render_world(frame.command_buffer);
    vkCmdEndRendering(frame.command_buffer);

    // ---------- Pass 2: TAA into history_image_[history_write_slot_] ----------
    // Flip scene_color & motion & depth & history slots into shader-read.
    vkinit::transition_image(frame.command_buffer, scene_color_image_,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // Motion-vec image is currently unread; transitioning it to SHADER_READ
    // anyway means a future TAA/SVGF pass that adds a sampler binding for
    // motion_vec_image_ doesn't need to also touch this transition site —
    // the layout is already correct. Cost: one extra (small) barrier per frame.
    vkinit::transition_image(frame.command_buffer, motion_vec_image_,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkinit::transition_image_aspect(frame.command_buffer, depth_image_,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
    // Both history slots end the previous frame in SHADER_READ_ONLY (write slot
    // exited at frame end; read slot wasn't touched). Read slot needs no
    // transition; write slot moves into COLOR_ATTACHMENT.
    vkinit::transition_image(frame.command_buffer, history_image_[history_write_slot_],
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Update TAA UBO.
    {
        struct TaaUBO {
            glm::mat4 inv_vp;
            glm::mat4 prev_vp;
            glm::vec4 viewport;
            glm::vec4 params;
        } u{};
        u.inv_vp = glm::inverse(last_view_proj_);
        u.prev_vp = prev_view_proj_;
        // TAA reads + writes at render_extent_, so the viewport vec it uses
        // for `gl_FragCoord → uv` conversion must reflect that.
        float w = static_cast<float>(render_extent_.width);
        float h = static_cast<float>(render_extent_.height);
        u.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
        u.params = glm::vec4(rt_.taa_history_blend,
                             0.05f,
                             prev_view_proj_valid_ ? 1.0f : 0.0f,
                             rt_.taa_spatial_strength);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, taa_ubo_alloc_, &ai);
        if (ai.pMappedData) std::memcpy(ai.pMappedData, &u, sizeof(u));
    }

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

    // ---------- Copy denoised history → swapchain ----------
    // ---------- Pass 2.5: Compose history (HDR) into swapchain (LDR) with ACES tonemap ----------
    vkinit::transition_image(frame.command_buffer, history_image_[history_write_slot_],
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Bloom mip chain reads the now-SHADER_READ history image (mip 0 input)
    // and writes mips 0..N down then back up additively. Final state leaves
    // mip 0 in SHADER_READ_ONLY which is what compose's binding 3 expects.
    // Skipped internally if rt_.bloom_enabled is false, but compose still
    // samples the texture — the previous frame's bloom mip 0 stays bound,
    // which is fine because compose multiplies by bloom_params.x and the
    // strength slider is the user-facing toggle.
    run_bloom_chain(frame.command_buffer);
    vkinit::transition_image(frame.command_buffer, swapchain_images_[img_index],
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
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
            glm::mat4 inv_view_proj;
        } pc_data{};
        float w = static_cast<float>(swapchain_extent_.width);
        float h = static_cast<float>(swapchain_extent_.height);
        pc_data.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
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
        pc_data.sun_dir   = glm::vec4(sun, 0.0f);
        pc_data.sun_color = glm::vec4(rt_.sun_color, rt_.sun_intensity);
        pc_data.sky_color = glm::vec4(rt_.sky_color, 0.0f);
        pc_data.flare_params = glm::vec4(rt_.lens_flare_strength,
                                          rt_.lens_flare_threshold,
                                          rt_.lens_flare_dispersal,
                                          rt_.lens_flare_halo_width);
        pc_data.flare_params2 = glm::vec4(static_cast<float>(rt_.lens_flare_ghosts),
                                           rt_.lens_flare_aberration,
                                           rt_.lens_flare_enabled ? 1.0f : 0.0f,
                                           0.0f);
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
                pc_data.sun_screen = glm::vec4(sun_uv.x, sun_uv.y, 1.0f, 0.0f);
            } else {
                pc_data.sun_screen = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
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
                                            0.0f, 0.0f);
        pc_data.inv_view_proj = glm::inverse(last_view_proj_);
        vkCmdPushConstants(frame.command_buffer, compose_pipeline_layout_,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc_data), &pc_data);
        vkCmdDraw(frame.command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(frame.command_buffer);
    }

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
        if (dd) ImGui_ImplVulkan_RenderDrawData(dd, frame.command_buffer);
        vkCmdEndRendering(frame.command_buffer);
    }

    // Two screenshot triggers: the --screenshot CLI one-shot AND F12 in-game.
    std::string screenshot_path_this_frame;
    bool screenshot_is_oneshot = false;
    if (!pending_screenshot_path_.empty()) {
        screenshot_path_this_frame = pending_screenshot_path_;
        pending_screenshot_path_.clear();
    } else if (!opts_.screenshot_path.empty() && !screenshot_taken_ &&
               static_cast<int>(frame_number_) >= opts_.screenshot_after_frames) {
        screenshot_path_this_frame = opts_.screenshot_path;
        screenshot_is_oneshot = true;
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

    vk_check(vkEndCommandBuffer(frame.command_buffer), "vkEndCommandBuffer");

    // Snapshot for next frame's reprojection.
    prev_view_proj_ = last_view_proj_;
    prev_view_proj_valid_ = true;
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
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, frame.tlas_build_done),
    };
    auto signal = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, render_semaphores_[img_index]);
    auto cmd_info = vkinit::command_buffer_submit_info(frame.command_buffer);
    VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr, .flags = 0,
        .waitSemaphoreInfoCount = 2,
        .pWaitSemaphoreInfos = waits,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal,
    };

    vk_check(vkQueueSubmit2(graphics_queue_, 1, &submit, frame.render_fence),
             "vkQueueSubmit2");

    if (want_screenshot) {
        vk_check(vkWaitForFences(device_, 1, &frame.render_fence, VK_TRUE, UINT64_MAX),
                 "screenshot vkWaitForFences");
        write_ppm(screenshot_path_this_frame, swapchain_extent_.width, swapchain_extent_.height);
        if (screenshot_is_oneshot) screenshot_taken_ = true;
        log::infof("[screenshot] saved %s", screenshot_path_this_frame.c_str());
    }

    VkPresentInfoKHR present{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_semaphores_[img_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain_,
        .pImageIndices = &img_index,
        .pResults = nullptr,
    };
    VkResult pres = vkQueuePresentKHR(graphics_queue_, &present);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        resize_requested_ = true;
    } else if (pres != VK_SUCCESS) {
        vk_check(pres, "vkQueuePresentKHR");
    }
}

void VulkanEngine::run(const RunOptions& opts) {
    opts_ = opts;
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
        last = now;
        if (frame_dt > 1.0f / 15.0f) frame_dt = 1.0f / 15.0f;
        if (frame_dt < 1e-6f)        frame_dt = 1.0f / 240.0f;
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
            // Synthetic input: walk forward (occasional jump), turn the view
            // back-and-forth so projectiles hit a variety of brushes/boxes,
            // hold fire. fire_held drives the auto-fire path so we get a
            // continuous stream of projectiles regardless of fire_rate_rps.
            in = InputFrame{};
            in.fwd  = true;
            in.jump = (static_cast<int>(demo_t * 0.5f) % 4) == 0;  // every ~2s
            in.fire_held = true;
            in.fire = (static_cast<int>(demo_t * 10.0f) & 1) == 0;
            // Sinusoidal turn — sweeps the view across the room.
            float turn = std::sin(demo_t * 0.7f) * 3.0f;
            in.mouse_dx = static_cast<double>(turn);
            in.mouse_dy = std::sin(demo_t * 1.3f) * 0.4;
        }
        bool menu_edge = in.menu_key && !prev_menu_key_;
        prev_menu_key_ = in.menu_key;
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
            constexpr float kHalfPi = 1.57079632679f;
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
        if (state_ == State::Playing && window_->cursor_captured()) {
            if (terrain_edit_mode_) {
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
                if (in.fire_held && terrain_brush_has_hit_) {
                    apply_terrain_brush(frame_dt);
                    rebuild_dirty_terrain_chunks();
                    if (!terrain_stroke_active_) terrain_stroke_active_ = true;
                } else if (terrain_stroke_active_ && !in.fire_held) {
                    // Mouse-up: heavy refresh (BLAS + Jolt) deferred until
                    // here so the stroke itself stays cheap.
                    refresh_terrain_blas();
                    refresh_terrain_collision();
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
                                     world_.aabbs.size());
                // Terrain ground clamp. update_player only collides
                // against the AABB list (static brushes + dyn props);
                // the heightfield isn't an AABB so the player would
                // free-fall through it. Sample bilinearly and snap up
                // if the player's feet sank below the ground.
                {
                    constexpr float kHalfHeight = 0.9f;
                    float ground = sample_terrain_height(player_.position.x,
                                                          player_.position.z);
                    float feet = player_.position.y - kHalfHeight;
                    if (feet < ground) {
                        player_.position.y = ground + kHalfHeight;
                        if (player_.velocity.y < 0.0f) player_.velocity.y = 0.0f;
                        player_.on_ground = true;
                    }
                }
                // Audio triggers from the player tick.
                if (audio_ && state_ == State::Playing) {
                    // Jump: rising edge of leaving ground while pressing jump.
                    if (was_on_ground && !player_.on_ground && pin.jump) {
                        audio_->play_local("jump", 0.7f, 0.06f, 0.10f);
                    }
                    // Land: falling edge — touched ground after being airborne.
                    if (!was_on_ground && player_.on_ground) {
                        audio_->play_local("land", 0.6f, 0.05f, 0.10f);
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
            }
        }

        update_scene_ubo();
        // Audio listener follows the camera. Reaping drained one-shots
        // also runs here so the active-voice list stays small.
        if (audio_) {
            audio_->set_listener(player_.eye_position(), player_.forward());
        }
        maybe_autosave_settings(frame_dt);

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
            auto& frame = current_frame();
            QLIKE_VK_CHECK(vkWaitForFences(device_, 1, &frame.render_fence,
                                           VK_TRUE, UINT64_MAX),
                           "vkWaitForFences");

            uint32_t img_index = 0;
            VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
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

            draw(img_index);

            ++frame_number_;
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
    guarded("destroy_imgui",  [&]{ destroy_imgui(); });
    guarded("destroy_viewmodel", [&]{ destroy_viewmodel(); });
    guarded("destroy_taa",    [&]{ destroy_taa(); });
    // Order: tear down the renderer's GPU dependencies on dynamic-prop world
    // matrices (TLAS instance buffer) BEFORE destroying the physics world that
    // owns those bodies. Otherwise the renderer's deferred destruction could
    // deref freed Jolt body data.
    guarded("destroy_rt",     [&]{ destroy_rt(); });
    guarded("physics_.reset", [&]{ physics_.reset(); });
    guarded("destroy_descriptors", [&]{ destroy_descriptors(); });
    guarded("destroy_meshes", [&]{
        destroy_mesh(allocator_, cube_mesh_);
        destroy_mesh(allocator_, cylinder_mesh_);
    });
    guarded("destroy_skybox", [&]{ destroy_skybox_resources(); });
    guarded("destroy_textures", [&]{ destroy_textures(); });
    guarded("destroy_grass_pipeline", [&]{ destroy_grass_pipeline(); });
    guarded("destroy_pipeline", [&]{ destroy_pipeline(); });
    guarded("destroy_grass", [&]{ destroy_grass(allocator_, grass_); });
    guarded("destroy_terrain_shadow_texture", [&]{ destroy_terrain_shadow_texture(); });
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
    guarded("destroy_swapchain",   [&]{ destroy_swapchain(); });

    guarded("vmaDestroyAllocator", [&]{
        if (allocator_) vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
    });

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
