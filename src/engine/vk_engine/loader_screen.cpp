// Loading-screen presentation: decodes the loader background and pushes
// progress frames to the swapchain during the (multi-second) asset load,
// before the main render loop starts. Split out of vk_engine.cpp.

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

void VulkanEngine::ensure_loader_bg_loaded_() {
    if (loader_bg_tried_) return;
    loader_bg_tried_ = true;
    // Probe a few roots so the engine finds the asset whether the cwd
    // is the project root or `build/`.
    const char* roots[] = {
        "assets/loader/loading_bg.jpg",
        "../assets/loader/loading_bg.jpg",
        "../../assets/loader/loading_bg.jpg",
    };
    std::string path;
    for (const char* r : roots) {
        std::ifstream f(r);
        if (f.good()) { path = r; break; }
    }
    if (path.empty()) {
        log::warnf("[loader] background image missing on all probe paths");
        return;
    }
    Texture2D r = upload_texture_from_file(
        device_, allocator_, graphics_queue_, graphics_queue_family_,
        path, VK_FORMAT_R8G8B8A8_SRGB);
    if (!r.ok) {
        log::warnf("[loader] failed to load %s", path.c_str());
        return;
    }
    // Free the sampler view -- we only need the raw image for
    // vkCmdBlitImage. Saves a tiny bit of memory and keeps the
    // descriptor pool budget intact.
    if (r.view) vkDestroyImageView(device_, r.view, nullptr);
    // Transition once from SHADER_READ_ONLY_OPTIMAL (where upload
    // leaves it) -> TRANSFER_SRC_OPTIMAL so the per-frame blit's
    // src layout matches. The image lives in this layout forever
    // since the loader is the only consumer.
    //
    // Don't assign to loader_bg_image_ until the transition succeeds.
    // present_loader_frame() picks up the asset based on that handle
    // and assumes the layout is TRANSFER_SRC_OPTIMAL; if pool create
    // / cmd alloc / queue submit fail, we want present_loader_frame to
    // fall through to the cleared-navy + vignette fallback instead of
    // issuing a vkCmdBlitImage with the wrong source layout.
    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = graphics_queue_family_,
    };
    if (vkCreateCommandPool(device_, &cpi, nullptr, &cp) != VK_SUCCESS) {
        log::warnf("[loader] cmd pool create failed; dropping bg image");
        vmaDestroyImage(allocator_, r.image, r.alloc);
        return;
    }
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = cp,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(device_, &cai, &cmd) != VK_SUCCESS) {
        log::warnf("[loader] cmd alloc failed; dropping bg image");
        vkDestroyCommandPool(device_, cp, nullptr);
        vmaDestroyImage(allocator_, r.image, r.alloc);
        return;
    }
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(cmd, &bi);
    vkinit::transition_image(cmd, r.image,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr,
    };
    VkResult sub = vkQueueSubmit(graphics_queue_, 1, &si, VK_NULL_HANDLE);
    if (sub == VK_SUCCESS) sub = vkQueueWaitIdle(graphics_queue_);
    vkDestroyCommandPool(device_, cp, nullptr);
    if (sub != VK_SUCCESS) {
        log::warnf("[loader] layout transition submit failed (%d); dropping bg image", int(sub));
        vmaDestroyImage(allocator_, r.image, r.alloc);
        return;
    }
    // Transition succeeded -- publish to engine state. destroy_loader_bg_()
    // at shutdown owns the image from this point.
    loader_bg_image_  = r.image;
    loader_bg_alloc_  = r.alloc;
    loader_bg_extent_ = { static_cast<uint32_t>(r.width),
                          static_cast<uint32_t>(r.height) };
    log::infof("[loader] background loaded (%ux%u) from %s",
                loader_bg_extent_.width, loader_bg_extent_.height, path.c_str());
}

void VulkanEngine::destroy_loader_bg_() {
    if (loader_bg_image_) {
        vmaDestroyImage(allocator_, loader_bg_image_, loader_bg_alloc_);
        loader_bg_image_ = VK_NULL_HANDLE;
        loader_bg_alloc_ = nullptr;
    }
}

void VulkanEngine::present_loader_frame(const char* label, float progress) {
    if (!device_ || !swapchain_ || swapchain_images_.empty()) return;

    if (window_) {
        std::string title = "quake-like — ";
        title += label ? label : "Loading…";
        glfwSetWindowTitle(window_->handle(), title.c_str());
        // Pump GLFW so the OS doesn't mark the window as "not responding"
        // during the long init steps that sit between loader frames.
        window_->poll_events();
    }

    auto& frame = current_frame();

    QLIKE_VK_CHECK(vkWaitForFences(device_, 1, &frame.render_fence,
                                    VK_TRUE, UINT64_MAX),
                   "loader: vkWaitForFences");

    uint32_t img_idx = 0;
    // Routes through FSR3 SwapChain proxy fns when active (session 4),
    // plain Vulkan otherwise.
    VkResult acq = acquire_next_image(UINT64_MAX, frame.swapchain_semaphore,
                                       VK_NULL_HANDLE, &img_idx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { resize_requested_ = true; return; }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        QLIKE_VK_CHECK(acq, "loader: vkAcquireNextImageKHR");
    }

    QLIKE_VK_CHECK(vkResetFences(device_, 1, &frame.render_fence),
                   "loader: vkResetFences");
    QLIKE_VK_CHECK(vkResetCommandBuffer(frame.command_buffer, 0),
                   "loader: vkResetCommandBuffer");

    auto bi = vkinit::command_buffer_begin_info(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    QLIKE_VK_CHECK(vkBeginCommandBuffer(frame.command_buffer, &bi),
                   "loader: vkBeginCommandBuffer");

    // Try to load the background image on the first call. Cached
    // across all subsequent loader frames.
    ensure_loader_bg_loaded_();

    // Blit path: if we have a background image, copy it to the
    // swapchain (stretched to fit) BEFORE the render pass begins.
    // vkCmdBlitImage runs without any graphics pipeline -- perfect
    // for the loader screen which fires before pipelines exist.
    // Falls back to the cleared navy + radial vignette if the asset
    // is missing.
    bool used_bg = (loader_bg_image_ != VK_NULL_HANDLE);
    if (used_bg) {
        vkinit::transition_image_src_stage(frame.command_buffer,
                                  swapchain_images_[img_idx],
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);   // chains with the acquire wait
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = {
            static_cast<int32_t>(loader_bg_extent_.width),
            static_cast<int32_t>(loader_bg_extent_.height), 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = {
            static_cast<int32_t>(swapchain_extent_.width),
            static_cast<int32_t>(swapchain_extent_.height), 1 };
        vkCmdBlitImage(frame.command_buffer,
                        loader_bg_image_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain_images_[img_idx],
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_LINEAR);
        vkinit::transition_image(frame.command_buffer,
                                  swapchain_images_[img_idx],
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    } else {
        vkinit::transition_image_src_stage(frame.command_buffer,
                                  swapchain_images_[img_idx],
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);   // chains with the acquire wait
    }

    VkClearValue clear_bg{};
    clear_bg.color = { { 0.020f, 0.025f, 0.040f, 1.0f } };  // deep navy
    // When a BG image was just blit'd into the swapchain, we MUST use
    // LOAD_OP_LOAD on the colour attachment, otherwise the rendering
    // pass starts with LOAD_OP_CLEAR which wipes the blit'd image back
    // to navy before the overlays draw -- "loader image is black".
    auto sw_color = vkinit::color_attachment_info(
        swapchain_views_[img_idx],
        used_bg ? nullptr : &clear_bg,
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

    // Everything is drawn with vkCmdClearAttachments rectangles — we
    // can't use ImGui/pipelines here (this runs DURING init, before
    // any pipeline exists). The pixel-block font below lets us show
    // a real percentage and title without a font atlas.
    auto clear_rect = [&](int x, int y, int w, int h,
                          float r, float g, float b) {
        if (w <= 0 || h <= 0) return;
        VkClearAttachment ca{};
        ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ca.colorAttachment = 0;
        ca.clearValue.color = { { r, g, b, 1.0f } };
        VkClearRect cr{};
        cr.rect.offset = { x, y };
        cr.rect.extent = { static_cast<uint32_t>(w),
                            static_cast<uint32_t>(h) };
        cr.baseArrayLayer = 0;
        cr.layerCount = 1;
        vkCmdClearAttachments(frame.command_buffer, 1, &ca, 1, &cr);
    };

    int W = static_cast<int>(swapchain_extent_.width);
    int H = static_cast<int>(swapchain_extent_.height);

    // Background gradient: dim radial vignette from centre. Skipped
    // entirely when a loader BG image was blit'd into the swapchain
    // above -- the rects would just cover the image otherwise. The
    // text + progress overlay below still draws on top so the user
    // can see the loading state on either background.
    if (!used_bg) {
        for (int b = 0; b < 8; ++b) {
            float t = float(b) / 8.0f;
            float d = 0.020f - 0.010f * t;
            int pad = static_cast<int>(t * std::max(W, H) * 0.5f);
            clear_rect(pad, pad, W - 2 * pad, H - 2 * pad, d, d * 1.1f, d * 1.4f);
        }
    }

    // ---- Pixel font (5×7), drawn as rectangle blocks at any scale ----
    // Bit 6 of each column byte is the TOP row, bit 0 is the BOTTOM row.
    // Only the chars actually rendered are defined: digits, %, space,
    // and "QUAKELIKEKLAOIDNG" letters.
    struct Glyph { uint8_t col[5]; };
    auto G = [](uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) {
        Glyph g{}; g.col[0]=a; g.col[1]=b; g.col[2]=c; g.col[3]=d; g.col[4]=e;
        return g;
    };
    auto glyph_for = [&](char ch) -> Glyph {
        switch (ch) {
            case '0': return G(0x3E,0x51,0x49,0x45,0x3E);
            case '1': return G(0x00,0x42,0x7F,0x40,0x00);
            case '2': return G(0x42,0x61,0x51,0x49,0x46);
            case '3': return G(0x21,0x41,0x45,0x4B,0x31);
            case '4': return G(0x18,0x14,0x12,0x7F,0x10);
            case '5': return G(0x27,0x45,0x45,0x45,0x39);
            case '6': return G(0x3C,0x4A,0x49,0x49,0x30);
            case '7': return G(0x01,0x71,0x09,0x05,0x03);
            case '8': return G(0x36,0x49,0x49,0x49,0x36);
            case '9': return G(0x06,0x49,0x49,0x29,0x1E);
            case '%': return G(0x23,0x13,0x08,0x64,0x62);
            case 'Q': return G(0x3E,0x41,0x51,0x21,0x5E);
            case 'U': return G(0x3F,0x40,0x40,0x40,0x3F);
            case 'A': return G(0x7E,0x09,0x09,0x09,0x7E);
            case 'K': return G(0x7F,0x08,0x14,0x22,0x41);
            case 'E': return G(0x7F,0x49,0x49,0x49,0x41);
            case 'L': return G(0x7F,0x40,0x40,0x40,0x40);
            case 'I': return G(0x00,0x41,0x7F,0x41,0x00);
            case 'O': return G(0x3E,0x41,0x41,0x41,0x3E);
            case 'D': return G(0x7F,0x41,0x41,0x41,0x3E);
            case 'N': return G(0x7F,0x02,0x0C,0x30,0x7F);
            case 'G': return G(0x3E,0x41,0x49,0x49,0x7A);
            case ' ': default: return G(0,0,0,0,0);
        }
    };
    auto draw_text = [&](int x, int y, int px, const char* s,
                         float r, float g, float b) {
        for (const char* p_s = s; *p_s; ++p_s) {
            Glyph g_ = glyph_for(*p_s);
            for (int c = 0; c < 5; ++c) {
                uint8_t col = g_.col[c];
                for (int r2 = 0; r2 < 7; ++r2) {
                    if (col & (1 << r2)) {
                        clear_rect(x + c * px, y + r2 * px, px, px, r, g, b);
                    }
                }
            }
            x += 6 * px;        // 5 px + 1 px spacing
        }
    };

    // ---- "QUAKE LIKE" title centred near the top third ----
    {
        const char* title = "QUAKE LIKE";
        int px = std::max(3, W / 280);
        int title_w = static_cast<int>(strlen(title)) * 6 * px - px;
        int tx = (W - title_w) / 2;
        int ty = H / 3;
        // Drop shadow for legibility
        draw_text(tx + px, ty + px, px, title, 0.0f, 0.0f, 0.0f);
        draw_text(tx,      ty,      px, title, 0.92f, 0.94f, 0.98f);
        // Accent rule below title
        int rule_w = title_w + px * 8;
        int rule_x = (W - rule_w) / 2;
        clear_rect(rule_x, ty + 7 * px + px * 3, rule_w, std::max(1, px / 2),
                   0.32f, 0.55f, 0.92f);
    }

    // ---- Progress bar — bigger, framed, with bright leading edge ----
    int bar_w = std::min(720, W - 120);
    int bar_h = std::max(18, H / 60);
    int bar_x = (W - bar_w) / 2;
    int bar_y = H * 5 / 8;
    int frame_t = std::max(2, bar_h / 8);
    // Outer frame (subtle blue)
    clear_rect(bar_x - frame_t, bar_y - frame_t,
               bar_w + 2 * frame_t, bar_h + 2 * frame_t,
               0.18f, 0.26f, 0.38f);
    // Track (deep)
    clear_rect(bar_x, bar_y, bar_w, bar_h, 0.04f, 0.06f, 0.10f);
    float p = progress;
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    int fill_w = static_cast<int>(static_cast<float>(bar_w) * p + 0.5f);
    // Fill — bright brand blue
    clear_rect(bar_x, bar_y, fill_w, bar_h, 0.32f, 0.58f, 0.95f);
    // Leading edge — brighter highlight column for "active" feel
    int edge_w = std::max(2, bar_h / 4);
    if (fill_w >= edge_w) {
        clear_rect(bar_x + fill_w - edge_w, bar_y, edge_w, bar_h,
                   0.85f, 0.92f, 1.00f);
    }

    // ---- Percentage display centred below the bar ("XX%") ----
    {
        int pct = static_cast<int>(p * 100.0f + 0.5f);
        char buf[6];
        std::snprintf(buf, sizeof(buf), "%d%%", pct);
        int px = std::max(2, W / 360);
        int txt_w = static_cast<int>(strlen(buf)) * 6 * px - px;
        int tx = (W - txt_w) / 2;
        int ty = bar_y + bar_h + std::max(16, bar_h);
        draw_text(tx + 1, ty + 1, px, buf, 0.0f, 0.0f, 0.0f);
        draw_text(tx,     ty,     px, buf, 0.82f, 0.92f, 1.00f);
    }

    // ---- "LOADING" label at the bottom ----
    {
        const char* lbl = "LOADING";
        int px = std::max(2, W / 480);
        int txt_w = static_cast<int>(strlen(lbl)) * 6 * px - px;
        int tx = (W - txt_w) / 2;
        int ty = H - H / 10;
        draw_text(tx, ty, px, lbl, 0.42f, 0.50f, 0.62f);
    }

    vkCmdEndRendering(frame.command_buffer);

    vkinit::transition_image(frame.command_buffer,
                              swapchain_images_[img_idx],
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    QLIKE_VK_CHECK(vkEndCommandBuffer(frame.command_buffer),
                   "loader: vkEndCommandBuffer");

    auto wait = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,   // blit (transfer) also writes the image
        frame.swapchain_semaphore);
    auto signal = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
        render_semaphores_[img_idx]);
    auto cmd_info = vkinit::command_buffer_submit_info(frame.command_buffer);
    VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr, .flags = 0,
        .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal,
    };
    QLIKE_VK_CHECK(vkQueueSubmit2(graphics_queue_, 1, &submit,
                                   frame.render_fence),
                   "loader: vkQueueSubmit2");

    // Loader-frame present — same proxy-aware path as the gameplay one.
    VkResult pres = queue_present_image(img_idx, render_semaphores_[img_idx]);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        resize_requested_ = true;
    } else if (pres != VK_SUCCESS) {
        QLIKE_VK_CHECK(pres, "loader: vkQueuePresentKHR");
    }

    ++frame_number_;
}


} // namespace qlike
