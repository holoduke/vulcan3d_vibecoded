// FSR2 integration — calls AMD's vendored FidelityFX-FSR2 SDK (MIT,
// see external/FidelityFX-FSR2/LICENSE.txt) via its documented public
// API. No port, no clean-room reimplementation; the actual upscaling
// is done by AMD's shipped compute shaders inside the SDK.
//
// Phase 3 = context lifecycle (init/destroy/recreate)
// Phase 4 = per-frame dispatch + HR output image, compose toggle.
//
// All FSR2 calls are gated on QLIKE_HAVE_FSR2 so the engine still
// builds if the submodule wasn't initialized.

#include "engine/vk_engine/internal.h"

#include <algorithm>

#ifdef QLIKE_HAVE_FSR2
  #include "ffx_fsr2.h"
  #include "vk/ffx_fsr2_vk.h"
  #include <cstdlib>
#endif

namespace qlike {

#ifdef QLIKE_HAVE_FSR2
namespace {
// Allocate the HR storage image FSR2 writes its upscaled output into.
// Lives for the lifetime of the FSR2 context — recreated on swapchain
// resize via recreate_fsr2_context.
bool allocate_fsr2_output(VmaAllocator allocator, VkDevice device,
                          VkExtent2D extent, VkFormat format,
                          VkImage& out_img, VmaAllocation& out_alloc,
                          VkImageView& out_view) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = format;
    ici.extent    = { extent.width, extent.height, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling  = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE: FSR2 writes via compute shader. SAMPLED: compose reads it.
    // TRANSFER_DST so we can clear it on first frame (helps spot dispatch
    // failures — black instead of garbage).
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(allocator, &ici, &aci, &out_img, &out_alloc, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = out_img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(device, &vci, nullptr, &out_view) != VK_SUCCESS) {
        vmaDestroyImage(allocator, out_img, out_alloc);
        out_img = VK_NULL_HANDLE; out_alloc = nullptr;
        return false;
    }
    return true;
}
} // namespace
#endif

void VulkanEngine::init_fsr2() {
#ifdef QLIKE_HAVE_FSR2
    // Need the swapchain (= displaySize) before context creation. Using
    // displaySize as the worst-case render size lets the context handle
    // any render_scale ≤ 1.0 without recreate per scale change.
    if (fsr2_context_valid_ || swapchain_extent_.width == 0) return;

    fsr2_scratch_size_ = ffxFsr2GetScratchMemorySizeVK(physical_device_);
    fsr2_scratch_ = std::malloc(fsr2_scratch_size_);
    if (!fsr2_scratch_) {
        log::warnf("[fsr2] scratch malloc failed (%zu bytes)", fsr2_scratch_size_);
        return;
    }

    FfxFsr2ContextDescription desc{};
    desc.flags = FFX_FSR2_ENABLE_AUTO_EXPOSURE
               | FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE
               | FFX_FSR2_ENABLE_DEBUG_CHECKING;
    desc.maxRenderSize.width  = swapchain_extent_.width;
    desc.maxRenderSize.height = swapchain_extent_.height;
    desc.displaySize.width    = swapchain_extent_.width;
    desc.displaySize.height   = swapchain_extent_.height;
    desc.device = ffxGetDeviceVK(device_);

    FfxErrorCode rc = ffxFsr2GetInterfaceVK(
        &desc.callbacks, fsr2_scratch_, fsr2_scratch_size_,
        physical_device_, vkGetDeviceProcAddr);
    if (rc != FFX_OK) {
        log::warnf("[fsr2] ffxFsr2GetInterfaceVK failed: %d", int(rc));
        std::free(fsr2_scratch_);
        fsr2_scratch_ = nullptr;
        return;
    }

    auto* ctx = static_cast<FfxFsr2Context*>(std::calloc(1, sizeof(FfxFsr2Context)));
    rc = ffxFsr2ContextCreate(ctx, &desc);
    if (rc != FFX_OK) {
        log::warnf("[fsr2] ffxFsr2ContextCreate failed: %d", int(rc));
        std::free(ctx);
        std::free(fsr2_scratch_);
        fsr2_scratch_ = nullptr;
        return;
    }

    if (!allocate_fsr2_output(allocator_, device_, swapchain_extent_,
                              fsr2_output_format_,
                              fsr2_output_image_, fsr2_output_alloc_,
                              fsr2_output_view_)) {
        log::warnf("[fsr2] output image alloc failed");
        ffxFsr2ContextDestroy(ctx);
        std::free(ctx);
        std::free(fsr2_scratch_);
        fsr2_scratch_ = nullptr;
        return;
    }

    fsr2_context_ = ctx;
    fsr2_context_valid_ = true;
    fsr2_reset_history_ = true;
    log::infof("[fsr2] context created at displaySize=%ux%u (max render=%ux%u, scratch=%zuKB)",
               swapchain_extent_.width, swapchain_extent_.height,
               desc.maxRenderSize.width, desc.maxRenderSize.height,
               fsr2_scratch_size_ / 1024);
#endif
}

void VulkanEngine::destroy_fsr2() {
#ifdef QLIKE_HAVE_FSR2
    if (fsr2_context_valid_ && fsr2_context_) {
        // Defensive wait — recreate_fsr2_context can fire mid-session
        // when the user changes render scale; engine shutdown already
        // wait-idles. ffxFsr2ContextDestroy frees the SDK's Vulkan objects.
        if (device_) vkDeviceWaitIdle(device_);
        ffxFsr2ContextDestroy(static_cast<FfxFsr2Context*>(fsr2_context_));
        std::free(fsr2_context_);
        fsr2_context_ = nullptr;
    }
    fsr2_context_valid_ = false;
    if (fsr2_output_view_)  { vkDestroyImageView(device_, fsr2_output_view_, nullptr); fsr2_output_view_ = VK_NULL_HANDLE; }
    if (fsr2_output_image_) { vmaDestroyImage(allocator_, fsr2_output_image_, fsr2_output_alloc_); fsr2_output_image_ = VK_NULL_HANDLE; fsr2_output_alloc_ = nullptr; }
    if (fsr2_scratch_) {
        std::free(fsr2_scratch_);
        fsr2_scratch_ = nullptr;
        fsr2_scratch_size_ = 0;
    }
#endif
}

void VulkanEngine::recreate_fsr2_context() {
#ifdef QLIKE_HAVE_FSR2
    // FSR2 bakes maxRenderSize/displaySize and the output image dims
    // into its allocations, so any swapchain change requires a full
    // recreate. compose_uses_fsr2_ stays true if the toggle is on; the
    // post-pass rewires bindings on the next frame via rewrite_compose_for_fsr2.
    destroy_fsr2();
    init_fsr2();
    // Force a compose rebind on the new view next frame.
    compose_uses_fsr2_ = false;
#endif
}

void VulkanEngine::dispatch_fsr2(VkCommandBuffer cmd) {
#ifdef QLIKE_HAVE_FSR2
    if (!fsr2_context_valid_ || !fsr2_output_image_) return;

    // Pre-dispatch: output → GENERAL so FSR2's compute writes succeed.
    // FSR2 docs say declared resource state == current Vulkan layout; for
    // FFX_RESOURCE_STATE_UNORDERED_ACCESS the backend assumes GENERAL.
    vkinit::transition_image(cmd, fsr2_output_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);

    auto* ctx = static_cast<FfxFsr2Context*>(fsr2_context_);

    FfxFsr2DispatchDescription d{};
    d.commandList = ffxGetCommandListVK(cmd);

    // Inputs are at render_extent_ and currently in SHADER_READ_ONLY
    // (kPostWorld batch transition just happened in the post pass).
    d.color = ffxGetTextureResourceVK(ctx,
        scene_color_image_, scene_color_view_,
        render_extent_.width, render_extent_.height,
        scene_color_format_, L"qlike_scene_color",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    d.depth = ffxGetTextureResourceVK(ctx,
        depth_image_, depth_view_,
        render_extent_.width, render_extent_.height,
        depth_format_, L"qlike_depth",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = ffxGetTextureResourceVK(ctx,
        motion_vec_image_, motion_vec_view_,
        render_extent_.width, render_extent_.height,
        motion_vec_format_, L"qlike_motion",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    d.output = ffxGetTextureResourceVK(ctx,
        fsr2_output_image_, fsr2_output_view_,
        swapchain_extent_.width, swapchain_extent_.height,
        fsr2_output_format_, L"qlike_fsr2_out",
        FFX_RESOURCE_STATE_UNORDERED_ACCESS);

    // jitter conversion: world.cpp writes proj-matrix offset
    //   jx_ndc = (halton-0.5) * 2 / render_w   (with strength=1 for fsr2)
    // FSR2 wants pixel-space jitter at render res, range ~[-0.5, 0.5]:
    //   jx_pix = jx_ndc * 0.5 * render_w  =  (halton-0.5)
    d.jitterOffset.x = current_frame_view_.jitter.x * 0.5f *
                       static_cast<float>(render_extent_.width);
    d.jitterOffset.y = current_frame_view_.jitter.y * 0.5f *
                       static_cast<float>(render_extent_.height);

    // Our shaders write motion as (cur_uv - prev_uv) in UV space. FSR2
    // wants pixel-space motion at render res, so multiply by render
    // dimensions. Sign convention matches (current - previous).
    d.motionVectorScale.x = static_cast<float>(render_extent_.width);
    d.motionVectorScale.y = static_cast<float>(render_extent_.height);
    d.renderSize.width    = render_extent_.width;
    d.renderSize.height   = render_extent_.height;

    d.enableSharpening = true;
    d.sharpness        = std::clamp(rt_.compose_sharpen_strength * 0.5f, 0.0f, 1.0f);
    d.frameTimeDelta   = std::max(0.1f, last_frame_dt_ * 1000.0f);
    d.preExposure      = 1.0f;   // AUTO_EXPOSURE is enabled in context flags
    d.reset            = fsr2_reset_history_;
    fsr2_reset_history_ = false;

    // Camera params — must match the projection compute_frame_view used.
    // 80° vertical FOV, 0.05/1500 near/far (see world.cpp).
    d.cameraNear              = 0.05f;
    d.cameraFar               = 1500.0f;
    d.cameraFovAngleVertical  = glm::radians(80.0f);
    d.viewSpaceToMetersFactor = 1.0f;

    FfxErrorCode rc = ffxFsr2ContextDispatch(ctx, &d);
    if (rc != FFX_OK) {
        log::warnf("[fsr2] dispatch failed: %d", int(rc));
    }

    // Post-dispatch: FSR2 leaves the output in UNORDERED_ACCESS (= GENERAL).
    // Compose samples it via combined-image-sampler, so transition to
    // SHADER_READ_ONLY for the upcoming bloom + compose passes.
    vkinit::transition_image(cmd, fsr2_output_image_,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#else
    (void)cmd;
#endif
}

// Compose-source toggle: when fsr2 turns on/off (or after the context
// is recreated by a swapchain resize), point compose's history sampler
// at fsr2_output_view_ vs the TAA history. wait_idle is only paid on
// the actual switch, not every frame.
void VulkanEngine::rewrite_compose_for_fsr2() {
#ifdef QLIKE_HAVE_FSR2
    const bool want_fsr2 = rt_.fsr2_enabled && fsr2_output_view_ != VK_NULL_HANDLE;
    if (want_fsr2 == compose_uses_fsr2_) return;
    if (device_) vkDeviceWaitIdle(device_);
    rewrite_compose_image_bindings();
    log::infof("compose source switched to %s", want_fsr2 ? "FSR2" : "TAA/TAAU");
#endif
}

} // namespace qlike
