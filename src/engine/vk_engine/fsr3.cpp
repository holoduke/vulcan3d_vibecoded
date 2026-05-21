// FSR3 upscaler integration via AMD FidelityFX SDK 1.1.4 ffx-api
// (vendored at external/FidelityFX-SDK, MIT, see LICENSE.txt).
//
// IMPORTANT — the SDK's ffx_api.h hardcodes FFX_API_ENTRY as
// __declspec(dllexport). Static-link consumers get NULL function
// pointers. AMD's intended consumer path is runtime LoadLibrary +
// GetProcAddress via ffx_api_loader.h. We follow that here: load
// amd_fidelityfx_vk.dll on first init, resolve the 5 entry points,
// call through function pointers from then on.
//
// Session 2 of docs/fsr3_plan.md (upscaler swap). Sessions 3-6
// (frame generation context, swapchain proxy, UI mask, present
// pacing, tuning) are NOT yet implemented — they need substantial
// architecture work outside one session.

#include "engine/vk_engine/internal.h"

#include <algorithm>
#include <cstring>

#ifdef QLIKE_HAVE_FSR3
  // Includes <windows.h> internally — keep ABOVE Vulkan/Win32 headers
  // already pulled in by internal.h to avoid macro collisions.
  #include "ffx_api/ffx_api.h"
  #include "ffx_api/ffx_api_loader.h"
  #include "ffx_api/ffx_upscale.h"
  #include "ffx_api/ffx_framegeneration.h"
  #include "ffx_api/vk/ffx_api_vk.h"
#endif

namespace qlike {

#ifdef QLIKE_HAVE_FSR3
namespace {

// Process-wide DLL handle + resolved function table. Loaded lazily on
// the first init_fsr3() call; never explicitly unloaded (process exit
// handles it). All access goes through `g_ffx`; if `g_ffx_loaded` is
// false the dispatch path no-ops.
HMODULE       g_ffx_dll      = nullptr;
ffxFunctions  g_ffx          = {};
bool          g_ffx_loaded   = false;

// SDK message callback. Routes the SDK's internal diagnostics through
// our logger. Wired into both the global verbose debug channel and
// the per-context fpMessage so we get warnings/errors before any
// crash bubbles up as an unhelpful AV.
void ffx_message_cb(uint32_t type, const wchar_t* message) {
    if (!message) return;
    char ascii[1024];
    int n = WideCharToMultiByte(CP_UTF8, 0, message, -1, ascii,
                                sizeof(ascii) - 1, nullptr, nullptr);
    if (n <= 0) { std::strncpy(ascii, "(unprintable)", sizeof(ascii) - 1); }
    ascii[sizeof(ascii) - 1] = '\0';
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        log::warnf("[fsr3 SDK ERR] %s", ascii);
    } else {
        log::warnf("[fsr3 SDK]     %s", ascii);
    }
}

bool ensure_loaded() {
    if (g_ffx_loaded) return true;
    g_ffx_dll = LoadLibraryA("amd_fidelityfx_vk.dll");
    if (!g_ffx_dll) {
        log::warnf("[fsr3] LoadLibrary amd_fidelityfx_vk.dll failed (err %lu) — "
                   "ensure the DLL is staged next to the exe (CMake post-build)",
                   GetLastError());
        return false;
    }
    ffxLoadFunctions(&g_ffx, g_ffx_dll);
    if (!g_ffx.CreateContext || !g_ffx.DestroyContext ||
        !g_ffx.Dispatch     || !g_ffx.Configure) {
        log::warnf("[fsr3] DLL loaded but a function pointer is null");
        FreeLibrary(g_ffx_dll);
        g_ffx_dll = nullptr;
        return false;
    }
    g_ffx_loaded = true;
    log::infof("[fsr3] DLL loaded — fps: Create=%p Destroy=%p Cfg=%p Disp=%p Q=%p",
               (void*)g_ffx.CreateContext,  (void*)g_ffx.DestroyContext,
               (void*)g_ffx.Configure,      (void*)g_ffx.Dispatch,
               (void*)g_ffx.Query);
    return true;
}

FfxApiResource make_resource(VkImage img, uint32_t w, uint32_t h,
                             uint32_t ffx_format, uint32_t state,
                             uint32_t usage = FFX_API_RESOURCE_USAGE_READ_ONLY) {
    FfxApiResource r{};
    r.resource = reinterpret_cast<void*>(img);
    r.description.type     = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format   = ffx_format;
    r.description.width    = w;
    r.description.height   = h;
    r.description.depth    = 1;
    r.description.mipCount = 1;
    r.description.flags    = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage    = usage;
    r.state                = state;
    return r;
}

uint32_t vk_to_ffx_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R16G16_SFLOAT:       return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_D32_SFLOAT:          return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case VK_FORMAT_R32_SFLOAT:          return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case VK_FORMAT_R8G8B8A8_UNORM:      return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    default:                            return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

bool allocate_fsr3_output(VmaAllocator allocator, VkDevice device,
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
    vci.format = format;
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

void VulkanEngine::init_fsr3() {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_context_valid_ || swapchain_extent_.width == 0) return;
    if (!ensure_loaded()) return;

    // (Skipped global debug configure — ffxConfigure(nullptr, ...)
    // returns ERROR_PARAMETER even though the header docs say it
    // accepts a null context for global state. Per-context fpMessage
    // below is what actually routes diagnostics in practice.)

    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.header.pNext = nullptr;
    backend.vkDevice         = device_;
    backend.vkPhysicalDevice = physical_device_;
    backend.vkDeviceProcAddr = vkGetDeviceProcAddr;

    ffxCreateContextDescUpscale up{};
    up.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    up.header.pNext = &backend.header;
    // DEBUG_CHECKING flag stripped — it triggers a code path inside
    // the SDK that AVs in our environment. Whatever validation it
    // does isn't load-bearing for the upscale itself. Re-add once
    // session 2 is stable and we want extra diagnostics.
    // DEPTH_INFINITE — our projection uses far = 1500 m which is
    // effectively infinite for the visible scene (mountains, sky-box).
    // Without this hint, FSR3 reverse-projects depth values through
    // a finite-far frustum and gets wrong world positions for distant
    // pixels — produces visible silhouette wobble on mountains + castle
    // edges. cameraFar still passed but DEPTH_INFINITE takes precedence.
    //
    // JITTER_CANCELLATION removed — empirically made things WORSE, not
    // better. Our motion vec for static geometry computes to
    // (cur_jitter − prev_jitter) which is the naturally-correct value
    // for FSR's default expectation of "motion vecs in jittered NDC."
    // The flag's behaviour appears to assume motion vecs were authored
    // in unjittered NDC + camera adds the jitter — we do the opposite.
    up.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE
             | FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE
             | FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    up.maxRenderSize.width   = swapchain_extent_.width;
    up.maxRenderSize.height  = swapchain_extent_.height;
    up.maxUpscaleSize.width  = swapchain_extent_.width;
    up.maxUpscaleSize.height = swapchain_extent_.height;
    up.fpMessage = ffx_message_cb;

    // SEH guard around ffxCreateContext — the SDK's v1.1.4 prebuilt VK
    // DLL is observed to AV at NULL inside CreateContext on this
    // device + Vulkan SDK 1.4.341.1 combo. The root cause is unclear
    // (likely API/ABI skew between the prebuilt DLL and our driver
    // stack); a debugger session would be needed to pinpoint it.
    // Trapping the AV keeps the engine alive — fsr3_fatal_ records
    // the failure so the next-frame check flips backend back to FSR2
    // and we don't retry.
    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = FFX_API_RETURN_ERROR;
    bool seh_av = false;
    __try {
        rc = g_ffx.CreateContext(&ctx, &up.header, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        seh_av = true;
    }
    if (seh_av) {
        log::warnf("[fsr3] CreateContext AVed inside the SDK DLL — disabling "
                   "FSR3 backend permanently; falling back to FSR2");
        fsr3_fatal_ = true;
        return;
    }
    if (rc != FFX_API_RETURN_OK) {
        log::warnf("[fsr3] ffxCreateContext upscaler failed: %u", rc);
        fsr3_fatal_ = true;
        return;
    }

    if (!allocate_fsr3_output(allocator_, device_, swapchain_extent_,
                              VK_FORMAT_R16G16B16A16_SFLOAT,
                              fsr3_output_image_, fsr3_output_alloc_,
                              fsr3_output_view_)) {
        log::warnf("[fsr3] output image alloc failed");
        g_ffx.DestroyContext(&ctx, nullptr);
        return;
    }

    fsr3_upscaler_context_ = ctx;
    fsr3_context_valid_    = true;
    fsr3_reset_history_    = true;
    log::infof("[fsr3] upscaler context created (display=%ux%u, ffx-api DLL)",
               swapchain_extent_.width, swapchain_extent_.height);
#endif
}

void VulkanEngine::destroy_fsr3() {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_context_valid_ && fsr3_upscaler_context_ && g_ffx_loaded) {
        if (device_) vkDeviceWaitIdle(device_);
        ffxContext ctx = static_cast<ffxContext>(fsr3_upscaler_context_);
        g_ffx.DestroyContext(&ctx, nullptr);
        fsr3_upscaler_context_ = nullptr;
    }
    fsr3_context_valid_ = false;
    if (fsr3_output_view_)  { vkDestroyImageView(device_, fsr3_output_view_, nullptr); fsr3_output_view_ = VK_NULL_HANDLE; }
    if (fsr3_output_image_) { vmaDestroyImage(allocator_, fsr3_output_image_, fsr3_output_alloc_); fsr3_output_image_ = VK_NULL_HANDLE; fsr3_output_alloc_ = nullptr; }
#endif
}

void VulkanEngine::recreate_fsr3_context() {
#ifdef QLIKE_HAVE_FSR3
    // Only re-init if a context already exists. Without this guard, a
    // swapchain recreate at startup (when load_settings applies a non-
    // 1.0 render_scale) would force-init FSR3 even when the user has
    // FSR2 selected — which is bad given the SDK's session-2 init
    // currently AVs in our environment.
    const bool was_valid = fsr3_context_valid_;
    destroy_fsr3();
    if (was_valid) init_fsr3();
    compose_uses_fsr3_ = false;
#endif
}

void VulkanEngine::dispatch_fsr3(VkCommandBuffer cmd) {
#ifdef QLIKE_HAVE_FSR3
    if (!fsr3_context_valid_ || !fsr3_output_image_ || !g_ffx_loaded) return;

    vkinit::transition_image(cmd, fsr3_output_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);

    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.header.pNext = nullptr;
    d.commandList = reinterpret_cast<void*>(cmd);

    d.color = make_resource(scene_color_image_,
                            render_extent_.width, render_extent_.height,
                            vk_to_ffx_format(scene_color_format_),
                            FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = make_resource(depth_image_,
                            render_extent_.width, render_extent_.height,
                            vk_to_ffx_format(depth_format_),
                            FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = make_resource(motion_vec_image_,
                                    render_extent_.width, render_extent_.height,
                                    vk_to_ffx_format(motion_vec_format_),
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.exposure = FfxApiResource{};
    d.reactive = FfxApiResource{};
    d.transparencyAndComposition = FfxApiResource{};
    d.output = make_resource(fsr3_output_image_,
                             swapchain_extent_.width, swapchain_extent_.height,
                             vk_to_ffx_format(VK_FORMAT_R16G16B16A16_SFLOAT),
                             FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
                             FFX_API_RESOURCE_USAGE_UAV);

    d.jitterOffset.x = current_frame_view_.jitter.x * 0.5f *
                       static_cast<float>(render_extent_.width);
    d.jitterOffset.y = current_frame_view_.jitter.y * 0.5f *
                       static_cast<float>(render_extent_.height);
    d.motionVectorScale.x = static_cast<float>(render_extent_.width);
    d.motionVectorScale.y = static_cast<float>(render_extent_.height);
    d.renderSize.width   = render_extent_.width;
    d.renderSize.height  = render_extent_.height;
    d.upscaleSize.width  = swapchain_extent_.width;
    d.upscaleSize.height = swapchain_extent_.height;
    // Sharpening off — RCAS at any strength amplifies edge variance from
    // FSR3's temporal accumulation, which on this scene reads as visible
    // per-frame wobble at object silhouettes. compose.frag's sharpener
    // still runs on the FSR3 output downstream if the user wants it.
    d.enableSharpening = false;
    d.sharpness        = 0.0f;
    d.frameTimeDelta   = std::max(0.1f, last_frame_dt_ * 1000.0f);
    d.preExposure      = 1.0f;
    d.reset            = fsr3_reset_history_;
    fsr3_reset_history_ = false;
    d.cameraNear              = 0.05f;
    d.cameraFar               = 1500.0f;
    d.cameraFovAngleVertical  = glm::radians(80.0f);
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags                   = 0;

    ffxContext ctx = static_cast<ffxContext>(fsr3_upscaler_context_);
    ffxReturnCode_t rc = g_ffx.Dispatch(&ctx, &d.header);
    if (rc != FFX_API_RETURN_OK) {
        log::warnf("[fsr3] ffxDispatch upscale failed: %u", rc);
    }

    vkinit::transition_image(cmd, fsr3_output_image_,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#else
    (void)cmd;
#endif
}

// ============================================================================
// FSR3 SwapChain proxy — session 4 of docs/fsr3_plan.md
// ============================================================================

void VulkanEngine::init_fsr3_swapchain() {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_swapchain_active_ || fsr3_swapchain_fatal_) return;
    if (!ensure_loaded()) return;
    if (swapchain_ == VK_NULL_HANDLE) return;

    // Build a fresh VkSwapchainCreateInfoKHR matching what init_swapchain
    // chose via vk-bootstrap. The SDK will destroy our existing swapchain
    // (passed via the in/out pointer) and recreate from this descriptor,
    // returning a wrapped handle we use henceforth.
    VkSwapchainCreateInfoKHR sc_info{};
    sc_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sc_info.surface          = surface_;
    sc_info.minImageCount    = static_cast<uint32_t>(swapchain_images_.size());
    sc_info.imageFormat      = swapchain_format_;
    sc_info.imageColorSpace  = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sc_info.imageExtent      = swapchain_extent_;
    sc_info.imageArrayLayers = 1;
    sc_info.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sc_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc_info.preTransform     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    sc_info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc_info.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sc_info.clipped          = VK_TRUE;
    // oldSwapchain = NULL — SDK observed to RUNTIME_ERROR-fail when
    // we hint at the existing swapchain even though docs say it's
    // safe. The SDK will destroy our swapchain (via the in/out
    // `swapchain` pointer below) before creating its own from scratch.
    sc_info.oldSwapchain     = VK_NULL_HANDLE;

    // All four queues collapse onto our existing graphics + compute pair
    // — we don't have separate dedicated present / image-acquire queues.
    // The SDK serializes submissions internally; sharing the queue is
    // safe per AMD's docs (samples do this on hardware that lacks
    // dedicated families).
    // SDK requires all 4 queues to be DISTINCT VkQueue handles
    // (init() at FrameInterpolationSwapchainVK.cpp:1503-1510 fails with
    // VK_ERROR_INITIALIZATION_FAILED otherwise). On weaker GPUs that
    // expose only 1 graphics queue, fsr3_extra_queue_* stay null and
    // we bail before the SDK can reject us.
    if (!fsr3_extra_queue_present_ || !fsr3_extra_queue_acquire_ ||
        !fsr3_extra_queue_compute_) {
        log::warnf("[fsr3 sc] need 4 distinct queues but device exposes only 1 "
                   "graphics queue — FG unavailable on this GPU");
        fsr3_swapchain_fatal_ = true;
        return;
    }
    VkQueueInfoFFXAPI gameQ{};
    gameQ.queue = graphics_queue_; gameQ.familyIndex = graphics_queue_family_;
    VkQueueInfoFFXAPI computeQ{};
    computeQ.queue = fsr3_extra_queue_compute_;
    computeQ.familyIndex = (fsr3_extra_queue_compute_ == compute_queue_)
                            ? compute_queue_family_ : graphics_queue_family_;
    VkQueueInfoFFXAPI presentQ{};
    presentQ.queue = fsr3_extra_queue_present_;
    presentQ.familyIndex = graphics_queue_family_;
    VkQueueInfoFFXAPI acquireQ{};
    acquireQ.queue = fsr3_extra_queue_acquire_;
    acquireQ.familyIndex = graphics_queue_family_;

    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = device_;
    backend.vkPhysicalDevice = physical_device_;
    backend.vkDeviceProcAddr = vkGetDeviceProcAddr;

    VkSwapchainKHR sdk_sc = swapchain_;   // in/out — SDK overwrites
    ffxCreateContextDescFrameGenerationSwapChainVK desc{};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    desc.header.pNext = &backend.header;
    desc.physicalDevice    = physical_device_;
    desc.device            = device_;
    desc.swapchain         = &sdk_sc;
    desc.allocator         = nullptr;
    desc.createInfo        = sc_info;
    desc.gameQueue         = gameQ;
    desc.asyncComputeQueue = computeQ;
    desc.presentQueue      = presentQ;
    desc.imageAcquireQueue = acquireQ;

    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = FFX_API_RETURN_ERROR;
    bool seh_av = false;
    __try {
        rc = g_ffx.CreateContext(&ctx, &desc.header, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        seh_av = true;
    }
    if (seh_av || rc != FFX_API_RETURN_OK) {
        log::warnf("[fsr3 sc] ffxCreateContext %s — recovering plain swapchain",
                   seh_av ? "AVed" : "failed");
        // SDK destroyed our original swapchain BEFORE checking parameters
        // — `swapchain_` now points at a dead handle. Tear the rest of
        // our swapchain state (image views, semaphores) down and rebuild
        // a plain swapchain so rendering keeps working without FG.
        if (device_) vkDeviceWaitIdle(device_);
        for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
        swapchain_views_.clear();
        swapchain_images_.clear();
        for (auto s : render_semaphores_) {
            if (s) vkDestroySemaphore(device_, s, nullptr);
        }
        render_semaphores_.clear();
        swapchain_ = VK_NULL_HANDLE;
        init_swapchain();
        fsr3_swapchain_fatal_ = true;
        return;
    }

    // Query the function-pointer replacements. From here on, the engine
    // calls these instead of standard Vulkan.
    ffxQueryDescSwapchainReplacementFunctionsVK qfns{};
    qfns.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    bool query_av = false;
    __try {
        rc = g_ffx.Query(&ctx, &qfns.header);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        query_av = true;
    }
    if (query_av) {
        log::warnf("[fsr3 sc] Query AVed inside SDK — recovering plain swapchain");
        if (device_) vkDeviceWaitIdle(device_);
        for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
        swapchain_views_.clear();
        swapchain_images_.clear();
        for (auto s : render_semaphores_) {
            if (s) vkDestroySemaphore(device_, s, nullptr);
        }
        render_semaphores_.clear();
        swapchain_ = VK_NULL_HANDLE;
        init_swapchain();
        fsr3_swapchain_fatal_ = true;
        return;
    }
    if (rc != FFX_API_RETURN_OK) {
        log::warnf("[fsr3 sc] Query replacement functions failed: %u", rc);
        ffxContext c = ctx; g_ffx.DestroyContext(&c, nullptr);
        // Same recovery path as the create-failure branch above.
        if (device_) vkDeviceWaitIdle(device_);
        for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
        swapchain_views_.clear();
        swapchain_images_.clear();
        for (auto s : render_semaphores_) {
            if (s) vkDestroySemaphore(device_, s, nullptr);
        }
        render_semaphores_.clear();
        swapchain_ = VK_NULL_HANDLE;
        init_swapchain();
        fsr3_swapchain_fatal_ = true;
        return;
    }
    fsr3_acquire_fn_     = qfns.pOutAcquireNextImageKHR;
    fsr3_present_fn_     = qfns.pOutQueuePresentKHR;
    fsr3_get_images_fn_  = qfns.pOutGetSwapchainImagesKHR;
    if (!fsr3_acquire_fn_ || !fsr3_present_fn_) {
        log::warnf("[fsr3 sc] replacement fns null (acq=%p pres=%p)",
                   (void*)fsr3_acquire_fn_, (void*)fsr3_present_fn_);
        ffxContext c = ctx; g_ffx.DestroyContext(&c, nullptr);
        fsr3_swapchain_fatal_ = true;
        return;
    }

    // SDK now owns our swapchain. Update our handle to its wrapper so
    // every site that reads `swapchain_` (image acquisition target,
    // image-view creation target) sees the proxy.
    swapchain_ = sdk_sc;
    fsr3_swapchain_context_ = ctx;
    fsr3_swapchain_active_  = true;
    log::infof("[fsr3 sc] SwapChain proxy active — acquire/present "
               "routed through SDK function pointers");

    // Re-query swapchain images via the SDK's replacement fn — our
    // cached vk-bootstrap images point at the destroyed original
    // swapchain. Same for image views + per-image semaphores.
    if (fsr3_get_images_fn_) {
        for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
        swapchain_views_.clear();
        for (auto s : render_semaphores_) {
            if (s) vkDestroySemaphore(device_, s, nullptr);
        }
        render_semaphores_.clear();

        uint32_t img_count = 0;
        fsr3_get_images_fn_(device_, swapchain_, &img_count, nullptr);
        swapchain_images_.assign(img_count, VK_NULL_HANDLE);
        fsr3_get_images_fn_(device_, swapchain_, &img_count, swapchain_images_.data());

        swapchain_views_.assign(img_count, VK_NULL_HANDLE);
        for (uint32_t i = 0; i < img_count; ++i) {
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = swapchain_images_[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = swapchain_format_;
            vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vk_check(vkCreateImageView(device_, &vci, nullptr, &swapchain_views_[i]),
                     "fsr3 sc image view");
        }
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        render_semaphores_.assign(img_count, VK_NULL_HANDLE);
        for (uint32_t i = 0; i < img_count; ++i) {
            vk_check(vkCreateSemaphore(device_, &si, nullptr, &render_semaphores_[i]),
                     "fsr3 sc render semaphore");
        }
        log::infof("[fsr3 sc] re-queried %u swapchain images via SDK fn", img_count);
    }

    // Configure FG with this swapchain so the pacer knows to insert
    // generated frames. Requires the FG context (session 3) to exist.
    if (fsr3_fg_context_valid_) {
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        // CRITICAL: cfg.swapChain expects the underlying
        // FrameInterpolationSwapChainVK* (cast as void*), NOT the
        // ffx-api ffxContext wrapper. The SDK gives us back a
        // VkSwapchainKHR via init_fsr3_swapchain that is internally
        // exactly that pointer cast — pass `swapchain_` directly.
        // Passing `ctx` here AVs at RtlEnterCriticalSection because
        // the SDK reinterpret_casts it to the wrong type, then deref's
        // an uninitialized critical-section field at struct offset.
        cfg.swapChain   = swapchain_;
        cfg.frameGenerationEnabled = true;
        cfg.allowAsyncWorkloads    = true;  // session 6 tuning
        cfg.flags = 0;
        cfg.frameGenerationCallback        = nullptr;  // we dispatch manually
        cfg.frameGenerationCallbackUserContext = nullptr;
        cfg.presentCallback                = nullptr;
        cfg.presentCallbackUserContext     = nullptr;
        cfg.HUDLessColor                   = FfxApiResource{};
        cfg.onlyPresentGenerated           = false;
        cfg.generationRect.left = 0;
        cfg.generationRect.top  = 0;
        cfg.generationRect.width  = swapchain_extent_.width;
        cfg.generationRect.height = swapchain_extent_.height;
        cfg.frameID = fsr3_fg_frame_id_;

        ffxContext fg_ctx = static_cast<ffxContext>(fsr3_fg_context_);
        ffxReturnCode_t crc = FFX_API_RETURN_ERROR;
        bool cfg_av = false;
        __try {
            crc = g_ffx.Configure(&fg_ctx, &cfg.header);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            cfg_av = true;
        }
        if (cfg_av) {
            log::warnf("[fsr3 sc] FG Configure AVed inside SDK — disabling FG");
            fsr3_fg_fatal_ = true;
        } else if (crc != FFX_API_RETURN_OK) {
            log::warnf("[fsr3 sc] FG Configure(enabled=true) failed: %u", crc);
        } else {
            log::info("[fsr3 sc] FG configured with swapchain — generation active");
        }
    }
#endif
}

// Session 5 — copy the post-compose, pre-ImGui swapchain image to our
// "hudless" snapshot. SDK's FG dispatch later extracts the HUD by
// diffing (current backbuffer − HUDLessColor); without this the HUD
// gets motion-interpolated and shimmers on every other frame.
void VulkanEngine::capture_hudless_snapshot(VkCommandBuffer cmd,
                                             uint32_t swapchain_idx) {
#ifdef QLIKE_HAVE_FSR3
    if (!fsr3_swapchain_active_ || !fsr3_hudless_image_) return;

    // swapchain → TRANSFER_SRC, hudless → TRANSFER_DST.
    vkinit::transition_image(cmd, swapchain_images_[swapchain_idx],
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkinit::transition_image(cmd, fsr3_hudless_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageCopy region{};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.extent = { swapchain_extent_.width, swapchain_extent_.height, 1 };
    vkCmdCopyImage(cmd,
                   swapchain_images_[swapchain_idx],
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   fsr3_hudless_image_,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region);

    // swapchain → COLOR_ATTACHMENT (for ImGui pass), hudless →
    // SHADER_READ for the SDK's sampling later.
    vkinit::transition_image(cmd, swapchain_images_[swapchain_idx],
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkinit::transition_image(cmd, fsr3_hudless_image_,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#else
    (void)cmd; (void)swapchain_idx;
#endif
}

// Session 5 — per-frame FG configure with the latest HUDLessColor
// snapshot. Called near present time, after the hudless copy. The
// SDK's pacer uses the configure'd state for the next pair of
// (real, generated) presents. frameID must increment by exactly 1.
void VulkanEngine::update_fsr3_fg_per_frame_config() {
#ifdef QLIKE_HAVE_FSR3
    if (!fsr3_fg_context_valid_ || !fsr3_swapchain_active_) return;

    FfxApiResource hudless = (fsr3_hudless_image_)
        ? make_resource(fsr3_hudless_image_,
                        swapchain_extent_.width, swapchain_extent_.height,
                        FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM,
                        FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ)
        : FfxApiResource{};

    // FG is active iff FSR upscale + FG checkbox + auto-disable all
    // agree. When FSR toggles off mid-session, SwapChain proxy stays
    // wrapped (acquire/present still route through SDK), but we MUST
    // tell the SDK to stop generating — otherwise it paces frames
    // built from non-FSR-upscaled inputs and the pacer's history goes
    // wrong → world geometry never reaches the screen, only the sky-
    // clear color survives.
    const bool fg_should_run = rt_.fsr2_enabled && rt_.fg_enabled &&
                               !fg_runtime_disabled_ && !fsr3_fg_fatal_;
    ffxConfigureDescFrameGeneration cfg{};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain   = swapchain_;       // wrapped VkSwapchainKHR (= proxy ptr)
    cfg.frameGenerationEnabled = fg_should_run;
    cfg.allowAsyncWorkloads    = true;
    cfg.flags = 0;
    cfg.HUDLessColor = hudless;         // SDK extracts UI by diffing
    cfg.onlyPresentGenerated = false;
    cfg.generationRect.width  = swapchain_extent_.width;
    cfg.generationRect.height = swapchain_extent_.height;
    cfg.frameID = fsr3_fg_frame_id_;

    ffxContext fg_ctx = static_cast<ffxContext>(fsr3_fg_context_);
    ffxReturnCode_t rc = g_ffx.Configure(&fg_ctx, &cfg.header);
    if (rc != FFX_API_RETURN_OK) {
        log::warnf("[fsr3 fg] per-frame Configure failed: %u", rc);
    }
    ++fsr3_fg_frame_id_;
#endif
}

void VulkanEngine::destroy_fsr3_swapchain() {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_swapchain_active_ && fsr3_swapchain_context_ && g_ffx_loaded) {
        if (device_) vkDeviceWaitIdle(device_);
        // Tell the FG context to stop generating BEFORE we kill the
        // SwapChain context — otherwise the next FG dispatch could
        // reference a dangling swapchain pointer.
        if (fsr3_fg_context_valid_) {
            ffxConfigureDescFrameGeneration cfg{};
            cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            cfg.swapChain   = swapchain_;       // same handle, matched type
            cfg.frameGenerationEnabled = false;
            ffxContext fg_ctx = static_cast<ffxContext>(fsr3_fg_context_);
            g_ffx.Configure(&fg_ctx, &cfg.header);
        }
        ffxContext ctx = static_cast<ffxContext>(fsr3_swapchain_context_);
        g_ffx.DestroyContext(&ctx, nullptr);
        fsr3_swapchain_context_ = nullptr;
        // SDK destroyed our wrapped swapchain handle. Our render path
        // will hit invalid swapchain_ until init_swapchain() rebuilds
        // it. Caller (the toggle handler) is responsible for that.
        swapchain_ = VK_NULL_HANDLE;
    }
    fsr3_swapchain_active_ = false;
    fsr3_acquire_fn_ = nullptr;
    fsr3_present_fn_ = nullptr;
#endif
}

VkResult VulkanEngine::acquire_next_image(uint64_t timeout, VkSemaphore sem,
                                          VkFence fence, uint32_t* out_idx) {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_swapchain_active_ && fsr3_acquire_fn_) {
        return fsr3_acquire_fn_(device_, swapchain_, timeout, sem, fence, out_idx);
    }
#endif
    return vkAcquireNextImageKHR(device_, swapchain_, timeout, sem, fence, out_idx);
}

VkResult VulkanEngine::queue_present_image(uint32_t image_index,
                                           VkSemaphore wait_sem) {
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &image_index;
    if (wait_sem) {
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &wait_sem;
    }
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_swapchain_active_ && fsr3_present_fn_) {
        return fsr3_present_fn_(graphics_queue_, &present);
    }
#endif
    return vkQueuePresentKHR(graphics_queue_, &present);
}

// ============================================================================
// FSR3 Frame Generation — session 3 of docs/fsr3_plan.md
// ============================================================================
//
// Creates the FG context, runs Prepare + Generation each frame to a
// private HR output image. Does NOT integrate with the swapchain
// (session 4) — the generated frame sits in fsr3_fg_output_image_
// and goes nowhere. Goal of session 3: prove the SDK can dispatch
// without crashing, validation clean, no GPU device-lost.

void VulkanEngine::init_fsr3_fg() {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_fg_context_valid_ || fsr3_fg_fatal_) return;
    if (!ensure_loaded()) return;
    if (swapchain_extent_.width == 0) return;

    // FG context backend desc — same shape as the upscaler's.
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice         = device_;
    backend.vkPhysicalDevice = physical_device_;
    backend.vkDeviceProcAddr = vkGetDeviceProcAddr;

    ffxCreateContextDescFrameGeneration fg{};
    fg.header.type     = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    fg.header.pNext    = &backend.header;
    // No HDR flag — our actual swapchain is B8G8R8A8_UNORM (LDR). The
    // SwapChain proxy in session 4 will reject FG creation if the FG
    // backBufferFormat doesn't match the real swapchain format.
    // ASYNC_WORKLOAD_SUPPORT lets the FG context dispatch its
    // generation work onto the asyncCompute queue we passed at
    // SwapChain context creation. Without it FG steals graphics queue
    // time from the next real frame's render, eating the perf benefit.
    fg.flags           = FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT;
    fg.displaySize.width   = swapchain_extent_.width;
    fg.displaySize.height  = swapchain_extent_.height;
    fg.maxRenderSize.width  = swapchain_extent_.width;
    fg.maxRenderSize.height = swapchain_extent_.height;
    fg.backBufferFormat = FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM;

    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = FFX_API_RETURN_ERROR;
    bool seh_av = false;
    __try {
        rc = g_ffx.CreateContext(&ctx, &fg.header, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        seh_av = true;
    }
    if (seh_av) {
        log::warnf("[fsr3 fg] CreateContext AVed inside SDK — disabling FG");
        fsr3_fg_fatal_ = true;
        return;
    }
    if (rc != FFX_API_RETURN_OK) {
        log::warnf("[fsr3 fg] ffxCreateContext failed: %u", rc);
        fsr3_fg_fatal_ = true;
        return;
    }

    if (!allocate_fsr3_output(allocator_, device_, swapchain_extent_,
                              VK_FORMAT_R16G16B16A16_SFLOAT,
                              fsr3_fg_output_image_, fsr3_fg_output_alloc_,
                              fsr3_fg_output_view_)) {
        log::warnf("[fsr3 fg] FG output image alloc failed");
        ffxContext c = ctx;
        g_ffx.DestroyContext(&c, nullptr);
        fsr3_fg_fatal_ = true;
        return;
    }

    // Session 5 — hudless snapshot at swapchain extent + format. Used
    // by the SDK to extract UI pixels from the final backbuffer. UNDEFINED
    // → SHADER_READ_ONLY transition happens on the first frame's copy.
    {
        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format    = swapchain_format_;
        ici.extent    = { swapchain_extent_.width, swapchain_extent_.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling  = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
                  | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(allocator_, &ici, &aci,
                           &fsr3_hudless_image_, &fsr3_hudless_alloc_, nullptr) == VK_SUCCESS) {
            VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            vci.image = fsr3_hudless_image_;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = swapchain_format_;
            vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCreateImageView(device_, &vci, nullptr, &fsr3_hudless_view_);
        }
    }

    fsr3_fg_context_       = ctx;
    fsr3_fg_context_valid_ = true;
    fsr3_fg_frame_id_      = 0;
    log::infof("[fsr3 fg] context created (display=%ux%u)",
               swapchain_extent_.width, swapchain_extent_.height);
#endif
}

void VulkanEngine::destroy_fsr3_fg() {
#ifdef QLIKE_HAVE_FSR3
    if (fsr3_fg_context_valid_ && fsr3_fg_context_ && g_ffx_loaded) {
        if (device_) vkDeviceWaitIdle(device_);
        ffxContext ctx = static_cast<ffxContext>(fsr3_fg_context_);
        g_ffx.DestroyContext(&ctx, nullptr);
        fsr3_fg_context_ = nullptr;
    }
    fsr3_fg_context_valid_ = false;
    if (fsr3_fg_output_view_)  { vkDestroyImageView(device_, fsr3_fg_output_view_, nullptr); fsr3_fg_output_view_ = VK_NULL_HANDLE; }
    if (fsr3_fg_output_image_) { vmaDestroyImage(allocator_, fsr3_fg_output_image_, fsr3_fg_output_alloc_); fsr3_fg_output_image_ = VK_NULL_HANDLE; fsr3_fg_output_alloc_ = nullptr; }
    if (fsr3_hudless_view_)    { vkDestroyImageView(device_, fsr3_hudless_view_, nullptr); fsr3_hudless_view_ = VK_NULL_HANDLE; }
    if (fsr3_hudless_image_)   { vmaDestroyImage(allocator_, fsr3_hudless_image_, fsr3_hudless_alloc_); fsr3_hudless_image_ = VK_NULL_HANDLE; fsr3_hudless_alloc_ = nullptr; }
#endif
}

void VulkanEngine::dispatch_fsr3_fg(VkCommandBuffer cmd) {
#ifdef QLIKE_HAVE_FSR3
    if (!fsr3_fg_context_valid_ || !fsr3_fg_output_image_) return;
    // FG dispatch needs the FSR3 upscaler output as presentColor — must
    // already be in SHADER_READ from dispatch_fsr3 above.
    if (!fsr3_output_image_) return;

    // FG output → GENERAL for the SDK's compute writes.
    vkinit::transition_image(cmd, fsr3_fg_output_image_,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);

    // 1. Prepare pass — consumes LR depth + motion. Builds the
    //    dilated motion field FG uses internally.
    {
        ffxDispatchDescFrameGenerationPrepare prep{};
        prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        prep.frameID     = fsr3_fg_frame_id_;
        prep.flags       = 0;
        prep.commandList = reinterpret_cast<void*>(cmd);
        prep.renderSize.width  = render_extent_.width;
        prep.renderSize.height = render_extent_.height;
        prep.jitterOffset.x = current_frame_view_.jitter.x * 0.5f *
                              static_cast<float>(render_extent_.width);
        prep.jitterOffset.y = current_frame_view_.jitter.y * 0.5f *
                              static_cast<float>(render_extent_.height);
        prep.motionVectorScale.x = static_cast<float>(render_extent_.width);
        prep.motionVectorScale.y = static_cast<float>(render_extent_.height);
        prep.frameTimeDelta   = std::max(0.1f, last_frame_dt_ * 1000.0f);
        prep.unused_reset     = false;
        prep.cameraNear       = 0.05f;
        prep.cameraFar        = 1500.0f;
        prep.cameraFovAngleVertical = glm::radians(80.0f);
        prep.viewSpaceToMetersFactor = 1.0f;
        prep.depth = make_resource(depth_image_,
                                   render_extent_.width, render_extent_.height,
                                   vk_to_ffx_format(depth_format_),
                                   FFX_API_RESOURCE_STATE_COMPUTE_READ);
        prep.motionVectors = make_resource(motion_vec_image_,
                                           render_extent_.width, render_extent_.height,
                                           vk_to_ffx_format(motion_vec_format_),
                                           FFX_API_RESOURCE_STATE_COMPUTE_READ);

        ffxContext ctx = static_cast<ffxContext>(fsr3_fg_context_);
        ffxReturnCode_t rc = g_ffx.Dispatch(&ctx, &prep.header);
        if (rc != FFX_API_RETURN_OK) {
            log::warnf("[fsr3 fg] Prepare dispatch failed: %u (frame=%llu)",
                       rc, (unsigned long long)fsr3_fg_frame_id_);
            return;
        }
    }

    // 2. Generation pass — consumes presentColor (HR FSR3 upscale
    //    output), produces 1 in-between frame in outputs[0].
    {
        ffxDispatchDescFrameGeneration gen{};
        gen.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        gen.commandList = reinterpret_cast<void*>(cmd);
        gen.presentColor = make_resource(fsr3_output_image_,
                                         swapchain_extent_.width, swapchain_extent_.height,
                                         vk_to_ffx_format(VK_FORMAT_R16G16B16A16_SFLOAT),
                                         FFX_API_RESOURCE_STATE_COMPUTE_READ);
        gen.outputs[0] = make_resource(fsr3_fg_output_image_,
                                       swapchain_extent_.width, swapchain_extent_.height,
                                       vk_to_ffx_format(VK_FORMAT_R16G16B16A16_SFLOAT),
                                       FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
                                       FFX_API_RESOURCE_USAGE_UAV);
        gen.numGeneratedFrames = 1;
        gen.reset = false;
        gen.backbufferTransferFunction = 0; // SRGB-ish, our scene_color is linear HDR
        gen.minMaxLuminance[0] = 0.001f;
        gen.minMaxLuminance[1] = 1000.0f;
        gen.generationRect.left = 0;
        gen.generationRect.top  = 0;
        gen.generationRect.width  = swapchain_extent_.width;
        gen.generationRect.height = swapchain_extent_.height;
        gen.frameID = fsr3_fg_frame_id_;

        ffxContext ctx = static_cast<ffxContext>(fsr3_fg_context_);
        ffxReturnCode_t rc = g_ffx.Dispatch(&ctx, &gen.header);
        if (rc != FFX_API_RETURN_OK) {
            log::warnf("[fsr3 fg] Generation dispatch failed: %u (frame=%llu)",
                       rc, (unsigned long long)fsr3_fg_frame_id_);
        }
    }

    // FG output → SHADER_READ for compose to sample later (session 4
    // will wire that up; today it just sits unused).
    vkinit::transition_image(cmd, fsr3_fg_output_image_,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // SDK contract: frameID must increment by exactly 1 each frame.
    // Any other delta resets the FG state machine.
    ++fsr3_fg_frame_id_;
#else
    (void)cmd;
#endif
}

void VulkanEngine::rewrite_compose_for_fsr3() {
#ifdef QLIKE_HAVE_FSR3
    // CRITICAL: must also check rt_.fsr2_enabled — that's the master
    // toggle for ANY FSR upscale. Without it, when the user turns FSR
    // off the compose desc set stays bound to the (stale) FSR3 output
    // image, so the screen freezes on the last FSR-upscaled frame
    // while the sky (recomputed inside compose.frag from sun_dir)
    // keeps animating. Same predicate the actual binding write uses.
    const bool want_fsr3 = rt_.fsr2_enabled &&
                           (rt_.fsr_backend == 1) &&
                           (fsr3_output_view_ != VK_NULL_HANDLE);
    if (want_fsr3 == compose_uses_fsr3_) return;
    if (device_) vkDeviceWaitIdle(device_);
    rewrite_compose_image_bindings();
    log::infof("compose source switched to %s", want_fsr3 ? "FSR3" : "fallback");
#endif
}

} // namespace qlike
