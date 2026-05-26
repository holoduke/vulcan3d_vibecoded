// Vulkan device + swapchain + depth + per-frame command/sync state +
// readback buffer + raster pipelines + persistent pipeline cache. Everything
// the engine spins up once at init() before the first draw, plus the
// recreate_swapchain() path that handles window resizes.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"
#include "engine/window.h"

#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qlike {

void VulkanEngine::init_vulkan() {
    vkb::InstanceBuilder builder;
    auto& b = builder.set_app_name("quake-like")
                  .request_validation_layers(vk_validation_enabled())
                  .require_api_version(1, 3, 0)
                  .set_debug_callback(&debug_callback)
                  .set_debug_messenger_severity(
                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                  .set_debug_messenger_type(
                      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
    if (vk_validation_enabled()) {
        // Synchronization validation: catches the kind of cross-queue / image-
        // layout / barrier race that produces VK_ERROR_DEVICE_LOST in
        // production. Best-practices flags driver hints. GPU-assisted
        // validation injects checks for shader-side OOB / NaN producers.
        b.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
        b.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
        b.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
        log::info("vk validation features: SYNC + BEST_PRACTICES + GPU_ASSISTED");
    }
    auto inst_ret = b.build();
    if (!inst_ret) throw std::runtime_error("vkb::InstanceBuilder: " +
                                            inst_ret.error().message());
    auto vkb_inst = inst_ret.value();
    instance_ = vkb_inst.instance;
    debug_messenger_ = vkb_inst.debug_messenger;
    log::infof("vk instance created (validation=%s)", vk_validation_enabled() ? "on" : "off");

    surface_ = reinterpret_cast<VkSurfaceKHR>(
        window_->create_surface(reinterpret_cast<VkInstance_T*>(instance_)));

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_features.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rq_features{};
    rq_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq_features.rayQuery = VK_TRUE;

    // Base (Vulkan 1.0) features. tessellationShader is required for the
    // near-terrain GPU tessellation pipeline (distance-adaptive subdivision
    // + displacement). Nothing else in the base feature set is requested
    // today, so this is purely additive.
    VkPhysicalDeviceFeatures base_feats{};
    base_feats.tessellationShader = VK_TRUE;
    // Anisotropic filtering — the shared albedo/normal sampler enables it
    // so the big near-horizontal terrain ground plane stays sharp at
    // grazing angles instead of mip-blurring along the view direction.
    base_feats.samplerAnisotropy = VK_TRUE;
    // Wireframe debug view (terrain mesh / tessellation density).
    base_feats.fillModeNonSolid = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    auto phys_ret = selector.set_minimum_version(1, 3)
                        .set_required_features(base_feats)
                        .set_required_features_13(f13)
                        .set_required_features_12(f12)
                        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
                        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
                        // FSR3 (FidelityFX SDK 1.1.4) backend looks up Vulkan
                        // functions via the KHR-suffixed names (e.g.
                        // vkCmdBeginRenderingKHR, vkCmdPipelineBarrier2KHR)
                        // even on a 1.3 device. Without the extensions
                        // explicitly enabled, vkGetDeviceProcAddr returns
                        // NULL for those names and the SDK AVs on first call.
                        // VK_EXT_subgroup_size_control lets the SDK query
                        // wave64 capability for its compute shaders.
                        .add_required_extension(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
                        .add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
                        .add_required_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
                        // VK_KHR_get_memory_requirements2 — promoted to
                        // Vulkan 1.1 core, but the FSR3 SDK loads the
                        // KHR-suffixed alias `vkGetBufferMemoryRequirements2KHR`
                        // via vkGetDeviceProcAddr. That alias only exists when
                        // this extension is explicitly enabled on the device.
                        // Without it the SDK gets NULL and AVs in
                        // ffxCreateContext (proven via procaddr-trace shim).
                        .add_required_extension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME)
                        .add_required_extension_features(as_features)
                        .add_required_extension_features(rq_features)
                        .set_surface(surface_)
                        .select();
    if (!phys_ret) throw std::runtime_error("vkb device select: " +
                                            phys_ret.error().message());
    auto vkb_phys = phys_ret.value();
    physical_device_ = vkb_phys.physical_device;
    log::infof("vk physical device: %s", vkb_phys.properties.deviceName);

    // Attachment-based Variable Rate Shading is optional — enabled only
    // when the device exposes both the extension and the
    // attachmentFragmentShadingRate feature. NVIDIA Turing+, AMD RDNA2+
    // and Intel Arc support it; MoltenVK and older Intel iGPUs don't.
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrs_feat{};
    vrs_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
    bool wants_vrs = vkb_phys.is_extension_present(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
    if (wants_vrs) {
        VkPhysicalDeviceFeatures2 feats2{};
        feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feats2.pNext = &vrs_feat;
        vkGetPhysicalDeviceFeatures2(physical_device_, &feats2);
        if (vrs_feat.attachmentFragmentShadingRate == VK_TRUE) {
            // Query tile size — needed for the attachment dimensions.
            VkPhysicalDeviceFragmentShadingRatePropertiesKHR vrs_props{};
            vrs_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &vrs_props;
            vkGetPhysicalDeviceProperties2(physical_device_, &props2);
            vrs_texel_size_ = vrs_props.minFragmentShadingRateAttachmentTexelSize;
            // Strip the other VRS features we don't use — leaves the
            // attachment-only path enabled and avoids opting in to
            // pipeline/primitive paths the engine doesn't drive.
            vrs_feat.pipelineFragmentShadingRate  = VK_FALSE;
            vrs_feat.primitiveFragmentShadingRate = VK_FALSE;
            vkb_phys.enable_extension_if_present(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
            vrs_supported_ = true;
            log::infof("VRS supported — tile %ux%u",
                       vrs_texel_size_.width, vrs_texel_size_.height);
        } else {
            log::info("VRS extension present but attachmentFragmentShadingRate "
                      "feature not exposed — disabling");
        }
    } else {
        log::info("VRS extension not present — full-rate shading everywhere");
    }

    // VK_EXT_device_fault (optional) — lets us call vkGetDeviceFaultInfoEXT
    // on a device-lost to retrieve the faulting GPU address + vendor
    // fault code. Pure diagnostics; no runtime cost when no fault occurs.
    VkPhysicalDeviceFaultFeaturesEXT fault_feat{};
    fault_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
    if (vkb_phys.is_extension_present(VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 ff2{};
        ff2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        ff2.pNext = &fault_feat;
        vkGetPhysicalDeviceFeatures2(physical_device_, &ff2);
        if (fault_feat.deviceFault == VK_TRUE) {
            vkb_phys.enable_extension_if_present(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
            device_fault_supported_ = true;
            log::info("VK_EXT_device_fault supported — device-lost will dump GPU fault info");
        }
    }
    if (!device_fault_supported_) {
        log::info("VK_EXT_device_fault not present — device-lost dumps CPU stack only");
    }

    vkb::DeviceBuilder dev_builder{ vkb_phys };
    if (vrs_supported_) dev_builder.add_pNext(&vrs_feat);
    if (device_fault_supported_) dev_builder.add_pNext(&fault_feat);
    // FSR3 SwapChain proxy needs FOUR distinct VkQueue handles
    // (game/asyncCompute/present/imageAcquire). Request 4 queues from
    // the graphics family so we can hand 3 distinct handles plus the
    // compute family's queue to the SDK. Discrete GPUs typically expose
    // 16 queues per family; iGPUs may expose only 1, in which case FG
    // is unavailable but the rest of the engine works.
    {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());
        uint32_t gfx_family = 0;
        for (uint32_t i = 0; i < qf_count; ++i) {
            if (qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { gfx_family = i; break; }
        }
        const uint32_t gfx_count = qf_props[gfx_family].queueCount;
        const uint32_t want = (gfx_count >= 4) ? 4u : 1u;
        std::vector<vkb::CustomQueueDescription> qsetup;
        qsetup.emplace_back(gfx_family, std::vector<float>(want, 1.0f));
        // Add the compute family if it's distinct (NVIDIA RTX has one).
        for (uint32_t i = 0; i < qf_count; ++i) {
            if (i == gfx_family) continue;
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qsetup.emplace_back(i, std::vector<float>{1.0f});
                break;
            }
        }
        dev_builder.custom_queue_setup(qsetup);
        log::infof("queue setup: graphics family %u × %u (avail %u), plus %zu more family queue(s)",
                   gfx_family, want, gfx_count, qsetup.size() - 1);
    }
    auto dev_ret = dev_builder.build();
    if (!dev_ret) throw std::runtime_error("vkb device build: " +
                                           dev_ret.error().message());
    auto vkb_dev = dev_ret.value();
    device_ = vkb_dev.device;
    graphics_queue_ = vkb_dev.get_queue(vkb::QueueType::graphics).value();
    graphics_queue_family_ = vkb_dev.get_queue_index(vkb::QueueType::graphics).value();
    log::infof("vk device created, graphics queue family=%u", graphics_queue_family_);

    // Async-compute queue. vk-bootstrap returns a queue from a dedicated
    // compute-only family if available (NVIDIA RTX cards expose one with 8
    // queues), falling back to the graphics family otherwise. We track
    // whether the queues are distinct — if not, the "async TLAS build"
    // path simply records onto the same queue and gains nothing, which is
    // fine and means no behavior change on weaker hardware.
    auto compute_q = vkb_dev.get_queue(vkb::QueueType::compute);
    auto compute_qi = vkb_dev.get_queue_index(vkb::QueueType::compute);
    if (compute_q && compute_qi) {
        compute_queue_ = compute_q.value();
        compute_queue_family_ = compute_qi.value();
    } else {
        compute_queue_ = graphics_queue_;
        compute_queue_family_ = graphics_queue_family_;
    }
    compute_queue_distinct_ = (compute_queue_family_ != graphics_queue_family_);
    log::infof("vk compute queue family=%u (%s)", compute_queue_family_,
               compute_queue_distinct_ ? "ASYNC — separate from graphics"
                                       : "shared with graphics");

    // Pull the extra graphics-family queues for the FSR3 SwapChain
    // proxy. Indices 1..3 are populated only when custom_queue_setup
    // requested 4 graphics queues (gfx_count >= 4). On hardware where
    // we got only 1, these stay VK_NULL_HANDLE and FG is unavailable.
    fsr3_extra_queue_present_ = VK_NULL_HANDLE;
    fsr3_extra_queue_acquire_ = VK_NULL_HANDLE;
    fsr3_extra_queue_compute_ = VK_NULL_HANDLE;
    {
        VkQueue q1 = VK_NULL_HANDLE, q2 = VK_NULL_HANDLE, q3 = VK_NULL_HANDLE;
        vkGetDeviceQueue(device_, graphics_queue_family_, 1, &q1);
        vkGetDeviceQueue(device_, graphics_queue_family_, 2, &q2);
        vkGetDeviceQueue(device_, graphics_queue_family_, 3, &q3);
        fsr3_extra_queue_present_ = q1;
        fsr3_extra_queue_acquire_ = q2;
        // Compute role for SDK: if we have 4 graphics queues, take the
        // 4th and route the SDK to it (keeps async-TLAS on its dedicated
        // family). Otherwise reuse our compute_queue_ as before.
        fsr3_extra_queue_compute_ = q3 ? q3 : compute_queue_;
        log::infof("[fsr3 sc] extra graphics queues: present=%p acquire=%p compute=%p",
                   (void*)q1, (void*)q2, (void*)q3);
    }

    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = physical_device_;
    aci.device = device_;
    aci.instance = instance_;
    aci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vk_check(vmaCreateAllocator(&aci, &allocator_), "vmaCreateAllocator");

    load_rt_functions(device_);
    log::info("RT extension function pointers loaded");
}

void VulkanEngine::init_swapchain() {
    uint32_t w = 0, h = 0;
    window_->framebuffer_size(w, h);

    vkb::SwapchainBuilder sb{ physical_device_, device_, surface_ };
    auto sc_ret = sb
        .set_desired_format(VkSurfaceFormatKHR{ VK_FORMAT_B8G8R8A8_UNORM,
                                                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(w, h)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .build();
    if (!sc_ret) throw std::runtime_error("swapchain build: " + sc_ret.error().message());

    auto vkb_sc = sc_ret.value();
    swapchain_ = vkb_sc.swapchain;
    swapchain_extent_ = vkb_sc.extent;
    swapchain_format_ = vkb_sc.image_format;
    swapchain_images_ = vkb_sc.get_images().value();
    swapchain_views_ = vkb_sc.get_image_views().value();

    // Derive render_extent_ from swapchain × user's render_scale slider. Min
    // 1×1 (sub-pixel scales would produce 0-extent images, which is invalid
    // and would crash the image-create call). Round to nearest int.
    {
        float sc = std::max(0.1f, rt_.render_scale);
        uint32_t rw = std::max<uint32_t>(1,
            static_cast<uint32_t>(static_cast<float>(swapchain_extent_.width)  * sc + 0.5f));
        uint32_t rh = std::max<uint32_t>(1,
            static_cast<uint32_t>(static_cast<float>(swapchain_extent_.height) * sc + 0.5f));
        render_extent_ = { rw, rh };
        log::infof("render_extent: %ux%u (swapchain %ux%u × %.2f scale)",
                   render_extent_.width, render_extent_.height,
                   swapchain_extent_.width, swapchain_extent_.height, sc);
    }

    auto sem_info = vkinit::semaphore_create_info();
    render_semaphores_.assign(swapchain_images_.size(), VK_NULL_HANDLE);
    for (auto& s : render_semaphores_) {
        vk_check(vkCreateSemaphore(device_, &sem_info, nullptr, &s),
                 "vkCreateSemaphore render (per-image)");
    }
    log::infof("swapchain: %ux%u, %u images, format=%d",
               swapchain_extent_.width, swapchain_extent_.height,
               static_cast<unsigned>(swapchain_images_.size()),
               static_cast<int>(swapchain_format_));
}

void VulkanEngine::destroy_swapchain() {
    // When the FSR3 SwapChain proxy is active, swapchain_ is a wrapped
    // handle internally pointing at FrameInterpolationSwapChainVK*.
    // Calling the standard vkDestroySwapchainKHR on it AVs because
    // the loader expects a real Vulkan swapchain. Tear the proxy down
    // first via the SDK's DestroyContext — that frees the wrapped
    // swapchain and nulls our handle, then the rest of this function
    // skips the (now-NULL) vkDestroySwapchainKHR call below.
    if (fsr3_swapchain_active_) {
        destroy_fsr3_swapchain();
    }
    for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
    swapchain_views_.clear();
    swapchain_images_.clear();
    for (auto s : render_semaphores_) {
        if (s) vkDestroySemaphore(device_, s, nullptr);
    }
    render_semaphores_.clear();
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void VulkanEngine::init_depth_image() {
    // Depth attached to the world raster pass — sized to render_extent_ so
    // it matches scene_color / motion_vec. Compose runs at swapchain_extent_
    // and doesn't sample depth-image past TAA (which reads at render res).
    VkImageCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = depth_format_,
        .extent = { render_extent_.width, render_extent_.height, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &depth_image_, &depth_alloc_, nullptr),
             "vmaCreateImage depth");

    VkImageViewCreateInfo vci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .image = depth_image_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = depth_format_,
        .components = {},
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &depth_view_),
             "vkCreateImageView depth");
    log::info("depth image created");
}

void VulkanEngine::destroy_depth_image() {
    if (depth_view_) vkDestroyImageView(device_, depth_view_, nullptr);
    if (depth_image_) vmaDestroyImage(allocator_, depth_image_, depth_alloc_);
    depth_view_ = VK_NULL_HANDLE;
    depth_image_ = VK_NULL_HANDLE;
    depth_alloc_ = nullptr;
}

void VulkanEngine::recreate_swapchain() {
    uint32_t w = 0, h = 0;
    window_->framebuffer_size(w, h);
    while (w == 0 || h == 0) {
        glfwWaitEvents();
        window_->framebuffer_size(w, h);
    }
    vkDeviceWaitIdle(device_);
    destroy_swapchain();
    destroy_depth_image();
    init_swapchain();
    init_depth_image();
    if (taa_desc_pool_) recreate_taa_targets();
    // TAAU targets depend on swapchain_extent_; rebuild after swapchain change.
    if (taau_desc_pool_) recreate_taau_targets();
    // Bloom mip 0 is sized to render_extent_/2; if we don't recreate it
    // alongside the swapchain, compose samples a stale-sized texture and
    // bloom appears offset against everything else. The compose desc-set's
    // image bindings (history at 0, depth at 1, bloom at 3) all reference
    // image views that were just destroyed, so rebind them.
    if (bloom_image_) recreate_bloom_targets();
    // LR raymarch targets are sized at render_extent_ × scale; rebuild
    // them too so the upscale compose continues to read valid storage.
    if (shadow_lr_image_) {
        destroy_shadow_lr();
        init_shadow_lr();
    }
    if (svgf_gi_image_) {
        // recreate_svgf_targets also rewrites scene_desc binding 19
        // so it points at the new view.
        recreate_svgf_targets();
    }
    if (tr_lr_color_image_) {
        destroy_terrain_raymarch_lowres();
        init_terrain_raymarch_lowres();
    }
    // VRS attachment is sized in LR-tile units — depends on tr_lr_extent_.
    if (vrs_supported_) recreate_vrs_attachment();
    // FSR2 bakes maxRenderSize/displaySize into its internal allocations,
    // so any swapchain change requires a context recreate. No-op when the
    // SDK isn't compiled in or the context never came up.
    recreate_fsr2_context();
    recreate_fsr3_context();
    // FG context bakes display size into its allocations the same way
    // the upscaler does — recreate when the swapchain changes. No-op
    // when FG was never enabled.
    if (fsr3_fg_context_valid_) { destroy_fsr3_fg(); init_fsr3_fg(); }
    // ReSTIR reservoir buffers are sized to render_extent_ and rebound
    // on scene_desc bindings 15/16. recreate_restir_buffers handles the
    // destroy + alloc + descriptor rewrite atomically.
    recreate_restir_buffers();
    rewrite_compose_image_bindings();

    VkDeviceSize needed = static_cast<VkDeviceSize>(swapchain_extent_.width) *
                          swapchain_extent_.height * 4;
    if (needed > readback_size_) {
        destroy_readback_buffer();
        init_readback_buffer();
    }
    resize_requested_ = false;
}

void VulkanEngine::init_commands() {
    auto pool_info = vkinit::command_pool_create_info(
        graphics_queue_family_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    auto compute_pool_info = vkinit::command_pool_create_info(
        compute_queue_family_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    for (auto& f : frames_) {
        vk_check(vkCreateCommandPool(device_, &pool_info, nullptr, &f.command_pool),
                 "vkCreateCommandPool");
        auto alloc_info = vkinit::command_buffer_allocate_info(f.command_pool, 1);
        vk_check(vkAllocateCommandBuffers(device_, &alloc_info, &f.command_buffer),
                 "vkAllocateCommandBuffers");

        // Compute-side: separate pool because Vulkan command pools are bound
        // to one queue family. If the compute family equals the graphics
        // family we still allocate a separate pool/buffer pair for clarity
        // (and to avoid threading two command-buffer recordings through one
        // pool, which would force serial recording).
        vk_check(vkCreateCommandPool(device_, &compute_pool_info, nullptr, &f.compute_pool),
                 "vkCreateCommandPool compute");
        auto compute_alloc = vkinit::command_buffer_allocate_info(f.compute_pool, 1);
        vk_check(vkAllocateCommandBuffers(device_, &compute_alloc, &f.compute_buffer),
                 "vkAllocateCommandBuffers compute");
    }
}

void VulkanEngine::init_sync() {
    auto fence_info = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    auto sem_info = vkinit::semaphore_create_info();
    for (auto& f : frames_) {
        vk_check(vkCreateFence(device_, &fence_info, nullptr, &f.render_fence),
                 "vkCreateFence");
        vk_check(vkCreateSemaphore(device_, &sem_info, nullptr, &f.swapchain_semaphore),
                 "vkCreateSemaphore swapchain");
        // Signaled by the compute queue's TLAS rebuild submission, awaited
        // by the graphics queue before fragment shaders sample the TLAS.
        vk_check(vkCreateSemaphore(device_, &sem_info, nullptr, &f.tlas_build_done),
                 "vkCreateSemaphore tlas_build_done");
    }
}

void VulkanEngine::init_gpu_query_pool() {
    // Read the device's timestamp period (ns per tick) and capture it
    // for results-conversion. Devices that don't support graphics-queue
    // timestamps would have timestampComputeAndGraphics=false; on every
    // mainstream desktop GPU it's true so we don't gate on the bit.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    gpu_timestamp_period_ns_ = props.limits.timestampPeriod;
    if (gpu_timestamp_period_ns_ <= 0.0f) {
        log::warn("[gpu-watchdog] device reports timestampPeriod=0 — GPU "
                  "timing disabled");
        return;
    }
    VkQueryPoolCreateInfo qpci{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = static_cast<uint32_t>(kQueriesPerFrame * kFrameOverlap),
        .pipelineStatistics = 0,
    };
    vk_check(vkCreateQueryPool(device_, &qpci, nullptr, &gpu_query_pool_),
             "vkCreateQueryPool gpu");
    // NO host-side vkResetQueryPool here. That call requires the
    // OPTIONAL hostQueryReset feature to be explicitly enabled at
    // device creation (it is NOT implied by Vulkan 1.2/1.3 — the
    // earlier comment was wrong). Calling it without the feature is
    // undefined behaviour and validation flagged it
    // (VUID-vkResetQueryPool-None-02665) — a real device-lost risk.
    // The per-frame vkCmdResetQueryPool in draw() (device-side,
    // always legal) resets each slot before it's written, and
    // gpu_query_recorded_mask_ blocks any read before the first
    // write, so no host reset is needed at all.
    gpu_query_recorded_mask_ = 0;
    log::infof("[gpu-watchdog] query pool created (%u queries, %.1f ns/tick)",
               kQueriesPerFrame * kFrameOverlap, gpu_timestamp_period_ns_);
}

void VulkanEngine::destroy_gpu_query_pool() {
    if (gpu_query_pool_) {
        vkDestroyQueryPool(device_, gpu_query_pool_, nullptr);
        gpu_query_pool_ = VK_NULL_HANDLE;
    }
    gpu_query_recorded_mask_ = 0;
}

void VulkanEngine::read_gpu_timestamps_for_slot(uint32_t slot) {
    // Skip until this slot has been written at least once (first
    // kFrameOverlap frames). The mask bit goes high when draw() records
    // the timestamps for this slot; it stays high for the engine's
    // lifetime since we always re-record each frame.
    if (!gpu_query_pool_) return;
    if ((gpu_query_recorded_mask_ & (uint64_t{1} << slot)) == 0) return;

    // draw() only writes the Scene begin/end pair (kStageScene*2,
    // +1) — TLAS/Taa/BlitUi slots are reset but never written. The
    // earlier code queried all kQueriesPerFrame at once; the 6
    // unwritten queries are "unavailable" so vkGetQueryPoolResults
    // returned VK_NOT_READY and we bailed EVERY frame → the HUD
    // permanently showed 0.00 ms and the TDR watchdog could never
    // fire. Query ONLY the 2 Scene timestamps that are actually
    // written. base_scene = slot's Scene-pair start index.
    const uint32_t base_scene = slot *
        static_cast<uint32_t>(kQueriesPerFrame) +
        static_cast<uint32_t>(kStageScene) * 2u;
    uint64_t ts2[2] = {};
    VkResult r = vkGetQueryPoolResults(device_, gpu_query_pool_,
                                        base_scene, 2,
                                        sizeof(ts2), ts2, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return;

    float scene_ms = 0.0f;
    if (ts2[1] > ts2[0]) {
        scene_ms = static_cast<float>(
            static_cast<double>(ts2[1] - ts2[0]) *
            static_cast<double>(gpu_timestamp_period_ns_) * 1.0e-6);
    }
    // TLAS/Taa/BlitUi stages aren't individually instrumented yet —
    // the Scene pair brackets the whole frame command buffer, so
    // scene_ms IS the total frame GPU time. Report 0 for the others.
    float tlas_ms    = 0.0f;
    float taa_ms     = 0.0f;
    float blit_ui_ms = 0.0f;
    // EMA so a single spike doesn't dominate the HUD readout. The
    // raw-frame TDR check uses the un-smoothed total below.
    auto ema = [](float prev, float cur) {
        return (prev <= 0.0f) ? cur : (prev * 0.85f + cur * 0.15f);
    };
    gpu_timers_.tlas_ms    = ema(gpu_timers_.tlas_ms,    tlas_ms);
    gpu_timers_.scene_ms   = ema(gpu_timers_.scene_ms,   scene_ms);
    gpu_timers_.taa_ms     = ema(gpu_timers_.taa_ms,     taa_ms);
    gpu_timers_.blit_ui_ms = ema(gpu_timers_.blit_ui_ms, blit_ui_ms);

    float total_raw = tlas_ms + scene_ms + taa_ms + blit_ui_ms;
    // Windows TDR is ~2 s per submit. Warn at 1.5 s so we have signal
    // BEFORE the device-lost fires. Use the raw single-frame total,
    // not the EMA — a single 1.8 s frame will still TDR even if the
    // average is fine.
    if (total_raw > 1500.0f) {
        log::warnf("[gpu-watchdog] frame GPU=%.0f ms (tlas=%.0f scene=%.0f "
                   "taa=%.0f blit=%.0f) — approaching Windows TDR (2000 ms). "
                   "Lower render scale / disable RT to recover.",
                   total_raw, tlas_ms, scene_ms, taa_ms, blit_ui_ms);
    }
}

void VulkanEngine::init_readback_buffer() {
    readback_size_ = static_cast<VkDeviceSize>(swapchain_extent_.width) *
                     swapchain_extent_.height * 4;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .size = readback_size_,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                             &readback_buffer_, &readback_alloc_, nullptr),
             "vmaCreateBuffer readback");
    log::infof("readback buffer allocated (%llu bytes)",
               static_cast<unsigned long long>(readback_size_));
}

void VulkanEngine::destroy_readback_buffer() {
    if (readback_buffer_) {
        vmaDestroyBuffer(allocator_, readback_buffer_, readback_alloc_);
        readback_buffer_ = VK_NULL_HANDLE;
        readback_alloc_ = nullptr;
        readback_size_ = 0;
    }
}

void VulkanEngine::init_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    vert_module_ = vkpipe::load_shader_module(device_, sd + "/cube.vert.spv");
    frag_module_ = vkpipe::load_shader_module(device_, sd + "/cube.frag.spv");

    VkPushConstantRange pc{};
    // Tessellation control/eval stages read pc (mvp/model/prev_mvp) for the
    // near-terrain tess pipeline, so the range must advertise them too —
    // referencing an unbacked stage is the same spec violation that bit
    // shadow.vert (VUID-...-10069).
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    pc.offset = 0;
    pc.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &scene_desc_set_layout_,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
    };
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &pipeline_layout_),
             "vkCreatePipelineLayout");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = vert_module_;
    cfg.frag = frag_module_;
    cfg.layout = pipeline_layout_;
    // Two color attachments: location 0 = HDR scene color, location 1 = RG16F
    // motion vector. cube.frag emits real per-instance motion via prev_mvp.
    cfg.color_formats = { scene_color_format_, motion_vec_format_ };
    cfg.depth_format = depth_format_;

    VkVertexInputBindingDescription vb{};
    vb.binding = 0;
    vb.stride = sizeof(Vertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    cfg.vbindings.push_back(vb);

    VkVertexInputAttributeDescription a0{};
    a0.location = 0; a0.binding = 0;
    a0.format = VK_FORMAT_R32G32B32_SFLOAT;
    a0.offset = offsetof(Vertex, position);
    VkVertexInputAttributeDescription a1{};
    a1.location = 1; a1.binding = 0;
    a1.format = VK_FORMAT_R32G32B32_SFLOAT;
    a1.offset = offsetof(Vertex, normal);
    VkVertexInputAttributeDescription a2{};
    a2.location = 2; a2.binding = 0;
    a2.format = VK_FORMAT_R32G32_SFLOAT;
    a2.offset = offsetof(Vertex, uv);
    cfg.vattrs = { a0, a1, a2 };
    // Alpha blending DISABLED for now while diagnosing a "gray floor"
    // flicker on the castle ground when walking backwards. The blend
    // existed to let SPOM wall silhouette pixels (alpha = 0) show
    // terrain through brick cavities, but it appears to also produce
    // slight color drift on floor pixels under TAA + motion. With
    // blend off, opaque cube fragments are unchanged; only the
    // wall-silhouette cavity transparency is lost — visible as the
    // brick edge clipping at the geometric corner instead of
    // showing-through.
    cfg.alpha_blend_color0_only = false;

    pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

    // Depth-only pre-pass pipeline. draw() runs render_world_depth_pass()
    // before the color pass to populate depth_image_ so the color pass's
    // depth_compare=LESS_OR_EQUAL early-rejects occluded fragments before
    // cube.frag's heavy inline-RT body executes. Particles, projectiles and
    // the viewmodel skip the prepass (sparse / screen-space).
    {
        std::string sd2 = QLIKE_SHADER_DIR;
        depth_frag_module_ = vkpipe::load_shader_module(device_, sd2 + "/cube_depth.frag.spv");
        depth_vert_module_ = vkpipe::load_shader_module(device_, sd2 + "/cube_depth.vert.spv");
    }
    vkpipe::GraphicsPipelineConfig dcfg = cfg;
    // IMPORTANT: depth pre-pass uses the SAME vert shader as the color
    // pass. Splitting it into a minimal cube_depth.vert sounds like a
    // perf win (no varyings, fewer matrix multiplies) but the SPIR-V
    // optimiser produced gl_Position values that differed from
    // cube.vert by 1 ULP — enough that the color pass's
    // LESS_OR_EQUAL test against the prepass depth flipped per-pixel
    // every frame on the castle floor. Symptom: gray-blue (sky clear
    // colour) flicker on close-by floor pixels when the camera moves.
    dcfg.vert = vert_module_;
    dcfg.frag = depth_frag_module_;
    // Depth-only pre-pass: no color attachments at all. Clear the inherited
    // color_formats vector — otherwise build_graphics_pipeline would pick its
    // size over color_attachment_count and we'd end up with a pipeline that
    // declares 2 color attachments while the prepass render-pass has none.
    dcfg.color_formats.clear();
    dcfg.color_attachment_count = 0;
    dcfg.depth_write = true;
    dcfg.depth_test = true;
    dcfg.depth_compare = VK_COMPARE_OP_LESS;
    depth_pipeline_ = vkpipe::build_graphics_pipeline(device_, dcfg);
    log::info("graphics + depth-prepass pipelines built");
}

void VulkanEngine::init_terrain_pipelines() {
    std::string sd = QLIKE_SHADER_DIR;
    terrain_vert_module_ = vkpipe::load_shader_module(device_, sd + "/terrain.vert.spv");

    // Vertex bindings: 0 = Vertex (pos/normal/uv), 1 = parent_y (float).
    VkVertexInputBindingDescription b0{};
    b0.binding = 0; b0.stride = sizeof(Vertex);
    b0.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputBindingDescription b1{};
    b1.binding = 1; b1.stride = sizeof(float);
    b1.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    auto attr = [](uint32_t loc, uint32_t binding, VkFormat fmt, uint32_t off) {
        VkVertexInputAttributeDescription a{};
        a.location = loc; a.binding = binding; a.format = fmt; a.offset = off;
        return a;
    };
    std::vector<VkVertexInputAttributeDescription> vattrs = {
        attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)),
        attr(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
        attr(2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)),
        attr(3, 1, VK_FORMAT_R32_SFLOAT,       0),
    };

    // Color pass — same color/depth formats as cube pipeline.
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = terrain_vert_module_;
        cfg.frag = frag_module_;       // shared cube.frag
        cfg.layout = pipeline_layout_;
        cfg.color_formats = { scene_color_format_, motion_vec_format_ };
        cfg.depth_format = depth_format_;
        cfg.vbindings = { b0, b1 };
        cfg.vattrs = vattrs;
        terrain_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
        // Wireframe variant (debug mesh-density view).
        {
            vkpipe::GraphicsPipelineConfig wcfg = cfg;
            wcfg.polygon_mode = VK_POLYGON_MODE_LINE;
            wcfg.cull = VK_CULL_MODE_NONE;
            terrain_wire_pipeline_ = vkpipe::build_graphics_pipeline(device_, wcfg);
        }
    }

    // Depth pre-pass — same morph as color pass so LESS_OR_EQUAL passes.
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = terrain_vert_module_;
        cfg.frag = depth_frag_module_;
        cfg.layout = pipeline_layout_;
        cfg.color_attachment_count = 0;
        cfg.depth_format = depth_format_;
        cfg.depth_compare = VK_COMPARE_OP_LESS;
        cfg.vbindings = { b0, b1 };
        cfg.vattrs = vattrs;
        terrain_depth_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    // Near-terrain GPU tessellation pipeline. Triangle patches (3 CP) so
    // it reuses the chunk's LOD0 triangle index buffer as a PATCH_LIST.
    // Only binding 0 (pos/normal/uv) — no parent_y / morph (near chunks
    // are full LOD0). Distance-adaptive subdivision in .tesc, detail
    // displacement + normal recompute in .tese, shared cube.frag.
    {
        terrain_tess_vert_module_ = vkpipe::load_shader_module(device_, sd + "/terrain_tess.vert.spv");
        terrain_tesc_module_      = vkpipe::load_shader_module(device_, sd + "/terrain_tess.tesc.spv");
        terrain_tese_module_      = vkpipe::load_shader_module(device_, sd + "/terrain_tess.tese.spv");
        std::vector<VkVertexInputAttributeDescription> tattrs = {
            attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)),
            attr(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
            attr(2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)),
        };
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = terrain_tess_vert_module_;
        cfg.tesc = terrain_tesc_module_;
        cfg.tese = terrain_tese_module_;
        cfg.patch_control_points = 3;
        cfg.frag = frag_module_;       // shared cube.frag
        // Cull NONE. The tessellator's primitive winding through the
        // reused triangle index buffer (with fractional_odd spacing)
        // does NOT reliably match the rasteriser's front-face test, so
        // back-face culling silently dropped ~half the patches → black
        // gaps + "no tessellation visible". Disabling cull renders every
        // patch; the .tese computes the surface normal analytically (not
        // from gl_FrontFacing), so both facings shade identically — no
        // back-face darkening, just a benign same-depth double-shade.
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.layout = pipeline_layout_;
        cfg.color_formats = { scene_color_format_, motion_vec_format_ };
        cfg.depth_format = depth_format_;
        cfg.vbindings = { b0 };
        cfg.vattrs = tattrs;
        terrain_tess_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
        // Wireframe variant — shows every tessellated sub-triangle.
        {
            vkpipe::GraphicsPipelineConfig wcfg = cfg;
            wcfg.polygon_mode = VK_POLYGON_MODE_LINE;
            terrain_tess_wire_pipeline_ = vkpipe::build_graphics_pipeline(device_, wcfg);
        }

        // Tessellation DEPTH-prepass pipeline — same vert/tesc/tese so
        // the primed depth is bit-identical to the color pass (priming
        // near chunks with the plain terrain.vert instead left a 1-ULP
        // gl_Position delta → LESS_OR_EQUAL rejects → sky shows through).
        vkpipe::GraphicsPipelineConfig dcfg = cfg;
        dcfg.frag = depth_frag_module_;
        dcfg.color_formats.clear();
        dcfg.color_attachment_count = 0;
        dcfg.depth_compare = VK_COMPARE_OP_LESS;
        terrain_tess_depth_pipeline_ = vkpipe::build_graphics_pipeline(device_, dcfg);
    }
    log::info("terrain pipelines built (CD-LOD morph + near tessellation)");
}

void VulkanEngine::destroy_terrain_pipelines() {
    if (terrain_pipeline_)       vkDestroyPipeline(device_, terrain_pipeline_, nullptr);
    if (terrain_depth_pipeline_) vkDestroyPipeline(device_, terrain_depth_pipeline_, nullptr);
    if (terrain_tess_pipeline_)  vkDestroyPipeline(device_, terrain_tess_pipeline_, nullptr);
    if (terrain_tess_depth_pipeline_) vkDestroyPipeline(device_, terrain_tess_depth_pipeline_, nullptr);
    if (terrain_wire_pipeline_)  vkDestroyPipeline(device_, terrain_wire_pipeline_, nullptr);
    if (terrain_tess_wire_pipeline_) vkDestroyPipeline(device_, terrain_tess_wire_pipeline_, nullptr);
    if (terrain_vert_module_)    vkDestroyShaderModule(device_, terrain_vert_module_, nullptr);
    if (terrain_tess_vert_module_) vkDestroyShaderModule(device_, terrain_tess_vert_module_, nullptr);
    if (terrain_tesc_module_)    vkDestroyShaderModule(device_, terrain_tesc_module_, nullptr);
    if (terrain_tese_module_)    vkDestroyShaderModule(device_, terrain_tese_module_, nullptr);
    terrain_pipeline_ = terrain_depth_pipeline_ = terrain_tess_pipeline_ = VK_NULL_HANDLE;
    terrain_vert_module_ = terrain_tess_vert_module_ = VK_NULL_HANDLE;
    terrain_tesc_module_ = terrain_tese_module_ = VK_NULL_HANDLE;
}

void VulkanEngine::init_terrain_raymarch_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    terrain_raymarch_vert_module_ = vkpipe::load_shader_module(
        device_, sd + "/terrain_raymarch.vert.spv");
    terrain_raymarch_frag_module_ = vkpipe::load_shader_module(
        device_, sd + "/terrain_raymarch.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = terrain_raymarch_vert_module_;
    cfg.frag = terrain_raymarch_frag_module_;
    cfg.layout = pipeline_layout_;          // shares the cube push-constant layout
    cfg.color_formats = { scene_color_format_, motion_vec_format_ };
    cfg.depth_format = depth_format_;
    // No vertex bindings — the vert builds NDC corners from gl_VertexIndex.
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    // LESS_OR_EQUAL so gl_FragDepth = far_plane fragments don't get
    // rejected by the cleared depth buffer (which is also 1.0).
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    // Attachment-based VRS — center 1×1, edges 2×2, corners 4×4. Pipeline
    // takes the per-pixel rate from the attachment bound at draw begin.
    cfg.enable_vrs = vrs_supported_;

    terrain_raymarch_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::infof("terrain raymarch pipeline built (vrs=%s)",
               vrs_supported_ ? "on" : "off");
}

void VulkanEngine::destroy_terrain_raymarch_pipeline() {
    if (terrain_raymarch_pipeline_) {
        vkDestroyPipeline(device_, terrain_raymarch_pipeline_, nullptr);
        terrain_raymarch_pipeline_ = VK_NULL_HANDLE;
    }
    if (terrain_raymarch_vert_module_) {
        vkDestroyShaderModule(device_, terrain_raymarch_vert_module_, nullptr);
        terrain_raymarch_vert_module_ = VK_NULL_HANDLE;
    }
    if (terrain_raymarch_frag_module_) {
        vkDestroyShaderModule(device_, terrain_raymarch_frag_module_, nullptr);
        terrain_raymarch_frag_module_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::init_terrain_water_pipeline() {
    // 64x64 subdivided water grid covering +-2048 m (~64 m per cell).
    // Was a single 2-triangle quad spanning +-8192 m. Perspective-
    // correct interpolation of the world-position varying (vWPos)
    // across triangles that huge has bounded float32 precision in
    // the barycentric weights. At the shoreline (~50-200 m from
    // camera) that translated to ~cm-scale per-frame wobble of
    // wpos.xz as the camera changed, which the user perceived as
    // "the water height jumps" since meshHeight(wpos.xz) -> shore
    // alpha is sensitive to small xz shifts at the iso-line.
    // Smaller triangles bound the barycentric precision tightly
    // (per-triangle vert distance <= 64 m means barycentric error
    // -> world error stays <= mm).
    constexpr int   kGrid = 64;             // 64x64 quads = 65x65 verts
    constexpr float kHalf = 2048.0f;        // +-2048 m total span
    constexpr float kStep = (2.0f * kHalf) / float(kGrid);
    std::vector<Vertex> verts;
    verts.reserve((kGrid + 1) * (kGrid + 1));
    for (int j = 0; j <= kGrid; ++j) {
        for (int i = 0; i <= kGrid; ++i) {
            float x = -kHalf + float(i) * kStep;
            float z = -kHalf + float(j) * kStep;
            Vertex vt{};
            vt.position = { x, 0.0f, z };
            vt.normal   = { 0.0f, 1.0f, 0.0f };
            vt.uv       = { float(i) / float(kGrid),
                            float(j) / float(kGrid) };
            verts.push_back(vt);
        }
    }
    std::vector<uint32_t> indices;
    indices.reserve(kGrid * kGrid * 6);
    for (int j = 0; j < kGrid; ++j) {
        for (int i = 0; i < kGrid; ++i) {
            uint32_t a = uint32_t(j * (kGrid + 1) + i);
            uint32_t b = a + 1;
            uint32_t c = a + (kGrid + 1);
            uint32_t d = c + 1;
            indices.push_back(a); indices.push_back(b); indices.push_back(d);
            indices.push_back(a); indices.push_back(d); indices.push_back(c);
        }
    }
    water_plane_mesh_ = create_mesh_from_data(
        device_, allocator_, graphics_queue_, graphics_queue_family_,
        verts.data(), static_cast<uint32_t>(verts.size()),
        indices.data(), static_cast<uint32_t>(indices.size()));

    std::string sd = QLIKE_SHADER_DIR;
    water_vert_module_ = vkpipe::load_shader_module(
        device_, sd + "/water.vert.spv");
    water_frag_module_ = vkpipe::load_shader_module(
        device_, sd + "/water.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = water_vert_module_;
    cfg.frag = water_frag_module_;
    cfg.layout = pipeline_layout_;          // shares the cube push-constant layout
    cfg.color_formats = { scene_color_format_, motion_vec_format_ };
    cfg.depth_format = depth_format_;

    VkVertexInputBindingDescription vb{};
    vb.binding = 0;
    vb.stride = sizeof(Vertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    cfg.vbindings.push_back(vb);
    VkVertexInputAttributeDescription a0{};
    a0.location = 0; a0.binding = 0;
    a0.format = VK_FORMAT_R32G32B32_SFLOAT;
    a0.offset = offsetof(Vertex, position);
    cfg.vattrs = { a0 };

    cfg.cull = VK_CULL_MODE_NONE;           // visible from above and below
    // depth_test ON — the only reliable way to hide water at pixels
    // where terrain is in front. Multiple software approaches
    // (12-step ray-vs-heightmap, adaptive 16-48 step, with/without
    // bias) all had failure modes: foam floating above terrain,
    // residual see-through, killed shore band, etc. The hardware
    // depth-test silhouette wobbles per TAA jitter, which is mostly
    // hidden by the +2 m water-plane lift in water.vert — wobble
    // lives in the transparent band where shore_alpha = 0.
    cfg.depth_test = true;
    cfg.depth_write = false;
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    // Depth bias enabled so the water plane sits "slightly behind" the
    // terrain mesh at every coplanar pixel. Without bias, the flat
    // water plane and the displaced terrain produce identical NDC
    // depth at their geometric intersection — perspective-interp
    // precision differences between a 2-triangle plane and a finely
    // tessellated mesh flip depth-test pass/fail per pixel as the
    // camera moves, and that flipping reads to the user as "the
    // water height is jumping" at the shoreline. With +bias terrain
    // wins all ties, the visible water/land boundary is the heightmap
    // iso-line decided shader-side via shore_alpha, frame-stable.
    cfg.depth_bias_enable = true;
    // Alpha-blend colour only (motion stays opaque for TAA). The frag
    // feathers the water alpha to 0 over the last ~0.35 m of depth at
    // the shoreline so the terrain↔water plane intersection dissolves
    // into the wet shore instead of z-fighting into a jittery line.
    cfg.alpha_blend_color0_only = true;
    cfg.enable_vrs = vrs_supported_;

    terrain_water_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::infof("rasterised water-plane pipeline built (vrs=%s)",
               vrs_supported_ ? "on" : "off");
}

void VulkanEngine::destroy_terrain_water_pipeline() {
    if (terrain_water_pipeline_) {
        vkDestroyPipeline(device_, terrain_water_pipeline_, nullptr);
        terrain_water_pipeline_ = VK_NULL_HANDLE;
    }
    if (water_vert_module_) {
        vkDestroyShaderModule(device_, water_vert_module_, nullptr);
        water_vert_module_ = VK_NULL_HANDLE;
    }
    if (water_frag_module_) {
        vkDestroyShaderModule(device_, water_frag_module_, nullptr);
        water_frag_module_ = VK_NULL_HANDLE;
    }
    destroy_mesh(allocator_, water_plane_mesh_);
}

// === Low-res raymarch upscale targets ====================================
//
// Three images sized at render_extent_ × terrain_raymarch_scale: HDR
// color, motion-vec, and a D32 depth. The raymarch pipeline draws into
// these in a separate render pass; the compose pipeline samples them
// (bilinear) and writes upscaled color / motion / gl_FragDepth into
// scene_color / motion_vec_view / depth_view at the start of the main
// world pass with depth-test LESS_OR_EQUAL so the already-prepass-
// populated cube/castle depths properly occlude the procedural surface.
bool VulkanEngine::tr_use_lowres() const {
    return rt_.terrain_raymarch_enabled &&
           rt_.terrain_raymarch_scale < 0.999f;
}

void VulkanEngine::init_terrain_raymarch_lowres() {
    // Compute the scaled extent. Floor to int, min 1.
    float s = std::clamp(rt_.terrain_raymarch_scale, 0.25f, 1.0f);
    // Water-only mode (mesh terrain active + water on) skips the whole
    // terrain march — it's cheap, so render it at FULL res. Half/quarter
    // res water was the cause of the gap + stair-step line artifacts
    // along terrain↔water silhouettes (the depth-aware upscale can't
    // reconstruct a clean edge between full-res rasterised terrain and
    // a low-res water buffer).
    if (!rt_.terrain_raymarch_enabled) s = 1.0f;
    uint32_t lw = std::max<uint32_t>(1u, static_cast<uint32_t>(
        std::round(static_cast<float>(render_extent_.width)  * s)));
    uint32_t lh = std::max<uint32_t>(1u, static_cast<uint32_t>(
        std::round(static_cast<float>(render_extent_.height) * s)));
    tr_lr_extent_ = { lw, lh };

    auto make_color = [&](VkFormat fmt, VkImage& img, VmaAllocation& alloc,
                           VkImageView& view, const char* tag) {
        VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = fmt,
            .extent = { lw, lh, 1 },
            .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &img, &alloc, nullptr),
                 tag);
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .image = img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
            .components = {},
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = 1,
            },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &view), tag);
    };
    make_color(scene_color_format_,
               tr_lr_color_image_,  tr_lr_color_alloc_,  tr_lr_color_view_,
               "tr_lr_color");
    make_color(motion_vec_format_,
               tr_lr_motion_image_, tr_lr_motion_alloc_, tr_lr_motion_view_,
               "tr_lr_motion");

    // Depth — the compose shader samples this as a normal sampler2D
    // (R32 read), so the image's view aspect is DEPTH but the
    // descriptor type is still combined_image_sampler.
    {
        VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = depth_format_,
            .extent = { lw, lh, 1 },
            .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci,
                                 &tr_lr_depth_image_, &tr_lr_depth_alloc_, nullptr),
                 "tr_lr_depth image");
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .image = tr_lr_depth_image_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = depth_format_,
            .components = {},
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = 1,
            },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &tr_lr_depth_view_),
                 "tr_lr_depth view");
    }

    // Linear sampler — the bilinear upscale is the whole reason we
    // can ship this at half-res without it looking like Minecraft.
    if (!tr_lr_sampler_) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &tr_lr_sampler_),
                 "tr_lr_sampler");
    }
    // Depth-dedicated NEAREST sampler — see vk_engine.h comment.
    if (!tr_lr_depth_sampler_) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &tr_lr_depth_sampler_),
                 "tr_lr_depth_sampler");
    }

    // Write descriptor bindings 9, 10, 11. These are stable across
    // resize for the lifetime of the views (we destroy + rewrite on
    // recreate).
    VkDescriptorImageInfo dii_c{};
    dii_c.sampler     = tr_lr_sampler_;
    dii_c.imageView   = tr_lr_color_view_;
    dii_c.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo dii_m{};
    dii_m.sampler     = tr_lr_sampler_;
    dii_m.imageView   = tr_lr_motion_view_;
    dii_m.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo dii_d{};
    dii_d.sampler     = tr_lr_depth_sampler_;   // NEAREST — see header comment
    dii_d.imageView   = tr_lr_depth_view_;
    dii_d.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = scene_desc_set_;
        writes[i].dstBinding = static_cast<uint32_t>(9 + i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    writes[0].pImageInfo = &dii_c;
    writes[1].pImageInfo = &dii_m;
    writes[2].pImageInfo = &dii_d;
    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    log::infof("terrain raymarch low-res targets: %ux%u (%.0f%%)",
               lw, lh, s * 100.0f);
}

void VulkanEngine::init_vrs_attachment() {
    if (!vrs_supported_) return;
    if (vrs_image_ != VK_NULL_HANDLE) return;
    // Attachment is sized so each texel covers a (tile.w × tile.h) block
    // of the LR raymarch render extent. NVIDIA + AMD currently report
    // 8×8 or 16×16. Round up so the attachment covers every pixel even
    // if (extent % tile) != 0; the trailing partial-tile texels just
    // shade slightly extra area, which is harmless.
    const uint32_t base_w = std::max<uint32_t>(1, tr_lr_extent_.width);
    const uint32_t base_h = std::max<uint32_t>(1, tr_lr_extent_.height);
    const uint32_t tw = std::max<uint32_t>(1, vrs_texel_size_.width);
    const uint32_t th = std::max<uint32_t>(1, vrs_texel_size_.height);
    vrs_extent_.width  = (base_w + tw - 1) / tw;
    vrs_extent_.height = (base_h + th - 1) / th;

    VkImageCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8_UINT,
        .extent = { vrs_extent_.width, vrs_extent_.height, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &vrs_image_, &vrs_alloc_,
                            nullptr), "vrs image");

    VkImageViewCreateInfo vci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .image = vrs_image_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8_UINT,
        .components = {},
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &vrs_view_), "vrs view");

    // Build the per-texel rate pattern on CPU, then upload via a staging
    // buffer + one-time submit. Pattern is a center→edge vignette:
    //   - inside 60% radius: rate 0  (1×1, full)
    //   - 60–85%:             rate 5 (2×2)
    //   - outside 85%:        rate 10 (4×4)
    // Encoding: `(width_log2 << 2) | height_log2`. Center-priority makes
    // sense because the player's attention is mid-screen and the FBM
    // raymarch detail beyond ~600 m is already LOD'd to a few octaves
    // (so 2×2 / 4×4 shading is invisible).
    const size_t n = static_cast<size_t>(vrs_extent_.width) *
                     static_cast<size_t>(vrs_extent_.height);
    std::vector<uint8_t> rates(n, 0);
    const float cx = (vrs_extent_.width  - 1) * 0.5f;
    const float cy = (vrs_extent_.height - 1) * 0.5f;
    const float r_max = std::sqrt(cx * cx + cy * cy);
    for (uint32_t y = 0; y < vrs_extent_.height; ++y) {
        for (uint32_t x = 0; x < vrs_extent_.width; ++x) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float r = std::sqrt(dx * dx + dy * dy) / std::max(r_max, 1.0f);
            uint8_t rate;
            if      (r < 0.60f) rate = 0;            // 1×1
            else if (r < 0.85f) rate = (1u << 2) | 1u; // 2×2 = 5
            else                rate = (2u << 2) | 2u; // 4×4 = 10
            rates[static_cast<size_t>(y) * vrs_extent_.width + x] = rate;
        }
    }

    // Stage + copy + final transition to FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL.
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = nullptr;
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = n,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo bac{};
        bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo bai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &bac, &staging, &staging_alloc,
                                 &bai), "vrs staging");
        std::memcpy(bai.pMappedData, rates.data(), n);
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, vrs_image_,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { vrs_extent_.width, vrs_extent_.height, 1 };
        vkCmdCopyBufferToImage(cb, staging, vrs_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        vkinit::transition_image(cb, vrs_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR);
    });
    vmaDestroyBuffer(allocator_, staging, staging_alloc);
    log::infof("VRS attachment %ux%u uploaded (tile %ux%u)",
               vrs_extent_.width, vrs_extent_.height,
               vrs_texel_size_.width, vrs_texel_size_.height);
}

void VulkanEngine::destroy_vrs_attachment() {
    if (vrs_view_) {
        vkDestroyImageView(device_, vrs_view_, nullptr);
        vrs_view_ = VK_NULL_HANDLE;
    }
    if (vrs_image_) {
        vmaDestroyImage(allocator_, vrs_image_, vrs_alloc_);
        vrs_image_ = VK_NULL_HANDLE;
        vrs_alloc_ = nullptr;
    }
    vrs_extent_ = {};
}

void VulkanEngine::recreate_vrs_attachment() {
    if (!vrs_supported_) return;
    vkDeviceWaitIdle(device_);
    destroy_vrs_attachment();
    init_vrs_attachment();
}

void VulkanEngine::destroy_terrain_raymarch_lowres() {
    if (tr_lr_color_view_) {
        vkDestroyImageView(device_, tr_lr_color_view_, nullptr);
        tr_lr_color_view_ = VK_NULL_HANDLE;
    }
    if (tr_lr_color_image_) {
        vmaDestroyImage(allocator_, tr_lr_color_image_, tr_lr_color_alloc_);
        tr_lr_color_image_ = VK_NULL_HANDLE;
        tr_lr_color_alloc_ = nullptr;
    }
    if (tr_lr_motion_view_) {
        vkDestroyImageView(device_, tr_lr_motion_view_, nullptr);
        tr_lr_motion_view_ = VK_NULL_HANDLE;
    }
    if (tr_lr_motion_image_) {
        vmaDestroyImage(allocator_, tr_lr_motion_image_, tr_lr_motion_alloc_);
        tr_lr_motion_image_ = VK_NULL_HANDLE;
        tr_lr_motion_alloc_ = nullptr;
    }
    if (tr_lr_depth_view_) {
        vkDestroyImageView(device_, tr_lr_depth_view_, nullptr);
        tr_lr_depth_view_ = VK_NULL_HANDLE;
    }
    if (tr_lr_depth_image_) {
        vmaDestroyImage(allocator_, tr_lr_depth_image_, tr_lr_depth_alloc_);
        tr_lr_depth_image_ = VK_NULL_HANDLE;
        tr_lr_depth_alloc_ = nullptr;
    }
    if (tr_lr_depth_sampler_) {
        vkDestroySampler(device_, tr_lr_depth_sampler_, nullptr);
        tr_lr_depth_sampler_ = VK_NULL_HANDLE;
    }
    if (tr_lr_sampler_) {
        vkDestroySampler(device_, tr_lr_sampler_, nullptr);
        tr_lr_sampler_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::recreate_terrain_raymarch_lowres() {
    vkDeviceWaitIdle(device_);
    destroy_terrain_raymarch_lowres();
    init_terrain_raymarch_lowres();
}

// Half-rate sun-shadow occlusion buffer: R8_UNORM at render_extent_/2.
// Rendered by a fullscreen PCSS pass (Phase 2), bilateral-upsampled in
// cube.frag (Phase 3). Allocated unconditionally (cheap, ~0.5 MB) so
// the runtime toggle needs no resize; sampled via tr_lr_sampler_.
void VulkanEngine::init_shadow_lr() {
    uint32_t sw = std::max<uint32_t>(1u, render_extent_.width  / 2u);
    uint32_t sh = std::max<uint32_t>(1u, render_extent_.height / 2u);
    shadow_lr_extent_ = { sw, sh };
    VkImageCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8_UNORM,
        .extent = { sw, sh, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                            &shadow_lr_image_, &shadow_lr_alloc_, nullptr),
             "shadow_lr image");
    VkImageViewCreateInfo vci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .image = shadow_lr_image_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8_UNORM,
        .components = {},
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &shadow_lr_view_),
             "shadow_lr view");
    log::infof("half-rate shadow buffer: %ux%u R8", sw, sh);
}

void VulkanEngine::destroy_shadow_lr() {
    if (shadow_lr_view_) {
        vkDestroyImageView(device_, shadow_lr_view_, nullptr);
        shadow_lr_view_ = VK_NULL_HANDLE;
    }
    if (shadow_lr_image_) {
        vmaDestroyImage(allocator_, shadow_lr_image_, shadow_lr_alloc_);
        shadow_lr_image_ = VK_NULL_HANDLE;
        shadow_lr_alloc_ = nullptr;
    }
}

// Rewrite scene_desc binding 18 to point at the (possibly newly created)
// shadow_lr_view_. Called after recreate_swapchain destroys+rebuilds the
// half-res image.
static void rewrite_binding18_shadow_lr(VkDevice device,
                                        VkDescriptorSet set,
                                        VkImageView view,
                                        VkSampler sampler) {
    if (!set || !view || !sampler) return;
    VkDescriptorImageInfo sh_bi{ sampler, view,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set; w.dstBinding = 18; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &sh_bi;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

void VulkanEngine::recreate_shadow_lr() {
    vkDeviceWaitIdle(device_);
    destroy_shadow_lr();
    init_shadow_lr();
    rewrite_binding18_shadow_lr(device_, scene_desc_set_,
                                shadow_lr_view_, linear_sampler_);
}

// SVGF GI denoiser — Session 1 plumbing (docs/svgf_plan.md). Allocates
// the R16G16B16A16F storage image cube.frag writes raw per-pixel GI
// irradiance into via imageStore. STORAGE + SAMPLED usage so later
// sessions can also read it from the denoiser passes. Layout starts
// UNDEFINED; vk_engine.cpp transitions to GENERAL before the main
// color pass and leaves it there (storage images can both write and
// sample-read in GENERAL).
void VulkanEngine::init_svgf_targets() {
    uint32_t w = std::max<uint32_t>(1u, render_extent_.width);
    uint32_t h = std::max<uint32_t>(1u, render_extent_.height);
    VkImageCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent = { w, h, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,   // needed by clear-to-zero below
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                            &svgf_gi_image_, &svgf_gi_alloc_, nullptr),
             "svgf gi image");
    VkImageViewCreateInfo vci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .image = svgf_gi_image_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .components = {},
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &svgf_gi_view_),
             "svgf gi view");

    // History ping-pong pair (Session 2). Same format/size. Same one-
    // time UNDEFINED→GENERAL transition. cube.frag indexes by
    // (frame & 1) — even frames read slot 0 / write slot 1, odd frames
    // swap. Race-free under kFrameOverlap=2 because frame F's write
    // target was last touched by F-2 (already retired by F-1's fence).
    for (int s = 0; s < 2; ++s) {
        vk_check(vmaCreateImage(allocator_, &ici, &aci,
                                &svgf_history_image_[s],
                                &svgf_history_alloc_[s], nullptr),
                 "svgf history image");
        VkImageViewCreateInfo hvci = vci;
        hvci.image = svgf_history_image_[s];
        vk_check(vkCreateImageView(device_, &hvci, nullptr,
                                   &svgf_history_view_[s]),
                 "svgf history view");
    }

    // Variance-moments ping-pong pair (4a-deep). R32G32 SFLOAT keeps
    // luminance + luminance^2 EMA estimates without precision loss on
    // HDR pixels (R16F would clip the squared term well below 1, killing
    // the variance estimate). Same render_extent + GENERAL layout +
    // zero-clear behaviour as the irradiance history slots so the very
    // first frame reads a sane prevMoments = (0, 0) and the EMA fully
    // takes the fresh sample (alpha=1 path inside svgf_moments.frag).
    // Also gets STORAGE+SAMPLED so svgf_atrous.frag can sample with
    // texelFetch later if we ever switch to a sampler-based bind.
    VkImageCreateInfo mici = ici;
    mici.format = VK_FORMAT_R32G32_SFLOAT;
    VkImageViewCreateInfo mvci = vci;
    mvci.format = VK_FORMAT_R32G32_SFLOAT;
    for (int s = 0; s < 2; ++s) {
        vk_check(vmaCreateImage(allocator_, &mici, &aci,
                                &svgf_moments_image_[s],
                                &svgf_moments_alloc_[s], nullptr),
                 "svgf moments image");
        mvci.image = svgf_moments_image_[s];
        vk_check(vkCreateImageView(device_, &mvci, nullptr,
                                   &svgf_moments_view_[s]),
                 "svgf moments view");
    }

    // A-trous ping-pong pair (4a-deep). Stays as a colour-attachment +
    // sampled-image pair (no STORAGE) because svgf_atrous.frag is a
    // standard fragment-output pipeline; descriptor uses sampler2D.
    // Format matches svgf_gi_image_/scene_color so the 3-pass chain
    // (scene_color -> atrous[0] -> atrous[1] -> scene_color) doesn't
    // bottleneck on a precision drop.
    VkImageCreateInfo atici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = scene_color_format_,
        .extent = { w, h, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageViewCreateInfo atvci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .image = VK_NULL_HANDLE,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = scene_color_format_,
        .components = {},
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    for (int s = 0; s < 2; ++s) {
        vk_check(vmaCreateImage(allocator_, &atici, &aci,
                                &svgf_atrous_image_[s],
                                &svgf_atrous_alloc_[s], nullptr),
                 "svgf atrous image");
        atvci.image = svgf_atrous_image_[s];
        vk_check(vkCreateImageView(device_, &atvci, nullptr,
                                   &svgf_atrous_view_[s]),
                 "svgf atrous view");
    }

    // One-time transitions + ZERO CLEAR. R16G16B16A16_SFLOAT and
    // R32G32_SFLOAT allocations can hold NaN/Inf garbage; cube.frag's
    // first-frame imageLoad on history (or svgf_moments.frag on the
    // prev moments slot) would pull those into mix(rgb, NaN, NaN) ->
    // black/firefly pixels everywhere SVGF is enabled. Clear all of
    // gi + history(2) + moments(2) to (0,0,0,0) so M-count=0 ->
    // temporal blend takes the fresh sample only.
    //
    // svgf_atrous_image_ entries do not need clearing; they're written
    // before being read in the same frame (always fully overwritten by
    // the previous a-trous pass). They do need to land in
    // SHADER_READ_ONLY so the first atrous pass sampling its OTHER
    // ping-pong slot doesn't read UNDEFINED.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, svgf_gi_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL);
        for (int s = 0; s < 2; ++s) {
            vkinit::transition_image(cb, svgf_history_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_GENERAL);
            vkinit::transition_image(cb, svgf_moments_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_GENERAL);
            vkinit::transition_image(cb, svgf_atrous_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        VkClearColorValue zero{};
        VkImageSubresourceRange sub{
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
        };
        vkCmdClearColorImage(cb, svgf_gi_image_,
                             VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &sub);
        for (int s = 0; s < 2; ++s) {
            vkCmdClearColorImage(cb, svgf_history_image_[s],
                                 VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &sub);
            vkCmdClearColorImage(cb, svgf_moments_image_[s],
                                 VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &sub);
        }
    });

    // Initial descriptor writes at bindings 19 / 20 / 21 / 22 / 23.
    if (scene_desc_set_) {
        VkDescriptorImageInfo infos[5] = {
            { VK_NULL_HANDLE, svgf_gi_view_,         VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, svgf_history_view_[0], VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, svgf_history_view_[1], VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, svgf_moments_view_[0], VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, svgf_moments_view_[1], VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet wr[5]{};
        for (int i = 0; i < 5; ++i) {
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = scene_desc_set_;
            wr[i].dstBinding = static_cast<uint32_t>(19 + i);
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            wr[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 5, wr, 0, nullptr);
    }
    log::infof("[svgf] gi + history(x2) + moments(x2) + atrous(x2) at %ux%u",
               w, h);
}

void VulkanEngine::destroy_svgf_targets() {
    if (svgf_gi_view_) {
        vkDestroyImageView(device_, svgf_gi_view_, nullptr);
        svgf_gi_view_ = VK_NULL_HANDLE;
    }
    if (svgf_gi_image_) {
        vmaDestroyImage(allocator_, svgf_gi_image_, svgf_gi_alloc_);
        svgf_gi_image_ = VK_NULL_HANDLE;
        svgf_gi_alloc_ = nullptr;
    }
    for (int s = 0; s < 2; ++s) {
        if (svgf_history_view_[s]) {
            vkDestroyImageView(device_, svgf_history_view_[s], nullptr);
            svgf_history_view_[s] = VK_NULL_HANDLE;
        }
        if (svgf_history_image_[s]) {
            vmaDestroyImage(allocator_, svgf_history_image_[s],
                            svgf_history_alloc_[s]);
            svgf_history_image_[s] = VK_NULL_HANDLE;
            svgf_history_alloc_[s] = nullptr;
        }
        if (svgf_moments_view_[s]) {
            vkDestroyImageView(device_, svgf_moments_view_[s], nullptr);
            svgf_moments_view_[s] = VK_NULL_HANDLE;
        }
        if (svgf_moments_image_[s]) {
            vmaDestroyImage(allocator_, svgf_moments_image_[s],
                            svgf_moments_alloc_[s]);
            svgf_moments_image_[s] = VK_NULL_HANDLE;
            svgf_moments_alloc_[s] = nullptr;
        }
        if (svgf_atrous_view_[s]) {
            vkDestroyImageView(device_, svgf_atrous_view_[s], nullptr);
            svgf_atrous_view_[s] = VK_NULL_HANDLE;
        }
        if (svgf_atrous_image_[s]) {
            vmaDestroyImage(allocator_, svgf_atrous_image_[s],
                            svgf_atrous_alloc_[s]);
            svgf_atrous_image_[s] = VK_NULL_HANDLE;
            svgf_atrous_alloc_[s] = nullptr;
        }
    }
}

// Rewrite scene_desc bindings 19..23 to point at the (possibly newly
// allocated) SVGF storage views. Used both by initial wire-up (via
// init_svgf_targets's own write block) and the post-swapchain-recreate
// path here. Belt-and-braces: init_svgf_targets already wrote these
// once at allocation, this confirms after a destroy+realloc.
static void rewrite_svgf_storage_bindings(VkDevice device,
                                          VkDescriptorSet set,
                                          VkImageView gi_view,
                                          const VkImageView (&hist)[2],
                                          const VkImageView (&moments)[2]) {
    if (!set || !gi_view) return;
    VkDescriptorImageInfo infos[5] = {
        { VK_NULL_HANDLE, gi_view,     VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, hist[0],     VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, hist[1],     VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, moments[0],  VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, moments[1],  VK_IMAGE_LAYOUT_GENERAL },
    };
    VkWriteDescriptorSet w[5]{};
    for (int i = 0; i < 5; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].dstBinding = static_cast<uint32_t>(19 + i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, 5, w, 0, nullptr);
}

void VulkanEngine::recreate_svgf_targets() {
    vkDeviceWaitIdle(device_);
    destroy_svgf_targets();
    init_svgf_targets();
    rewrite_svgf_storage_bindings(device_, scene_desc_set_,
                                  svgf_gi_view_,
                                  svgf_history_view_,
                                  svgf_moments_view_);
    // The a-trous fragment pipeline samples scene_color / depth /
    // moments / atrous_view via its own desc sets; rewrite those too
    // so the chained 3-pass filter follows the new image handles.
    recreate_svgf_pass_targets();
}

void VulkanEngine::init_terrain_raymarch_compose_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    tr_compose_frag_module_ = vkpipe::load_shader_module(
        device_, sd + "/terrain_raymarch_compose.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = terrain_raymarch_vert_module_;   // reuse the fullscreen-tri vert
    cfg.frag = tr_compose_frag_module_;
    cfg.layout = pipeline_layout_;
    cfg.color_formats = { scene_color_format_, motion_vec_format_ };
    cfg.depth_format = depth_format_;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    // LESS_OR_EQUAL — we want raymarch fragments that beat (or tie) the
    // cube prepass depth. Cube fragments drawn later will pass the
    // same comparison only where they're actually closer than the
    // raymarch.
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    tr_compose_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::info("terrain raymarch compose pipeline built");
}

void VulkanEngine::destroy_terrain_raymarch_compose_pipeline() {
    if (tr_compose_pipeline_) {
        vkDestroyPipeline(device_, tr_compose_pipeline_, nullptr);
        tr_compose_pipeline_ = VK_NULL_HANDLE;
    }
    if (tr_compose_frag_module_) {
        vkDestroyShaderModule(device_, tr_compose_frag_module_, nullptr);
        tr_compose_frag_module_ = VK_NULL_HANDLE;
    }
}

// init_sun_shadow_pipeline / destroy_sun_shadow_pipeline moved to
// vk_engine/sun_shadow.cpp.

void VulkanEngine::init_grass_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    grass_vert_module_ = vkpipe::load_shader_module(device_, sd + "/grass.vert.spv");
    grass_frag_module_ = vkpipe::load_shader_module(device_, sd + "/grass.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = grass_vert_module_;
    cfg.frag = grass_frag_module_;
    cfg.layout = pipeline_layout_;     // shares cube's layout (push consts)
    cfg.color_formats = { scene_color_format_, motion_vec_format_ };
    cfg.depth_format = depth_format_;

    // Grass uses CULL_NONE — blades are double-sided, the back of a
    // billboard would otherwise vanish when its random rotation faces
    // away from the camera.
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Vertex bindings:
    //   0 = blade mesh per-vertex (Vertex layout, same as cube)
    //   1 = per-instance GrassBlade (3 vec4 entries: pos+pad,
    //       rot+height+pad, tint+pad)
    VkVertexInputBindingDescription b0{};
    b0.binding = 0;
    b0.stride = sizeof(Vertex);
    b0.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputBindingDescription b1{};
    b1.binding = 1;
    b1.stride = static_cast<uint32_t>(sizeof(GrassBlade));
    b1.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    cfg.vbindings = { b0, b1 };

    auto attr = [](uint32_t loc, uint32_t binding, VkFormat fmt, uint32_t off) {
        VkVertexInputAttributeDescription a{};
        a.location = loc; a.binding = binding; a.format = fmt; a.offset = off;
        return a;
    };
    cfg.vattrs = {
        attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)),
        attr(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
        attr(2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)),
        // GrassBlade layout (matches header):
        //   vec3 pos + float pad     -> vec4 at offset 0
        //   float rot + float h + 2pad -> vec4 at offset 16
        //   vec3 tint + float pad     -> vec4 at offset 32
        attr(3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0),
        attr(4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16),
        attr(5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32),
    };

    grass_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::info("grass pipeline built");
}

void VulkanEngine::init_grass_raymarch_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    grass_rm_frag_module_ = vkpipe::load_shader_module(device_,
                                sd + "/grass_raymarch.frag.spv");
    vkpipe::GraphicsPipelineConfig cfg{};
    // Reuse taa.vert (fullscreen triangle from gl_VertexIndex). The
    // taa shader module was created in init_taa() and stays alive for
    // the engine's lifetime, so it's safe to reference here.
    cfg.vert = taa_vert_module_;
    cfg.frag = grass_rm_frag_module_;
    cfg.layout = pipeline_layout_;            // shares the cube push-constant layout
    cfg.color_formats = { scene_color_format_, motion_vec_format_ };
    cfg.depth_format  = depth_format_;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    // LESS_OR_EQUAL so the fullscreen-tri's gl_FragCoord.z (=1.0) doesn't
    // immediately fail vs whatever the depth pre-pass wrote — the actual
    // depth test gate is gl_FragDepth (the marched hit's projected z),
    // which the shader writes per-pixel.
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    // Real alpha blending on attachment 0 (color) only. The shader
    // outputs a true edge alpha derived from the SDF distance at the
    // converged hit, so blade silhouettes blend smoothly into the
    // underlying terrain instead of dither-discarding. Attachment 1
    // (motion vector) stays opaque — TAA needs a clean motion read
    // and a blended motion vector would be meaningless.
    cfg.alpha_blend_color0_only = true;
    grass_rm_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::info("grass raymarch pipeline built");
}

void VulkanEngine::destroy_grass_raymarch_pipeline() {
    if (grass_rm_pipeline_) {
        vkDestroyPipeline(device_, grass_rm_pipeline_, nullptr);
        grass_rm_pipeline_ = VK_NULL_HANDLE;
    }
    if (grass_rm_frag_module_) {
        vkDestroyShaderModule(device_, grass_rm_frag_module_, nullptr);
        grass_rm_frag_module_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::destroy_grass_pipeline() {
    if (grass_pipeline_) vkDestroyPipeline(device_, grass_pipeline_, nullptr);
    if (grass_vert_module_) vkDestroyShaderModule(device_, grass_vert_module_, nullptr);
    if (grass_frag_module_) vkDestroyShaderModule(device_, grass_frag_module_, nullptr);
    grass_pipeline_ = VK_NULL_HANDLE;
    grass_vert_module_ = grass_frag_module_ = VK_NULL_HANDLE;
}

void VulkanEngine::destroy_pipeline() {
    if (pipeline_)         vkDestroyPipeline(device_, pipeline_, nullptr);
    if (depth_pipeline_)   vkDestroyPipeline(device_, depth_pipeline_, nullptr);
    if (shadow_lr_pipeline_) vkDestroyPipeline(device_, shadow_lr_pipeline_, nullptr);
    if (pipeline_layout_)  vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (vert_module_)      vkDestroyShaderModule(device_, vert_module_, nullptr);
    if (frag_module_)      vkDestroyShaderModule(device_, frag_module_, nullptr);
    if (depth_frag_module_) vkDestroyShaderModule(device_, depth_frag_module_, nullptr);
    if (depth_vert_module_) vkDestroyShaderModule(device_, depth_vert_module_, nullptr);
    if (shadow_lr_frag_module_) vkDestroyShaderModule(device_, shadow_lr_frag_module_, nullptr);
    pipeline_ = depth_pipeline_ = shadow_lr_pipeline_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    vert_module_ = frag_module_ = depth_frag_module_ = depth_vert_module_ = VK_NULL_HANDLE;
    shadow_lr_frag_module_ = VK_NULL_HANDLE;
}

// Half-rate shadow producer pipeline (roadmap item #4, Phase 2). Re-uses
// pipeline_layout_ + vert_module_ (cube.vert) so push-constants and
// descriptor sets stay identical to the main color pass. Only delta:
// single R8_UNORM colour attachment, no depth attachment (no depth test
// — accepts a small amount of overdraw to avoid plumbing a half-res
// depth buffer for Phase 2), and the trivial shadow_lr.frag.
void VulkanEngine::init_shadow_lr_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    shadow_lr_frag_module_ = vkpipe::load_shader_module(
        device_, sd + "/shadow_lr.frag.spv");

    // Replicate the same vertex-attribute layout the main cube pipeline uses.
    VkVertexInputBindingDescription b0{};
    b0.binding = 0; b0.stride = sizeof(Vertex);
    b0.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription a0{};
    a0.location = 0; a0.binding = 0;
    a0.format = VK_FORMAT_R32G32B32_SFLOAT;
    a0.offset = offsetof(Vertex, position);
    VkVertexInputAttributeDescription a1{};
    a1.location = 1; a1.binding = 0;
    a1.format = VK_FORMAT_R32G32B32_SFLOAT;
    a1.offset = offsetof(Vertex, normal);
    VkVertexInputAttributeDescription a2{};
    a2.location = 2; a2.binding = 0;
    a2.format = VK_FORMAT_R32G32_SFLOAT;
    a2.offset = offsetof(Vertex, uv);

    vkpipe::GraphicsPipelineConfig scfg{};
    scfg.layout = pipeline_layout_;
    scfg.vert = vert_module_;
    scfg.frag = shadow_lr_frag_module_;
    scfg.vbindings = { b0 };
    scfg.vattrs = { a0, a1, a2 };
    scfg.color_formats = { VK_FORMAT_R8_UNORM };
    scfg.color_attachment_count = 1;
    scfg.depth_format = VK_FORMAT_UNDEFINED;
    scfg.depth_test = false;
    scfg.depth_write = false;
    scfg.cull = VK_CULL_MODE_BACK_BIT;
    scfg.alpha_blend_color0_only = false;
    shadow_lr_pipeline_ = vkpipe::build_graphics_pipeline(device_, scfg);

    // Wire the produced image as a sampled texture at scene_desc binding 18
    // for cube.frag's bilateral-upsample consumer (Phase 3). linear_sampler_
    // (CLAMP_TO_EDGE, LINEAR mag/min) gives the bilinear interpolation that
    // is the "bilateral" in this minimum-viable upsample — Phase 3.5 may
    // add depth/normal edge-stopping if silhouette bleed is visible.
    if (scene_desc_set_ && shadow_lr_view_ && linear_sampler_) {
        VkDescriptorImageInfo sh_bi{ linear_sampler_, shadow_lr_view_,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 18;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &sh_bi;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    log::info("shadow_lr pipeline built (half-rate sun shadow producer)");
}

void VulkanEngine::init_pipeline_cache() {
    // Load any existing cache blob from disk so subsequent runs reuse the
    // GPU's pre-compiled pipelines. First run on a clean machine is a few
    // hundred ms slower; every run after is near-instant.
    constexpr const char* kCachePath = "qlike_pipeline.cache";
    std::vector<char> initial;
    if (std::ifstream f(kCachePath, std::ios::binary | std::ios::ate); f) {
        std::streamsize sz = f.tellg();
        if (sz > 0) {
            initial.resize(static_cast<size_t>(sz));
            f.seekg(0);
            f.read(initial.data(), sz);
            log::infof("pipeline cache loaded: %s (%lld bytes)", kCachePath,
                       static_cast<long long>(sz));
        }
    }
    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = initial.size();
    ci.pInitialData = initial.empty() ? nullptr : initial.data();
    if (vkCreatePipelineCache(device_, &ci, nullptr, &pipeline_cache_) == VK_SUCCESS) {
        vkpipe::set_pipeline_cache(pipeline_cache_);
    } else {
        log::warnf("pipeline cache create failed; pipelines won't be cached");
        pipeline_cache_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::destroy_pipeline_cache() {
    if (!pipeline_cache_) return;
    constexpr const char* kCachePath = "qlike_pipeline.cache";
    size_t sz = 0;
    vkGetPipelineCacheData(device_, pipeline_cache_, &sz, nullptr);
    if (sz > 0) {
        std::vector<char> blob(sz);
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, blob.data())
            == VK_SUCCESS) {
            std::ofstream f(kCachePath, std::ios::binary);
            if (f) {
                f.write(blob.data(), static_cast<std::streamsize>(sz));
                log::infof("pipeline cache saved: %s (%zu bytes)", kCachePath, sz);
            }
        }
    }
    vkpipe::set_pipeline_cache(VK_NULL_HANDLE);
    vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
    pipeline_cache_ = VK_NULL_HANDLE;
}

} // namespace qlike
