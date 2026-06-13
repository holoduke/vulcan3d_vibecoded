// Vulkan device + swapchain + depth + per-frame command/sync state + GPU
// timestamp queries + readback buffer, plus the recreate_swapchain() path
// that handles window resizes. The pipeline and render-target creation that
// used to live here moved to pipelines_setup.cpp.

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
    // SSDM disp image sizes to render_extent_; rebuild on swapchain change.
    if (ssdm_disp_pipeline_) recreate_ssdm_targets();
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

} // namespace qlike
