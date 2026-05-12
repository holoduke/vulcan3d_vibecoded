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
                  .request_validation_layers(kUseValidationLayers)
                  .require_api_version(1, 3, 0)
                  .set_debug_callback(&debug_callback)
                  .set_debug_messenger_severity(
                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                  .set_debug_messenger_type(
                      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
    if (kUseValidationLayers) {
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
    log::infof("vk instance created (validation=%s)", kUseValidationLayers ? "on" : "off");

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

    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    auto phys_ret = selector.set_minimum_version(1, 3)
                        .set_required_features_13(f13)
                        .set_required_features_12(f12)
                        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
                        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
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

    vkb::DeviceBuilder dev_builder{ vkb_phys };
    if (vrs_supported_) dev_builder.add_pNext(&vrs_feat);
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
    // Bloom mip 0 is sized to render_extent_/2; if we don't recreate it
    // alongside the swapchain, compose samples a stale-sized texture and
    // bloom appears offset against everything else. The compose desc-set's
    // image bindings (history at 0, depth at 1, bloom at 3) all reference
    // image views that were just destroyed, so rebind them.
    if (bloom_image_) recreate_bloom_targets();
    // LR raymarch targets are sized at render_extent_ × scale; rebuild
    // them too so the upscale compose continues to read valid storage.
    if (tr_lr_color_image_) {
        destroy_terrain_raymarch_lowres();
        init_terrain_raymarch_lowres();
    }
    // VRS attachment is sized in LR-tile units — depends on tr_lr_extent_.
    if (vrs_supported_) recreate_vrs_attachment();
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
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
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

    pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

    // Depth-only pre-pass pipeline. draw() runs render_world_depth_pass()
    // before the color pass to populate depth_image_ so the color pass's
    // depth_compare=LESS_OR_EQUAL early-rejects occluded fragments before
    // cube.frag's heavy inline-RT body executes. Particles, projectiles and
    // the viewmodel skip the prepass (sparse / screen-space).
    {
        std::string sd2 = QLIKE_SHADER_DIR;
        depth_frag_module_ = vkpipe::load_shader_module(device_, sd2 + "/cube_depth.frag.spv");
    }
    vkpipe::GraphicsPipelineConfig dcfg = cfg;
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
    log::info("terrain pipelines built (CD-LOD morph)");
}

void VulkanEngine::destroy_terrain_pipelines() {
    if (terrain_pipeline_)       vkDestroyPipeline(device_, terrain_pipeline_, nullptr);
    if (terrain_depth_pipeline_) vkDestroyPipeline(device_, terrain_depth_pipeline_, nullptr);
    if (terrain_vert_module_)    vkDestroyShaderModule(device_, terrain_vert_module_, nullptr);
    terrain_pipeline_ = terrain_depth_pipeline_ = VK_NULL_HANDLE;
    terrain_vert_module_ = VK_NULL_HANDLE;
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
    if (pipeline_layout_)  vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (vert_module_)      vkDestroyShaderModule(device_, vert_module_, nullptr);
    if (frag_module_)      vkDestroyShaderModule(device_, frag_module_, nullptr);
    if (depth_frag_module_) vkDestroyShaderModule(device_, depth_frag_module_, nullptr);
    pipeline_ = depth_pipeline_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    vert_module_ = frag_module_ = depth_frag_module_ = VK_NULL_HANDLE;
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
