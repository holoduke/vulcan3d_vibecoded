// Vulkan device + swapchain + depth + per-frame command/sync state +
// readback buffer + raster pipelines + persistent pipeline cache. Everything
// the engine spins up once at init() before the first draw, plus the
// recreate_swapchain() path that handles window resizes.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"
#include "engine/window.h"

#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

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

    vkb::DeviceBuilder dev_builder{ vkb_phys };
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
