// TAA (temporal anti-aliasing) pass — history pingpong + motion-vector
// reproject. Lives in its own TU so post.cpp doesn't carry the TAA
// init/teardown plus the compose pass plus screenshot read-back.
//
// Still implemented as VulkanEngine member methods (init_taa,
// recreate_taa_targets, destroy_taa). State fields stay in vk_engine.h
// for now; a future pass can wrap them in a TaaPass RAII class once
// linear_sampler_/history_view_ ownership is disentangled from the
// compose pass that also samples them.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

#include <string>

namespace qlike {

void VulkanEngine::init_taa() {
    // Linear sampler for history+depth lookups. Owned by TAA but the
    // compose pass also samples through it (created first; compose's
    // init_compose() relies on it being live).
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &linear_sampler_),
                 "linear sampler");
    }

    // UBO buffer. The motion-vec-driven reproject path (svgf-tao-rewrite)
    // dropped the inverse/prev VP matrices that the depth-reconstruct path
    // used to need; only viewport + params remain.
    struct TaaUBO { glm::vec4 vp; glm::vec4 params; };
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(TaaUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &taa_ubo_buffer_, &taa_ubo_alloc_, nullptr),
                 "taa ubo");
    }

    // Descriptor pool: 2 sets (one per history slot we sample as the "previous"),
    // each with 4 sampled images (current_color + history_color + current_depth
    // + motion_vec) + 1 UBO.
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = kHistorySlots, .poolSizeCount = 2, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &taa_desc_pool_),
                 "taa desc pool");
    }

    // Set layout: 0 = current_color, 1 = history_color, 2 = current_depth,
    //             3 = UBO, 4 = motion vector image.
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .bindingCount = 5, .pBindings = bindings,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &taa_desc_set_layout_),
             "taa set layout");

    VkDescriptorSetLayout layouts[kHistorySlots] = {
        taa_desc_set_layout_, taa_desc_set_layout_
    };
    VkDescriptorSetAllocateInfo dai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = taa_desc_pool_,
        .descriptorSetCount = kHistorySlots,
        .pSetLayouts = layouts,
    };
    vk_check(vkAllocateDescriptorSets(device_, &dai, taa_desc_sets_),
             "taa set alloc");

    // Pipeline layout (just the set, no push constants).
    VkPipelineLayoutCreateInfo plci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &taa_desc_set_layout_,
        .pushConstantRangeCount = 0, .pPushConstantRanges = nullptr,
    };
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &taa_pipeline_layout_),
             "taa pipeline layout");

    std::string sd = QLIKE_SHADER_DIR;
    taa_vert_module_ = vkpipe::load_shader_module(device_, sd + "/taa.vert.spv");
    taa_frag_module_ = vkpipe::load_shader_module(device_, sd + "/taa.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = taa_vert_module_;
    cfg.frag = taa_frag_module_;
    cfg.layout = taa_pipeline_layout_;
    cfg.color_format = scene_color_format_;
    cfg.depth_format = VK_FORMAT_UNDEFINED;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = false;
    cfg.depth_write = false;
    taa_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

    recreate_taa_targets();
    log::info("TAA initialized");
}

void VulkanEngine::recreate_taa_targets() {
    // Drop existing targets.
    if (scene_color_view_)  vkDestroyImageView(device_, scene_color_view_, nullptr);
    if (scene_color_image_) vmaDestroyImage(allocator_, scene_color_image_, scene_color_alloc_);
    scene_color_image_ = VK_NULL_HANDLE; scene_color_view_ = VK_NULL_HANDLE;
    scene_color_alloc_ = nullptr;
    if (motion_vec_view_)   vkDestroyImageView(device_, motion_vec_view_, nullptr);
    if (motion_vec_image_)  vmaDestroyImage(allocator_, motion_vec_image_, motion_vec_alloc_);
    motion_vec_image_ = VK_NULL_HANDLE; motion_vec_view_ = VK_NULL_HANDLE;
    motion_vec_alloc_ = nullptr;
    for (int s = 0; s < kHistorySlots; ++s) {
        if (history_view_[s])  vkDestroyImageView(device_, history_view_[s], nullptr);
        if (history_image_[s]) vmaDestroyImage(allocator_, history_image_[s], history_alloc_[s]);
        history_image_[s] = VK_NULL_HANDLE; history_view_[s] = VK_NULL_HANDLE;
        history_alloc_[s] = nullptr;
    }

    auto make_image = [&](VkFormat fmt, VkImageUsageFlags usage,
                          VkImage& img, VmaAllocation& alloc, VkImageView& view) {
        VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
            // Render-resolution targets: scene_color / motion_vec / history
            // all sized to render_extent_, not swapchain_extent_, so the
            // user's render_scale slider actually changes per-pixel cost.
            .extent = { render_extent_.width, render_extent_.height, 1 },
            .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &img, &alloc, nullptr),
                 "taa image");
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .image = img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
            .components = {},
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &view),
                 "taa view");
    };

    make_image(scene_color_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
               scene_color_image_, scene_color_alloc_, scene_color_view_);
    make_image(motion_vec_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
               motion_vec_image_, motion_vec_alloc_, motion_vec_view_);
    for (int s = 0; s < kHistorySlots; ++s) {
        make_image(scene_color_format_,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   history_image_[s], history_alloc_[s], history_view_[s]);
    }

    for (int set = 0; set < kHistorySlots; ++set) {
        VkDescriptorImageInfo i_cur{ linear_sampler_, scene_color_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_hist{ linear_sampler_,
                                       history_view_[1 - set],
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth{ linear_sampler_, depth_view_,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_motion{ linear_sampler_, motion_vec_view_,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorBufferInfo i_ubo{ taa_ubo_buffer_, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet w[5]{};
        for (int i = 0; i < 3; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = taa_desc_sets_[set];
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[0].pImageInfo = &i_cur;
        w[1].pImageInfo = &i_hist;
        w[2].pImageInfo = &i_depth;
        w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[3].dstSet = taa_desc_sets_[set];
        w[3].dstBinding = 3;
        w[3].descriptorCount = 1;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[3].pBufferInfo = &i_ubo;
        w[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[4].dstSet = taa_desc_sets_[set];
        w[4].dstBinding = 4;
        w[4].descriptorCount = 1;
        w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[4].pImageInfo = &i_motion;
        vkUpdateDescriptorSets(device_, 5, w, 0, nullptr);
    }

    prev_view_proj_valid_ = false;

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        for (int s = 0; s < kHistorySlots; ++s) {
            vkinit::transition_image(cb, history_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });
}

void VulkanEngine::destroy_taa() {
    if (taa_pipeline_)        vkDestroyPipeline(device_, taa_pipeline_, nullptr);
    if (taa_pipeline_layout_) vkDestroyPipelineLayout(device_, taa_pipeline_layout_, nullptr);
    if (taa_vert_module_)     vkDestroyShaderModule(device_, taa_vert_module_, nullptr);
    if (taa_frag_module_)     vkDestroyShaderModule(device_, taa_frag_module_, nullptr);
    if (taa_desc_set_layout_) vkDestroyDescriptorSetLayout(device_, taa_desc_set_layout_, nullptr);
    if (taa_desc_pool_)       vkDestroyDescriptorPool(device_, taa_desc_pool_, nullptr);
    if (taa_ubo_buffer_)      vmaDestroyBuffer(allocator_, taa_ubo_buffer_, taa_ubo_alloc_);
    if (linear_sampler_)      vkDestroySampler(device_, linear_sampler_, nullptr);

    for (int s = 0; s < kHistorySlots; ++s) {
        if (history_view_[s])  vkDestroyImageView(device_, history_view_[s], nullptr);
        if (history_image_[s]) vmaDestroyImage(allocator_, history_image_[s], history_alloc_[s]);
        history_image_[s] = VK_NULL_HANDLE;
        history_view_[s] = VK_NULL_HANDLE;
        history_alloc_[s] = nullptr;
    }
    if (scene_color_view_)  vkDestroyImageView(device_, scene_color_view_, nullptr);
    if (scene_color_image_) vmaDestroyImage(allocator_, scene_color_image_, scene_color_alloc_);
    if (motion_vec_view_)   vkDestroyImageView(device_, motion_vec_view_, nullptr);
    if (motion_vec_image_)  vmaDestroyImage(allocator_, motion_vec_image_, motion_vec_alloc_);

    taa_pipeline_ = VK_NULL_HANDLE;
    taa_pipeline_layout_ = VK_NULL_HANDLE;
    taa_vert_module_ = taa_frag_module_ = VK_NULL_HANDLE;
    taa_desc_set_layout_ = VK_NULL_HANDLE;
    taa_desc_pool_ = VK_NULL_HANDLE;
    taa_ubo_buffer_ = VK_NULL_HANDLE;
    taa_ubo_alloc_ = nullptr;
    linear_sampler_ = VK_NULL_HANDLE;
    scene_color_image_ = VK_NULL_HANDLE;
    scene_color_view_ = VK_NULL_HANDLE;
    scene_color_alloc_ = nullptr;
    motion_vec_image_ = VK_NULL_HANDLE;
    motion_vec_view_ = VK_NULL_HANDLE;
    motion_vec_alloc_ = nullptr;
}

} // namespace qlike
