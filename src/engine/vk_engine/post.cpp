// Post-process passes: TAA (history pingpong, motion-vector consumed) +
// bloom mip chain (Karis-prefiltered downsample → additive upsample) +
// compose (compose.frag tonemaps history, blends bloom, draws lens flare).
// Plus the screenshot read-back path (capture_screenshot / write_ppm) since
// it's tightly coupled to the post-pass output.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace qlike {

void VulkanEngine::init_taa() {
    // Linear sampler for history+depth lookups.
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

    // UBO buffer (matrices + params).
    struct TaaUBO { glm::mat4 inv_vp; glm::mat4 prev_vp; glm::vec4 vp; glm::vec4 params; };
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

    // Bloom mip chain comes BEFORE the compose-desc-set write below — the
    // compose set's binding 3 samples bloom_mip_views_[0], which only exists
    // after init_bloom() → recreate_bloom_targets() runs.
    init_bloom();

    // --- Compose pass: samples history (HDR) → tonemap → swapchain (LDR) ---
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kHistorySlots * 4 },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = kHistorySlots, .poolSizeCount = 1, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &compose_desc_pool_),
                 "compose desc pool");
    }
    {
        VkDescriptorSetLayoutBinding b[4]{};
        for (int i = 0; i < 4; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .bindingCount = 4, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                             &compose_desc_set_layout_),
                 "compose set layout");
    }
    {
        VkDescriptorSetLayout layouts[kHistorySlots] = {
            compose_desc_set_layout_, compose_desc_set_layout_
        };
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = compose_desc_pool_,
            .descriptorSetCount = kHistorySlots, .pSetLayouts = layouts,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai, compose_desc_sets_),
                 "compose set alloc");
        rewrite_compose_image_bindings();
    }

    VkPushConstantRange compose_pc{};
    compose_pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    compose_pc.offset = 0;
    // 9 vec4 (viewport, bloom, sun_dir, sun_color, sky, flare, flare2,
    //         sun_screen, sharpen_params) + mat4 inv_view_proj.
    compose_pc.size = sizeof(glm::vec4) * 9 + sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo cplci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &compose_desc_set_layout_,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &compose_pc,
    };
    vk_check(vkCreatePipelineLayout(device_, &cplci, nullptr, &compose_pipeline_layout_),
             "compose pipeline layout");

    {
        std::string sd2 = QLIKE_SHADER_DIR;
        compose_frag_module_ = vkpipe::load_shader_module(device_, sd2 + "/compose.frag.spv");
    }
    vkpipe::GraphicsPipelineConfig ccfg{};
    ccfg.vert = taa_vert_module_;       // reuse the fullscreen-triangle VS
    ccfg.frag = compose_frag_module_;
    ccfg.layout = compose_pipeline_layout_;
    ccfg.color_format = swapchain_format_;
    ccfg.depth_format = VK_FORMAT_UNDEFINED;
    ccfg.cull = VK_CULL_MODE_NONE;
    ccfg.depth_test = false;
    ccfg.depth_write = false;
    compose_pipeline_ = vkpipe::build_graphics_pipeline(device_, ccfg);
    log::info("Compose pass initialized (ACES Fitted tonemap + sRGB encode)");
}

void VulkanEngine::init_bloom() {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .bindingCount = 1, .pBindings = &b,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                         &bloom_desc_set_layout_),
             "bloom set layout");

    VkDescriptorPoolSize sz{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              static_cast<uint32_t>(kHistorySlots * kBloomMips + kBloomMips) };
    VkDescriptorPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .maxSets = static_cast<uint32_t>(kHistorySlots * kBloomMips + kBloomMips),
        .poolSizeCount = 1, .pPoolSizes = &sz,
    };
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &bloom_desc_pool_),
             "bloom desc pool");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.offset = 0;
    pc.size = sizeof(glm::vec4) * 3;
    VkPipelineLayoutCreateInfo plci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &bloom_desc_set_layout_,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
    };
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &bloom_pipeline_layout_),
             "bloom pipeline layout");

    std::string sd = QLIKE_SHADER_DIR;
    bloom_down_frag_ = vkpipe::load_shader_module(device_, sd + "/bloom_down.frag.spv");
    bloom_up_frag_   = vkpipe::load_shader_module(device_, sd + "/bloom_up.frag.spv");

    vkpipe::GraphicsPipelineConfig dcfg{};
    dcfg.vert = taa_vert_module_;
    dcfg.frag = bloom_down_frag_;
    dcfg.layout = bloom_pipeline_layout_;
    dcfg.color_format = scene_color_format_;
    dcfg.depth_format = VK_FORMAT_UNDEFINED;
    dcfg.cull = VK_CULL_MODE_NONE;
    dcfg.depth_test = false;
    dcfg.depth_write = false;
    bloom_down_pipeline_ = vkpipe::build_graphics_pipeline(device_, dcfg);

    vkpipe::GraphicsPipelineConfig ucfg = dcfg;
    ucfg.frag = bloom_up_frag_;
    ucfg.additive_blend = true;
    bloom_up_pipeline_ = vkpipe::build_graphics_pipeline(device_, ucfg);

    recreate_bloom_targets();
    log::info("Bloom mip chain initialized");
}

void VulkanEngine::rewrite_compose_image_bindings() {
    if (!compose_desc_set_layout_) return;
    for (int s = 0; s < kHistorySlots; ++s) {
        VkDescriptorImageInfo i_hist{ linear_sampler_, history_view_[s],
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth{ linear_sampler_, depth_view_,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_sky { skybox_.sampler, skybox_.view,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_bloom{ linear_sampler_, bloom_mip_views_[0],
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[4]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = compose_desc_sets_[s];
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &i_hist;
        w[1] = w[0]; w[1].dstBinding = 1; w[1].pImageInfo = &i_depth;
        w[2] = w[0]; w[2].dstBinding = 2; w[2].pImageInfo = &i_sky;
        w[3] = w[0]; w[3].dstBinding = 3; w[3].pImageInfo = &i_bloom;
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
    }
}

void VulkanEngine::recreate_bloom_targets() {
    for (int i = 0; i < kBloomMips; ++i) {
        if (bloom_mip_views_[i]) {
            vkDestroyImageView(device_, bloom_mip_views_[i], nullptr);
            bloom_mip_views_[i] = VK_NULL_HANDLE;
        }
    }
    if (bloom_image_) {
        vmaDestroyImage(allocator_, bloom_image_, bloom_alloc_);
        bloom_image_ = VK_NULL_HANDLE;
        bloom_alloc_ = nullptr;
    }
    // The bloom desc pool is sized for exactly one allocation's worth of
    // sets. Without resetting it, the second call (resize / resolution
    // change) hits VK_ERROR_OUT_OF_POOL_MEMORY → throw → engine exits.
    if (bloom_desc_pool_) {
        vkResetDescriptorPool(device_, bloom_desc_pool_, 0);
    }

    // Bloom mip 0 is half-res of the *render* target, not the swapchain —
    // bloom samples history (which is at render_extent_) so its source
    // resolution scales with render_scale too.
    uint32_t w0 = std::max<uint32_t>(1, render_extent_.width  / 2);
    uint32_t h0 = std::max<uint32_t>(1, render_extent_.height / 2);
    for (int i = 0; i < kBloomMips; ++i) {
        bloom_mip_extents_[i] = { std::max(1u, w0 >> i),
                                  std::max(1u, h0 >> i) };
    }

    VkImageCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = scene_color_format_,
        .extent = { w0, h0, 1 },
        .mipLevels = static_cast<uint32_t>(kBloomMips),
        .arrayLayers = 1,
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
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &bloom_image_, &bloom_alloc_, nullptr),
             "bloom image");

    for (int i = 0; i < kBloomMips; ++i) {
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .image = bloom_image_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = scene_color_format_,
            .components = {},
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                  static_cast<uint32_t>(i), 1, 0, 1 },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &bloom_mip_views_[i]),
                 "bloom mip view");
    }

    {
        std::vector<VkDescriptorSetLayout> layouts(kHistorySlots * kBloomMips,
                                                    bloom_desc_set_layout_);
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = bloom_desc_pool_,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };
        std::vector<VkDescriptorSet> sets(layouts.size());
        vk_check(vkAllocateDescriptorSets(device_, &dai, sets.data()),
                 "bloom down desc sets");
        for (int s = 0; s < kHistorySlots; ++s)
            for (int m = 0; m < kBloomMips; ++m)
                bloom_down_sets_[s][m] = sets[s * kBloomMips + m];
    }
    {
        std::vector<VkDescriptorSetLayout> layouts(kBloomMips, bloom_desc_set_layout_);
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = bloom_desc_pool_,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai, bloom_up_sets_),
                 "bloom up desc sets");
    }

    for (int s = 0; s < kHistorySlots; ++s) {
        for (int m = 0; m < kBloomMips; ++m) {
            VkImageView src_view = (m == 0)
                ? history_view_[s] : bloom_mip_views_[m - 1];
            VkDescriptorImageInfo info{ linear_sampler_, src_view,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = bloom_down_sets_[s][m];
            w.dstBinding = 0; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &info;
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }
    }
    for (int m = 0; m < kBloomMips - 1; ++m) {
        VkDescriptorImageInfo info{ linear_sampler_, bloom_mip_views_[m + 1],
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = bloom_up_sets_[m];
        w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
}

void VulkanEngine::destroy_bloom() {
    if (bloom_down_pipeline_) vkDestroyPipeline(device_, bloom_down_pipeline_, nullptr);
    if (bloom_up_pipeline_)   vkDestroyPipeline(device_, bloom_up_pipeline_, nullptr);
    if (bloom_pipeline_layout_) vkDestroyPipelineLayout(device_, bloom_pipeline_layout_, nullptr);
    if (bloom_down_frag_)     vkDestroyShaderModule(device_, bloom_down_frag_, nullptr);
    if (bloom_up_frag_)       vkDestroyShaderModule(device_, bloom_up_frag_, nullptr);
    if (bloom_desc_set_layout_) vkDestroyDescriptorSetLayout(device_, bloom_desc_set_layout_, nullptr);
    if (bloom_desc_pool_)     vkDestroyDescriptorPool(device_, bloom_desc_pool_, nullptr);
    for (int i = 0; i < kBloomMips; ++i) {
        if (bloom_mip_views_[i]) vkDestroyImageView(device_, bloom_mip_views_[i], nullptr);
    }
    if (bloom_image_) vmaDestroyImage(allocator_, bloom_image_, bloom_alloc_);
    bloom_down_pipeline_ = bloom_up_pipeline_ = VK_NULL_HANDLE;
    bloom_pipeline_layout_ = VK_NULL_HANDLE;
    bloom_down_frag_ = bloom_up_frag_ = VK_NULL_HANDLE;
    bloom_desc_set_layout_ = VK_NULL_HANDLE;
    bloom_desc_pool_ = VK_NULL_HANDLE;
    bloom_image_ = VK_NULL_HANDLE; bloom_alloc_ = nullptr;
    for (auto& v : bloom_mip_views_) v = VK_NULL_HANDLE;
}

void VulkanEngine::run_bloom_chain(VkCommandBuffer cmd) {
    if (!bloom_image_) return;
    // Run unconditionally even if bloom_enabled is off — compose's binding 3
    // points at this image and Vulkan requires it to be in a defined layout
    // when the descriptor is bound.

    auto draw_fullscreen = [&](VkPipeline pipe, VkDescriptorSet set,
                               VkImageView dst_view, VkExtent2D dst_ext,
                               VkExtent2D src_ext, float param_x, float param_y,
                               bool additive) {
        VkRenderingAttachmentInfo att{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = dst_view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = additive ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {},
        };
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0,0}, dst_ext },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &att,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{};
        vp.width = static_cast<float>(dst_ext.width);
        vp.height = static_cast<float>(dst_ext.height);
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0,0}, dst_ext };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                bloom_pipeline_layout_, 0, 1, &set, 0, nullptr);
        struct PC {
            glm::vec4 src_extent;
            glm::vec4 dst_extent;
            glm::vec4 params;
        } pc_data{};
        pc_data.src_extent = glm::vec4(src_ext.width, src_ext.height,
                                       1.0f / float(src_ext.width),
                                       1.0f / float(src_ext.height));
        pc_data.dst_extent = glm::vec4(dst_ext.width, dst_ext.height,
                                       1.0f / float(dst_ext.width),
                                       1.0f / float(dst_ext.height));
        pc_data.params = glm::vec4(param_x, param_y, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, bloom_pipeline_layout_,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc_data), &pc_data);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    };

    // All mips start UNDEFINED → COLOR_ATTACHMENT (single barrier).
    vkinit::transition_image_mip(cmd, bloom_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 0, kBloomMips);

    // 1. Downsample chain. Mip 0 reads from history at render_extent_;
    // subsequent mips read the previous bloom mip.
    for (int m = 0; m < kBloomMips; ++m) {
        VkExtent2D src_ext = (m == 0) ? render_extent_ : bloom_mip_extents_[m - 1];
        if (m > 0) {
            vkinit::transition_image_mip(cmd, bloom_image_,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                static_cast<uint32_t>(m - 1), 1);
        }
        draw_fullscreen(bloom_down_pipeline_,
                        bloom_down_sets_[history_write_slot_][m],
                        bloom_mip_views_[m],
                        bloom_mip_extents_[m],
                        src_ext,
                        m == 0 ? 1.0f : 0.0f,
                        m == 0 ? rt_.bloom_threshold : 0.0f,
                        /*additive=*/false);
    }

    // 2. Upsample chain (additive). Source = mip m+1, dest = mip m.
    for (int m = kBloomMips - 2; m >= 0; --m) {
        VkImageMemoryBarrier2 barriers[2]{};
        uint32_t bn = 0;
        auto fill = [&](VkImageMemoryBarrier2& b, VkImageLayout from,
                        VkImageLayout to, uint32_t mip) {
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT |
                              VK_ACCESS_2_MEMORY_READ_BIT;
            b.oldLayout = from;
            b.newLayout = to;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = bloom_image_;
            b.subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1,
            };
        };
        fill(barriers[bn++], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, static_cast<uint32_t>(m + 1));
        fill(barriers[bn++], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, static_cast<uint32_t>(m));
        VkDependencyInfo dep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr, .dependencyFlags = 0,
            .memoryBarrierCount = 0, .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0, .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = bn, .pImageMemoryBarriers = barriers,
        };
        vkCmdPipelineBarrier2(cmd, &dep);

        draw_fullscreen(bloom_up_pipeline_,
                        bloom_up_sets_[m],
                        bloom_mip_views_[m],
                        bloom_mip_extents_[m],
                        bloom_mip_extents_[m + 1],
                        /*filter_radius=*/1.0f,
                        /*param_y=*/0.0f,
                        /*additive=*/true);
    }

    // Final transition: mip 0 → SHADER_READ so compose can sample it.
    vkinit::transition_image_mip(cmd, bloom_image_,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 0, 1);
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
    destroy_bloom();
    if (compose_pipeline_)         vkDestroyPipeline(device_, compose_pipeline_, nullptr);
    if (compose_pipeline_layout_)  vkDestroyPipelineLayout(device_, compose_pipeline_layout_, nullptr);
    if (compose_frag_module_)      vkDestroyShaderModule(device_, compose_frag_module_, nullptr);
    if (compose_desc_set_layout_)  vkDestroyDescriptorSetLayout(device_, compose_desc_set_layout_, nullptr);
    if (compose_desc_pool_)        vkDestroyDescriptorPool(device_, compose_desc_pool_, nullptr);

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
