// Compose pass — samples TAA history (HDR), tonemaps, blends bloom +
// lens flare, encodes to swapchain (LDR). The TAA pass that produces
// that history lives in vk_engine/taa.cpp; the bloom mip chain that
// feeds binding 3 lives in vk_engine/bloom.cpp.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

namespace qlike {

void VulkanEngine::init_compose() {
    // Compose set's binding 3 samples bloom_mip_views_[0], so init_bloom()
    // must have run first. linear_sampler_ comes from init_taa(). Both are
    // sequenced from VulkanEngine::init() — this method just assumes them.
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

void VulkanEngine::rewrite_compose_image_bindings() {
    if (!compose_desc_set_layout_) return;
    // When TAAU is on, compose reads the native upscaled output instead of
    // the LR TAA history. taau_view_ ping-pong matches history_view_ slot.
    const bool use_taau = rt_.taau_enabled && taau_image_[0] != VK_NULL_HANDLE;
    compose_uses_taau_ = use_taau;
    for (int s = 0; s < kHistorySlots; ++s) {
        VkImageView hist_src = use_taau ? taau_view_[s] : history_view_[s];
        VkDescriptorImageInfo i_hist{ linear_sampler_, hist_src,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth{ linear_sampler_, depth_view_,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_sky { skybox_.sampler, skybox_.view,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_bloom{ linear_sampler_,
                                        bloom_full_view_ ? bloom_full_view_
                                                          : bloom_mip_views_[0],
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

void VulkanEngine::destroy_compose() {
    if (compose_pipeline_)         vkDestroyPipeline(device_, compose_pipeline_, nullptr);
    if (compose_pipeline_layout_)  vkDestroyPipelineLayout(device_, compose_pipeline_layout_, nullptr);
    if (compose_frag_module_)      vkDestroyShaderModule(device_, compose_frag_module_, nullptr);
    if (compose_desc_set_layout_)  vkDestroyDescriptorSetLayout(device_, compose_desc_set_layout_, nullptr);
    if (compose_desc_pool_)        vkDestroyDescriptorPool(device_, compose_desc_pool_, nullptr);
    compose_pipeline_         = VK_NULL_HANDLE;
    compose_pipeline_layout_  = VK_NULL_HANDLE;
    compose_frag_module_      = VK_NULL_HANDLE;
    compose_desc_set_layout_  = VK_NULL_HANDLE;
    compose_desc_pool_        = VK_NULL_HANDLE;
}

} // namespace qlike
