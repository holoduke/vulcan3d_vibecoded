// DLSS Ray Reconstruction guide buffers.
//
// DLSS-RR replaces the upscaler AND the denoisers: it takes the noisy
// ray-traced frame plus per-pixel material guides and resolves a clean,
// upscaled image. The guides it needs that the engine did not have as
// separate textures - diffuse albedo, specular albedo, world normal +
// roughness - are decoded from the G-buffer by rr_guides.comp right before
// the DLSS dispatch (dlss.cpp tags them).

#include "engine/vk_engine/internal.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <string>

namespace qlike {

namespace {
constexpr VkFormat kRrFormats[3] = {
    VK_FORMAT_R8G8B8A8_UNORM,        // diffuse albedo
    VK_FORMAT_R8G8B8A8_UNORM,        // specular albedo
    VK_FORMAT_R16G16B16A16_SFLOAT,   // normal xyz + roughness
};
struct RrGuidesPC {
    glm::mat4 inv_vp;
    glm::vec4 cam_pos;
};
}

VkFormat VulkanEngine::rr_guide_format(int i) const { return kRrFormats[i]; }

void VulkanEngine::destroy_rr_guide_images() {
    for (int i = 0; i < 3; ++i) {
        if (rr_view_[i]) vkDestroyImageView(device_, rr_view_[i], nullptr);
        if (rr_image_[i]) vmaDestroyImage(allocator_, rr_image_[i], rr_alloc_[i]);
        rr_view_[i] = VK_NULL_HANDLE; rr_image_[i] = VK_NULL_HANDLE; rr_alloc_[i] = nullptr;
    }
    rr_extent_ = { 0, 0 };
}

void VulkanEngine::destroy_rr_guides() {
    destroy_rr_guide_images();
    if (rr_pipeline_) vkDestroyPipeline(device_, rr_pipeline_, nullptr);
    if (rr_pipeline_layout_) vkDestroyPipelineLayout(device_, rr_pipeline_layout_, nullptr);
    if (rr_pool_) vkDestroyDescriptorPool(device_, rr_pool_, nullptr);
    if (rr_set_layout_) vkDestroyDescriptorSetLayout(device_, rr_set_layout_, nullptr);
    if (rr_sampler_) vkDestroySampler(device_, rr_sampler_, nullptr);
    rr_pipeline_ = VK_NULL_HANDLE; rr_pipeline_layout_ = VK_NULL_HANDLE;
    rr_pool_ = VK_NULL_HANDLE; rr_set_layout_ = VK_NULL_HANDLE; rr_set_ = VK_NULL_HANDLE;
    rr_sampler_ = VK_NULL_HANDLE;
}

// Called before command recording (ensure_dlss_bindings), with the GPU idle:
// (re)creates the pipeline once and the images whenever the render extent
// or the G-buffer views changed.
void VulkanEngine::ensure_rr_guides() {
    if (!rr_pipeline_) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST; sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &sci, nullptr, &rr_sampler_), "rr sampler");

        VkDescriptorSetLayoutBinding b[7]{};
        for (int i = 0; i < 7; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i < 3 || i == 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 7; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &rr_set_layout_), "rr set layout");
        VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
                                       { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 } };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &rr_pool_), "rr pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = rr_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &rr_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &rr_set_), "rr set");

        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RrGuidesPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &rr_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &rr_pipeline_layout_), "rr layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/rr_guides.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = rr_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &rr_pipeline_),
                 "rr guides pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }

    const bool stale = rr_extent_.width != render_extent_.width ||
                       rr_extent_.height != render_extent_.height ||
                       rr_src_views_[0] != gbuffer0_view_ || rr_src_views_[1] != gbuffer1_view_ ||
                       rr_src_views_[2] != depth_view_ || rr_src_views_[3] != scene_color_view_;
    if (!stale) return;

    destroy_rr_guide_images();
    for (int i = 0; i < 3; ++i) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = kRrFormats[i];
        ici.extent = { render_extent_.width, render_extent_.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &rr_image_[i], &rr_alloc_[i], nullptr), "rr guide");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = rr_image_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = kRrFormats[i];
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &rr_view_[i]), "rr guide view");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (int i = 0; i < 3; ++i)
            vkinit::transition_image(cb, rr_image_[i], VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    rr_extent_ = render_extent_;
    rr_src_views_[0] = gbuffer0_view_; rr_src_views_[1] = gbuffer1_view_; rr_src_views_[2] = depth_view_;
    rr_src_views_[3] = scene_color_view_;

    VkDescriptorImageInfo ii[7] = {
        { rr_sampler_, gbuffer0_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { rr_sampler_, gbuffer1_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { rr_sampler_, depth_view_,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { VK_NULL_HANDLE, rr_view_[0], VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, rr_view_[1], VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, rr_view_[2], VK_IMAGE_LAYOUT_GENERAL },
        { rr_sampler_, scene_color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
    };
    VkWriteDescriptorSet w[7]{};
    for (int i = 0; i < 7; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = rr_set_; w[i].dstBinding = uint32_t(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i < 3 || i == 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device_, 7, w, 0, nullptr);
    log::infof("[dlss-rr] guide buffers %ux%u", rr_extent_.width, rr_extent_.height);
}

// G-buffer + depth are SHADER_READ_ONLY here (post-world transition). The
// guides are left SHADER_READ_ONLY for the DLSS-RR evaluate.
void VulkanEngine::dispatch_rr_guides(VkCommandBuffer cmd) {
    if (!rr_pipeline_ || !rr_image_[0]) return;
    for (int i = 0; i < 3; ++i)
        vkinit::transition_image(cmd, rr_image_[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rr_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rr_pipeline_layout_, 0, 1, &rr_set_, 0, nullptr);
    RrGuidesPC pc{};
    pc.inv_vp = last_inv_view_proj_;
    pc.cam_pos = glm::vec4(current_frame_view_.eye_pos, 0.0f);
    vkCmdPushConstants(cmd, rr_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (rr_extent_.width + 7) / 8, (rr_extent_.height + 7) / 8, 1);
    for (int i = 0; i < 3; ++i)
        vkinit::transition_image(cmd, rr_image_[i], VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace qlike
