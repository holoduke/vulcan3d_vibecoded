// Compose pass — samples TAA history (HDR), tonemaps, blends bloom +
// lens flare, encodes to swapchain (LDR). The TAA pass that produces
// that history lives in vk_engine/taa.cpp; the bloom mip chain that
// feeds binding 3 lives in vk_engine/bloom.cpp.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

#include <GLFW/glfw3.h>
#include <algorithm>

namespace qlike {

void VulkanEngine::init_compose() {
    // Compose set's binding 3 samples bloom_mip_views_[0], so init_bloom()
    // must have run first. linear_sampler_ comes from init_taa(). Both are
    // sequenced from VulkanEngine::init() — this method just assumes them.
    init_exposure();   // SSBO + compute; compose binding 4 reads the buffer
    init_atmosphere();  // LUTs + aerial volume; compose bindings 5-7 sample them
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kHistorySlots * 8 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kHistorySlots },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = kHistorySlots, .poolSizeCount = 2, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &compose_desc_pool_),
                 "compose desc pool");
    }
    {
        VkDescriptorSetLayoutBinding b[9]{};
        for (int i = 0; i < 9; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // exposure SSBO
        VkDescriptorSetLayoutCreateInfo lci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .bindingCount = 9, .pBindings = b,
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
    // 10 vec4 (viewport, bloom, sun_dir, sun_color, sky, flare, flare2,
    //          sun_screen, sharpen_params, brush_indicator) +
    //          mat4 inv_view_proj.
    compose_pc.size = sizeof(glm::vec4) * 11 + sizeof(glm::mat4);   // + atmo params

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
    // Four sources for compose's history binding, in priority order:
    //   FSR3  — HR upscaled output via ffx-api (FidelityFX SDK 1.1.4)
    //   FSR2  — HR upscaled output via legacy FSR2 static lib
    //   TAAU  — HR temporal upsample
    //   TAA   — LR temporal AA history (default)
    // FSR3 only wins when fsr2_enabled is on AND fsr_backend == 1; the
    // FSR2/FSR3 selection is mutually exclusive (one upscale per frame).
    const bool use_fsr3 = rt_.fsr2_enabled && rt_.fsr_backend == 1 &&
                          fsr3_output_view_ != VK_NULL_HANDLE;
    const bool use_fsr2 = !use_fsr3 && rt_.fsr2_enabled &&
                          fsr2_output_view_ != VK_NULL_HANDLE;
    const bool use_taau = !use_fsr3 && !use_fsr2 && rt_.taau_enabled &&
                          taau_image_[0] != VK_NULL_HANDLE;
    compose_uses_fsr3_ = use_fsr3;
    compose_uses_fsr2_ = use_fsr2;
    compose_uses_taau_ = use_taau;
    for (int s = 0; s < kHistorySlots; ++s) {
        VkImageView hist_src = use_fsr3 ? fsr3_output_view_
                              : use_fsr2 ? fsr2_output_view_
                              : use_taau ? taau_view_[s]
                              : history_view_[s];
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
        VkDescriptorBufferInfo bi{ exposure_buf_, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet wb = w[0];
        wb.dstBinding = 4;
        wb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wb.pImageInfo = nullptr;
        wb.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
        if (exposure_buf_) vkUpdateDescriptorSets(device_, 1, &wb, 0, nullptr);
        // Atmosphere LUTs (GENERAL layout, see atmosphere.cpp).
        if (atmo_sampler_) {
            VkDescriptorImageInfo ai[4] = {
                { atmo_sampler_, atmo_trans_.view, VK_IMAGE_LAYOUT_GENERAL },
                { atmo_sampler_, atmo_sky_.view,   VK_IMAGE_LAYOUT_GENERAL },
                { atmo_sampler_, atmo_ap_.view,    VK_IMAGE_LAYOUT_GENERAL },
                { atmo_sampler_, cloud_img_.view ? cloud_img_.view : atmo_sky_.view, VK_IMAGE_LAYOUT_GENERAL },
            };
            VkWriteDescriptorSet wa[4]{};
            for (int k = 0; k < 4; ++k) {
                wa[k] = w[0];
                wa[k].dstBinding = uint32_t(5 + k);
                wa[k].pImageInfo = &ai[k];
            }
            vkUpdateDescriptorSets(device_, 4, wa, 0, nullptr);
        }
        // Exposure compute reads the same image compose shows.
        if (exposure_sets_[s]) {
            VkWriteDescriptorSet we[3]{};
            we[0] = w[0]; we[0].dstSet = exposure_sets_[s]; we[0].dstBinding = 0;
            we[1] = w[1]; we[1].dstSet = exposure_sets_[s]; we[1].dstBinding = 1;
            we[2] = wb;   we[2].dstSet = exposure_sets_[s]; we[2].dstBinding = 2;
            vkUpdateDescriptorSets(device_, 3, we, 0, nullptr);
        }
    }
}

void VulkanEngine::destroy_compose() {
    destroy_exposure();
    destroy_atmosphere();
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


// ---------------------------------------------------------------------
// Auto-exposure (shaders/exposure.comp)
// ---------------------------------------------------------------------
void VulkanEngine::init_exposure() {
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 16;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &exposure_buf_, &exposure_alloc_, nullptr),
                 "exposure buffer");
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkCmdFillBuffer(cb, exposure_buf_, 0, 16, 0u);   // valid = 0 -> first frame snaps
        });
    }
    {
        VkDescriptorSetLayoutBinding b[3]{};
        for (int i = 0; i < 3; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i == 2) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &exposure_set_layout_),
                 "exposure set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kHistorySlots * 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kHistorySlots },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = kHistorySlots; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &exposure_pool_), "exposure pool");
        VkDescriptorSetLayout layouts[kHistorySlots] = { exposure_set_layout_, exposure_set_layout_ };
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = exposure_pool_;
        dai.descriptorSetCount = kHistorySlots; dai.pSetLayouts = layouts;
        vk_check(vkAllocateDescriptorSets(device_, &dai, exposure_sets_), "exposure sets");
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 4 };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &exposure_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &exposure_pipeline_layout_),
                 "exposure pipeline layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/exposure.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = exposure_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &exposure_pipeline_),
                 "exposure pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    log::info("[exposure] luminance-reduction auto-exposure ready");
}

void VulkanEngine::destroy_exposure() {
    if (exposure_pipeline_) vkDestroyPipeline(device_, exposure_pipeline_, nullptr);
    if (exposure_pipeline_layout_) vkDestroyPipelineLayout(device_, exposure_pipeline_layout_, nullptr);
    if (exposure_pool_) vkDestroyDescriptorPool(device_, exposure_pool_, nullptr);
    if (exposure_set_layout_) vkDestroyDescriptorSetLayout(device_, exposure_set_layout_, nullptr);
    if (exposure_buf_) vmaDestroyBuffer(allocator_, exposure_buf_, exposure_alloc_);
    exposure_pipeline_ = VK_NULL_HANDLE; exposure_pipeline_layout_ = VK_NULL_HANDLE;
    exposure_pool_ = VK_NULL_HANDLE; exposure_set_layout_ = VK_NULL_HANDLE;
    exposure_buf_ = VK_NULL_HANDLE; exposure_alloc_ = nullptr;
    exposure_sets_[0] = exposure_sets_[1] = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_exposure(VkCommandBuffer cmd) {
    if (!exposure_pipeline_ || rt_.auto_exposure_strength <= 0.0f) return;
    const double now = glfwGetTime();
    float dt = exposure_last_t_ > 0.0 ? float(now - exposure_last_t_) : 0.0f;
    exposure_last_t_ = now;
    dt = std::min(dt, 0.25f);
    // Sky luminance estimate: the sky colour the compose pass paints.
    const glm::vec3 sky = rt_.sky_color;
    float sky_lum = 0.2126f * sky.r + 0.7152f * sky.g + 0.0722f * sky.b;
    if (skybox_.hdr && skybox_.avg_lum > 0.0f) sky_lum = skybox_.avg_lum;   // what compose paints
    sky_lum = std::max(0.05f, sky_lum);
    struct { float dt, up, down, sky; } pc{ dt, 2.5f, 0.9f, sky_lum };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_pipeline_layout_, 0, 1,
                            &exposure_sets_[history_write_slot_], 0, nullptr);
    vkCmdPushConstants(cmd, exposure_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
    VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

} // namespace qlike
