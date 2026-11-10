// World-space radiance cache (SHaRC-style hash grid).
//
// deferred_lighting.frag scatters the diffuse outgoing radiance of ~1/8 of
// the on-screen pixels into a spatial hash (cell size grows with camera
// distance, 6 normal-axis buckets per cell). GI rays that hit geometry look
// the hit cell up instead of shading the hit with a proxy albedo + one sun
// ray, so bounce light carries the real surface colour and - because the
// cached radiance itself includes GI - multiple bounces for free.
//
// Buffers:  keys  uint[N]          0 = empty, else hash check value
//           acc   uvec4[N]         rgb fixed-point sum (x256) + sample count
//           res   vec4[N]          resolved radiance rgb + age (frames)
// rc_resolve.comp averages acc into res (EMA), clears acc, ages and evicts.

#include "engine/vk_engine/internal.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <string>

namespace qlike {

void VulkanEngine::init_radiance_cache() {
    const VkDeviceSize n = kRadianceCacheEntries;
    const VkDeviceSize sizes[3] = { n * 4, n * 16, n * 16 };
    for (int i = 0; i < 3; ++i) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizes[i];
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &rc_buf_[i], &rc_alloc_[i], nullptr),
                 "radiance cache buffer");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (int i = 0; i < 3; ++i) vkCmdFillBuffer(cb, rc_buf_[i], 0, VK_WHOLE_SIZE, 0u);
    });

    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; ++i) {
        b[i].binding = uint32_t(i);
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 3; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &rc_set_layout_), "rc set layout");
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &rc_pool_), "rc pool");
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = rc_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &rc_set_layout_;
    vk_check(vkAllocateDescriptorSets(device_, &dai, &rc_set_), "rc set");
    VkDescriptorBufferInfo bi[3];
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) {
        bi[i] = { rc_buf_[i], 0, VK_WHOLE_SIZE };
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = rc_set_; w[i].dstBinding = uint32_t(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);

    VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 4 };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &rc_set_layout_;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &rc_pipeline_layout_), "rc layout");
    VkShaderModule m = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/rc_resolve.comp.spv");
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = m;
    cci.stage.pName = "main";
    cci.layout = rc_pipeline_layout_;
    vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &rc_pipeline_),
             "rc pipeline");
    vkDestroyShaderModule(device_, m, nullptr);
    log::infof("[rcache] world-space radiance cache ready (%u entries, %.0f MB)",
               kRadianceCacheEntries, double(n * 36) / (1024.0 * 1024.0));
}

void VulkanEngine::destroy_radiance_cache() {
    if (rc_pipeline_) vkDestroyPipeline(device_, rc_pipeline_, nullptr);
    if (rc_pipeline_layout_) vkDestroyPipelineLayout(device_, rc_pipeline_layout_, nullptr);
    if (rc_pool_) vkDestroyDescriptorPool(device_, rc_pool_, nullptr);
    if (rc_set_layout_) vkDestroyDescriptorSetLayout(device_, rc_set_layout_, nullptr);
    for (int i = 0; i < 3; ++i) {
        if (rc_buf_[i]) vmaDestroyBuffer(allocator_, rc_buf_[i], rc_alloc_[i]);
        rc_buf_[i] = VK_NULL_HANDLE; rc_alloc_[i] = nullptr;
    }
    rc_pipeline_ = VK_NULL_HANDLE; rc_pipeline_layout_ = VK_NULL_HANDLE;
    rc_pool_ = VK_NULL_HANDLE; rc_set_layout_ = VK_NULL_HANDLE; rc_set_ = VK_NULL_HANDLE;
}

// Runs right after the deferred lighting pass (which scattered this
// frame's samples into acc). The trailing barrier orders the resolved
// buffer before the NEXT frame's deferred reads (barriers cover all later
// submissions on the queue).
void VulkanEngine::dispatch_radiance_cache(VkCommandBuffer cmd) {
    if (!rc_pipeline_) return;
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);

    if (!rt_.radiance_cache) {
        // Off: drop this frame's samples and invalidate every resolved
        // entry (age 0 = never resolved), so rc_lookup always misses.
        mb.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier2(cmd, &di);
        vkCmdFillBuffer(cmd, rc_buf_[1], 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(cmd, rc_buf_[2], 0, VK_WHOLE_SIZE, 0u);
        mb.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        vkCmdPipelineBarrier2(cmd, &di);
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rc_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rc_pipeline_layout_, 0, 1, &rc_set_, 0, nullptr);
    float pc[4] = { float(kRadianceCacheEntries), 0.12f, 0.0f, 0.0f };
    vkCmdPushConstants(cmd, rc_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, kRadianceCacheEntries / 256, 1, 1);

    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &di);
}

} // namespace qlike
