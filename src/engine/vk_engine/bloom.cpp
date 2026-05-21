// Bloom mip chain — Karis-prefiltered downsample + additive upsample.
// Lives in its own TU so post.cpp doesn't carry both TAA and bloom plus
// compose-binding rewrite all in one ~700-line file.
//
// Still implemented as VulkanEngine member methods (init_bloom,
// recreate_bloom_targets, destroy_bloom, run_bloom_chain). The state
// fields stay in vk_engine.h for now; a future pass can wrap them in
// a BloomChain RAII class when the cross-cutting dependencies on
// taa_vert_module_, history_view_, linear_sampler_ etc. are
// disentangled.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <vector>

namespace qlike {

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

void VulkanEngine::recreate_bloom_targets() {
    for (int i = 0; i < kBloomMips; ++i) {
        if (bloom_mip_views_[i]) {
            vkDestroyImageView(device_, bloom_mip_views_[i], nullptr);
            bloom_mip_views_[i] = VK_NULL_HANDLE;
        }
    }
    if (bloom_full_view_) {
        vkDestroyImageView(device_, bloom_full_view_, nullptr);
        bloom_full_view_ = VK_NULL_HANDLE;
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
    // All-mips view for compose. Lets compose.frag textureLod into the
    // smallest mip (kBloomMips-1) as a coarse scene-average proxy for
    // auto-exposure.
    if (bloom_full_view_) {
        vkDestroyImageView(device_, bloom_full_view_, nullptr);
        bloom_full_view_ = VK_NULL_HANDLE;
    }
    {
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .image = bloom_image_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = scene_color_format_,
            .components = {},
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                  0, static_cast<uint32_t>(kBloomMips), 0, 1 },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &bloom_full_view_),
                 "bloom full view");
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
    if (bloom_full_view_) vkDestroyImageView(device_, bloom_full_view_, nullptr);
    if (bloom_image_) vmaDestroyImage(allocator_, bloom_image_, bloom_alloc_);
    bloom_down_pipeline_ = bloom_up_pipeline_ = VK_NULL_HANDLE;
    bloom_pipeline_layout_ = VK_NULL_HANDLE;
    bloom_down_frag_ = bloom_up_frag_ = VK_NULL_HANDLE;
    bloom_desc_set_layout_ = VK_NULL_HANDLE;
    bloom_desc_pool_ = VK_NULL_HANDLE;
    bloom_image_ = VK_NULL_HANDLE; bloom_alloc_ = nullptr;
    bloom_full_view_ = VK_NULL_HANDLE;
    for (auto& v : bloom_mip_views_) v = VK_NULL_HANDLE;
}

void VulkanEngine::run_bloom_chain(VkCommandBuffer cmd) {
    if (!bloom_image_) return;

    // Bloom-disabled fast path. compose's binding 3 still needs a valid
    // image layout, but it doesn't need actual bloom contents — compose's
    // bloom_params.x (strength) is gated separately, and a zero-clear
    // means even if the user toggles strength back on without bloom we
    // contribute zero color. Skips the whole downsample + upsample chain
    // (~kBloomMips × 2 fullscreen draws + ~kBloomMips barriers) when the
    // user has bloom off in the UI.
    if (!rt_.bloom_enabled) {
        vkinit::transition_image_mip(cmd, bloom_image_,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     0, kBloomMips);
        VkClearColorValue clear_c{};                       // zero
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT,
                                        0, kBloomMips, 0, 1 };
        vkCmdClearColorImage(cmd, bloom_image_,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              &clear_c, 1, &range);
        vkinit::transition_image_mip(cmd, bloom_image_,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     0, kBloomMips);
        return;
    }

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
    // Each iteration needs the previous draw's COLOR_ATTACHMENT write
    // visible to this draw's FRAGMENT_SHADER read. BY_REGION_BIT lets
    // tile-based GPUs do the dependency per-tile instead of full-frame
    // — desktop drivers ignore the flag, so it's free downside.
    for (int m = kBloomMips - 2; m >= 0; --m) {
        VkImageMemoryBarrier2 barriers[2]{};
        uint32_t bn = 0;
        auto fill = [&](VkImageMemoryBarrier2& b, VkImageLayout from,
                        VkImageLayout to, uint32_t mip) {
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            // src covers BOTH possible prior uses of this mip — it was
            // either just written as a colour attachment (downsample /
            // previous upsample dest) or sampled as a shader texture.
            // Covering both is safe and avoids a stale-src hazard.
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_2_SHADER_READ_BIT;
            // dst MUST match how the mip is consumed next, derived
            // from the target layout — NOT hard-coded. The previous
            // code used FRAGMENT_SHADER/SHADER_READ for both barriers,
            // but the →COLOR_ATTACHMENT barrier feeds an additive
            // loadOp=LOAD draw which reads+writes at
            // COLOR_ATTACHMENT_OUTPUT. That mismatch was the
            // sync-validation READ_AFTER_WRITE hazard on the bloom
            // LOAD attachment (a real intermittent device-lost risk).
            if (to == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                b.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            } else {  // → SHADER_READ_ONLY_OPTIMAL (sampled next)
                b.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            }
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
            .pNext = nullptr,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
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

} // namespace qlike
