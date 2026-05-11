#include "engine/vk_initializers.h"

#include <vector>

namespace qlike::vkinit {

VkCommandPoolCreateInfo command_pool_create_info(uint32_t queue_family,
                                                 VkCommandPoolCreateFlags flags) {
    return VkCommandPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .queueFamilyIndex = queue_family,
    };
}

VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count) {
    return VkCommandBufferAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count,
    };
}

VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags) {
    return VkCommandBufferBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = flags,
        .pInheritanceInfo = nullptr,
    };
}

VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags) {
    return VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
    };
}

VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags) {
    return VkSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
    };
}

VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stage, VkSemaphore sem) {
    return VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = sem,
        .value = 1,
        .stageMask = stage,
        .deviceIndex = 0,
    };
}

VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd) {
    return VkCommandBufferSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmd,
        .deviceMask = 0,
    };
}

VkSubmitInfo2 submit_info(const VkCommandBufferSubmitInfo* cmd,
                          const VkSemaphoreSubmitInfo* signal,
                          const VkSemaphoreSubmitInfo* wait) {
    return VkSubmitInfo2{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = wait ? 1u : 0u,
        .pWaitSemaphoreInfos = wait,
        .commandBufferInfoCount = cmd ? 1u : 0u,
        .pCommandBufferInfos = cmd,
        .signalSemaphoreInfoCount = signal ? 1u : 0u,
        .pSignalSemaphoreInfos = signal,
    };
}

VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspect) {
    return VkImageSubresourceRange{
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
}

VkRenderingAttachmentInfo color_attachment_info(VkImageView view,
                                                const VkClearValue* clear,
                                                VkImageLayout layout) {
    VkRenderingAttachmentInfo info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = view,
        .imageLayout = layout,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear ? *clear : VkClearValue{},
    };
    return info;
}

VkRenderingInfo rendering_info(VkExtent2D extent,
                               const VkRenderingAttachmentInfo* color,
                               const VkRenderingAttachmentInfo* depth) {
    return VkRenderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderArea = VkRect2D{ {0, 0}, extent },
        .layerCount = 1,
        .viewMask = 0,
        .colorAttachmentCount = color ? 1u : 0u,
        .pColorAttachments = color,
        .pDepthAttachment = depth,
        .pStencilAttachment = nullptr,
    };
}

// Access mask for a target image layout. The Vulkan validation layer warns
// when a barrier passes generic MEMORY_READ/MEMORY_WRITE for a layout that
// has a more specific allowed access set — the warnings aren't bugs (the
// generic masks are spec-legal), but they crowd the log and hide real
// problems. Mapping each layout to its canonical mask kills the noise.
static VkAccessFlags2 access_for_layout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return 0;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_2_SHADER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;
        default:
            return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
}

// Pipeline-stage mask derived from a layout. Avoids the ALL_COMMANDS_BIT
// hammer that forces a full pipeline serialization on every transition.
// For layouts that genuinely could be touched by multiple stages we still
// fall back to ALL_GRAPHICS / ALL_COMMANDS — the perf wins come from the
// common cases (color out → shader read, transfer dst → shader read).
static VkPipelineStageFlags2 stage_for_layout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // Could be VS or FS; safe to cover both. Avoids ALL_COMMANDS
            // which would also fence ray queries / compute / transfer.
            return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_2_COPY_BIT |
                   VK_PIPELINE_STAGE_2_BLIT_BIT |
                   VK_PIPELINE_STAGE_2_RESOLVE_BIT;
        case VK_IMAGE_LAYOUT_GENERAL:
        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

void transition_image_aspect(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout from, VkImageLayout to,
                             VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = stage_for_layout(from),
        .srcAccessMask = access_for_layout(from),
        .dstStageMask = stage_for_layout(to),
        .dstAccessMask = access_for_layout(to),
        .oldLayout = from, .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = image_subresource_range(aspect),
    };
    VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr, .dependencyFlags = 0,
        .memoryBarrierCount = 0, .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0, .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageLayout from, VkImageLayout to) {
    VkImageAspectFlags aspect =
        (to == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL ||
         from == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
        ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    transition_image_aspect(cmd, image, from, to, aspect);
}

void transition_image_mip(VkCommandBuffer cmd, VkImage image,
                          VkImageLayout from, VkImageLayout to,
                          uint32_t base_mip, uint32_t mip_count) {
    VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = stage_for_layout(from),
        .srcAccessMask = access_for_layout(from),
        .dstStageMask = stage_for_layout(to),
        .dstAccessMask = access_for_layout(to),
        .oldLayout = from, .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = base_mip,
            .levelCount = mip_count,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr, .dependencyFlags = 0,
        .memoryBarrierCount = 0, .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0, .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

void transition_images_batch(VkCommandBuffer cmd,
                              const ImageTransition* entries,
                              uint32_t count) {
    if (count == 0) return;
    // Stack-allocate up to 8 barriers (current call sites use 2-3); fall
    // back to vector if a future caller exceeds. Hot path stays alloc-
    // free.
    constexpr uint32_t kStackMax = 8;
    VkImageMemoryBarrier2 stack_barriers[kStackMax];
    std::vector<VkImageMemoryBarrier2> heap_barriers;
    VkImageMemoryBarrier2* barriers = stack_barriers;
    if (count > kStackMax) {
        heap_barriers.resize(count);
        barriers = heap_barriers.data();
    }
    for (uint32_t i = 0; i < count; ++i) {
        const ImageTransition& e = entries[i];
        VkImageAspectFlags aspect = e.aspect;
        if (aspect == 0) {
            aspect = (e.to   == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL ||
                      e.from == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                     ? VK_IMAGE_ASPECT_DEPTH_BIT
                     : VK_IMAGE_ASPECT_COLOR_BIT;
        }
        barriers[i] = VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask  = stage_for_layout(e.from),
            .srcAccessMask = access_for_layout(e.from),
            .dstStageMask  = stage_for_layout(e.to),
            .dstAccessMask = access_for_layout(e.to),
            .oldLayout = e.from, .newLayout = e.to,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = e.image,
            .subresourceRange = image_subresource_range(aspect),
        };
    }
    VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr, .dependencyFlags = 0,
        .memoryBarrierCount = 0, .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0, .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = count, .pImageMemoryBarriers = barriers,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace qlike::vkinit
