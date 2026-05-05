#pragma once

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <string>

#include "engine/log.h"

namespace qlike {

// Shared error-check helper. Use anywhere a Vulkan call can fail. The throw
// site is logged BEFORE the throw so we still see the cause if the exception
// object's storage gets corrupted later (we've seen 0xDD freed-heap poison
// in `e.what()` at the outer catch). Use the FILE/LINE-aware macro form when
// possible so the logged line points at the caller, not at this header.
inline void vk_check_loc(VkResult r, const char* what,
                         const char* file, int line) {
    if (r != VK_SUCCESS) {
        qlike::log::errorf("[vk_check] %s failed: VkResult=%d at %s:%d",
                           what, static_cast<int>(r), file, line);
        throw std::runtime_error(std::string("vulkan error: ") + what +
                                 " (VkResult=" + std::to_string(static_cast<int>(r)) +
                                 ") at " + file + ":" + std::to_string(line));
    }
}

inline void vk_check(VkResult r, const char* what) {
    vk_check_loc(r, what, "?", 0);
}

} // namespace qlike

// Prefer this form at call sites — it captures the caller's location.
#define QLIKE_VK_CHECK(expr, what) ::qlike::vk_check_loc((expr), (what), __FILE__, __LINE__)

namespace qlike::vkinit {

VkCommandPoolCreateInfo command_pool_create_info(uint32_t queue_family,
                                                 VkCommandPoolCreateFlags flags = 0);
VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1);
VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags = 0);

VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0);
VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags = 0);

VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stage, VkSemaphore sem);
VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd);
VkSubmitInfo2 submit_info(const VkCommandBufferSubmitInfo* cmd,
                          const VkSemaphoreSubmitInfo* signal,
                          const VkSemaphoreSubmitInfo* wait);

VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspect);

VkRenderingAttachmentInfo color_attachment_info(VkImageView view,
                                                const VkClearValue* clear,
                                                VkImageLayout layout);
VkRenderingInfo rendering_info(VkExtent2D extent,
                               const VkRenderingAttachmentInfo* color,
                               const VkRenderingAttachmentInfo* depth);

void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageLayout from, VkImageLayout to);

// Explicit-aspect overload for depth/stencil images.
void transition_image_aspect(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout from, VkImageLayout to,
                             VkImageAspectFlags aspect);

// Per-mip-range overload — used by the bloom mip-chain so we don't transition
// every mip when only one is being read/written. base_mip + mip_count cover
// the affected subresource; aspect is COLOR (the bloom image is always color).
void transition_image_mip(VkCommandBuffer cmd, VkImage image,
                          VkImageLayout from, VkImageLayout to,
                          uint32_t base_mip, uint32_t mip_count);

// Run `body(cmd)` in a one-time-submit command buffer, then wait for the
// queue to idle before returning. Centralizes the transient-pool / allocate /
// begin / submit / wait / cleanup boilerplate that BLAS builds, texture
// uploads, and image-layout transitions all reproduced inline.
template <typename F>
void one_time_submit(VkDevice device, VkQueue queue, uint32_t queue_family,
                     F&& body) {
    VkCommandPoolCreateInfo pci = command_pool_create_info(
        queue_family, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(device, &pci, nullptr, &pool),
             "one_time_submit: pool");

    VkCommandBufferAllocateInfo cbai = command_buffer_allocate_info(pool, 1);
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &cbai, &cb),
             "one_time_submit: alloc");

    VkCommandBufferBeginInfo bi = command_buffer_begin_info(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vk_check(vkBeginCommandBuffer(cb, &bi), "one_time_submit: begin");

    body(cb);

    vk_check(vkEndCommandBuffer(cb), "one_time_submit: end");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vk_check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE),
             "one_time_submit: submit");
    vk_check(vkQueueWaitIdle(queue), "one_time_submit: wait");

    vkDestroyCommandPool(device, pool, nullptr);
}

} // namespace qlike::vkinit
