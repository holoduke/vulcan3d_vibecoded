// Frames in flight: per-frame CPU -> GPU uploads through a per-slot staging
// ring. See the block comment on UploadRing in vk_engine.h.
//
// Why a staging copy instead of writing the persistently mapped buffers:
// with two frames overlapping, frame N-1 may still be reading (say) the
// scene UBO when the CPU prepares frame N. Writing the mapped memory would
// change the data under the running frame. Instead the CPU writes slot
// N % 2 of the ring (the GPU finished with that slot two frames ago - its
// fence was waited on), and the GPU copies it into the real buffer at the
// start of frame N, after frame N-1's commands in queue order.

#include "engine/vk_engine/internal.h"

#include <cstring>

namespace qlike {

namespace {
// Scene UBO (~6 KB) + pass UBOs + TLAS materials / prev transforms for up
// to (brushes + 1 + 8192) slots (~0.9 MB). Overflow falls back to a direct
// write with a one-time warning rather than corrupting anything.
constexpr VkDeviceSize kUploadRingBytes = 2u * 1024u * 1024u;
constexpr VkDeviceSize kUploadAlign = 16;
}

void VulkanEngine::init_upload_ring() {
    upload_ring_.capacity = kUploadRingBytes;
    for (uint32_t s = 0; s < kFrameOverlap; ++s) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = kUploadRingBytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &upload_ring_.buf[s],
                                 &upload_ring_.alloc[s], &ai),
                 "upload ring");
        upload_ring_.map[s] = static_cast<uint8_t*>(ai.pMappedData);
    }

    // Timeline semaphore: every graphics frame submit signals ++value; the
    // next frame's compute (TLAS rebuild) waits for it.
    VkSemaphoreTypeCreateInfo tci{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue = 0;
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &tci;
    vk_check(vkCreateSemaphore(device_, &sci, nullptr, &gfx_timeline_), "gfx timeline");
    gfx_timeline_value_ = 0;
}

void VulkanEngine::destroy_upload_ring() {
    for (uint32_t s = 0; s < kFrameOverlap; ++s) {
        if (upload_ring_.buf[s]) vmaDestroyBuffer(allocator_, upload_ring_.buf[s], upload_ring_.alloc[s]);
        upload_ring_.buf[s] = VK_NULL_HANDLE;
        upload_ring_.map[s] = nullptr;
    }
    if (gfx_timeline_) vkDestroySemaphore(device_, gfx_timeline_, nullptr);
    gfx_timeline_ = VK_NULL_HANDLE;
}

void VulkanEngine::begin_frame_uploads() {
    upload_ring_.used = 0;
    upload_ring_.copies.clear();
    upload_ring_.open = upload_ring_.buf[0] != VK_NULL_HANDLE;
    upload_ring_.in_frame = upload_ring_.open;
}

void* VulkanEngine::upload_alloc(VkBuffer dst, VkDeviceSize dst_offset, VkDeviceSize bytes,
                                 void* dst_mapped) {
    if (bytes == 0) return nullptr;
    UploadRing& r = upload_ring_;
    const VkDeviceSize at = (r.used + kUploadAlign - 1) & ~(kUploadAlign - 1);
    if (!r.open && r.in_frame) {
        // A write after the batch was flushed would land under the
        // previous, still-running frame: every per-frame upload must be
        // staged before flush_frame_uploads().
        static bool warned = false;
        if (!warned) { warned = true; log::warnf("[upload] write after flush (dst %p, %llu B) - direct, unsynchronised", (void*)dst, (unsigned long long)bytes); }
    }
    if (!r.open || at + bytes > r.capacity) {
        if (r.open) {
            static bool warned = false;
            if (!warned) { warned = true; log::warnf("[upload] ring full (%llu B) - direct write", (unsigned long long)r.capacity); }
        }
        return dst_mapped ? static_cast<uint8_t*>(dst_mapped) + dst_offset : nullptr;
    }
    const uint32_t slot = uint32_t(frame_number_ % kFrameOverlap);
    r.used = at + bytes;
    r.copies.push_back({ dst, VkBufferCopy{ at, dst_offset, bytes } });
    return r.map[slot] + at;
}

void VulkanEngine::upload(VkBuffer dst, VkDeviceSize dst_offset, const void* src, VkDeviceSize bytes,
                          void* dst_mapped) {
    if (void* p = upload_alloc(dst, dst_offset, bytes, dst_mapped)) std::memcpy(p, src, size_t(bytes));
}

void VulkanEngine::flush_frame_uploads(VkCommandBuffer cmd) {
    UploadRing& r = upload_ring_;
    r.open = false;
    const uint32_t slot = uint32_t(frame_number_ % kFrameOverlap);

    // Frame-start barrier, ALWAYS emitted (not only when there are copies):
    // besides ordering the upload copies after the previous frame's reads
    // (WAR/WAW), it is what orders this frame's UNDEFINED -> attachment
    // discards of the shared render targets (scene colour, G-buffer,
    // depth, ...; recorded with TOP_OF_PIPE sources) after the previous,
    // possibly still running, frame finished sampling them.
    VkMemoryBarrier2 pre{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    pre.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    pre.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    pre.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;   // full: chains to TOP_OF_PIPE barriers too
    pre.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &pre;
    vkCmdPipelineBarrier2(cmd, &di);
    if (r.copies.empty()) return;
    vmaFlushAllocation(allocator_, r.alloc[slot], 0, r.used);

    // One vkCmdCopyBuffer per destination (regions are in staging order).
    static std::vector<VkBufferCopy> regions;   // reused (render thread only)
    for (size_t i = 0; i < r.copies.size();) {
        const VkBuffer dst = r.copies[i].dst;
        regions.clear();
        size_t j = i;
        for (; j < r.copies.size() && r.copies[j].dst == dst; ++j) regions.push_back(r.copies[j].region);
        vkCmdCopyBuffer(cmd, r.buf[slot], dst, uint32_t(regions.size()), regions.data());
        i = j;
    }

    VkMemoryBarrier2 post{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    post.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    post.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    post.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    di.pMemoryBarriers = &post;
    vkCmdPipelineBarrier2(cmd, &di);
    r.copies.clear();
}

} // namespace qlike

namespace qlike {

// Rare runtime descriptor rewrites (resize, toggles, a reallocated TLAS)
// must not touch a set the in-flight frame still uses
// (VUID-vkUpdateDescriptorSets-None-03047): wait for every submitted
// graphics frame first. Cheap when nothing is pending.
void VulkanEngine::wait_gpu_frames_idle() {
    if (!gfx_timeline_ || gfx_timeline_value_ == 0) return;
    VkSemaphoreWaitInfo wi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    wi.semaphoreCount = 1;
    wi.pSemaphores = &gfx_timeline_;
    wi.pValues = &gfx_timeline_value_;
    vk_check(vkWaitSemaphores(device_, &wi, UINT64_MAX), "wait gpu frames");
}

} // namespace qlike

namespace qlike {

// Frames in flight, TLAS side: this frame's TLAS was built into
// tlas_stage_[slot] on the compute queue while the previous frame was
// still running. Clone it into the bound tlas_ once the previous frame's
// ray queries are done (queue order + the WAR barrier). All descriptor sets
// keep pointing at tlas_. The clone of an 8K-instance TLAS is a few MB of
// device copy - microseconds.
void VulkanEngine::record_tlas_publish(VkCommandBuffer cmd) {
    if (!tlas_publish_pending_ || tlas_last_built_slot_ < 0 || !tlas_) return;
    tlas_publish_pending_ = false;
    const VkPipelineStageFlags2 readers = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mb.srcStageMask = readers;
    mb.srcAccessMask = 0;                                   // WAR: execution dependency
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                       VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);

    VkCopyAccelerationStructureInfoKHR ci{ VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
    ci.src = tlas_stage_[tlas_last_built_slot_];
    ci.dst = tlas_;
    ci.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
    g_rt.cmd_copy_as(cmd, &ci);

    mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    mb.dstStageMask = readers;
    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier2(cmd, &di);
}

} // namespace qlike
