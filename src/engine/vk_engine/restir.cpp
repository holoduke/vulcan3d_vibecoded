// ReSTIR GI — reservoir SSBO foundation (session 2 of multi-session
// plan in docs/restir_plan.md).
//
// Each pixel owns a Reservoir (48 B std430). The reservoir holds the
// current "best" GI sample direction + its radiance, the WRS weight
// (W), the running sum (w_sum), the sample count (M), and (session 5)
// the writing surface's camera distance packed into `pad` for
// depth-aware disocclusion.
//
// Session 5 ping-pong model — RING, NOT DESCRIPTOR-SWAP:
// One physical SSBO holds THREE per-pixel reservoir regions. cube.frag
// indexes region (frame % 3) for the current write and region
// ((frame + 2) % 3) for the previous-frame read, deriving both from
// scene.rt_flags.w. Bindings 15/16 still alias this one buffer (no
// per-frame descriptor sets, no UPDATE_AFTER_BIND) — the offset, not
// the descriptor, selects prev vs cur.
//
// Why 3 regions and why it's race-free under kFrameOverlap == 2:
// frame F writes region F%3. That region was last touched by frame
// F-3 (its own previous write) and last read by frame F-2 (as F-2's
// prev region, since (F-2+2)%3 == F%3). Before the CPU records frame
// F it waits on frame F-2's fence, so all of F-2's (and earlier)
// GPU work is fully retired — F's write cannot race any in-flight
// read/write of that region. Two regions would NOT be safe (F's
// write region would equal F-1's, still in flight).

#include "engine/vk_engine/internal.h"

namespace qlike {

namespace {
// Must match the GLSL Reservoir struct in cube.frag (std430).
//   layout(std430) struct Reservoir {
//       vec3  sample_dir;   // unit vector, hemisphere of N
//       vec3  radiance;     // path-traced contribution along sample_dir
//       float W;            // unbiased contribution weight
//       float w_sum;        // WRS running sum (for future combines)
//       uint  M;            // sample count
//       uint  pad;          // session 5: floatBitsToUint(cam_dist) of
//                           //   the writing surface — depth disoccl.
//   };
constexpr VkDeviceSize kReservoirBytes = 48;
// Triple-buffered ring (see file header) — one SSBO, 3 per-pixel
// regions, frame%3 selects cur / (frame+2)%3 selects prev.
constexpr uint32_t kReservoirRing = 3;

VkDeviceSize reservoir_buffer_bytes(VkExtent2D extent) {
    // Defensive cap: the buffer can't exceed Vulkan's 4 GB SSBO limit.
    // At 4K (3840×2160) × 48 B = ~395 MB, well under. Below 256x256
    // we still allocate a useful min so the early-init descriptor
    // write doesn't take a NULL buffer.
    uint32_t w = std::max<uint32_t>(extent.width,  256);
    uint32_t h = std::max<uint32_t>(extent.height, 256);
    return VkDeviceSize(w) * VkDeviceSize(h) *
           kReservoirBytes * VkDeviceSize(kReservoirRing);
}
} // namespace

void VulkanEngine::init_restir() {
    // Called from init() after init_descriptors but before
    // write_scene_descriptors_once so the desc write captures the
    // live buffer handles. Idempotent — safe to call again from
    // recreate_restir_buffers after destroying the old pair.
    if (reservoir_buf_[0]) return;
    const VkDeviceSize bytes = reservoir_buffer_bytes(render_extent_);
    // Session 5: one physical buffer, 3 per-pixel ring regions (see
    // file header). Both scene_desc bindings 15 (prev) and 16 (cur)
    // alias buf[0]; cube.frag offsets into the right region by
    // frame%3, so the descriptor never has to change. buf[1] stays
    // unused.
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                             &reservoir_buf_[0], &reservoir_alloc_[0],
                             nullptr),
             "reservoir buffer");
    // Zero-init — random garbage in `M` would make the temporal-combine
    // math read a stale "Mâ‰«0" value on the first frame and lock the
    // accumulation to noise.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkCmdFillBuffer(cb, reservoir_buf_[0], 0, bytes, 0u);
    });
    reservoir_write_slot_ = 0;
    log::infof("[restir] reservoir SSBO allocated: %llu MB (%ux%u px)",
               (unsigned long long)(bytes / (1024 * 1024)),
               render_extent_.width, render_extent_.height);
}

void VulkanEngine::destroy_restir() {
    for (int s = 0; s < 2; ++s) {
        if (reservoir_buf_[s]) {
            vmaDestroyBuffer(allocator_, reservoir_buf_[s], reservoir_alloc_[s]);
            reservoir_buf_[s]   = VK_NULL_HANDLE;
            reservoir_alloc_[s] = nullptr;
        }
    }
}

void VulkanEngine::recreate_restir_buffers() {
    // Called from recreate_swapchain after render_extent_ has changed.
    // Wait-idle is paid by the caller (recreate_swapchain wait-idles).
    // The destroy + init alloc-pair leaves scene_desc bindings 15/16
    // dangling at the old (destroyed) buffers, so we rewrite just
    // those two bindings inline here.
    destroy_restir();
    init_restir();
    if (!scene_desc_set_) return;
    // Both bindings alias to buf[0] for session 3 (see init_restir).
    VkDescriptorBufferInfo res_prev_bi{ reservoir_buf_[0], 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo res_cur_bi { reservoir_buf_[0], 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[2]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = scene_desc_set_; w[0].dstBinding = 15;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = &res_prev_bi;
    w[1] = w[0];
    w[1].dstBinding = 16;
    w[1].pBufferInfo = &res_cur_bi;
    vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
}

} // namespace qlike
