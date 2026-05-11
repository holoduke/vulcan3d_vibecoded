#include "engine/texture.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"

// stb_image declarations — the implementation is owned by skybox.cpp's TU.
#include <stb_image.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace qlike {

namespace {

// On-disk cache for decoded textures. Skips the JPG decode (the
// dominant cost for the engine's 8K source textures) on every run
// after the first. Layout:
//   header (32 bytes, see CacheHeader)
//   width × height × 4 bytes of RGBA8 pixels
// Cache is stored as `<source_path>.qtc` next to the JPG. Staleness
// detected via stored source-file mtime; if the JPG is newer than what
// the cache was baked from we re-decode + overwrite.
struct CacheHeader {
    char     magic[4];        // 'Q','T','C','1'
    uint32_t width;
    uint32_t height;
    uint32_t format_tag;      // 1 = RGBA8 (only one we cache)
    uint64_t source_mtime;    // file_time_type.time_since_epoch().count()
    uint64_t reserved;
};
static_assert(sizeof(CacheHeader) == 32, "qtc header layout");

constexpr uint32_t kCacheRGBA8 = 1u;

uint64_t file_mtime_count(const std::string& path) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec) return 0;
    return static_cast<uint64_t>(t.time_since_epoch().count());
}

// Try to read a .qtc cache file. On success, fills `out_w`, `out_h`
// and returns the malloc()'d RGBA8 pixel buffer (caller frees with
// free()). On miss / stale / corrupt, returns nullptr.
unsigned char* try_load_cache(const std::string& source_path,
                               int* out_w, int* out_h) {
    const std::string cache_path = source_path + ".qtc";
    std::error_code ec;
    if (!std::filesystem::exists(cache_path, ec)) return nullptr;
    uint64_t src_mtime = file_mtime_count(source_path);
    if (src_mtime == 0) return nullptr;     // can't validate → re-decode

    std::ifstream f(cache_path, std::ios::binary);
    if (!f.is_open()) return nullptr;
    CacheHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f.good()) return nullptr;
    if (std::memcmp(hdr.magic, "QTC1", 4) != 0)   return nullptr;
    if (hdr.format_tag != kCacheRGBA8)            return nullptr;
    if (hdr.source_mtime != src_mtime)            return nullptr;
    if (hdr.width == 0 || hdr.height == 0)        return nullptr;

    const size_t bytes = static_cast<size_t>(hdr.width) * hdr.height * 4;
    auto* px = static_cast<unsigned char*>(std::malloc(bytes));
    if (!px) return nullptr;
    f.read(reinterpret_cast<char*>(px), static_cast<std::streamsize>(bytes));
    if (!f.good()) { std::free(px); return nullptr; }

    *out_w = static_cast<int>(hdr.width);
    *out_h = static_cast<int>(hdr.height);
    return px;
}

// Save a freshly-decoded RGBA8 buffer to the .qtc cache so subsequent
// runs skip the JPG decode. Best-effort: failures are logged but don't
// break the engine — we still have the in-memory pixels for this run.
void save_cache(const std::string& source_path,
                const unsigned char* pixels, int w, int h) {
    const std::string cache_path = source_path + ".qtc";
    std::ofstream f(cache_path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        log::warnf("[texture] could not write cache %s", cache_path.c_str());
        return;
    }
    CacheHeader hdr{};
    std::memcpy(hdr.magic, "QTC1", 4);
    hdr.width  = static_cast<uint32_t>(w);
    hdr.height = static_cast<uint32_t>(h);
    hdr.format_tag = kCacheRGBA8;
    hdr.source_mtime = file_mtime_count(source_path);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    const size_t bytes = static_cast<size_t>(w) * h * 4;
    f.write(reinterpret_cast<const char*>(pixels),
            static_cast<std::streamsize>(bytes));
}

void layout_transition(VkCommandBuffer cb, VkImage img,
                       VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                       VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                       uint32_t base_mip = 0, uint32_t mip_count = 1) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseMipLevel = base_mip;
    b.subresourceRange.levelCount = mip_count;
    b.subresourceRange.baseArrayLayer = 0; b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &dep);
}

// floor(log2(max(w,h))) + 1 — number of mips down to 1×1.
uint32_t mip_levels_for(int w, int h) {
    uint32_t m = 1;
    int dim = (w > h ? w : h);
    while (dim > 1) { dim >>= 1; ++m; }
    return m;
}

} // namespace

Texture2D upload_texture_from_file(VkDevice device, VmaAllocator alloc,
                                   VkQueue queue, uint32_t qf,
                                   const std::string& path, VkFormat format) {
    Texture2D r{};
    int w = 0, h = 0;
    bool from_cache = false;
    // Fast path: try the .qtc cache. Falls back to JPG decode on miss
    // / stale / corrupt. The cache stores raw RGBA8 mip-0 pixels;
    // mip generation still happens on the GPU below — much faster than
    // baking mips on the CPU.
    unsigned char* pixels = try_load_cache(path, &w, &h);
    if (pixels) {
        from_cache = true;
    } else {
        int comps = 0;
        pixels = stbi_load(path.c_str(), &w, &h, &comps, 4);
        if (!pixels) {
            // Caller probes multiple paths — only worth logging at the call site
            // when every probe fails.
            return r;
        }
        // Best-effort cache write so the next launch skips the decode.
        save_cache(path, pixels, w, h);
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;

    VkBuffer stage_buf = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(alloc, &bci, &aci, &stage_buf, &stage_alloc, nullptr),
                 "tex staging");
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(alloc, stage_alloc, &info);
        std::memcpy(info.pMappedData, pixels, static_cast<size_t>(bytes));
    }
    stbi_image_free(pixels);

    const uint32_t mip_count = mip_levels_for(w, h);

    {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
        ici.mipLevels = mip_count; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_SRC needed for the blit chain that generates mip levels.
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(alloc, &ici, &aci, &r.image, &r.alloc, nullptr),
                 "tex image");
    }
    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = r.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = mip_count;
        vci.subresourceRange.baseArrayLayer = 0; vci.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device, &vci, nullptr, &r.view), "tex view");
    }

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = qf,
    };
    vk_check(vkCreateCommandPool(device, &pci, nullptr, &pool), "tex pool");
    VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr, .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &cbai, &cb), "tex cb");
    VkCommandBufferBeginInfo cbi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vk_check(vkBeginCommandBuffer(cb, &cbi), "tex begin");

    // Upload mip 0 (the loaded pixels). All other mips stay UNDEFINED until
    // the blit chain below populates them.
    layout_transition(cb, r.image,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      0, 1);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    vkCmdCopyBufferToImage(cb, stage_buf, r.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Mip chain via vkCmdBlitImage with linear filter. Each level reads from
    // the (now TRANSFER_SRC) parent and writes a half-sized version.
    int32_t mip_w = w, mip_h = h;
    for (uint32_t lvl = 1; lvl < mip_count; ++lvl) {
        // Parent (lvl-1): TRANSFER_DST → TRANSFER_SRC.
        // Src-stage depends on what wrote the parent: lvl 0 was written by
        // vkCmdCopyBufferToImage (COPY_BIT), every other level was written by
        // vkCmdBlitImage (BLIT_BIT). Picking the wrong one is the
        // WRITE_AFTER_WRITE hazard sync validation flags.
        const VkPipelineStageFlags2 parent_writer_stage =
            (lvl == 1) ? VK_PIPELINE_STAGE_2_COPY_BIT
                       : VK_PIPELINE_STAGE_2_BLIT_BIT;
        layout_transition(cb, r.image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          parent_writer_stage, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                          lvl - 1, 1);
        // Child (lvl): UNDEFINED → TRANSFER_DST.
        layout_transition(cb, r.image,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                          VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          lvl, 1);

        int32_t child_w = (mip_w / 2 > 1) ? (mip_w / 2) : 1;
        int32_t child_h = (mip_h / 2 > 1) ? (mip_h / 2) : 1;
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = lvl - 1;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = { mip_w, mip_h, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = lvl;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = { child_w, child_h, 1 };
        vkCmdBlitImage(cb, r.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       r.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        mip_w = child_w;
        mip_h = child_h;
    }

    // Move the parent levels (TRANSFER_SRC, last read by BLIT) and the last
    // child (TRANSFER_DST) into SHADER_READ_ONLY. Empty parent range when
    // mip_count==1 (single mip written by COPY) is skipped — Vulkan spec
    // forbids levelCount=0.
    if (mip_count > 1) {
        layout_transition(cb, r.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT,
                          0, mip_count - 1);
    }
    // Last mip's writer is BLIT_BIT (mip_count > 1) or COPY_BIT (mip_count == 1).
    const VkPipelineStageFlags2 last_writer =
        (mip_count == 1) ? VK_PIPELINE_STAGE_2_COPY_BIT
                         : VK_PIPELINE_STAGE_2_BLIT_BIT;
    layout_transition(cb, r.image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      last_writer, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_READ_BIT,
                      mip_count - 1, 1);

    vk_check(vkEndCommandBuffer(cb), "tex end");
    VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = nullptr,
                     .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
                     .pWaitDstStageMask = nullptr,
                     .commandBufferCount = 1, .pCommandBuffers = &cb,
                     .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr };
    vk_check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "tex submit");
    vk_check(vkQueueWaitIdle(queue), "tex wait");
    vkDestroyCommandPool(device, pool, nullptr);
    vmaDestroyBuffer(alloc, stage_buf, stage_alloc);

    log::infof("[texture] uploaded %s: %dx%d", path.c_str(), w, h);
    r.ok = true;
    return r;
}

void destroy_texture_2d(VkDevice device, VmaAllocator alloc, Texture2D& t) {
    if (t.view)  vkDestroyImageView(device, t.view, nullptr);
    if (t.image) vmaDestroyImage(alloc, t.image, t.alloc);
    t = {};
}

} // namespace qlike
