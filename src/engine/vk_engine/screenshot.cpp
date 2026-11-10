// Frame capture path. capture_screenshot() blits the current swapchain
// image into the readback_buffer_ (host-mapped); write_ppm() encodes that
// buffer as a P6 PPM file. Used by the --screenshot CLI flag and (later
// hopefully) any in-game screenshot key.

#include "engine/vk_engine/internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace qlike {

void VulkanEngine::capture_screenshot(VkCommandBuffer cmd, VkImage src,
                                      VkExtent2D extent, VkImageLayout pre_layout) {
    vkinit::transition_image(cmd, src, pre_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = { extent.width, extent.height, 1 };

    vkCmdCopyImageToBuffer(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback_buffer_, 1, &region);

    vkinit::transition_image(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pre_layout);
}

void VulkanEngine::write_ppm(const std::string& path, uint32_t w, uint32_t h) {
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(allocator_, readback_alloc_, &info);
    void* mapped = info.pMappedData;
    if (!mapped) {
        log::errorf("readback buffer has no mapped pointer");
        return;
    }
    const VkDeviceSize need = static_cast<VkDeviceSize>(w) * h * 4;
    if (need > readback_size_) {
        log::errorf("readback buffer too small: have %llu, need %llu",
                    static_cast<unsigned long long>(readback_size_),
                    static_cast<unsigned long long>(need));
        return;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        log::errorf("failed to open screenshot path: %s", path.c_str());
        return;
    }
    out << "P6\n" << w << " " << h << "\n255\n";

    const uint8_t* src = static_cast<const uint8_t*>(mapped);
    const float paper = std::max(80.0f, rt_.hdr_paper_nits);
    std::vector<uint8_t> row(w * 3);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const uint8_t* px = src + (y * w + x) * 4;
            if (hdr_active_) {
                // A2B10G10R10 PQ / BT.2020 -> nits -> paper-white-relative
                // BT.709 -> sRGB (highlights above paper white clip).
                uint32_t v; std::memcpy(&v, px, 4);
                float c[3] = { float(v & 1023u) / 1023.0f, float((v >> 10) & 1023u) / 1023.0f,
                               float((v >> 20) & 1023u) / 1023.0f };
                for (float& e : c) {
                    float p = std::pow(e, 1.0f / 78.84375f);
                    float n = std::max(p - 0.8359375f, 0.0f) / (18.8515625f - 18.6875f * p);
                    e = std::pow(n, 1.0f / 0.1593017578125f) * 10000.0f / paper;
                }
                float r =  1.6605f * c[0] - 0.5876f * c[1] - 0.0728f * c[2];
                float g = -0.1246f * c[0] + 1.1329f * c[1] - 0.0083f * c[2];
                float b = -0.0182f * c[0] - 0.1006f * c[1] + 1.1187f * c[2];
                float o[3] = { r, g, b };
                for (int k = 0; k < 3; ++k) {
                    float l = std::clamp(o[k], 0.0f, 1.0f);
                    float s = l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
                    row[x * 3 + k] = static_cast<uint8_t>(s * 255.0f + 0.5f);
                }
                continue;
            }
            row[x * 3 + 0] = px[2];  // R from BGRA's B-position
            row[x * 3 + 1] = px[1];
            row[x * 3 + 2] = px[0];
        }
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size()));
    }
    log::infof("screenshot written: %s (%ux%u)  brushes=%zu  dyn=%zu  proj=%zu  spark=%zu",
               path.c_str(), w, h,
               world_.brushes.size(), dyn_props_.size(),
               projectiles_.size(), particles_.size());
}

} // namespace qlike
