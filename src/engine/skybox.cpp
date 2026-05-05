#include "engine/skybox.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"

// stb_image is vendored inside tinygltf-src; reuse it instead of pulling in a
// separate FetchContent. tinygltf was configured to not own stb_image, so we
// own the implementation in this TU.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR  // we ship the LDR JPG path; .hdr/.exr would need tinyexr
#include <stb_image.h>

// tinyexr — HDR equirect loader (.exr). Implementation lives in the upstream
// tinyexr CMake target (which also pulls in miniz); we just include the
// header here and call into the linked symbols.
#include <tinyexr.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cmath>
#include <cstring>

namespace qlike {

namespace {

void image_layout_transition(VkCommandBuffer cmd, VkImage img,
                             VkImageLayout from, VkImageLayout to,
                             VkPipelineStageFlags2 src_stage,
                             VkAccessFlags2 src_access,
                             VkPipelineStageFlags2 dst_stage,
                             VkAccessFlags2 dst_access) {
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
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

bool path_ends_with(const std::string& s, const char* suffix) {
    size_t n = std::strlen(suffix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        char a = s[s.size() - n + i];
        char b = suffix[i];
        // case-insensitive ASCII compare
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b) return false;
    }
    return true;
}

// Find the brightest pixel in the equirect panorama and convert its UV to a
// world direction. Equirect mapping (u,v) ∈ [0,1]² →
//   longitude (yaw) = (u - 0.5) · 2π
//   latitude  (pitch) = (0.5 - v) · π
//   dir = (cos(lat)·sin(lon), sin(lat), cos(lat)·cos(lon))
//
// We work on the loaded sRGB→linear-ish 8-bit data; that's fine because we
// only care about argmax, not absolute luminance.
glm::vec3 extract_sun_direction(const unsigned char* pixels, int w, int h, int comps) {
    int best_idx = 0;
    int best_lum = -1;
    // Sample every Nth pixel for speed. 4K image at stride 8 is still 1M
    // samples — plenty to find a sun disc that's tens of pixels wide.
    const int stride = std::max(1, w / 1024);
    for (int y = 0; y < h; y += stride) {
        // Skip the lower hemisphere — sun is always in the sky half. Speeds
        // things up and avoids picking up bright ground tiles.
        if (y > h * 6 / 10) break;
        for (int x = 0; x < w; x += stride) {
            int idx = (y * w + x) * comps;
            int r = pixels[idx + 0];
            int g = comps > 1 ? pixels[idx + 1] : r;
            int b = comps > 2 ? pixels[idx + 2] : r;
            // Crude luminance proxy biased to bright + warm pixels (the sun).
            int lum = r * 2 + g * 3 + b;
            if (lum > best_lum) {
                best_lum = lum;
                best_idx = y * w + x;
            }
        }
    }
    int by = best_idx / w;
    int bx = best_idx % w;
    float u = (static_cast<float>(bx) + 0.5f) / static_cast<float>(w);
    float v = (static_cast<float>(by) + 0.5f) / static_cast<float>(h);

    float lon = (u - 0.5f) * glm::two_pi<float>();
    float lat = (0.5f - v) * glm::pi<float>();

    glm::vec3 d(std::cos(lat) * std::sin(lon),
                std::sin(lat),
                std::cos(lat) * std::cos(lon));
    log::infof("[skybox] sun extracted at uv=(%.3f, %.3f) → dir=(%.2f, %.2f, %.2f)",
               u, v, d.x, d.y, d.z);
    return d;
}

// Same idea as the LDR sun extractor, but operates on linear float pixels —
// matters because EXR sun pixels can be 50,000+ nits, dwarfing the rest of
// the sky. argmax over linear luminance is correct here.
glm::vec3 extract_sun_direction_float(const float* pixels, int w, int h) {
    int best_idx = 0;
    float best_lum = -1.0f;
    const int stride = (w / 1024 > 1) ? (w / 1024) : 1;
    for (int y = 0; y < h; y += stride) {
        if (y > h * 6 / 10) break;  // sun is in the upper hemisphere
        for (int x = 0; x < w; x += stride) {
            int idx = (y * w + x) * 4;
            float r = pixels[idx + 0];
            float g = pixels[idx + 1];
            float b = pixels[idx + 2];
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (lum > best_lum) {
                best_lum = lum;
                best_idx = y * w + x;
            }
        }
    }
    int by = best_idx / w;
    int bx = best_idx % w;
    float u = (static_cast<float>(bx) + 0.5f) / static_cast<float>(w);
    float v = (static_cast<float>(by) + 0.5f) / static_cast<float>(h);
    float lon = (u - 0.5f) * glm::two_pi<float>();
    float lat = (0.5f - v) * glm::pi<float>();
    glm::vec3 d(std::cos(lat) * std::sin(lon),
                std::sin(lat),
                std::cos(lat) * std::cos(lon));
    log::infof("[skybox] HDR sun extracted at uv=(%.3f, %.3f) lum=%.2f → "
               "dir=(%.2f, %.2f, %.2f)",
               u, v, best_lum, d.x, d.y, d.z);
    return d;
}

} // namespace

// HDR (EXR) path. Returns ok=true on success.
struct HdrLoadResult {
    int w = 0, h = 0;
    float* rgba = nullptr;  // 4 floats per pixel; caller frees with free()
    bool ok = false;
};

static HdrLoadResult load_exr(const std::string& path) {
    HdrLoadResult r{};
    const char* err = nullptr;
    int code = LoadEXR(&r.rgba, &r.w, &r.h, path.c_str(), &err);
    if (code != TINYEXR_SUCCESS) {
        if (err) {
            log::errorf("[skybox] EXR load failed for %s: %s", path.c_str(), err);
            FreeEXRErrorMessage(err);
        }
        r.rgba = nullptr;
        return r;
    }
    r.ok = true;
    log::infof("[skybox] EXR loaded %s: %dx%d (RGBA32F)", path.c_str(), r.w, r.h);
    return r;
}

Skybox load_skybox(VkDevice device, VmaAllocator alloc, VkQueue queue,
                   uint32_t queue_family, const std::string& path) {
    Skybox sb{};

    // EXR path: preserves real sun radiance — the sun pixel keeps its
    // 10,000–50,000 nit brightness, so the bloom mip chain naturally turns
    // it into a glaring disc instead of needing a procedural overlay.
    const bool is_exr = path_ends_with(path, ".exr");
    if (is_exr) {
        HdrLoadResult hdr = load_exr(path);
        if (!hdr.ok) return sb;
        sb.width = static_cast<uint32_t>(hdr.w);
        sb.height = static_cast<uint32_t>(hdr.h);
        sb.sun_direction = extract_sun_direction_float(hdr.rgba, hdr.w, hdr.h);

        const VkDeviceSize bytes = static_cast<VkDeviceSize>(hdr.w) * hdr.h * 16;
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
                     "skybox-hdr staging");
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(alloc, stage_alloc, &info);
            std::memcpy(info.pMappedData, hdr.rgba, static_cast<size_t>(bytes));
        }
        free(hdr.rgba);

        const VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
        {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = fmt;
            ici.extent = { sb.width, sb.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateImage(alloc, &ici, &aci, &sb.image, &sb.alloc, nullptr),
                     "skybox-hdr image");
        }
        {
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = sb.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = fmt;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.baseMipLevel = 0; vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.baseArrayLayer = 0; vci.subresourceRange.layerCount = 1;
            vk_check(vkCreateImageView(device, &vci, nullptr, &sb.view),
                     "skybox-hdr view");
        }
        {
            VkSamplerCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            si.magFilter = VK_FILTER_LINEAR;
            si.minFilter = VK_FILTER_LINEAR;
            si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.maxLod = 0.0f;
            si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            vk_check(vkCreateSampler(device, &si, nullptr, &sb.sampler),
                     "skybox-hdr sampler");
        }

        // One-shot upload + transition.
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = queue_family,
        };
        vk_check(vkCreateCommandPool(device, &pci, nullptr, &pool), "skybox-hdr pool");
        VkCommandBufferAllocateInfo cbai{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr, .commandPool = pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
        };
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vk_check(vkAllocateCommandBuffers(device, &cbai, &cb), "skybox-hdr cb");
        VkCommandBufferBeginInfo cbi{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        vk_check(vkBeginCommandBuffer(cb, &cbi), "skybox-hdr begin");
        image_layout_transition(cb, sb.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { sb.width, sb.height, 1 };
        vkCmdCopyBufferToImage(cb, stage_buf, sb.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        image_layout_transition(cb, sb.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        vk_check(vkEndCommandBuffer(cb), "skybox-hdr end");
        VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = nullptr,
                         .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
                         .pWaitDstStageMask = nullptr,
                         .commandBufferCount = 1, .pCommandBuffers = &cb,
                         .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr };
        vk_check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "skybox-hdr submit");
        vk_check(vkQueueWaitIdle(queue), "skybox-hdr wait");
        vkDestroyCommandPool(device, pool, nullptr);
        vmaDestroyBuffer(alloc, stage_buf, stage_alloc);
        return sb;
    }

    int w = 0, h = 0, comps = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &comps, STBI_rgb_alpha);
    if (!pixels) {
        log::errorf("[skybox] failed to load %s: %s", path.c_str(),
                    stbi_failure_reason() ? stbi_failure_reason() : "(unknown)");
        return sb;
    }
    log::infof("[skybox] loaded %s: %dx%d, %d source comps",
               path.c_str(), w, h, comps);

    sb.width = static_cast<uint32_t>(w);
    sb.height = static_cast<uint32_t>(h);
    sb.sun_direction = extract_sun_direction(pixels, w, h, 4);

    const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;

    // Staging buffer (host visible) → device-local image.
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
                 "skybox staging");
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(alloc, stage_alloc, &info);
        std::memcpy(info.pMappedData, pixels, static_cast<size_t>(bytes));
    }
    stbi_image_free(pixels);

    {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_SRGB;
        ici.extent = { sb.width, sb.height, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(alloc, &ici, &aci, &sb.image, &sb.alloc, nullptr),
                 "skybox image");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = sb.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device, &vci, nullptr, &sb.view), "skybox view");
    }

    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        // Wrap U (longitude) repeats around the panorama; clamp V (latitude)
        // so we don't bleed the bottom row across the pole.
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        vk_check(vkCreateSampler(device, &si, nullptr, &sb.sampler), "skybox sampler");
    }

    // One-shot copy via a transient command buffer.
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    vk_check(vkCreateCommandPool(device, &pci, nullptr, &pool), "skybox pool");
    VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr, .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &cbai, &cb), "skybox cb");
    VkCommandBufferBeginInfo cbi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vk_check(vkBeginCommandBuffer(cb, &cbi), "skybox begin");

    image_layout_transition(cb, sb.image,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { sb.width, sb.height, 1 };
    vkCmdCopyBufferToImage(cb, stage_buf, sb.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    image_layout_transition(cb, sb.image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_2_SHADER_READ_BIT);

    vk_check(vkEndCommandBuffer(cb), "skybox end");
    VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = nullptr,
                     .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
                     .pWaitDstStageMask = nullptr,
                     .commandBufferCount = 1, .pCommandBuffers = &cb,
                     .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr };
    vk_check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "skybox submit");
    vk_check(vkQueueWaitIdle(queue), "skybox wait");
    vkDestroyCommandPool(device, pool, nullptr);
    vmaDestroyBuffer(alloc, stage_buf, stage_alloc);
    return sb;
}

void destroy_skybox(VkDevice device, VmaAllocator alloc, Skybox& sb) {
    if (sb.sampler) vkDestroySampler(device, sb.sampler, nullptr);
    if (sb.view)    vkDestroyImageView(device, sb.view, nullptr);
    if (sb.image)   vmaDestroyImage(alloc, sb.image, sb.alloc);
    sb = {};
}

} // namespace qlike
