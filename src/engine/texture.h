#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string>

namespace qlike {

// One uploaded 2D texture: device-local image + view + the VMA allocation
// that backs it. Sampler is shared across textures and owned elsewhere.
struct Texture2D {
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView   view  = VK_NULL_HANDLE;
    bool          ok    = false;
};

// Load + upload an image file (JPG/PNG via stb_image) as a 2D Vulkan
// texture. Returns ok=false if the file is missing or can't be decoded —
// callers using a probe-paths pattern should check `ok` and try the next
// path. The implementation owns its own staging buffer and one-shot upload
// command; it waits queue idle before returning.
//
// `format` controls how the GPU samples the data: pass VK_FORMAT_R8G8B8A8_SRGB
// for albedo (sampler does the gamma decode for free), R8G8B8A8_UNORM for
// data-textures like normal/roughness maps.
Texture2D upload_texture_from_file(VkDevice device, VmaAllocator alloc,
                                   VkQueue queue, uint32_t queue_family,
                                   const std::string& path, VkFormat format);

void destroy_texture_2d(VkDevice device, VmaAllocator alloc, Texture2D& t);

} // namespace qlike
