#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/vec3.hpp>

#include <cstdint>
#include <string>

namespace qlike {

struct Skybox {
    VkImage       image  = VK_NULL_HANDLE;
    VmaAllocation alloc  = nullptr;
    VkImageView   view   = VK_NULL_HANDLE;
    VkSampler     sampler = VK_NULL_HANDLE;
    uint32_t      width = 0;
    uint32_t      height = 0;
    // World-space "toward sun" direction extracted from the brightest pixel of
    // the loaded image. Use this to drive the directional light so shadows in
    // the scene match the visible sun in the sky.
    glm::vec3     sun_direction{ 0.0f, 1.0f, 0.0f };
};

// Load an LDR equirectangular sky panorama (JPG/PNG). The image is uploaded
// as an RGBA8 sRGB Vulkan texture; the sampler does the sRGB→linear decode
// for us. Caller owns the resources and must call destroy_skybox() to free.
Skybox load_skybox(VkDevice device, VmaAllocator alloc, VkQueue queue,
                   uint32_t queue_family, const std::string& path);

void destroy_skybox(VkDevice device, VmaAllocator alloc, Skybox& sb);

} // namespace qlike
