#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string>

namespace qlike {

// One uploaded 2D texture: device-local image + view + the VMA allocation
// that backs it. Sampler is shared across textures and owned elsewhere.
struct Texture2D {
    VkImage       image  = VK_NULL_HANDLE;
    VmaAllocation alloc  = nullptr;
    VkImageView   view   = VK_NULL_HANDLE;
    int           width  = 0;
    int           height = 0;
    bool          ok     = false;
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

// Upload an already-decoded, tightly-packed RGBA8 mip-0 buffer (w*h*4
// bytes) as a 2D texture with a GPU-generated mip chain. Same proven
// path as the file loader; used for procedurally-baked textures that
// never touch disk. `label` is for logging only.
Texture2D upload_texture_from_pixels(VkDevice device, VmaAllocator alloc,
                                     VkQueue queue, uint32_t queue_family,
                                     const unsigned char* pixels,
                                     int w, int h, VkFormat format,
                                     const char* label);

void destroy_texture_2d(VkDevice device, VmaAllocator alloc, Texture2D& t);

// ---- Streaming / parallel texture loading -----------------------------
// CPU-side decode result. Holds malloc()'d RGBA8 pixel buffer that must
// be freed via free() after the corresponding GPU upload. Width/height
// are zero on failure (file missing or stbi_load failed).
struct TextureCpuPixels {
    unsigned char* pixels = nullptr;   // malloc()'d, RGBA8 w*h*4 bytes
    int            width  = 0;
    int            height = 0;
    std::string    debug_path;         // for logging; copied from request
};

// Decode-only helper. Tries .qtc cache then stbi_load(). No GPU work —
// safe to call from any thread. Returns empty struct on failure. Caller
// must free `pixels` with std::free() after upload completes.
TextureCpuPixels decode_texture_pixels(const std::string& path);

// Upload an already-decoded buffer (from decode_texture_pixels). This
// MUST be called from the thread that owns the graphics queue (Vulkan
// queues are externally synchronised). Frees the input pixel buffer on
// successful upload OR on failure — the caller need not free it.
Texture2D upload_decoded_pixels(VkDevice device, VmaAllocator alloc,
                                 VkQueue queue, uint32_t queue_family,
                                 TextureCpuPixels&& pixels, VkFormat format);

} // namespace qlike
