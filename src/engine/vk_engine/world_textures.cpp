// World texture array, HDR skybox, procedural terrain-material baking, and
// the FSR2 sampler swap. Owns the disk->GPU decode path (resolve_tex_path,
// the kWorldTexSpec table) and the photogrammetry PBR splat overrides.
// Split out of the old monolithic world.cpp (maintainability pass).

#include "engine/vk_engine/internal.h"
#include "engine/gltf_loader.h"
#include "engine/grass.h"
#include "engine/skybox.h"
#include "engine/terrain.h"
#include "engine/texture.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <utility>
#include <vector>

// stb_image raw loader (implementation lives in skybox.cpp's TU) - used to
// AO-multiply the stylized-grass albedo before upload.
extern "C" {
    unsigned char* stbi_load(const char*, int*, int*, int*, int);
    void           stbi_image_free(void*);
}

namespace qlike {

void VulkanEngine::init_skybox() {
    // EXR first — keeps real sun radiance so the bloom pass naturally produces
    // a glaring sun disc. Falls back to the LDR JPG if EXR is missing or
    // tinyexr can't decode it. Engine is invoked from build/ on Windows but
    // from the project root in CI, so probe both.
    static const char* kSkyboxCandidates[] = {
        "assets/sky/MorningSkyHDRI002B_4K_HDR.exr",
        "../assets/sky/MorningSkyHDRI002B_4K_HDR.exr",
        "../../assets/sky/MorningSkyHDRI002B_4K_HDR.exr",
        "assets/sky/MorningSkyHDRI002B_4K_TONEMAPPED.jpg",
        "../assets/sky/MorningSkyHDRI002B_4K_TONEMAPPED.jpg",
        "../../assets/sky/MorningSkyHDRI002B_4K_TONEMAPPED.jpg",
    };
    for (const char* path : kSkyboxCandidates) {
        skybox_ = load_skybox(device_, allocator_, graphics_queue_,
                              graphics_queue_family_, path);
        if (skybox_.image != VK_NULL_HANDLE) break;
    }
    if (skybox_.image == VK_NULL_HANDLE) {
        log::info("[skybox] no asset loaded — creating 1x1 placeholder so the "
                  "compose descriptor stays bound");
        // Build a 1x1 black image manually (cheap fallback).
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_SRGB;
        ici.extent = { 1, 1, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci,
                                 &skybox_.image, &skybox_.alloc, nullptr),
                 "skybox dummy image");

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = skybox_.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0; vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0; vci.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device_, &vci, nullptr, &skybox_.view),
                 "skybox dummy view");

        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &skybox_.sampler),
                 "skybox dummy sampler");

        skybox_.width = 1; skybox_.height = 1;
        skybox_.sun_direction = glm::vec3(0.45f, 0.7f, 0.55f);

        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, skybox_.image,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    } else {
        // Real asset loaded — sync the directional-light angles to the
        // brightest pixel in the panorama so cast shadows match the visible
        // sun.
        glm::vec3 d = glm::normalize(skybox_.sun_direction);
        float pitch_deg = glm::degrees(std::asin(glm::clamp(d.y, -1.0f, 1.0f)));
        float yaw_deg = glm::degrees(std::atan2(d.x, d.z));
        rt_.sun_pitch_deg = pitch_deg;
        rt_.sun_yaw_deg = yaw_deg;
        log::infof("[skybox] sun synced to UI: pitch=%.1f° yaw=%.1f°",
                   pitch_deg, yaw_deg);
    }
}

void VulkanEngine::destroy_skybox_resources() {
    destroy_skybox(device_, allocator_, skybox_);
}

// Resolve "assets/tex/<tail>" trying the three roots the engine
// supports (build/, ../, ../../). Shared between the early-kickoff
// decode path and the in-init upload path so the two see the same
// resolved files.
static std::string resolve_tex_path(const std::string& tail) {
    const std::string roots[] = {
        "assets/tex/" + tail,
        "../assets/tex/" + tail,
        "../../assets/tex/" + tail,
    };
    for (const std::string& p : roots) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return std::string();
}

// Canonical list of the file-loaded world textures. Kept here so the
// early-kickoff decode and the in-init upload walk the same indices.
namespace {
struct WorldTexSpec {
    const char* color_jpg;
    const char* normal_jpg;
};
const WorldTexSpec kWorldTexSpecs[] = {
    { "Ground054/Ground054_2K-JPG_Color.jpg",
      "Ground054/Ground054_2K-JPG_NormalGL.jpg" },
    { "Bricks078/Bricks078_8K-JPG_Color.jpg",
      "Bricks078/Bricks078_8K-JPG_NormalGL.jpg" },
    { "Wood048/Wood048_2K-JPG_Color.jpg",
      "Wood048/Wood048_2K-JPG_NormalGL.jpg" },
    { "Metal042A/Metal042A_2K-JPG_Color.jpg",
      "Metal042A/Metal042A_2K-JPG_NormalGL.jpg" },
    { "PaintedBricks001/PaintedBricks001_8K-JPG_Color.jpg",
      "PaintedBricks001/PaintedBricks001_8K-JPG_NormalGL.jpg" },
    { "Tiles130/Tiles130_8K-JPG_Color.jpg",
      "Tiles130/Tiles130_8K-JPG_NormalGL.jpg" },
    { "Tiles074/Tiles074_8K-JPG_Color.jpg",
      "Tiles074/Tiles074_8K-JPG_NormalGL.jpg" },
};
const char* kWorldSpomPaths[] = {
    "Bricks078/Bricks078_8K-JPG_Displacement.jpg",
    "PaintedBricks001/PaintedBricks001_8K-JPG_Displacement.jpg",
    "Tiles130/Tiles130_8K-JPG_Displacement.jpg",
    "Tiles074/Tiles074_8K-JPG_Displacement.jpg",
};
}  // namespace

void VulkanEngine::start_world_texture_decodes() {
    // The 4K HDR sky decode (~0.4 s) overlaps world generation too.
    skybox_prefetch("assets/sky/MorningSkyHDRI002B_4K_HDR.exr");
    // Resolve paths NOW on the main thread (filesystem access is fast)
    // so the worker threads only do JPG decode. paths[i] empty => skip.
    std::vector<std::string> paths;
    paths.reserve(VulkanEngine::kFileTextureCount * 2);
    for (int i = 0; i < VulkanEngine::kFileTextureCount; ++i) {
        paths.push_back(resolve_tex_path(kWorldTexSpecs[i].color_jpg));
        paths.push_back(resolve_tex_path(kWorldTexSpecs[i].normal_jpg));
    }
    world_tex_decode_handle_ = start_background_decode(paths);

    std::vector<std::string> spom_paths;
    spom_paths.reserve(4);
    for (int i = 0; i < 4; ++i) {
        spom_paths.push_back(resolve_tex_path(kWorldSpomPaths[i]));
    }
    world_spom_decode_handle_ = start_background_decode(spom_paths);
}

void VulkanEngine::init_textures() {
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.anisotropyEnable = VK_FALSE;
        // VK_LOD_CLAMP_NONE lets the sampler pick from the entire mip chain
        // texture.cpp generates — eliminates far-distance moiré and halves
        // average sample bandwidth on receding floors/walls.
        si.maxLod = VK_LOD_CLAMP_NONE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &texture_sampler_),
                 "texture sampler");
    }

    auto resolve_path = [](const std::string& tail) -> std::string {
        const std::string roots[] = {
            "assets/tex/" + tail,
            "../assets/tex/" + tail,
            "../../assets/tex/" + tail,
        };
        for (const std::string& p : roots) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    auto probe = [&](const std::string& tail, VkFormat fmt) -> TextureSlot {
        std::string p = resolve_path(tail);
        if (!p.empty()) {
            Texture2D r = upload_texture_from_file(
                device_, allocator_, graphics_queue_,
                graphics_queue_family_, p, fmt);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                return s;
            }
        }
        log::warnf("[texture] %s missing on every probe path", tail.c_str());
        return {};
    };
    // Parallel-decode helper: kicks off N JPG decodes on worker threads,
    // then serialises the GPU uploads on the calling (graphics-queue)
    // thread. Wall-clock for init drops from sum(decode+upload) to
    // max(decode) + sum(upload) — ~5× on 8K JPGs with 8 CPU cores.
    auto probe_parallel = [&](const std::vector<std::pair<std::string, VkFormat>>& items,
                              std::vector<TextureSlot>& out) {
        out.assign(items.size(), {});
        std::vector<std::string> paths(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            paths[i] = resolve_path(items[i].first);
        }
        std::vector<std::future<TextureCpuPixels>> futs(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            if (paths[i].empty()) continue;
            futs[i] = std::async(std::launch::async,
                                  decode_texture_pixels, paths[i]);
        }
        for (size_t i = 0; i < items.size(); ++i) {
            if (paths[i].empty()) {
                log::warnf("[texture] %s missing on every probe path",
                            items[i].first.c_str());
                continue;
            }
            TextureCpuPixels cpu = futs[i].get();
            if (!cpu.pixels) {
                log::warnf("[texture] %s decode failed",
                            items[i].first.c_str());
                continue;
            }
            Texture2D r = upload_decoded_pixels(device_, allocator_,
                                                  graphics_queue_,
                                                  graphics_queue_family_,
                                                  std::move(cpu),
                                                  items[i].second);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                out[i] = s;
            }
        }
    };

    struct Spec {
        const char* name;
        const char* color_jpg;
        const char* normal_jpg;
    };
    // Index → category. Indices ALSO match what level.cpp / spawn_random_box
    // pass into Brush::tex_albedo, so reordering here means reordering there.
    const Spec specs[] = {
        { "Ground054",         "Ground054/Ground054_2K-JPG_Color.jpg",
                               "Ground054/Ground054_2K-JPG_NormalGL.jpg" },
        { "Bricks078",         "Bricks078/Bricks078_8K-JPG_Color.jpg",
                               "Bricks078/Bricks078_8K-JPG_NormalGL.jpg" },
        { "Wood048",           "Wood048/Wood048_2K-JPG_Color.jpg",
                               "Wood048/Wood048_2K-JPG_NormalGL.jpg" },
        { "Metal042A",         "Metal042A/Metal042A_2K-JPG_Color.jpg",
                               "Metal042A/Metal042A_2K-JPG_NormalGL.jpg" },
        { "PaintedBricks001",  "PaintedBricks001/PaintedBricks001_8K-JPG_Color.jpg",
                               "PaintedBricks001/PaintedBricks001_8K-JPG_NormalGL.jpg" },
        { "Tiles130",          "Tiles130/Tiles130_8K-JPG_Color.jpg",
                               "Tiles130/Tiles130_8K-JPG_NormalGL.jpg" },
        { "Tiles074",          "Tiles074/Tiles074_8K-JPG_Color.jpg",
                               "Tiles074/Tiles074_8K-JPG_NormalGL.jpg" },
    };

    static_assert(sizeof(specs) / sizeof(specs[0]) == kFileTextureCount,
                  "specs[] must list exactly the file-loaded textures");
    // Albedos go through an sRGB sampler so the GPU does the gamma decode
    // automatically; normals stay UNORM (linear-space, already in [0,1]).
    // Decodes were kicked off in start_world_texture_decodes() BEFORE
    // init_world ran, so they overlapped with the 6-second terrain mesh
    // upload + chunk build. By the time we walk the futures here the
    // decode work is typically already done and only the serial uploads
    // remain — saves another ~1s over the pure-batched approach.
    {
        std::vector<TextureCpuPixels> cpu =
            finish_background_decode(world_tex_decode_handle_);
        world_tex_decode_handle_ = {};
        for (int i = 0; i < kFileTextureCount; ++i) {
            const int ci = i * 2 + 0, ni = i * 2 + 1;
            // Re-do the path probe-log on miss so users still see which
            // file was missing.
            if (cpu.size() <= static_cast<size_t>(ni) ||
                !cpu[ci].pixels) {
                log::warnf("[texture] %s missing on every probe path",
                            specs[i].color_jpg);
            } else {
                Texture2D r = upload_decoded_pixels(device_, allocator_,
                                                     graphics_queue_,
                                                     graphics_queue_family_,
                                                     std::move(cpu[ci]),
                                                     VK_FORMAT_R8G8B8A8_SRGB);
                if (r.ok) {
                    TextureSlot s{};
                    s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                    albedo_textures_[i] = s;
                }
            }
            if (cpu.size() <= static_cast<size_t>(ni) ||
                !cpu[ni].pixels) {
                log::warnf("[texture] %s missing on every probe path",
                            specs[i].normal_jpg);
            } else {
                Texture2D r = upload_decoded_pixels(device_, allocator_,
                                                     graphics_queue_,
                                                     graphics_queue_family_,
                                                     std::move(cpu[ni]),
                                                     VK_FORMAT_R8G8B8A8_UNORM);
                if (r.ok) {
                    TextureSlot s{};
                    s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                    normal_textures_[i] = s;
                }
            }
        }
    }
    // Slots 7-11: procedurally-baked seamless terrain materials.
    bake_terrain_materials();

    // SPOM displacement maps — bound to cube.frag as sampler2D u_height[5].
    // Order MUST match cube.frag's height_idx_for_albedo() switch:
    //   [0] Bricks078         (albedo idx 1) — castle outer walls
    //   [1] PaintedBricks001  (albedo idx 4) — keep walls
    //   [2] Tiles130          (albedo idx 5) — courtyard floor
    //   [3] Tiles074          (albedo idx 6) — keep interior floor
    //   [4] stylized-grass1 (albedo idx kTexGrass / 8) -- terrain grass
    //                       band material (ground under blades); also
    //                       sampled by .tese for vertex displacement
    //                       on the grass band.
    // Displacement maps — already decoded during init_world by the
    // early-kickoff handle. Just upload the finished CPU buffers.
    {
        std::vector<TextureCpuPixels> cpu =
            finish_background_decode(world_spom_decode_handle_);
        world_spom_decode_handle_ = {};
        for (int i = 0; i < 4; ++i) {
            if (cpu.size() <= static_cast<size_t>(i) || !cpu[i].pixels) {
                log::warnf("[texture] %s missing on every probe path",
                            kWorldSpomPaths[i]);
                continue;
            }
            Texture2D r = upload_decoded_pixels(device_, allocator_,
                                                  graphics_queue_,
                                                  graphics_queue_family_,
                                                  std::move(cpu[i]),
                                                  VK_FORMAT_R8G8B8A8_UNORM);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                spom_height_textures_[i] = s;
            }
        }
    }

    // Stylized-grass texture set (assets/stylized_grass/...). Used as the
    // ground beneath where grass blades grow -- replaces the procedurally
    // baked grass at slot kTexGrass (8). Feeds:
    //   - cube.frag's is_terrain splat path (per-texel grass detail under
    //     the blades) via albedo_textures_[8] + normal_textures_[8].
    //   - cube.frag's per-pixel SPOM displacement in the is_terrain block
    //     and terrain_tess.tese's vertex displacement, both via
    //     spom_height_textures_[4] (height_idx_for_albedo returns 4 for
    //     albedo idx 8).
    // The albedo is multiplied by the AO map at load time to bake in the
    // crevice darkening before sampling (no extra texture binding, no
    // per-frame ALU). Paths probed under multiple roots so the engine
    // runs from build/ (Windows) or the project root (CI) alike.
    auto resolve_grass_path = [&](const char* tail) -> std::string {
        const std::string roots[] = {
            std::string("assets/stylized_grass/stylized-grass1-bl/") + tail,
            std::string("../assets/stylized_grass/stylized-grass1-bl/") + tail,
            std::string("../../assets/stylized_grass/stylized-grass1-bl/") + tail,
        };
        for (const std::string& p : roots) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    auto probe_grass = [&](const char* tail, VkFormat fmt) -> TextureSlot {
        std::string p = resolve_grass_path(tail);
        if (!p.empty()) {
            Texture2D r = upload_texture_from_file(
                device_, allocator_, graphics_queue_,
                graphics_queue_family_, p, fmt);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                return s;
            }
        }
        log::warnf("[texture] stylized-grass %s missing on every probe path",
                   tail);
        return {};
    };
    // Build the AO-multiplied albedo CPU-side. stbi_load is brought in by
    // texture.cpp's TU; the prototypes are re-declared at file scope above
    // to avoid pulling stb_image.h into world.cpp.
    TextureSlot sg_alb{};
    {
        std::string alb_path = resolve_grass_path("stylized-grass1_albedo.png");
        std::string ao_path  = resolve_grass_path("stylized-grass1_ao.png");
        if (!alb_path.empty()) {
            int aw = 0, ah = 0, ac = 0;
            unsigned char* alb_px = stbi_load(alb_path.c_str(),
                                              &aw, &ah, &ac, 4);
            if (alb_px) {
                int ow = 0, oh = 0, oc = 0;
                unsigned char* ao_px = ao_path.empty()
                    ? nullptr
                    : stbi_load(ao_path.c_str(), &ow, &oh, &oc, 4);
                // Multiply RGB by AO.r. If the AO map size differs from
                // the albedo (it shouldn't with this asset, but be safe)
                // use nearest sampling rather than skipping the AO bake.
                if (ao_px) {
                    for (int y = 0; y < ah; ++y) {
                        int ay = (oh > 0) ? (y * oh / ah) : 0;
                        for (int x = 0; x < aw; ++x) {
                            int ax = (ow > 0) ? (x * ow / aw) : 0;
                            int ai = (y  * aw + x ) * 4;
                            int oi = (ay * ow + ax) * 4;
                            unsigned int ao = ao_px[oi]; // R channel
                            // gamma-aware-ish: AO maps are usually linear,
                            // so multiply straight on the sRGB-encoded
                            // albedo. Slightly stronger than linear-space
                            // would be, which matches the artistic intent
                            // (deepen the crevices).
                            for (int c = 0; c < 3; ++c) {
                                unsigned int v = alb_px[ai + c];
                                alb_px[ai + c] =
                                    static_cast<unsigned char>((v * ao) / 255U);
                            }
                        }
                    }
                    stbi_image_free(ao_px);
                }
                Texture2D r = upload_texture_from_pixels(
                    device_, allocator_, graphics_queue_,
                    graphics_queue_family_, alb_px, aw, ah,
                    VK_FORMAT_R8G8B8A8_SRGB, "stylized-grass1_albedo*AO");
                stbi_image_free(alb_px);
                if (r.ok) {
                    sg_alb.image = r.image;
                    sg_alb.alloc = r.alloc;
                    sg_alb.view  = r.view;
                }
            } else {
                log::warnf("[texture] stylized-grass albedo decode failed");
            }
        } else {
            log::warnf("[texture] stylized-grass albedo missing on every probe path");
        }
    }
    TextureSlot sg_nrm = probe_grass(
        "stylized-grass1_normal-ogl.png", VK_FORMAT_R8G8B8A8_UNORM);
    TextureSlot sg_h   = probe_grass(
        "stylized-grass1_height.png",     VK_FORMAT_R8G8B8A8_UNORM);
    // Swap the procedural-grass slot for the stylized-grass albedo/normal.
    // The procedural bake just ran above, so destroy its slot-8 image to
    // avoid leaking before overwriting. If the disk asset failed to load
    // (probe miss) we keep the procedural fallback intact.
    auto kill_slot = [&](TextureSlot& s) {
        if (s.image) texture_upload_batch_sync(device_, allocator_, graphics_queue_);   // may be mid-upload
        if (s.view)  vkDestroyImageView(device_, s.view, nullptr);
        if (s.image) vmaDestroyImage(allocator_, s.image, s.alloc);
        s = {};
    };
    if (sg_alb.image) {
        kill_slot(albedo_textures_[kTexGrass]);
        albedo_textures_[kTexGrass] = sg_alb;
    }
    if (sg_nrm.image) {
        kill_slot(normal_textures_[kTexGrass]);
        normal_textures_[kTexGrass] = sg_nrm;
    }
    // SPOM slot 4 is the stylized-grass height. Always populated; cube.frag's
    // height_idx_for_albedo() returns 4 for albedo idx kTexGrass, and the
    // .tese reads it through u_height[4] for vertex displacement.
    spom_height_textures_[4] = sg_h;

    // Restore the rocky-rugged-terrain PNG on the ROCK slot (kTexRock).
    // The user originally asked for this asset; the earlier integration
    // put it on the grass slot and the stylized-grass swap replaced it.
    // Rocky terrain (cliffs / steep slopes) is the natural home for a
    // realistic rock texture, so we overwrite the procedural rock slot
    // with the PNGs. Probes follow the same multi-root pattern as
    // probe_grass so the engine still finds them from build/ or root.
    auto resolve_rock_path = [&](const char* tail) -> std::string {
        const std::string roots[] = {
            std::string("assets/rocky_terrain/rocky-rugged-terrain-bl/") + tail,
            std::string("../assets/rocky_terrain/rocky-rugged-terrain-bl/") + tail,
            std::string("../../assets/rocky_terrain/rocky-rugged-terrain-bl/") + tail,
        };
        for (const std::string& p : roots) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    auto probe_rock = [&](const char* tail, VkFormat fmt) -> TextureSlot {
        std::string p = resolve_rock_path(tail);
        if (!p.empty()) {
            Texture2D r = upload_texture_from_file(
                device_, allocator_, graphics_queue_,
                graphics_queue_family_, p, fmt);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                return s;
            }
        }
        log::warnf("[texture] rocky-rugged %s missing on every probe path",
                   tail);
        return {};
    };
    TextureSlot rr_alb = probe_rock(
        "rocky-rugged-terrain_1_albedo.png",    VK_FORMAT_R8G8B8A8_SRGB);
    TextureSlot rr_nrm = probe_rock(
        "rocky-rugged-terrain_1_normal-ogl.png", VK_FORMAT_R8G8B8A8_UNORM);
    TextureSlot rr_h   = probe_rock(
        "rocky-rugged-terrain_1_height.png",    VK_FORMAT_R8G8B8A8_UNORM);
    // SPOM slot 5 = rocky-rugged height. Lets the tess displacement +
    // per-pixel SPOM bump where the visible STONES are (not where the
    // grass-height noise wanted them).
    if (rr_h.image) {
        kill_slot(spom_height_textures_[5]);
        spom_height_textures_[5] = rr_h;
    }
    // Rocky-rugged goes into the GRASS-BAND NON-MASK FALLBACK slot
    // (kTexProcGrass / slot 12). cube.frag's grass splat lerps:
    //   grass_mask = 1  -> kTexGrass slot 8  (stylized green grass)
    //   grass_mask = 0  -> kTexProcGrass slot 12 (now rocky-rugged)
    // So on the green terrain band, areas without grass blades read
    // as rocky ground -- which is what the user wants visible.
    // Cliffs / steep slopes keep the procedural rock at kTexRock.
    if (rr_alb.image) {
        kill_slot(albedo_textures_[kTexProcGrass]);
        albedo_textures_[kTexProcGrass] = rr_alb;
    }
    if (rr_nrm.image) {
        kill_slot(normal_textures_[kTexProcGrass]);
        normal_textures_[kTexProcGrass] = rr_nrm;
    }

    load_terrain_pbr_overrides();
    load_foliage_textures();
}

// ----------------------------------------------------------------------
// AAA-terrain Phase 1: real photogrammetry PBR sets over the procedural
// bakes. For each splat layer, assets/terrain_pbr/<Name>/ holds an
// ambientCG 2K-JPG set; we bake AO into the albedo at load (same trick
// as the stylized-grass path) and pack Displacement into the normal
// map's ALPHA — cube.frag's rock POM reads height from
// u_normal[kMatRock].a, so every layer that ships displacement gets
// parallax-capable normals for free. Missing sets leave the procedural
// bake in place (probe-and-fallback like every other asset).
// Foliage textures (assets/foliage): the leaf / needle card atlas (RGBA,
// alpha = coverage) and tree bark. The bark ships as albedo only; its
// normal map (height in alpha) is derived from the luminance.
void VulkanEngine::load_foliage_textures() {
    auto resolve = [](const std::string& tail) -> std::string {
        for (const std::string& p : { "assets/foliage/" + tail, "../assets/foliage/" + tail,
                                      "../../assets/foliage/" + tail }) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    std::vector<TextureSlot> retired;
    auto put = [&](TextureSlot& slot, const unsigned char* px, int w, int h, VkFormat fmt, const char* label) {
        Texture2D t = upload_texture_from_pixels(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                                 px, w, h, fmt, label);
        if (!t.ok) return;
        retired.push_back(slot);
        slot = { t.image, t.alloc, t.view };
    };
    int loaded = 0;
    if (std::string p = resolve("leaf_atlas.png"); !p.empty()) {
        int w = 0, h = 0, c = 0;
        if (unsigned char* px = stbi_load(p.c_str(), &w, &h, &c, 4)) {
            put(albedo_textures_[kTexLeaf], px, w, h, VK_FORMAT_R8G8B8A8_SRGB, "foliage/leaf_atlas");
            stbi_image_free(px);
            ++loaded;
        }
    }
    if (std::string p = resolve("bark.png"); !p.empty()) {
        int w = 0, h = 0, c = 0;
        if (unsigned char* px = stbi_load(p.c_str(), &w, &h, &c, 4)) {
            put(albedo_textures_[kTexBark], px, w, h, VK_FORMAT_R8G8B8A8_SRGB, "foliage/bark");
            // Height = luminance (furrows are dark), normal = its gradient.
            std::vector<float> ht(size_t(w) * size_t(h));
            for (size_t i = 0; i < ht.size(); ++i)
                ht[i] = (0.2126f * px[i * 4] + 0.7152f * px[i * 4 + 1] + 0.0722f * px[i * 4 + 2]) / 255.0f;
            std::vector<unsigned char> nrm(size_t(w) * size_t(h) * 4);
            auto H = [&](int x, int y) { return ht[size_t((y + h) % h) * size_t(w) + size_t((x + w) % w)]; };
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const float dx = (H(x + 1, y) - H(x - 1, y)) * 3.0f, dy = (H(x, y + 1) - H(x, y - 1)) * 3.0f;
                    const float il = 1.0f / std::sqrt(dx * dx + dy * dy + 1.0f);
                    unsigned char* o = &nrm[(size_t(y) * size_t(w) + size_t(x)) * 4];
                    o[0] = static_cast<unsigned char>(std::clamp((-dx * il * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                    o[1] = static_cast<unsigned char>(std::clamp((dy * il * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                    o[2] = static_cast<unsigned char>(std::clamp((il * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                    o[3] = static_cast<unsigned char>(std::clamp(H(x, y) * 255.0f, 0.0f, 255.0f));
                }
            put(normal_textures_[kTexBark], nrm.data(), w, h, VK_FORMAT_R8G8B8A8_UNORM, "foliage/bark_n");
            stbi_image_free(px);
            ++loaded;
        }
    }
    if (!retired.empty()) texture_upload_batch_sync(device_, allocator_, graphics_queue_);
    for (TextureSlot& s : retired) {
        if (s.view)  vkDestroyImageView(device_, s.view, nullptr);
        if (s.image) vmaDestroyImage(allocator_, s.image, s.alloc);
    }
    log::infof("[foliage] %d/2 textures loaded (leaf atlas, bark)", loaded);
}

void VulkanEngine::load_terrain_pbr_overrides() {

    struct PbrSet { int slot; const char* name; };
    const PbrSet sets[] = {
        { kTexRock,  "Rock035"   },
        { kTexGrass, "Grass004"  },
        { kTexDirt,  "Ground024" },
        { kTexSand,  "Ground027" },
        { kTexSnow,  "Snow006"   },
        { kTexRoof,  "RoofingTiles012A" },
        { kTexRoad,  "PavingStones131" },
    };
    auto resolve = [](const std::string& tail) -> std::string {
        const std::string roots[] = {
            "assets/terrain_pbr/" + tail,
            "../assets/terrain_pbr/" + tail,
            "../../assets/terrain_pbr/" + tail,
        };
        for (const std::string& p : roots) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    auto kill_slot = [&](TextureSlot& s) {
        if (s.image) texture_upload_batch_sync(device_, allocator_, graphics_queue_);   // may be mid-upload
        if (s.view)  vkDestroyImageView(device_, s.view, nullptr);
        if (s.image) vmaDestroyImage(allocator_, s.image, s.alloc);
        s = {};
    };
    // Phase 1 (worker threads): decode + AO bake + albedo lift + displacement
    // pack for every set in parallel - ~21 2K JPG decodes were ~1.2 s of
    // serial startup. Phase 2 (this thread): GPU uploads in order.
    struct CpuSet {
        unsigned char* alb = nullptr; int aw = 0, ah = 0;
        unsigned char* nrm = nullptr; int nw = 0, nh = 0;
    };
    constexpr int kSets = int(sizeof(sets) / sizeof(sets[0]));
    CpuSet cpu[kSets];
    auto prepare = [&](int si) {
        const PbrSet& set = sets[si];
        CpuSet& out = cpu[si];
        std::string base = std::string(set.name) + "/" + set.name + "_2K-JPG_";
        std::string p_alb  = resolve(base + "Color.jpg");
        std::string p_nrm  = resolve(base + "NormalGL.jpg");
        std::string p_disp = resolve(base + "Displacement.jpg");
        std::string p_ao   = resolve(base + "AmbientOcclusion.jpg");
        if (p_alb.empty() || p_nrm.empty()) {
            log::warnf("[terrain pbr] %s incomplete - keeping procedural slot %d",
                       set.name, set.slot);
            return;
        }
        // Albedo x AO.
        int aw = 0, ah = 0, ac = 0;
        unsigned char* alb = stbi_load(p_alb.c_str(), &aw, &ah, &ac, 4);
        if (!alb) return;
        if (!p_ao.empty()) {
            int ow = 0, oh = 0, oc = 0;
            unsigned char* ao = stbi_load(p_ao.c_str(), &ow, &oh, &oc, 4);
            if (ao) {
                for (int y = 0; y < ah; ++y) {
                    int ay = (oh > 0) ? (y * oh / ah) : 0;
                    for (int x = 0; x < aw; ++x) {
                        int ax = (ow > 0) ? (x * ow / aw) : 0;
                        int ai = (y * aw + x) * 4, oi = (ay * ow + ax) * 4;
                        unsigned int o = ao[oi];
                        for (int c = 0; c < 3; ++c) {
                            alb[ai + c] = static_cast<unsigned char>(
                                (static_cast<unsigned int>(alb[ai + c]) * o) / 255U);
                        }
                    }
                }
                stbi_image_free(ao);
            }
        }
        // Albedo floor. Rock035 is a near-black basalt (mean sRGB ~23/255
        // ~ 0.9% linear reflectance) - rendered as pitch-black clutter
        // rocks and cliffs. Real weathered rock sits ~15-30%. Scale in
        // LINEAR space so the texture's contrast is kept, only exposure
        // shifts. Sets already brighter than the floor are untouched.
        if (set.slot == kTexRock) {
            float lut[256];
            for (int i = 0; i < 256; ++i) {
                float c = i / 255.0f;
                lut[i] = c <= 0.04045f ? c / 12.92f
                                       : std::pow((c + 0.055f) / 1.055f, 2.4f);
            }
            double sum = 0.0;
            const size_t px = static_cast<size_t>(aw) * ah;
            for (size_t i = 0; i < px; ++i)
                sum += 0.2126 * lut[alb[i * 4]] + 0.7152 * lut[alb[i * 4 + 1]] +
                       0.0722 * lut[alb[i * 4 + 2]];
            const float mean = static_cast<float>(sum / std::max<size_t>(px, 1));
            const float kFloor = 0.15f;
            if (mean > 1e-4f && mean < kFloor) {
                const float k = kFloor / mean;
                // Basalt is cool-tinted; the 13x lift turned that into a
                // teal cast. Pull 60% toward per-pixel luminance so the
                // rock reads as neutral grey stone.
                const float kDesat = 0.6f;
                // linear -> sRGB byte via a 4096-entry table (was 12 M pow()).
                unsigned char enc[4096];
                for (int i = 0; i < 4096; ++i) {
                    float l = (float(i) + 0.5f) / 4096.0f;
                    float s = l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
                    enc[i] = static_cast<unsigned char>(std::clamp(s * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                for (size_t i = 0; i < px; ++i) {
                    const float lr = lut[alb[i * 4]], lg = lut[alb[i * 4 + 1]],
                                lb = lut[alb[i * 4 + 2]];
                    const float y = 0.2126f * lr + 0.7152f * lg + 0.0722f * lb;
                    const float ch[3] = { lr + (y - lr) * kDesat,
                                          lg + (y - lg) * kDesat,
                                          lb + (y - lb) * kDesat };
                    for (int c = 0; c < 3; ++c) {
                        float l = std::clamp(ch[c] * k, 0.0f, 1.0f);
                        alb[i * 4 + c] = enc[std::min(4095, int(l * 4096.0f))];
                    }
                }
                log::infof("[terrain pbr] %s albedo lifted: mean %.3f -> %.2f linear",
                           set.name, mean, kFloor);
            }
        }
        out.alb = alb; out.aw = aw; out.ah = ah;

        // Normal with Displacement packed into alpha.
        int nw = 0, nh = 0, nc = 0;
        unsigned char* nrm = stbi_load(p_nrm.c_str(), &nw, &nh, &nc, 4);
        if (nrm && !p_disp.empty()) {
            int dw = 0, dh = 0, dc = 0;
            unsigned char* dsp = stbi_load(p_disp.c_str(), &dw, &dh, &dc, 4);
            if (dsp) {
                for (int y = 0; y < nh; ++y) {
                    int dy = (dh > 0) ? (y * dh / nh) : 0;
                    for (int x = 0; x < nw; ++x) {
                        int dx = (dw > 0) ? (x * dw / nw) : 0;
                        nrm[(y * nw + x) * 4 + 3] = dsp[(dy * dw + dx) * 4];
                    }
                }
                stbi_image_free(dsp);
            }
        }
        out.nrm = nrm; out.nw = nw; out.nh = nh;
    };
    {
        std::vector<std::future<void>> futs;
        for (int si = 0; si < kSets; ++si)
            futs.push_back(std::async(std::launch::async, prepare, si));
        for (auto& f : futs) f.get();
    }
    // Replaced procedural slots are destroyed after ONE batch sync at the end
    // (they may still be in a pending batched upload).
    std::vector<TextureSlot> retired;
    int loaded = 0;
    for (int si = 0; si < kSets; ++si) {
        const PbrSet& set = sets[si];
        CpuSet& c = cpu[si];
        if (!c.alb) { if (c.nrm) stbi_image_free(c.nrm); continue; }
        Texture2D t_alb = upload_texture_from_pixels(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            c.alb, c.aw, c.ah, VK_FORMAT_R8G8B8A8_SRGB, set.name);
        stbi_image_free(c.alb);
        Texture2D t_nrm{};
        if (c.nrm) {
            t_nrm = upload_texture_from_pixels(
                device_, allocator_, graphics_queue_, graphics_queue_family_,
                c.nrm, c.nw, c.nh, VK_FORMAT_R8G8B8A8_UNORM, set.name);
            stbi_image_free(c.nrm);
        }
        if (t_alb.ok) {
            retired.push_back(albedo_textures_[set.slot]);
            albedo_textures_[set.slot] = { t_alb.image, t_alb.alloc, t_alb.view };
        }
        if (t_nrm.ok) {
            retired.push_back(normal_textures_[set.slot]);
            normal_textures_[set.slot] = { t_nrm.image, t_nrm.alloc, t_nrm.view };
        }
        ++loaded;
    }
    for (TextureSlot& s : retired) kill_slot(s);   // first one syncs the batch
    log::infof("[terrain pbr] %d/7 photogrammetry layer sets loaded "
               "(AO-baked albedo, displacement in normal alpha)", loaded);
}

// ----------------------------------------------------------------------
// Procedural terrain materials.
//
// No ground textures ship on disk (only Ground054), so the 5 splat
// materials are generated at load: seamless tiling albedo (sRGB) + a
// matching tangent-space normal map (UNORM, GL/+Y convention) for each
// of rock / grass / dirt / sand / snow. cube.frag's is_terrain block
// blends them by the existing height/slope weights.
//
// Seamlessness is by construction: every noise octave uses a lattice
// whose period equals the number of cells across the [0,1) tile, so the
// value at u=1 hashes to the same lattice point as u=0. The normal map
// is the central-difference gradient of the same periodic height field,
// so it tiles too — no visible seams at any tiling rate.
// ----------------------------------------------------------------------
namespace {

inline uint32_t tm_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
// Hash of a lattice node, wrapped to `period` so the tile is seamless.
inline float tm_lat(int xi, int yi, int period, uint32_t seed) {
    xi = ((xi % period) + period) % period;
    yi = ((yi % period) + period) % period;
    uint32_t h = tm_hash(static_cast<uint32_t>(xi) * 73856093U ^
                          static_cast<uint32_t>(yi) * 19349663U ^ seed);
    return static_cast<float>(h & 0xFFFFFFu) / float(0x1000000);
}
// Periodic value noise. `cells` lattice cells span the whole tile, and
// the lattice period equals `cells`, so f(u=0)==f(u=1) → seamless.
inline float tm_vnoise(float u, float v, int cells, uint32_t seed) {
    float x = u * cells, y = v * cells;
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    float fx = x - x0, fy = y - y0;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    float a = tm_lat(x0,     y0,     cells, seed);
    float b = tm_lat(x0 + 1, y0,     cells, seed);
    float c = tm_lat(x0,     y0 + 1, cells, seed);
    float d = tm_lat(x0 + 1, y0 + 1, cells, seed);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}
// Tileable fBm: each octave's period == its cell count, so the sum
// stays seamless.
inline float tm_fbm(float u, float v, int base_cells, int octaves,
                     uint32_t seed) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    int cells = base_cells;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * tm_vnoise(u, v, cells, seed + o * 1013u);
        norm += amp;
        amp  *= 0.5f;
        cells *= 2;
    }
    return sum / norm;                       // ~0..1
}
inline float tm_clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
// Ridged multifractal — the standard "eroded mountain" basis. Each
// octave is a sharp ridge (1-|2n-1|), and its amplitude is WEIGHTED by
// the previous octave's value, so detail accumulates on exposed ridges
// and valleys stay smooth (mirrors how real erosion distributes relief
// — crisp crests, clean gullies). `gain` sharpens, `offset` ~1.
inline float tm_ridged_mf(float u, float v, int base_cells, int octaves,
                          uint32_t seed) {
    float sum = 0.0f, freqAmp = 0.5f, norm = 0.0f, prev = 1.0f;
    int   cells = base_cells;
    for (int o = 0; o < octaves; ++o) {
        float n = tm_vnoise(u, v, cells, seed + o * 2237u);
        float r = 1.0f - std::fabs(2.0f * n - 1.0f);   // ridge
        r = r * r;                                      // sharpen crest
        float contrib = r * freqAmp * tm_clamp01(prev); // weight by prev
        sum  += contrib;
        norm += freqAmp;
        prev  = r;
        freqAmp *= 0.55f;
        cells   *= 2;
    }
    return sum / norm;                       // ~0..1, ridged
}
inline float tm_smooth(float a, float b, float x) {
    float t = tm_clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}
// Linear → sRGB 8-bit. Authored colours are linear (matching the old
// in-shader palette); the array sampler is sRGB and decodes on read, so
// we must store the sRGB-encoded byte here.
inline unsigned char tm_lin2srgb8(float c) {
    c = tm_clamp01(c);
    float s = (c <= 0.0031308f) ? (c * 12.92f)
                                : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
    int b = static_cast<int>(s * 255.0f + 0.5f);
    return static_cast<unsigned char>(b < 0 ? 0 : (b > 255 ? 255 : b));
}
inline unsigned char tm_unorm8(float c) {
    int b = static_cast<int>(tm_clamp01(c) * 255.0f + 0.5f);
    return static_cast<unsigned char>(b);
}

enum TmKind { TM_ROCK, TM_GRASS, TM_DIRT, TM_SAND, TM_SNOW };

// Fill albedo (sRGB bytes) + normal (UNORM bytes, GL +Y) for one
// material. Builds a periodic height field first, then derives both the
// colour ramp and the normal from it so relief and shading agree.
void tm_generate(TmKind kind, int W, int H,
                 std::vector<unsigned char>& albedo,
                 std::vector<unsigned char>& normal) {
    albedo.resize(static_cast<size_t>(W) * H * 4);
    normal.resize(static_cast<size_t>(W) * H * 4);
    std::vector<float> hf(static_cast<size_t>(W) * H);  // low-freq: drives albedo
    std::vector<float> rf(static_cast<size_t>(W) * H);  // fine relief: drives normal

    // Per-material recipe. n_strength is the bump depth of the detail
    // normal map. The normal is the gradient of a HIGH-frequency relief
    // field (rf) so adjacent texels actually differ — the old code took
    // the gradient of the low-freq albedo field, which barely changed
    // texel-to-texel, so every normal map came out flat (в‰€(0,0,1)) and
    // the strength slider did nothing.
    int   base_cells, octaves;
    float n_strength;            // 0..1-ish bump depth
    uint32_t seed;
    switch (kind) {
        case TM_ROCK:  base_cells = 6;  octaves = 9; n_strength = 1.45f; seed = 11u; break;
        case TM_GRASS: base_cells = 4;  octaves = 6; n_strength = 0.45f; seed = 23u; break;
        case TM_DIRT:  base_cells = 5;  octaves = 6; n_strength = 0.65f; seed = 37u; break;
        case TM_SAND:  base_cells = 3;  octaves = 5; n_strength = 0.55f; seed = 53u; break;
        default:       base_cells = 3;  octaves = 5; n_strength = 0.30f; seed = 71u; break; // SNOW
    }

    // Height field (periodic).
    for (int y = 0; y < H; ++y) {
        float v = static_cast<float>(y) / H;
        for (int x = 0; x < W; ++x) {
            float u = static_cast<float>(x) / W;
            float f = tm_fbm(u, v, base_cells, octaves, seed);
            if (kind == TM_ROCK) {
                // Eroded-mountain rock: ridged multifractal (crisp
                // crests, smooth valleys) + a sharper high-frequency
                // drainage/gully layer carved in for fine cracks.
                float mf    = tm_ridged_mf(u, v, base_cells, octaves, seed);
                float gully = tm_ridged_mf(u, v, base_cells * 4, 4,
                                           seed + 901u);
                gully = gully * gully;                  // tighten ravines
                f = tm_clamp01(mf - 0.35f * gully);
            } else if (kind == TM_SAND) {
                // Isotropic — no baked directional content (that caused
                // either fixed world stripes or, when rotated per-pixel
                // in cube.frag, per-triangle UV seams). The shore-
                // following ripple is added smoothly in cube.frag from
                // the height field instead. Soft dunes + fine grain.
                float dune  = tm_fbm(u, v, 6,  4, seed + 7u);
                float grain = tm_fbm(u, v, 48, 2, seed + 13u);
                f = 0.70f * dune + 0.30f * grain;
            } else if (kind == TM_SNOW) {
                // Smooth dunes + sparse sparkle specks.
                float spark = tm_vnoise(u, v, 256, seed + 9u);
                f = 0.85f * f + (spark > 0.93f ? 0.15f : 0.0f);
            }
            hf[static_cast<size_t>(y) * W + x] = f;

            // Fine relief for the NORMAL map: a high-frequency field
            // whose neighbouring texels differ enough to give a real
            // gradient. Mix in the low-freq f so big shapes still cast
            // the right macro tilt.
            float fine_r = tm_fbm(u, v, base_cells * 6, octaves, seed + 101u);
            float micro  = tm_vnoise(u, v, 256, seed + 211u);
            float relief = 0.45f * f + 0.40f * fine_r + 0.15f * micro;
            if (kind == TM_ROCK) {
                // Crisp chiselled relief: the eroded height (f, already
                // ridged-MF) plus a fine high-frequency ridged-MF layer
                // → sharp ridge/crack normal at close range.
                float fine_mf = tm_ridged_mf(u, v, base_cells * 5, 5,
                                             seed + 701u);
                relief = 0.40f * f + 0.60f * fine_mf;
            } else if (kind == TM_SAND) {
                // Isotropic relief (matches the isotropic albedo). The
                // directional ripple is a smooth cube.frag overlay.
                float dn = tm_fbm(u, v, 5,  4, seed + 211u);
                float gr = tm_fbm(u, v, 40, 3, seed + 311u);
                relief = 0.6f * dn + 0.4f * gr;
            }
            rf[static_cast<size_t>(y) * W + x] = relief;
        }
    }

    auto Hsamp = [&](int x, int y) -> float {
        x = (x % W + W) % W; y = (y % H + H) % H;
        return rf[static_cast<size_t>(y) * W + x];
    };

    for (int y = 0; y < H; ++y) {
        float v = static_cast<float>(y) / H;
        for (int x = 0; x < W; ++x) {
            float u = static_cast<float>(x) / W;
            float h = hf[static_cast<size_t>(y) * W + x];
            // Fine break-up so flat patches don't read as one colour.
            float fine = tm_vnoise(u, v, 256, seed + 3u) - 0.5f;

            float rl, gl, bl;       // linear albedo
            switch (kind) {
                case TM_ROCK: {
                    float t = tm_smooth(0.25f, 0.85f, h);
                    rl = 0.085f + (0.215f - 0.085f) * t;
                    gl = 0.080f + (0.205f - 0.080f) * t;
                    bl = 0.072f + (0.190f - 0.072f) * t;
                    float s = fine * 0.045f; rl += s; gl += s; bl += s;
                } break;
                case TM_GRASS: {
                    float t = tm_smooth(0.30f, 0.72f, h);
                    // Brighter, lower-contrast green so the cross-fade
                    // with neighbouring layers reads as a smooth tonal
                    // gradient, not dark speckle over green.
                    rl = 0.085f + (0.150f - 0.085f) * t;
                    gl = 0.150f + (0.240f - 0.150f) * t;
                    bl = 0.045f + (0.075f - 0.045f) * t;
                    float clump = (tm_vnoise(u, v, 96, seed + 5u) - 0.5f) * 0.020f;
                    rl += clump * 0.6f; gl += clump; bl += clump * 0.3f;
                } break;
                case TM_DIRT: {
                    float t = tm_smooth(0.25f, 0.80f, h);
                    rl = 0.075f + (0.185f - 0.075f) * t;
                    gl = 0.048f + (0.115f - 0.048f) * t;
                    bl = 0.025f + (0.060f - 0.025f) * t;
                    // gentle embedded pebbles (no harsh dark stamps)
                    float peb = tm_vnoise(u, v, 64, seed + 6u);
                    if (peb > 0.82f) { rl *= 0.85f; gl *= 0.85f; bl *= 0.85f; }
                    float s = fine * 0.025f; rl += s; gl += s; bl += s;
                } break;
                case TM_SAND: {
                    float t = tm_smooth(0.30f, 0.75f, h);
                    rl = 0.300f + 0.070f * t;
                    gl = 0.235f + 0.060f * t;
                    bl = 0.120f + 0.040f * t;
                    float s = fine * 0.030f; rl += s; gl += s; bl += s;
                } break;
                default: { // SNOW
                    float t = tm_smooth(0.35f, 0.80f, h);
                    rl = 0.760f + 0.180f * t;
                    gl = 0.790f + 0.170f * t;
                    bl = 0.880f + 0.110f * t;     // faint cool tint
                    float s = fine * 0.020f; rl += s; gl += s; bl += s;
                } break;
            }

            size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            albedo[idx + 0] = tm_lin2srgb8(rl);
            albedo[idx + 1] = tm_lin2srgb8(gl);
            albedo[idx + 2] = tm_lin2srgb8(bl);
            albedo[idx + 3] = 255;

            // Tangent-space normal from the FINE relief field's central
            // difference. GL convention: G = +Y; a bump (higher centre)
            // tilts the normal toward -gradient. The RELIEF_GAIN turns
            // the small texel-to-texel relief delta into a visible
            // surface slope so the detail-normal strength slider has a
            // real, wide effect.
            const float RELIEF_GAIN = 14.0f;
            float g = n_strength * RELIEF_GAIN;
            float dhx = (Hsamp(x + 1, y) - Hsamp(x - 1, y)) * g;
            float dhy = (Hsamp(x, y + 1) - Hsamp(x, y - 1)) * g;
            dhx = dhx < -4.0f ? -4.0f : (dhx > 4.0f ? 4.0f : dhx);
            dhy = dhy < -4.0f ? -4.0f : (dhy > 4.0f ? 4.0f : dhy);
            float nx = -dhx, ny = -dhy, nz = 1.0f;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            normal[idx + 0] = tm_unorm8(nx * inv * 0.5f + 0.5f);
            normal[idx + 1] = tm_unorm8(ny * inv * 0.5f + 0.5f);
            normal[idx + 2] = tm_unorm8(nz * inv * 0.5f + 0.5f);
            // Alpha = the relief HEIGHT field itself (0..1). cube.frag's
            // terrain parallax-occlusion march samples this — free, no
            // extra texture/binding. Rock gets an extra eroded-gully
            // term so the POM carves cracked, weathered rock.
            float reliefH = rf[static_cast<size_t>(y) * W + x];
            if (kind == TM_ROCK) {
                // POM height = eroded ridged-MF + deep drainage gullies
                // so parallax carves crisp ravines into the rock.
                float ru = static_cast<float>(x) / W;
                float rv = static_cast<float>(y) / H;
                float mf  = tm_ridged_mf(ru, rv, base_cells * 2, 6,
                                         seed + 401u);
                float gul = tm_ridged_mf(ru, rv, base_cells * 5, 4,
                                         seed + 411u);
                reliefH = tm_clamp01(0.60f * mf + 0.40f * gul * gul);
            }
            normal[idx + 3] = tm_unorm8(reliefH);
        }
    }
}

} // namespace

void VulkanEngine::bake_terrain_materials() {
    const int W = 512, H = 512;
    struct Mat { TmKind kind; int slot; const char* name; };
    const Mat mats[kTerrainMatCount] = {
        { TM_ROCK,  kTexRock,      "terrain/rock"  },
        { TM_GRASS, kTexGrass,     "terrain/grass" },
        { TM_DIRT,  kTexDirt,      "terrain/dirt"  },
        { TM_SAND,  kTexSand,      "terrain/sand"  },
        { TM_SNOW,  kTexSnow,      "terrain/snow"  },
        // Second copy of the procedural grass into slot 12. The
        // probe_grass step will overwrite slot 8 with the stylized
        // PNG; slot 12 keeps the procedural grass so cube.frag can
        // lerp grass-mask=0 -> procedural, grass-mask=1 -> stylized.
        { TM_GRASS, kTexProcGrass, "terrain/proc_grass" },
        // Placeholders for the roof / road slots (replaced by the
        // ambientCG sets in load_terrain_pbr_overrides()).
        { TM_ROCK,  kTexRoof,      "terrain/roof_placeholder" },
        { TM_DIRT,  kTexRoad,      "terrain/road_placeholder" },
        // Placeholders for the foliage slots (load_foliage_textures()).
        { TM_GRASS, kTexLeaf,      "foliage/leaf_placeholder" },
        { TM_ROCK,  kTexBark,      "foliage/bark_placeholder" },
    };
    // CPU generation of all materials in parallel (pure functions of
    // kind + size); uploads stay on this thread (queue owner).
    std::vector<unsigned char> albs[kTerrainMatCount], nrms[kTerrainMatCount];
    {
        std::vector<std::future<void>> futs;
        for (int i = 0; i < kTerrainMatCount; ++i)
            futs.push_back(std::async(std::launch::async, [&, i]() {
                tm_generate(mats[i].kind, W, H, albs[i], nrms[i]);
            }));
        for (auto& f : futs) f.get();
    }
    for (int i = 0; i < kTerrainMatCount; ++i) {
        const Mat& m = mats[i];
        std::vector<unsigned char>& alb = albs[i];
        std::vector<unsigned char>& nrm = nrms[i];
        Texture2D a = upload_texture_from_pixels(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            alb.data(), W, H, VK_FORMAT_R8G8B8A8_SRGB, m.name);
        Texture2D n = upload_texture_from_pixels(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            nrm.data(), W, H, VK_FORMAT_R8G8B8A8_UNORM, m.name);
        albedo_textures_[m.slot] = { a.image, a.alloc, a.view };
        normal_textures_[m.slot] = { n.image, n.alloc, n.view };
    }
    log::infof("[texture] baked %d procedural terrain materials (%dx%d)",
               kTerrainMatCount, W, H);
}

void VulkanEngine::destroy_textures() {
    auto kill = [&](TextureSlot& s) {
        if (s.view)  vkDestroyImageView(device_, s.view, nullptr);
        if (s.image) vmaDestroyImage(allocator_, s.image, s.alloc);
        s = {};
    };
    for (int i = 0; i < kTextureCount; ++i) {
        kill(albedo_textures_[i]);
        kill(normal_textures_[i]);
    }
    for (int i = 0; i < kSpomMaterialCount; ++i) {
        kill(spom_height_textures_[i]);
    }
    if (texture_sampler_) {
        vkDestroySampler(device_, texture_sampler_, nullptr);
        texture_sampler_ = VK_NULL_HANDLE;
    }
}

// FSR2 mip-bias (AMD recommendation): when input is rendered below the
// display resolution, the sampler should bias towards sharper mips so
// the upscaler's frequency content matches the display. Formula from
// FSR2 docs:  log2(render_w / display_w) - 1.
//
// Called from init() between apply_render_scale() and the scene
// descriptor write so the new sampler is captured by the descriptor.
// Live toggling rt_.fsr2_enabled mid-game does NOT update the bias
// (would require waiting on the device + rewriting every descriptor
// set that captures the sampler — a follow-up).
void VulkanEngine::update_texture_sampler_for_fsr2() {
    if (!device_) return;
    float bias = 0.0f;
    if (rt_.fsr2_enabled && swapchain_extent_.width > 0 &&
        render_extent_.width > 0 &&
        swapchain_extent_.width != render_extent_.width) {
        float ratio = static_cast<float>(render_extent_.width) /
                      static_cast<float>(swapchain_extent_.width);
        bias = std::log2(ratio) - 1.0f;
    }
    if (texture_sampler_) {
        vkDestroySampler(device_, texture_sampler_, nullptr);
        texture_sampler_ = VK_NULL_HANDLE;
    }
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // Anisotropic filtering — keeps the terrain ground crisp at grazing
    // angles (the dominant "blurry everywhere" cause for a big flat
    // surface). Clamp to the device's supported max.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    // 4x, not 16x: terrain is triplanar (3 albedo + 3 normal taps per
    // fragment), so anisotropy cost is multiplied 6x over a screen-
    // filling ground. 4x removes essentially all the grazing-angle blur
    // while keeping the per-frame cost sane (16x cratered it to ~20fps).
    float aniso = std::min(4.0f, props.limits.maxSamplerAnisotropy);
    si.anisotropyEnable = (aniso > 1.0f) ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = aniso;
    si.maxLod = VK_LOD_CLAMP_NONE;
    si.mipLodBias = bias;
    vk_check(vkCreateSampler(device_, &si, nullptr, &texture_sampler_),
             "texture sampler (fsr2 update)");
    if (bias != 0.0f) {
        log::infof("[fsr2] texture sampler bias = %.2f", bias);
    }
}

// Bake grass eligibility into an RG8 texture once at level load.
//   R = presence  (low-freq FBM, smoothed 0..1)
//   G = slope mag (raw |в€‡h| clamped to [0..2] / 2 → [0..1])
// Storing them separately lets the shaders apply the slope-cutoff
// slider (`grass_slope_n_min`) at runtime without re-baking. Both
// raymarched grass + the terrain green-tint sample this same texture
// for consistent eligibility.

} // namespace qlike
