// World-side state: level + meshes + skybox + texture array + viewmodel +
// dyn-prop spawning + the per-frame `render_world` raster pass that drives
// every cube/cylinder draw the engine emits. Lives here because all of these
// touch `world_`, `dyn_props_`, `physics_`, the TLAS instance cache (via
// rebuild_dyn_render_cache), and the same pipeline_layout_/scene_desc_set_.

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
#include <limits>
#include <vector>

namespace qlike {

// Distance-based LOD selector for terrain chunks. Used by all three
// terrain draw paths (depth pre-pass, color pass, shadow pass) so LOD
// stays consistent вЂ” if the pre-pass picked half-LOD and the color
// pass picked full, the depth values would mismatch and the color
// pass's LESS_OR_EQUAL test would discard fragments. Thresholds are
// pulled from rt_ so the user can push higher-resolution terrain
// further out via the sliders in the Graphics Settings menu.
static int pick_terrain_lod(const TerrainChunk& c, glm::vec3 cam_xz,
                            float lod1, float lod2, float lod3) {
    float dx = c.center.x - cam_xz.x;
    float dz = c.center.z - cam_xz.z;
    float d = std::sqrt(dx * dx + dz * dz);
    if (d < lod1) return 0;
    if (d < lod2) return 1;
    if (d < lod3) return 2;
    return 3;
}

// CD-LOD morph factor for a chunk currently rendered at LOD 0. Ramps
// from 0 (full LOD-0 surface) to 1 (LOD-1 stride-2 interp) over the
// last 25% of the LOD-0 distance band. Threshold scales with the
// user-selected lod1 distance so morphing stays linked to where the
// LOD switch actually happens.
static float pick_terrain_morph(const TerrainChunk& c, glm::vec3 cam_xz,
                                int lod, float lod1) {
    if (lod != 0) return 0.0f;
    float dx = c.center.x - cam_xz.x;
    float dz = c.center.z - cam_xz.z;
    float d = std::sqrt(dx * dx + dz * dz);
    float fade_start = lod1 * 0.75f;
    float fade_end   = lod1;
    if (d <= fade_start) return 0.0f;
    if (d >= fade_end)   return 1.0f;
    float t = (d - fade_start) / (fade_end - fade_start);
    return t * t * (3.0f - 2.0f * t);
}

void VulkanEngine::apply_terrain_brush(float dt) {
    if (!terrain_brush_has_hit_ || terrain_chunks_.chunks.empty()) return;
    if (terrain_data_.heights.empty()) return;
    const float r = terrain_brush_radius_;
    const float w = terrain_data_.cell;
    const int dim = terrain_data_.dim;
    const int W = dim + 1;

    // Convert world hit point to grid coords + brush radius in cells.
    float cx = (terrain_brush_world_pos_.x - terrain_data_.origin_x) / w;
    float cz = (terrain_brush_world_pos_.z - terrain_data_.origin_z) / w;
    int rcells = static_cast<int>(std::ceil(r / w)) + 1;
    int ix0 = std::max(0, static_cast<int>(std::floor(cx)) - rcells);
    int ix1 = std::min(W - 1, static_cast<int>(std::ceil(cx))  + rcells);
    int iz0 = std::max(0, static_cast<int>(std::floor(cz)) - rcells);
    int iz1 = std::min(W - 1, static_cast<int>(std::ceil(cz))  + rcells);
    if (ix0 > ix1 || iz0 > iz1) return;

    const float strength = terrain_brush_strength_ * dt;

    for (int iz = iz0; iz <= iz1; ++iz) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            float dx = (static_cast<float>(ix) - cx) * w;
            float dz = (static_cast<float>(iz) - cz) * w;
            float d = std::sqrt(dx * dx + dz * dz);
            if (d > r) continue;
            // Smoothstep falloff so the brush has a soft edge.
            float t = 1.0f - (d / r);
            float falloff = t * t * (3.0f - 2.0f * t);
            size_t idx = static_cast<size_t>(iz) * static_cast<size_t>(W) +
                         static_cast<size_t>(ix);
            float& h = terrain_data_.heights[idx];
            switch (terrain_brush_mode_) {
                case TerrainBrushMode::Raise:
                    h += strength * falloff;
                    break;
                case TerrainBrushMode::Lower:
                    h -= strength * falloff;
                    break;
                case TerrainBrushMode::Smooth: {
                    // 4-tap blur toward neighbour mean.
                    int xm = std::max(0, ix - 1);
                    int xp = std::min(W - 1, ix + 1);
                    int zm = std::max(0, iz - 1);
                    int zp = std::min(W - 1, iz + 1);
                    auto at = [&](int x, int z) {
                        return terrain_data_.heights[
                            static_cast<size_t>(z) * static_cast<size_t>(W) +
                            static_cast<size_t>(x)];
                    };
                    float mean = 0.25f * (at(xm, iz) + at(xp, iz) +
                                          at(ix, zm) + at(ix, zp));
                    h += (mean - h) * std::min(1.0f, strength * 4.0f * falloff);
                    break;
                }
                case TerrainBrushMode::Flatten: {
                    float target = terrain_brush_flatten_target_;
                    h += (target - h) * std::min(1.0f, strength * 0.5f * falloff);
                    break;
                }
            }
        }
    }

    // Mark affected chunks dirty for next-frame mesh rebuild. A chunk's
    // sample range is [origin_ix, origin_ix + chunk_cells] in X and Z,
    // so we mark any chunk whose range overlaps the brush footprint.
    const int chunk_cells = terrain_chunks_.chunk_cells;
    auto mark_chunk_dirty = [&](int ci) {
        for (int existing : terrain_dirty_chunks_) {
            if (existing == ci) return;
        }
        terrain_dirty_chunks_.push_back(ci);
    };
    for (size_t ci = 0; ci < terrain_chunks_.chunks.size(); ++ci) {
        const auto& c = terrain_chunks_.chunks[ci];
        int cix0 = c.origin_ix;
        int cix1 = c.origin_ix + chunk_cells;
        int ciz0 = c.origin_iz;
        int ciz1 = c.origin_iz + chunk_cells;
        if (ix1 < cix0 || ix0 > cix1 || iz1 < ciz0 || iz0 > ciz1) continue;
        mark_chunk_dirty(static_cast<int>(ci));
    }
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
}

void VulkanEngine::rebuild_dirty_terrain_chunks() {
    if (terrain_dirty_chunks_.empty()) return;
    if (terrain_data_.heights.empty()) return;
    Heightmap hm{};
    hm.dim = terrain_data_.dim;
    hm.cell = terrain_data_.cell;
    hm.origin_x = terrain_data_.origin_x;
    hm.origin_z = terrain_data_.origin_z;
    hm.heights = terrain_data_.heights;
    for (int ci : terrain_dirty_chunks_) {
        if (ci < 0 || static_cast<size_t>(ci) >= terrain_chunks_.chunks.size()) continue;
        rebuild_chunk_vertices(device_, allocator_, graphics_queue_,
                               graphics_queue_family_, hm,
                               terrain_chunks_.chunks[ci]);
    }
    terrain_dirty_chunks_.clear();
}

void VulkanEngine::refresh_terrain_collision() {
    if (!terrain_jolt_dirty_) return;
    if (terrain_data_.heights.empty()) return;
    // Jolt has no incremental update for HeightFieldShape вЂ” we rebuild
    // the whole shape. Mirrors the sub-grid trim done at level load:
    // when sample_count would be odd Jolt's block-size requirement
    // forces us onto a 2048Г—2048 sub-grid taken from the 2049ВІ heightmap.
    if (physics_) {
        const int W = terrain_data_.dim + 1;
        const int physics_samples = (W % 2 == 0) ? W : (W - 1);
        if (physics_samples == W) {
            physics_->add_static_heightfield(terrain_data_.heights.data(),
                                             terrain_data_.dim,
                                             glm::vec2(terrain_data_.origin_x,
                                                        terrain_data_.origin_z),
                                             terrain_data_.cell);
        } else {
            std::vector<float> jolt_heights(static_cast<size_t>(physics_samples) *
                                             static_cast<size_t>(physics_samples));
            for (int z = 0; z < physics_samples; ++z) {
                std::memcpy(jolt_heights.data() +
                                static_cast<size_t>(z) *
                                static_cast<size_t>(physics_samples),
                            terrain_data_.heights.data() +
                                static_cast<size_t>(z) * static_cast<size_t>(W),
                            static_cast<size_t>(physics_samples) * sizeof(float));
            }
            physics_->add_static_heightfield(jolt_heights.data(),
                                             physics_samples - 1,
                                             glm::vec2(terrain_data_.origin_x,
                                                        terrain_data_.origin_z),
                                             terrain_data_.cell);
        }
    }
    terrain_jolt_dirty_ = false;
}

void VulkanEngine::refresh_terrain_blas() {
    if (!terrain_blas_dirty_) return;
    // BLAS rebuild is the heavy part. Defer to mouse-up so per-frame
    // cost stays low during a stroke. The merged terrain mesh's vertex
    // buffer is rebuilt from the current heightmap, then the BLAS is
    // re-built in place. RT shadows/GI will be slightly stale until
    // this fires, which reads as "lighting catches up after you stop
    // sculpting" вЂ” acceptable.
    if (terrain_mesh_.vertex_buffer && !terrain_data_.heights.empty()) {
        Heightmap hm{};
        hm.dim = terrain_data_.dim;
        hm.cell = terrain_data_.cell;
        hm.origin_x = terrain_data_.origin_x;
        hm.origin_z = terrain_data_.origin_z;
        hm.heights = terrain_data_.heights;
        // Rebuild the full-mesh vertex buffer in place by abusing the
        // chunk rebuild helper on a virtual "chunk" covering the whole
        // mesh.
        TerrainChunk fake{};
        fake.mesh = terrain_mesh_;
        fake.origin_ix = 0;
        fake.origin_iz = 0;
        fake.sample_dim = terrain_data_.dim + 1;
        rebuild_chunk_vertices(device_, allocator_, graphics_queue_,
                               graphics_queue_family_, hm, fake);
    }
    terrain_blas_dirty_ = false;
}

void VulkanEngine::destroy_terrain_shadow_texture() {
    if (terrain_shadow_view_) {
        vkDestroyImageView(device_, terrain_shadow_view_, nullptr);
        terrain_shadow_view_ = VK_NULL_HANDLE;
    }
    if (terrain_shadow_image_) {
        vmaDestroyImage(allocator_, terrain_shadow_image_, terrain_shadow_alloc_);
        terrain_shadow_image_ = VK_NULL_HANDLE;
        terrain_shadow_alloc_ = nullptr;
    }
    if (terrain_shadow_sampler_) {
        vkDestroySampler(device_, terrain_shadow_sampler_, nullptr);
        terrain_shadow_sampler_ = VK_NULL_HANDLE;
    }
    terrain_shadow_dim_ = 0;
}

namespace {
// First-time creation of the heightmap shadow image + view + sampler.
// Pulled out of rebuild_terrain_shadow_texture so the progressive tile
// path can rely on the resources existing without duplicating setup.
// Caller writes binding 6 separately; transitions the image into
// SHADER_READ_ONLY_OPTIMAL on first creation (so partial uploads can
// flip TRANSFER_DST в†’ SHADER_READ_ONLY symmetrically).
void ensure_shadow_resources(VkDevice device, VmaAllocator alloc,
                             VkQueue queue, uint32_t qfam,
                             int W, int H,
                             VkImage& img, VmaAllocation& img_alloc,
                             VkImageView& view, VkSampler& sampler,
                             int& dim_out) {
    if (img) return;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = { static_cast<uint32_t>(W),
                   static_cast<uint32_t>(H), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(alloc, &ici, &aci, &img, &img_alloc, nullptr),
             "terrain shadow image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8_UNORM;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device, &vci, nullptr, &view),
             "terrain shadow view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device, &si, nullptr, &sampler),
             "terrain shadow sampler");

    // Initialise to SHADER_READ_ONLY so the very first tick's
    // tile transition (SHADER_READ_ONLY в†’ TRANSFER_DST в†’ SHADER_READ_ONLY)
    // is symmetric. Without this the layout starts UNDEFINED and the
    // first frame would have to special-case the source layout.
    vkinit::one_time_submit(device, queue, qfam, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, img,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    dim_out = W;
}
} // namespace

void VulkanEngine::rebuild_terrain_shadow_texture() {
    if (terrain_data_.heights.empty()) return;
    // Reconstruct a Heightmap view over the cached data.
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;

    // Sun direction from the user settings (matches what cube.frag uses).
    float p = glm::radians(rt_.sun_pitch_deg);
    float y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun_dir(std::sin(y) * std::cos(p), std::sin(p),
                      std::cos(y) * std::cos(p));
    terrain_shadow_sun_dir_        = sun_dir;
    terrain_shadow_target_sun_dir_ = sun_dir;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_jobs_.clear();
        terrain_shadow_results_.clear();
    }

    // Sync bake stays at SS=1 for fast level load (~150ms). When the
    // user has supersample > 1 the texture is created at SS×heightmap-dim,
    // and we replicate each SS=1 texel into an SS×SS block so the
    // texture is fully populated from frame 1 — the worker then
    // re-bakes properly at the user's SS in the background.
    const int ss = std::max(1, rt_.terrain_bake_supersample);
    terrain_shadow_active_ss_ = ss;
    std::vector<uint8_t> base = bake_heightmap_shadow(hm, sun_dir);
    const int W = hm.width()  * ss;
    const int H = hm.height() * ss;
    std::vector<uint8_t> shadow;
    if (ss == 1) {
        shadow = std::move(base);
    } else {
        shadow.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
        for (int z = 0; z < H; ++z) {
            for (int x = 0; x < W; ++x) {
                int sx = x / ss;
                int sz = z / ss;
                shadow[static_cast<size_t>(z) * static_cast<size_t>(W) +
                       static_cast<size_t>(x)] =
                    base[static_cast<size_t>(sz) * static_cast<size_t>(hm.width()) +
                         static_cast<size_t>(sx)];
            }
        }
    }

    const bool first_create = (terrain_shadow_image_ == VK_NULL_HANDLE);
    ensure_shadow_resources(device_, allocator_, graphics_queue_,
                             graphics_queue_family_, W, H,
                             terrain_shadow_image_, terrain_shadow_alloc_,
                             terrain_shadow_view_, terrain_shadow_sampler_,
                             terrain_shadow_dim_);

    // Stage upload via a scratch host-visible buffer.
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(W) * static_cast<VkDeviceSize>(H);
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
        std::memcpy(ai.pMappedData, shadow.data(), static_cast<size_t>(bytes));
    }

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = { static_cast<uint32_t>(W),
                                static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(allocator_, stage, stage_alloc);

    // Write the texture into descriptor binding 6 only on first
    // creation; the image handle doesn't change after that, so partial
    // tile uploads don't need to re-write the descriptor.
    if (first_create) {
        VkDescriptorImageInfo dii{};
        dii.sampler = terrain_shadow_sampler_;
        dii.imageView = terrain_shadow_view_;
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 6;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &dii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
}

// ---------------- Sun shadow map (single-cascade ortho) ----------------
//
// Standard depth-only shadow pass: one 2048ВІ D32 depth target written
// from the sun's POV each frame, sampled by grass.vert as a
// sampler2DShadow at descriptor binding 7. Cube.frag continues to use
// RT shadow rays вЂ” we don't move the whole engine to CSM yet, this is
// the cheap path that lets dynamic occluders (castle, dyn-props) cast
// shadow on grass without firing a ray per blade.
//
// Light frustum: ortho box centred on the player. Half-width covers the
// grass distance + slack. Depth along the sun axis is large so far-off
// mountain ridges still register as casters even when the receiver
// (grass) is right under the camera. Texel-snapped each frame to avoid
// crawling shadow edges as the player walks.
void VulkanEngine::init_sun_shadow_resources() {
    // Snap requested resolution to the allowed list. Default 1024 if a
    // settings file from before this slider existed lacks the field.
    int requested = rt_.shadow_map_resolution;
    int snapped = kShadowMapResolutions[1];  // 1024
    int best_diff = std::abs(requested - snapped);
    for (int v : kShadowMapResolutions) {
        int d = std::abs(requested - v);
        if (d < best_diff) { best_diff = d; snapped = v; }
    }
    sun_shadow_dim_ = snapped;
    rt_.shadow_map_resolution = snapped;  // write-back so the UI shows the snapped value
    const uint32_t W = static_cast<uint32_t>(sun_shadow_dim_);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_D32_SFLOAT;
    ici.extent = { W, W, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &sun_shadow_image_, &sun_shadow_alloc_, nullptr),
             "sun shadow image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = sun_shadow_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr, &sun_shadow_view_),
             "sun shadow view");

    // Comparison sampler вЂ” sampler2DShadow returns the comparison result
    // (0..1 with PCF) instead of raw depth. LINEAR + 2x2 hardware PCF gives
    // soft penumbra essentially for free.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;  // out-of-bounds = lit
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    vk_check(vkCreateSampler(device_, &si, nullptr, &sun_shadow_sampler_),
             "sun shadow sampler");

    // Initial transition into SHADER_READ_ONLY so render_sun_shadow_pass's
    // first transition (READ_ONLY в†’ DEPTH_ATTACHMENT_OPTIMAL) is
    // symmetric with subsequent frames.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image_aspect(cb, sun_shadow_image_,
                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         VK_IMAGE_ASPECT_DEPTH_BIT);
    });

    // Descriptor binding 7 only needs writing once вЂ” the image handle
    // doesn't change for the lifetime of the engine.
    VkDescriptorImageInfo dii{};
    dii.sampler = sun_shadow_sampler_;
    dii.imageView = sun_shadow_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 7;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void VulkanEngine::destroy_sun_shadow_resources() {
    if (sun_shadow_view_) {
        vkDestroyImageView(device_, sun_shadow_view_, nullptr);
        sun_shadow_view_ = VK_NULL_HANDLE;
    }
    if (sun_shadow_image_) {
        vmaDestroyImage(allocator_, sun_shadow_image_, sun_shadow_alloc_);
        sun_shadow_image_ = VK_NULL_HANDLE;
        sun_shadow_alloc_ = nullptr;
    }
    if (sun_shadow_sampler_) {
        vkDestroySampler(device_, sun_shadow_sampler_, nullptr);
        sun_shadow_sampler_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::update_sun_shadow_light_vp() {
    // Sun direction from the user settings (matches cube.frag and the
    // heightmap bake). Light position floats along the sun axis above the
    // player so the ortho frustum's depth range covers nearby grass and
    // distant mountain ridges that need to cast on it.
    float p_rad = glm::radians(rt_.sun_pitch_deg);
    float y_rad = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun_dir = glm::normalize(glm::vec3(
        std::sin(y_rad) * std::cos(p_rad),
        std::sin(p_rad),
        std::cos(y_rad) * std::cos(p_rad)));

    // Half-extent of the ortho box, slider-controlled.
    const float half = std::max(rt_.shadow_map_world_half, 30.0f);
    // Sun-axis depth: large so the shadow camera can sit far back along
    // the sun direction and still capture distant cliff peaks.
    const float depth_back  = 600.0f;
    const float depth_front = 400.0f;

    glm::vec3 cam = player_.eye_position();

    // Texel-grid snap: round the light-space view-plane origin to the
    // nearest shadow texel so distant edges don't crawl as the player
    // walks. Pick world axes spanning the light-view plane (any two axes
    // perpendicular to sun_dir).
    glm::vec3 up_ref = std::abs(sun_dir.y) > 0.99f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right_axis = glm::normalize(glm::cross(up_ref, sun_dir));
    glm::vec3 up_axis    = glm::normalize(glm::cross(sun_dir, right_axis));
    float texel = (2.0f * half) / static_cast<float>(sun_shadow_dim_);
    float u = glm::dot(cam, right_axis);
    float v = glm::dot(cam, up_axis);
    u = std::round(u / texel) * texel;
    v = std::round(v / texel) * texel;
    // Reconstruct snapped centre (component along sun_dir doesn't matter
    // for an ortho frustum, but keep player-y so the box is positioned
    // sensibly).
    glm::vec3 snapped_centre = right_axis * u + up_axis * v +
                                sun_dir * glm::dot(cam, sun_dir);

    glm::vec3 light_eye = snapped_centre + sun_dir * depth_back;
    glm::mat4 view = glm::lookAt(light_eye, snapped_centre, up_axis);
    glm::mat4 proj = glm::ortho(-half, half, -half, half,
                                 0.0f, depth_back + depth_front);
    proj[1][1] *= -1.0f;  // match Vulkan-y-flipped convention used elsewhere
    sun_shadow_light_vp_ = proj * view;
}

void VulkanEngine::render_sun_shadow_pass(VkCommandBuffer cmd) {
    // Transition into DEPTH_ATTACHMENT_OPTIMAL for the depth-only render.
    vkinit::transition_image_aspect(cmd, sun_shadow_image_,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);

    VkClearValue clear_depth{};
    clear_depth.depthStencil.depth = 1.0f;

    VkRenderingAttachmentInfo depth_att{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = sun_shadow_view_,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear_depth,
    };
    VkExtent2D ext{ static_cast<uint32_t>(sun_shadow_dim_),
                    static_cast<uint32_t>(sun_shadow_dim_) };
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, ext },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 0, .pColorAttachments = nullptr,
        .pDepthAttachment = &depth_att, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width = static_cast<float>(sun_shadow_dim_);
    vp_state.height = static_cast<float>(sun_shadow_dim_);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, ext };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sun_shadow_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    // Slope-scale + constant bias to combat acne. Tuned conservatively;
    // peter-panning at thin geometry edges is the lesser evil here vs
    // shadow acne scrolling across grass as the sun rotates.
    vkCmdSetDepthBias(cmd, /*const*/ 1.5f, /*clamp*/ 0.0f, /*slope*/ 4.0f);

    glm::mat4 vp = sun_shadow_light_vp_;
    Frustum light_frustum = extract_frustum(vp);

    auto push_shadow = [&](const glm::mat4& model) {
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
    };

    // Static brushes (castle, towers). Light-frustum cull: only brushes
    // whose AABB intersects the ortho box need rasterising. Cuts ~80%
    // of brush draws when the player isn't inside the keep вЂ” was the
    // dominant cost behind the 14-second TDR repro.
    VkDeviceSize cube_off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &cube_off);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.aabbs[i];
        if (!aabb_visible(light_frustum, a.min, a.max)) continue;
        glm::mat4 model = glm::translate(glm::mat4(1.0f), b.center) *
                          glm::scale(glm::mat4(1.0f), b.size);
        push_shadow(model);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }

    // Dyn-props вЂ” same cube BLAS is the cube mesh; cylinders use their
    // own mesh. Light-frustum cull mirrors the brushes path.
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(light_frustum, dr.aabb_min, dr.aabb_max)) continue;
        glm::mat4 model = dr.world * glm::scale(glm::mat4(1.0f),
                                                 dyn_props_[i].full_size);
        push_shadow(model);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }

    // Terrain chunks вЂ” frustum-culled, distance-LOD'd. Distance metric is
    // camera-to-chunk (not light-to-chunk) so terrain casters share the
    // same LOD selection as the main render вЂ” keeps the cost bounded
    // and matches the resolution the user actually sees.
    if (!terrain_chunks_.chunks.empty()) {
        glm::vec3 cam = player_.eye_position();
        for (const auto& c : terrain_chunks_.chunks) {
            if (!aabb_visible(light_frustum, c.aabb_min, c.aabb_max)) continue;
            int lod = pick_terrain_lod(c, cam, rt_.terrain_lod1 * rt_.terrain_lod_scale, rt_.terrain_lod2 * rt_.terrain_lod_scale, rt_.terrain_lod3 * rt_.terrain_lod_scale);
            VkDeviceSize toff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &c.mesh.vertex_buffer, &toff);
            VkBuffer ibo = (lod == 0) ? c.mesh.index_buffer : c.ibo_lod[lod - 1];
            vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
            push_shadow(glm::mat4(1.0f));
            vkCmdDrawIndexed(cmd, c.index_count_lod[lod], 1, 0, 0, 0);
        }
    }

    vkCmdEndRendering(cmd);

    // Transition back to SHADER_READ_ONLY so grass.vert can sample it.
    vkinit::transition_image_aspect(cmd, sun_shadow_image_,
                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);
}

void VulkanEngine::start_terrain_shadow_worker() {
    if (terrain_data_.heights.empty()) return;
    if (terrain_shadow_worker_.joinable()) return;     // already running
    // Snapshot the heightmap into a Heightmap struct the worker can read
    // without locks. Future sculpt edits won't propagate; if/when we
    // care, call stop_terrain_shadow_worker / start_terrain_shadow_worker
    // again to rebuild the snapshot.
    terrain_shadow_baker_hm_.dim       = terrain_data_.dim;
    terrain_shadow_baker_hm_.cell      = terrain_data_.cell;
    terrain_shadow_baker_hm_.origin_x  = terrain_data_.origin_x;
    terrain_shadow_baker_hm_.origin_z  = terrain_data_.origin_z;
    terrain_shadow_baker_hm_.heights   = terrain_data_.heights;
    terrain_shadow_stop_.store(false);
    terrain_shadow_worker_ = std::thread([this]() { terrain_shadow_worker_loop(); });
}

void VulkanEngine::stop_terrain_shadow_worker() {
    if (!terrain_shadow_worker_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_stop_.store(true);
        terrain_shadow_jobs_.clear();
    }
    terrain_shadow_cv_.notify_all();
    terrain_shadow_worker_.join();
}

void VulkanEngine::terrain_shadow_worker_loop() {
    // Background bake loop. Pops one job at a time, runs the tile bake
    // on the worker thread (zero impact on main FPS), and pushes the
    // result back. Main thread drains results in tick_terrain_shadow_progressive.
    while (true) {
        ShadowBakeJob job;
        {
            std::unique_lock<std::mutex> lk(terrain_shadow_mutex_);
            terrain_shadow_cv_.wait(lk, [&]{
                return !terrain_shadow_jobs_.empty() ||
                       terrain_shadow_stop_.load();
            });
            if (terrain_shadow_stop_.load()) return;
            job = terrain_shadow_jobs_.front();
            terrain_shadow_jobs_.pop_front();
        }
        // Bake outside the lock so other threads keep running.
        ShadowBakeResult r;
        r.ix = job.ix; r.iz = job.iz; r.w = job.w; r.h = job.h; r.ss = job.ss;
        r.sun_dir = job.sun_dir;
        r.data.resize(static_cast<size_t>(job.w) * static_cast<size_t>(job.h));
        if (job.ss <= 1) {
            bake_heightmap_shadow_tile(terrain_shadow_baker_hm_, job.sun_dir,
                                        job.ix, job.iz, job.w, job.h, r.data.data());
        } else {
            bake_heightmap_shadow_tile_ss(terrain_shadow_baker_hm_, job.sun_dir,
                                           job.ix, job.iz, job.w, job.h, job.ss,
                                           r.data.data());
        }
        {
            std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
            terrain_shadow_results_.push_back(std::move(r));
        }
    }
}

void VulkanEngine::enqueue_terrain_shadow_rebake(glm::vec3 target_sun) {
    if (terrain_data_.heights.empty()) return;
    terrain_shadow_target_sun_dir_ = target_sun;

    // Tile coordinates are in TEXTURE space (heightmap_dim+1 × ss).
    const int ss = std::max(1, terrain_shadow_active_ss_);
    const int W = (terrain_data_.dim + 1) * ss;
    const int H = (terrain_data_.dim + 1) * ss;
    const int ts = kShadowTileSize;
    const glm::vec3 cam = player_.eye_position();

    // Build the full tile list, sorted nearest-first so the worker
    // bakes near-camera tiles before far ones.
    std::vector<ShadowBakeTile> tiles;
    int n_tiles = ((W + ts - 1) / ts) * ((H + ts - 1) / ts);
    tiles.reserve(static_cast<size_t>(n_tiles));
    const float sub_cell = terrain_data_.cell / static_cast<float>(ss);
    for (int iz = 0; iz < H; iz += ts) {
        int th = std::min(ts, H - iz);
        for (int ix = 0; ix < W; ix += ts) {
            int tw = std::min(ts, W - ix);
            float wx = terrain_data_.origin_x +
                       (static_cast<float>(ix) + 0.5f * static_cast<float>(tw)) *
                       sub_cell;
            float wz = terrain_data_.origin_z +
                       (static_cast<float>(iz) + 0.5f * static_cast<float>(th)) *
                       sub_cell;
            float dx = wx - cam.x, dz = wz - cam.z;
            tiles.push_back({ix, iz, tw, th, dx * dx + dz * dz});
        }
    }
    std::sort(tiles.begin(), tiles.end(),
              [](const ShadowBakeTile& a, const ShadowBakeTile& b) {
                  return a.dist_sq < b.dist_sq;
              });

    // Replace the worker's job queue. Drop any older results too — they
    // were baked against a stale sun direction or a different ss.
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_jobs_.clear();
        terrain_shadow_results_.clear();
        for (const auto& t : tiles) {
            terrain_shadow_jobs_.push_back({t.ix, t.iz, t.w, t.h, ss, target_sun});
        }
    }
    terrain_shadow_cv_.notify_all();
}

void VulkanEngine::tick_terrain_shadow_progressive() {
    if (terrain_shadow_image_ == VK_NULL_HANDLE) return;
    // Async path: drain a few finished bake results from the worker
    // thread and upload them to the GPU image. Bake CPU work happens
    // off-thread so there's no impact on FPS even when many tiles are
    // baking. We early-return if no results are ready.
    std::vector<ShadowBakeResult> ready;
    bool jobs_remaining = false;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        const int max_take = kShadowUploadsPerFrame;
        while (ready.size() < static_cast<size_t>(max_take) &&
               !terrain_shadow_results_.empty()) {
            auto& r = terrain_shadow_results_.front();
            // Drop stale results (sun has changed since this tile was
            // queued, or supersample doesn't match the live texture).
            if (glm::distance(r.sun_dir, terrain_shadow_target_sun_dir_) < 0.001f &&
                r.ss == terrain_shadow_active_ss_) {
                ready.push_back(std::move(r));
            }
            terrain_shadow_results_.pop_front();
        }
        jobs_remaining = !terrain_shadow_jobs_.empty() ||
                         !terrain_shadow_results_.empty();
    }
    if (ready.empty()) {
        if (!jobs_remaining) {
            terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
        }
        return;
    }
    // Pack tiles into one staging buffer, upload via a single submit.
    size_t total_bytes = 0;
    for (const auto& r : ready) {
        total_bytes += static_cast<size_t>(r.w) * static_cast<size_t>(r.h);
    }
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = total_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    uint8_t* mapped = static_cast<uint8_t*>(ai.pMappedData);

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(ready.size());
    size_t off = 0;
    for (const auto& r : ready) {
        size_t bytes = static_cast<size_t>(r.w) * static_cast<size_t>(r.h);
        std::memcpy(mapped + off, r.data.data(), bytes);
        VkBufferImageCopy reg{};
        reg.bufferOffset = off;
        reg.bufferRowLength = 0;
        reg.bufferImageHeight = 0;
        reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.imageSubresource.mipLevel = 0;
        reg.imageSubresource.baseArrayLayer = 0;
        reg.imageSubresource.layerCount = 1;
        reg.imageOffset = { r.ix, r.iz, 0 };
        reg.imageExtent = { static_cast<uint32_t>(r.w),
                            static_cast<uint32_t>(r.h), 1 };
        regions.push_back(reg);
        off += bytes;
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                static_cast<uint32_t>(regions.size()),
                                regions.data());
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    if (!jobs_remaining) {
        terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
    }
}

#if 0  // Old inline-bake body — replaced by the worker thread above.
namespace _terrain_shadow_dead_code_removed {

    // Re-sort the remaining tail if the player has drifted enough that
    // priorities changed. 16m threshold в‰€ a quarter of a tile so the
    // ordering stays meaningful without re-sorting every frame.
    const glm::vec3 cam = player_.eye_position();
    if (glm::distance(cam, terrain_shadow_last_sort_pos_) > 16.0f) {
        for (auto& t : terrain_shadow_pending_tiles_) {
            float wx = terrain_data_.origin_x +
                       (static_cast<float>(t.ix) + 0.5f * static_cast<float>(t.w)) *
                       terrain_data_.cell;
            float wz = terrain_data_.origin_z +
                       (static_cast<float>(t.iz) + 0.5f * static_cast<float>(t.h)) *
                       terrain_data_.cell;
            float dx = wx - cam.x, dz = wz - cam.z;
            t.dist_sq = dx * dx + dz * dz;
        }
        std::sort(terrain_shadow_pending_tiles_.begin(),
                  terrain_shadow_pending_tiles_.end(),
                  [](const ShadowBakeTile& a, const ShadowBakeTile& b) {
                      return a.dist_sq < b.dist_sq;
                  });
        terrain_shadow_last_sort_pos_ = cam;
    }

    // Pop a budget of tiles off the front (nearest first).
    int budget = std::min<int>(kShadowTilesPerFrame,
                               static_cast<int>(terrain_shadow_pending_tiles_.size()));
    if (budget <= 0) return;

    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;
    const glm::vec3 sun_dir = terrain_shadow_target_sun_dir_;

    // Pack all tiles' bytes into a single staging buffer with
    // contiguous per-tile regions, one BufferImageCopy per tile.
    struct Pending { int ix, iz, w, h; size_t offset; };
    std::vector<Pending> pendings;
    pendings.reserve(budget);
    size_t total_bytes = 0;
    for (int i = 0; i < budget; ++i) {
        const auto& t = terrain_shadow_pending_tiles_[i];
        pendings.push_back({t.ix, t.iz, t.w, t.h, total_bytes});
        total_bytes += static_cast<size_t>(t.w) * static_cast<size_t>(t.h);
    }

    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = total_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    uint8_t* mapped = static_cast<uint8_t*>(ai.pMappedData);

    // Bake each tile into its slot in the staging buffer. Tiles are
    // small (64Г—64 = 4 KiB), CPU-cheap enough that single-threaded
    // here is fine вЂ” the whole tick stays well under a frame.
    for (int i = 0; i < budget; ++i) {
        const auto& p = pendings[i];
        bake_heightmap_shadow_tile(hm, sun_dir, p.ix, p.iz, p.w, p.h,
                                   mapped + p.offset);
    }

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(budget);
        for (int i = 0; i < budget; ++i) {
            const auto& p = pendings[i];
            VkBufferImageCopy r{};
            r.bufferOffset = p.offset;
            r.bufferRowLength = 0;
            r.bufferImageHeight = 0;
            r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            r.imageSubresource.mipLevel = 0;
            r.imageSubresource.baseArrayLayer = 0;
            r.imageSubresource.layerCount = 1;
            r.imageOffset = { p.ix, p.iz, 0 };
            r.imageExtent = { static_cast<uint32_t>(p.w),
                              static_cast<uint32_t>(p.h), 1 };
            regions.push_back(r);
        }
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                static_cast<uint32_t>(regions.size()),
                                regions.data());
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(allocator_, stage, stage_alloc);

    // Drop the consumed tiles from the front. erase() of a contiguous
    // range is O(N) but the tail is small and the per-frame budget is
    // tiny вЂ” measured negligible vs. the bake itself.
    terrain_shadow_pending_tiles_.erase(
        terrain_shadow_pending_tiles_.begin(),
        terrain_shadow_pending_tiles_.begin() + budget);

    // Once the queue drains, snap the "current" sun to the target so
    // future change-detection compares against what's actually on the
    // GPU. While tiles are still pending, we leave terrain_shadow_sun_dir_
    // alone; the threshold check uses the target field anyway.
    if (terrain_shadow_pending_tiles_.empty()) {
        terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
    }
}
}  // namespace _terrain_shadow_dead_code_removed
#endif

float VulkanEngine::sample_terrain_height(float x, float z) const {
    if (terrain_data_.heights.empty() || terrain_data_.dim <= 0) return 0.0f;
    const int W = terrain_data_.dim + 1;
    float fx = (x - terrain_data_.origin_x) / terrain_data_.cell;
    float fz = (z - terrain_data_.origin_z) / terrain_data_.cell;
    if (fx < 0.0f || fz < 0.0f) return 0.0f;
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    if (ix >= terrain_data_.dim || iz >= terrain_data_.dim) return 0.0f;
    float tx = fx - static_cast<float>(ix);
    float tz = fz - static_cast<float>(iz);
    auto h = [&](int x_, int z_) {
        return terrain_data_.heights[static_cast<size_t>(z_) * static_cast<size_t>(W) +
                                     static_cast<size_t>(x_)];
    };
    float h00 = h(ix,     iz);
    float h10 = h(ix + 1, iz);
    float h01 = h(ix,     iz + 1);
    float h11 = h(ix + 1, iz + 1);
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    return h0 * (1.0f - tz) + h1 * tz;
}

void VulkanEngine::init_world() {
    cube_mesh_ = create_cube_mesh(device_, allocator_, graphics_queue_, graphics_queue_family_);
    cylinder_mesh_ = create_cylinder_mesh(device_, allocator_,
                                          graphics_queue_, graphics_queue_family_);
    world_ = game::make_arena();
    log::infof("world: %u brushes", static_cast<unsigned>(world_.brushes.size()));

    // Procedural terrain: generated first so the castle can be lifted to
    // sit on the plateau height. Phase 1 is non-streaming вЂ” one big mesh
    // (~1kmВІ). See docs/terrain_plan.md for streaming/LOD/sculpt phases.
    {
        HeightmapParams hp{};
        // Two constraints that fight each other:
        //   1. chunk_cells must be divisible by every LOD stride (1, 2, 4, 8)
        //      so the stride-N index loop covers every cell вЂ” otherwise the
        //      lower-LOD index buffers leave a strip of cells uncovered at
        //      the chunk's right/bottom edge в†’ visible square seams.
        //   2. Jolt's HeightFieldShape needs sample_count divisible by
        //      block_size in [2, 8].
        //
        // chunk_cells / 8 means dim is divisible by 8 в†’ sample_count = dim+1
        // is odd в†’ Jolt fails. Resolution: pick dim = 2048 cells (=32Г—64),
        // chunk_cells = 64 (divisible by all four strides), and feed Jolt a
        // 2048Г—2048 sub-grid by truncating the last row/col of the 2049ВІ
        // heightmap. The visual mesh uses the full 2049ВІ so the world edge
        // stays continuous. Jolt loses one cell-width at the world edge вЂ”
        // invisible since the player can't reach the world boundary.
        // Heightmap resolution scale — the user can pick higher density
        // via the Terrain settings menu. Defaults to 1× = the baseline
        // 2048-cell grid. Higher values keep the world extent the same
        // (cell shrinks proportionally) so castle/grass/spawn positions
        // are unaffected.
        const int hres = std::max(1, std::min(4, rt_.terrain_heightmap_scale));
        hp.dim = 2048 * hres;
        hp.cell_size = 1.0f / static_cast<float>(hres);
        hp.height_scale = 140.0f;
        hp.plateau_extent = glm::vec2(28.0f, 28.0f);
        hp.plateau_height = 22.0f;
        hp.plateau_blend  = 24.0f;
        hp.frequency      = 0.0014f;
        Heightmap hm = generate_heightmap(hp);
        // Cache for collision + Phase 4 sculpt access.
        terrain_data_.dim = hm.dim;
        terrain_data_.cell = hm.cell;
        terrain_data_.origin_x = hm.origin_x;
        terrain_data_.origin_z = hm.origin_z;
        terrain_data_.heights = hm.heights;
        terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                           graphics_queue_, graphics_queue_family_, hm);
        // 32Г—32 chunks of 64 cells each вЂ” 64 is divisible by every LOD
        // stride (1/2/4/8) so all four LODs cover every cell. в‰€64 m per
        // chunk side at 1 m cells. Skirt strips on each LOD's index
        // buffer hide LOD-mismatch cracks at chunk seams.
        const int chunks_per_side = 32;
        terrain_chunks_ = build_terrain_chunks(device_, allocator_,
                                               graphics_queue_, graphics_queue_family_,
                                               hm, chunks_per_side);
        // Lift every level brush by plateau_height so the castle's y=0
        // baseline lands on the plateau surface. Cheaper than rewriting
        // level.cpp вЂ” the brush AABBs / world matrices update through the
        // same vector.
        for (auto& b : world_.brushes) b.center.y += hp.plateau_height;
        for (auto& a : world_.aabbs)   { a.min.y += hp.plateau_height;
                                          a.max.y += hp.plateau_height; }
        // Lift player's default spawn point above the new ground.
        player_.position.y += hp.plateau_height;

        // Grass: scatter blades on the heightmap. Acceptance band +
        // slope test mean only the grass-coloured layer of the
        // terrain (low altitude, gentle slope) gets covered. Cap at
        // 200k blades for a healthy density without bloating VRAM.
        GrassParams gp{};
        // Cover the plateau too вЂ” slopes, valleys, and the plateau
        // courtyard around the castle. Castle stones are excluded by
        // the keep-out rectangle below so blades never poke through
        // the brushes themselves.
        // Placement bounds are GENEROUS вЂ” the runtime altitude
        // sliders (rt_.grass_alt_min/max) shape the visible band
        // without re-baking. Cover most reasonable terrain heights.
        gp.height_min = -20.0f;
        gp.height_max = 200.0f;
        // Concentrate blades in a 200m square around the castle so the
        // density is high (~12 blades/mВІ) where the player actually
        // sees grass. Spreading them across the full 1km map gave
        // ~0.5/mВІ (about a metre between blades вЂ” looks bald).
        gp.half_extent = 200.0f;
        // Inner keep is ~6m square at origin. Keep blades out of the
        // keep interior only; the outer courtyard between keep and
        // perimeter walls gets grass too. Blades whose footprint
        // overlaps a brush stay invisible вЂ” the wall geometry occludes
        // them from outside views.
        gp.keep_out_xz = glm::vec2(4.0f, 4.0f);
        // 800k placed blades: density slider 0..4 maps to render
        // fraction (density / 4) Г— placed, so density=1.0 в†’ 200k,
        // density=4.0 в†’ 800k. Earlier 2M cap meant density=4 fired
        // ~10M sun shadow rays/frame (5 verts each) and pushed the
        // GPU into TDR. 800k keeps the worst case at ~4M rays which
        // the 4080 handles comfortably even with the rest of the
        // RT load (terrain shadows, AO, GI).
        gp.max_blades  = 800000;
        grass_ = build_grass(device_, allocator_,
                             graphics_queue_, graphics_queue_family_,
                             hm, gp);
    }

    physics_ = std::make_unique<PhysicsWorld>();
    // Batch-add the level brushes вЂ” Jolt's AddBodiesPrepare/Finalize takes
    // the broadphase mutex once for the whole pile instead of per body.
    // For 150+ static brushes (the castle) this is the difference between
    // ~30 ms and ~3 ms at level load.
    std::vector<PhysicsWorld::StaticBox> sboxes;
    sboxes.reserve(world_.brushes.size());
    for (const auto& b : world_.brushes) {
        sboxes.push_back({b.center, b.size * 0.5f});
    }
    physics_->add_static_boxes(sboxes.data(), sboxes.size());
    // Heightfield collision for terrain. Player + crates collide cleanly
    // against the bumpy ground. Jolt requires the sample-count side to be
    // divisible by some block_size in [2, 8]; with dim=2048 (sample_count
    // = 2049, odd) we'd violate that, so we copy the first 2048Г—2048 sub-
    // grid into a packed buffer and pass that with dim=2047 (sample_count
    // = 2048, block_size 8 вњ“). Player can't reach the world edge so the
    // single-cell shrinkage is invisible.
    if (!terrain_data_.heights.empty()) {
        const int W = terrain_data_.dim + 1;        // 2049
        const int physics_samples = (W % 2 == 0) ? W : (W - 1);  // 2048
        std::vector<float> jolt_heights;
        if (physics_samples == W) {
            // Already even вЂ” pass through.
            physics_->add_static_heightfield(
                terrain_data_.heights.data(), terrain_data_.dim,
                glm::vec2(terrain_data_.origin_x, terrain_data_.origin_z),
                terrain_data_.cell);
        } else {
            jolt_heights.resize(static_cast<size_t>(physics_samples) *
                                 static_cast<size_t>(physics_samples));
            for (int z = 0; z < physics_samples; ++z) {
                std::memcpy(jolt_heights.data() +
                                static_cast<size_t>(z) *
                                static_cast<size_t>(physics_samples),
                            terrain_data_.heights.data() +
                                static_cast<size_t>(z) * static_cast<size_t>(W),
                            static_cast<size_t>(physics_samples) * sizeof(float));
            }
            physics_->add_static_heightfield(
                jolt_heights.data(), physics_samples - 1,
                glm::vec2(terrain_data_.origin_x, terrain_data_.origin_z),
                terrain_data_.cell);
        }
    }

    log::infof("Jolt: %d static bodies; dynamic boxes will spawn over time "
               "(max %d, every %.2fs)",
               physics_->body_count(),
               kMaxDynProps, kSpawnInterval);
}

void VulkanEngine::init_skybox() {
    // EXR first вЂ” keeps real sun radiance so the bloom pass naturally produces
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
        log::info("[skybox] no asset loaded вЂ” creating 1x1 placeholder so the "
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
        // Real asset loaded вЂ” sync the directional-light angles to the
        // brightest pixel in the panorama so cast shadows match the visible
        // sun.
        glm::vec3 d = glm::normalize(skybox_.sun_direction);
        float pitch_deg = glm::degrees(std::asin(glm::clamp(d.y, -1.0f, 1.0f)));
        float yaw_deg = glm::degrees(std::atan2(d.x, d.z));
        rt_.sun_pitch_deg = pitch_deg;
        rt_.sun_yaw_deg = yaw_deg;
        log::infof("[skybox] sun synced to UI: pitch=%.1fВ° yaw=%.1fВ°",
                   pitch_deg, yaw_deg);
    }
}

void VulkanEngine::destroy_skybox_resources() {
    destroy_skybox(device_, allocator_, skybox_);
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
        // texture.cpp generates вЂ” eliminates far-distance moirГ© and halves
        // average sample bandwidth on receding floors/walls.
        si.maxLod = VK_LOD_CLAMP_NONE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &texture_sampler_),
                 "texture sampler");
    }

    auto probe = [&](const std::string& tail, VkFormat fmt) -> TextureSlot {
        const std::string roots[] = {
            "assets/tex/" + tail,
            "../assets/tex/" + tail,
            "../../assets/tex/" + tail,
        };
        for (const std::string& p : roots) {
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

    struct Spec {
        const char* name;
        const char* color_jpg;
        const char* normal_jpg;
    };
    // Index в†’ category. Indices ALSO match what level.cpp / spawn_random_box
    // pass into Brush::tex_albedo, so reordering here means reordering there.
    const Spec specs[kTextureCount] = {
        { "Ground054",         "Ground054/Ground054_2K-JPG_Color.jpg",
                               "Ground054/Ground054_2K-JPG_NormalGL.jpg" },
        { "Bricks067",         "Bricks067/Bricks067_2K-JPG_Color.jpg",
                               "Bricks067/Bricks067_2K-JPG_NormalGL.jpg" },
        { "Wood048",           "Wood048/Wood048_2K-JPG_Color.jpg",
                               "Wood048/Wood048_2K-JPG_NormalGL.jpg" },
        { "Metal042A",         "Metal042A/Metal042A_2K-JPG_Color.jpg",
                               "Metal042A/Metal042A_2K-JPG_NormalGL.jpg" },
        { "PaintedPlaster017", "PaintedPlaster017/PaintedPlaster017_2K-JPG_Color.jpg",
                               "PaintedPlaster017/PaintedPlaster017_2K-JPG_NormalGL.jpg" },
    };

    for (int i = 0; i < kTextureCount; ++i) {
        // Albedos go through an sRGB sampler so the GPU does the gamma decode
        // automatically; normals stay UNORM (linear-space, already in [0,1]).
        albedo_textures_[i] = probe(specs[i].color_jpg,  VK_FORMAT_R8G8B8A8_SRGB);
        normal_textures_[i] = probe(specs[i].normal_jpg, VK_FORMAT_R8G8B8A8_UNORM);
    }
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
    if (texture_sampler_) {
        vkDestroySampler(device_, texture_sampler_, nullptr);
        texture_sampler_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::init_viewmodel() {
    // Procedural placeholder gun: a barrel (cylinder) + body (box) + grip
    // (box). Camera-space coords: +X right, +Y up, -Z forward.
    auto rotX = [](float deg) {
        return glm::rotate(glm::mat4(1.0f), glm::radians(deg),
                           glm::vec3(1.0f, 0.0f, 0.0f));
    };
    auto T = [](glm::vec3 t) { return glm::translate(glm::mat4(1.0f), t); };
    auto S = [](glm::vec3 s) { return glm::scale(glm::mat4(1.0f), s); };

    const glm::vec4 metal(0.32f, 0.34f, 0.40f, 1.0f);
    const glm::vec4 dark (0.16f, 0.17f, 0.20f, 1.0f);
    const glm::vec4 grip (0.18f, 0.13f, 0.10f, 1.0f);

    constexpr float gx = 0.18f;
    constexpr float gy = -0.18f;
    constexpr float body_hz = 0.16f;
    constexpr float body_zc = -0.32f;
    constexpr float barrel_hl = 0.12f;
    constexpr float barrel_zc = body_zc - body_hz - barrel_hl;

    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy, body_zc)) * S(glm::vec3(0.045f, 0.045f, body_hz)),
        metal,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cylinder,
        T(glm::vec3(gx, gy, barrel_zc)) * rotX(90.0f) *
            S(glm::vec3(0.040f, barrel_hl, 0.040f)),
        dark,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy + 0.050f, body_zc)) *
            S(glm::vec3(0.018f, 0.005f, body_hz * 0.55f)),
        dark,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy + 0.040f, barrel_zc - barrel_hl + 0.01f)) *
            S(glm::vec3(0.006f, 0.012f, 0.006f)),
        dark,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy - 0.10f, body_zc + 0.06f)) *
            S(glm::vec3(0.038f, 0.075f, 0.045f)),
        grip,
    });

    GltfModel gltf = load_gltf("assets/gun.glb");
    if (!gltf.primitives.empty()) {
        glm::vec3 center = (gltf.aabb_min + gltf.aabb_max) * 0.5f;
        glm::vec3 extent = gltf.aabb_max - gltf.aabb_min;
        float max_side = std::max({extent.x, extent.y, extent.z, 1e-3f});
        float scale = 0.30f / max_side;

        viewmodel_root_offset_ =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.16f, -0.20f, -0.35f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(scale)) *
            glm::translate(glm::mat4(1.0f), -center);

        for (const auto& prim : gltf.primitives) {
            ViewmodelMesh vm{};
            vm.mesh = create_mesh_from_data(device_, allocator_, graphics_queue_,
                                            graphics_queue_family_,
                                            prim.vertices.data(),
                                            static_cast<uint32_t>(prim.vertices.size()),
                                            prim.indices.data(),
                                            static_cast<uint32_t>(prim.indices.size()));
            vm.base_color = prim.base_color;
            viewmodel_gltf_.push_back(std::move(vm));
        }
        log::infof("viewmodel: glTF loaded with %zu primitives (placeholder hidden)",
                   viewmodel_gltf_.size());
        viewmodel_proc_parts_.clear();
    } else {
        log::info("viewmodel: using procedural placeholder "
                  "(drop assets/gun.glb to override)");
    }
}

void VulkanEngine::destroy_viewmodel() {
    for (auto& vm : viewmodel_gltf_) destroy_mesh(allocator_, vm.mesh);
    viewmodel_gltf_.clear();
    viewmodel_proc_parts_.clear();
}

void VulkanEngine::draw_viewmodel(VkCommandBuffer cmd, const glm::mat4& vp,
                                  const glm::mat4& view_inv) {
    auto draw_part = [&](const Mesh& m, const glm::mat4& model,
                         const glm::vec4& color, const glm::vec3& emissive,
                         bool full_emissive) {
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, m.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        // Viewmodel is camera-attached вЂ” prev_mvp = mvp gives zero motion,
        // which is what we want for SVGF (the gun shouldn't smear when the
        // camera turns even though world points "moved").
        pc.prev_mvp = pc.mvp;
        pc.color = color;
        pc.emissive = glm::vec4(emissive, full_emissive ? 1.0f : 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, m.index_count, 1, 0, 0, 0);
    };

    // Recoil: shift the entire viewmodel back along camera +Z (toward the
    // camera) by the kick curve. Camera looks down -Z, so positive +Z = back.
    float kick = recoil_kick(recoil_timer_, kRecoilDuration, kRecoilStroke);
    glm::mat4 recoil_offset = glm::translate(glm::mat4(1.0f),
                                             glm::vec3(0.0f, 0.0f, kick));
    glm::mat4 view_inv_recoil = view_inv * recoil_offset;

    if (!viewmodel_gltf_.empty()) {
        glm::mat4 root = view_inv_recoil * viewmodel_root_offset_;
        for (const auto& vm : viewmodel_gltf_) {
            draw_part(vm.mesh, root, vm.base_color, glm::vec3(0.0f), false);
        }
        return;
    }

    for (const auto& p : viewmodel_proc_parts_) {
        glm::mat4 world = view_inv_recoil * p.local;
        const Mesh& mesh = (p.kind == ViewmodelPart::Kind::Cube)
                              ? cube_mesh_ : cylinder_mesh_;
        draw_part(mesh, world, p.color, glm::vec3(0.0f), false);
    }

    // Muzzle flash: a bright, short-lived emissive blob just past the muzzle
    // tip. Bloom downstream halos it dramatically. Anchored so the
    // back face of the (scale-animated) cube always sits past the
    // barrel tip вЂ” without this, the small fade-in/out cube floats
    // forward of the tip because scale shrinks but center stays fixed.
    if (muzzle_flash_timer_ > 0.0f) {
        const float barrel_tip_z = -0.72f;  // matches barrel_zc - barrel_hl in init
        float t = std::min(1.0f, muzzle_flash_timer_ / kMuzzleFlashDuration);
        float intensity = t * t;
        float core_size = 0.04f * (0.6f + 0.4f * t);   // full cube size
        // Center the cube so its back face (toward camera) is exactly at
        // the barrel tip independent of size: center_z = tip - half_size.
        float core_z = barrel_tip_z - core_size * 0.5f;
        glm::vec3 muzzle(0.18f, -0.18f, core_z);
        glm::mat4 base = glm::translate(glm::mat4(1.0f), muzzle);

        glm::mat4 core = view_inv_recoil * base *
                         glm::scale(glm::mat4(1.0f), glm::vec3(core_size));
        glm::vec3 core_emi = glm::vec3(8.0f, 6.5f, 3.0f) * intensity;
        draw_part(cube_mesh_, core, glm::vec4(1.0f, 0.85f, 0.45f, 1.0f),
                  core_emi, true);

        glm::mat4 rotX90 = glm::rotate(glm::mat4(1.0f),
                                       glm::radians(90.0f),
                                       glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 tongue = view_inv_recoil * base *
                           glm::translate(glm::mat4(1.0f),
                                          glm::vec3(0.0f, 0.0f, -0.06f * t)) *
                           rotX90 *
                           glm::scale(glm::mat4(1.0f),
                                      glm::vec3(0.025f, 0.06f * t, 0.025f));
        glm::vec3 tongue_emi = glm::vec3(6.0f, 4.5f, 1.6f) * intensity;
        draw_part(cylinder_mesh_, tongue, glm::vec4(1.0f, 0.78f, 0.30f, 1.0f),
                  tongue_emi, true);
    }
}

void VulkanEngine::rebuild_dyn_render_cache() {
    // Resize without reset so we can reuse cached transforms for sleeping
    // bodies.
    dyn_render_cache_.resize(dyn_props_.size());
    if (!physics_) return;
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        DynRender& dr = dyn_render_cache_[i];
        const uint32_t body = dyn_props_[i].body_id;
        const auto handle = dyn_props_[i].jolt_handle;
        const bool slot_matches = dr.valid && dr.body_id == body;
        // Reuse cached transform if the slot still maps to the same body AND
        // Jolt has put it to sleep.
        if (slot_matches && !physics_->is_body_active_h(handle)) {
            // Sleeping в†’ no motion this frame. Drag prev_world forward.
            dr.prev_world = dr.world;
            continue;
        }
        // Capture the prior world before we overwrite it.
        glm::mat4 captured_prev = slot_matches ? dr.world : glm::mat4(1.0f);
        if (!physics_->get_body_world_matrix_h(handle, dr.world)) {
            dr.valid = false;
            dr.body_id = 0;
            continue;
        }
        if (!slot_matches) captured_prev = dr.world;
        dr.prev_world = captured_prev;
        glm::vec3 he = dyn_props_[i].full_size * 0.5f;
        glm::vec3 mn(std::numeric_limits<float>::max());
        glm::vec3 mx(std::numeric_limits<float>::lowest());
        for (int j = 0; j < 8; ++j) {
            glm::vec4 c((j & 1) ? he.x : -he.x,
                        (j & 2) ? he.y : -he.y,
                        (j & 4) ? he.z : -he.z, 1.0f);
            glm::vec3 wc(dr.world * c);
            mn = glm::min(mn, wc);
            mx = glm::max(mx, wc);
        }
        dr.valid = true;
        dr.body_id = body;
        dr.aabb_min = mn;
        dr.aabb_max = mx;
    }
}

void VulkanEngine::rebuild_tick_aabbs() {
    // Static level + every dynamic box's world-space AABB envelope. With
    // rotated boxes we use the AABB of the rotated cube (over-approximation)
    // вЂ” the player's collision is still solid against tilted falling crates.
    tick_aabbs_.clear();
    tick_aabbs_.reserve(world_.aabbs.size() + dyn_props_.size());
    for (const auto& a : world_.aabbs) tick_aabbs_.push_back(a);
    if (!physics_) return;
    if (dyn_tick_aabb_cache_.size() != dyn_props_.size()) {
        dyn_tick_aabb_cache_.resize(dyn_props_.size());
    }
    for (size_t idx = 0; idx < dyn_props_.size(); ++idx) {
        const auto& dp = dyn_props_[idx];
        DynTickAabb& slot = dyn_tick_aabb_cache_[idx];
        // Slot identity check: the cached entry is only valid if it was
        // populated for THIS body. FIFO eviction (front-erase + push_back)
        // shifts every other slot down by 1, so a slot's body identity can
        // change without dyn_props_.size() changing. Without this guard the
        // stale cached AABB describes a now-removed box and the player
        // walks through the new occupant.
        const bool slot_matches = slot.body_id == dp.body_id;
        const bool active = physics_->is_body_active_h(dp.jolt_handle);
        if (slot_matches && !active) {
            // Sleeping body in a slot we already populated в†’ AABB hasn't
            // moved since last compute, reuse.
            tick_aabbs_.push_back(slot.aabb);
            continue;
        }
        glm::mat4 m;
        if (!physics_->get_body_world_matrix_h(dp.jolt_handle, m)) {
            slot.body_id = 0;   // mark invalid
            continue;
        }
        glm::vec3 he = dp.full_size * 0.5f;
        glm::vec3 mn(std::numeric_limits<float>::max());
        glm::vec3 mx(std::numeric_limits<float>::lowest());
        for (int i = 0; i < 8; ++i) {
            glm::vec4 c((i & 1) ? he.x : -he.x,
                        (i & 2) ? he.y : -he.y,
                        (i & 4) ? he.z : -he.z, 1.0f);
            glm::vec3 wc(m * c);
            mn = glm::min(mn, wc);
            mx = glm::max(mx, wc);
        }
        slot.aabb = {mn, mx};
        slot.body_id = dp.body_id;
        tick_aabbs_.push_back(slot.aabb);
    }
}

void VulkanEngine::spawn_random_box() {
    if (!physics_) return;

    // Spawn around the castle: XY/Z spread covers the keep + courtyard;
    // height seeds well above the highest tower so crates fall onto the
    // walkway / merlons / outside the perimeter.
    //   - terrain plateau ~y22, castle interior ~y22..27, towers ~y32
    //   - spawn 38..50m so crates have airtime to scatter realistically.
    const float plateau_y = terrain_data_.heights.empty() ? 0.0f : 22.0f;
    glm::vec3 pos(frand_range(spawn_rng_state_, -16.0f, 16.0f),
                  plateau_y + frand_range(spawn_rng_state_, 16.0f, 28.0f),
                  frand_range(spawn_rng_state_, -16.0f, 16.0f));

    glm::vec3 euler(frand_range(spawn_rng_state_, -1.2f, 1.2f),
                    frand_range(spawn_rng_state_, -1.2f, 1.2f),
                    frand_range(spawn_rng_state_, -1.2f, 1.2f));

    float side = frand_range(spawn_rng_state_, 0.30f, 0.75f);
    glm::vec3 he(side, side * frand_range(spawn_rng_state_, 0.7f, 1.3f), side);

    static const glm::vec3 palette[] = {
        {1.00f, 0.40f, 0.40f}, {0.40f, 1.00f, 0.40f}, {0.40f, 0.55f, 1.00f},
        {1.00f, 1.00f, 0.30f}, {1.00f, 0.50f, 1.00f}, {0.40f, 1.00f, 1.00f},
        {1.00f, 0.70f, 0.30f}, {0.70f, 0.30f, 0.85f}, {0.85f, 0.85f, 0.85f},
    };
    constexpr int kPaletteCount = sizeof(palette) / sizeof(palette[0]);
    glm::vec3 col = palette[xorshift32(spawn_rng_state_) % kPaletteCount];

    uint32_t id = physics_->add_dynamic_box(pos, he, euler);
    if (id == 0) return;
    DynamicProp p{};
    p.body_id = id;
    p.jolt_handle = physics_->handle_of(id);
    p.full_size = he * 2.0f;
    p.fallback_color = glm::vec4(col, 1.0f);
    int tex = 1 + static_cast<int>(xorshift32(spawn_rng_state_) % 4u);
    p.tex_albedo = tex;
    p.tex_normal = tex;
    p.uv_scale = 1.5f;
    p.color = glm::vec4(glm::mix(glm::vec3(0.9f), col, 0.4f), 1.0f);
    dyn_props_.push_back(p);

    // FIFO eviction: drop the oldest while we exceed the cap.
    while (static_cast<int>(dyn_props_.size()) > kMaxDynProps) {
        physics_->remove_body(dyn_props_.front().body_id);
        dyn_props_.erase(dyn_props_.begin());
    }
}

void VulkanEngine::render_world_depth_pass(VkCommandBuffer cmd) {
    // Mirror render_world()'s view setup. We don't try to share state across
    // the two passes because that would require threading FrameView through
    // both, and the saving is in the noise vs cube.frag's inline-RT cost
    // which the prepass exists to skip.
    float aspect = static_cast<float>(render_extent_.width) /
                   static_cast<float>(render_extent_.height);

    constexpr float kFixedDt = 1.0f / 120.0f;
    float alpha = physics_accumulator_ / kFixedDt;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    glm::vec3 render_pos = glm::mix(prev_player_position_, player_.position, alpha);

    glm::vec3 saved = player_.position;
    player_.position = render_pos;
    glm::mat4 view = player_.view_matrix();
    player_.position = saved;
    // Far plane: 1500m. Tradeoff between two artefacts:
    //   - too short (e.g. 500m): visible terrain disc is asymmetric
    //     because the corner ray at 80В° FOV reaches further world
    //     distance than the centre, swept across the screen on turn;
    //   - too long  (e.g. 3000m): float32 depth precision at the
    //     silhouette of steep distant peaks runs out of bits and
    //     adjacent triangles z-fight, looking "see-through".
    // 1500m + atmospheric fog reaching ~95% sky tint by ~1.3km keeps
    // the asymmetry hidden under fog AND keeps depth precision tight
    // enough that distant cliffs are stable.
    glm::mat4 proj = glm::perspective(glm::radians(80.0f), aspect, 0.05f, 1500.0f);
    proj[1][1] *= -1.0f;

    // Match the color-pass jitter exactly so depth-prepass and color-pass
    // depth values are bit-identical (LESS_OR_EQUAL passes the equal-depth
    // pixel; if jitter mismatched, the color pass would mostly fail the test
    // and the screen would go black).
    if (rt_.taa_jitter_enabled) {
        auto halton = [](int b, int i) {
            float f = 1.0f, r = 0.0f;
            while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
            return r;
        };
        int idx = static_cast<int>(frame_number_ % 16) + 1;
        float jx = (halton(2, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.width);
        float jy = (halton(3, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.height);
        proj[2][0] += jx;
        proj[2][1] += jy;
    }

    glm::mat4 vp = proj * view;
    Frustum frustum = extract_frustum(vp);

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(render_extent_.width);
    vp_state.height = static_cast<float>(render_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    auto push_depth = [&](const glm::mat4& model) {
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) continue;
        glm::mat4 model = glm::translate(glm::mat4(1.0f), b.center) *
                          glm::scale(glm::mat4(1.0f), b.size);
        push_depth(model);
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) continue;
        glm::mat4 model = dr.world * glm::scale(glm::mat4(1.0f), dyn_props_[i].full_size);
        push_depth(model);
    }
    // Terrain: distance-LOD per chunk. Pre-pass and color pass MUST
    // pick the same LOD AND morph factor per chunk so the rasterised
    // depth values match вЂ” otherwise the LESS_OR_EQUAL test in the
    // color pass would discard fragments. We bind terrain_depth_pipeline_
    // (same terrain.vert as the color pass, just routed to depth-only).
    if (!terrain_chunks_.chunks.empty()) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_depth_pipeline_);
        glm::vec3 cam = player_.eye_position();
        for (const auto& c : terrain_chunks_.chunks) {
            if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
            int lod = pick_terrain_lod(c, cam, rt_.terrain_lod1 * rt_.terrain_lod_scale, rt_.terrain_lod2 * rt_.terrain_lod_scale, rt_.terrain_lod3 * rt_.terrain_lod_scale);
            float morph = pick_terrain_morph(c, cam, lod, rt_.terrain_lod1 * rt_.terrain_lod_scale);
            VkBuffer vbufs[2] = { c.mesh.vertex_buffer, c.parent_y_buffer };
            VkDeviceSize voffs[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
            VkBuffer ibo = (lod == 0) ? c.mesh.index_buffer : c.ibo_lod[lod - 1];
            vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
            PushConstants pc{};
            pc.mvp = vp;
            pc.model = glm::mat4(1.0f);
            pc.color = glm::vec4(1.0f, 1.0f, 1.0f, morph);
            vkCmdPushConstants(cmd, pipeline_layout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, c.index_count_lod[lod], 1, 0, 0, 0);
        }
        // Caller binds depth_pipeline_ before invoking us; it will
        // continue using that for any non-terrain post-terrain draws.
        // Re-bind so the rest of the depth pre-pass goes through it.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline_);
    }
}

void VulkanEngine::render_world(VkCommandBuffer cmd) {
    float aspect = static_cast<float>(render_extent_.width) /
                   static_cast<float>(render_extent_.height);

    // Render at the interpolated position between the last two physics ticks,
    // so visuals are smooth even when tick rate != render rate.
    constexpr float kFixedDt = 1.0f / 120.0f;
    float alpha = physics_accumulator_ / kFixedDt;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    glm::vec3 render_pos = glm::mix(prev_player_position_, player_.position, alpha);

    glm::vec3 saved = player_.position;
    player_.position = render_pos;
    glm::mat4 view = player_.view_matrix();
    player_.position = saved;
    // Far plane: 1500m. Tradeoff between two artefacts:
    //   - too short (e.g. 500m): visible terrain disc is asymmetric
    //     because the corner ray at 80В° FOV reaches further world
    //     distance than the centre, swept across the screen on turn;
    //   - too long  (e.g. 3000m): float32 depth precision at the
    //     silhouette of steep distant peaks runs out of bits and
    //     adjacent triangles z-fight, looking "see-through".
    // 1500m + atmospheric fog reaching ~95% sky tint by ~1.3km keeps
    // the asymmetry hidden under fog AND keeps depth precision tight
    // enough that distant cliffs are stable.
    glm::mat4 proj = glm::perspective(glm::radians(80.0f), aspect, 0.05f, 1500.0f);
    proj[1][1] *= -1.0f;

    // Sub-pixel Halton jitter вЂ” TAA integrates these offsets into a super-
    // sampled-looking result across ~16 frames.
    if (rt_.taa_jitter_enabled) {
        auto halton = [](int b, int i) {
            float f = 1.0f, r = 0.0f;
            while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
            return r;
        };
        int idx = static_cast<int>(frame_number_ % 16) + 1;
        float jx = (halton(2, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.width);
        float jy = (halton(3, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.height);
        proj[2][0] += jx;
        proj[2][1] += jy;
    }

    last_view_proj_ = proj * view;
    glm::mat4 vp = last_view_proj_;
    Frustum frustum = extract_frustum(vp);

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(render_extent_.width);
    vp_state.height = static_cast<float>(render_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);

    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const bool tex_on = rt_.textures_enabled;
    // prev_view_proj_ is captured at the END of each frame from the previous
    // frame's last_view_proj_; on frame 0 it's identity (which produces a
    // single-frame motion-vector glitch вЂ” TAA's "history_valid" gate handles
    // the invalid-history first frame, so this doesn't surface visually).
    const glm::mat4 prev_vp = prev_view_proj_;
    auto draw_brush = [&](const glm::mat4& model, const glm::mat4& prev_model,
                          const glm::vec4& color,
                          const glm::vec3& emissive, bool full_emissive,
                          int tex_albedo, int tex_normal, float uv_scale,
                          bool object_space) {
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        pc.prev_mvp = prev_vp * prev_model;
        pc.color = color;
        pc.emissive = glm::vec4(emissive, full_emissive ? 1.0f : 0.0f);
        if (tex_on) {
            pc.tex_params = glm::vec4(static_cast<float>(tex_albedo),
                                      static_cast<float>(tex_normal),
                                      uv_scale, object_space ? 1.0f : 0.0f);
        } else {
            pc.tex_params = glm::vec4(-1.0f, -1.0f, uv_scale, 0.0f);
        }
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    int culled_static = 0, culled_dyn = 0, drawn_static = 0, drawn_dyn = 0;
    // Terrain pass вЂ” chunked. Each visible chunk is drawn at full LOD
    // when within `near_lod` metres of the camera, otherwise half LOD.
    // The merged-static-BLAS-style "single big draw" is reserved for the
    // RT path; for raster we want LOD + frustum cull per chunk so the
    // 2km terrain stays cheap from any viewpoint.
    if (!terrain_chunks_.chunks.empty()) {
        // Switch to the terrain-specific pipeline (terrain.vert + cube.frag,
        // 2 vertex bindings вЂ” pos/norm/uv + parent_y) so we can morph
        // between LOD 0 and LOD 1 in the vertex shader. Skirts still hide
        // LOD-mismatch cracks at higher LOD seams.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_pipeline_);
        glm::vec3 cam = player_.eye_position();
        for (const auto& c : terrain_chunks_.chunks) {
            if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
            int lod = pick_terrain_lod(c, cam, rt_.terrain_lod1 * rt_.terrain_lod_scale, rt_.terrain_lod2 * rt_.terrain_lod_scale, rt_.terrain_lod3 * rt_.terrain_lod_scale);
            float morph = pick_terrain_morph(c, cam, lod, rt_.terrain_lod1 * rt_.terrain_lod_scale);
            VkBuffer vbufs[2] = { c.mesh.vertex_buffer, c.parent_y_buffer };
            VkDeviceSize voffs[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
            VkBuffer ibo = (lod == 0) ? c.mesh.index_buffer : c.ibo_lod[lod - 1];
            vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
            PushConstants pc{};
            pc.mvp = vp;
            pc.model = glm::mat4(1.0f);
            pc.prev_mvp = prev_vp;
            // .w doubles as the morph factor вЂ” terrain.vert lerps Y by it.
            pc.color = glm::vec4(1.0f, 1.0f, 1.0f, morph);
            pc.emissive = glm::vec4(0.0f);
            pc.tex_params = tex_on
                ? glm::vec4(0.0f, 0.0f, 16.0f, 2.0f)
                : glm::vec4(-1.0f, -1.0f, 16.0f, 2.0f);
            vkCmdPushConstants(cmd, pipeline_layout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, c.index_count_lod[lod], 1, 0, 0, 0);
        }
        // Switch back to the cube pipeline + cube mesh for the brush loop.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    // Grass вЂ” instanced billboard blades. Single draw call covers
    // every blade; the grass.vert collapses out-of-range blades to
    // a degenerate triangle (NaN clip space) so no fragment work runs
    // for them. Skipped entirely when disabled in Settings or when
    // build_grass produced an empty set.
    if (rt_.grass_enabled && grass_.instance_count > 0 &&
        grass_pipeline_ != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grass_pipeline_);
        // descriptor set is the same scene set, already bound at the
        // top of render_world; no need to rebind.
        VkBuffer  bufs[2]    = { grass_.blade_mesh.vertex_buffer,
                                  grass_.instance_buffer };
        VkDeviceSize offs[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, bufs, offs);
        vkCmdBindIndexBuffer(cmd, grass_.blade_mesh.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        // Grass push constants вЂ” only grass_params is read by grass.vert,
        // the rest are dummies. Time is just the frame number scaled to
        // a sane wind period.
        PushConstants gpc{};
        gpc.mvp = vp;
        gpc.model = glm::mat4(1.0f);
        gpc.prev_mvp = prev_vp;
        gpc.color = glm::vec4(1.0f);
        gpc.emissive = glm::vec4(0.0f);
        gpc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
        float t = static_cast<float>(frame_number_) * 0.016f;
        gpc.grass_params = glm::vec4(rt_.grass_distance, rt_.grass_wind, t, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &gpc);
        // Density slider goes 0..4. We map (density / 4.0) onto the
        // [0, total_placed] range so density=1.0 gives 25% of the
        // placed blades (the original visual default after the 2M
        // bump) and density=4.0 unleashes the whole pile. Halton
        // placement order means any prefix is a uniformly sparser
        // sampling of the field.
        const float frac = std::clamp(rt_.grass_density, 0.0f, 4.0f) * 0.25f;
        uint32_t inst_n = static_cast<uint32_t>(
            static_cast<float>(grass_.instance_count) * frac);
        if (inst_n > 0) {
            vkCmdDrawIndexed(cmd, grass_.blade_mesh.index_count,
                             inst_n, 0, 0, 0);
        }
        // Restore the world pipeline + cube mesh bindings for the
        // brush loop below.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) { ++culled_static; continue; }
        glm::mat4 model = glm::translate(glm::mat4(1.0f), b.center) *
                          glm::scale(glm::mat4(1.0f), b.size);
        glm::vec4 brush_base = tex_on ? b.color : b.fallback_color;
        // Static brushes don't move вЂ” prev_model = current.
        draw_brush(model, model, brush_base, b.emissive, b.full_emissive,
                   b.tex_albedo, b.tex_normal, b.uv_scale, false);
        ++drawn_static;
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) { ++culled_dyn; continue; }
        const glm::mat4 scale_m = glm::scale(glm::mat4(1.0f), dyn_props_[i].full_size);
        glm::mat4 model = dr.world * scale_m;
        glm::mat4 prev_model = dr.prev_world * scale_m;
        glm::vec4 dyn_base = tex_on
            ? dyn_props_[i].color
            : dyn_props_[i].fallback_color;
        draw_brush(model, prev_model, dyn_base, glm::vec3(0.0f), false,
                   dyn_props_[i].tex_albedo, dyn_props_[i].tex_normal,
                   dyn_props_[i].uv_scale, /*object_space=*/true);
        ++drawn_dyn;
    }

    // Bullet-impact decals вЂ” drawn AFTER static + dyn brushes so they sit
    // on top of the wall textures, BEFORE sparks so the spark glow
    // overlaps the scorch mark visually. Uses the still-bound cube mesh.
    if (!decals_.empty()) {
        draw_decals(cmd, vp);
    }

    // Hit sparks (raster path).
    if (physics_ && !particles_.empty()) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        draw_spark_trails(cmd, vp);
    }
    // Shadow debug overlay (gated by rt_.shadow_debug_overlay).
    draw_shadow_debug(cmd, vp);

    // Projectiles: switch to cylinder mesh, draw each in-flight bullet.
    if (!projectiles_.empty() && physics_) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        for (const auto& p : projectiles_) {
            glm::mat4 world;
            if (!physics_->get_body_world_matrix(p.body_id, world)) continue;
            glm::vec3 pos(world[3]);
            glm::vec3 vel = physics_->get_linear_velocity(p.body_id);
            glm::vec3 dir = glm::length(vel) > 1e-3f ? glm::normalize(vel) : p.initial_dir;
            glm::mat4 model = align_local_y_to(pos, dir) *
                              glm::scale(glm::mat4(1.0f),
                                         glm::vec3(p.radius, p.half_length, p.radius));
            PushConstants pc{};
            pc.mvp = vp * model;
            pc.model = model;
            // Projectiles don't track a prev_pose yet вЂ” zero-motion approx.
            pc.prev_mvp = pc.mvp;
            pc.color = p.color;
            pc.emissive = glm::vec4(p.emissive, 1.0f);  // full_emissive
            vkCmdPushConstants(cmd, pipeline_layout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
        }
    }

    // Viewmodel: rendered last so it overdraws world geometry only when its
    // depth wins. Its depth values are tiny (close to camera), so it sits in
    // front of everything in practice.
    draw_viewmodel(cmd, vp, glm::inverse(view));

    last_draw_static_ = drawn_static;
    last_draw_dyn_    = drawn_dyn;
    last_culled_      = culled_static + culled_dyn;
}

} // namespace qlike


