// Voxel buildings — Session A engine plumbing.
//
// Builds a single procedural voxel tower 100 m north of the castle origin,
// uploads its brickmap to GPU (brick atlas SSBO + shape directory SSBO),
// and ray-marches it during the main world MRT pass via a dedicated
// pipeline (voxel.vert + voxel.frag). Fully isolated from the existing
// scene descriptor set / cube pipeline — separate set layout, separate
// pipeline layout, separate push-constant range.
//
// The voxel pass writes to scene_color + motion_vec + depth, the same
// MRT targets the world colour pass uses, so it composites naturally
// with brushes, terrain, grass, water. Depth is written via
// gl_FragDepth from the DDA hit, so SVGF / TAA / FSR3 can reproject
// voxel pixels.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"
#include "engine/voxel/voxel_world.h"
#include "engine/frustum.h"

#include <glm/gtc/matrix_transform.hpp>   // glm::translate for per-chunk model

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace qlike {

namespace {

// Mirrors the layout in voxel.{vert,frag} — keep in lock-step.
struct VoxelCameraUboData {
    glm::mat4 view_proj;
    glm::mat4 prev_view_proj;
    glm::vec4 camera_pos;
    glm::vec4 sun_dir;
    glm::vec4 sun_color;
    glm::vec4 ambient;
    glm::vec4 viewport;
    glm::vec4 pal[16];
};

// Mirrors voxel.{vert,frag}'s push-constant layout — keep in lock-step.
// Rigid transform is split into R + T so the fragment shader can build
// the world→local inverse analytically (transpose(R), -T) — much cheaper
// than a per-pixel mat4 inverse(). std430 mat3 occupies 3 vec4 columns
// (48 B), so the total size matches the previous mat4 layout exactly.
struct VoxelPushConstants {
    glm::vec4  R0;           // mat3 col 0 (xyz used, w pad)
    glm::vec4  R1;           // mat3 col 1
    glm::vec4  R2;           // mat3 col 2
    glm::vec4  T;            // xyz translation, w pad
    glm::vec4  dims_vs;      // xyz = shape extent (local), w = voxel size
    glm::ivec4 grid_dir;     // xyz = brick dims, w = directory base offset
};
static_assert(sizeof(VoxelPushConstants) == 96,
              "VoxelPC size — fits 128B PC limit");

// Atlas + directory growth budget. The tower starts with ~892 bricks /
// ~2.6k directory entries; chunks add per collapse. These caps cover many
// dozens of chunks without ever resizing the GPU buffers (which would
// require a queue idle). 4 000 bricks ≈ 18 MB host-mapped (BAR), 64 k
// directory entries = 256 KB — both fit comfortably.
constexpr uint32_t kVoxelAtlasCapacityBricks = 4000;
constexpr uint32_t kVoxelDirCapacityEntries  = 64 * 1024;

// Tower placement — 50 m north of the castle origin. Tower extends
// +X/+Y/+Z from the base corner; centre on (0, _, 50).
constexpr float kTowerCenterZ = 50.0f;

} // namespace

void VulkanEngine::init_voxel() {
    using namespace voxel;

    // ---- CPU side: build the procedural tower ----
    voxel_world_ = std::make_unique<VoxelWorld>();

    // 8×24×8 bricks → 12.8 × 38.4 × 12.8 m. Slimmer footprint than the
    // original 12×18×12 + a stepped narrower upper turret on top, so
    // the new total is taller AND thinner. add_procedural_tower owns
    // both layers; these constants only need to cover the BASE so the
    // ground-snap math here can anchor the bottom.
    constexpr float kTowerSizeX = 8.0f * kBrickSize;   // 12.8
    constexpr float kTowerSizeZ = 8.0f * kBrickSize;
    // Anchor the tower's base 2 m below sampled terrain at (0, kTowerCenterZ)
    // so the bottom row of voxels embeds in the ground rather than floating.
    // Sample the four corners of the footprint and take the min so the tower
    // doesn't levitate over a sloped patch.
    const float h0 = sample_terrain_height(-kTowerSizeX * 0.5f,
                                            kTowerCenterZ - kTowerSizeZ * 0.5f);
    const float h1 = sample_terrain_height( kTowerSizeX * 0.5f,
                                            kTowerCenterZ - kTowerSizeZ * 0.5f);
    const float h2 = sample_terrain_height(-kTowerSizeX * 0.5f,
                                            kTowerCenterZ + kTowerSizeZ * 0.5f);
    const float h3 = sample_terrain_height( kTowerSizeX * 0.5f,
                                            kTowerCenterZ + kTowerSizeZ * 0.5f);
    const float base_y = std::min(std::min(h0, h1), std::min(h2, h3)) - 2.0f;
    glm::vec3 base_corner(-kTowerSizeX * 0.5f, base_y,
                          kTowerCenterZ - kTowerSizeZ * 0.5f);
    const int shape_idx = voxel_world_->add_procedural_tower(base_corner);
    (void)shape_idx;

    const auto& bricks = voxel_world_->bricks();
    const auto& shapes = voxel_world_->shapes();
    if (shapes.empty() || bricks.empty()) {
        log::error("[voxel] no shape/brick data produced; skipping init");
        voxel_world_.reset();
        return;
    }
    const VoxelShape& s0 = shapes[0];

    // ---- GPU side: brick atlas SSBO (device-local via staging) ----
    auto create_device_ssbo = [&](VkDeviceSize bytes, const void* data,
                                  const char* tag,
                                  VkBuffer& out_buf, VmaAllocation& out_alloc) {
        // Staging buffer (host-visible).
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation staging_alloc = nullptr;
        {
            VkBufferCreateInfo bci{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr, .flags = 0,
                .size = bytes,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            };
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo ai{};
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                     &staging, &staging_alloc, &ai),
                     tag);
            std::memcpy(ai.pMappedData, data, static_cast<size_t>(bytes));
        }
        // Device-local destination.
        {
            VkBufferCreateInfo bci{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr, .flags = 0,
                .size = bytes,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            };
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                     &out_buf, &out_alloc, nullptr),
                     tag);
        }
        // Copy staging → device.
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            VkBufferCopy region{0, 0, bytes};
            vkCmdCopyBuffer(cb, staging, out_buf, 1, &region);
        });
        vmaDestroyBuffer(allocator_, staging, staging_alloc);
    };

    // Brick atlas is host-mapped (not staged) so carves + chunk extraction
    // can flush touched / appended bricks directly without command-buffer
    // round-trips. Sized to a generous capacity so future chunk growth
    // doesn't need a reallocation. Device-local-host-visible (ReBAR) when
    // available — 18 MB easily fits the guaranteed BAR window.
    const VkDeviceSize atlas_bytes =
        VkDeviceSize(kVoxelAtlasCapacityBricks) * sizeof(voxel::BrickPayload);
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = atlas_bytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &voxel_atlas_buffer_, &voxel_atlas_alloc_, &ai),
                 "voxel brick atlas");
        // Upload only the populated prefix; the rest stays uninitialised.
        const size_t init_bytes = bricks.size() * sizeof(voxel::BrickPayload);
        std::memcpy(ai.pMappedData, bricks.data(), init_bytes);
        voxel_atlas_mapped_   = ai.pMappedData;
        voxel_atlas_bytes_    = atlas_bytes;
        voxel_atlas_uploaded_ = static_cast<uint32_t>(bricks.size());
    }

    // Directory: a single FLAT global array of `uint entries[]`. The main
    // shape occupies the first N slots; each extracted chunk appends its
    // own segment at the current head. Host-mapped for cheap appends.
    const VkDeviceSize dir_bytes =
        VkDeviceSize(kVoxelDirCapacityEntries) * sizeof(uint32_t);
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = dir_bytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &voxel_dir_buffer_, &voxel_dir_alloc_, &ai),
                 "voxel shape directory");
        std::memcpy(ai.pMappedData, s0.directory.data(),
                    s0.directory.size() * sizeof(uint32_t));
        voxel_dir_mapped_   = ai.pMappedData;
        voxel_dir_bytes_    = dir_bytes;
        voxel_dir_uploaded_   = static_cast<uint32_t>(s0.directory.size());
        voxel_dir_next_shape_ = 1;   // shape 0 (main) is already uploaded
    }

    log::infof("[voxel] GPU pools: atlas cap=%llu KB (%u bricks live), "
               "dir cap=%llu KB (%u entries live)",
               (unsigned long long)(atlas_bytes / 1024),
               voxel_atlas_uploaded_,
               (unsigned long long)(dir_bytes  / 1024),
               voxel_dir_uploaded_);

    // ---- Camera UBO (host-mapped, per-frame memcpy) ----
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(VoxelCameraUboData),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &voxel_camera_ubo_, &voxel_camera_alloc_, &ai),
                 "voxel camera ubo");
        voxel_camera_mapped_ = ai.pMappedData;
    }

    // ---- Descriptor set layout ----
    {
        VkDescriptorSetLayoutBinding b[4]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        // Wall texture (shared with the castle brushes — same Bricks078
        // albedo at index 1 of the main pipeline's u_albedo array).
        b[3].binding = 3;
        b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[3].descriptorCount = 1;
        b[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &dlci, nullptr,
                                              &voxel_desc_set_layout_),
                 "voxel desc set layout");
    }

    // ---- Pipeline layout (set + push constants) ----
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = sizeof(VoxelPushConstants);
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &voxel_desc_set_layout_,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &voxel_pipeline_layout_),
                 "voxel pipeline layout");
    }

    // ---- Pipeline ----
    {
        std::string sd = QLIKE_SHADER_DIR;
        voxel_vert_module_ = vkpipe::load_shader_module(device_, sd + "/voxel.vert.spv");
        voxel_frag_module_ = vkpipe::load_shader_module(device_, sd + "/voxel.frag.spv");

        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = voxel_vert_module_;
        cfg.frag = voxel_frag_module_;
        cfg.layout = voxel_pipeline_layout_;
        cfg.color_formats = { scene_color_format_, motion_vec_format_,
                              gbuffer0_format_,    gbuffer1_format_ };
        cfg.depth_format = depth_format_;
        cfg.depth_test = true;
        cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
        // CULL_NONE: rasterize both front + back faces. CULL_FRONT
        // (back-only) halves fragment invocations but caused visible
        // dropouts on the far wall from some camera angles — likely
        // near-plane clipping of back-face geometry when the camera
        // approaches the AABB. Reverted; the cheap analytic inverse +
        // tighter max_steps cap recover most of the win.
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.alpha_blend_color0_only = false;
        // No vertex buffer — vertex shader derives corners from gl_VertexIndex.
        cfg.vbindings.clear();
        cfg.vattrs.clear();
        voxel_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    // ---- Descriptor pool + set + write ----
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         2 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 3, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr,
                                         &voxel_desc_pool_),
                 "voxel desc pool");

        VkDescriptorSetAllocateInfo dsai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = voxel_desc_pool_,
            .descriptorSetCount = 1, .pSetLayouts = &voxel_desc_set_layout_,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dsai, &voxel_desc_set_),
                 "voxel alloc desc set");

        VkDescriptorBufferInfo ubo_bi  { voxel_camera_ubo_,   0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo atlas_bi{ voxel_atlas_buffer_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dir_bi  { voxel_dir_buffer_,   0, VK_WHOLE_SIZE };
        // Bricks078 lives at albedo index 1 (see init_textures specs[]). If
        // that ever drops below 2, the fallback writes the first available
        // albedo so the shader still gets a real texture.
        const int wall_idx = (kFileTextureCount > 1) ? 1 : 0;
        VkDescriptorImageInfo wall_ii{
            .sampler     = texture_sampler_,
            .imageView   = albedo_textures_[wall_idx].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet w[4]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = voxel_desc_set_; w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &ubo_bi;
        w[1] = w[0];
        w[1].dstBinding = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &atlas_bi;
        w[2] = w[1];
        w[2].dstBinding = 2;
        w[2].pBufferInfo = &dir_bi;
        w[3] = w[0];
        w[3].dstBinding = 3;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[3].pBufferInfo = nullptr;
        w[3].pImageInfo  = &wall_ii;
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
    }

    // ---- Session B: also bind the brick atlas + directory into the
    // SCENE descriptor set (bindings 24 / 25) so cube.frag's inline-RT
    // shadow + GI rays can DDA-march the same buffers. init_voxel runs
    // after init_descriptors + write_scene_descriptors_once, so the set
    // exists; we just fill the two trailing bindings here.
    if (scene_desc_set_ != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo atlas_bi{ voxel_atlas_buffer_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dir_bi  { voxel_dir_buffer_,   0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = scene_desc_set_; w[0].dstBinding = 24;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &atlas_bi;
        w[1] = w[0];
        w[1].dstBinding = 25;
        w[1].pBufferInfo = &dir_bi;
        vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    }

    // ---- Session C: build collision (player AABBs + Jolt static body) ----
    rebuild_voxel_collision();

    log::info("[voxel] init complete — tower live, 1 shape, 1 pipeline");
}

void VulkanEngine::rebuild_voxel_collision() {
    if (!voxel_world_ || voxel_world_->shapes().empty()) return;
    std::vector<voxel::CollisionBox> boxes;
    voxel_world_->build_collision_boxes(0, boxes);

    // Player kinematic collision (game::collision AABBs). Rebuilt wholesale;
    // rebuild_tick_aabbs appends these to the static prefix each frame.
    voxel_collision_aabbs_.clear();
    voxel_collision_aabbs_.reserve(boxes.size());
    for (const auto& b : boxes) {
        voxel_collision_aabbs_.push_back(
            collision::AABB{ b.center - b.half, b.center + b.half });
    }

    // Projectile collision: one Jolt StaticCompound body.
    if (physics_) {
        std::vector<PhysicsWorld::StaticBox> jb;
        jb.reserve(boxes.size());
        for (const auto& b : boxes) jb.push_back({ b.center, b.half });
        physics_->set_voxel_collision(jb.data(), jb.size());
    }
}

void VulkanEngine::flush_voxel_brick_(uint32_t slot) {
    if (!voxel_atlas_mapped_) return;
    const voxel::BrickPayload& bp = voxel_world_->brick(slot);
    size_t off = static_cast<size_t>(slot) * sizeof(voxel::BrickPayload);
    if (off + sizeof(voxel::BrickPayload) > voxel_atlas_bytes_) return;
    std::memcpy(static_cast<char*>(voxel_atlas_mapped_) + off,
                &bp, sizeof(voxel::BrickPayload));
    // Flush just this brick's range (non-coherent memory). Cheap vs the
    // whole-atlas flush we used to do every carve.
    vmaFlushAllocation(allocator_, voxel_atlas_alloc_, off,
                       sizeof(voxel::BrickPayload));
}

void VulkanEngine::flush_voxel_bricks_batched_(const std::vector<uint32_t>& slots) {
    // Batched memcpy + single vmaFlushAllocation across the [min,max] slot
    // range. One driver call instead of N. Wasted flush bytes for unmodified
    // bricks inside the range are cheap (non-coherent memory; flush is a
    // CPU-side cache line writeback, not a GPU upload).
    if (!voxel_atlas_mapped_ || slots.empty()) return;
    uint32_t mn = slots.front(), mx = slots.front();
    for (uint32_t s : slots) {
        if (s < mn) mn = s;
        if (s > mx) mx = s;
        // memcpy each brick individually (sparse) — keeps unmodified bricks
        // in the range as-is. A whole-range memcpy would also work since the
        // host mirror IS authoritative, but per-slot is symmetric with the
        // single-brick path.
        const voxel::BrickPayload& bp = voxel_world_->brick(s);
        size_t off = static_cast<size_t>(s) * sizeof(voxel::BrickPayload);
        if (off + sizeof(voxel::BrickPayload) > voxel_atlas_bytes_) continue;
        std::memcpy(static_cast<char*>(voxel_atlas_mapped_) + off,
                    &bp, sizeof(voxel::BrickPayload));
    }
    size_t off0 = static_cast<size_t>(mn) * sizeof(voxel::BrickPayload);
    size_t bytes = static_cast<size_t>(mx - mn + 1) * sizeof(voxel::BrickPayload);
    if (off0 + bytes > voxel_atlas_bytes_)
        bytes = voxel_atlas_bytes_ - off0;
    vmaFlushAllocation(allocator_, voxel_atlas_alloc_, off0, bytes);
}

int VulkanEngine::apply_voxel_carve(int shape_idx, glm::vec3 world_center,
                                    float radius) {
    if (!voxel_world_ || shape_idx < 0 ||
        shape_idx >= (int)voxel_world_->shapes().size()) return 0;

    // For falling chunks (shape_idx > 0), the carve point is in world space
    // but carve_sphere works in shape-local. Compute the chunk's CURRENT
    // local-to-world from its Jolt body, invert, transform the world point
    // into shape-local, then translate back by the shape's recorded
    // origin_world (since carve_sphere subtracts that out internally).
    glm::vec3 carve_pos = world_center;
    if (shape_idx > 0 && physics_) {
        for (const auto& ch : voxel_chunks_) {
            if (ch.shape_index != shape_idx) continue;
            glm::mat4 body_world;
            if (!physics_->get_body_world_matrix_h(ch.jolt_handle, body_world))
                return 0;
            const auto& s = voxel_world_->shapes()[shape_idx];
            glm::vec3 half = 0.5f * glm::vec3(s.dim_bricks[0],
                                              s.dim_bricks[1],
                                              s.dim_bricks[2]) * voxel::kBrickSize;
            // chunk_world maps shape-local (0..dims) → world. Built the same
            // way render_voxels does it: body_world * translate(-half).
            glm::mat4 chunk_world = body_world *
                glm::translate(glm::mat4(1.0f), -half);
            glm::mat4 inv = glm::inverse(chunk_world);
            glm::vec3 local = glm::vec3(inv * glm::vec4(world_center, 1.0f));
            // carve_sphere does (carve_pos - s.origin_world) → reinjecting
            // origin_world here cancels that subtraction so the local point
            // makes it through to the DDA.
            carve_pos = s.origin_world + local;
            break;
        }
    }

    // CHEAP path (runs per bullet impact): clear the sphere of voxels and
    // push ONLY the touched bricks to the GPU so the hole shows instantly.
    // The expensive collapse + collision rebuild are deferred to
    // process_voxel_updates (debounced) so rapid fire never stalls a frame.
    std::vector<uint32_t> dirty;
    int removed = voxel_world_->carve_sphere(shape_idx, carve_pos, radius, dirty);
    if (removed == 0) return 0;
    flush_voxel_bricks_batched_(dirty);
    voxel_update_pending_ = true;
    voxel_removed_accum_ += removed;
    // Mark this shape dirty (dedupe).
    bool already = false;
    for (int s : voxel_dirty_shapes_) if (s == shape_idx) { already = true; break; }
    if (!already) voxel_dirty_shapes_.push_back(shape_idx);
    return removed;
}

void VulkanEngine::upload_voxel_growth_() {
    // Push any newly-allocated bricks (slot ≥ voxel_atlas_uploaded_) to the
    // host-mapped atlas, and any newly-claimed directory entries (offset ≥
    // voxel_dir_uploaded_) to the host-mapped dir buffer. Both are bounded
    // memcpys + a single vmaFlushAllocation each — cheap.
    if (!voxel_world_) return;
    if (voxel_atlas_mapped_) {
        uint32_t target = voxel_world_->brick_pool_size();
        if (target > voxel_atlas_uploaded_) {
            const auto& bricks = voxel_world_->bricks();
            if (target > kVoxelAtlasCapacityBricks) {
                target = kVoxelAtlasCapacityBricks;     // bound to capacity
                log::error("[voxel] brick atlas capacity exceeded — chunks dropped");
            }
            size_t off = size_t(voxel_atlas_uploaded_) * sizeof(voxel::BrickPayload);
            size_t cnt = size_t(target - voxel_atlas_uploaded_)
                       * sizeof(voxel::BrickPayload);
            std::memcpy((char*)voxel_atlas_mapped_ + off,
                        bricks.data() + voxel_atlas_uploaded_, cnt);
            vmaFlushAllocation(allocator_, voxel_atlas_alloc_, off, cnt);
            voxel_atlas_uploaded_ = target;
        }
    }
    if (voxel_dir_mapped_) {
        uint32_t target = voxel_world_->total_directory_size();
        if (target > voxel_dir_uploaded_) {
            if (target > kVoxelDirCapacityEntries) {
                target = kVoxelDirCapacityEntries;
                log::error("[voxel] directory capacity exceeded — chunks dropped");
            }
            // Walk only the shapes added since the last call (tracked by
            // voxel_dir_next_shape_). Old code re-scanned every shape ever
            // created on each collapse — cost grew linearly with session-
            // long debris history.
            const auto& shapes = voxel_world_->shapes();
            for (size_t si = voxel_dir_next_shape_; si < shapes.size(); ++si) {
                const auto& s = shapes[si];
                size_t off = size_t(s.dir_base) * sizeof(uint32_t);
                size_t bytes = s.directory.size() * sizeof(uint32_t);
                if (off + bytes > size_t(voxel_dir_bytes_)) continue;
                std::memcpy((char*)voxel_dir_mapped_ + off,
                            s.directory.data(), bytes);
                vmaFlushAllocation(allocator_, voxel_dir_alloc_, off, bytes);
            }
            voxel_dir_next_shape_ = static_cast<uint32_t>(shapes.size());
            voxel_dir_uploaded_   = target;
        }
    }
}

void VulkanEngine::process_voxel_updates(float dt) {
    if (!voxel_world_ || voxel_world_->shapes().empty()) return;
    if (voxel_collapse_cd_  > 0.0f) voxel_collapse_cd_  -= dt;
    if (voxel_collision_cd_ > 0.0f) voxel_collision_cd_ -= dt;

    // Tick chunk lifetime each frame regardless of pending state. A chunk
    // despawns when EITHER its TTL hits zero (hard cap) or Jolt has put
    // the body to sleep AFTER the chunk has been alive long enough that
    // sleep means "settled on terrain" (not "just hasn't been ticked yet").
    // The age gate replaces a magic-number TTL threshold (was `ttl < 6.5f`,
    // which silently broke if the initial TTL was changed).
    constexpr float kMinAgeBeforeSleepCull = 1.0f;   // seconds
    for (auto it = voxel_chunks_.begin(); it != voxel_chunks_.end(); ) {
        it->ttl -= dt;
        it->age += dt;
        const bool active = physics_ &&
                            physics_->is_body_active_h(it->jolt_handle);
        const bool slept = it->age >= kMinAgeBeforeSleepCull && !active;
        if (it->ttl <= 0.0f || slept) {
            if (physics_) physics_->remove_body(it->body_id);
            voxel_world_->drop_shape(it->shape_index);
            it = voxel_chunks_.erase(it);
        } else { ++it; }
    }

    if (!voxel_update_pending_) return;

    // Collapse (the ~20 ms BFS) only when enough has been carved to plausibly
    // disconnect a chunk, and rate-limited — a single bullet hole never pays
    // for it. A big cut (≥ ~3000 voxels of accumulated damage) triggers the
    // structural check; each detached connected component falls as its own
    // VOXEL CHUNK (own shape, own brick atlas slice, own Jolt body).
    bool collapsed = false;
    if (voxel_removed_accum_ >= 3000 && voxel_collapse_cd_ <= 0.0f) {
        voxel_removed_accum_ = 0;
        voxel_collapse_cd_   = 0.25f;
        collapsed = true;

        std::vector<uint32_t> main_dirty;
        std::vector<voxel::VoxelWorld::ChunkOut> chunks;
        voxel_world_->collapse_into_chunks(0, 8, chunks, main_dirty);
        // Flush the main shape's cleared voxels (batched).
        flush_voxel_bricks_batched_(main_dirty);
        // Push the new chunks' bricks + directories to the GPU buffers.
        upload_voxel_growth_();

        // Spawn a Jolt dynamic body per chunk (a single box for now — sized
        // to the chunk's bounding extent). Each chunk's voxel structure is
        // preserved in its own VoxelShape; voxel.frag draws it at the body's
        // pose, so it READS as a falling voxel piece (not a smooth log).
        if (physics_ && !chunks.empty()) {
            constexpr size_t kMaxChunks = 24;
            const size_t n = std::min(chunks.size(), kMaxChunks);
            for (size_t i = 0; i < n; ++i) {
                glm::vec3 he = chunks[i].half_extents;
                // Skip degenerate slivers (any axis < 5 cm OR total volume
                // < 0.001 m³, ~one 10-cm voxel). They produce wasted Jolt
                // bodies and invisible debris that drags physics for no
                // visual gain — drop the shape entry, don't spawn.
                float vol = 8.0f * he.x * he.y * he.z;
                if (std::min({he.x, he.y, he.z}) < 0.05f || vol < 0.001f) {
                    voxel_world_->drop_shape(chunks[i].shape_index);
                    continue;
                }
                uint32_t id = physics_->add_dynamic_box(
                    chunks[i].center_world, he, glm::vec3(0.0f), 250.0f);
                if (id == 0) {
                    voxel_world_->drop_shape(chunks[i].shape_index);
                    continue;
                }
                VoxelChunk vc{};
                vc.shape_index = chunks[i].shape_index;
                vc.body_id     = id;
                vc.jolt_handle = physics_->handle_of(id);
                vc.ttl         = 9.0f;
                voxel_chunks_.push_back(vc);
            }
            // Drop any chunks past the cap (they were extracted from the main
            // shape but we don't want a fleet of dynamic bodies — keep the
            // biggest, leak the rest as orphaned shapes that don't render).
            for (size_t i = n; i < chunks.size(); ++i)
                voxel_world_->drop_shape(chunks[i].shape_index);
            log::infof("[voxel] collapse: %zu voxel chunks spawned (of %zu)",
                       n, chunks.size());
        }
    }

    // Per-chunk self-split. If a bullet carved a chunk and the carve broke
    // it into multiple pieces, drop the smaller pieces as new chunks (the
    // largest stays in the parent's Jolt body). This is the "chunks can
    // break again" path — voxels remain voxels at every level.
    if (physics_ && !voxel_dirty_shapes_.empty()) {
        // Snapshot inheritance: pulling the parent body's pose + velocity
        // before split lets each new piece get a Jolt body that's roughly
        // where it visibly was (no teleport) and continues the bulk motion.
        for (int dirty_shape : voxel_dirty_shapes_) {
            if (dirty_shape <= 0) continue;  // shape 0 = main, handled above
            // Find the parent chunk entry.
            VoxelChunk* parent_ch = nullptr;
            for (auto& c : voxel_chunks_) {
                if (c.shape_index == dirty_shape) { parent_ch = &c; break; }
            }
            if (!parent_ch) continue;
            glm::mat4 parent_world;
            if (!physics_->get_body_world_matrix_h(parent_ch->jolt_handle,
                                                   parent_world)) continue;
            const glm::vec3 parent_lin =
                physics_->get_linear_velocity_h(parent_ch->jolt_handle);

            std::vector<uint32_t> piece_dirty;
            std::vector<voxel::VoxelWorld::ChunkOut> pieces;
            constexpr int kMinPieceVoxels = 8;   // tiny shards become dust
            int removed = voxel_world_->split_floating_chunk(
                dirty_shape, kMinPieceVoxels, pieces, piece_dirty);
            if (removed == 0 && pieces.empty()) continue;  // still 1 CC

            // Flush parent's cleared bricks + upload new shapes' bricks/dir.
            flush_voxel_bricks_batched_(piece_dirty);
            upload_voxel_growth_();

            // Spawn a body per new piece. Inherit parent velocity so they
            // continue along its trajectory + a small outward kick from the
            // parent centroid so they separate visibly.
            const auto& parent_sh = voxel_world_->shapes()[dirty_shape];
            const glm::vec3 parent_half = 0.5f * glm::vec3(
                parent_sh.dim_bricks[0],
                parent_sh.dim_bricks[1],
                parent_sh.dim_bricks[2]) * voxel::kBrickSize;
            const glm::mat4 parent_chunk_world =
                parent_world * glm::translate(glm::mat4(1.0f), -parent_half);
            for (const auto& p : pieces) {
                glm::vec3 he = p.half_extents;
                float vol = 8.0f * he.x * he.y * he.z;
                if (std::min({he.x, he.y, he.z}) < 0.05f || vol < 0.001f) {
                    voxel_world_->drop_shape(p.shape_index);
                    continue;
                }
                // p.center_world is in the parent's SHAPE-LOCAL frame
                // (split_floating_chunk built it from origin_world which
                // is the parent's REST origin, not its current world pos).
                // Transform it through parent_chunk_world to current world.
                glm::vec3 local_centre = p.center_world - parent_sh.origin_world;
                glm::vec3 spawn_world = glm::vec3(
                    parent_chunk_world * glm::vec4(local_centre, 1.0f));
                // Outward kick: spawn_world − parent body centre, scaled.
                glm::vec3 outward = spawn_world - glm::vec3(parent_world[3]);
                float ol = glm::length(outward);
                glm::vec3 kick = (ol > 1e-3f) ? outward * (1.5f / ol)
                                              : glm::vec3(0.0f);
                glm::vec3 init_vel = parent_lin + kick;
                uint32_t id = physics_->add_dynamic_box(
                    spawn_world, he, init_vel, 100.0f);
                if (id == 0) { voxel_world_->drop_shape(p.shape_index); continue; }
                VoxelChunk vc{};
                vc.shape_index = p.shape_index;
                vc.body_id     = id;
                vc.jolt_handle = physics_->handle_of(id);
                vc.ttl         = 9.0f;
                voxel_chunks_.push_back(vc);
            }
            log::infof("[voxel] chunk split: shape %d -> %zu new pieces",
                       dirty_shape, pieces.size());
            // parent_ch pointer may now dangle if voxel_chunks_ reallocated
            // — do NOT touch it past this point.
        }
    }
    voxel_dirty_shapes_.clear();

    // Collision rebuild (player AABBs + Jolt body): voxel_update_pending_ is
    // the "collision is stale" flag, set by every carve. Rebuild on a slow
    // cadence (continuous hole-poking shouldn't rebuild every frame), forced
    // right after a collapse, and SKIPPED entirely while the player is far
    // from the tower — shooting it from a distance never needs up-to-date
    // physics solidity (the hole's VISUAL already updated in the carve). It
    // catches up the moment they approach. Stays flagged dirty until then.
    if (!voxel_update_pending_) return;
    const auto& s0 = voxel_world_->shapes()[0];
    glm::vec3 tower_c = s0.origin_world + 0.5f * glm::vec3(
        s0.dim_bricks[0], s0.dim_bricks[1], s0.dim_bricks[2]) * voxel::kBrickSize;
    glm::vec3 d = player_.eye_position() - tower_c;
    const bool near_tower = glm::dot(d, d) < (45.0f * 45.0f);
    if (collapsed) voxel_collision_cd_ = 0.0f;
    if (near_tower && voxel_collision_cd_ <= 0.0f) {
        rebuild_voxel_collision();
        voxel_collision_cd_   = 0.5f;
        voxel_update_pending_ = false;
    }
}

void VulkanEngine::destroy_voxel() {
    if (voxel_pipeline_) {
        vkDestroyPipeline(device_, voxel_pipeline_, nullptr);
        voxel_pipeline_ = VK_NULL_HANDLE;
    }
    if (voxel_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, voxel_pipeline_layout_, nullptr);
        voxel_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (voxel_desc_pool_) {
        vkDestroyDescriptorPool(device_, voxel_desc_pool_, nullptr);
        voxel_desc_pool_ = VK_NULL_HANDLE;
        voxel_desc_set_ = VK_NULL_HANDLE;
    }
    if (voxel_desc_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, voxel_desc_set_layout_, nullptr);
        voxel_desc_set_layout_ = VK_NULL_HANDLE;
    }
    if (voxel_vert_module_) {
        vkDestroyShaderModule(device_, voxel_vert_module_, nullptr);
        voxel_vert_module_ = VK_NULL_HANDLE;
    }
    if (voxel_frag_module_) {
        vkDestroyShaderModule(device_, voxel_frag_module_, nullptr);
        voxel_frag_module_ = VK_NULL_HANDLE;
    }
    if (voxel_camera_ubo_) {
        vmaDestroyBuffer(allocator_, voxel_camera_ubo_, voxel_camera_alloc_);
        voxel_camera_ubo_ = VK_NULL_HANDLE;
        voxel_camera_alloc_ = nullptr;
        voxel_camera_mapped_ = nullptr;
    }
    if (voxel_atlas_buffer_) {
        vmaDestroyBuffer(allocator_, voxel_atlas_buffer_, voxel_atlas_alloc_);
        voxel_atlas_buffer_ = VK_NULL_HANDLE;
        voxel_atlas_alloc_ = nullptr;
        voxel_atlas_mapped_ = nullptr;
        voxel_atlas_bytes_ = 0;
        voxel_atlas_uploaded_ = 0;
    }
    if (voxel_dir_buffer_) {
        vmaDestroyBuffer(allocator_, voxel_dir_buffer_, voxel_dir_alloc_);
        voxel_dir_buffer_ = VK_NULL_HANDLE;
        voxel_dir_alloc_ = nullptr;
        voxel_dir_mapped_ = nullptr;
        voxel_dir_bytes_ = 0;
        voxel_dir_uploaded_ = 0;
        voxel_dir_next_shape_ = 0;
    }
    // Remove any active chunks' Jolt bodies before tearing physics down.
    if (physics_) {
        for (const auto& ch : voxel_chunks_) physics_->remove_body(ch.body_id);
    }
    voxel_chunks_.clear();
    voxel_dirty_shapes_.clear();
    voxel_world_.reset();
}

void VulkanEngine::update_voxel_camera_ubo() {
    if (!voxel_camera_mapped_) return;

    VoxelCameraUboData data{};
    const FrameView& fv = current_frame_view_;
    data.view_proj      = fv.vp;
    data.prev_view_proj = prev_view_proj_;
    data.camera_pos     = glm::vec4(fv.eye_pos, 1.0f);

    // Match the sun direction computation used by descriptors.cpp /
    // combat.cpp / sun_shadow.cpp — keeps voxel shading consistent
    // with the rest of the scene.
    float p_rad = glm::radians(rt_.sun_pitch_deg);
    float y_rad = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun = glm::normalize(glm::vec3(
        std::cos(p_rad) * std::sin(y_rad),
        std::sin(p_rad),
        std::cos(p_rad) * std::cos(y_rad)));
    // shader expects sun_dir to be "toward the sun" — the dot uses
    // (-sun_dir) so we flip sign here to match.
    data.sun_dir   = glm::vec4(-sun, 0.0f);
    data.sun_color = glm::vec4(1.0f, 0.96f, 0.88f, 1.0f);
    data.ambient   = glm::vec4(0.18f, 0.20f, 0.24f, 1.0f);

    float w = static_cast<float>(render_extent_.width);
    float h = static_cast<float>(render_extent_.height);
    data.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);

    // 16-entry palette. 0 = grey, 1 = sandstone, 2 = warm wood, 3-15 =
    // procedurally interpolated copper-gold → rust-brown ramp so any
    // material id below 16 lands on a plausible masonry tone instead of
    // the old magenta sentinels.
    data.pal[0]  = glm::vec4(0.50f, 0.50f, 0.50f, 1.0f);   // grey stone
    data.pal[1]  = glm::vec4(0.95f, 0.78f, 0.45f, 1.0f);   // bright sandstone
    data.pal[2]  = glm::vec4(0.75f, 0.30f, 0.15f, 1.0f);   // warm red wood
    const glm::vec3 c_copper(0.85f, 0.55f, 0.30f);
    const glm::vec3 c_rust  (0.45f, 0.22f, 0.12f);
    for (int i = 3; i < 16; ++i) {
        float t = float(i - 3) / 12.0f;
        glm::vec3 col = glm::mix(c_copper, c_rust, t);
        data.pal[i] = glm::vec4(col, 1.0f);
    }

    std::memcpy(voxel_camera_mapped_, &data, sizeof(data));
}

void VulkanEngine::render_voxels(VkCommandBuffer cmd) {
    if (!voxel_pipeline_ || !voxel_world_ || voxel_world_->shapes().empty()) return;

    update_voxel_camera_ubo();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, voxel_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            voxel_pipeline_layout_, 0, 1,
                            &voxel_desc_set_, 0, nullptr);

    // Helper: push a shape's PC + draw the 36-vert AABB cube.
    const Frustum& frustum = current_frame_view_.frustum;
    auto draw_shape = [&](const voxel::VoxelShape& s, const glm::mat4& model) {
        // Skip dropped shapes (dim_bricks zeroed) or shapes whose grid is
        // empty.
        if (s.dim_bricks[0] <= 0 || s.dim_bricks[1] <= 0 || s.dim_bricks[2] <= 0)
            return;
        // World-space AABB frustum cull. Even with rotation, the shape's
        // axis-aligned box around its CURRENT transform is a conservative
        // bound — rotate the 8 local corners and take min/max. Off-screen
        // chunks skip the proxy-cube draw + per-pixel DDA entirely.
        glm::vec3 ext = glm::vec3(s.dim_bricks[0], s.dim_bricks[1], s.dim_bricks[2])
                      * voxel::kBrickSize;
        glm::vec3 lo( std::numeric_limits<float>::infinity());
        glm::vec3 hi(-std::numeric_limits<float>::infinity());
        for (int ci = 0; ci < 8; ++ci) {
            glm::vec3 c(((ci >> 0) & 1) ? ext.x : 0.0f,
                        ((ci >> 1) & 1) ? ext.y : 0.0f,
                        ((ci >> 2) & 1) ? ext.z : 0.0f);
            glm::vec3 w = glm::vec3(model * glm::vec4(c, 1.0f));
            lo = glm::min(lo, w);
            hi = glm::max(hi, w);
        }
        if (!aabb_visible(frustum, lo, hi)) return;
        // Decompose the rigid model into rotation R + translation T so the
        // fragment shader can use the analytic rigid-inverse (transpose,
        // -T) instead of mat4 inverse() per pixel. glm matrices are
        // column-major; model[0..2] are the rotation columns, model[3]
        // is the translation column.
        VoxelPushConstants pc{};
        pc.R0       = glm::vec4(glm::vec3(model[0]), 0.0f);
        pc.R1       = glm::vec4(glm::vec3(model[1]), 0.0f);
        pc.R2       = glm::vec4(glm::vec3(model[2]), 0.0f);
        pc.T        = glm::vec4(glm::vec3(model[3]), 0.0f);
        pc.dims_vs  = glm::vec4(
            s.dim_bricks[0] * voxel::kBrickSize,
            s.dim_bricks[1] * voxel::kBrickSize,
            s.dim_bricks[2] * voxel::kBrickSize,
            voxel::kVoxelSize);
        pc.grid_dir = glm::ivec4(s.dim_bricks[0], s.dim_bricks[1],
                                  s.dim_bricks[2], (int)s.dir_base);
        vkCmdPushConstants(cmd, voxel_pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 36, 1, 0, 0);
    };

    // Draw the main shape (translation-only).
    const auto& shapes = voxel_world_->shapes();
    {
        const voxel::VoxelShape& s = shapes[0];
        glm::mat4 model = glm::translate(glm::mat4(1.0f), s.origin_world);
        draw_shape(s, model);
    }

    // Draw each chunk with its Jolt-driven transform. The shape origin is
    // the chunk's rest position (in shape-local 0..dims), so the model
    // matrix moves to the Jolt body's centre AND rotates around the chunk's
    // own centre — translate(centre) · rotate · translate(-half) — so the
    // chunk's local centre lines up with the body's centre.
    if (physics_) {
        for (const auto& ch : voxel_chunks_) {
            if (ch.shape_index < 0 || ch.shape_index >= (int)shapes.size()) continue;
            const voxel::VoxelShape& s = shapes[ch.shape_index];
            glm::mat4 body_world;
            if (!physics_->get_body_world_matrix_h(ch.jolt_handle, body_world))
                continue;
            // body_world is centred at the chunk's centre. The shape's local
            // origin is at (0,0,0); its centre is at half_extent. We want:
            //   world = body_world * translate(-half_extent) when applied to
            //   shape-local coords (so a local (0,0,0) maps to body_centre
            //   − half_extent, and local (dims) maps to body_centre + half).
            glm::vec3 half = 0.5f * glm::vec3(
                s.dim_bricks[0], s.dim_bricks[1], s.dim_bricks[2]) *
                voxel::kBrickSize;
            glm::mat4 model = body_world *
                              glm::translate(glm::mat4(1.0f), -half);
            draw_shape(s, model);
        }
    }
}

} // namespace qlike
