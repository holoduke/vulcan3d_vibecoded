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

struct VoxelPushConstants {
    glm::vec4  origin_world;
    glm::vec4  dims_world;
    glm::ivec4 dims_bricks;
    glm::vec4  voxel_size;   // x = vs, y = bs
    glm::vec4  shape_idx;
};
static_assert(sizeof(VoxelPushConstants) == 80, "VoxelPC size — fits 128B PC limit");

// Tower placement — 100 m north of the castle origin. Tower extends
// +X/+Y/+Z from the base corner; centre on (0, _, 100).
constexpr float kTowerCenterZ = 100.0f;

} // namespace

void VulkanEngine::init_voxel() {
    using namespace voxel;

    // ---- CPU side: build the procedural tower ----
    voxel_world_ = std::make_unique<VoxelWorld>();

    // 12×18×12 bricks → 19.2 × 28.8 × 19.2 m.
    constexpr float kTowerSizeX = 12.0f * kBrickSize;   // 19.2
    constexpr float kTowerSizeZ = 12.0f * kBrickSize;
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

    const VkDeviceSize atlas_bytes = static_cast<VkDeviceSize>(bricks.size()) *
                                     sizeof(voxel::BrickPayload);
    // Brick atlas is host-mapped (not staged) so carves can flush touched
    // bricks directly — see voxel_atlas_mapped_. Device-local-host-visible
    // (ReBAR) when available; 4 MB fits the guaranteed BAR window.
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
        std::memcpy(ai.pMappedData, bricks.data(), static_cast<size_t>(atlas_bytes));
        voxel_atlas_mapped_ = ai.pMappedData;
        voxel_atlas_bytes_  = atlas_bytes;
    }

    // Shape directory: header (uvec4 = dims) + entries[].
    // GLSL std430 lays this out tightly when the buffer is declared as
    // `uvec4 hdr; uint entries[];`.
    std::vector<uint32_t> dir_blob;
    dir_blob.reserve(4 + s0.directory.size());
    dir_blob.push_back(static_cast<uint32_t>(s0.dim_bricks[0]));
    dir_blob.push_back(static_cast<uint32_t>(s0.dim_bricks[1]));
    dir_blob.push_back(static_cast<uint32_t>(s0.dim_bricks[2]));
    dir_blob.push_back(0u);  // padding for uvec4
    dir_blob.insert(dir_blob.end(), s0.directory.begin(), s0.directory.end());
    const VkDeviceSize dir_bytes = dir_blob.size() * sizeof(uint32_t);
    create_device_ssbo(dir_bytes, dir_blob.data(), "voxel shape directory",
                       voxel_dir_buffer_, voxel_dir_alloc_);

    log::infof("[voxel] GPU upload: atlas=%llu KB, dir=%llu KB",
               (unsigned long long)(atlas_bytes / 1024),
               (unsigned long long)(dir_bytes  / 1024));

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
        VkDescriptorSetLayoutBinding b[3]{};
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
        VkDescriptorSetLayoutCreateInfo dlci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3, .pBindings = b,
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
        cfg.color_formats = { scene_color_format_, motion_vec_format_ };
        cfg.depth_format = depth_format_;
        cfg.depth_test = true;
        cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
        // CULL_NONE: rasterize all 12 tris of the AABB. The near (front)
        // face supplies a depth value < the back face, so depth-test
        // LESS_OR_EQUAL against existing world depth correctly admits
        // the fragment when no terrain occludes the near face. CULL_FRONT
        // would rasterize ONLY the back face, whose far depth fails the
        // depth-test against any terrain in front of the box → invisible
        // voxel. Cost: up to 2x fragment invocations per AABB pixel; the
        // second invocation writes the same gl_FragDepth, so depth_write
        // is idempotent and the perf hit is negligible for one shape.
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
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 2, .pPoolSizes = sizes,
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
        VkWriteDescriptorSet w[3]{};
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
        vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
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

int VulkanEngine::apply_voxel_carve(glm::vec3 center, float radius) {
    if (!voxel_world_ || voxel_world_->shapes().empty()) return 0;
    // CHEAP path (runs per bullet impact): clear the sphere of voxels and
    // push ONLY the touched bricks to the GPU so the hole shows instantly.
    // The expensive collapse + collision rebuild are deferred to
    // process_voxel_updates (debounced) so rapid fire never stalls a frame.
    std::vector<uint32_t> dirty;
    int removed = voxel_world_->carve_sphere(0, center, radius, dirty);
    if (removed == 0) return 0;
    for (uint32_t slot : dirty) flush_voxel_brick_(slot);
    voxel_update_pending_ = true;
    voxel_removed_accum_ += removed;
    return removed;
}

void VulkanEngine::process_voxel_updates(float dt) {
    if (!voxel_world_ || voxel_world_->shapes().empty()) return;
    if (voxel_collapse_cd_  > 0.0f) voxel_collapse_cd_  -= dt;
    if (voxel_collision_cd_ > 0.0f) voxel_collision_cd_ -= dt;
    if (!voxel_update_pending_) return;

    // Collapse (the ~20 ms BFS) only when enough has been carved to plausibly
    // disconnect a chunk, and rate-limited — a single bullet hole never pays
    // for it. A big cut (≥ ~3000 voxels of accumulated damage) triggers the
    // structural check; the detached top falls as debris.
    bool collapsed = false;
    if (voxel_removed_accum_ >= 3000 && voxel_collapse_cd_ <= 0.0f) {
        voxel_removed_accum_ = 0;
        voxel_collapse_cd_   = 0.25f;
        collapsed = true;

        std::vector<uint32_t> dirty;
        std::vector<voxel::CollisionBox> debris;
        voxel_world_->collapse_unsupported(0, 8, debris, dirty);
        for (uint32_t slot : dirty) flush_voxel_brick_(slot);

    // Spawn the detached chunk as falling debris boxes (reuses dynamic-prop
    // physics + rendering). Largest chunks first so the visible mass is
    // preserved; the rest were already cleared from the shape.
    if (physics_ && !debris.empty()) {
        std::sort(debris.begin(), debris.end(),
                  [](const voxel::CollisionBox& a, const voxel::CollisionBox& b) {
                      return a.half.x * a.half.y * a.half.z >
                             b.half.x * b.half.y * b.half.z;
                  });
        constexpr int kMaxDebris = 40;
        int n = std::min((int)debris.size(), kMaxDebris);
        for (int i = 0; i < n; ++i) {
            glm::vec3 he = debris[i].half;
            if (std::min({he.x, he.y, he.z}) < 0.05f) continue;
            uint32_t id = physics_->add_dynamic_box(debris[i].center, he,
                                                    glm::vec3(0.0f), 400.0f);
            if (id == 0) continue;
            DynamicProp p{};
            p.body_id = id;
            p.jolt_handle = physics_->handle_of(id);
            p.full_size = he * 2.0f;
            p.fallback_color = glm::vec4(0.62f, 0.50f, 0.36f, 1.0f);  // sandstone
            p.color = glm::vec4(0.62f, 0.50f, 0.36f, 1.0f);
            p.tex_albedo = -1; p.tex_normal = -1; p.uv_scale = 1.0f;
            dyn_props_.push_back(p);
            while (static_cast<int>(dyn_props_.size()) > kMaxDynProps) {
                physics_->remove_body(dyn_props_.front().body_id);
                dyn_props_.erase(dyn_props_.begin());
            }
        }
            log::infof("[voxel] collapse: %d debris boxes spawned (of %zu)",
                       n, debris.size());
        }
    }

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
    }
    if (voxel_dir_buffer_) {
        vmaDestroyBuffer(allocator_, voxel_dir_buffer_, voxel_dir_alloc_);
        voxel_dir_buffer_ = VK_NULL_HANDLE;
        voxel_dir_alloc_ = nullptr;
    }
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

    // 16-entry palette. Index 0 reserved; 1 = stone, 2 = warm accent.
    for (int i = 0; i < 16; ++i) data.pal[i] = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    data.pal[0]  = glm::vec4(0.50f, 0.50f, 0.50f, 1.0f);
    // Saturated sandstone + warm wood — tuned to stand out against the
    // distant terrain (which sits around grey-green at 100 m) and pink
    // sky. Will tone down once normal lighting / shadows land in Session B.
    data.pal[1]  = glm::vec4(0.95f, 0.78f, 0.45f, 1.0f);  // bright sandstone
    data.pal[2]  = glm::vec4(0.75f, 0.30f, 0.15f, 1.0f);  // warm red wood accent

    std::memcpy(voxel_camera_mapped_, &data, sizeof(data));
}

void VulkanEngine::render_voxels(VkCommandBuffer cmd) {
    if (!voxel_pipeline_ || !voxel_world_ || voxel_world_->shapes().empty()) return;

    update_voxel_camera_ubo();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, voxel_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            voxel_pipeline_layout_, 0, 1,
                            &voxel_desc_set_, 0, nullptr);

    const voxel::VoxelShape& s = voxel_world_->shapes()[0];
    VoxelPushConstants pc{};
    pc.origin_world = glm::vec4(s.origin_world, 0.0f);
    pc.dims_world   = glm::vec4(
        s.dim_bricks[0] * voxel::kBrickSize,
        s.dim_bricks[1] * voxel::kBrickSize,
        s.dim_bricks[2] * voxel::kBrickSize,
        0.0f);
    pc.dims_bricks  = glm::ivec4(s.dim_bricks[0], s.dim_bricks[1],
                                  s.dim_bricks[2], 0);
    pc.voxel_size   = glm::vec4(voxel::kVoxelSize, voxel::kBrickSize, 0.0f, 0.0f);
    pc.shape_idx    = glm::vec4(0.0f);

    vkCmdPushConstants(cmd, voxel_pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 36, 1, 0, 0);
}

} // namespace qlike
