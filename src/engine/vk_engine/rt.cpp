// Ray tracing acceleration structures + per-frame TLAS rebuild. The cube
// and cylinder BLAS are built once at init from their meshes; the TLAS is
// host-mapped (768 instances) and rewritten every frame from the static
// brush bake + DynRender cache + particles + projectiles. UPDATE mode is
// used when the instance count is unchanged (the common case); a full
// BUILD only happens on count changes (spawn / despawn).

#include "engine/vk_engine/internal.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace qlike {

void VulkanEngine::init_rt() {
    // RT buffers (BLAS/TLAS storage, instance buffer, scratch) are touched
    // by both the graphics queue (cube.frag's ray queries) and the compute
    // queue (per-frame TLAS rebuild). When those queues live in different
    // families we use SHARING_MODE_CONCURRENT to avoid emitting queue-family
    // ownership-transfer barriers around every cross-queue access. The cost
    // is a small driver-level overhead vs. EXCLUSIVE on some hardware.
    const uint32_t shared_families[2] = { graphics_queue_family_, compute_queue_family_ };
    const VkSharingMode rt_sharing = compute_queue_distinct_
        ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    const uint32_t rt_share_count = compute_queue_distinct_ ? 2u : 0u;
    const uint32_t* rt_share_indices = compute_queue_distinct_ ? shared_families : nullptr;

    // Build a BLAS for an arbitrary qlike::Mesh. Reused for cube + cylinder.
    auto build_mesh_blas = [&](const Mesh& mesh, const char* tag,
                               VkBuffer& out_buffer, VmaAllocation& out_alloc,
                               VkAccelerationStructureKHR& out_as,
                               VkDeviceAddress& out_addr) {
        const uint32_t tri_count = mesh.index_count / 3;
        VkDeviceAddress vbo_addr = buffer_device_address(device_, mesh.vertex_buffer);
        VkDeviceAddress ibo_addr = buffer_device_address(device_, mesh.index_buffer);

        VkAccelerationStructureGeometryTrianglesDataKHR tri{};
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = vbo_addr;
        tri.vertexStride = sizeof(Vertex);
        tri.maxVertex = mesh.vertex_count - 1;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = ibo_addr;

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tri;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi.geometryCount = 1;
        bgi.pGeometries = &geom;

        uint32_t prim_count = tri_count;
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        g_rt.get_as_build_sizes(device_,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &prim_count, &sizes);

        {
            VkBufferCreateInfo bci{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr, .flags = 0,
                .size = sizes.accelerationStructureSize,
                .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                .sharingMode = rt_sharing,
                .queueFamilyIndexCount = rt_share_count,
                .pQueueFamilyIndices = rt_share_indices,
            };
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                     &out_buffer, &out_alloc, nullptr),
                     "blas buffer");
        }
        VkAccelerationStructureCreateInfoKHR aci_blas{};
        aci_blas.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        aci_blas.buffer = out_buffer;
        aci_blas.size = sizes.accelerationStructureSize;
        aci_blas.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vk_check(g_rt.create_as(device_, &aci_blas, nullptr, &out_as),
                 "vkCreateAccelerationStructureKHR blas");
        bgi.dstAccelerationStructure = out_as;

        VkBuffer scratch = VK_NULL_HANDLE;
        VmaAllocation scratch_alloc = nullptr;
        {
            VkBufferCreateInfo bci{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr, .flags = 0,
                .size = sizes.buildScratchSize,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            };
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                     &scratch, &scratch_alloc, nullptr),
                     "blas scratch");
        }
        bgi.scratchData.deviceAddress = buffer_device_address(device_, scratch);

        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = tri_count;
            const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
            g_rt.cmd_build_as(cb, 1, &bgi, &p_range);
        });

        vmaDestroyBuffer(allocator_, scratch, scratch_alloc);
        out_addr = as_device_address(device_, out_as);
        log::infof("BLAS (%s) built: %u tris", tag, tri_count);
    };

    build_mesh_blas(cube_mesh_, "cube",
                    blas_buffer_, blas_alloc_, blas_, blas_device_address_);
    build_mesh_blas(cylinder_mesh_, "cylinder",
                    cylinder_blas_buffer_, cylinder_blas_alloc_,
                    cylinder_blas_, cylinder_blas_device_address_);

    // --- Allocate TLAS storage + scratch + instance buffer (rebuilt each frame) ---
    // Sized for the FIFO box cap (200) + ~32 static brushes/lanterns +
    // ~80 in-flight projectiles + ~256 hit particles + headroom.
    tlas_max_instances_ = 768;

    {
        VkDeviceSize ibuf_size = sizeof(VkAccelerationStructureInstanceKHR) *
                                 tlas_max_instances_;
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = ibuf_size,
            .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = rt_sharing,
            .queueFamilyIndexCount = rt_share_count,
            .pQueueFamilyIndices = rt_share_indices,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &instances_buffer_, &instances_alloc_, nullptr),
                 "tlas instance buffer");
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, instances_alloc_, &ai);
        instances_mapped_ = ai.pMappedData;
        instances_buffer_address_ = buffer_device_address(device_, instances_buffer_);
    }

    // Per-instance material buffer (host-visible, mapped). Three vec4 per
    // entry: color (rgb) + full_emissive flag (a); emissive (rgb) + reserved;
    // tex_params (albedo idx, normal idx, uv scale, reserved).
    {
        VkDeviceSize size = static_cast<VkDeviceSize>(tlas_max_instances_) *
                            (3u * sizeof(glm::vec4));
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &materials_buffer_, &materials_alloc_, nullptr),
                 "materials buffer");
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, materials_alloc_, &ai);
        materials_mapped_ = ai.pMappedData;
    }

    // Per-instance previous-frame transforms (mat4 per slot). cube.frag's
    // motion-vector output uses prev_mvp passed via push constants today;
    // this SSBO stays around for a future compute-shader motion-vec pass
    // that can read prev_world via gl_InstanceCustomIndex without needing
    // the engine to re-emit prev_mvp on every draw.
    {
        VkDeviceSize size = static_cast<VkDeviceSize>(tlas_max_instances_) *
                            sizeof(glm::mat4);
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &prev_transforms_buffer_,
                                 &prev_transforms_alloc_, nullptr),
                 "prev transforms buffer");
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, prev_transforms_alloc_, &ai);
        prev_transforms_mapped_ = ai.pMappedData;
    }

    // Query TLAS size at max instance count.
    VkAccelerationStructureGeometryInstancesDataKHR inst_data{};
    inst_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst_data.arrayOfPointers = VK_FALSE;
    inst_data.data.deviceAddress = instances_buffer_address_;

    VkAccelerationStructureGeometryKHR tlas_geom{};
    tlas_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geom.geometry.instances = inst_data;
    tlas_geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR tlas_bgi{};
    tlas_bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlas_bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // ALLOW_UPDATE lets rebuild_tlas() pick UPDATE mode (vs full BUILD) for
    // frames where instance count is unchanged. Update is far cheaper than
    // build for moving instances.
    tlas_bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                     VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    tlas_bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlas_bgi.geometryCount = 1;
    tlas_bgi.pGeometries = &tlas_geom;

    VkAccelerationStructureBuildSizesInfoKHR tlas_sizes{};
    tlas_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    g_rt.get_as_build_sizes(device_,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_bgi,
        &tlas_max_instances_, &tlas_sizes);

    tlas_buffer_size_ = tlas_sizes.accelerationStructureSize;
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = tlas_buffer_size_,
            .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = rt_sharing,
            .queueFamilyIndexCount = rt_share_count,
            .pQueueFamilyIndices = rt_share_indices,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &tlas_buffer_, &tlas_alloc_, nullptr),
                 "tlas buffer");
    }
    VkAccelerationStructureCreateInfoKHR aci_tlas{};
    aci_tlas.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    aci_tlas.buffer = tlas_buffer_;
    aci_tlas.size = tlas_buffer_size_;
    aci_tlas.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vk_check(g_rt.create_as(device_, &aci_tlas, nullptr, &tlas_),
             "vkCreateAccelerationStructureKHR tlas");

    {
        // Scratch must hold whichever of {build, update} is bigger — the same
        // buffer is reused for both modes in rebuild_tlas().
        VkDeviceSize scratch_size = tlas_sizes.buildScratchSize;
        if (tlas_sizes.updateScratchSize > scratch_size) {
            scratch_size = tlas_sizes.updateScratchSize;
        }
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = scratch_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            // Scratch is only touched by the compute queue's AS build, but
            // we include CONCURRENT here for symmetry — the BLAS build at
            // init time uses one_time_submit on the graphics queue, which
            // means the FIRST scratch use is on graphics. CONCURRENT avoids
            // the implicit ownership claim getting "stuck" on graphics
            // family when the per-frame TLAS build then runs on compute.
            .sharingMode = rt_sharing,
            .queueFamilyIndexCount = rt_share_count,
            .pQueueFamilyIndices = rt_share_indices,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &tlas_scratch_buffer_, &tlas_scratch_alloc_, nullptr),
                 "tlas scratch");
        tlas_scratch_address_ = buffer_device_address(device_, tlas_scratch_buffer_);
    }

    rt_initialized_ = true;
    log::infof("RT initialized: max %u instances, TLAS buffer %llu B",
               tlas_max_instances_,
               static_cast<unsigned long long>(tlas_buffer_size_));
}

void VulkanEngine::destroy_rt() {
    if (tlas_)               g_rt.destroy_as(device_, tlas_, nullptr);
    if (blas_)               g_rt.destroy_as(device_, blas_, nullptr);
    if (cylinder_blas_)      g_rt.destroy_as(device_, cylinder_blas_, nullptr);
    if (tlas_scratch_buffer_) vmaDestroyBuffer(allocator_, tlas_scratch_buffer_, tlas_scratch_alloc_);
    if (instances_buffer_)    vmaDestroyBuffer(allocator_, instances_buffer_, instances_alloc_);
    if (materials_buffer_)    vmaDestroyBuffer(allocator_, materials_buffer_, materials_alloc_);
    if (prev_transforms_buffer_) vmaDestroyBuffer(allocator_, prev_transforms_buffer_, prev_transforms_alloc_);
    if (tlas_buffer_)         vmaDestroyBuffer(allocator_, tlas_buffer_, tlas_alloc_);
    if (blas_buffer_)         vmaDestroyBuffer(allocator_, blas_buffer_, blas_alloc_);
    if (cylinder_blas_buffer_) vmaDestroyBuffer(allocator_, cylinder_blas_buffer_, cylinder_blas_alloc_);
    tlas_ = blas_ = cylinder_blas_ = VK_NULL_HANDLE;
    tlas_buffer_ = blas_buffer_ = cylinder_blas_buffer_ = tlas_scratch_buffer_ =
        instances_buffer_ = materials_buffer_ = prev_transforms_buffer_ = VK_NULL_HANDLE;
    tlas_alloc_ = blas_alloc_ = cylinder_blas_alloc_ = tlas_scratch_alloc_ =
        instances_alloc_ = materials_alloc_ = prev_transforms_alloc_ = nullptr;
    cylinder_blas_device_address_ = 0;
    materials_mapped_ = nullptr;
    prev_transforms_mapped_ = nullptr;
    rt_initialized_ = false;
}

void VulkanEngine::bake_static_brushes() {
    constexpr glm::vec4 kNoTex(-1.0f, -1.0f, 1.0f, 0.0f);
    const bool tex_on = rt_.textures_enabled;

    static_brush_instances_.clear();
    static_brush_materials_.clear();
    static_brush_worlds_.clear();
    static_brush_instances_.reserve(world_.brushes.size());
    static_brush_materials_.reserve(world_.brushes.size() * 3);
    static_brush_worlds_.reserve(world_.brushes.size());

    uint32_t i = 0;
    for (const auto& b : world_.brushes) {
        glm::mat4 m = glm::translate(glm::mat4(1.0f), b.center) *
                      glm::scale(glm::mat4(1.0f), b.size);
        static_brush_worlds_.push_back(m);
        glm::vec4 base = tex_on ? b.color : b.fallback_color;
        glm::vec4 tex = tex_on
            ? glm::vec4(static_cast<float>(b.tex_albedo),
                        static_cast<float>(b.tex_normal),
                        b.uv_scale, 0.0f)
            : kNoTex;

        VkAccelerationStructureInstanceKHR inst{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                inst.transform.matrix[row][col] = m[col][row];
            }
        }
        inst.instanceCustomIndex = i;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas_device_address_;
        static_brush_instances_.push_back(inst);

        static_brush_materials_.push_back(
            glm::vec4(glm::vec3(base), b.full_emissive ? 1.0f : 0.0f));
        static_brush_materials_.push_back(glm::vec4(b.emissive, 0.0f));
        static_brush_materials_.push_back(tex);
        ++i;
    }
    static_brush_tex_on_ = tex_on;
    static_brush_dirty_ = false;
}

void VulkanEngine::rebuild_tlas(VkCommandBuffer cmd) {
    if (!rt_initialized_) return;

    // Inter-submission AS-build hazard fence. Two TLAS builds across
    // consecutive frames write to the same `tlas_` buffer; submission order
    // on a single queue does NOT imply memory ordering for AS builds. Without
    // this barrier, the sync validator reports WAW hazards and NVIDIA can
    // (and does, ~40% of autodemo runs) eventually surface this as
    // VK_ERROR_DEVICE_LOST. The barrier waits for any prior AS-build write to
    // drain before the next AS-build starts, scoped to the AS-build stage so
    // it doesn't penalize anything else.
    {
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    const bool tex_on = rt_.textures_enabled;
    if (static_brush_dirty_ || static_brush_tex_on_ != tex_on) {
        bake_static_brushes();
    }

    auto* dst = static_cast<VkAccelerationStructureInstanceKHR*>(instances_mapped_);
    auto* mats = static_cast<glm::vec4*>(materials_mapped_);  // 3 vec4 per material
    auto* prevs = static_cast<glm::mat4*>(prev_transforms_mapped_);

    // Static-brush block: baked once per (level, textures_enabled). Memcpy
    // straight in instead of recomputing the world matrices and material
    // tuples every frame. Prev_world for static brushes equals current world
    // (no motion).
    const uint32_t static_n = std::min<uint32_t>(
        static_cast<uint32_t>(static_brush_instances_.size()), tlas_max_instances_);
    std::memcpy(dst, static_brush_instances_.data(),
                sizeof(VkAccelerationStructureInstanceKHR) * static_n);
    if (mats && static_n > 0) {
        std::memcpy(mats, static_brush_materials_.data(),
                    sizeof(glm::vec4) * 3 * static_n);
    }
    if (prevs && static_n > 0) {
        std::memcpy(prevs, static_brush_worlds_.data(),
                    sizeof(glm::mat4) * static_n);
    }
    uint32_t n = static_n;

    auto write_instance = [&](const glm::mat4& m, const glm::mat4& prev_m,
                              const glm::vec4& color,
                              const glm::vec3& emissive, bool full_emissive,
                              VkDeviceAddress blas_ref,
                              const glm::vec4& tex_params,
                              uint32_t mask = 0xFF) {
        if (n >= tlas_max_instances_) return;
        VkAccelerationStructureInstanceKHR inst{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                inst.transform.matrix[row][col] = m[col][row];
            }
        }
        inst.instanceCustomIndex = n;
        // Mask convention:
        //   bit 0 = "shadow caster / AO occluder"
        //   bit 1 = "visible to GI / reflection"
        // Shadow + AO rays use cull-mask 0x01 (excludes anything without
        // bit 0 set — sparks + bullets are mask 0xFE so they're skipped);
        // GI + reflection rays use 0xFF and see everything.
        inst.mask = mask;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas_ref;
        dst[n] = inst;

        if (mats) {
            mats[n * 3 + 0] = glm::vec4(glm::vec3(color), full_emissive ? 1.0f : 0.0f);
            mats[n * 3 + 1] = glm::vec4(emissive, 0.0f);
            mats[n * 3 + 2] = tex_params;
        }
        if (prevs) {
            prevs[n] = prev_m;
        }
        ++n;
    };

    constexpr glm::vec4 kNoTex(-1.0f, -1.0f, 1.0f, 0.0f);

    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        const glm::mat4 scale_m = glm::scale(glm::mat4(1.0f), dyn_props_[i].full_size);
        glm::mat4 m = dr.world * scale_m;
        glm::mat4 prev_m = dr.prev_world * scale_m;
        glm::vec4 tex = tex_on
            ? glm::vec4(static_cast<float>(dyn_props_[i].tex_albedo),
                        static_cast<float>(dyn_props_[i].tex_normal),
                        dyn_props_[i].uv_scale, 1.0f)
            : kNoTex;
        glm::vec4 dyn_base = tex_on
            ? dyn_props_[i].color
            : dyn_props_[i].fallback_color;
        write_instance(m, prev_m, dyn_base, glm::vec3(0.0f), false,
                       blas_device_address_, tex);
    }
    // Sparks are intentionally NOT added to the TLAS. With kMaxParticles=384
    // and high fire rates, including them costs ~hundreds of instance writes
    // + AS-build work per frame, plus they get traversed by every GI and
    // reflection ray for negligible visual contribution (sub-mm cylinders
    // moving fast). Their visual is fully handled by the trail rasterization
    // in draw_spark_trails — far cheaper, and indistinguishable on screen.

    for (const auto& p : projectiles_) {
        glm::mat4 world;
        if (!physics_->get_body_world_matrix(p.body_id, world)) continue;
        glm::vec3 pos(world[3]);
        glm::vec3 vel = physics_->get_linear_velocity(p.body_id);
        glm::vec3 dir = glm::length(vel) > 1e-3f ? glm::normalize(vel) : p.initial_dir;
        glm::mat4 align = align_local_y_to(pos, dir);
        glm::mat4 scale_m = glm::scale(glm::mat4(1.0f),
                                       glm::vec3(p.radius, p.half_length, p.radius));
        glm::mat4 m = align * scale_m;
        // 0xFE: bullets are sub-mm cylinders moving at 600 m/s — their
        // contact-shadow streaks alias visibly in the soft-shadow penumbra
        // before they hit. Reflections/GI still see them so the tracer line
        // shows up in shiny pedestals.
        write_instance(m, m, p.color, p.emissive,
                       /*full_emissive=*/true, cylinder_blas_device_address_,
                       kNoTex, 0xFE);
    }

    VkAccelerationStructureGeometryInstancesDataKHR inst_data{};
    inst_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst_data.arrayOfPointers = VK_FALSE;
    inst_data.data.deviceAddress = instances_buffer_address_;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = inst_data;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    // UPDATE mode if instance count is unchanged; BUILD mode otherwise.
    const bool can_update = (n == prev_tlas_n_) && rt_initialized_;
    VkAccelerationStructureBuildGeometryInfoKHR bgi{};
    bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    bgi.mode = can_update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                          : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.srcAccelerationStructure = can_update ? tlas_ : VK_NULL_HANDLE;
    bgi.dstAccelerationStructure = tlas_;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;
    bgi.scratchData.deviceAddress = tlas_scratch_address_;
    prev_tlas_n_ = n;

    vmaFlushAllocation(allocator_, instances_alloc_,  0,
                       sizeof(VkAccelerationStructureInstanceKHR) * n);
    vmaFlushAllocation(allocator_, materials_alloc_, 0,
                       sizeof(glm::vec4) * 3 * n);
    vmaFlushAllocation(allocator_, prev_transforms_alloc_, 0,
                       sizeof(glm::mat4) * n);

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = n;
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
    g_rt.cmd_build_as(cmd, 1, &bgi, &p_range);

    last_tlas_n_ = n;

    // No in-CB memory barrier here — this command buffer is submitted on the
    // compute queue, and the cross-queue dependency to the graphics queue's
    // FRAGMENT_SHADER stage (where cube.frag reads the TLAS) is carried by
    // the binary semaphore signaled by this submission. Binary semaphores
    // implicitly carry both execution AND memory dependencies, making the
    // AS write visible to the graphics queue's reads. Adding an explicit
    // intra-queue barrier with dstStage=FRAGMENT_SHADER would also be a spec
    // error on a compute-only queue (which doesn't support that stage).
}

} // namespace qlike
