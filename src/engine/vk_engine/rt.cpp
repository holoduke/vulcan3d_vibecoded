// Ray tracing acceleration structures + per-frame TLAS rebuild. The cube
// and cylinder BLAS are built once at init from their meshes; the TLAS is
// host-mapped (768 instances) and rewritten every frame from the static
// brush bake + DynRender cache + particles + projectiles. UPDATE mode is
// used when the instance count is unchanged (the common case); a full
// BUILD only happens on count changes (spawn / despawn).

#include "engine/vk_engine/internal.h"
#include "engine/terrain.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qlike {

// Sentinel instanceCustomIndex for the merged-static-BLAS TLAS instance.
// cube.frag's GI loop sees this and switches to per-primitive material
// lookup (`materials[gl_PrimitiveID / 12]`) instead of per-instance
// (`materials[customIndex]`). Picked at the top of the 24-bit range so it
// never collides with a legitimate materials-buffer slot.
inline constexpr uint32_t kStaticBlasInstSentinel = 0xFFFFFFu;

// Tris per cube вЂ” shader does `prim_id / kCubeTrisPerBox` to recover the
// brush index inside the merged BLAS. Mirrored as `12` in cube.frag.
inline constexpr uint32_t kCubeTrisPerBox = 12;

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
    // Terrain BLAS вЂ” same lambda, since the terrain mesh is just a
    // standard Vertex/uint32 indexed mesh (built in init_world via
    // build_terrain_mesh).
    if (terrain_mesh_.index_count > 0) {
        build_mesh_blas(terrain_mesh_, "terrain",
                        terrain_blas_buffer_, terrain_blas_alloc_,
                        terrain_blas_, terrain_blas_device_address_);
    }
    // Single big BLAS over the entire static castle. Replaces ~190 small
    // per-brush TLAS instances with one вЂ” drops top-level traversal cost
    // and (more importantly) keeps RT load below the GPU TDR threshold
    // when many dyn props + projectiles + particles share the TLAS.
    bake_merged_static_blas();

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
        // Device-local with BAR-mapped host write. The AS-build reads
        // this every frame on the compute queue; over PCIe it would
        // burn 1-3 ms per frame on large TLAS rebuilds. AUTO_PREFER_DEVICE
        // + HOST_ACCESS_SEQUENTIAL_WRITE asks VMA for the resizable BAR
        // path on supported GPUs, falling back to internal staging.
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
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

    // Per-instance material buffer. Read by every cube.frag fragment as
    // an SSBO вЂ” placing it in DEVICE_LOCAL memory keeps that read off
    // PCIe. AUTO_PREFER_DEVICE + HOST_ACCESS_SEQUENTIAL_WRITE asks VMA
    // to use BAR-mapped device memory if the GPU supports it, falling
    // back to internal staging otherwise. Three vec4 per entry: color
    // (rgb)+full_emissive(a); emissive(rgb)+reserved; tex_params
    // (albedo idx, normal idx, uv scale, reserved).
    //
    // After the static-brush BLAS merge, customIndex ranges from 0
    // (per-brush material via gl_PrimitiveID/12 вЂ” the merged-BLAS
    // instance itself uses the sentinel custom_index) up to
    // M + tlas_max_instances_ for dynamic entries. Size accordingly so
    // dynamic slot writes never overflow.
    const uint32_t kStaticMatSlots = static_cast<uint32_t>(world_.brushes.size());
    // +1 reserves a slot for the terrain instance (Phase 1 single
    // heightmap mesh) right after the per-brush materials.
    const uint32_t kTotalMatSlots  = kStaticMatSlots + 1u + tlas_max_instances_;
    {
        VkDeviceSize size = static_cast<VkDeviceSize>(kTotalMatSlots) *
                            (3u * sizeof(glm::vec4));
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
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
    //
    // Sized to match the materials buffer's customIndex range so dynamic
    // entries with customIndex = M + i can write without overflow.
    {
        VkDeviceSize size = static_cast<VkDeviceSize>(kTotalMatSlots) *
                            sizeof(glm::mat4);
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        // Device-local with BAR-mapped host write вЂ” same rationale as
        // materials_buffer_ above (read by every cube.frag fragment).
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
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
        // Scratch must hold whichever of {build, update} is bigger вЂ” the same
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
            // we include CONCURRENT here for symmetry вЂ” the BLAS build at
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
    if (merged_static_blas_) g_rt.destroy_as(device_, merged_static_blas_, nullptr);
    if (terrain_blas_)       g_rt.destroy_as(device_, terrain_blas_, nullptr);
    if (tlas_scratch_buffer_) vmaDestroyBuffer(allocator_, tlas_scratch_buffer_, tlas_scratch_alloc_);
    if (instances_buffer_)    vmaDestroyBuffer(allocator_, instances_buffer_, instances_alloc_);
    if (materials_buffer_)    vmaDestroyBuffer(allocator_, materials_buffer_, materials_alloc_);
    if (prev_transforms_buffer_) vmaDestroyBuffer(allocator_, prev_transforms_buffer_, prev_transforms_alloc_);
    if (tlas_buffer_)         vmaDestroyBuffer(allocator_, tlas_buffer_, tlas_alloc_);
    if (blas_buffer_)         vmaDestroyBuffer(allocator_, blas_buffer_, blas_alloc_);
    if (cylinder_blas_buffer_) vmaDestroyBuffer(allocator_, cylinder_blas_buffer_, cylinder_blas_alloc_);
    if (merged_static_blas_buffer_) vmaDestroyBuffer(allocator_, merged_static_blas_buffer_, merged_static_blas_alloc_);
    if (merged_static_mesh_.vertex_buffer) destroy_mesh(allocator_, merged_static_mesh_);
    if (terrain_blas_buffer_) vmaDestroyBuffer(allocator_, terrain_blas_buffer_, terrain_blas_alloc_);
    if (terrain_mesh_.vertex_buffer) destroy_mesh(allocator_, terrain_mesh_);
    destroy_terrain_chunks(device_, allocator_, terrain_chunks_);
    tlas_ = blas_ = cylinder_blas_ = merged_static_blas_ = terrain_blas_ = VK_NULL_HANDLE;
    tlas_buffer_ = blas_buffer_ = cylinder_blas_buffer_ = merged_static_blas_buffer_ =
        terrain_blas_buffer_ =
        tlas_scratch_buffer_ = instances_buffer_ = materials_buffer_ =
        prev_transforms_buffer_ = VK_NULL_HANDLE;
    tlas_alloc_ = blas_alloc_ = cylinder_blas_alloc_ = merged_static_blas_alloc_ =
        terrain_blas_alloc_ =
        tlas_scratch_alloc_ = instances_alloc_ = materials_alloc_ =
        prev_transforms_alloc_ = nullptr;
    cylinder_blas_device_address_ = 0;
    merged_static_blas_device_address_ = 0;
    merged_static_brush_count_ = 0;
    terrain_blas_device_address_ = 0;
    materials_mapped_ = nullptr;
    prev_transforms_mapped_ = nullptr;
    rt_initialized_ = false;
}

void VulkanEngine::bake_merged_static_blas() {
    // Concatenate every world_.brush's cube into one world-space triangle
    // soup. The unit-cube vertex template lives in cube_mesh_ but we need
    // *positions only* in world space вЂ” normals and UVs are unused for RT
    // (cube.frag's primary shading reads from raster, RT only sees triangle
    // intersections). We still pack the full Vertex struct because that's
    // the only path our `create_mesh_from_data` upload helper takes.
    //
    // 24 verts Г— N brushes; 36 indices Г— N brushes (12 tris Г— N brushes).
    // For a 200-brush castle that's 4800 verts / 7200 indices вЂ” trivial.
    const uint32_t M = static_cast<uint32_t>(world_.brushes.size());
    if (M == 0) {
        merged_static_brush_count_ = 0;
        return;
    }

    // Cube template (positions only вЂ” normals/UVs zeroed since RT ignores).
    constexpr float h = 0.5f;
    static const glm::vec3 kCubePos[24] = {
        // +X
        { h,-h,-h}, { h, h,-h}, { h, h, h}, { h,-h, h},
        // -X
        {-h,-h, h}, {-h, h, h}, {-h, h,-h}, {-h,-h,-h},
        // +Y
        {-h, h,-h}, {-h, h, h}, { h, h, h}, { h, h,-h},
        // -Y
        {-h,-h, h}, {-h,-h,-h}, { h,-h,-h}, { h,-h, h},
        // +Z
        {-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h},
        // -Z
        { h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h},
    };
    static const uint32_t kCubeIdx[36] = {
        0, 1, 2,    0, 2, 3,
        4, 5, 6,    4, 6, 7,
        8, 9,10,    8,10,11,
       12,13,14,   12,14,15,
       16,17,18,   16,18,19,
       20,21,22,   20,22,23,
    };

    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;
    verts.reserve(M * 24);
    indices.reserve(M * 36);
    // CRITICAL: brush ordering in the merged buffer must match the order the
    // shader recovers via `gl_PrimitiveID / 12`. We push in `world_.brushes`
    // order so brush[i] occupies primitive range [i*12, i*12 + 12).
    for (uint32_t bi = 0; bi < M; ++bi) {
        const auto& b = world_.brushes[bi];
        const uint32_t v_off = bi * 24;
        for (int v = 0; v < 24; ++v) {
            glm::vec3 wp = b.center + kCubePos[v] * b.size;
            verts.push_back({wp, glm::vec3(0.0f), glm::vec2(0.0f)});
        }
        for (int k = 0; k < 36; ++k) {
            indices.push_back(v_off + kCubeIdx[k]);
        }
    }

    merged_static_mesh_ = create_mesh_from_data(
        device_, allocator_, graphics_queue_, graphics_queue_family_,
        verts.data(), static_cast<uint32_t>(verts.size()),
        indices.data(), static_cast<uint32_t>(indices.size()));
    merged_static_brush_count_ = M;

    // Build a BLAS over the merged triangle soup. PREFER_FAST_TRACE +
    // single-build (no UPDATE flag) вЂ” this geometry never changes after
    // level load.
    const uint32_t tri_count = merged_static_mesh_.index_count / 3;
    VkDeviceAddress vbo_addr = buffer_device_address(device_, merged_static_mesh_.vertex_buffer);
    VkDeviceAddress ibo_addr = buffer_device_address(device_, merged_static_mesh_.index_buffer);

    VkAccelerationStructureGeometryTrianglesDataKHR tri{};
    tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.vertexData.deviceAddress = vbo_addr;
    tri.vertexStride = sizeof(Vertex);
    tri.maxVertex = merged_static_mesh_.vertex_count - 1;
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
    // ALLOW_COMPACTION lets us re-create the AS at its actual minimum size
    // after the initial build. NVIDIA's BVH builder typically over-allocates
    // by 30-50% for worst-case fits; compacting reclaims that VRAM AND tends
    // to give a slightly faster traversal because the post-compact tree fits
    // tighter in cache. Required because the merged BLAS for ~1800 tris is
    // permanent вЂ” we'd carry the over-allocation forever otherwise.
    bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;

    uint32_t prim_count = tri_count;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    g_rt.get_as_build_sizes(device_,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &prim_count, &sizes);

    // Mirror init_rt()'s rt_sharing decision so the BLAS storage is visible
    // from BOTH the graphics queue (during cube.frag ray traversal) and the
    // compute queue (during the per-frame TLAS rebuild that references this
    // BLAS by device address). Without this, NVIDIA's driver eventually
    // surfaces the EXCLUSIVE-with-cross-queue-access as a DEVICE_LOST.
    const uint32_t shared_families[2] = {
        graphics_queue_family_, compute_queue_family_
    };
    const VkSharingMode rt_sharing = compute_queue_distinct_
        ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    const uint32_t rt_share_count = compute_queue_distinct_ ? 2u : 0u;
    const uint32_t* rt_share_indices = compute_queue_distinct_ ? shared_families : nullptr;
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
                                 &merged_static_blas_buffer_,
                                 &merged_static_blas_alloc_, nullptr),
                 "merged static blas buffer");
    }
    VkAccelerationStructureCreateInfoKHR aci_blas{};
    aci_blas.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    aci_blas.buffer = merged_static_blas_buffer_;
    aci_blas.size = sizes.accelerationStructureSize;
    aci_blas.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vk_check(g_rt.create_as(device_, &aci_blas, nullptr, &merged_static_blas_),
             "vkCreateAccelerationStructureKHR merged static blas");
    bgi.dstAccelerationStructure = merged_static_blas_;

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
                 "merged static blas scratch");
    }
    bgi.scratchData.deviceAddress = buffer_device_address(device_, scratch);

    // Single-shot query pool to read back the post-build compacted size.
    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qpci.queryCount = 1;
    VkQueryPool qpool = VK_NULL_HANDLE;
    vk_check(vkCreateQueryPool(device_, &qpci, nullptr, &qpool),
             "compaction query pool");

    // Build + write compacted-size query in one submit. The barrier between
    // the build and the query is mandatory: writeProperties needs the build
    // result fully visible.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkCmdResetQueryPool(cb, qpool, 0, 1);
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = tri_count;
        const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
        g_rt.cmd_build_as(cb, 1, &bgi, &p_range);

        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &dep);

        g_rt.cmd_write_as_props(cb, 1, &merged_static_blas_,
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            qpool, 0);
    });

    VkDeviceSize compacted_size = 0;
    vk_check(vkGetQueryPoolResults(device_, qpool, 0, 1,
                                   sizeof(VkDeviceSize), &compacted_size,
                                   sizeof(VkDeviceSize),
                                   VK_QUERY_RESULT_WAIT_BIT |
                                   VK_QUERY_RESULT_64_BIT),
             "compaction size query result");
    vkDestroyQueryPool(device_, qpool, nullptr);

    if (compacted_size > 0 && compacted_size < sizes.accelerationStructureSize) {
        // Allocate a new tighter buffer + AS, copy with COMPACT mode, swap in.
        VkBuffer compact_buf = VK_NULL_HANDLE;
        VmaAllocation compact_alloc = nullptr;
        {
            VkBufferCreateInfo bci{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr, .flags = 0,
                .size = compacted_size,
                .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                .sharingMode = rt_sharing,
                .queueFamilyIndexCount = rt_share_count,
                .pQueueFamilyIndices = rt_share_indices,
            };
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                     &compact_buf, &compact_alloc, nullptr),
                     "merged static blas compact buffer");
        }
        VkAccelerationStructureKHR compact_as = VK_NULL_HANDLE;
        VkAccelerationStructureCreateInfoKHR aci_c{};
        aci_c.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        aci_c.buffer = compact_buf;
        aci_c.size = compacted_size;
        aci_c.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vk_check(g_rt.create_as(device_, &aci_c, nullptr, &compact_as),
                 "compact AS");

        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            VkCopyAccelerationStructureInfoKHR cinfo{};
            cinfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
            cinfo.src = merged_static_blas_;
            cinfo.dst = compact_as;
            cinfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
            g_rt.cmd_copy_as(cb, &cinfo);
        });

        const VkDeviceSize before = sizes.accelerationStructureSize;
        // Swap and free the original. Safe to do here because the
        // one_time_submit above already returned with vkQueueWaitIdle.
        g_rt.destroy_as(device_, merged_static_blas_, nullptr);
        vmaDestroyBuffer(allocator_, merged_static_blas_buffer_, merged_static_blas_alloc_);
        merged_static_blas_ = compact_as;
        merged_static_blas_buffer_ = compact_buf;
        merged_static_blas_alloc_ = compact_alloc;
        log::infof("BLAS (merged-static) compacted: %llu -> %llu bytes (%.0f%%)",
                   static_cast<unsigned long long>(before),
                   static_cast<unsigned long long>(compacted_size),
                   100.0 * static_cast<double>(compacted_size) /
                          static_cast<double>(before));
    }

    vmaDestroyBuffer(allocator_, scratch, scratch_alloc);
    merged_static_blas_device_address_ = as_device_address(device_, merged_static_blas_);
    log::infof("BLAS (merged-static): %u brushes, %u tris", M, tri_count);
}

void VulkanEngine::bake_static_brushes() {
    constexpr glm::vec4 kNoTex(-1.0f, -1.0f, 1.0f, 0.0f);
    const bool tex_on = rt_.textures_enabled;
    const bool merged = rt_.use_merged_static_blas &&
                        merged_static_blas_device_address_ != 0;

    static_brush_instances_.clear();
    static_brush_materials_.clear();
    static_brush_worlds_.clear();
    static_brush_materials_.reserve(world_.brushes.size() * 3);
    static_brush_worlds_.reserve(world_.brushes.size());

    if (merged && !world_.brushes.empty()) {
        // Merged path: one TLAS instance covering the whole castle. Material
        // lookup in cube.frag is routed through gl_PrimitiveID/12 by the
        // sentinel custom index.
        static_brush_instances_.reserve(1);
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[2][2] = 1.0f;
        inst.instanceCustomIndex = kStaticBlasInstSentinel;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = merged_static_blas_device_address_;
        static_brush_instances_.push_back(inst);
    } else {
        // Legacy path: one TLAS instance per brush, all pointing at the
        // shared cube BLAS. customIndex matches the brush index so the
        // shader's `materials[customIndex]` keeps working.
        static_brush_instances_.reserve(world_.brushes.size());
        uint32_t i = 0;
        for (const auto& b : world_.brushes) {
            glm::mat4 m = glm::translate(glm::mat4(1.0f), b.center) *
                          glm::scale(glm::mat4(1.0f), b.size);
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
            ++i;
        }
    }

    // Per-brush material table вЂ” populated identically for both paths so
    // the materials[brush_idx] layout is consistent.
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
        static_brush_materials_.push_back(
            glm::vec4(glm::vec3(base), b.full_emissive ? 1.0f : 0.0f));
        static_brush_materials_.push_back(glm::vec4(b.emissive, 0.0f));
        static_brush_materials_.push_back(tex);
    }
    // Terrain instance + material вЂ” appended at slot M (one entry per
    // call to bake_static_brushes). cube.frag's GI hit recovers the
    // terrain material via materials[customIndex] just like dynamic
    // instances. customIndex == M (= world_.brushes.size()).
    if (terrain_blas_device_address_ != 0) {
        VkAccelerationStructureInstanceKHR tinst{};
        tinst.transform.matrix[0][0] = 1.0f;
        tinst.transform.matrix[1][1] = 1.0f;
        tinst.transform.matrix[2][2] = 1.0f;
        tinst.instanceCustomIndex = static_cast<uint32_t>(world_.brushes.size());
        // Mask convention extended for the terrain receiver case:
        //   bit 0 = "shadow caster, ray-mask 0x01 sees it" (legacy)
        //   bit 1 = "non-terrain shadow caster" вЂ” terrain receivers
        //           use ray mask 0x02 to fire RT shadow rays that
        //           skip the terrain BLAS (so the BLAS-vs-LOD-raster
        //           mismatch can't false-hit per triangle), but still
        //           hit castle / dyn-props for box-shadow-on-ground.
        // Terrain BLAS keeps bit 0 (non-terrain receivers still see it
        // for mountain-shadow-on-castle) but clears bit 1.
        tinst.mask = 0x01;
        tinst.instanceShaderBindingTableRecordOffset = 0;
        tinst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        tinst.accelerationStructureReference = terrain_blas_device_address_;
        static_brush_instances_.push_back(tinst);
        // Material table slot for terrain.
        static_brush_materials_.push_back(glm::vec4(terrain_color_, 0.0f));
        static_brush_materials_.push_back(glm::vec4(0.0f));
        static_brush_materials_.push_back(kNoTex);
        static_brush_worlds_.push_back(glm::mat4(1.0f));
    }
    static_brush_tex_on_ = tex_on;
    static_brush_dirty_ = false;
}

void VulkanEngine::rebuild_tlas(VkCommandBuffer cmd) {
    if (!rt_initialized_) return;

    // Skip the entire build when nothing in the dynamic set moved since
    // last frame. Static brushes are baked + memcpyed once per change;
    // dynamic instances are the only thing the per-frame TLAS rebuild
    // exists to track. Hash the (translation, projectile pose) state and
    // compare. On a static scene (no dyn motion, no projectiles) this
    // skips the AS-build + scratch + cross-queue semaphore signal path
    // entirely — frees the compute queue most frames.
    auto mix64 = [](uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    };
    auto hash_f = [](float f) {
        uint32_t u; std::memcpy(&u, &f, 4); return uint64_t(u);
    };
    uint64_t sig = 0xcbf29ce484222325ULL;
    sig = mix64(sig, dyn_props_.size());
    sig = mix64(sig, projectiles_.size());
    sig = mix64(sig, static_brush_dirty_ ? 1u : 0u);
    sig = mix64(sig, rt_.textures_enabled ? 1u : 0u);
    sig = mix64(sig, rt_.use_merged_static_blas ? 1u : 0u);
    for (size_t i = 0; i < dyn_render_cache_.size(); ++i) {
        const DynRender& dr = dyn_render_cache_[i];
        if (!dr.valid) continue;
        sig = mix64(sig, hash_f(dr.world[3].x));
        sig = mix64(sig, hash_f(dr.world[3].y));
        sig = mix64(sig, hash_f(dr.world[3].z));
        sig = mix64(sig, hash_f(dr.world[0].x));   // rotation hint
        sig = mix64(sig, hash_f(dr.world[2].z));
    }
    for (const auto& p : projectiles_) {
        glm::mat4 w;
        if (!physics_->get_body_world_matrix_h(p.jolt_handle, w)) continue;
        sig = mix64(sig, hash_f(w[3].x));
        sig = mix64(sig, hash_f(w[3].y));
        sig = mix64(sig, hash_f(w[3].z));
    }
    if (tlas_first_build_done_ && sig == prev_dyn_signature_) {
        return;   // tlas_ contents still valid — skip build + signal
    }
    prev_dyn_signature_ = sig;
    tlas_first_build_done_ = true;

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
    // Detect a flipped use_merged_static_blas toggle: the cached static
    // instances were emitted under the OLD value, so re-bake to match.
    const bool merged_active = !static_brush_instances_.empty() &&
        static_brush_instances_.front().instanceCustomIndex == kStaticBlasInstSentinel;
    const bool merged_wanted = rt_.use_merged_static_blas &&
                               merged_static_blas_device_address_ != 0 &&
                               !world_.brushes.empty();
    if (static_brush_dirty_ || static_brush_tex_on_ != tex_on ||
        merged_active != merged_wanted) {
        bake_static_brushes();
        // Force a full TLAS rebuild (not UPDATE) on the next frame because
        // the instance count just changed.
        prev_tlas_n_ = UINT32_MAX;
        static_brush_uploaded_ = false;
    }

    auto* dst = static_cast<VkAccelerationStructureInstanceKHR*>(instances_mapped_);
    auto* mats = static_cast<glm::vec4*>(materials_mapped_);  // 3 vec4 per material
    auto* prevs = static_cast<glm::mat4*>(prev_transforms_mapped_);

    // Materials/prev_transforms layout:
    //   slots [0..M-1]   per-brush data for the castle (always populated)
    //   slot M           terrain (only if terrain BLAS exists)
    //   slots [M+T..]    dynamic instances (write_instance fills these)
    const uint32_t M = static_cast<uint32_t>(world_.brushes.size());
    const uint32_t T = (terrain_blas_device_address_ != 0) ? 1u : 0u;
    const uint32_t static_mats = M + T;
    // Static brush materials + worlds change rarely (texture toggle,
    // merged-vs-per-brush swap, brush count change). Gate the per-frame
    // memcpy of ~200 KB on the dirty flag вЂ” saves PCIe traffic when the
    // static set is stable.
    if (!static_brush_uploaded_) {
        if (mats && static_mats > 0) {
            std::memcpy(mats, static_brush_materials_.data(),
                        sizeof(glm::vec4) * 3 * static_mats);
        }
        if (prevs && static_mats > 0) {
            std::memcpy(prevs, static_brush_worlds_.data(),
                        sizeof(glm::mat4) * static_mats);
        }
        static_brush_uploaded_ = true;
    }
    // TLAS instance block: 1 entry (merged) or M entries (legacy) +
    // optionally 1 terrain entry. memcpy the prebuilt instances directly.
    const uint32_t static_n = static_cast<uint32_t>(
        std::min<size_t>(static_brush_instances_.size(), tlas_max_instances_));
    if (static_n > 0) {
        std::memcpy(dst, static_brush_instances_.data(),
                    sizeof(VkAccelerationStructureInstanceKHR) * static_n);
    }
    uint32_t n = static_n;
    // Dynamic customIndex starts past the static-mat block (brushes+terrain).
    uint32_t mat_idx = static_mats;

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
        // customIndex is the materials buffer slot, NOT the TLAS slot. The
        // shader reads materials[customIndex] for dynamic hits. The merged
        // static BLAS uses kStaticBlasInstSentinel and is handled separately.
        inst.instanceCustomIndex = mat_idx;
        // Mask convention:
        //   bit 0 = "shadow caster / AO occluder"
        //   bit 1 = "visible to GI / reflection"
        // Shadow + AO rays use cull-mask 0x01 (excludes anything without
        // bit 0 set вЂ” sparks + bullets are mask 0xFE so they're skipped);
        // GI + reflection rays use 0xFF and see everything.
        inst.mask = mask;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas_ref;
        dst[n] = inst;

        if (mats) {
            mats[mat_idx * 3 + 0] = glm::vec4(glm::vec3(color), full_emissive ? 1.0f : 0.0f);
            mats[mat_idx * 3 + 1] = glm::vec4(emissive, 0.0f);
            mats[mat_idx * 3 + 2] = tex_params;
        }
        if (prevs) {
            prevs[mat_idx] = prev_m;
        }
        ++n;
        ++mat_idx;
    };

    constexpr glm::vec4 kNoTex(-1.0f, -1.0f, 1.0f, 0.0f);

    // Angular-size TLAS culling (Khronos hybrid-RT best practice; Wolfenstein
    // uses the same heuristic). A dyn-prop whose AABB subtends less than this
    // angle from the camera is small enough that its shadow / GI contribution
    // is sub-pixel; skipping it from the TLAS saves traversal cost and BLAS-
    // instance overhead. Threshold в‰€ 0.5В° = ~tan(0.5В°) в‰€ 0.0087 rad.
    constexpr float kMinAngularSize = 0.0087f;
    const glm::vec3 cam_pos = player_.eye_position();
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        glm::vec3 prop_pos = glm::vec3(dr.world[3]);
        float dist = glm::distance(prop_pos, cam_pos);
        float radius = 0.5f * glm::length(dyn_props_[i].full_size);
        if (dist > radius * 2.0f &&
            (radius / std::max(dist, 0.001f)) < kMinAngularSize) {
            continue;
        }
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
    // in draw_spark_trails вЂ” far cheaper, and indistinguishable on screen.

    for (const auto& p : projectiles_) {
        glm::mat4 world;
        if (!physics_->get_body_world_matrix_h(p.jolt_handle, world)) continue;
        glm::vec3 pos(world[3]);
        glm::vec3 vel = physics_->get_linear_velocity_h(p.jolt_handle);
        glm::vec3 dir = glm::length(vel) > 1e-3f ? glm::normalize(vel) : p.initial_dir;
        glm::mat4 align = align_local_y_to(pos, dir);
        glm::mat4 scale_m = glm::scale(glm::mat4(1.0f),
                                       glm::vec3(p.radius, p.half_length, p.radius));
        glm::mat4 m = align * scale_m;
        // Mask 0xFC: clears bit 0 (no shadow-caster for non-terrain
        // receivers) AND bit 1 (no shadow-caster for terrain receivers).
        // Bullets are sub-mm cylinders moving at 600 m/s вЂ” their contact
        // shadow streaks alias visibly in the soft-shadow penumbra. GI/
        // reflection rays still see them via mask 0xFF so the tracer
        // line shows up in shiny surfaces.
        write_instance(m, m, p.color, p.emissive,
                       /*full_emissive=*/true, cylinder_blas_device_address_,
                       kNoTex, 0xFC);
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
    // Materials/prev_transforms layout: [0..M-1] static brushes,
    // [M..mat_idx-1] dynamic. Flush the full populated prefix.
    if (mat_idx > 0) {
        vmaFlushAllocation(allocator_, materials_alloc_, 0,
                           sizeof(glm::vec4) * 3 * mat_idx);
        vmaFlushAllocation(allocator_, prev_transforms_alloc_, 0,
                           sizeof(glm::mat4) * mat_idx);
    }

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = n;
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
    g_rt.cmd_build_as(cmd, 1, &bgi, &p_range);

    last_tlas_n_ = n;

    // No in-CB memory barrier here вЂ” this command buffer is submitted on the
    // compute queue, and the cross-queue dependency to the graphics queue's
    // FRAGMENT_SHADER stage (where cube.frag reads the TLAS) is carried by
    // the binary semaphore signaled by this submission. Binary semaphores
    // implicitly carry both execution AND memory dependencies, making the
    // AS write visible to the graphics queue's reads. Adding an explicit
    // intra-queue barrier with dstStage=FRAGMENT_SHADER would also be a spec
    // error on a compute-only queue (which doesn't support that stage).
}

} // namespace qlike
