// Castle mesh parts (round towers, conical roofs, gate arch).
//
// Level::meshes is built in game/level.cpp in castle-local space; init_world
// lifts it onto the plateau and calls init_castle_meshes() to upload it.
// init_rt() calls build_castle_mesh_blas() (one BLAS per part); the parts
// are written into the TLAS every rebuild like the gate doors. Raster goes
// through the regular world pipeline with grass_params.w = -1, which cube.frag
// / cube.vert treat as "static mesh": vertex-UV texturing with a derivative
// tangent frame, and none of the axis-aligned-box-only paths (box relief,
// SPOM, corner softening, shell extrusion).

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"

#include <cstring>

namespace qlike {

static_assert(sizeof(game::LevelVertex) == sizeof(Vertex),
              "LevelVertex must match the engine Vertex layout");

void VulkanEngine::init_castle_meshes() {
    for (const auto& lm : world_.meshes) {
        if (lm.verts.empty() || lm.idx.empty()) continue;
        CastleMeshGpu g{};
        g.mesh = create_mesh_from_data(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            reinterpret_cast<const Vertex*>(lm.verts.data()),
            static_cast<uint32_t>(lm.verts.size()),
            lm.idx.data(), static_cast<uint32_t>(lm.idx.size()));
        g.tex_albedo = lm.tex_albedo;
        g.tex_normal = lm.tex_normal;
        g.tint = lm.tint;
        g.fallback = lm.fallback;
        g.bmin = lm.bmin;
        g.bmax = lm.bmax;
        castle_meshes_.push_back(g);
        log::infof("[castle] mesh part: %zu verts, %zu tris (albedo %d)",
                   lm.verts.size(), lm.idx.size() / 3, lm.tex_albedo);
    }
}

void VulkanEngine::build_castle_mesh_blas() {
    const uint32_t shared_families[2] = { graphics_queue_family_, compute_queue_family_ };
    const VkSharingMode rt_sharing = compute_queue_distinct_
        ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    for (auto& g : castle_meshes_) {
        if (g.blas) continue;
        const uint32_t tri_count = g.mesh.index_count / 3;
        VkAccelerationStructureGeometryTrianglesDataKHR tri{};
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = buffer_device_address(device_, g.mesh.vertex_buffer);
        tri.vertexStride = sizeof(Vertex);
        tri.maxVertex = g.mesh.vertex_count - 1;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = buffer_device_address(device_, g.mesh.index_buffer);

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

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        g_rt.get_as_build_sizes(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &bgi, &tri_count, &sizes);

        {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizes.accelerationStructureSize;
            bci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bci.sharingMode = rt_sharing;
            bci.queueFamilyIndexCount = compute_queue_distinct_ ? 2u : 0u;
            bci.pQueueFamilyIndices = compute_queue_distinct_ ? shared_families : nullptr;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &g.blas_buf,
                                     &g.blas_alloc, nullptr),
                     "castle mesh blas buffer");
        }
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = g.blas_buf;
        ci.size = sizes.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vk_check(g_rt.create_as(device_, &ci, nullptr, &g.blas), "castle mesh blas");
        bgi.dstAccelerationStructure = g.blas;

        VkBuffer scratch = VK_NULL_HANDLE;
        VmaAllocation scratch_alloc = nullptr;
        {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizes.buildScratchSize;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &scratch,
                                     &scratch_alloc, nullptr),
                     "castle mesh blas scratch");
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
        g.blas_addr = as_device_address(device_, g.blas);
    }
}

void VulkanEngine::destroy_castle_meshes() {
    for (auto& g : castle_meshes_) {
        if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
        if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
        if (g.mesh.vertex_buffer) destroy_mesh(allocator_, g.mesh);
    }
    castle_meshes_.clear();
}

void VulkanEngine::draw_castle_meshes(VkCommandBuffer cmd, const glm::mat4& vp,
                                      const glm::mat4& prev_vp, const Frustum& fr,
                                      bool depth_only, VkPipelineLayout shadow_layout) {
    if (castle_meshes_.empty()) return;
    const bool tex_on = rt_.textures_enabled;
    VkDeviceSize off = 0;
    for (const auto& g : castle_meshes_) {
        if (!aabb_visible(fr, g.bmin, g.bmax)) continue;
        vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, g.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        if (shadow_layout) {
            glm::mat4 mvp = vp;   // model = identity (world-space verts)
            vkCmdPushConstants(cmd, shadow_layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &mvp);
        } else {
            PushConstants pc{};
            pc.mvp = vp;
            pc.model = glm::mat4(1.0f);
            pc.prev_mvp = prev_vp;
            // Static-mesh mode flag (see file header).
            pc.grass_params = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
            if (!depth_only) {
                pc.color = glm::vec4(tex_on ? g.tint : g.fallback, 1.0f);
                pc.emissive = glm::vec4(0.0f);
                pc.tex_params = tex_on
                    ? glm::vec4(static_cast<float>(g.tex_albedo),
                                static_cast<float>(g.tex_normal), 1.0f, 0.0f)
                    : glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
            }
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                               0, sizeof(PushConstants), &pc);
        }
        vkCmdDrawIndexed(cmd, g.mesh.index_count, 1, 0, 0, 0);
    }
}

} // namespace qlike
