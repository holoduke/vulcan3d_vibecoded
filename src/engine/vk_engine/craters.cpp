// Dynamic craters: big ground blasts (bombs, crashes) press a bowl with a
// raised, slightly irregular rim into the terrain. Everything that holds a
// copy of the ground height is updated BY REGION only (a sculpt-style full
// refresh would rebuild 2049^2 vertices, the whole heightfield and the
// horizon bake per explosion):
//   * terrain_data_.heights (CPU: gameplay height queries, sample_terrain_height)
//   * CLOD min/max pyramid (culling) for the touched nodes
//   * both height textures (absolute + sculpt delta), sub-rect copies
//   * the merged terrain mesh vertices (rows of the rect) + terrain BLAS
//     refit, so ray-traced sun shadows / GI see the hole (a raster surface
//     below the BLAS would self-shadow black)
//   * the Jolt heightfield (HeightFieldShape::SetHeights) - player, crates
//     and debris collide with the crater
//   * flowers / rocks: removed inside the bowl, re-seated on the slopes.
// Not on the castle plateau and not in the water.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"
#include "engine/physics_world.h"
#include "engine/terrain_clod.h"
#include "engine/vk_initializers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace qlike {

void VulkanEngine::make_crater(glm::vec3 pos, float size) {
    TerrainData& td = terrain_data_;
    if (td.heights.empty() || size < 0.6f) return;
    // The ray-marched terrain is procedural (FBM in the shader): a heightmap
    // crater would only sink the physics ground with no visible hole.
    if (rt_.terrain_raymarch_enabled) return;
    // Only at the surface (callers pass ground blasts, but stay defensive).
    if (pos.y > sample_terrain_height(pos.x, pos.z) + 4.0f) return;
    if (std::max(std::abs(pos.x), std::abs(pos.z)) < 26.0f) return;      // castle plateau
    const float h0 = sample_terrain_height(pos.x, pos.z);
    if (h0 < rt_.water_level + 0.4f) return;                              // water / shoreline
    const auto t0 = std::chrono::steady_clock::now();
    // Frames in flight: the previous frame may still read the heightmap
    // images, terrain vertices / BLAS and the clutter instances this edits
    // in place. A crater is a one-off event - drain the GPU first.
    vkDeviceWaitIdle(device_);

    auto& R = spacejet_rng_state_;
    const float Rc   = 2.4f + 2.4f * size;          // bowl radius (m): bomb ~4.8, crash ~7.9
    const float D    = 0.45f + 0.7f * size;         // depth at the centre
    const float rimh = 0.22f * D;                   // raised ejecta rim
    const float Rmax = Rc * 1.7f;
    const float phase = frand_range(R, 0.0f, 6.28f);
    // Height change at a point: paraboloid bowl + Gaussian rim, both with
    // a little angular irregularity so it doesn't look stamped.
    auto delta = [&](float x, float z) {
        const float dx = x - pos.x, dz = z - pos.z;
        const float r = std::sqrt(dx * dx + dz * dz);
        if (r >= Rmax) return 0.0f;
        const float ang = std::atan2(dz, dx);
        const float wob = 1.0f + 0.12f * std::sin(ang * 3.0f + phase) + 0.07f * std::sin(ang * 7.0f + phase * 2.1f);
        const float rr = r / (Rc * wob);
        float d = rr < 1.0f ? -D * (1.0f - rr * rr) : 0.0f;
        const float k = (rr - 1.0f) / 0.38f;
        d += rimh * std::exp(-k * k);
        return d;
    };

    const int W = td.dim + 1;
    auto to_ix = [&](float w, float o) { return int(std::floor((w - o) / td.cell)); };
    const int x0 = std::clamp(to_ix(pos.x - Rmax, td.origin_x) - 1, 0, W - 1);
    const int x1 = std::clamp(to_ix(pos.x + Rmax, td.origin_x) + 2, 0, W - 1);
    const int z0 = std::clamp(to_ix(pos.z - Rmax, td.origin_z) - 1, 0, W - 1);
    const int z1 = std::clamp(to_ix(pos.z + Rmax, td.origin_z) + 2, 0, W - 1);
    const float floor_h = rt_.water_level + 0.15f;   // never dig below the water plane
    for (int iz = z0; iz <= z1; ++iz)
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = td.origin_x + float(ix) * td.cell, wz = td.origin_z + float(iz) * td.cell;
            float& h = td.heights[size_t(iz) * size_t(W) + size_t(ix)];
            const float d = delta(wx, wz);
            if (d < 0.0f) h = std::max(h + d, std::min(h, floor_h));
            else          h += d;
        }

    // Culling pyramid for the touched nodes.
    update_clod_pyramid_region(terrain_clod_pyramid_, td.heights.data(), td.dim, td.cell, 16.0f,
                               x0, x1, z0, z1);

    // ---- GPU: height textures (sub-rect) + terrain vertices + BLAS refit ----
    const int rw = x1 - x0 + 1, rh = z1 - z0 + 1;
    // Vertex rect one cell wider: normals of the neighbours change too.
    const int vx0 = std::max(0, x0 - 1), vx1 = std::min(W - 1, x1 + 1);
    const int vz0 = std::max(0, z0 - 1), vz1 = std::min(W - 1, z1 + 1);
    const int vw = vx1 - vx0 + 1, vh = vz1 - vz0 + 1;
    const VkDeviceSize tex_bytes = VkDeviceSize(rw) * rh * sizeof(float);
    const VkDeviceSize vtx_bytes = VkDeviceSize(vw) * vh * sizeof(Vertex);
    const VkDeviceSize stage_bytes = tex_bytes * 2 + vtx_bytes;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = stage_bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer stage = VK_NULL_HANDLE; VmaAllocation stage_alloc = nullptr; VmaAllocationInfo sai{};
    if (vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, &sai) != VK_SUCCESS) return;
    auto* full = static_cast<float*>(sai.pMappedData);
    float* dlt = full + size_t(rw) * size_t(rh);
    auto* vtx = reinterpret_cast<Vertex*>(dlt + size_t(rw) * size_t(rh));
    const bool has_base = terrain_baseline_heights_.size() == td.heights.size();
    for (int z = 0; z < rh; ++z)
        for (int x = 0; x < rw; ++x) {
            const size_t i = size_t(z0 + z) * size_t(W) + size_t(x0 + x);
            full[size_t(z) * rw + x] = td.heights[i];
            dlt[size_t(z) * rw + x] = has_base ? td.heights[i] - terrain_baseline_heights_[i] : td.heights[i];
        }
    // Same layout as gen_chunk_verts (interior grid, row-major, 1-cell
    // central-difference normal, uv = world xz).
    auto H = [&](int x, int z) {
        x = std::clamp(x, 0, W - 1); z = std::clamp(z, 0, W - 1);
        return td.heights[size_t(z) * size_t(W) + size_t(x)];
    };
    for (int z = 0; z < vh; ++z)
        for (int x = 0; x < vw; ++x) {
            const int gx = vx0 + x, gz = vz0 + z;
            const glm::vec3 p(td.origin_x + float(gx) * td.cell, H(gx, gz), td.origin_z + float(gz) * td.cell);
            const int xm = std::max(0, gx - 1), xp = std::min(W - 1, gx + 1);
            const int zm = std::max(0, gz - 1), zp = std::min(W - 1, gz + 1);
            const float dhdx = (H(xp, gz) - H(xm, gz)) / (float(xp - xm) * td.cell);
            const float dhdz = (H(gx, zp) - H(gx, zm)) / (float(zp - zm) * td.cell);
            vtx[size_t(z) * vw + x] = { p, glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz)), glm::vec2(p.x, p.z) };
        }
    vmaFlushAllocation(allocator_, stage_alloc, 0, VK_WHOLE_SIZE);

    VkBuffer scratch = VK_NULL_HANDLE; VmaAllocation scratch_alloc = nullptr;
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        auto copy_tex = [&](VkImage img, VkDeviceSize off) {
            if (!img) return;
            vkinit::transition_image(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            r.imageSubresource.layerCount = 1;
            r.imageOffset = { x0, z0, 0 };
            r.imageExtent = { uint32_t(rw), uint32_t(rh), 1 };
            vkCmdCopyBufferToImage(cb, stage, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            vkinit::transition_image(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        };
        copy_tex(terrain_height_full_image_, 0);
        copy_tex(terrain_height_image_, tex_bytes);
        if (terrain_mesh_.vertex_buffer) {
            std::vector<VkBufferCopy> rows(static_cast<size_t>(vh));
            for (int z = 0; z < vh; ++z)
                rows[size_t(z)] = { tex_bytes * 2 + VkDeviceSize(z) * vw * sizeof(Vertex),
                                    (VkDeviceSize(vz0 + z) * W + vx0) * sizeof(Vertex),
                                    VkDeviceSize(vw) * sizeof(Vertex) };
            vkCmdCopyBuffer(cb, stage, terrain_mesh_.vertex_buffer, uint32_t(rows.size()), rows.data());
            VkMemoryBarrier2 mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            VkDependencyInfo di{};
            di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cb, &di);
            record_terrain_blas_refit(cb, &scratch, &scratch_alloc);
            mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier2(cb, &di);
        }
    });
    if (scratch) vmaDestroyBuffer(allocator_, scratch, scratch_alloc);
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    // The TLAS caches the terrain BLAS bounds at build time (a crater can
    // dig below them): force a full rebuild next frame.
    prev_dyn_signature_ = 0;
    prev_tlas_n_ = UINT32_MAX;

    // ---- Physics: heightfield region in place ----
    if (physics_) physics_->update_heightfield_region(td.heights.data(), W, x0, z0, x1, z1);

    // ---- Flowers / rocks: gone in the bowl, re-seated on slope and rim ----
    auto reseat = [&](std::vector<ScatterInstance>& cpu, void* mapped,
                      const std::vector<float>& dy) {
        if (!mapped) return;
        auto* gpu = static_cast<ScatterInstance*>(mapped);
        for (size_t i = 0; i < cpu.size(); ++i) {
            ScatterInstance& s = cpu[i];
            if (s.pos_scale.w <= 0.0f) continue;
            const float dx = s.pos_scale.x - pos.x, dz = s.pos_scale.z - pos.z;
            const float r2 = dx * dx + dz * dz;
            if (r2 >= Rmax * Rmax) continue;
            if (r2 < (0.85f * Rc) * (0.85f * Rc)) s.pos_scale.w = 0.0f;   // blasted away
            else s.pos_scale.y += delta(s.pos_scale.x, s.pos_scale.z);
            ScatterInstance g = s;
            if (scatter_.rm_surface && i < dy.size()) g.pos_scale.y += dy[i];
            gpu[i] = g;
        }
    };
    reseat(scatter_.inst_cpu, scatter_.inst_mapped, scatter_.inst_dy);
    reseat(scatter_.plant_cpu, scatter_.plant_mapped, scatter_.plant_dy);

    log::infof("[crater] r %.1f m depth %.1f m at (%.0f, %.0f, %.0f): %dx%d cells, %.2f ms",
               Rc, D, pos.x, pos.y, pos.z, rw, rh,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

} // namespace qlike
