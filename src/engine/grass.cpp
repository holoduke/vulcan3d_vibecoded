#include "engine/grass.h"

#include "engine/log.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#ifdef max
#  undef max
#endif
#ifdef min
#  undef min
#endif

namespace qlike {

namespace {

// 2D Halton(2, 3) — low-discrepancy sequence so the rejection-sampled
// blade positions don't form visible grids or clumps.
float halton(int i, int b) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
    return r;
}

// Local-space blade mesh: two-segment ribbon, 5 vertices, 3 triangles.
//   v0 ----- v1     y=0  (base, full width)
//    \      /
//     v2 - v3       y=0.5 (mid, full width — used for the curve)
//      \  /
//       v4         y=1.0  (tip)
//
// Width = ±blade_width at base/mid, 0 at the tip. The vertex shader
// scales y by per-instance height_factor and applies wind sway.
void make_blade_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                     uint32_t queue_family, float w, float h, Mesh& out) {
    const float h2 = h * 0.5f;
    Vertex verts[5] = {
        // pos              normal               uv (x = side, y = height ratio)
        {{-w, 0.0f,  0.0f}, {0, 0, 1}, {0.0f, 0.0f}}, // 0 base-left
        {{ w, 0.0f,  0.0f}, {0, 0, 1}, {1.0f, 0.0f}}, // 1 base-right
        {{-w, h2,    0.0f}, {0, 0, 1}, {0.0f, 0.5f}}, // 2 mid-left
        {{ w, h2,    0.0f}, {0, 0, 1}, {1.0f, 0.5f}}, // 3 mid-right
        {{ 0.0f, h,  0.0f}, {0, 0, 1}, {0.5f, 1.0f}}, // 4 tip
    };
    uint32_t indices[9] = {
        0, 1, 3,    0, 3, 2,   // base segment (quad as 2 tris)
        2, 3, 4               // top triangle to the tip
    };
    out = create_mesh_from_data(device, alloc, queue, queue_family,
                                verts, 5, indices, 9);
}

} // namespace

GrassMesh build_grass(VkDevice device, VmaAllocator alloc, VkQueue queue,
                      uint32_t queue_family,
                      const Heightmap& hm, const GrassParams& params) {
    GrassMesh g{};
    make_blade_mesh(device, alloc, queue, queue_family,
                    params.blade_width, params.blade_height, g.blade_mesh);

    if (hm.heights.empty() || params.max_blades == 0) {
        log::infof("[grass] heightmap empty or max_blades=0; no blades placed");
        return g;
    }

    // Heightmap-gradient normal sampler — same formula as terrain.cpp's
    // gen_chunk_verts so accept/reject is consistent with the rendered
    // surface.
    auto sample_n = [&](int gx, int gz) {
        int xm = std::max(0, gx - 1);
        int xp = std::min(hm.width()  - 1, gx + 1);
        int zm = std::max(0, gz - 1);
        int zp = std::min(hm.height() - 1, gz + 1);
        float inv_2dx = 1.0f / (static_cast<float>(xp - xm) * hm.cell);
        float inv_2dz = 1.0f / (static_cast<float>(zp - zm) * hm.cell);
        float dhdx = (hm.at(xp, gz) - hm.at(xm, gz)) * inv_2dx;
        float dhdz = (hm.at(gx, zp) - hm.at(gx, zm)) * inv_2dz;
        return glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));
    };

    std::vector<GrassBlade> blades;
    blades.reserve(params.max_blades);

    std::mt19937 rng(static_cast<uint32_t>(params.seed));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_real_distribution<float> rot(0.0f, 6.28318530718f);
    // Tints: warm grass greens with occasional yellow + dark-green
    // outliers so big fields don't read as a single flat colour.
    auto random_tint = [&]() {
        float r = u01(rng);
        if (r < 0.10f) {
            return glm::vec3(0.55f, 0.50f, 0.20f);   // dry yellow
        } else if (r < 0.25f) {
            return glm::vec3(0.18f, 0.32f, 0.12f);   // dark green
        } else if (r < 0.45f) {
            return glm::vec3(0.42f, 0.55f, 0.20f);   // bright lime
        }
        return glm::vec3(0.30f, 0.45f, 0.18f);       // mid green (default)
    };

    // Halton-sampled candidate stream; reject candidates that fall
    // outside the heightmap, off the height band, or on too steep a
    // slope. Cap the attempt count so a level with no acceptable
    // band still terminates.
    const float side = 2.0f * params.half_extent;
    const int   max_attempts = static_cast<int>(params.max_blades * 8u);
    int kept = 0;
    int attempts = 0;
    int hi = 1;
    while (kept < static_cast<int>(params.max_blades) &&
           attempts < max_attempts) {
        ++attempts;
        float hx = halton(hi, 2);
        float hz = halton(hi, 3);
        ++hi;

        float wx = -params.half_extent + hx * side;
        float wz = -params.half_extent + hz * side;

        // Castle keep-out: skip blades inside the rectangle.
        if (params.keep_out_xz.x > 0.0f && params.keep_out_xz.y > 0.0f &&
            std::abs(wx) < params.keep_out_xz.x &&
            std::abs(wz) < params.keep_out_xz.y) {
            continue;
        }

        // Bilinear sample of heightmap at (wx, wz). Reject if outside.
        float fx = (wx - hm.origin_x) / hm.cell;
        float fz = (wz - hm.origin_z) / hm.cell;
        if (fx < 0.0f || fz < 0.0f) continue;
        int ix = static_cast<int>(std::floor(fx));
        int iz = static_cast<int>(std::floor(fz));
        if (ix >= hm.dim || iz >= hm.dim) continue;
        float tx = fx - static_cast<float>(ix);
        float tz = fz - static_cast<float>(iz);
        float h00 = hm.at(ix, iz);
        float h10 = hm.at(ix + 1, iz);
        float h01 = hm.at(ix, iz + 1);
        float h11 = hm.at(ix + 1, iz + 1);
        float h0 = h00 * (1.0f - tx) + h10 * tx;
        float h1 = h01 * (1.0f - tx) + h11 * tx;
        float wy = h0 * (1.0f - tz) + h1 * tz;

        // Reject by altitude band.
        if (wy < params.height_min || wy > params.height_max) continue;
        // Reject by slope.
        glm::vec3 n = sample_n(ix, iz);
        if (n.y < params.min_normal_y) continue;

        // Per-blade jitter — rotation, height variation, tint.
        GrassBlade b{};
        b.pos = glm::vec3(wx, wy, wz);
        b.rotation_y = rot(rng);
        b.height_factor = 0.65f + u01(rng) * 0.7f;   // 0.65..1.35
        b.tint = random_tint();
        b.pad = 0.0f;
        blades.push_back(b);
        ++kept;
    }

    log::infof("[grass] placed %d blades (%d attempts, %d max)",
               kept, attempts, params.max_blades);

    if (blades.empty()) return g;

    // Upload instance buffer. Host-visible so we can update in-place
    // for the Phase 2 sculpt-aware re-placement (not yet wired). Stays
    // mapped persistently — cheap for ~5 MB max.
    const VkDeviceSize size = sizeof(GrassBlade) *
                              static_cast<VkDeviceSize>(blades.size());
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = size,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (vmaCreateBuffer(alloc, &bci, &aci,
                        &g.instance_buffer, &g.instance_alloc, nullptr) != VK_SUCCESS) {
        log::error("[grass] instance buffer alloc failed");
        return g;
    }
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(alloc, g.instance_alloc, &ai);
    if (ai.pMappedData == nullptr) {
        log::error("[grass] instance buffer not mapped");
        return g;
    }
    std::memcpy(ai.pMappedData, blades.data(), static_cast<size_t>(size));
    g.instance_count = static_cast<uint32_t>(blades.size());
    return g;
}

void destroy_grass(VmaAllocator alloc, GrassMesh& g) {
    if (g.blade_mesh.vertex_buffer) destroy_mesh(alloc, g.blade_mesh);
    if (g.instance_buffer) {
        vmaDestroyBuffer(alloc, g.instance_buffer, g.instance_alloc);
        g.instance_buffer = VK_NULL_HANDLE;
        g.instance_alloc = nullptr;
    }
    g.instance_count = 0;
}

} // namespace qlike
