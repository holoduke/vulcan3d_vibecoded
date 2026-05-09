#include "engine/grass.h"

#include "engine/log.h"

#include <FastNoiseLite.h>

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

// Higher-resolution blade for Bezier curve evaluation in the vertex
// shader. 4 mid levels + base + tip = 9 vertices, 7 triangles (3 mid
// quads + 1 tip cap). The vertex shader treats inUv.y as the t
// parameter on a quadratic Bezier (base → control → tip) and bends
// the blade per-instance via tilt + tip_bend params. Width tapers
// from full at the base to 0 at the tip via a smooth (1 - t)
// envelope so the silhouette reads as a natural blade.
//
// inUv layout is preserved from the previous mesh:
//   .x = side (0 left / 1 right; 0.5 for the tip)
//   .y = height ratio in [0, 1]
void make_blade_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                     uint32_t queue_family, float w, float h, Mesh& out) {
    // 4 evenly-spaced mid levels at t = 0.25, 0.50, 0.75 (then tip at 1.0).
    // Plus base level at t = 0. Keeps the level count modest for vertex
    // throughput while still giving a smooth Bezier curve.
    const float t_levels[4] = { 0.0f, 0.25f, 0.50f, 0.75f };
    Vertex verts[9];
    int v = 0;
    for (int i = 0; i < 4; ++i) {
        float t = t_levels[i];
        float yw = h * t;
        float ww = w * (1.0f - t * 0.35f);   // gentle linear taper
        verts[v++] = { { -ww, yw, 0.0f }, { 0, 0, 1 }, { 0.0f, t } }; // left
        verts[v++] = { {  ww, yw, 0.0f }, { 0, 0, 1 }, { 1.0f, t } }; // right
    }
    // Tip
    verts[v++] = { { 0.0f, h, 0.0f }, { 0, 0, 1 }, { 0.5f, 1.0f } };

    uint32_t indices[24];
    int idx = 0;
    // 3 mid-quads between consecutive (left, right) pairs.
    for (int i = 0; i < 3; ++i) {
        int bL = i * 2 + 0;
        int bR = i * 2 + 1;
        int tL = (i + 1) * 2 + 0;
        int tR = (i + 1) * 2 + 1;
        // CCW from outside (looking down +z).
        indices[idx++] = bL; indices[idx++] = bR; indices[idx++] = tR;
        indices[idx++] = bL; indices[idx++] = tR; indices[idx++] = tL;
    }
    // Tip cap: top mid-quad → tip.
    int tipL = 6, tipR = 7, tip = 8;
    indices[idx++] = tipL; indices[idx++] = tipR; indices[idx++] = tip;

    out = create_mesh_from_data(device, alloc, queue, queue_family,
                                verts, 9, indices, idx);
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

    // Low-frequency simplex noise sampled at the candidate XZ — used
    // to nudge the slope-acceptance threshold up/down by ~20% so the
    // grass/cliff boundary dissolves into patches rather than tracing
    // a single linear contour line.
    FastNoiseLite slope_noise;
    slope_noise.SetSeed(params.seed + 7);
    slope_noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    slope_noise.SetFrequency(0.06f);
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

        // Slope sampling — placement keeps a generous superset of
        // blades (n.y >= 0.55, ~57° max). The strict slope cutoff is
        // applied at shader-time via the grass_slope_max uniform so
        // the user's UI slider reshapes the field without rebuilding
        // the instance buffer. The per-position simplex offset still
        // breaks up the boundary into patches rather than a clean
        // contour line.
        glm::vec3 n = sample_n(ix, iz);
        float noise = slope_noise.GetNoise(wx, wz);   // -1..1
        float n_with_noise = n.y + noise * 0.10f;
        if (n_with_noise < 0.55f) continue;

        // Per-blade jitter — rotation, height variation, tint.
        // Layout matches grass.vert: 3 vec4s (pos+pad, rot/height+pad,
        // tint+pad). Earlier vec3+float packed layout caused the
        // shader to read the wrong fields — tint came out as garbage
        // and blades rendered teal instead of green.
        GrassBlade b{};
        // Per-blade lean angle (tilt of the blade off vertical, before
        // wind / Bezier curve). Stored in pos_pad.w. Sane range is a
        // few tenths of a radian — enough that a field of blades has
        // visible "leaning here, upright there" texture instead of
        // every blade pointing straight up.
        float tilt = (u01(rng) - 0.5f) * 0.6f;       // [-0.30, 0.30] rad
        b.pos_pad        = glm::vec4(wx, wy, wz, tilt);
        float rotation   = rot(rng);
        float h_factor   = 0.65f + u01(rng) * 0.7f;
        // Clump id: a per-blade hash that the shader blends with the
        // base tint so neighbouring blades tend to share colour, giving
        // the "patches of slightly different green" look characteristic
        // of clump-style grass renderers (Unity-Grass / Ghost of Tsushima).
        float clump_id = std::floor(u01(rng) * 32.0f);
        b.rot_height_pad = glm::vec4(rotation, h_factor, n_with_noise, clump_id);
        // Tip bend amount — how strongly the blade curves forward at
        // the top. > 1 = pronounced cane-shape; < 1 = nearly straight.
        // Stored in tint_pad.w.
        float tip_bend = 0.55f + u01(rng) * 0.95f;   // [0.55, 1.50]
        b.tint_pad       = glm::vec4(random_tint(), tip_bend);
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
