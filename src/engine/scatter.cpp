#include "engine/scatter.h"
#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/log.h"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace qlike {
namespace {

float hash31(glm::vec3 p) {
    float h = p.x * 12.9898f + p.y * 78.233f + p.z * 37.719f;
    float s = std::sin(h) * 43758.5453f;
    return s - std::floor(s);
}

// Lumpy rock: icosahedron subdivided twice, vertices displaced by a
// per-direction hash so it reads as an irregular stone, bottom
// flattened so it sits on the ground without floating edges.
void build_rock_mesh_data(std::vector<Vertex>& verts,
                          std::vector<uint32_t>& idx) {
    // Icosahedron base.
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<glm::vec3> pos = {
        {-1,  t, 0}, {1,  t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1,  t}, {0, 1,  t}, {0, -1, -t}, {0, 1, -t},
        { t, 0, -1}, {t, 0,  1}, {-t, 0, -1}, {-t, 0,  1},
    };
    for (auto& p : pos) p = glm::normalize(p);
    std::vector<uint32_t> faces = {
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    };
    // ONE subdivision round (80 tris / 240 verts). Two rounds looked
    // marginally rounder but cost 4x the vertex invocations — at tens of
    // thousands of instances the vertex stage dominated the whole frame
    // (60k x 960 verts = 58M VS invocations, ~100 ms). Clutter reads at
    // a glance; nobody counts facets on a fist-sized rock.
    for (int round = 0; round < 1; ++round) {
        std::vector<glm::vec3> np;
        std::vector<uint32_t>  nf;
        np.reserve(faces.size() * 2);
        for (size_t f = 0; f < faces.size(); f += 3) {
            glm::vec3 a = pos[faces[f]], b = pos[faces[f + 1]],
                      c = pos[faces[f + 2]];
            glm::vec3 ab = glm::normalize(a + b);
            glm::vec3 bc = glm::normalize(b + c);
            glm::vec3 ca = glm::normalize(c + a);
            uint32_t base = uint32_t(np.size());
            np.insert(np.end(), { a, ab, ca,  ab, b, bc,
                                  ca, bc, c,  ab, bc, ca });
            for (int i = 0; i < 12; ++i) nf.push_back(base + i);
        }
        pos = std::move(np);
        faces = std::move(nf);
    }
    // Displace by direction hash (3 octaves of pseudo-noise) and
    // flatten the bottom third.
    for (auto& p : pos) {
        float n = hash31(p * 2.1f) * 0.5f +
                  hash31(p * 4.7f + glm::vec3(13.0f)) * 0.3f +
                  hash31(p * 9.3f + glm::vec3(41.0f)) * 0.2f;
        float r = 0.78f + n * 0.44f;            // 0.78..1.22
        p *= r;
        if (p.y < -0.35f) p.y = -0.35f - (p.y + 0.35f) * 0.15f;
    }
    // Flat-face normals (one normal per triangle, vertices unshared).
    verts.clear();
    idx.clear();
    verts.reserve(faces.size());
    idx.reserve(faces.size());
    for (size_t f = 0; f < faces.size(); f += 3) {
        glm::vec3 a = pos[faces[f]], b = pos[faces[f + 1]],
                  c = pos[faces[f + 2]];
        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        uint32_t base = uint32_t(verts.size());
        verts.push_back({ a, n, glm::vec2(0.0f) });
        verts.push_back({ b, n, glm::vec2(0.0f) });
        verts.push_back({ c, n, glm::vec2(0.0f) });
        idx.insert(idx.end(), { base, base + 1, base + 2 });
    }
}

} // namespace

ScatterSet build_scatter(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family, const Heightmap& hm,
                         const ScatterParams& params) {
    ScatterSet s{};
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    build_rock_mesh_data(verts, idx);
    s.rock_mesh = create_mesh_from_data(device, alloc, queue, queue_family,
                                        verts.data(), uint32_t(verts.size()),
                                        idx.data(), uint32_t(idx.size()));
    if (hm.heights.empty() || params.max_instances <= 0) return s;

    std::mt19937 rng(uint32_t(params.seed));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int W = hm.width();
    auto h_at = [&](float wx, float wz) { return hm.sample_world(wx, wz); };
    auto n_y_at = [&](float wx, float wz) {
        float e = 1.0f;
        float dx = h_at(wx + e, wz) - h_at(wx - e, wz);
        float dz = h_at(wx, wz + e) - h_at(wx, wz - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    (void)W;

    std::vector<ScatterInstance> inst;
    inst.reserve(size_t(params.max_instances));
    const int attempts = params.max_instances * 4;
    for (int i = 0; i < attempts &&
                    int(inst.size()) < params.max_instances; ++i) {
        float wx = (u01(rng) * 2.0f - 1.0f) * params.half_extent;
        float wz = (u01(rng) * 2.0f - 1.0f) * params.half_extent;
        if (std::abs(wx) < params.keep_out_xz.x &&
            std::abs(wz) < params.keep_out_xz.y) continue;
        if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
            wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
            params.road->distance(wx, wz) <
                params.road->half_width + params.road->shoulder + 0.6f) continue;
        float h = h_at(wx, wz);
        if (h < params.water_level + 0.4f) continue;     // not in/at water
        float ny = n_y_at(wx, wz);
        float slope = 1.0f - ny;

        // Rule weights. Boulders: steep ground or the rock band, plus a
        // sparse scattering everywhere. Pebbles: gentler ground,
        // clustered by a low-frequency hash so they come in patches.
        float rock_w = std::clamp(slope * 2.4f, 0.0f, 1.0f);
        rock_w = (std::max)(rock_w,
                            std::clamp((h - params.rock_band_h) / 25.0f,
                                       0.0f, 1.0f));
        rock_w *= 1.0f - std::clamp((h - params.snow_band_h) / 20.0f,
                                    0.0f, 0.7f);
        float clump = hash31(glm::vec3(std::floor(wx / 9.0f), 7.0f,
                                       std::floor(wz / 9.0f)));
        float pebble_w = (1.0f - slope * 1.5f) * (clump > 0.45f ? 1.0f : 0.15f);

        float r = u01(rng);
        bool boulder = r < rock_w * 0.16f + 0.012f;       // sparse big rocks
        bool pebble  = !boulder && r < 0.55f * std::clamp(pebble_w, 0.0f, 1.0f);
        if (!boulder && !pebble) continue;

        ScatterInstance si{};
        float sc = boulder
            ? 0.15f + u01(rng) * u01(rng) * 0.45f          // 0.15-0.6 m
            : 0.035f + u01(rng) * 0.085f;                  // 0.035-0.12 m
        // Sink proportionally so the flattened base sits in the ground.
        si.pos_scale = glm::vec4(wx, h - sc * 0.18f, wz, sc);
        float tint = 0.8f + u01(rng) * 0.4f;
        // Slight warm/cool per-rock variation around neutral.
        glm::vec3 tc(tint * (0.96f + u01(rng) * 0.08f),
                     tint,
                     tint * (0.94f + u01(rng) * 0.08f));
        si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(),
                                tc.x, tc.y, tc.z);
        inst.push_back(si);
    }

    if (inst.empty()) return s;
    const VkDeviceSize bytes = sizeof(ScatterInstance) *
                               VkDeviceSize(inst.size());
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    if (vmaCreateBuffer(alloc, &bci, &aci, &s.instance_buffer,
                        &s.instance_alloc, &ai) != VK_SUCCESS) {
        log::warn("[scatter] instance buffer alloc failed");
        return s;
    }
    std::memcpy(ai.pMappedData, inst.data(), size_t(bytes));
    s.instance_count = uint32_t(inst.size());
    log::infof("[scatter] %u clutter instances placed (rock mesh %u tris)",
               s.instance_count, s.rock_mesh.index_count / 3);
    return s;
}

void destroy_scatter(VmaAllocator alloc, ScatterSet& s) {
    destroy_mesh(alloc, s.rock_mesh);
    if (s.instance_buffer) {
        vmaDestroyBuffer(alloc, s.instance_buffer, s.instance_alloc);
        s.instance_buffer = VK_NULL_HANDLE;
        s.instance_alloc = nullptr;
    }
    s.instance_count = 0;
}

} // namespace qlike
