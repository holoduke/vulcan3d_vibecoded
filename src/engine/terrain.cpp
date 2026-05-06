#include "engine/terrain.h"

#include "engine/log.h"

#define FASTNOISE_LITE_IMPLEMENTATION
#include <FastNoiseLite.h>

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef max
#  undef max
#endif
#ifdef min
#  undef min
#endif

namespace qlike {

float Heightmap::sample_world(float x, float z) const {
    float fx = (x - origin_x) / cell;
    float fz = (z - origin_z) / cell;
    if (fx < 0.0f || fz < 0.0f) return 0.0f;
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    if (ix >= dim || iz >= dim) return 0.0f;
    float tx = fx - static_cast<float>(ix);
    float tz = fz - static_cast<float>(iz);
    float h00 = at(ix,     iz);
    float h10 = at(ix + 1, iz);
    float h01 = at(ix,     iz + 1);
    float h11 = at(ix + 1, iz + 1);
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    return h0 * (1.0f - tz) + h1 * tz;
}

Heightmap generate_heightmap(const HeightmapParams& p) {
    FastNoiseLite n;
    n.SetSeed(p.seed);
    n.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    n.SetFractalType(FastNoiseLite::FractalType_FBm);
    n.SetFractalOctaves(p.octaves);
    n.SetFractalLacunarity(p.lacunarity);
    n.SetFractalGain(p.gain);
    n.SetFrequency(p.frequency);

    // Ridged secondary noise gives ridges on top of the rolling base
    // — gives the terrain something rocky to look at without specific
    // texturing in Phase 1.
    FastNoiseLite ridge;
    ridge.SetSeed(p.seed + 1);
    ridge.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    ridge.SetFractalType(FastNoiseLite::FractalType_Ridged);
    ridge.SetFractalOctaves(4);
    ridge.SetFractalLacunarity(2.1f);
    ridge.SetFractalGain(0.5f);
    ridge.SetFrequency(p.frequency * 1.7f);

    Heightmap hm;
    hm.dim  = p.dim;
    hm.cell = p.cell_size;
    hm.origin_x = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    hm.origin_z = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    hm.heights.resize(static_cast<size_t>(hm.width()) * static_cast<size_t>(hm.height()));

    auto smoothstep = [](float a, float b, float t) {
        if (a == b) return t < a ? 0.0f : 1.0f;
        float u = std::clamp((t - a) / (b - a), 0.0f, 1.0f);
        return u * u * (3.0f - 2.0f * u);
    };

    const float plat_outer_x = p.plateau_extent.x + p.plateau_blend;
    const float plat_outer_z = p.plateau_extent.y + p.plateau_blend;

    for (int iz = 0; iz < hm.height(); ++iz) {
        for (int ix = 0; ix < hm.width(); ++ix) {
            float wx = hm.origin_x + static_cast<float>(ix) * hm.cell;
            float wz = hm.origin_z + static_cast<float>(iz) * hm.cell;

            float base = n.GetNoise(wx, wz);              // [-1, 1]
            float r    = ridge.GetNoise(wx, wz);          // [-1, 1]
            // Mix: base rolling hills + ridge contribution at higher
            // elevations only. Empirical weights — pick something
            // believable.
            float h = base * 0.7f + std::pow(std::max(0.0f, r), 1.5f) * 0.5f;
            h *= p.height_scale;

            // Plateau blend: inside the inner extent we hard-set to
            // plateau_height. In the blend ring we ease from terrain
            // into plateau via smoothstep on the *outer* distance.
            float dx = std::abs(wx - p.plateau_center.x);
            float dz = std::abs(wz - p.plateau_center.y);
            float d_in_x  = std::max(0.0f, dx - p.plateau_extent.x);
            float d_in_z  = std::max(0.0f, dz - p.plateau_extent.y);
            float d = std::max(d_in_x, d_in_z);
            // d=0 inside plateau, d>0 outside. Blend over [0, plateau_blend].
            float t = 1.0f - smoothstep(0.0f, p.plateau_blend, d);
            float final_h = h * (1.0f - t) + p.plateau_height * t;
            (void)plat_outer_x; (void)plat_outer_z;

            hm.heights[static_cast<size_t>(iz) * static_cast<size_t>(hm.width()) +
                       static_cast<size_t>(ix)] = final_h;
        }
    }
    log::infof("[terrain] heightmap %dx%d cells, %.0fm side, plateau h=%.1f",
               p.dim, p.dim, hm.side(), p.plateau_height);
    return hm;
}

Mesh build_terrain_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                        uint32_t queue_family, const Heightmap& hm) {
    const int W = hm.width();
    const int H = hm.height();
    std::vector<Vertex> verts;
    verts.reserve(static_cast<size_t>(W) * static_cast<size_t>(H));
    // First pass: positions + zero normals.
    for (int iz = 0; iz < H; ++iz) {
        for (int ix = 0; ix < W; ++ix) {
            glm::vec3 p(hm.origin_x + static_cast<float>(ix) * hm.cell,
                        hm.at(ix, iz),
                        hm.origin_z + static_cast<float>(iz) * hm.cell);
            // UV: world units per metre / 8m repeat — Phase 3 picks up
            // proper triplanar so this is just a hint for raster shaders.
            glm::vec2 uv(p.x / 8.0f, p.z / 8.0f);
            verts.push_back({p, glm::vec3(0.0f), uv});
        }
    }

    // Index buffer. 2 triangles per cell.
    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(hm.dim) * static_cast<size_t>(hm.dim) * 6u);
    for (int iz = 0; iz < hm.dim; ++iz) {
        for (int ix = 0; ix < hm.dim; ++ix) {
            uint32_t i00 = static_cast<uint32_t>(iz * W + ix);
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + static_cast<uint32_t>(W);
            uint32_t i11 = i01 + 1;
            // CCW when viewed from +Y so the terrain's outward normal is
            // up. With X × Z = -Y, the (i00, i10, i11) order produced
            // -Y normals; (i00, i11, i10) produces +Y. Same flip on the
            // diagonal-opposite triangle.
            indices.push_back(i00); indices.push_back(i11); indices.push_back(i10);
            indices.push_back(i00); indices.push_back(i01); indices.push_back(i11);
        }
    }

    // Smooth per-vertex normals: accumulate face normal into each tri's
    // verts, then normalize.
    auto add_face = [&](uint32_t a, uint32_t b, uint32_t c) {
        glm::vec3 e1 = verts[b].position - verts[a].position;
        glm::vec3 e2 = verts[c].position - verts[a].position;
        glm::vec3 n = glm::cross(e1, e2);
        verts[a].normal += n;
        verts[b].normal += n;
        verts[c].normal += n;
    };
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        add_face(indices[i], indices[i + 1], indices[i + 2]);
    }
    for (auto& v : verts) {
        float L = glm::length(v.normal);
        v.normal = (L > 1e-6f) ? (v.normal / L) : glm::vec3(0.0f, 1.0f, 0.0f);
    }

    Mesh m = create_mesh_from_data(device, alloc, queue, queue_family,
                                   verts.data(), static_cast<uint32_t>(verts.size()),
                                   indices.data(), static_cast<uint32_t>(indices.size()));
    log::infof("[terrain] mesh: %zu verts, %zu indices",
               verts.size(), indices.size());
    return m;
}

} // namespace qlike
