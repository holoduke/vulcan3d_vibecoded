#pragma once

// Phase 1 procedural heightmap terrain. One non-streaming region; the
// streaming chunk manager arrives in Phase 2 (see docs/terrain_plan.md).
//
// Coordinate convention:
//   - World +Y is up.
//   - The heightmap grid spans [-extent/2, +extent/2] in X and Z.
//   - height[ix, iz] is the world Y at the cell center.

#include "engine/mesh.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

namespace qlike {

struct HeightmapParams {
    int   dim         = 256;     // grid cells per side; vertex grid is dim+1.
    float cell_size   = 4.0f;    // world meters per cell. dim*cell_size = total side.
    int   seed        = 1337;
    float frequency   = 0.0024f; // base fBM frequency (1/m).
    int   octaves     = 6;
    float lacunarity  = 2.05f;
    float gain        = 0.5f;
    float height_scale = 80.0f;  // amplitude.
    // Plateau: a rectangular region forced to a constant height. Used so
    // the castle has a flat foundation regardless of the noise. Disable
    // by setting plateau_extent to zero.
    glm::vec2 plateau_center = glm::vec2(0.0f, 0.0f);
    glm::vec2 plateau_extent = glm::vec2(40.0f, 40.0f); // half-extent
    float     plateau_height = 12.0f;
    float     plateau_blend  = 12.0f;  // smoothstep falloff width
};

struct Heightmap {
    int   dim   = 0;
    float cell  = 0.0f;
    float origin_x = 0.0f; // world X of grid index 0
    float origin_z = 0.0f; // world Z of grid index 0
    std::vector<float> heights; // (dim+1) * (dim+1)

    int   width()  const { return dim + 1; }
    int   height() const { return dim + 1; }
    float at(int ix, int iz) const {
        return heights[static_cast<size_t>(iz) * static_cast<size_t>(width()) + static_cast<size_t>(ix)];
    }
    float side() const { return static_cast<float>(dim) * cell; }
    // World-space sample at arbitrary (x, z) via bilinear interpolation.
    // Returns 0 if outside the grid (caller decides what to render there).
    float sample_world(float x, float z) const;
};

Heightmap generate_heightmap(const HeightmapParams& p);

// Builds a triangulated terrain mesh + VRAM upload via create_mesh_from_data.
// Vertex normals are smoothed per-vertex from neighbouring triangle normals.
Mesh build_terrain_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                        uint32_t queue_family, const Heightmap& hm);

} // namespace qlike
