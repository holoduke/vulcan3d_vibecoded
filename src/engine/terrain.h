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

// Builds a single triangulated terrain mesh covering the whole heightmap.
// Used by the merged BLAS for ray traversal — RT shadows / GI / AO see
// the full-resolution surface regardless of raster LOD.
Mesh build_terrain_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                        uint32_t queue_family, const Heightmap& hm);

// CPU-baked sun-shadow map. For each heightmap cell, marches a ray
// toward `sun_dir` and tests whether the ray dips below the heightmap
// at any step. Result is a (width × height) byte array, 0 = lit,
// 255 = in shadow. Used by grass.vert as a pre-baked replacement
// for per-fragment RT shadow rays — zero per-frame cost, scales to
// arbitrary grass density.
std::vector<uint8_t> bake_heightmap_shadow(const Heightmap& hm,
                                           glm::vec3 sun_dir);

// Bake a single rectangular tile [ix0..ix0+w) × [iz0..iz0+h) of the
// heightmap into `out_tile` (w*h bytes, row-major). Used by the
// progressive sun-change re-bake to amortize the work across frames
// and prioritize tiles near the camera. Cells outside the heightmap
// bounds are written as 0 (lit) — clamp at sample sites only.
void bake_heightmap_shadow_tile(const Heightmap& hm, glm::vec3 sun_dir,
                                int ix0, int iz0, int w, int h,
                                uint8_t* out_tile);

// Phase 2 chunked terrain: split the heightmap into a grid of chunks,
// each with its own full-LOD vertex+index buffer plus a sparser
// stride-N index buffers that re-index the same vertex buffer (LOD 0 = full,
// LOD 1 = stride 2, LOD 2 = stride 4, LOD 3 = stride 8). Per-frame, the
// renderer picks LOD per chunk based on distance to the camera. Each LOD's
// index buffer also includes a "skirt" — a thin downward strip around the
// chunk perimeter — to hide cracks at LOD-mismatched chunk seams without
// needing edge stitching.
//
// The vertex buffer is also kept on the host side (`positions` stores
// per-vertex world Y) so Phase 4 sculpt can apply heightmap edits then
// rebuild the affected chunk meshes incrementally. Skirt vertices live
// in the same buffer past the interior block (offset = sample_dim²).
constexpr int kTerrainLodCount = 4;
struct TerrainChunk {
    Mesh mesh{};                                // VBO (interior + skirt verts) and LOD-0 IBO
    VkBuffer      ibo_lod[kTerrainLodCount - 1] = { VK_NULL_HANDLE };
    VmaAllocation ibo_lod_alloc[kTerrainLodCount - 1] = { nullptr };
    uint32_t      index_count_lod[kTerrainLodCount] = { 0 };  // [0] mirrors mesh.index_count
    int cx = 0, cz = 0;                // grid coords (0..N-1)
    int origin_ix = 0, origin_iz = 0;  // first heightmap sample
    int sample_dim = 0;                // samples per chunk side (chunk_cells+1)
    glm::vec3 aabb_min{0.0f};
    glm::vec3 aabb_max{0.0f};
    glm::vec3 center{0.0f};            // world-space chunk centre — used for LOD distance
};

struct TerrainChunkSet {
    std::vector<TerrainChunk> chunks;
    int chunks_per_side = 0;
    int chunk_cells = 0;       // cells per chunk side
};

// Builds chunks_per_side² chunks from `hm`. `chunks_per_side` must
// divide `hm.dim`. Each chunk covers chunk_cells = hm.dim/chunks_per_side
// cells per side.
TerrainChunkSet build_terrain_chunks(VkDevice device, VmaAllocator alloc,
                                     VkQueue queue, uint32_t queue_family,
                                     const Heightmap& hm,
                                     int chunks_per_side);

void destroy_terrain_chunks(VkDevice device, VmaAllocator alloc,
                            TerrainChunkSet& set);

// Rebuild a single chunk's vertex buffer from the (possibly mutated)
// heightmap. Index buffers don't change — only positions and normals.
// Used by Phase 4 sculpt after a brush stroke modifies heights.
void rebuild_chunk_vertices(VkDevice device, VmaAllocator alloc, VkQueue queue,
                            uint32_t queue_family,
                            const Heightmap& hm, TerrainChunk& c);

} // namespace qlike
