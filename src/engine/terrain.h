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

#include <cmath>

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

// FBM heightmap that mirrors the shader's terrain_raymarch.frag exactly:
// 9-octave value-noise FBM with derivative-erosion, ~37° per-octave
// rotation, plateau blend toward p.plateau_height inside the gameplay
// plateau footprint. Used so physics / grass / chunked mesh / BLAS share
// the visible surface when rt_.terrain_raymarch_enabled. Multi-threaded
// — ~38 M FBM evaluations at 2048² takes ~2-3 s on a desktop CPU.
Heightmap generate_heightmap_raymarch(const HeightmapParams& p);

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

// Supersampled tile bake. tx0, tz0 are texel coords on a virtual
// `(hm.dim+1) * ss` × `(hm.dim+1) * ss` grid; the world XZ for each
// texel is `origin + (tx + 0.5) * (cell / ss)` so the bake captures
// sub-cell shadow precision (smoother edges, finer medium/far
// detail). Heights are sampled via Heightmap::sample_world (bilinear)
// — no extra heightmap data needed. ss must be ≥ 1.
void bake_heightmap_shadow_tile_ss(const Heightmap& hm, glm::vec3 sun_dir,
                                    int tx0, int tz0, int w, int h, int ss,
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
// in the same buffer past the interior block (offset = sample_dim^2).
//
// LOD count bumped from 4 to 8 so the user can taper density much further
// out before hitting the degenerate single-quad LOD. Each level's default
// stride doubles: 1, 2, 4, 8, 16, 32, 64, 128. Strides exceeding
// chunk_cells (typically 64) are clamped by gen_indices_for_lod to N-1
// (one giant quad covering the chunk). The user-tunable per-LOD stride
// array in rt_ lets the runtime override these defaults.
constexpr int kTerrainLodCount = 8;
struct TerrainChunk {
    Mesh mesh{};                                // VBO (interior + skirt verts) and LOD-0 IBO
    VkBuffer      ibo_lod[kTerrainLodCount - 1] = { VK_NULL_HANDLE };
    VmaAllocation ibo_lod_alloc[kTerrainLodCount - 1] = { nullptr };
    uint32_t      index_count_lod[kTerrainLodCount] = { 0 };  // [0] mirrors mesh.index_count
    // Per-vertex "parent Y" for CD-LOD morphing -- see gen_chunk_parent_y.
    // One float per vertex (same vertex order as the mesh's VBO), bound at
    // vertex binding 1 by the terrain pipeline. Stays VK_NULL_HANDLE if
    // morphing is disabled.
    //
    // ONLY morphs LOD0 <-> LOD1 today. Bake is hard-coded for the
    // stride-1 -> stride-2 vertex grid. Transitions LOD1->2, LOD2->3,
    // ... pop visibly. Workaround: thresholds for LODs 2+ are pushed
    // far out in the default ladder so pops are fog-masked / subtend
    // few pixels (see Runtime::terrain_lod_distance). Follow-up to ship
    // the "right" fix: bake per-LOD-pair parent streams (parent_y[lod]
    // referencing LOD lod+1) and bind them per draw; adds
    // (kTerrainLodCount - 1) extra vertex buffers per chunk plus the
    // descriptor / pipeline plumbing to switch streams. Not shipped
    // because the threshold workaround hides the artefact at low cost.
    VkBuffer      parent_y_buffer = VK_NULL_HANDLE;
    VmaAllocation parent_y_alloc  = nullptr;
    int cx = 0, cz = 0;                // grid coords (0..N-1)
    int origin_ix = 0, origin_iz = 0;  // first heightmap sample
    int sample_dim = 0;                // samples per chunk side (chunk_cells*densify+1)
    // Per-chunk densification multiplier. 1 = one vertex per heightmap
    // cell (the original baseline); 2/4/8 = sub-divide each heightmap
    // cell into N x N quads via bilinear sampling of the source
    // heightmap. Drives the chunk VBO + parent_y + IBO sample_dim. Each
    // chunk tracks its own densify so only the chunks near the camera
    // pay the VRAM + triangle cost.
    int densify = 1;
    glm::vec3 aabb_min{0.0f};
    glm::vec3 aabb_max{0.0f};
    glm::vec3 center{0.0f};            // world-space chunk centre — used for LOD distance
};

struct TerrainChunkSet {
    std::vector<TerrainChunk> chunks;
    int chunks_per_side = 0;
    int chunk_cells = 0;       // cells per chunk side
};

// Camera-distance LOD picker. Same thresholds drive depth pre-pass +
// color pass + sun-shadow pass so depth values match across passes
// (mismatch breaks the depth pre-pass's LESS_OR_EQUAL test).
//
// `thresholds[i]` is the camera-XZ distance at which the chunk leaves
// LOD i and switches to LOD i+1. `n_thresholds` must equal
// kTerrainLodCount - 1; the returned LOD is in [0, kTerrainLodCount-1].
//
// Distance is the CLOSEST POINT on the chunk AABB (XZ) to the camera.
// The chunk you're standing IN gives distance 0 (clamps inside the
// AABB), so it always picks LOD 0 regardless of where the chunk's
// centre is. Was chunk-CENTRE distance which on 64 m chunks could put
// a chunk you're standing in at distance up to ~45 m (half-diagonal)
// and bump it to a higher LOD than expected -- the user reported this
// as "sometimes the closeup LOD is not the 0-10 one but a less dense
// one." pick_terrain_morph below uses the same AABB-clamp distance so
// the picker and morph fade window stay aligned across chunks.
inline int pick_terrain_lod(const TerrainChunk& c, glm::vec3 cam_xz,
                            const float* thresholds, int n_thresholds) {
    // Parens around std::min/max bypass the windows.h macro clobbering.
    float cx = (std::min)((std::max)(cam_xz.x, c.aabb_min.x), c.aabb_max.x);
    float cz = (std::min)((std::max)(cam_xz.z, c.aabb_min.z), c.aabb_max.z);
    float dx = cx - cam_xz.x;
    float dz = cz - cam_xz.z;
    float d = std::sqrt(dx * dx + dz * dz);
    for (int i = 0; i < n_thresholds; ++i) {
        if (d < thresholds[i]) return i;
    }
    return n_thresholds;
}

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

// Rebuild every chunk's LOD index buffers from the supplied per-LOD
// strides. Used when the user tweaks the per-LOD stride sliders. The
// vertex buffers are untouched; only the kTerrainLodCount index buffers
// per chunk are regenerated. Stride values <= 0 are clamped to 1. Each
// chunk's IBOs are sized by its own c.sample_dim (which may differ when
// per-chunk densification is in use).
void rebuild_chunk_lod_indices(VkDevice device, VmaAllocator alloc,
                                VkQueue queue, uint32_t queue_family,
                                TerrainChunkSet& set,
                                const int strides[kTerrainLodCount]);

// Re-bake a single chunk at a new densify multiplier. Destroys + recreates
// the chunk's VBO (via Mesh), parent_y stream, AND every LOD index buffer
// (their topology depends on sample_dim = chunk_cells * densify + 1). The
// per-LOD `strides` apply unchanged: stride N in densified vertex space
// gives stride N * densify in source heightmap cells, so stride 4 at
// densify 4 reproduces the densify-1 stride-1 mesh exactly. The chunk's
// origin_ix/iz, cx/cz, center, etc are preserved; the AABB is recomputed
// from the new vertex positions. Caller is expected to vkQueueWaitIdle
// before calling so the old buffers are not in flight.
void rebuild_chunk_at_density(VkDevice device, VmaAllocator alloc,
                               VkQueue queue, uint32_t queue_family,
                               const Heightmap& hm, TerrainChunk& c,
                               int densify,
                               const int strides[kTerrainLodCount]);

} // namespace qlike
