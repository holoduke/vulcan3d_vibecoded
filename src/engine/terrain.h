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

// FBM heightmap that mirrors the shader's terrain_raymarch.frag exactly:
// 9-octave value-noise FBM with derivative-erosion, ~37° per-octave
// rotation, plateau blend toward p.plateau_height inside the gameplay
// plateau footprint. Used so physics / grass / chunked mesh / BLAS share
// the visible surface when rt_.terrain_raymarch_enabled. Multi-threaded
// — ~38 M FBM evaluations at 2048² takes ~2-3 s on a desktop CPU.
Heightmap generate_heightmap_raymarch(const HeightmapParams& p);

// Height offset from the CPU heightmap (bilinear 9-octave FBM) to the
// surface the raymarcher draws near the camera (`octaves` FBM octaves).
// Used to sit ground clutter on the raymarched terrain.
float raymarch_fine_correction(const Heightmap& hm, const HeightmapParams& p,
                               float x, float z, int octaves);

// Builds a single triangulated terrain mesh covering the whole heightmap.
// Used by the merged BLAS for ray traversal — RT shadows / GI / AO see
// the full-resolution surface regardless of raster LOD.
Mesh build_terrain_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                        uint32_t queue_family, const Heightmap& hm);

// CPU-baked sun-shadow map. For each heightmap cell, marches a ray
// toward `sun_dir` and tests whether the ray dips below the heightmap
// at any step. Result is a (width × height × 2) byte array, RG8
// interleaved: R = 0 lit / 255 shadowed, G = blocker distance along
// the sun ray (t / 400 m, 0-255; 0 where lit). Shaders run PCSS on
// it — penumbra width scales with G — while .r-only consumers
// (grass.vert, scatter.vert) keep the old binary semantics.
std::vector<uint8_t> bake_heightmap_shadow(const Heightmap& hm,
                                           glm::vec3 sun_dir);

// Bake a single rectangular tile [ix0..ix0+w) × [iz0..iz0+h) of the
// heightmap into `out_tile` (w*h*2 bytes, row-major RG8). Used by the
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

// Raster terrain is the quadtree CDLOD path (engine/terrain_clod.h) —
// the chunk-stride LOD system is gone. TerrainChunk survives ONLY as
// the argument shape for rebuild_chunk_vertices, which the sculpt
// pipeline uses to rewrite the FULL BLAS mesh's vertex buffer in place
// (the BLAS holds the buffer's device address, so the buffer must be
// updated, not recreated).
struct TerrainChunk {
    Mesh mesh{};                       // VBO being rewritten (not owned)
    int cx = 0, cz = 0;
    int origin_ix = 0, origin_iz = 0;  // first heightmap sample
    int sample_dim = 0;                // samples per side
    int densify = 1;
    glm::vec3 aabb_min{0.0f};
    glm::vec3 aabb_max{0.0f};
    glm::vec3 center{0.0f};
};

// Rewrite a mesh's vertex buffer (positions + normals + skirts) from the
// (possibly sculpt-mutated) heightmap. Vertex layout matches
// build_terrain_mesh's output, so passing a TerrainChunk wrapping
// terrain_mesh_ with sample_dim = dim+1 refreshes the whole BLAS source
// mesh after a brush stroke.
void rebuild_chunk_vertices(VkDevice device, VmaAllocator alloc, VkQueue queue,
                            uint32_t queue_family,
                            const Heightmap& hm, TerrainChunk& c);

} // namespace qlike
