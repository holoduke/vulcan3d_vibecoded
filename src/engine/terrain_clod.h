#pragma once

// Quadtree CDLOD terrain (Decima-class "vertex-free" continuous LOD).
//
// Replaces the chunked stride-LOD + tessellation + densify stack with:
//   - ONE shared grid patch mesh (kClodGridCells² quads + skirt ring),
//     instanced over quadtree nodes. No per-chunk vertex/index buffers.
//   - CPU quadtree selection per frame by camera distance: a node whose
//     closest point lies inside its level's range recurses into its 4
//     children; otherwise it is emitted whole. Few hundred instances.
//   - terrain.vert fetches absolute height from the heightmap texture
//     (binding 17) and geomorphs odd grid vertices toward their even
//     neighbours inside each level's morph band — the fully-morphed
//     fine edge is bit-identical to the adjacent coarser node's edge,
//     so LOD seams are crack-free without stitching. A short skirt is
//     kept as insurance against sub-pixel fp seams.
//
// Physics, the RT BLAS, grass and the brush sculpt keep consuming the
// SAME Heightmap; the raster surface now follows the heightmap texture
// directly, so every height consumer agrees by construction (the old
// densify path resampled the mesh at a different density from the
// texture, which broke the grass raymarch on densified chunks).

#include "engine/mesh.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <vector>

namespace qlike {

struct Heightmap;

// Grid resolution per patch side, in cells. 64 → 65² interior vertices,
// 8192 interior triangles. At the 16 m leaf node that is 0.25 m
// triangles near the camera (4× the old LOD0 density before densify).
constexpr int kClodGridCells = 64;

// Per-instance data consumed by terrain.vert / terrain_shadow.vert at
// vertex binding 1 (input rate INSTANCE). 32 bytes.
struct ClodInstance {
    float origin_x;     // node min-corner world X
    float origin_z;     // node min-corner world Z
    float size_m;       // node side length (world metres)
    float lod;          // quadtree level (0 = leaf) — debug tint only
    float morph_start;  // camera distance where geomorph begins
    float morph_end;    // camera distance where geomorph completes (=range)
    float skirt_depth;  // world metres the skirt ring drops
    float _pad;
};
static_assert(sizeof(ClodInstance) == 32, "instance layout is GPU ABI");

// Selection output: instance + AABB for per-pass frustum culling.
struct ClodNode {
    ClodInstance inst;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
};

// Min/max height pyramid for node AABBs. Level 0 covers kClodLeafCells
// heightmap cells per texel; each higher level halves the resolution.
struct ClodHeightPyramid {
    int leaf_blocks = 0;             // blocks per side at level 0
    int levels = 0;
    // levels[l] is leaf_blocks>>l per side, row-major (min, max) pairs.
    std::vector<std::vector<glm::vec2>> minmax;
};

struct ClodParams {
    float world_origin_x = 0.0f;  // heightmap min-corner world X
    float world_origin_z = 0.0f;
    float world_side     = 0.0f;  // heightmap side length (m)
    float leaf_size_m    = 16.0f; // leaf node side
    int   levels         = 8;     // leaf..root (root side = leaf<<(levels-1))
    // Distance at which a LEAF node stops subdividing (range of level 0).
    // range[l] = range0 * 2^l. Larger = denser terrain further out.
    float range0         = 50.0f;
    // Fraction of the range where geomorph begins (e.g. 0.7 → morph
    // runs over the last 30 % of each level's range).
    float morph_start_frac = 0.70f;
};

// Build the shared instanced patch: interior grid + skirt ring.
// Vertex layout: 3 floats { grid_x, grid_z, skirt_flag }, grid coords in
// cells [0..kClodGridCells]. Skirt vertices duplicate the border cell
// coords with skirt_flag = 1; terrain.vert drops them by
// inst.skirt_depth after the height fetch.
Mesh build_clod_patch_mesh(VkDevice device, VmaAllocator alloc,
                           VkQueue queue, uint32_t queue_family);

// Build (or rebuild after sculpt) the min/max pyramid from raw height
// data ((dim+1)² row-major samples, `cell` metres apart). Raw fields
// rather than Heightmap so the engine's TerrainData (same layout, its
// own type) can call without copying 16 MB of heights.
ClodHeightPyramid build_clod_pyramid(const float* heights, int dim,
                                     float cell, float leaf_size_m);

// Incremental pyramid refresh after a brush stroke touched heightmap
// samples [ix0..ix1] × [iz0..iz1]. Rescans only the leaf blocks that
// overlap the region and propagates the 2×2 reduce upward.
void update_clod_pyramid_region(ClodHeightPyramid& pyr,
                                const float* heights, int dim, float cell,
                                float leaf_size_m,
                                int ix0, int ix1, int iz0, int iz1);

// Quadtree selection. Emits visible-LOD nodes for the whole world; the
// caller frustum-culls per pass (camera passes + light shadow pass must
// share this list so every pass rasterises identical geometry).
void select_clod_nodes(const ClodParams& p, const ClodHeightPyramid& pyr,
                       glm::vec3 cam, std::vector<ClodNode>& out);

} // namespace qlike
