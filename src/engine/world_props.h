#pragma once

// Procedural world props baked into static world-space meshes:
//   * rock formations - clustered, strata-squashed, noise-displaced
//     boulders on slopes / ridges / the rock band, plus lone erratics;
//   * trees           - pines (stacked jagged cones) and broadleaf trees
//     (branching trunk + lumpy canopy clusters), forest density from a
//     low-frequency noise field and the height / slope bands;
//   * bushes          - low lumpy shrubs around forest edges.
// Output goes through the castle static-mesh path (vk_engine/castle_mesh):
// BLAS + RT shadows, sun shadow map, depth prepass, deferred lighting.
// Meshes are chunked on a grid so frustum culling works per chunk.

#include "game/level.h"

#include <glm/vec4.hpp>

#include <vector>

namespace qlike {

struct Heightmap;
struct Road;

struct WorldPropParams {
    float water_level   = 20.9f;
    float grass_band_lo = 17.0f;    // trees start above the beach
    float tree_line     = 72.0f;    // no trees above
    float rock_band_lo  = 60.0f;    // formations favour above this
    float half_extent   = 520.0f;   // placement square around origin
    float castle_clear  = 34.0f;    // keep the castle surroundings open
    int   seed          = 0x5EED;
    // GPU-instanced vegetation (whole map).
    float veg_half_extent = 1000.0f;
    int   max_trees       = 80000;
    int   tree_attempts   = 3000000;
    int   max_bushes      = 30000;
    int   max_formations  = 650;
    // Circles (xz, radius) kept free of trees / rocks (AA vehicles etc.).
    std::vector<glm::vec3> keep_out;
    int   max_boulders    = 1800;
    int   bush_attempts   = 500000;
};

struct WorldPropStats { int formations = 0, boulders = 0, pines = 0, broadleaf = 0, bushes = 0; };

// Vegetation archetype: one species variant with a 4-level LOD chain in
// local space (base at origin, y up). Vertex uv = (wind weight, material:
// 0 bark, 1 needles, 2 leaves, 3 bush, 4 rock).
constexpr int kVegLods = 4;
constexpr int kVegRtLevels = 3;   // BLAS detail levels for ray tracing
struct VegArchetype {
    int species = 0;                 // 0 pine, 1 broadleaf, 2 bush, 3 rock
    game::LevelMesh lods[kVegLods];
    float radius = 1.0f;             // bounding-sphere radius (scale 1)
    float height = 1.0f;
};
// GPU instance (std430, 32 B): .xyz pos, .w scale | .x rotY, .y archetype,
// .z wind phase, .w vertical scale (0 = 1).
struct VegInstance {
    glm::vec4 pos_scale;
    glm::vec4 rot_arch_phase;
};

std::vector<VegArchetype> build_veg_archetypes(int seed);
std::vector<VegInstance> place_vegetation(const Heightmap& hm, const Road* road,
                                          const WorldPropParams& p,
                                          const std::vector<VegArchetype>& arch,
                                          std::vector<collision::AABB>& trunk_colliders,
                                          WorldPropStats& stats);

// Appends meshes (world space) and player colliders.
WorldPropStats build_world_props(const Heightmap& hm, const Road* road,
                                 const WorldPropParams& p,
                                 std::vector<game::LevelMesh>& out_meshes,
                                 std::vector<collision::AABB>& out_colliders);

} // namespace qlike
