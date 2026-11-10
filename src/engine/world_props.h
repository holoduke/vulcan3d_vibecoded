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
};

struct WorldPropStats { int formations = 0, boulders = 0, pines = 0, broadleaf = 0, bushes = 0; };

// Appends meshes (world space) and player colliders.
WorldPropStats build_world_props(const Heightmap& hm, const Road* road,
                                 const WorldPropParams& p,
                                 std::vector<game::LevelMesh>& out_meshes,
                                 std::vector<collision::AABB>& out_colliders);

} // namespace qlike
