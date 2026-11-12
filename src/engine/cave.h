#pragma once

// A walkable cave beside the castle road.
//
// The terrain is a heightfield on both renderers (raymarched FBM and the
// CDLOD mesh), in physics and for every placement system, so it cannot
// hold an overhang. The cave is therefore built in two parts that all of
// those systems already understand:
//   * a TRENCH carved into the source heights (like the road): its floor
//     and lower walls are ordinary terrain, so walking, grass keep-out,
//     shadows and both terrain renderers work unchanged;
//   * a rock SHELL mesh over the trench (static world mesh: ray traced,
//     shadow casting): an inner vault that forms the ceiling and upper
//     walls, and an outer skin that follows the hillside the trench cut
//     away, so from outside the hill is whole again apart from the mouth.

#include "engine/road.h"
#include "game/level.h"

#include <glm/glm.hpp>

#include <vector>

namespace qlike {

struct Heightmap;

struct Cave {
    // Centreline samples (~1 m apart), mouth first. y = floor height.
    std::vector<glm::vec3> pts;
    std::vector<float> half_w;      // inner half-width of the passage
    std::vector<float> height;      // inner height above the floor
    std::vector<float> cover;       // 1 over the cave, fading to 0 past its end (outer skin only)
    int   n_inner = 0;              // samples that carry the inner vault (the rest only close the skin)
    // Outer skin heights, (pts.size() x kCross), sampled BEFORE the carve.
    static constexpr int kCross = 17;
    std::vector<float> skin_y;
    std::vector<float> skin_u;      // lateral offset of each column (pts.size() x kCross)
    glm::vec3 mouth{0.0f};          // centre of the entrance, on the floor
    glm::vec3 mouth_dir{1.0f, 0.0f, 0.0f};   // into the cave
    bool empty() const { return pts.size() < 4; }
};

// Lays the cave out on the right-hand side of the road (seen leaving the
// castle), `along` metres down the road. Must run on the heights the road
// was already carved into.
Cave make_cave(const Heightmap& hm, const Road& road, float water_level, float along = 62.0f);

// Cuts the trench (floor + steep walls) into the heightmap.
void carve_cave(const Cave& cave, Heightmap& hm);

// Appends the rock shell (world space) and coarse roof colliders.
void append_cave_mesh(game::Level& lv, const Cave& cave);

// The cave footprint as a Road (centreline + half width), for the systems
// that keep vegetation off the road.
Road cave_footprint(const Cave& cave);

}  // namespace qlike
