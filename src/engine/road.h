#pragma once

// Castle road: a smoothed spline from the castle gate out across the land
// bridge into the southern valley. One CPU description drives everything
// so the pieces can't disagree:
//   * carve_heights()   flattens/levels the heightmap under the road
//                       (raster terrain, BLAS, Jolt and the bakes all read
//                       the carved heights),
//   * clear_grass_mask() removes grass blades from the road surface,
//   * distance()        lets scatter skip rocks on the road,
//   * gpu_points()      feeds the terrain shader a polyline so the cobble
//                       material gets crisp, exact edges per pixel.

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <vector>

namespace qlike {

struct Heightmap;

struct Road {
    // Densified centreline, y = levelled road surface height.
    std::vector<glm::vec3> pts;
    float half_width = 2.3f;     // paved half-width (m)
    float shoulder   = 1.8f;     // dirt verge past the paving (m)
    float blend      = 5.0f;     // terrain cut/fill falloff past the verge (m)
    glm::vec2 bb_min{0.0f}, bb_max{0.0f};   // XZ bounds incl. all margins

    bool empty() const { return pts.size() < 2; }
    // XZ distance to the centreline. Also returns the road height at the
    // closest point.
    float distance(float x, float z, float* out_h = nullptr) const;
};

// Builds the castle road over the current heights. `water_level` keeps
// the surface above the lake on the land bridge; `gate_h` is the castle
// floor height the road starts at.
Road make_castle_road(const Heightmap& hm, float water_level, float gate_h);

// Cut/fill the heightmap to the road surface with a smooth falloff.
void carve_heights(const Road& road, Heightmap& hm);

// Zero grass presence (R channel of the RG8 mask) on paving + verge.
void clear_grass_mask(const Road& road, std::vector<uint8_t>& mask, int dim,
                      float world_side);

// The noisy wobble terrain_raymarch.frag puts on the drawn road edges
// (`jit` in its road weights): the paving ends where distance + jitter =
// half_width, the dirt verge where distance + 1.6 * jitter ~ half_width +
// shoulder. Placement (grass, plants, rocks) adds it to the plain centre-
// line distance so nothing grows on the visible paving and the grass meets
// the drawn verge. Defined in terrain.cpp (shares the shader noise twin).
float road_edge_jitter(float x, float z);

// Polyline for the shader: (x, y, z, 0) per point, at most `max_pts`
// (re-sampled evenly along the length if the road has more).
std::vector<glm::vec4> road_gpu_points(const Road& road, int max_pts);

} // namespace qlike
