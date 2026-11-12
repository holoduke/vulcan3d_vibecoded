#pragma once

// Instanced grass on top of the heightmap terrain.
//
//   Generation: CPU, once at level load. Halton-sampled candidate XZ
//   across the map, each candidate accepts only if the heightmap slope
//   and altitude fall in the grass band (matches cube.frag's layer
//   thresholds — sand/snow/cliff regions stay bare).
//
//   Render: a tiny shared blade mesh (5 verts, 3 tris) drawn instanced
//   from a host-mapped instance buffer. Wind, sway and per-blade height
//   variation live in grass.vert. Distant blades collapse to a clip-
//   space NaN so the rasteriser drops them with zero fragment cost —
//   no compute culling needed for the first iteration.

#include "engine/mesh.h"
#include "engine/terrain.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace qlike {

// Layout MUST match grass.vert's vertex inputs (locations 3/4/5 each
// declared as vec4, so each entry consumes 16 bytes). Earlier 36-byte
// packed layout caused the shader to read garbage tint colours from
// the next blade's data, which made grass render as bright cyan/teal.
struct alignas(16) GrassBlade {
    glm::vec4 pos_pad;        // .xyz = world base position, .w unused
    glm::vec4 rot_height_pad; // .x = rotation_y, .y = height_factor
    glm::vec4 tint_pad;       // .xyz = tint, .w unused
};
static_assert(sizeof(GrassBlade) == 48, "GrassBlade layout must match grass.vert");

struct GrassParams {
    // Cap on total blades stored in the instance buffer. 200k is a
    // healthy density across a ~150 m grass band on the 2 km world
    // and well under the GPU's vertex throughput at instance-rate.
    uint32_t max_blades   = 100000;
    // Acceptance band for grass placement. Heights match cube.frag's
    // grass layer (~12..40m by default).
    float    height_min   = 8.0f;
    float    height_max   = 38.0f;
    // Reject blades on slopes steeper than this (cosine of the angle
    // between the heightmap normal and +Y). 0.85 ≈ 32° max slope.
    float    min_normal_y = 0.85f;
    // World-space placement budget. We sample candidates uniformly in
    // a square centred on the heightmap, sized to cover most of the
    // terrain. Larger = sparser grass, smaller = denser.
    float    half_extent  = 600.0f;
    // Meadow-scale blades, matched to the raymarched field (~0.35-0.5 m
    // at the default height slider). The original 0.55 m + wide
    // h_factor range stuck isolated 0.75 m spears above the field;
    // 0.30 m read as too short in playtesting.
    float    blade_height = 0.38f;   // base unit blade height (m)
    float    blade_width  = 0.040f;  // base half-width (m)
    int      seed         = 0xC0DE;
    // Rectangular keep-out zone, centered on (0, 0). Candidates with
    // |x| < keep_out_xz.x AND |z| < keep_out_xz.y are rejected — so
    // the castle's footprint stays clear of blades that would poke
    // through the brushes. (0,0) disables the zone.
    glm::vec2 keep_out_xz = glm::vec2(0.0f, 0.0f);
    // No blades on the road (paving + verge). Null = no road.
    const struct Road* road = nullptr;
};

struct GrassMesh {
    // Shared 5-vertex blade — a 2-segment ribbon: two bottom corners,
    // two mid corners, one tip. Mid corners let the blade curve under
    // wind without subdividing further.
    Mesh blade_mesh{};
    // Instance buffer: one GrassBlade per blade. Uploaded once at
    // build time; never updated unless the level reloads.
    VkBuffer instance_buffer = VK_NULL_HANDLE;
    VmaAllocation instance_alloc = nullptr;
    uint32_t instance_count = 0;

    // Spatial buckets. Blades are stably sorted by cell (Halton order is
    // kept inside each cell, so a per-cell prefix is still a uniform
    // thinning for the density slider). The renderer draws only cells
    // within grass distance AND in the view frustum — before this every
    // blade on the 400 m field went through grass.vert each frame just
    // to be culled there.
    struct Cell {
        glm::vec3 aabb_min;
        uint32_t  first;
        glm::vec3 aabb_max;
        uint32_t  count;
    };
    std::vector<Cell> cells;
};

GrassMesh build_grass(VkDevice device, VmaAllocator alloc, VkQueue queue,
                      uint32_t queue_family,
                      const Heightmap& hm, const GrassParams& params);

void destroy_grass(VmaAllocator alloc, GrassMesh& g);

} // namespace qlike
