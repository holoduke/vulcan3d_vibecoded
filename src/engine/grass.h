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

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace qlike {

struct GrassBlade {
    glm::vec3 pos;        // world-space base of the blade
    float rotation_y;     // radians, around the up axis
    float height_factor;  // 0.6..1.4, multiplies the unit blade height
    glm::vec3 tint;       // small per-blade albedo variation (greens/yellows)
    float pad;            // padding to vec4 alignment for the SSBO/instance buffer
};

struct GrassParams {
    // Cap on total blades stored in the instance buffer. 200k is a
    // healthy density across a ~150 m grass band on the 2 km world
    // and well under the GPU's vertex throughput at instance-rate.
    uint32_t max_blades   = 200000;
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
    float    blade_height = 0.55f;   // base unit blade height (m)
    float    blade_width  = 0.045f;  // base half-width (m)
    int      seed         = 0xC0DE;
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
};

GrassMesh build_grass(VkDevice device, VmaAllocator alloc, VkQueue queue,
                      uint32_t queue_family,
                      const Heightmap& hm, const GrassParams& params);

void destroy_grass(VmaAllocator alloc, GrassMesh& g);

} // namespace qlike
