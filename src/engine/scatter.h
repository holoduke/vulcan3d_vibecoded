#pragma once

// AAA-terrain Phase 3: ground clutter scatter. Procedurally-generated
// rock meshes instanced over the heightmap by splat-style rules
// (boulders favour steep/rocky ground, pebbles favour dirt/sand/grass).
// AAA terrain reads "AAA" largely because the near field is covered in
// small meshes that break the bare-heightfield look — this is that
// layer, built on the same instancing pattern as the grass field.

#include "engine/mesh.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>

namespace qlike {

struct Heightmap;

// Per-instance data — layout consumed by scatter.vert at binding 1.
struct ScatterInstance {
    glm::vec4 pos_scale;   // .xyz = base world pos, .w = uniform scale (m)
    glm::vec4 rot_tint;    // .x = Y rotation (rad), .yzw = albedo tint
};
static_assert(sizeof(ScatterInstance) == 32, "instance layout is GPU ABI");

struct ScatterSet {
    Mesh          rock_mesh{};                  // shared lumpy-rock mesh
    VkBuffer      instance_buffer = VK_NULL_HANDLE;
    VmaAllocation instance_alloc  = nullptr;
    uint32_t      instance_count  = 0;
};

struct ScatterParams {
    float half_extent   = 220.0f;   // placement square around origin (m)
    int   max_instances = 60000;
    int   seed          = 0xB0BA;
    // Rules (heights in world Y, matching the splat bands).
    float water_level   = 20.9f;    // reject below water + margin
    float rock_band_h   = 60.0f;    // boulders favoured above this
    float snow_band_h   = 110.0f;   // density tapers off into snow
    // Castle keep-out (same as grass).
    glm::vec2 keep_out_xz = glm::vec2(14.0f, 14.0f);
};

// Builds the rock mesh + places instances. Boulders: 0.15-0.55 m on
// steep slopes / rock band / scattered through dirt. Pebbles:
// 0.04-0.12 m clustered on flatter ground. One mesh, one instance
// buffer, one draw.
ScatterSet build_scatter(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family, const Heightmap& hm,
                         const ScatterParams& params);

void destroy_scatter(VmaAllocator alloc, ScatterSet& s);

} // namespace qlike
