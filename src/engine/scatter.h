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
#include <vector>

namespace qlike {

struct Heightmap;
struct HeightmapParams;

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
    // Small plants (flowers / ferns): one mesh per species, instances in
    // one buffer sorted by species. Vertex uv.x = part (0 leaf / stem,
    // 1 petal, 2 flower centre), uv.y = wind sway weight.
    static constexpr int kPlantSpecies = 4;      // fern, daisy, poppy, bluebell
    Mesh          plant_mesh[kPlantSpecies]{};
    uint32_t      plant_first[kPlantSpecies]{};
    uint32_t      plant_count[kPlantSpecies]{};
    VkBuffer      plant_buffer = VK_NULL_HANDLE;
    VmaAllocation plant_alloc  = nullptr;
    // Persistently-mapped instance data + per-instance height offset to
    // the raymarched surface (finer FBM than the CPU heightmap). Applied
    // / removed in place by scatter_set_raymarch_surface().
    void*              inst_mapped  = nullptr;
    void*              plant_mapped = nullptr;
    std::vector<float> inst_dy, plant_dy;
    bool               rm_surface = false;
};

struct ScatterParams {
    float half_extent   = 220.0f;   // placement square around origin (m)
    // 24k x 240 verts ≈ 5.8M VS invocations — bounded vertex cost.
    int   max_instances = 24000;
    int   seed          = 0xB0BA;
    // Rules (heights in world Y, matching the splat bands).
    float water_level   = 20.9f;    // reject below water + margin
    float rock_band_h   = 60.0f;    // boulders favoured above this
    float snow_band_h   = 110.0f;   // density tapers off into snow
    // Castle keep-out (same as grass).
    glm::vec2 keep_out_xz = glm::vec2(14.0f, 14.0f);
    // Optional extra keep-out (the castle road). Null = none.
    const struct Road* road = nullptr;
    // When set, per-instance offsets to the raymarched surface are
    // precomputed (`rm_octaves` FBM octaves, the near-camera detail).
    const HeightmapParams* rm_params = nullptr;
    int   rm_octaves = 14;
};

// Builds the rock mesh + places instances. Boulders: 0.15-0.55 m on
// steep slopes / rock band / scattered through dirt. Pebbles:
// 0.04-0.12 m clustered on flatter ground. One mesh, one instance
// buffer, one draw.
ScatterSet build_scatter(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family, const Heightmap& hm,
                         const ScatterParams& params);

void destroy_scatter(VmaAllocator alloc, ScatterSet& s);

// Move all instances onto the raymarched surface (on) or back onto the
// CPU heightmap surface (off). Cheap: rewrites the mapped buffers.
void scatter_set_raymarch_surface(ScatterSet& s, bool on);

} // namespace qlike
