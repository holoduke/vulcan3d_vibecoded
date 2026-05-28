#pragma once

// Voxel buildings — Session A (visuals only).
//
// Storage model: "brickmap". A VoxelShape owns a dense 3D directory of
// brick slot indices; each brick is a 16³ block of voxels (4096) backed
// by a 512-bit occupancy bitmap + 4096 per-voxel palette indices.
// Empty bricks consume zero pool space (directory entry = kEmptyBrick).
//
// All bricks across all shapes live in one pool (`VoxelWorld::bricks_`)
// so the GPU sees a single flat SSBO; per-shape directories are a
// separate SSBO per shape (only one shape for now).
//
// Coord conventions:
//   - Voxel size = 0.1 m. Brick edge = 16 voxels = 1.6 m.
//   - Shape voxel coords are unsigned ivec3 in [0, dim_bricks*16).
//   - Shape origin_world is the world-space position of voxel corner (0,0,0).
//   - Session A is translation-only — no rotation. Sessions B+ add quat.
//
// Voxel index inside a brick:  i = z*16*16 + y*16 + x          (0..4095)
// Occupancy bit:                word = i >> 5; bit = i & 31

#include <glm/vec3.hpp>
#include <cstdint>
#include <vector>

namespace qlike::voxel {

constexpr int   kBrickDim    = 16;
constexpr int   kBrickVoxels = kBrickDim * kBrickDim * kBrickDim;  // 4096
constexpr float kVoxelSize   = 0.1f;
constexpr float kBrickSize   = kBrickDim * kVoxelSize;             // 1.6 m

// 4096 bits + 4096 uint8 palette indices = (128 + 1024) * 4 B = 4608 B per brick.
// Tight std430 layout — pal[] is packed 4 indices per uint32.
struct alignas(16) BrickPayload {
    uint32_t occ[128];
    uint32_t pal[1024];
};
static_assert(sizeof(BrickPayload) == (128 + 1024) * 4, "BrickPayload size");

constexpr uint32_t kEmptyBrick = 0xFFFFFFFFu;

struct VoxelShape {
    glm::vec3             origin_world{0.0f};
    int                   dim_bricks[3]{0, 0, 0};
    // Row-major: idx = (bz * dim_bricks[1] + by) * dim_bricks[0] + bx.
    // Values: brick pool slot, or kEmptyBrick.
    std::vector<uint32_t> directory;
};

class VoxelWorld {
public:
    // Build a hollow procedural tower (axis-aligned, no rotation) with
    // walls, floor, door cutout on south face (toward -Z), and merlons
    // around the top. base_corner_world is the world-space position of
    // voxel (0,0,0) — tower extends +X/+Y/+Z from there.
    int add_procedural_tower(glm::vec3 base_corner_world);

    const std::vector<BrickPayload>& bricks() const { return bricks_; }
    const std::vector<VoxelShape>&   shapes() const { return shapes_; }

    uint64_t occupied_voxel_count() const;

private:
    VoxelShape& new_shape_(glm::vec3 origin_world, int bx, int by, int bz);
    void        poke_(VoxelShape& s, int vx, int vy, int vz, uint8_t pal);

    std::vector<BrickPayload> bricks_;
    std::vector<VoxelShape>   shapes_;
};

} // namespace qlike::voxel
