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

namespace qlike::collision { struct AABB; }

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

// World-space solid box produced by greedy decomposition — fed to both the
// player kinematic collision (game::collision AABBs) and Jolt static boxes
// (projectile collision). center/half are world-space metres.
struct CollisionBox {
    glm::vec3 center;
    glm::vec3 half;
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

    // Read a single voxel (world index). True if solid. Out-of-range = false.
    bool solid_at(int shape, int vx, int vy, int vz) const;

    // Greedy box decomposition of shape `shape`'s solid voxels into a small
    // set of world-space AABBs. Used for player + projectile collision.
    void build_collision_boxes(int shape, std::vector<CollisionBox>& out) const;

    // Clear every voxel within `radius` metres of world-space `center` in
    // shape `shape`. Returns the number of voxels removed. Records touched
    // brick pool slots in `dirty_bricks` (deduped) so the caller can re-
    // upload just those to the GPU. Does NOT free emptied bricks (directory
    // entry stays valid → DDA still visits it, finds no set bits).
    int carve_sphere(int shape, glm::vec3 center, float radius,
                     std::vector<uint32_t>& dirty_bricks);

    // Mutable brick access for GPU re-upload of dirty slots.
    const BrickPayload& brick(uint32_t slot) const { return bricks_[slot]; }

    // Structural collapse (Session D). Flood-fills "support" from anchor
    // voxels (the bottom `anchor_layers` voxel layers, i.e. the base resting
    // on the ground) through 6-connected solid voxels. Any solid voxel NOT
    // reached is unsupported — it gets greedy-boxed into `out_debris` (world-
    // space boxes), CLEARED from the shape (touched bricks recorded in
    // `dirty`), so the caller can spawn falling rigid-body debris and re-
    // upload the holes. Returns the number of voxels that detached.
    int collapse_unsupported(int shape, int anchor_layers,
                             std::vector<CollisionBox>& out_debris,
                             std::vector<uint32_t>& dirty);

private:
    VoxelShape& new_shape_(glm::vec3 origin_world, int bx, int by, int bz);
    void        poke_(VoxelShape& s, int vx, int vy, int vz, uint8_t pal);
    bool        get_(const VoxelShape& s, int vx, int vy, int vz) const;
    void        clear_(VoxelShape& s, int vx, int vy, int vz,
                       std::vector<uint32_t>& dirty);

    std::vector<BrickPayload> bricks_;
    std::vector<VoxelShape>   shapes_;
    // Reused scratch for the per-grid passes (collapse BFS + greedy box
    // decomposition). Sized to the shape's voxel count once, refilled per
    // call — avoids ~10 MB alloc/free churn on every structural update.
    // mutable: build_collision_boxes is const but uses scratch_a_.
    mutable std::vector<uint8_t> scratch_a_;
    mutable std::vector<uint8_t> scratch_b_;
};

} // namespace qlike::voxel
