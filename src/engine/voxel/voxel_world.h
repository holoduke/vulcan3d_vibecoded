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
    // Offset into the global, flat GPU directory SSBO where this shape's
    // entries[] start. Assigned in append order: main shape = 0, the next
    // extracted chunk gets the main shape's size, and so on.
    uint32_t              dir_base = 0;
    // True for the main static shape; false for extracted falling chunks
    // (used to gate which collision / collapse passes touch it).
    bool                  is_main  = false;
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

    // Voxel DDA raycast against shape `shape` (world-space ray; `dir` should
    // be normalized). Returns true on the first solid-voxel hit within
    // `max_dist`, filling `out_t` (world distance) and `out_normal` (the
    // axis-aligned face normal of the voxel that was hit, pointing back
    // toward the ray). Used for exact, lag-free bullet-vs-voxel destruction
    // — independent of the Jolt collision body (which is debounced for perf
    // and would otherwise miss fresh holes). The normal lets callers scale
    // the crater by impact angle.
    bool raycast(int shape, glm::vec3 origin, glm::vec3 dir,
                 float max_dist, float& out_t, glm::vec3& out_normal) const;

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

    // Multi-shape collapse (Session E). Like collapse_unsupported but,
    // instead of greedy-boxing the detached voxels into Jolt-box "logs",
    // groups them into 6-connected ISLANDS and EXTRACTS each island into
    // its own brand-new VoxelShape (own brick slots in the global atlas,
    // own directory) — so the caller can render the chunks via voxel.frag
    // with their own model matrix and they look like falling voxel pieces
    // (preserving the wall's voxel structure). For each new chunk shape
    // returns its shape index plus its world-space bounding-box size
    // (used to size the Jolt dynamic body). The main shape's voxels are
    // cleared and its touched brick slots reported in `main_dirty` so the
    // caller can flush them to the GPU. Returns total voxels detached.
    struct ChunkOut {
        int       shape_index;
        glm::vec3 center_world;   // world centre of the chunk's AABB
        glm::vec3 half_extents;   // world half-size of the chunk's AABB
    };
    int collapse_into_chunks(int shape, int anchor_layers,
                             std::vector<ChunkOut>& out_chunks,
                             std::vector<uint32_t>& main_dirty);

    // Remove a chunk shape (created by collapse_into_chunks). Its directory
    // slot is leaked (the dir_base of later chunks would shift if compacted;
    // leaks are fine — the directory buffer is sized for thousands of slots).
    // Its bricks are dropped from the active atlas-occupancy view but the
    // pool grows monotonically (also fine for the session length).
    void drop_shape(int shape);

    // Free-floating chunk split. Used when a CHUNK (an extracted piece, not
    // the anchored main shape) is carved by a bullet — checks whether the
    // carve disconnected the chunk's voxels. If 2+ connected components
    // remain, keep the largest in-place (the parent chunk continues to be
    // simulated by its existing Jolt body) and extract the rest into new
    // VoxelShapes (ChunkOut entries). No anchor concept is used — every
    // solid voxel is a candidate, since a free-floating chunk has no ground
    // to lean against. Returns total voxels detached. main_dirty contains
    // touched brick slots in the original (now smaller) chunk.
    int split_floating_chunk(int shape, int min_piece_voxels,
                             std::vector<ChunkOut>& out_pieces,
                             std::vector<uint32_t>& main_dirty);

private:
    VoxelShape& new_shape_(glm::vec3 origin_world, int bx, int by, int bz);
    void        poke_(VoxelShape& s, int vx, int vy, int vz, uint8_t pal);
    bool        get_(const VoxelShape& s, int vx, int vy, int vz) const;
    void        clear_(VoxelShape& s, int vx, int vy, int vz,
                       std::vector<uint32_t>& dirty);

    std::vector<BrickPayload> bricks_;
    std::vector<VoxelShape>   shapes_;
    // Monotonic head into the flat global directory: every new shape claims
    // [dir_head_, dir_head_ + dir_size) and bumps the head. Lets the engine
    // upload just the new segment to the GPU directory buffer.
    uint32_t                  dir_head_ = 0;
public:
    uint32_t total_directory_size() const { return dir_head_; }
    uint32_t brick_pool_size() const { return (uint32_t)bricks_.size(); }
private:
    // Reused scratch for the per-grid passes (collapse BFS + greedy box
    // decomposition). Sized to the shape's voxel count once, refilled per
    // call — avoids ~10 MB alloc/free churn on every structural update.
    // mutable: build_collision_boxes is const but uses scratch_a_.
    mutable std::vector<uint8_t> scratch_a_;
    mutable std::vector<uint8_t> scratch_b_;
    // Third scratch — used by the load-stress SI pass to hold per-voxel
    // "vertical load above" counts before mixing them into the cantilever
    // distance for fracture decisions.
    mutable std::vector<uint8_t> scratch_c_;
};

} // namespace qlike::voxel
