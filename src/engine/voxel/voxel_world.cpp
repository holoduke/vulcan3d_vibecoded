#include "engine/voxel/voxel_world.h"

#include "engine/log.h"

#include <bit>        // std::popcount
#include <cstring>

namespace qlike::voxel {

VoxelShape& VoxelWorld::new_shape_(glm::vec3 origin_world, int bx, int by, int bz) {
    shapes_.emplace_back();
    VoxelShape& s = shapes_.back();
    s.origin_world = origin_world;
    s.dim_bricks[0] = bx;
    s.dim_bricks[1] = by;
    s.dim_bricks[2] = bz;
    s.directory.assign(static_cast<size_t>(bx) * by * bz, kEmptyBrick);
    return s;
}

void VoxelWorld::poke_(VoxelShape& s, int vx, int vy, int vz, uint8_t pal) {
    const int bx = vx >> 4;
    const int by = vy >> 4;
    const int bz = vz >> 4;
    const int dir_i = (bz * s.dim_bricks[1] + by) * s.dim_bricks[0] + bx;
    uint32_t bi = s.directory[dir_i];
    if (bi == kEmptyBrick) {
        bi = static_cast<uint32_t>(bricks_.size());
        bricks_.emplace_back();
        std::memset(&bricks_.back(), 0, sizeof(BrickPayload));
        s.directory[dir_i] = bi;
    }
    BrickPayload& b = bricks_[bi];
    const int lx = vx & 15;
    const int ly = vy & 15;
    const int lz = vz & 15;
    const int i  = (lz * 16 * 16) + (ly * 16) + lx;
    b.occ[i >> 5] |= (1u << (i & 31));
    const int word  = i >> 2;
    const int shift = (i & 3) * 8;
    b.pal[word] = (b.pal[word] & ~(0xFFu << shift)) | (static_cast<uint32_t>(pal) << shift);
}

uint64_t VoxelWorld::occupied_voxel_count() const {
    uint64_t total = 0;
    for (const auto& b : bricks_) {
        for (int i = 0; i < 128; ++i) {
            total += static_cast<uint64_t>(std::popcount(b.occ[i]));
        }
    }
    return total;
}

int VoxelWorld::add_procedural_tower(glm::vec3 base_corner_world) {
    // 12 × 18 × 12 bricks = 192 × 288 × 192 voxels = 19.2 × 28.8 × 19.2 m.
    constexpr int BX = 12, BY = 18, BZ = 12;
    constexpr int VX = BX * kBrickDim;
    constexpr int VY = BY * kBrickDim;
    constexpr int VZ = BZ * kBrickDim;
    constexpr int wall_thick     = 6;          // 0.6 m
    constexpr int floor_thick    = 4;          // 0.4 m
    constexpr int door_w         = 12;         // 1.2 m
    constexpr int door_h         = 24;         // 2.4 m
    constexpr int batt_y         = VY - 32;    // last 3.2 m gets the battlement
    constexpr int merlon_period  = 8;          // 0.8 m crenellation period
    constexpr uint8_t kPalStone  = 1;
    constexpr uint8_t kPalAccent = 2;          // floor / lintel accent

    VoxelShape& s = new_shape_(base_corner_world, BX, BY, BZ);

    // Outer wall ring + battlement.
    for (int y = 0; y < VY; ++y) {
        const bool in_batt = (y >= batt_y);
        for (int z = 0; z < VZ; ++z) {
            for (int x = 0; x < VX; ++x) {
                const bool wx = (x < wall_thick) || (x >= VX - wall_thick);
                const bool wz = (z < wall_thick) || (z >= VZ - wall_thick);
                if (!(wx || wz)) continue;          // interior — skip
                // South-face door cutout (low z, not corners).
                if (z < wall_thick && !wx) {
                    if (y < door_h &&
                        x >= (VX / 2 - door_w / 2) &&
                        x <  (VX / 2 + door_w / 2)) {
                        continue;
                    }
                }
                // Merlon gaps along the top of the wall ring.
                if (in_batt) {
                    const int along = wx ? z : x;
                    const bool gap = ((along / merlon_period) & 1) == 1;
                    if (gap && y >= VY - 8) continue;  // top half = open
                }
                poke_(s, x, y, z, kPalStone);
            }
        }
    }

    // Floor slab (interior only — outside the wall ring).
    for (int z = wall_thick; z < VZ - wall_thick; ++z) {
        for (int x = wall_thick; x < VX - wall_thick; ++x) {
            for (int y = 0; y < floor_thick; ++y) {
                poke_(s, x, y, z, kPalAccent);
            }
        }
    }

    // Lintel above the door — fills the top of the door arch with the
    // accent palette so the entrance reads as a framed doorway.
    {
        const int lint_y0 = door_h;
        const int lint_y1 = door_h + 4;
        for (int y = lint_y0; y < lint_y1; ++y) {
            for (int x = (VX / 2 - door_w / 2); x < (VX / 2 + door_w / 2); ++x) {
                for (int z = 0; z < wall_thick; ++z) {
                    poke_(s, x, y, z, kPalAccent);
                }
            }
        }
    }

    log::infof("[voxel] tower built: %d bricks live, %llu occupied voxels (%dx%dx%d cell grid)",
               (int)bricks_.size(),
               (unsigned long long)occupied_voxel_count(),
               VX, VY, VZ);

    return static_cast<int>(shapes_.size()) - 1;
}

} // namespace qlike::voxel
