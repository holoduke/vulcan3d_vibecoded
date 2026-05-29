#include "engine/voxel/voxel_world.h"

#include "engine/log.h"

#include <algorithm>
#include <bit>        // std::popcount
#include <cmath>
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

bool VoxelWorld::get_(const VoxelShape& s, int vx, int vy, int vz) const {
    if (vx < 0 || vy < 0 || vz < 0) return false;
    const int bx = vx >> 4, by = vy >> 4, bz = vz >> 4;
    if (bx >= s.dim_bricks[0] || by >= s.dim_bricks[1] || bz >= s.dim_bricks[2])
        return false;
    const int dir_i = (bz * s.dim_bricks[1] + by) * s.dim_bricks[0] + bx;
    const uint32_t bi = s.directory[dir_i];
    if (bi == kEmptyBrick) return false;
    const int lx = vx & 15, ly = vy & 15, lz = vz & 15;
    const int i = (lz * 256) + (ly * 16) + lx;
    return (bricks_[bi].occ[i >> 5] & (1u << (i & 31))) != 0u;
}

bool VoxelWorld::solid_at(int shape, int vx, int vy, int vz) const {
    if (shape < 0 || shape >= (int)shapes_.size()) return false;
    return get_(shapes_[shape], vx, vy, vz);
}

void VoxelWorld::clear_(VoxelShape& s, int vx, int vy, int vz,
                        std::vector<uint32_t>& dirty) {
    if (vx < 0 || vy < 0 || vz < 0) return;
    const int bx = vx >> 4, by = vy >> 4, bz = vz >> 4;
    if (bx >= s.dim_bricks[0] || by >= s.dim_bricks[1] || bz >= s.dim_bricks[2])
        return;
    const int dir_i = (bz * s.dim_bricks[1] + by) * s.dim_bricks[0] + bx;
    const uint32_t bi = s.directory[dir_i];
    if (bi == kEmptyBrick) return;
    const int lx = vx & 15, ly = vy & 15, lz = vz & 15;
    const int i = (lz * 256) + (ly * 16) + lx;
    uint32_t& word = bricks_[bi].occ[i >> 5];
    const uint32_t bit = 1u << (i & 31);
    if ((word & bit) == 0u) return;             // already empty
    word &= ~bit;
    if (std::find(dirty.begin(), dirty.end(), bi) == dirty.end())
        dirty.push_back(bi);
}

int VoxelWorld::carve_sphere(int shape, glm::vec3 center, float radius,
                             std::vector<uint32_t>& dirty_bricks) {
    if (shape < 0 || shape >= (int)shapes_.size()) return 0;
    VoxelShape& s = shapes_[shape];
    // World → shape-local voxel coords.
    glm::vec3 local = (center - s.origin_world) / kVoxelSize;
    float rv = radius / kVoxelSize;
    int r = (int)std::ceil(rv);
    int cx = (int)std::floor(local.x);
    int cy = (int)std::floor(local.y);
    int cz = (int)std::floor(local.z);
    float r2 = rv * rv;
    int removed = 0;
    for (int z = cz - r; z <= cz + r; ++z)
    for (int y = cy - r; y <= cy + r; ++y)
    for (int x = cx - r; x <= cx + r; ++x) {
        float dx = (float)x + 0.5f - local.x;
        float dy = (float)y + 0.5f - local.y;
        float dz = (float)z + 0.5f - local.z;
        if (dx * dx + dy * dy + dz * dz > r2) continue;
        if (!get_(s, x, y, z)) continue;
        clear_(s, x, y, z, dirty_bricks);
        ++removed;
    }
    return removed;
}

int VoxelWorld::collapse_unsupported(int shape, int anchor_layers,
                                     std::vector<CollisionBox>& out_debris,
                                     std::vector<uint32_t>& dirty) {
    out_debris.clear();
    if (shape < 0 || shape >= (int)shapes_.size()) return 0;
    VoxelShape& s = shapes_[shape];
    const int VX = s.dim_bricks[0] * kBrickDim;
    const int VY = s.dim_bricks[1] * kBrickDim;
    const int VZ = s.dim_bricks[2] * kBrickDim;
    const size_t N = static_cast<size_t>(VX) * VY * VZ;
    auto vidx = [&](int x, int y, int z) {
        return (static_cast<size_t>(z) * VY + y) * VX + x;
    };

    // BFS support from the bottom anchor layers through solid voxels.
    // Reused scratch (zeroed each call) instead of a fresh 10 MB alloc.
    scratch_a_.assign(N, 0u);
    std::vector<uint8_t>& supported = scratch_a_;
    std::vector<int> stack;
    stack.reserve(1u << 16);
    for (int z = 0; z < VZ; ++z)
    for (int x = 0; x < VX; ++x)
    for (int y = 0; y < anchor_layers && y < VY; ++y) {
        if (get_(s, x, y, z) && !supported[vidx(x, y, z)]) {
            supported[vidx(x, y, z)] = 1u;
            stack.push_back((int)vidx(x, y, z));
        }
    }
    const int nx[6] = { 1, -1, 0, 0, 0, 0 };
    const int ny[6] = { 0, 0, 1, -1, 0, 0 };
    const int nz[6] = { 0, 0, 0, 0, 1, -1 };
    while (!stack.empty()) {
        int li = stack.back(); stack.pop_back();
        int x = li % VX;
        int y = (li / VX) % VY;
        int z = li / (VX * VY);
        for (int k = 0; k < 6; ++k) {
            int ax = x + nx[k], ay = y + ny[k], az = z + nz[k];
            if (ax < 0 || ay < 0 || az < 0 || ax >= VX || ay >= VY || az >= VZ)
                continue;
            size_t ai = vidx(ax, ay, az);
            if (supported[ai]) continue;
            if (!get_(s, ax, ay, az)) continue;
            supported[ai] = 1u;
            stack.push_back((int)ai);
        }
    }

    // Greedy-box every unsupported solid voxel, clearing as we go.
    scratch_b_.assign(N, 0u);
    std::vector<uint8_t>& consumed = scratch_b_;
    auto free_unsup = [&](int x, int y, int z) {
        return get_(s, x, y, z) && !supported[vidx(x, y, z)] &&
               !consumed[vidx(x, y, z)];
    };
    int removed = 0;
    for (int z = 0; z < VZ; ++z)
    for (int y = 0; y < VY; ++y)
    for (int x = 0; x < VX; ++x) {
        if (!free_unsup(x, y, z)) continue;
        int ex = x + 1;
        while (ex < VX && free_unsup(ex, y, z)) ++ex;
        int ey = y + 1;
        for (; ey < VY; ++ey) {
            bool ok = true;
            for (int xx = x; xx < ex && ok; ++xx) ok = free_unsup(xx, ey, z);
            if (!ok) break;
        }
        int ez = z + 1;
        for (; ez < VZ; ++ez) {
            bool ok = true;
            for (int yy = y; yy < ey && ok; ++yy)
                for (int xx = x; xx < ex && ok; ++xx) ok = free_unsup(xx, yy, ez);
            if (!ok) break;
        }
        for (int zz = z; zz < ez; ++zz)
            for (int yy = y; yy < ey; ++yy)
                for (int xx = x; xx < ex; ++xx) {
                    consumed[vidx(xx, yy, zz)] = 1u;
                    clear_(s, xx, yy, zz, dirty);
                    ++removed;
                }
        glm::vec3 lo = s.origin_world + glm::vec3(x, y, z) * kVoxelSize;
        glm::vec3 hi = s.origin_world + glm::vec3(ex, ey, ez) * kVoxelSize;
        out_debris.push_back({ (lo + hi) * 0.5f, (hi - lo) * 0.5f });
    }
    return removed;
}

void VoxelWorld::build_collision_boxes(int shape,
                                       std::vector<CollisionBox>& out) const {
    out.clear();
    if (shape < 0 || shape >= (int)shapes_.size()) return;
    const VoxelShape& s = shapes_[shape];
    const int VX = s.dim_bricks[0] * kBrickDim;
    const int VY = s.dim_bricks[1] * kBrickDim;
    const int VZ = s.dim_bricks[2] * kBrickDim;

    // Greedy 3D box decomposition of solid voxels. visited[] marks consumed
    // voxels. For each seed, grow +X, then +Y over the row, then +Z over the
    // slab — emitting the largest axis-aligned box of still-solid, unvisited
    // voxels. Blocky structures collapse to a few hundred boxes.
    scratch_a_.assign(static_cast<size_t>(VX) * VY * VZ, 0u);
    std::vector<uint8_t>& visited = scratch_a_;
    auto vidx = [&](int x, int y, int z) {
        return (static_cast<size_t>(z) * VY + y) * VX + x;
    };
    auto free_solid = [&](int x, int y, int z) {
        return get_(s, x, y, z) && !visited[vidx(x, y, z)];
    };

    for (int z = 0; z < VZ; ++z)
    for (int y = 0; y < VY; ++y)
    for (int x = 0; x < VX; ++x) {
        if (!free_solid(x, y, z)) continue;
        // Grow X.
        int ex = x + 1;
        while (ex < VX && free_solid(ex, y, z)) ++ex;
        // Grow Y over the [x,ex) row.
        int ey = y + 1;
        for (; ey < VY; ++ey) {
            bool ok = true;
            for (int xx = x; xx < ex && ok; ++xx) ok = free_solid(xx, ey, z);
            if (!ok) break;
        }
        // Grow Z over the [x,ex)×[y,ey) slab.
        int ez = z + 1;
        for (; ez < VZ; ++ez) {
            bool ok = true;
            for (int yy = y; yy < ey && ok; ++yy)
                for (int xx = x; xx < ex && ok; ++xx) ok = free_solid(xx, yy, ez);
            if (!ok) break;
        }
        for (int zz = z; zz < ez; ++zz)
            for (int yy = y; yy < ey; ++yy)
                for (int xx = x; xx < ex; ++xx) visited[vidx(xx, yy, zz)] = 1u;

        glm::vec3 lo = s.origin_world +
            glm::vec3(x, y, z) * kVoxelSize;
        glm::vec3 hi = s.origin_world +
            glm::vec3(ex, ey, ez) * kVoxelSize;
        out.push_back({ (lo + hi) * 0.5f, (hi - lo) * 0.5f });
    }
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
