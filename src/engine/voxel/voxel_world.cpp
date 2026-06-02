#include "engine/voxel/voxel_world.h"

#include "engine/log.h"

#include <glm/vec3.hpp>
#include <glm/common.hpp>   // glm::clamp

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
    // Claim a contiguous slice of the flat global directory: the engine
    // uploads each shape's segment to GPU at [dir_base, dir_base + size).
    s.dir_base = dir_head_;
    dir_head_ += static_cast<uint32_t>(s.directory.size());
    // First shape is the main static one; chunks created later default to
    // is_main = false (set explicitly in collapse_into_chunks for clarity).
    s.is_main = shapes_.size() == 1;
    return s;
}

void VoxelWorld::poke_(VoxelShape& s, int vx, int vy, int vz, uint8_t pal) {
    if (vx < 0 || vy < 0 || vz < 0) return;
    const int bx = vx >> 4;
    const int by = vy >> 4;
    const int bz = vz >> 4;
    // Bounds-check matches get_() / clear_(); silent return prevents an OOB
    // directory write if a caller passes a coord past the shape extent.
    if (bx >= s.dim_bricks[0] || by >= s.dim_bricks[1] || bz >= s.dim_bricks[2])
        return;
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
    const int i = (lz * kBrickDim * kBrickDim) + (ly * kBrickDim) + lx;
    return (bricks_[bi].occ[i >> 5] & (1u << (i & 31))) != 0u;
}

bool VoxelWorld::solid_at(int shape, int vx, int vy, int vz) const {
    if (shape < 0 || shape >= (int)shapes_.size()) return false;
    return get_(shapes_[shape], vx, vy, vz);
}

bool VoxelWorld::raycast(int shape, glm::vec3 ro, glm::vec3 rd,
                         float max_dist, float& out_t,
                         glm::vec3& out_normal) const {
    if (shape < 0 || shape >= (int)shapes_.size()) return false;
    const VoxelShape& s = shapes_[shape];
    const int VX = s.dim_bricks[0] * kBrickDim;
    const int VY = s.dim_bricks[1] * kBrickDim;
    const int VZ = s.dim_bricks[2] * kBrickDim;
    const float vs = kVoxelSize;

    // Ray into shape-local space (translation only). AABB clip first.
    glm::vec3 lo = ro - s.origin_world;
    glm::vec3 smax(VX * vs, VY * vs, VZ * vs);
    float t0 = 0.0f, t1 = max_dist;
    for (int k = 0; k < 3; ++k) {
        if (std::abs(rd[k]) < 1e-8f) {
            if (lo[k] < 0.0f || lo[k] > smax[k]) return false;  // parallel, outside
        } else {
            float inv = 1.0f / rd[k];
            float a = (0.0f    - lo[k]) * inv;
            float b = (smax[k] - lo[k]) * inv;
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
            if (t0 > t1) return false;
        }
    }
    float t = std::max(t0, 0.0f) + 1e-4f;
    if (t > max_dist) return false;

    // Amanatides-Woo voxel DDA (per-voxel get_(); bullet segments are short
    // so the empty-brick fast-skip isn't worth the extra code here).
    glm::vec3 p = lo + t * rd;
    glm::ivec3 vc(std::floor(p.x / vs), std::floor(p.y / vs), std::floor(p.z / vs));
    vc = glm::clamp(vc, glm::ivec3(0), glm::ivec3(VX - 1, VY - 1, VZ - 1));
    glm::ivec3 stp(rd.x >= 0 ? 1 : -1, rd.y >= 0 ? 1 : -1, rd.z >= 0 ? 1 : -1);
    auto tnext = [&](int k) {
        if (std::abs(rd[k]) < 1e-8f) return 1e30f;
        float face = (float(vc[k]) + (stp[k] > 0 ? 1.0f : 0.0f)) * vs;
        return t + (face - p[k]) / rd[k];
    };
    glm::vec3 tMax(tnext(0), tnext(1), tnext(2));
    glm::vec3 tDelta(
        std::abs(rd.x) < 1e-8f ? 1e30f : vs / std::abs(rd.x),
        std::abs(rd.y) < 1e-8f ? 1e30f : vs / std::abs(rd.y),
        std::abs(rd.z) < 1e-8f ? 1e30f : vs / std::abs(rd.z));

    // Initial face axis = the AABB-entry axis (largest entry-t), so an
    // immediate hit still reports a sensible normal.
    int last_axis = 0;
    {
        float te[3];
        for (int k = 0; k < 3; ++k)
            te[k] = (std::abs(rd[k]) < 1e-8f) ? -1e30f
                  : ((stp[k] > 0 ? 0.0f : smax[k]) - lo[k]) / rd[k];
        last_axis = (te[0] > te[1] && te[0] > te[2]) ? 0 : (te[1] > te[2] ? 1 : 2);
    }

    const int max_steps = VX + VY + VZ + 8;
    for (int i = 0; i < max_steps; ++i) {
        if (vc.x < 0 || vc.y < 0 || vc.z < 0 ||
            vc.x >= VX || vc.y >= VY || vc.z >= VZ) return false;
        if (get_(s, vc.x, vc.y, vc.z)) {
            out_t = t;
            out_normal = glm::vec3(0.0f);
            out_normal[last_axis] = -float(stp[last_axis]);
            return true;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            t = tMax.x; vc.x += stp.x; tMax.x += tDelta.x; last_axis = 0;
        } else if (tMax.y < tMax.z) {
            t = tMax.y; vc.y += stp.y; tMax.y += tDelta.y; last_axis = 1;
        } else {
            t = tMax.z; vc.z += stp.z; tMax.z += tDelta.z; last_axis = 2;
        }
        if (t > max_dist) return false;
    }
    return false;
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
    const int i = (lz * kBrickDim * kBrickDim) + (ly * kBrickDim) + lx;
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

int VoxelWorld::collapse_into_chunks(int shape, int anchor_layers,
                                     std::vector<ChunkOut>& out_chunks,
                                     std::vector<uint32_t>& main_dirty) {
    out_chunks.clear();
    if (shape < 0 || shape >= (int)shapes_.size()) return 0;
    VoxelShape& s = shapes_[shape];
    const int VX = s.dim_bricks[0] * kBrickDim;
    const int VY = s.dim_bricks[1] * kBrickDim;
    const int VZ = s.dim_bricks[2] * kBrickDim;
    const size_t N = static_cast<size_t>(VX) * VY * VZ;
    auto vidx = [&](int x, int y, int z) {
        return (static_cast<size_t>(z) * VY + y) * VX + x;
    };

    // 1. Load × cantilever "structural integrity" pass.
    //    Two combined signals decide whether a voxel survives the cut:
    //      (a) Cantilever distance — BFS hop count from the nearest voxel
    //          whose vertical column to y=0 is fully solid. Far-from-
    //          support voxels accumulate distance.
    //      (b) Vertical load — count of consecutive solid voxels stacked
    //          directly above this voxel in its own column. The heavier
    //          the load above, the lower the cantilever tolerance.
    //    Effective stress = dist + kLoadWeight * load. Voxels with stress
    //    above kStressMax are unstable; downstream CC + chunk extraction
    //    treats them like disconnected islands.
    //
    //    Tuning rationale:
    //      kStressMax  = 14 voxels (~1.4 m) of "effective" unsupported reach.
    //      kLoadWeight = 0.10: 30 voxels (~3 m) of stack above contributes
    //                    +3 to effective stress, so a column with a moderate
    //                    overhang AND a tall stack on top collapses earlier
    //                    than the same overhang with nothing on it.
    //    This matches the intuition: empty pillars survive long overhangs;
    //    weight-bearing walls fracture earlier under the same cut.
    // Tuned looser to keep single-bullet holes from cascading vertically.
    //   kMaxCantilever 20 voxels (~2 m) — voxels can reach support via a
    //                   longer lateral path before being flagged.
    //   kLoadWeight 0.05 — load above counts, but less; a 30-voxel stack
    //                   adds only 1.5 to effective stress.
    //   kStressMax 20 — matches kMaxCantilever for a pure cantilever
    //                   (no load), and stays loose under realistic load.
    const int   kMaxCantilever = 20;
    const int   kStressMax     = 20;
    const float kLoadWeight    = 0.05f;
    scratch_a_.assign(N, 255u);   // 255 = unreached
    std::vector<uint8_t>& dist = scratch_a_;
    std::vector<int> stk; stk.reserve(1 << 16);
    for (int z = 0; z < VZ; ++z)
    for (int x = 0; x < VX; ++x) {
        // Require the column's base layers to all be solid — matches
        // the old anchor-layers contract (otherwise a column resting on
        // air would count itself as supported).
        bool anchored = true;
        for (int y = 0; y < anchor_layers && y < VY; ++y) {
            if (!get_(s, x, y, z)) { anchored = false; break; }
        }
        if (!anchored) continue;
        // Mark the contiguous solid run from y=0 upward as direct
        // sources (dist 0). The whole intact column is supported,
        // not just the anchor layers — so a 25 m intact tower's top
        // crenellations start at dist 0, not 250.
        for (int y = 0; y < VY; ++y) {
            if (!get_(s, x, y, z)) break;
            dist[vidx(x, y, z)] = 0u;
            stk.push_back(static_cast<int>(vidx(x, y, z)));
        }
    }
    const int nx[6] = { 1, -1, 0, 0, 0, 0 };
    const int ny[6] = { 0, 0, 1, -1, 0, 0 };
    const int nz[6] = { 0, 0, 0, 0, 1, -1 };
    // FIFO BFS (head index, not pop_back) — uniform-cost BFS needs FIFO
    // ordering so the first time we see a voxel we have its shortest
    // distance. Stack-based traversal would over-assign distances.
    size_t head = 0;
    while (head < stk.size()) {
        int li = stk[head++];
        int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
        uint8_t d = dist[vidx(x, y, z)];
        if (d >= kMaxCantilever) continue;     // beyond cap → don't expand
        for (int k = 0; k < 6; ++k) {
            int ax = x + nx[k], ay = y + ny[k], az = z + nz[k];
            if (ax < 0 || ay < 0 || az < 0 || ax >= VX || ay >= VY || az >= VZ)
                continue;
            size_t ai = vidx(ax, ay, az);
            if (dist[ai] != 255u) continue;
            if (!get_(s, ax, ay, az)) continue;
            dist[ai] = static_cast<uint8_t>(d + 1);
            stk.push_back(static_cast<int>(ai));
        }
    }
    // Compute vertical load above each voxel — count of consecutive solid
    // voxels directly above it in the same column. Walk top-down per (x,z)
    // and accumulate. Capped at 255 (uint8) which is fine — a 25 m tower
    // is 250 voxels tall, and we only care that load is "large" past ~30.
    scratch_c_.assign(N, 0u);
    std::vector<uint8_t>& load_above = scratch_c_;
    for (int z = 0; z < VZ; ++z)
    for (int x = 0; x < VX; ++x) {
        int count = 0;
        for (int y = VY - 1; y >= 0; --y) {
            if (get_(s, x, y, z)) {
                load_above[vidx(x, y, z)] = static_cast<uint8_t>(count);
                count = (count < 255) ? count + 1 : 255;
            } else {
                count = 0;
            }
        }
    }

    // Translate dist → boolean supported using the combined stress signal:
    //   stress = dist + kLoadWeight * load_above
    //   supported = stress ≤ kStressMax (AND reached by BFS, i.e. dist < 255)
    //
    // Column-source voxels (dist == 0) are under DIRECT vertical
    // compression — their column to the ground is fully intact, so they
    // hold up their own load through that column, not through any
    // cantilever. Exempting them from the load penalty prevents the
    // false-positive where a tall wall's own bottom voxels would read
    // as overstressed by the weight stacked above them — that's the
    // wall *working as designed*, not a fracture.
    //
    // A voxel never reached by the BFS is unsupported regardless of load
    // (it's disconnected from the anchor entirely).
    for (size_t i = 0; i < N; ++i) {
        uint8_t d = dist[i];
        if (d == 255u) { dist[i] = 0u; continue; }
        if (d == 0u)    { dist[i] = 1u; continue; }   // direct column support
        float stress = float(d) + kLoadWeight * float(load_above[i]);
        dist[i] = (stress <= float(kStressMax)) ? 1u : 0u;
    }
    std::vector<uint8_t>& supported = dist;

    // 2. Connected components of unsupported solid voxels.
    scratch_b_.assign(N, 0u);
    std::vector<uint8_t>& visited_cc = scratch_b_;
    struct CCInfo {
        int vmin[3]; int vmax[3];
        std::vector<int> voxels;
    };
    std::vector<CCInfo> ccs;
    for (int z = 0; z < VZ; ++z)
    for (int y = 0; y < VY; ++y)
    for (int x = 0; x < VX; ++x) {
        size_t i = vidx(x, y, z);
        if (visited_cc[i] || supported[i] || !get_(s, x, y, z)) continue;
        ccs.emplace_back();
        CCInfo& cc = ccs.back();
        cc.vmin[0] = cc.vmax[0] = x;
        cc.vmin[1] = cc.vmax[1] = y;
        cc.vmin[2] = cc.vmax[2] = z;
        visited_cc[i] = 1u;
        stk.clear(); stk.push_back(static_cast<int>(i));
        cc.voxels.push_back(static_cast<int>(i));
        // FIFO head-index traversal (matches the cantilever BFS above).
        // DFS pop_back can blow the local stack on a pathological 100k-voxel
        // island; head-index walks the queue once with no recursion.
        size_t cc_head = 0;
        while (cc_head < stk.size()) {
            int li = stk[cc_head++];
            int cx = li % VX, cy = (li / VX) % VY, cz = li / (VX * VY);
            for (int k = 0; k < 6; ++k) {
                int ax = cx + nx[k], ay = cy + ny[k], az = cz + nz[k];
                if (ax < 0 || ay < 0 || az < 0 ||
                    ax >= VX || ay >= VY || az >= VZ) continue;
                size_t ai = vidx(ax, ay, az);
                if (visited_cc[ai] || supported[ai] || !get_(s, ax, ay, az))
                    continue;
                visited_cc[ai] = 1u;
                stk.push_back(static_cast<int>(ai));
                cc.voxels.push_back(static_cast<int>(ai));
                if (ax < cc.vmin[0]) cc.vmin[0] = ax;
                if (ay < cc.vmin[1]) cc.vmin[1] = ay;
                if (az < cc.vmin[2]) cc.vmin[2] = az;
                if (ax > cc.vmax[0]) cc.vmax[0] = ax;
                if (ay > cc.vmax[1]) cc.vmax[1] = ay;
                if (az > cc.vmax[2]) cc.vmax[2] = az;
            }
        }
    }

    // 2b. Cap component size — split CCs that would produce a too-large
    //     single chunk into smaller sub-pieces along their longest axis.
    //     Without this, a wide cut at mid-tower drops "everything above"
    //     as one giant rigid box that reads as cartoony. Recursive median
    //     split bounds each piece to a maximum size; the parent CC list is
    //     replaced with the split children.
    const int kMaxCCVoxels = 4000;     // ~1.6 m × 1.6 m × 1.6 m worth of solid
    {
        std::vector<CCInfo> split;
        split.reserve(ccs.size());
        std::vector<CCInfo> pending = std::move(ccs);
        while (!pending.empty()) {
            CCInfo cc = std::move(pending.back());
            pending.pop_back();
            if ((int)cc.voxels.size() <= kMaxCCVoxels) {
                split.push_back(std::move(cc));
                continue;
            }
            // Longest axis of the bounding box decides the split plane.
            int dims[3] = {
                cc.vmax[0] - cc.vmin[0] + 1,
                cc.vmax[1] - cc.vmin[1] + 1,
                cc.vmax[2] - cc.vmin[2] + 1,
            };
            int axis = 0;
            if (dims[1] > dims[axis]) axis = 1;
            if (dims[2] > dims[axis]) axis = 2;
            // If the longest axis is too narrow to split usefully, give up
            // and accept the oversized chunk (vs. shedding a 1-voxel
            // slab as a separate Jolt body).
            if (dims[axis] < 8) {
                split.push_back(std::move(cc));
                continue;
            }
            // Median plane along that axis (use the actual voxel-count
            // median, not the bbox midpoint, so the two halves have
            // roughly equal voxel counts even for irregular shapes).
            std::vector<int> coords; coords.reserve(cc.voxels.size());
            for (int li : cc.voxels) {
                int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
                coords.push_back(axis == 0 ? x : axis == 1 ? y : z);
            }
            std::nth_element(coords.begin(),
                             coords.begin() + coords.size() / 2,
                             coords.end());
            int median = coords[coords.size() / 2];
            CCInfo a, b;
            a.vmin[0] = b.vmin[0] = INT32_MAX;
            a.vmin[1] = b.vmin[1] = INT32_MAX;
            a.vmin[2] = b.vmin[2] = INT32_MAX;
            a.vmax[0] = b.vmax[0] = -INT32_MAX;
            a.vmax[1] = b.vmax[1] = -INT32_MAX;
            a.vmax[2] = b.vmax[2] = -INT32_MAX;
            for (int li : cc.voxels) {
                int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
                int c = axis == 0 ? x : axis == 1 ? y : z;
                CCInfo& dst = (c <= median) ? a : b;
                dst.voxels.push_back(li);
                if (x < dst.vmin[0]) dst.vmin[0] = x;
                if (y < dst.vmin[1]) dst.vmin[1] = y;
                if (z < dst.vmin[2]) dst.vmin[2] = z;
                if (x > dst.vmax[0]) dst.vmax[0] = x;
                if (y > dst.vmax[1]) dst.vmax[1] = y;
                if (z > dst.vmax[2]) dst.vmax[2] = z;
            }
            if (!a.voxels.empty()) pending.push_back(std::move(a));
            if (!b.voxels.empty()) pending.push_back(std::move(b));
        }
        ccs = std::move(split);
    }

    // 3. Extract each component into its own new VoxelShape.
    //    Tiny CCs (< ~8 voxels) become dust — we still clear them from
    //    the main shape but don't spawn a Jolt body for them. Prevents
    //    the "5-voxel sliver detaches as a standalone debris piece"
    //    artifact at the carve edge.
    const int kMinCCExtract = 8;
    int removed = 0;
    // CAUTION: every new_shape_() call below may reallocate shapes_, which
    // invalidates any VoxelShape& held across the call. We re-deref
    // shapes_[shape] each iteration to avoid using a dangling reference
    // (the original `s` capture at function entry would dangle after the
    // first non-tiny CC extracts and bumps shapes_).
    for (CCInfo& cc : ccs) {
        if (cc.voxels.empty()) continue;
        if ((int)cc.voxels.size() < kMinCCExtract) {
            VoxelShape& main_ref = shapes_[shape];
            for (int li : cc.voxels) {
                int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
                clear_(main_ref, x, y, z, main_dirty);
                ++removed;
            }
            continue;
        }
        const int cVX = cc.vmax[0] - cc.vmin[0] + 1;
        const int cVY = cc.vmax[1] - cc.vmin[1] + 1;
        const int cVZ = cc.vmax[2] - cc.vmin[2] + 1;
        const int cBX = (cVX + 15) / 16;
        const int cBY = (cVY + 15) / 16;
        const int cBZ = (cVZ + 15) / 16;
        const glm::vec3 chunk_origin = s.origin_world +
            glm::vec3(static_cast<float>(cc.vmin[0]),
                      static_cast<float>(cc.vmin[1]),
                      static_cast<float>(cc.vmin[2])) * kVoxelSize;

        // new_shape_ assigns dir_base and bumps dir_head_.
        // Take a reference AFTER the call (push_back may reallocate shapes_).
        const size_t main_idx = static_cast<size_t>(shape);
        VoxelShape& chunk = new_shape_(chunk_origin, cBX, cBY, cBZ);
        chunk.is_main = false;
        VoxelShape& main = shapes_[main_idx];   // re-deref after possible realloc

        for (int li : cc.voxels) {
            int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
            // Copy palette from the main brick at this voxel.
            uint8_t pal_idx = 1;     // sandstone fallback
            {
                int bx_m = x >> 4, by_m = y >> 4, bz_m = z >> 4;
                int dir_i_m = (bz_m * main.dim_bricks[1] + by_m) *
                              main.dim_bricks[0] + bx_m;
                uint32_t bi_m = main.directory[dir_i_m];
                if (bi_m != kEmptyBrick) {
                    int lx = x & 15, ly = y & 15, lz = z & 15;
                    int idx = lz * kBrickDim * kBrickDim + ly * kBrickDim + lx;
                    int wi = idx >> 2;
                    int sh = (idx & 3) * 8;
                    pal_idx = static_cast<uint8_t>(
                        (bricks_[bi_m].pal[wi] >> sh) & 0xFFu);
                }
            }
            int cx = x - cc.vmin[0];
            int cy = y - cc.vmin[1];
            int cz = z - cc.vmin[2];
            poke_(chunk, cx, cy, cz, pal_idx);
            clear_(main, x, y, z, main_dirty);
            ++removed;
        }

        ChunkOut co{};
        co.shape_index = static_cast<int>(shapes_.size() - 1);
        glm::vec3 mins = chunk_origin;
        glm::vec3 maxs = chunk_origin + glm::vec3(
            static_cast<float>(cVX),
            static_cast<float>(cVY),
            static_cast<float>(cVZ)) * kVoxelSize;
        co.center_world = (mins + maxs) * 0.5f;
        co.half_extents = (maxs - mins) * 0.5f;
        out_chunks.push_back(co);
    }
    return removed;
}

int VoxelWorld::split_floating_chunk(int shape, int min_piece_voxels,
                                     std::vector<ChunkOut>& out_pieces,
                                     std::vector<uint32_t>& main_dirty) {
    out_pieces.clear();
    if (shape < 0 || shape >= (int)shapes_.size()) return 0;
    VoxelShape& s = shapes_[shape];
    const int VX = s.dim_bricks[0] * kBrickDim;
    const int VY = s.dim_bricks[1] * kBrickDim;
    const int VZ = s.dim_bricks[2] * kBrickDim;
    if (VX <= 0 || VY <= 0 || VZ <= 0) return 0;
    const size_t N = static_cast<size_t>(VX) * VY * VZ;
    auto vidx = [&](int x, int y, int z) {
        return (static_cast<size_t>(z) * VY + y) * VX + x;
    };

    // Enumerate every 6-connected solid CC. No anchor — a chunk floating in
    // the air has no "ground", so we just identify all components and keep
    // the LARGEST in the original shape; smaller ones become new chunks.
    scratch_a_.assign(N, 0u);
    std::vector<uint8_t>& visited = scratch_a_;
    struct CCInfo {
        int vmin[3]; int vmax[3];
        std::vector<int> voxels;
    };
    std::vector<CCInfo> ccs;
    std::vector<int> stk; stk.reserve(1 << 14);
    const int nx[6] = { 1, -1, 0, 0, 0, 0 };
    const int ny[6] = { 0, 0, 1, -1, 0, 0 };
    const int nz[6] = { 0, 0, 0, 0, 1, -1 };
    for (int z = 0; z < VZ; ++z)
    for (int y = 0; y < VY; ++y)
    for (int x = 0; x < VX; ++x) {
        size_t i = vidx(x, y, z);
        if (visited[i] || !get_(s, x, y, z)) continue;
        ccs.emplace_back();
        CCInfo& cc = ccs.back();
        cc.vmin[0] = cc.vmax[0] = x;
        cc.vmin[1] = cc.vmax[1] = y;
        cc.vmin[2] = cc.vmax[2] = z;
        visited[i] = 1u;
        stk.clear(); stk.push_back(static_cast<int>(i));
        cc.voxels.push_back(static_cast<int>(i));
        size_t head = 0;
        while (head < stk.size()) {
            int li = stk[head++];
            int cx = li % VX, cy = (li / VX) % VY, cz = li / (VX * VY);
            for (int k = 0; k < 6; ++k) {
                int ax = cx + nx[k], ay = cy + ny[k], az = cz + nz[k];
                if (ax < 0 || ay < 0 || az < 0 ||
                    ax >= VX || ay >= VY || az >= VZ) continue;
                size_t ai = vidx(ax, ay, az);
                if (visited[ai] || !get_(s, ax, ay, az)) continue;
                visited[ai] = 1u;
                stk.push_back(static_cast<int>(ai));
                cc.voxels.push_back(static_cast<int>(ai));
                if (ax < cc.vmin[0]) cc.vmin[0] = ax;
                if (ay < cc.vmin[1]) cc.vmin[1] = ay;
                if (az < cc.vmin[2]) cc.vmin[2] = az;
                if (ax > cc.vmax[0]) cc.vmax[0] = ax;
                if (ay > cc.vmax[1]) cc.vmax[1] = ay;
                if (az > cc.vmax[2]) cc.vmax[2] = az;
            }
        }
    }

    if (ccs.size() <= 1) return 0;     // no split — chunk still intact

    // Find the largest CC; it stays in the original shape. Smaller ones
    // become new shapes. (Largest is kept so the parent's existing Jolt
    // body keeps approximating the bulk mass; smaller fragments get fresh
    // bodies via the engine spawn path.)
    size_t main_cc = 0;
    for (size_t i = 1; i < ccs.size(); ++i) {
        if (ccs[i].voxels.size() > ccs[main_cc].voxels.size()) main_cc = i;
    }

    int removed = 0;
    for (size_t ci = 0; ci < ccs.size(); ++ci) {
        if (ci == main_cc) continue;
        CCInfo& cc = ccs[ci];
        if ((int)cc.voxels.size() < min_piece_voxels) {
            // Too small to be worth a Jolt body — just clear from main and
            // skip extraction (becomes dust). Visible-as-hole.
            for (int li : cc.voxels) {
                int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
                clear_(s, x, y, z, main_dirty);
                ++removed;
            }
            continue;
        }
        const int cVX = cc.vmax[0] - cc.vmin[0] + 1;
        const int cVY = cc.vmax[1] - cc.vmin[1] + 1;
        const int cVZ = cc.vmax[2] - cc.vmin[2] + 1;
        const int cBX = (cVX + 15) / 16;
        const int cBY = (cVY + 15) / 16;
        const int cBZ = (cVZ + 15) / 16;
        // Origin for the new shape — in the SAME shape-local frame as the
        // parent. Engine layer converts to world via parent's body
        // transform.
        const glm::vec3 chunk_origin = s.origin_world +
            glm::vec3(static_cast<float>(cc.vmin[0]),
                      static_cast<float>(cc.vmin[1]),
                      static_cast<float>(cc.vmin[2])) * kVoxelSize;

        const size_t parent_idx = static_cast<size_t>(shape);
        VoxelShape& piece = new_shape_(chunk_origin, cBX, cBY, cBZ);
        piece.is_main = false;
        VoxelShape& parent = shapes_[parent_idx];

        for (int li : cc.voxels) {
            int x = li % VX, y = (li / VX) % VY, z = li / (VX * VY);
            uint8_t pal_idx = 1;
            {
                int bx_m = x >> 4, by_m = y >> 4, bz_m = z >> 4;
                int dir_i_m = (bz_m * parent.dim_bricks[1] + by_m) *
                              parent.dim_bricks[0] + bx_m;
                uint32_t bi_m = parent.directory[dir_i_m];
                if (bi_m != kEmptyBrick) {
                    int lx = x & 15, ly = y & 15, lz = z & 15;
                    int idx = lz * kBrickDim * kBrickDim + ly * kBrickDim + lx;
                    int wi = idx >> 2;
                    int sh = (idx & 3) * 8;
                    pal_idx = static_cast<uint8_t>(
                        (bricks_[bi_m].pal[wi] >> sh) & 0xFFu);
                }
            }
            int cx = x - cc.vmin[0];
            int cy = y - cc.vmin[1];
            int cz = z - cc.vmin[2];
            poke_(piece, cx, cy, cz, pal_idx);
            clear_(parent, x, y, z, main_dirty);
            ++removed;
        }

        ChunkOut co{};
        co.shape_index = static_cast<int>(shapes_.size() - 1);
        glm::vec3 mins = chunk_origin;
        glm::vec3 maxs = chunk_origin + glm::vec3(
            static_cast<float>(cVX),
            static_cast<float>(cVY),
            static_cast<float>(cVZ)) * kVoxelSize;
        co.center_world = (mins + maxs) * 0.5f;
        co.half_extents = (maxs - mins) * 0.5f;
        out_pieces.push_back(co);
    }
    return removed;
}

void VoxelWorld::drop_shape(int shape) {
    // Soft-drop: clear the shape's directory so the DDA visits empty bricks
    // only. The shape entry, brick pool, and dir_head_ stay (no compaction).
    if (shape < 0 || shape >= (int)shapes_.size()) return;
    VoxelShape& s = shapes_[shape];
    std::fill(s.directory.begin(), s.directory.end(), kEmptyBrick);
    s.dim_bricks[0] = s.dim_bricks[1] = s.dim_bricks[2] = 0;
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
    // Slimmer two-stage tower: a 8×16×8 brick base (12.8 × 25.6 × 12.8 m)
    // with a centred 5×6×5 brick upper turret (8.0 × 9.6 × 8.0 m) sitting
    // on top, capped by a stepped pyramid roof. Shape grid is sized for
    // BASE + TURRET + ROOF height; X/Z fits the base; the turret is
    // centred and indented from the base footprint.
    constexpr int BX = 8;
    constexpr int BZ = 8;
    constexpr int base_BY    = 16;                   // 25.6 m base
    constexpr int turret_BY  = 6;                    // 9.6 m turret
    constexpr int roof_VY    = 28;                   // 2.8 m roof — pyramid steps
    constexpr int turret_VY  = turret_BY * kBrickDim;
    constexpr int base_VY    = base_BY  * kBrickDim;
    constexpr int total_VY   = base_VY + turret_VY + roof_VY;
    constexpr int BY         = (total_VY + kBrickDim - 1) / kBrickDim;

    constexpr int VX = BX * kBrickDim;
    constexpr int VY = BY * kBrickDim;
    constexpr int VZ = BZ * kBrickDim;

    constexpr int wall_thick      = 5;               // 0.5 m walls
    constexpr int floor_thick     = 3;               // 0.3 m base floor
    constexpr int door_w          = 12;              // 1.2 m
    constexpr int door_h          = 24;              // 2.4 m
    constexpr int merlon_period   = 6;               // 0.6 m crenellation
    constexpr int base_batt_y     = base_VY - 8;     // 0.8 m crenellation band on base
    constexpr int turret_batt_y0  = base_VY + turret_VY - 8;

    // Turret footprint — 5×5 bricks (8.0×8.0 m), centred in X and Z.
    constexpr int turret_VXZ      = 5 * kBrickDim;
    constexpr int turret_x0       = (VX - turret_VXZ) / 2;
    constexpr int turret_z0       = (VZ - turret_VXZ) / 2;
    constexpr int turret_x1       = turret_x0 + turret_VXZ;
    constexpr int turret_z1       = turret_z0 + turret_VXZ;

    // Window cutouts: tall slits, 2 voxels wide × 6 voxels tall, with
    // small horizontal cross at the top (a cross-shaped arrow slit).
    // Three per face on the base, two per face on the turret.
    constexpr int win_w           = 2;
    constexpr int win_h           = 6;

    constexpr uint8_t kPalStone   = 1;               // base sandstone
    constexpr uint8_t kPalAccent  = 2;               // floor / lintel / window frame
    constexpr uint8_t kPalRoof    = 3;               // copper/rust roof
    constexpr uint8_t kPalWood    = 2;               // wood-tone for upper turret accents

    VoxelShape& s = new_shape_(base_corner_world, BX, BY, BZ);

    auto is_window_slot = [&](int along, int span, int win_count,
                              int y, int y_lo, int y_hi) -> bool {
        if (y < y_lo || y >= y_hi) return false;
        // win_count slits evenly spaced; centred per slit.
        for (int k = 0; k < win_count; ++k) {
            int cx = (span * (2 * k + 1)) / (2 * win_count);
            int half = win_w / 2;
            if (along >= cx - half && along < cx + half) return true;
        }
        return false;
    };

    // ---------- BASE: outer wall + battlement + window slits ----------
    for (int y = 0; y < base_VY; ++y) {
        const bool in_batt = (y >= base_batt_y);
        for (int z = 0; z < VZ; ++z) {
            for (int x = 0; x < VX; ++x) {
                const bool wx = (x < wall_thick) || (x >= VX - wall_thick);
                const bool wz = (z < wall_thick) || (z >= VZ - wall_thick);
                if (!(wx || wz)) continue;
                // South door cutout (low z, not in corner posts).
                if (z < wall_thick && !wx) {
                    if (y < door_h &&
                        x >= (VX / 2 - door_w / 2) &&
                        x <  (VX / 2 + door_w / 2)) {
                        continue;
                    }
                }
                // Window slits — 3 per face at two heights.
                if (!in_batt) {
                    const int along = wx ? z : x;
                    const int span  = wx ? VZ : VX;
                    if (is_window_slot(along, span, 3, y,
                                       int(base_VY * 0.30f),
                                       int(base_VY * 0.30f) + win_h) ||
                        is_window_slot(along, span, 3, y,
                                       int(base_VY * 0.65f),
                                       int(base_VY * 0.65f) + win_h)) {
                        // Skip corners.
                        const bool corner =
                            (x < wall_thick && z < wall_thick) ||
                            (x < wall_thick && z >= VZ - wall_thick) ||
                            (x >= VX - wall_thick && z < wall_thick) ||
                            (x >= VX - wall_thick && z >= VZ - wall_thick);
                        if (!corner) continue;
                    }
                }
                if (in_batt) {
                    const int along = wx ? z : x;
                    const bool gap = ((along / merlon_period) & 1) == 1;
                    if (gap && y >= base_VY - 4) continue;
                }
                poke_(s, x, y, z, kPalStone);
            }
        }
    }

    // Base floor slab (interior only).
    for (int z = wall_thick; z < VZ - wall_thick; ++z) {
        for (int x = wall_thick; x < VX - wall_thick; ++x) {
            for (int y = 0; y < floor_thick; ++y) {
                poke_(s, x, y, z, kPalAccent);
            }
        }
    }

    // Lintel above the door.
    for (int y = door_h; y < door_h + 3; ++y) {
        for (int x = (VX / 2 - door_w / 2); x < (VX / 2 + door_w / 2); ++x) {
            for (int z = 0; z < wall_thick; ++z) {
                poke_(s, x, y, z, kPalAccent);
            }
        }
    }

    // ---------- TURRET FLOOR: a stone slab covering the base top ----
    // Fills the gap so the base interior reads as ceilinged and the
    // turret sits on a visible deck.
    for (int z = wall_thick; z < VZ - wall_thick; ++z) {
        for (int x = wall_thick; x < VX - wall_thick; ++x) {
            for (int y = base_VY; y < base_VY + 3; ++y) {
                poke_(s, x, y, z, kPalAccent);
            }
        }
    }

    // ---------- UPPER TURRET: thinner ring + battlement + windows ----
    const int t_wall = 4;                            // 0.4 m thinner walls
    const int t_top  = base_VY + turret_VY;
    for (int y = base_VY; y < t_top; ++y) {
        const int ly = y - base_VY;
        const bool in_batt = (y >= turret_batt_y0);
        for (int z = turret_z0; z < turret_z1; ++z) {
            for (int x = turret_x0; x < turret_x1; ++x) {
                const bool wx = (x < turret_x0 + t_wall) ||
                                 (x >= turret_x1 - t_wall);
                const bool wz = (z < turret_z0 + t_wall) ||
                                 (z >= turret_z1 - t_wall);
                if (!(wx || wz)) continue;
                // Window slits on the turret — 2 per face, tall ones.
                if (!in_batt) {
                    const int along = wx ? (z - turret_z0) : (x - turret_x0);
                    const int span  = turret_VXZ;
                    if (is_window_slot(along, span, 2, ly,
                                       int(turret_VY * 0.30f),
                                       int(turret_VY * 0.30f) + win_h)) {
                        const bool corner =
                            (x < turret_x0 + t_wall && z < turret_z0 + t_wall) ||
                            (x < turret_x0 + t_wall && z >= turret_z1 - t_wall) ||
                            (x >= turret_x1 - t_wall && z < turret_z0 + t_wall) ||
                            (x >= turret_x1 - t_wall && z >= turret_z1 - t_wall);
                        if (!corner) continue;
                    }
                }
                if (in_batt) {
                    const int along = wx ? z : x;
                    const bool gap = ((along / merlon_period) & 1) == 1;
                    if (gap && y >= t_top - 4) continue;
                }
                poke_(s, x, y, z, kPalStone);
            }
        }
    }

    // ---------- ROOF: stepped square pyramid over the turret ----
    // Starts ONE voxel inside the turret outer face so the roof eaves
    // hang slightly past the wall, then steps inward by 1 voxel every
    // ~4 voxels of height. Tile palette gives a copper/rust read.
    {
        const int r_y0 = t_top;
        for (int dy = 0; dy < roof_VY; ++dy) {
            // Inset grows with height — pyramid taper.
            const int inset = (dy * (turret_VXZ / 2 - 2)) / roof_VY;
            const int rx0 = turret_x0 + inset;
            const int rx1 = turret_x1 - inset;
            const int rz0 = turret_z0 + inset;
            const int rz1 = turret_z1 - inset;
            if (rx0 >= rx1 || rz0 >= rz1) break;
            for (int z = rz0; z < rz1; ++z) {
                for (int x = rx0; x < rx1; ++x) {
                    // Only outline + top — hollow inside to keep voxel
                    // count down. The top 4 layers fill fully so the
                    // apex is solid.
                    const bool edge = (x == rx0 || x == rx1 - 1 ||
                                       z == rz0 || z == rz1 - 1);
                    const bool top  = (dy >= roof_VY - 4);
                    if (!(edge || top)) continue;
                    poke_(s, x, r_y0 + dy, z, kPalRoof);
                }
            }
        }
    }

    log::infof("[voxel] tower built: %d bricks live, %llu occupied voxels (%dx%dx%d cell grid)",
               (int)bricks_.size(),
               (unsigned long long)occupied_voxel_count(),
               VX, VY, VZ);
    (void)kPalWood;
    return static_cast<int>(shapes_.size()) - 1;
}

} // namespace qlike::voxel
