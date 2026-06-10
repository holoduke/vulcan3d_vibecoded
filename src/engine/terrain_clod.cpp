#include "engine/terrain_clod.h"

#include <algorithm>
#include <cmath>

namespace qlike {

// Patch vertex reuses the engine Vertex struct so create_mesh_from_data
// handles the upload: position = (grid_x, skirt_flag, grid_z). normal/uv
// are unused by the CLOD vertex shader.
Mesh build_clod_patch_mesh(VkDevice device, VmaAllocator alloc,
                           VkQueue queue, uint32_t queue_family) {
    const int K  = kClodGridCells;
    const int W  = K + 1;                       // interior verts per side
    std::vector<Vertex>   verts;
    std::vector<uint32_t> idx;
    verts.reserve(W * W + 4 * W);
    idx.reserve(K * K * 6 + 4 * K * 6);

    // Interior grid.
    for (int z = 0; z <= K; ++z)
        for (int x = 0; x <= K; ++x)
            verts.push_back({ glm::vec3(float(x), 0.0f, float(z)),
                              glm::vec3(0, 1, 0), glm::vec2(0) });
    auto vid = [&](int x, int z) { return uint32_t(z * W + x); };
    for (int z = 0; z < K; ++z) {
        for (int x = 0; x < K; ++x) {
            uint32_t a = vid(x, z),     b = vid(x + 1, z);
            uint32_t c = vid(x, z + 1), d = vid(x + 1, z + 1);
            idx.insert(idx.end(), { a, c, b,  b, c, d });
        }
    }

    // Skirt ring: duplicate each border vertex with skirt_flag = 1 and
    // stitch a quad strip from the border down to the skirt twin. The
    // twins share grid coords with their parents, so the geomorph in
    // terrain.vert moves both identically and the strip stays vertical.
    auto add_skirt_edge = [&](auto border_vid, int count) {
        uint32_t first_twin = uint32_t(verts.size());
        for (int i = 0; i < count; ++i) {
            Vertex v = verts[border_vid(i)];
            v.position.y = 1.0f;               // skirt flag
            verts.push_back(v);
        }
        for (int i = 0; i < count - 1; ++i) {
            uint32_t a = border_vid(i), b = border_vid(i + 1);
            uint32_t c = first_twin + i, d = first_twin + i + 1;
            // Two triangles per segment; winding is irrelevant (terrain
            // pipelines cull NONE on skirts is not guaranteed, so emit
            // both windings? No — terrain pipeline culls back faces;
            // emit both triangles double-sided to be safe).
            idx.insert(idx.end(), { a, c, b,  b, c, d });
            idx.insert(idx.end(), { a, b, c,  b, d, c });
        }
    };
    add_skirt_edge([&](int i) { return vid(i, 0); }, W);       // north (z=0)
    add_skirt_edge([&](int i) { return vid(i, K); }, W);       // south
    add_skirt_edge([&](int i) { return vid(0, i); }, W);       // west
    add_skirt_edge([&](int i) { return vid(K, i); }, W);       // east

    return create_mesh_from_data(device, alloc, queue, queue_family,
                                 verts.data(), uint32_t(verts.size()),
                                 idx.data(),  uint32_t(idx.size()));
}

ClodHeightPyramid build_clod_pyramid(const float* heights, int dim,
                                     float cell, float leaf_size_m) {
    ClodHeightPyramid pyr;
    if (!heights || dim <= 0 || cell <= 0.0f) return pyr;
    const int leaf_cells =
        (std::max)(1, int(std::lround(leaf_size_m / cell)));
    pyr.leaf_blocks = (std::max)(1, dim / leaf_cells);
    // Level count down to a single block.
    int lb = pyr.leaf_blocks;
    pyr.levels = 1;
    while (lb > 1) { lb >>= 1; ++pyr.levels; }
    pyr.minmax.resize(pyr.levels);

    // Level 0: scan each leaf block (+1 sample overlap so block borders
    // are covered by both neighbours).
    pyr.minmax[0].resize(size_t(pyr.leaf_blocks) * pyr.leaf_blocks);
    const int W = dim + 1;
    for (int bz = 0; bz < pyr.leaf_blocks; ++bz) {
        for (int bx = 0; bx < pyr.leaf_blocks; ++bx) {
            int x0 = bx * leaf_cells, z0 = bz * leaf_cells;
            int x1 = (std::min)(x0 + leaf_cells + 1, W);
            int z1 = (std::min)(z0 + leaf_cells + 1, W);
            float mn = 1e30f, mx = -1e30f;
            for (int z = z0; z < z1; ++z) {
                const float* row = heights + size_t(z) * W;
                for (int x = x0; x < x1; ++x) {
                    float h = row[x];
                    mn = (std::min)(mn, h);
                    mx = (std::max)(mx, h);
                }
            }
            pyr.minmax[0][size_t(bz) * pyr.leaf_blocks + bx] = { mn, mx };
        }
    }
    // Upper levels: 2x2 reduce.
    for (int l = 1; l < pyr.levels; ++l) {
        int side  = (std::max)(1, pyr.leaf_blocks >> l);
        int pside = (std::max)(1, pyr.leaf_blocks >> (l - 1));
        pyr.minmax[l].resize(size_t(side) * side);
        for (int z = 0; z < side; ++z) {
            for (int x = 0; x < side; ++x) {
                glm::vec2 r(1e30f, -1e30f);
                for (int dz = 0; dz < 2; ++dz)
                    for (int dx = 0; dx < 2; ++dx) {
                        int px = (std::min)(x * 2 + dx, pside - 1);
                        int pz = (std::min)(z * 2 + dz, pside - 1);
                        glm::vec2 v = pyr.minmax[l - 1][size_t(pz) * pside + px];
                        r.x = (std::min)(r.x, v.x);
                        r.y = (std::max)(r.y, v.y);
                    }
                pyr.minmax[l][size_t(z) * side + x] = r;
            }
        }
    }
    return pyr;
}

void update_clod_pyramid_region(ClodHeightPyramid& pyr,
                                const float* heights, int dim, float cell,
                                float leaf_size_m,
                                int ix0, int ix1, int iz0, int iz1) {
    if (pyr.levels == 0 || !heights || dim <= 0 || cell <= 0.0f) return;
    const int leaf_cells =
        (std::max)(1, int(std::lround(leaf_size_m / cell)));
    const int W = dim + 1;
    int bx0 = std::clamp(ix0 / leaf_cells, 0, pyr.leaf_blocks - 1);
    int bx1 = std::clamp(ix1 / leaf_cells, 0, pyr.leaf_blocks - 1);
    int bz0 = std::clamp(iz0 / leaf_cells, 0, pyr.leaf_blocks - 1);
    int bz1 = std::clamp(iz1 / leaf_cells, 0, pyr.leaf_blocks - 1);
    for (int bz = bz0; bz <= bz1; ++bz) {
        for (int bx = bx0; bx <= bx1; ++bx) {
            int x0 = bx * leaf_cells, z0 = bz * leaf_cells;
            int x1 = (std::min)(x0 + leaf_cells + 1, W);
            int z1 = (std::min)(z0 + leaf_cells + 1, W);
            float mn = 1e30f, mx = -1e30f;
            for (int z = z0; z < z1; ++z) {
                const float* row = heights + size_t(z) * W;
                for (int x = x0; x < x1; ++x) {
                    float h = row[x];
                    mn = (std::min)(mn, h);
                    mx = (std::max)(mx, h);
                }
            }
            pyr.minmax[0][size_t(bz) * pyr.leaf_blocks + bx] = { mn, mx };
        }
    }
    // Propagate the touched region up the reduce chain.
    for (int l = 1; l < pyr.levels; ++l) {
        bx0 >>= 1; bx1 >>= 1; bz0 >>= 1; bz1 >>= 1;
        int side  = (std::max)(1, pyr.leaf_blocks >> l);
        int pside = (std::max)(1, pyr.leaf_blocks >> (l - 1));
        for (int z = bz0; z <= (std::min)(bz1, side - 1); ++z) {
            for (int x = bx0; x <= (std::min)(bx1, side - 1); ++x) {
                glm::vec2 r(1e30f, -1e30f);
                for (int dz = 0; dz < 2; ++dz)
                    for (int dx = 0; dx < 2; ++dx) {
                        int px = (std::min)(x * 2 + dx, pside - 1);
                        int pz = (std::min)(z * 2 + dz, pside - 1);
                        glm::vec2 v = pyr.minmax[l - 1][size_t(pz) * pside + px];
                        r.x = (std::min)(r.x, v.x);
                        r.y = (std::max)(r.y, v.y);
                    }
                pyr.minmax[l][size_t(z) * side + x] = r;
            }
        }
    }
}

namespace {

float node_dist_xz(glm::vec3 cam, float ox, float oz, float size) {
    float cx = std::clamp(cam.x, ox, ox + size);
    float cz = std::clamp(cam.z, oz, oz + size);
    float dx = cx - cam.x, dz = cz - cam.z;
    return std::sqrt(dx * dx + dz * dz);
}

struct SelectCtx {
    const ClodParams* p;
    const ClodHeightPyramid* pyr;
    glm::vec3 cam;
    std::vector<ClodNode>* out;
};

// level: 0 = leaf. bx/bz: node coords on that level's grid.
void select_rec(const SelectCtx& c, int level, int bx, int bz) {
    const ClodParams& p = *c.p;
    float size = p.leaf_size_m * float(1 << level);
    float ox = p.world_origin_x + bx * size;
    float oz = p.world_origin_z + bz * size;

    // Height bounds from the pyramid. Pyramid level l and quadtree level
    // l share the same block grid by construction (leaf_blocks ==
    // world_side / leaf_size), so the lookup is a direct index.
    glm::vec2 mm(0.0f, 1.0f);
    if (c.pyr->levels > 0) {
        // Same-level lookup when the pyramid is deep enough; if the
        // quadtree has more levels than the pyramid (mismatched params),
        // pl saturates at the 1-block root where the index is 0 anyway.
        int pl   = (std::min)(level, c.pyr->levels - 1);
        int side = (std::max)(1, c.pyr->leaf_blocks >> pl);
        int px   = std::clamp(bx, 0, side - 1);
        int pz   = std::clamp(bz, 0, side - 1);
        mm = c.pyr->minmax[pl][size_t(pz) * side + px];
    }

    float range = p.range0 * float(1 << level);
    float d = node_dist_xz(c.cam, ox, oz, size);
    if (level > 0 && d < range * 0.5f) {
        // Within the next-finer ring — recurse. (range of level-1 ==
        // range*0.5; testing the node's own closest point against it
        // keeps neighbouring levels within one step of each other.)
        select_rec(c, level - 1, bx * 2,     bz * 2);
        select_rec(c, level - 1, bx * 2 + 1, bz * 2);
        select_rec(c, level - 1, bx * 2,     bz * 2 + 1);
        select_rec(c, level - 1, bx * 2 + 1, bz * 2 + 1);
        return;
    }

    ClodNode n;
    n.inst.origin_x = ox;
    n.inst.origin_z = oz;
    n.inst.size_m   = size;
    n.inst.lod      = float(level);
    // Geomorph band: last (1 - morph_start_frac) of this level's
    // selection band. A node drawn at level L exists at distances
    // [range*0.5 .. range] (closer would have recursed), so the band
    // sits inside that interval and reaches 1.0 exactly at range —
    // where the neighbouring node is one level coarser.
    float band_lo = range * 0.5f;
    float band_hi = range;
    n.inst.morph_start = band_lo + (band_hi - band_lo) * p.morph_start_frac;
    n.inst.morph_end   = band_hi;
    // Skirt depth scales with node size (hides sub-pixel seams; the
    // geomorph itself is crack-free at matched edges).
    n.inst.skirt_depth = (std::max)(1.5f, size * 0.04f);
    n.inst._pad = 0.0f;
    n.aabb_min = glm::vec3(ox, mm.x - n.inst.skirt_depth, oz);
    n.aabb_max = glm::vec3(ox + size, mm.y + 2.0f, oz + size);
    c.out->push_back(n);
}

} // namespace

void select_clod_nodes(const ClodParams& p, const ClodHeightPyramid& pyr,
                       glm::vec3 cam, std::vector<ClodNode>& out) {
    out.clear();
    SelectCtx c{ &p, &pyr, cam, &out };
    // Root level covers the whole world with one node.
    select_rec(c, p.levels - 1, 0, 0);
}

} // namespace qlike
