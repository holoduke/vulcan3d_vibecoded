#include "engine/terrain.h"

#include "engine/log.h"

#define FASTNOISE_LITE_IMPLEMENTATION
#include <FastNoiseLite.h>

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#ifdef max
#  undef max
#endif
#ifdef min
#  undef min
#endif

namespace qlike {

float Heightmap::sample_world(float x, float z) const {
    float fx = (x - origin_x) / cell;
    float fz = (z - origin_z) / cell;
    if (fx < 0.0f || fz < 0.0f) return 0.0f;
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    if (ix >= dim || iz >= dim) return 0.0f;
    float tx = fx - static_cast<float>(ix);
    float tz = fz - static_cast<float>(iz);
    float h00 = at(ix,     iz);
    float h10 = at(ix + 1, iz);
    float h01 = at(ix,     iz + 1);
    float h11 = at(ix + 1, iz + 1);
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    return h0 * (1.0f - tz) + h1 * tz;
}

Heightmap generate_heightmap(const HeightmapParams& p) {
    FastNoiseLite n;
    n.SetSeed(p.seed);
    n.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    n.SetFractalType(FastNoiseLite::FractalType_FBm);
    n.SetFractalOctaves(p.octaves);
    n.SetFractalLacunarity(p.lacunarity);
    n.SetFractalGain(p.gain);
    n.SetFrequency(p.frequency);

    // Ridged secondary noise gives ridges on top of the rolling base
    // — gives the terrain something rocky to look at without specific
    // texturing in Phase 1.
    FastNoiseLite ridge;
    ridge.SetSeed(p.seed + 1);
    ridge.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    ridge.SetFractalType(FastNoiseLite::FractalType_Ridged);
    ridge.SetFractalOctaves(4);
    ridge.SetFractalLacunarity(2.1f);
    ridge.SetFractalGain(0.5f);
    ridge.SetFrequency(p.frequency * 1.7f);

    Heightmap hm;
    hm.dim  = p.dim;
    hm.cell = p.cell_size;
    hm.origin_x = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    hm.origin_z = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    hm.heights.resize(static_cast<size_t>(hm.width()) * static_cast<size_t>(hm.height()));

    auto smoothstep = [](float a, float b, float t) {
        if (a == b) return t < a ? 0.0f : 1.0f;
        float u = std::clamp((t - a) / (b - a), 0.0f, 1.0f);
        return u * u * (3.0f - 2.0f * u);
    };

    const float plat_outer_x = p.plateau_extent.x + p.plateau_blend;
    const float plat_outer_z = p.plateau_extent.y + p.plateau_blend;

    for (int iz = 0; iz < hm.height(); ++iz) {
        for (int ix = 0; ix < hm.width(); ++ix) {
            float wx = hm.origin_x + static_cast<float>(ix) * hm.cell;
            float wz = hm.origin_z + static_cast<float>(iz) * hm.cell;

            float base = n.GetNoise(wx, wz);              // [-1, 1]
            float r    = ridge.GetNoise(wx, wz);          // [-1, 1]
            // Mix: base rolling hills + ridge contribution at higher
            // elevations only. Empirical weights — pick something
            // believable.
            float h = base * 0.7f + std::pow(std::max(0.0f, r), 1.5f) * 0.5f;
            h *= p.height_scale;

            // Plateau blend: inside the inner extent we hard-set to
            // plateau_height. In the blend ring we ease from terrain
            // into plateau via smoothstep on the *outer* distance.
            float dx = std::abs(wx - p.plateau_center.x);
            float dz = std::abs(wz - p.plateau_center.y);
            float d_in_x  = std::max(0.0f, dx - p.plateau_extent.x);
            float d_in_z  = std::max(0.0f, dz - p.plateau_extent.y);
            float d = std::max(d_in_x, d_in_z);
            // d=0 inside plateau, d>0 outside. Blend over [0, plateau_blend].
            float t = 1.0f - smoothstep(0.0f, p.plateau_blend, d);
            float final_h = h * (1.0f - t) + p.plateau_height * t;
            (void)plat_outer_x; (void)plat_outer_z;

            hm.heights[static_cast<size_t>(iz) * static_cast<size_t>(hm.width()) +
                       static_cast<size_t>(ix)] = final_h;
        }
    }
    log::infof("[terrain] heightmap %dx%d cells, %.0fm side, plateau h=%.1f",
               p.dim, p.dim, hm.side(), p.plateau_height);
    return hm;
}

namespace {

// Generate one chunk's vertex grid + smoothed per-vertex normals from
// `hm` covering [origin_ix, origin_ix + sample_dim) × [origin_iz, ...).
// Per-vertex normals come from the global heightmap gradient (central
// differences) so adjacent chunks agree on shared-edge normals — face-
// average accumulation only saw triangles WITHIN a chunk and produced
// faint shading seams at chunk boundaries.
// Skirt depth in metres. Each chunk-edge vertex has a "twin" vertex at the
// same XZ but lowered Y by this amount; the skirt strip connects them so
// that LOD-mismatch cracks at chunk boundaries are hidden by a vertical
// wall rather than producing a see-through gap. 8 m is enough to cover
// worst-case adjacent-cell height jumps on the steepest mountains.
constexpr float kTerrainSkirtDepth = 8.0f;

void gen_chunk_verts(const Heightmap& hm, int origin_ix, int origin_iz,
                     int sample_dim,
                     std::vector<Vertex>& verts,
                     glm::vec3& aabb_min, glm::vec3& aabb_max) {
    verts.clear();
    verts.reserve(static_cast<size_t>(sample_dim) * static_cast<size_t>(sample_dim) +
                   static_cast<size_t>(4) * static_cast<size_t>(sample_dim));
    aabb_min = glm::vec3( 1e9f);
    aabb_max = glm::vec3(-1e9f);
    // 4-cell radius for the gradient. Pre-LOD this was 1-cell, which
    // gave normals that exactly matched the local 1m face. After LOD
    // landed, distant chunks render at stride 2/4/8 — triangles span
    // 4–8 cells but the 1-cell-radius normals at the corners reflect
    // sub-cell detail the triangle's flat face doesn't represent.
    // Result: adjacent large LOD-2/3 triangles get noisy per-vertex
    // normals → patchy "different shaded faces" artifacts. A 4-cell
    // gradient samples a span comparable to the average rendered
    // triangle: smooths the LOD-far chunks without erasing the
    // gentle valleys at LOD 0 (which still get fine-detail shading
    // from the BLAS-derived face geometry).
    constexpr int kNormalRadius = 4;
    auto sample_normal = [&](int gx, int gz) {
        int xm = std::max(0, gx - kNormalRadius);
        int xp = std::min(hm.width()  - 1, gx + kNormalRadius);
        int zm = std::max(0, gz - kNormalRadius);
        int zp = std::min(hm.height() - 1, gz + kNormalRadius);
        float inv_2dx = 1.0f / (static_cast<float>(xp - xm) * hm.cell);
        float inv_2dz = 1.0f / (static_cast<float>(zp - zm) * hm.cell);
        float dhdx = (hm.at(xp, gz) - hm.at(xm, gz)) * inv_2dx;
        float dhdz = (hm.at(gx, zp) - hm.at(gx, zm)) * inv_2dz;
        return glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));
    };
    // Interior (sample_dim × sample_dim).
    for (int iz = 0; iz < sample_dim; ++iz) {
        for (int ix = 0; ix < sample_dim; ++ix) {
            int gx = std::clamp(origin_ix + ix, 0, hm.width()  - 1);
            int gz = std::clamp(origin_iz + iz, 0, hm.height() - 1);
            glm::vec3 p(hm.origin_x + static_cast<float>(gx) * hm.cell,
                        hm.at(gx, gz),
                        hm.origin_z + static_cast<float>(gz) * hm.cell);
            glm::vec2 uv(p.x, p.z);
            verts.push_back({p, sample_normal(gx, gz), uv});
            aabb_min = glm::min(aabb_min, p);
            aabb_max = glm::max(aabb_max, p);
        }
    }
    // Skirt twins: 4 edges × sample_dim verts. Each one is the interior
    // edge vertex with Y dropped by kTerrainSkirtDepth. Normal inherits
    // from the corresponding interior vertex (heightmap-gradient normal,
    // mostly upward) so the skirt shades like the terrain above it
    // rather than reading as a dark vertical wall — distant chunks at
    // low LOD would otherwise show their outward-facing skirts as
    // visible dark faces against the sky.
    auto add_skirt_edge = [&](int ix0, int iz0, int dx, int dz) {
        for (int k = 0; k < sample_dim; ++k) {
            int ix = ix0 + dx * k;
            int iz = iz0 + dz * k;
            int gx = std::clamp(origin_ix + ix, 0, hm.width()  - 1);
            int gz = std::clamp(origin_iz + iz, 0, hm.height() - 1);
            glm::vec3 p(hm.origin_x + static_cast<float>(gx) * hm.cell,
                        hm.at(gx, gz) - kTerrainSkirtDepth,
                        hm.origin_z + static_cast<float>(gz) * hm.cell);
            verts.push_back({p, sample_normal(gx, gz), glm::vec2(p.x, p.z)});
        }
    };
    add_skirt_edge(0,             0,             1, 0);  // top  (z=0)
    add_skirt_edge(sample_dim - 1, 0,             0, 1);  // right(x=N-1)
    add_skirt_edge(0,             sample_dim - 1, 1, 0);  // bot  (z=N-1)
    add_skirt_edge(0,             0,             0, 1);  // left (x=0)
}

// Per-vertex "parent Y" for CD-LOD morphing between LOD 0 and LOD 1.
// For each vertex (ix, iz) in a chunk:
//   - Both indices even → vertex is part of the LOD 1 grid, parent = self.
//   - Otherwise         → linear interpolation of the stride-2 neighbours
//                         that bracket this vertex, matching what LOD 1's
//                         stride-2 triangulation would render at this XZ.
// At runtime the vertex shader does lerp(actual_y, parent_y, morph_factor).
// morph_factor → 1 produces the LOD-1 surface; → 0 keeps the full LOD-0
// surface. Smooth transition with no popping. Skirt verts (after the N²
// interior block) inherit parent_y == self_y so the skirt stays put.
void gen_chunk_parent_y(const Heightmap& hm, int origin_ix, int origin_iz,
                        int sample_dim, std::vector<float>& out) {
    const int N = sample_dim;
    out.clear();
    out.resize(static_cast<size_t>(N) * static_cast<size_t>(N) +
                static_cast<size_t>(4) * static_cast<size_t>(N));
    auto h_at = [&](int lx, int lz) {
        int gx = std::clamp(origin_ix + lx, 0, hm.width()  - 1);
        int gz = std::clamp(origin_iz + lz, 0, hm.height() - 1);
        return hm.at(gx, gz);
    };
    for (int iz = 0; iz < N; ++iz) {
        for (int ix = 0; ix < N; ++ix) {
            float self_y = h_at(ix, iz);
            float py;
            int evx = ix - (ix & 1);
            int evz = iz - (iz & 1);
            int ex2 = std::min(evx + 2, N - 1);
            int ez2 = std::min(evz + 2, N - 1);
            if ((ix & 1) == 0 && (iz & 1) == 0) {
                py = self_y;
            } else if ((iz & 1) == 0) {
                py = 0.5f * (h_at(evx, iz) + h_at(ex2, iz));
            } else if ((ix & 1) == 0) {
                py = 0.5f * (h_at(ix, evz) + h_at(ix, ez2));
            } else {
                py = 0.25f * (h_at(evx, evz) + h_at(ex2, evz) +
                              h_at(evx, ez2) + h_at(ex2, ez2));
            }
            out[static_cast<size_t>(iz) * static_cast<size_t>(N) +
                 static_cast<size_t>(ix)] = py;
        }
    }
    // Skirt twins must NOT move during morph: their actual Y is the
    // interior height minus the skirt drop, so parent_y has to match
    // that exact value or terrain.vert will lerp them upward and
    // expose the gap the skirt was filling. Match the skirt's actual
    // dropped Y rather than the interior height.
    size_t skirt_base = static_cast<size_t>(N) * static_cast<size_t>(N);
    auto skirt_self = [&](int ix0, int iz0, int dx, int dz, size_t edge) {
        for (int k = 0; k < N; ++k) {
            int ix = ix0 + dx * k;
            int iz = iz0 + dz * k;
            float y = h_at(ix, iz) - kTerrainSkirtDepth;
            out[skirt_base + edge * static_cast<size_t>(N) +
                 static_cast<size_t>(k)] = y;
        }
    };
    skirt_self(0,             0,             1, 0, 0);
    skirt_self(sample_dim - 1, 0,             0, 1, 1);
    skirt_self(0,             sample_dim - 1, 1, 0, 2);
    skirt_self(0,             0,             0, 1, 3);
}

// LOD-stride index buffer + skirt strips. Stride 1 = full LOD (every
// vertex), stride 2 = half (every other), 4 = quarter, 8 = eighth.
// Skirt indexing: 4 edges × sample_dim skirt-twin verts laid out at
// offset M = sample_dim². Each LOD stride sweeps the same edge but
// only references the strided subset, while the skirt twins exist at
// every position so any neighbour LOD finds matching X,Z corners.
void gen_indices_for_lod(int sample_dim, int stride, std::vector<uint32_t>& out) {
    const int N = sample_dim;
    if (stride < 1) stride = 1;
    if (stride > N - 1) stride = N - 1;
    const int s = stride;
    const uint32_t M = static_cast<uint32_t>(N) * static_cast<uint32_t>(N);
    out.clear();
    // Approximate index count: (N/s)² quads × 6 + 4 edges × (N/s) quads × 6.
    out.reserve(static_cast<size_t>((N / s) * (N / s)) * 6u +
                 static_cast<size_t>(4 * (N / s)) * 6u);
    auto V = [N](int ix, int iz) {
        return static_cast<uint32_t>(iz * N + ix);
    };
    // Interior triangles at stride s. Cells whose +s neighbour falls past
    // N-1 are skipped — the skirt fills the resulting edge gap.
    for (int iz = 0; iz < N - s; iz += s) {
        for (int ix = 0; ix < N - s; ix += s) {
            uint32_t i00 = V(ix,     iz);
            uint32_t i10 = V(ix + s, iz);
            uint32_t i01 = V(ix,     iz + s);
            uint32_t i11 = V(ix + s, iz + s);
            out.push_back(i00); out.push_back(i11); out.push_back(i10);
            out.push_back(i00); out.push_back(i01); out.push_back(i11);
        }
    }
    // Skirt strips. Winding chosen so each strip's outward normal points
    // away from the chunk (so backface culling keeps the visible side).
    auto skirt_top = [&](int ix) {
        return M + 0u * static_cast<uint32_t>(N) + static_cast<uint32_t>(ix);
    };
    auto skirt_right = [&](int iz) {
        return M + 1u * static_cast<uint32_t>(N) + static_cast<uint32_t>(iz);
    };
    auto skirt_bot = [&](int ix) {
        return M + 2u * static_cast<uint32_t>(N) + static_cast<uint32_t>(ix);
    };
    auto skirt_left = [&](int iz) {
        return M + 3u * static_cast<uint32_t>(N) + static_cast<uint32_t>(iz);
    };
    // Top (z=0, outward -Z)
    for (int ix = 0; ix < N - s; ix += s) {
        uint32_t a  = V(ix,     0);
        uint32_t b  = V(ix + s, 0);
        uint32_t sa = skirt_top(ix);
        uint32_t sb = skirt_top(ix + s);
        out.push_back(a);  out.push_back(b);  out.push_back(sa);
        out.push_back(sa); out.push_back(b);  out.push_back(sb);
    }
    // Right (x=N-1, outward +X)
    for (int iz = 0; iz < N - s; iz += s) {
        uint32_t a  = V(N - 1, iz);
        uint32_t b  = V(N - 1, iz + s);
        uint32_t sa = skirt_right(iz);
        uint32_t sb = skirt_right(iz + s);
        out.push_back(a);  out.push_back(b);  out.push_back(sa);
        out.push_back(sa); out.push_back(b);  out.push_back(sb);
    }
    // Bottom (z=N-1, outward +Z)
    for (int ix = 0; ix < N - s; ix += s) {
        uint32_t a  = V(ix,     N - 1);
        uint32_t b  = V(ix + s, N - 1);
        uint32_t sa = skirt_bot(ix);
        uint32_t sb = skirt_bot(ix + s);
        out.push_back(a);  out.push_back(sa); out.push_back(b);
        out.push_back(sa); out.push_back(sb); out.push_back(b);
    }
    // Left (x=0, outward -X)
    for (int iz = 0; iz < N - s; iz += s) {
        uint32_t a  = V(0, iz);
        uint32_t b  = V(0, iz + s);
        uint32_t sa = skirt_left(iz);
        uint32_t sb = skirt_left(iz + s);
        out.push_back(a);  out.push_back(sa); out.push_back(b);
        out.push_back(sa); out.push_back(sb); out.push_back(b);
    }
}

// Full-LOD index buffer for a chunk of `sample_dim` × `sample_dim` verts.
void gen_full_indices(int sample_dim, std::vector<uint32_t>& out) {
    const int cells = sample_dim - 1;
    out.clear();
    out.reserve(static_cast<size_t>(cells) * static_cast<size_t>(cells) * 6u);
    for (int iz = 0; iz < cells; ++iz) {
        for (int ix = 0; ix < cells; ++ix) {
            uint32_t i00 = static_cast<uint32_t>(iz * sample_dim + ix);
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + static_cast<uint32_t>(sample_dim);
            uint32_t i11 = i01 + 1;
            out.push_back(i00); out.push_back(i11); out.push_back(i10);
            out.push_back(i00); out.push_back(i01); out.push_back(i11);
        }
    }
}

// Half-LOD index buffer: skip every other vertex on both axes. The
// vertex buffer stays at full resolution — we just reference a sparser
// subset. Cells span 2x in each axis. Cracks at chunk boundaries are
// hidden by the next-LOD-up chunks meeting at the seam (we always
// match neighbour LODs in this build because the LOD is purely
// distance-driven and neighbours are at most one ring apart).
void gen_half_indices(int sample_dim, std::vector<uint32_t>& out) {
    const int cells = sample_dim - 1;
    out.clear();
    if (cells < 2) return;
    out.reserve(static_cast<size_t>(cells / 2) *
                static_cast<size_t>(cells / 2) * 6u);
    for (int iz = 0; iz < cells - 1; iz += 2) {
        for (int ix = 0; ix < cells - 1; ix += 2) {
            uint32_t i00 = static_cast<uint32_t>(iz * sample_dim + ix);
            uint32_t i20 = i00 + 2;
            uint32_t i02 = i00 + static_cast<uint32_t>(2 * sample_dim);
            uint32_t i22 = i02 + 2;
            out.push_back(i00); out.push_back(i22); out.push_back(i20);
            out.push_back(i00); out.push_back(i02); out.push_back(i22);
        }
    }
}

} // namespace

namespace {
// Inner ray-march for a single rectangular region. `out` points to the
// top-left of a w×h sub-image with the given row stride (in bytes).
// 0 = lit, 255 = in shadow. The full-map and tile bake share this loop.
void bake_shadow_region(const Heightmap& hm, glm::vec3 L,
                        int ix0, int iz0, int w, int h,
                        uint8_t* out, int stride) {
    if (L.y < 0.05f) {
        for (int dz = 0; dz < h; ++dz)
            std::fill_n(out + static_cast<size_t>(dz) * static_cast<size_t>(stride),
                        w, uint8_t(255));
        return;
    }
    const float step = hm.cell * 0.5f;
    const float max_t = 400.0f;
    const int   max_steps = static_cast<int>(max_t / step);
    const int W = hm.width();
    const int H = hm.height();
    for (int dz = 0; dz < h; ++dz) {
        int iz = iz0 + dz;
        if (iz < 0 || iz >= H) {
            std::fill_n(out + static_cast<size_t>(dz) * static_cast<size_t>(stride),
                        w, uint8_t(0));
            continue;
        }
        uint8_t* row = out + static_cast<size_t>(dz) * static_cast<size_t>(stride);
        for (int dx = 0; dx < w; ++dx) {
            int ix = ix0 + dx;
            if (ix < 0 || ix >= W) { row[dx] = 0; continue; }
            float h0 = hm.at(ix, iz);
            glm::vec3 p0(hm.origin_x + static_cast<float>(ix) * hm.cell,
                         h0 + 0.20f,
                         hm.origin_z + static_cast<float>(iz) * hm.cell);
            bool shadowed = false;
            for (int s = 1; s <= max_steps; ++s) {
                float t = static_cast<float>(s) * step;
                glm::vec3 p = p0 + L * t;
                float h_at_p = hm.sample_world(p.x, p.z);
                if (p.y < h_at_p) { shadowed = true; break; }
                if (p.y - h0 > 100.0f) break;
            }
            row[dx] = shadowed ? 255 : 0;
        }
    }
}
} // namespace

void bake_heightmap_shadow_tile(const Heightmap& hm, glm::vec3 sun_dir,
                                int ix0, int iz0, int w, int h,
                                uint8_t* out_tile) {
    bake_shadow_region(hm, glm::normalize(sun_dir), ix0, iz0, w, h, out_tile, w);
}

std::vector<uint8_t> bake_heightmap_shadow(const Heightmap& hm,
                                           glm::vec3 sun_dir) {
    const int W = hm.width();
    const int H = hm.height();
    std::vector<uint8_t> out(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
    glm::vec3 L = glm::normalize(sun_dir);

    // Multi-threaded — split rows across hardware threads. ~1024² cells
    // × ~80 marching steps drops from ~1s single-threaded to ~100-150 ms
    // with 8-12 worker threads. Used at level load; sun-change re-bakes
    // go through the progressive tile path instead.
    unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    int rows_per = (H + static_cast<int>(n_threads) - 1) /
                   static_cast<int>(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        int z_lo = static_cast<int>(t) * rows_per;
        int z_hi = std::min(H, z_lo + rows_per);
        if (z_lo >= z_hi) break;
        workers.emplace_back([&, z_lo, z_hi]() {
            uint8_t* row0 = out.data() +
                static_cast<size_t>(z_lo) * static_cast<size_t>(W);
            bake_shadow_region(hm, L, 0, z_lo, W, z_hi - z_lo, row0, W);
        });
    }
    for (auto& th : workers) th.join();

    log::infof("[terrain] heightmap shadow baked: %dx%d (%u threads)",
               W, H, n_threads);
    return out;
}

Mesh build_terrain_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                        uint32_t queue_family, const Heightmap& hm) {
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;
    glm::vec3 amin, amax;
    gen_chunk_verts(hm, 0, 0, hm.dim + 1, verts, amin, amax);
    gen_full_indices(hm.dim + 1, indices);
    Mesh m = create_mesh_from_data(device, alloc, queue, queue_family,
                                   verts.data(), static_cast<uint32_t>(verts.size()),
                                   indices.data(), static_cast<uint32_t>(indices.size()));
    log::infof("[terrain] mesh: %zu verts, %zu indices",
               verts.size(), indices.size());
    return m;
}

namespace {
// Upload an arbitrary host-resident byte buffer to a device-local
// read-only buffer with the given usage flags. Used by both
// upload_index_buffer (UINT32 IBO) and the parent_y vertex stream.
void upload_typed_buffer(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family,
                         const void* data, VkDeviceSize size,
                         VkBufferUsageFlags usage,
                         VkBuffer& out_buf, VmaAllocation& out_alloc) {
    VkBufferCreateInfo bci_stage{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci_stage{};
    aci_stage.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    vmaCreateBuffer(alloc, &bci_stage, &aci_stage, &stage, &stage_alloc, nullptr);
    void* m = nullptr;
    vmaMapMemory(alloc, stage_alloc, &m);
    std::memcpy(m, data, static_cast<size_t>(size));
    vmaUnmapMemory(alloc, stage_alloc);

    VkBufferCreateInfo bci_dst{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci_dst{};
    aci_dst.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    vmaCreateBuffer(alloc, &bci_dst, &aci_dst, &out_buf, &out_alloc, nullptr);

    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy copy{ 0, 0, size };
    vkCmdCopyBuffer(cb, stage, out_buf, 1, &copy);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1, .pCommandBuffers = &cb,
        .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr,
    };
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkDestroyCommandPool(device, pool, nullptr);
    vmaDestroyBuffer(alloc, stage, stage_alloc);
}

// Upload a uint32_t index buffer as a device-local read-only IBO.
void upload_index_buffer(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family,
                         const uint32_t* data, uint32_t count,
                         VkBuffer& out_buf, VmaAllocation& out_alloc) {
    const VkDeviceSize size = sizeof(uint32_t) * static_cast<VkDeviceSize>(count);
    upload_typed_buffer(device, alloc, queue, queue_family, data, size,
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT, out_buf, out_alloc);
}
} // namespace

TerrainChunkSet build_terrain_chunks(VkDevice device, VmaAllocator alloc,
                                     VkQueue queue, uint32_t queue_family,
                                     const Heightmap& hm,
                                     int chunks_per_side) {
    TerrainChunkSet set;
    if (chunks_per_side <= 0 || hm.dim % chunks_per_side != 0) {
        log::errorf("[terrain] bad chunks_per_side=%d for dim=%d (must divide)",
                    chunks_per_side, hm.dim);
        return set;
    }
    set.chunks_per_side = chunks_per_side;
    set.chunk_cells = hm.dim / chunks_per_side;
    const int sample_dim = set.chunk_cells + 1;

    // Pre-build the four LOD index buffers (stride 1, 2, 4, 8). They're
    // identical for every chunk (same topology), so we share the host
    // arrays but each chunk gets its own GPU buffer for cache locality.
    std::vector<uint32_t> idx_lod[kTerrainLodCount];
    int strides[kTerrainLodCount] = { 1, 2, 4, 8 };
    for (int i = 0; i < kTerrainLodCount; ++i) {
        gen_indices_for_lod(sample_dim, strides[i], idx_lod[i]);
    }

    set.chunks.reserve(static_cast<size_t>(chunks_per_side) *
                       static_cast<size_t>(chunks_per_side));
    std::vector<Vertex> verts;
    std::vector<float>  parent_y;
    for (int cz = 0; cz < chunks_per_side; ++cz) {
        for (int cx = 0; cx < chunks_per_side; ++cx) {
            TerrainChunk c{};
            c.cx = cx; c.cz = cz;
            c.origin_ix = cx * set.chunk_cells;
            c.origin_iz = cz * set.chunk_cells;
            c.sample_dim = sample_dim;

            gen_chunk_verts(hm, c.origin_ix, c.origin_iz, sample_dim,
                            verts, c.aabb_min, c.aabb_max);
            c.center = 0.5f * (c.aabb_min + c.aabb_max);

            // LOD 0 lives in the main mesh's index buffer.
            c.mesh = create_mesh_from_data(
                device, alloc, queue, queue_family,
                verts.data(), static_cast<uint32_t>(verts.size()),
                idx_lod[0].data(), static_cast<uint32_t>(idx_lod[0].size()));
            c.index_count_lod[0] = static_cast<uint32_t>(idx_lod[0].size());

            // LODs 1..N-1 each get a standalone index buffer.
            for (int lod = 1; lod < kTerrainLodCount; ++lod) {
                upload_index_buffer(device, alloc, queue, queue_family,
                                     idx_lod[lod].data(),
                                     static_cast<uint32_t>(idx_lod[lod].size()),
                                     c.ibo_lod[lod - 1], c.ibo_lod_alloc[lod - 1]);
                c.index_count_lod[lod] = static_cast<uint32_t>(idx_lod[lod].size());
            }

            // CD-LOD parent_y stream — one float per vertex (interior +
            // skirt verts), bound at vertex binding 1 by the terrain
            // pipeline. Drives the LOD 0 ↔ LOD 1 morph in terrain.vert.
            gen_chunk_parent_y(hm, c.origin_ix, c.origin_iz, sample_dim, parent_y);
            upload_typed_buffer(device, alloc, queue, queue_family,
                                 parent_y.data(),
                                 static_cast<VkDeviceSize>(parent_y.size()) *
                                 sizeof(float),
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                 c.parent_y_buffer, c.parent_y_alloc);

            set.chunks.push_back(std::move(c));
        }
    }
    log::infof("[terrain] chunks: %dx%d (= %d) at %d cells/chunk; verts/chunk=%d "
               "lod0_tris=%u lod1=%u lod2=%u lod3=%u",
               chunks_per_side, chunks_per_side,
               static_cast<int>(set.chunks.size()),
               set.chunk_cells,
               sample_dim * sample_dim + 4 * sample_dim,
               static_cast<unsigned>(idx_lod[0].size() / 3u),
               static_cast<unsigned>(idx_lod[1].size() / 3u),
               static_cast<unsigned>(idx_lod[2].size() / 3u),
               static_cast<unsigned>(idx_lod[3].size() / 3u));
    return set;
}

void destroy_terrain_chunks(VkDevice device, VmaAllocator alloc,
                            TerrainChunkSet& set) {
    (void)device;
    for (auto& c : set.chunks) {
        if (c.mesh.vertex_buffer) destroy_mesh(alloc, c.mesh);
        for (int lod = 0; lod < kTerrainLodCount - 1; ++lod) {
            if (c.ibo_lod[lod]) {
                vmaDestroyBuffer(alloc, c.ibo_lod[lod], c.ibo_lod_alloc[lod]);
                c.ibo_lod[lod] = VK_NULL_HANDLE;
                c.ibo_lod_alloc[lod] = nullptr;
            }
        }
        if (c.parent_y_buffer) {
            vmaDestroyBuffer(alloc, c.parent_y_buffer, c.parent_y_alloc);
            c.parent_y_buffer = VK_NULL_HANDLE;
            c.parent_y_alloc = nullptr;
        }
    }
    set.chunks.clear();
    set.chunks_per_side = 0;
    set.chunk_cells = 0;
}

void rebuild_chunk_vertices(VkDevice device, VmaAllocator alloc, VkQueue queue,
                            uint32_t queue_family,
                            const Heightmap& hm, TerrainChunk& c) {
    std::vector<Vertex> verts;
    gen_chunk_verts(hm, c.origin_ix, c.origin_iz, c.sample_dim,
                    verts, c.aabb_min, c.aabb_max);
    // Re-upload via stage-then-copy. We could keep a persistent host-
    // mapped staging buffer for sculpt strokes if this becomes a hot
    // path, but a single chunk is ~40KB so the one-shot path is fine.
    const VkDeviceSize size = sizeof(Vertex) *
                              static_cast<VkDeviceSize>(verts.size());
    VkBufferCreateInfo bci_stage{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    vmaCreateBuffer(alloc, &bci_stage, &aci, &stage, &stage_alloc, nullptr);
    void* mapped = nullptr;
    vmaMapMemory(alloc, stage_alloc, &mapped);
    std::memcpy(mapped, verts.data(), static_cast<size_t>(size));
    vmaUnmapMemory(alloc, stage_alloc);

    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cb, stage, c.mesh.vertex_buffer, 1, &region);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = nullptr,
        .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1, .pCommandBuffers = &cb,
        .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr,
    };
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkDestroyCommandPool(device, pool, nullptr);
    vmaDestroyBuffer(alloc, stage, stage_alloc);
}

} // namespace qlike
