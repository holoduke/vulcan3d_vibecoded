#include "engine/terrain.h"

#include "engine/log.h"

#define FASTNOISE_LITE_IMPLEMENTATION
#include <FastNoiseLite.h>

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// === FBM heightmap matching terrain_raymarch.frag ============================
//
// The fragment shader's terrainM uses 2D value noise with analytical
// derivatives + derivative-erosion suppression. We mirror it here so the
// CPU heightmap (driving physics / grass / mesh / BLAS) matches what the
// shader draws. Constants are bit-identical to the shader; small float
// drift is OK because the visible raymarch is what the user sees, and a
// few cm of grass / physics offset is invisible.
namespace {

inline glm::vec3 rm_fract(glm::vec3 v) { return v - glm::floor(v); }
inline float     rm_fract1(float v)    { return v - std::floor(v); }

// hash21 вЂ” exactly mirrors the shader's hash function.
inline float rm_hash21(glm::vec2 p) {
    glm::vec3 p3 = rm_fract(glm::vec3(p.x, p.y, p.x) * 0.1031f);
    p3 += glm::dot(p3, glm::vec3(p3.y, p3.z, p3.x) + 19.19f);
    return rm_fract1((p3.x + p3.y) * p3.z);
}

inline float rm_noised(glm::vec2 p, glm::vec2& deriv) {
    glm::vec2 i = glm::floor(p);
    glm::vec2 f = p - i;
    // Hermite cubic вЂ” MUST stay byte-identical in meaning to
    // terrain_raymarch.frag / grass_raymarch.frag noised(). This is
    // the CPU FBM that bakes the physics heightfield + grass baseline;
    // if it diverges from the shader the collision surface no longer
    // matches the visible terrain. (Reverted quintic experiment.)
    glm::vec2 u  = f * f * (3.0f - 2.0f * f);
    glm::vec2 du = 6.0f * f * (1.0f - f);
    float a = rm_hash21(i + glm::vec2(0, 0));
    float b = rm_hash21(i + glm::vec2(1, 0));
    float c = rm_hash21(i + glm::vec2(0, 1));
    float d = rm_hash21(i + glm::vec2(1, 1));
    float val = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    deriv = du * (glm::vec2(b - a, c - a) +
                  (a - b - c + d) * glm::vec2(u.y, u.x));
    return val;
}

constexpr float kRmTerrainScale  = 0.003f;
constexpr float kRmTerrainHeight = 120.0f;
constexpr float kRmPlateauBlend  = 20.0f;   // matches shader/world plateau_blend

// Mirror of shader m2 = mat2(0.8, -0.6, 0.6, 0.8). GLSL is column-major;
// applying m * p = (col0*p.x + col1*p.y) gives:
//   x' =  0.8*p.x + 0.6*p.y
//   y' = -0.6*p.x + 0.8*p.y
inline glm::vec2 rm_rotate(glm::vec2 p) {
    return glm::vec2(0.8f * p.x + 0.6f * p.y,
                     -0.6f * p.x + 0.8f * p.y);
}

float rm_terrain(glm::vec2 worldXZ, const HeightmapParams& p, int octaves = 9) {
    glm::vec2 wp = worldXZ;
    glm::vec2 q  = worldXZ * kRmTerrainScale;
    float a = 0.0f, b = 1.0f;
    glm::vec2 d(0.0f);
    for (int i = 0; i < octaves; i++) {
        glm::vec2 deriv;
        float n = rm_noised(q, deriv);
        d += deriv;
        a += b * n / (1.0f + glm::dot(d, d));
        b *= 0.5f;
        q = rm_rotate(q) * 2.0f;
    }
    float h = a * kRmTerrainHeight;

    // Soft plateau blend (mirrors shader's plateauWeight + mix).
    glm::vec2 ext = p.plateau_extent;
    glm::vec2 ab  = glm::abs(wp - p.plateau_center) - ext;
    float dout = std::max(std::max(ab.x, 0.0f), std::max(ab.y, 0.0f));
    float t = dout / kRmPlateauBlend;
    t = std::clamp(t, 0.0f, 1.0f);
    float w = 1.0f - (t * t * (3.0f - 2.0f * t));   // smoothstep
    return h * (1.0f - w) + p.plateau_height * w;
}

} // namespace

float raymarch_fine_correction(const Heightmap& hm, const HeightmapParams& p,
                               float x, float z, int octaves) {
    // Exact high-octave FBM minus the bilinear 9-octave grid the CPU
    // heightmap stores (sculpt deltas cancel out of the difference).
    float fx = (x - hm.origin_x) / hm.cell, fz = (z - hm.origin_z) / hm.cell;
    float ix = std::floor(fx), iz = std::floor(fz);
    float tx = fx - ix, tz = fz - iz;
    auto g = [&](float gx, float gz) {
        return rm_terrain(glm::vec2(hm.origin_x + gx * hm.cell, hm.origin_z + gz * hm.cell), p, 9);
    };
    float bil = (g(ix, iz) * (1.0f - tx) + g(ix + 1.0f, iz) * tx) * (1.0f - tz) +
                (g(ix, iz + 1.0f) * (1.0f - tx) + g(ix + 1.0f, iz + 1.0f) * tx) * tz;
    return rm_terrain(glm::vec2(x, z), p, octaves) - bil;
}

Heightmap generate_heightmap_raymarch(const HeightmapParams& p) {
    Heightmap hm;
    hm.dim       = p.dim;
    hm.cell      = p.cell_size;
    hm.origin_x  = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    hm.origin_z  = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    const int W = hm.width();
    const int H = hm.height();
    hm.heights.resize(static_cast<size_t>(W) * static_cast<size_t>(H));

    // Multi-thread row-strip generation. 9 octaves Г— 2048ВІ в‰€ 38 M FBM
    // evals вЂ” single-threaded would be ~10 s, parallel ~2-3 s on 8+
    // cores.
    const int hw = std::max<int>(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    workers.reserve(hw);
    int rows = (H + hw - 1) / hw;
    for (int t = 0; t < hw; ++t) {
        int z0 = t * rows;
        int z1 = std::min(H, z0 + rows);
        if (z0 >= z1) break;
        workers.emplace_back([&, z0, z1] {
            for (int iz = z0; iz < z1; ++iz) {
                float wz = hm.origin_z + static_cast<float>(iz) * hm.cell;
                for (int ix = 0; ix < W; ++ix) {
                    float wx = hm.origin_x + static_cast<float>(ix) * hm.cell;
                    hm.heights[static_cast<size_t>(iz) *
                               static_cast<size_t>(W) +
                               static_cast<size_t>(ix)] =
                        rm_terrain(glm::vec2(wx, wz), p);
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    log::infof("[terrain] heightmap_raymarch %dx%d cells, %.0fm side, plateau h=%.1f",
               p.dim, p.dim, hm.side(), p.plateau_height);
    return hm;
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
    // вЂ” gives the terrain something rocky to look at without specific
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
            // elevations only. Empirical weights вЂ” pick something
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
// `hm` covering [origin_ix, origin_ix + sample_dim) Г— [origin_iz, ...).
// Per-vertex normals come from the global heightmap gradient (central
// differences) so adjacent chunks agree on shared-edge normals вЂ” face-
// average accumulation only saw triangles WITHIN a chunk and produced
// faint shading seams at chunk boundaries.
// Skirt depth in metres. Each chunk-edge vertex has a "twin" vertex at the
// same XZ but lowered Y by this amount; the skirt strip connects them so
// that LOD-mismatch cracks at chunk boundaries are hidden by a vertical
// wall rather than producing a see-through gap. 8 m is enough to cover
// worst-case adjacent-cell height jumps on the steepest mountains.
constexpr float kTerrainSkirtDepth = 8.0f;

// densify >= 1: sub-divide each source heightmap cell within the chunk
// into `densify` quads per axis. sample_dim therefore corresponds to
// chunk_cells * densify + 1 vertices per side. The vertex at densified
// integer coords (ix, iz) maps to source heightmap fractional coords
// (origin_ix + ix/densify, origin_iz + iz/densify) and the height is
// bilinearly sampled. At densify=1 the math collapses to integer sample
// lookups (the original code path), and at densify=N the vertex grid is
// N x N times finer in BOTH axes than the source heightmap.
void gen_chunk_verts(const Heightmap& hm, int origin_ix, int origin_iz,
                     int sample_dim,
                     std::vector<Vertex>& verts,
                     glm::vec3& aabb_min, glm::vec3& aabb_max,
                     int densify = 1) {
    if (densify < 1) densify = 1;
    verts.clear();
    verts.reserve(static_cast<size_t>(sample_dim) * static_cast<size_t>(sample_dim) +
                   static_cast<size_t>(4) * static_cast<size_t>(sample_dim));
    aabb_min = glm::vec3( 1e9f);
    aabb_max = glm::vec3(-1e9f);
    // 1-cell normal gradient вЂ” gives per-vertex normals that match the
    // 1m triangle face at LOD 0 for crisp near detail. The earlier
    // 4-cell smoothed setting was a band-aid for LOD-mismatched far
    // chunks producing patchy slope-driven rock blend; that's now
    // handled by fading slope-rock at distance and by the cull-mask
    // trick keeping RT shadow rays off the terrain BLAS, so we can
    // get sharp 1m detail back without the artifacts returning.
    constexpr int kNormalRadius = 1;
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
    // Bilinear-sample the source heightmap at fractional source-cell
    // coords. At densify=1 fx,fz are always integers and the math
    // reduces to a clamped hm.at() lookup (no precision loss vs the
    // original code).
    const float inv_d = 1.0f / static_cast<float>(densify);
    auto sample_h = [&](int ix, int iz) -> float {
        float fx = static_cast<float>(origin_ix) + static_cast<float>(ix) * inv_d;
        float fz = static_cast<float>(origin_iz) + static_cast<float>(iz) * inv_d;
        int ax = static_cast<int>(std::floor(fx));
        int az = static_cast<int>(std::floor(fz));
        float tx = fx - static_cast<float>(ax);
        float tz = fz - static_cast<float>(az);
        int axc = std::clamp(ax,     0, hm.width()  - 1);
        int axn = std::clamp(ax + 1, 0, hm.width()  - 1);
        int azc = std::clamp(az,     0, hm.height() - 1);
        int azn = std::clamp(az + 1, 0, hm.height() - 1);
        float h00 = hm.at(axc, azc);
        float h10 = hm.at(axn, azc);
        float h01 = hm.at(axc, azn);
        float h11 = hm.at(axn, azn);
        float h0 = h00 * (1.0f - tx) + h10 * tx;
        float h1 = h01 * (1.0f - tx) + h11 * tx;
        return h0 * (1.0f - tz) + h1 * tz;
    };
    // Source-aligned integer (gx, gz) used for the normal sample. For
    // sub-cell densified verts we round to the nearest source cell;
    // gives the same heightmap-gradient normal as the surrounding
    // source verts which avoids visible faceted shading on the
    // upsampled grid (the surface itself is the bilinear interpolant,
    // so its slope is already smooth between source verts).
    auto nearest_src = [&](int ix, int iz, int& gx, int& gz) {
        float fx = static_cast<float>(origin_ix) + static_cast<float>(ix) * inv_d;
        float fz = static_cast<float>(origin_iz) + static_cast<float>(iz) * inv_d;
        gx = std::clamp(static_cast<int>(std::floor(fx + 0.5f)),
                         0, hm.width()  - 1);
        gz = std::clamp(static_cast<int>(std::floor(fz + 0.5f)),
                         0, hm.height() - 1);
    };
    // Interior (sample_dim Г— sample_dim) at densified resolution.
    for (int iz = 0; iz < sample_dim; ++iz) {
        for (int ix = 0; ix < sample_dim; ++ix) {
            float wx = hm.origin_x + (static_cast<float>(origin_ix) +
                                       static_cast<float>(ix) * inv_d) * hm.cell;
            float wz = hm.origin_z + (static_cast<float>(origin_iz) +
                                       static_cast<float>(iz) * inv_d) * hm.cell;
            float hy = sample_h(ix, iz);
            int gx, gz; nearest_src(ix, iz, gx, gz);
            glm::vec3 p(wx, hy, wz);
            glm::vec2 uv(p.x, p.z);
            verts.push_back({p, sample_normal(gx, gz), uv});
            aabb_min = glm::min(aabb_min, p);
            aabb_max = glm::max(aabb_max, p);
        }
    }
    // Skirt twins: 4 edges Г— sample_dim verts (the densified perimeter).
    // Each one is the interior edge vertex with Y dropped by
    // kTerrainSkirtDepth. Normal inherits from the corresponding
    // interior vertex (heightmap-gradient normal, mostly upward) so the
    // skirt shades like the terrain above it rather than reading as a
    // dark vertical wall вЂ” distant chunks at low LOD would otherwise
    // show their outward-facing skirts as visible dark faces against
    // the sky.
    auto add_skirt_edge = [&](int ix0, int iz0, int dx, int dz) {
        for (int k = 0; k < sample_dim; ++k) {
            int ix = ix0 + dx * k;
            int iz = iz0 + dz * k;
            float wx = hm.origin_x + (static_cast<float>(origin_ix) +
                                       static_cast<float>(ix) * inv_d) * hm.cell;
            float wz = hm.origin_z + (static_cast<float>(origin_iz) +
                                       static_cast<float>(iz) * inv_d) * hm.cell;
            float hy = sample_h(ix, iz);
            int gx, gz; nearest_src(ix, iz, gx, gz);
            glm::vec3 p(wx, hy - kTerrainSkirtDepth, wz);
            verts.push_back({p, sample_normal(gx, gz), glm::vec2(p.x, p.z)});
        }
    };
    add_skirt_edge(0,             0,             1, 0);  // top  (z=0)
    add_skirt_edge(sample_dim - 1, 0,             0, 1);  // right(x=N-1)
    add_skirt_edge(0,             sample_dim - 1, 1, 0);  // bot  (z=N-1)
    add_skirt_edge(0,             0,             0, 1);  // left (x=0)

    // Expand the chunk AABB to cover (a) the skirt twins (dropped by
    // kTerrainSkirtDepth) and (b) the worst-case tessellated vertex
    // displacement applied in terrain_tess.tese. The .tese is bumps-
    // up-only. Worst-case stack at the current UI sliders:
    //   - per-type ampM peaks at ~1.5 m on rock pixels when the
    //     "Rock relief" slider hits its max (1.0) -- see the
    //     `ampM += kRockReliefV() * (0.40 + wRockN * 1.10)` line.
    //   - the rocky-grass height-map contribution adds up to
    //     kRockyAmp == `terrain_disp_amp` (slider max 1.0 m) on top of
    //     that on grass-band pixels.
    // Total peak ~2.5 m up. 4 m of headroom keeps a comfortable margin
    // so a chunk near the frustum edge whose interior verts barely
    // fall outside the frustum still has its displaced verts inside
    // the AABB -- otherwise we frustum-cull it and drop a chunk-sized
    // patch of terrain at the screen edge. Was 3 m before -- tight at
    // max slider settings; bumped to 4 m for safety headroom.
    aabb_min.y -= kTerrainSkirtDepth;
    aabb_max.y += 4.0f;
}



// Full-LOD index buffer for a chunk of `sample_dim` Г— `sample_dim` verts.
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


} // namespace

namespace {
// Inner ray-march for a single rectangular region. `out` points to the
// top-left of a wГ—h sub-image with the given row stride (in bytes).
// RG8 interleaved, 2 bytes per texel:
//   R: 0 = lit, 255 = in shadow
//   G: blocker distance along the sun ray, t / 400 m normalised to 0-255
//      (0 where lit). The shaders run PCSS on this — penumbra width
//      scales with blocker distance, so a tower top 60 m up casts a
//      wide soft shadow while a low wall stays tight and dark.
// The full-map and tile bake share this loop. `stride` is in TEXELS.
void bake_shadow_region(const Heightmap& hm, glm::vec3 L,
                        int ix0, int iz0, int w, int h,
                        uint8_t* out, int stride) {
    const float max_t = 400.0f;
    if (L.y < 0.05f) {
        for (int dz = 0; dz < h; ++dz) {
            uint8_t* row = out + static_cast<size_t>(dz) *
                                  static_cast<size_t>(stride) * 2;
            for (int dx = 0; dx < w; ++dx) {
                row[2 * dx + 0] = 255;
                row[2 * dx + 1] = 255;
            }
        }
        return;
    }
    const float step = hm.cell * 0.5f;
    const int   max_steps = static_cast<int>(max_t / step);
    const int W = hm.width();
    const int H = hm.height();
    for (int dz = 0; dz < h; ++dz) {
        int iz = iz0 + dz;
        uint8_t* row = out + static_cast<size_t>(dz) *
                              static_cast<size_t>(stride) * 2;
        if (iz < 0 || iz >= H) {
            std::fill_n(row, static_cast<size_t>(w) * 2, uint8_t(0));
            continue;
        }
        for (int dx = 0; dx < w; ++dx) {
            int ix = ix0 + dx;
            if (ix < 0 || ix >= W) { row[2 * dx] = 0; row[2 * dx + 1] = 0; continue; }
            float h0 = hm.at(ix, iz);
            glm::vec3 p0(hm.origin_x + static_cast<float>(ix) * hm.cell,
                         h0 + 0.20f,
                         hm.origin_z + static_cast<float>(iz) * hm.cell);
            bool shadowed = false;
            float t_hit = 0.0f;
            for (int s = 1; s <= max_steps; ++s) {
                float t = static_cast<float>(s) * step;
                glm::vec3 p = p0 + L * t;
                float h_at_p = hm.sample_world(p.x, p.z);
                if (p.y < h_at_p) { shadowed = true; t_hit = t; break; }
                if (p.y - h0 > 100.0f) break;
            }
            row[2 * dx + 0] = shadowed ? 255 : 0;
            row[2 * dx + 1] = shadowed
                ? static_cast<uint8_t>(std::min(t_hit * (255.0f / max_t), 255.0f))
                : 0;
        }
    }
}
} // namespace

void bake_heightmap_shadow_tile(const Heightmap& hm, glm::vec3 sun_dir,
                                int ix0, int iz0, int w, int h,
                                uint8_t* out_tile) {
    bake_shadow_region(hm, glm::normalize(sun_dir), ix0, iz0, w, h, out_tile, w);
}

void bake_heightmap_shadow_tile_ss(const Heightmap& hm, glm::vec3 sun_dir,
                                    int tx0, int tz0, int w, int h, int ss,
                                    uint8_t* out_tile) {
    if (ss < 1) ss = 1;
    glm::vec3 L = glm::normalize(sun_dir);
    const float max_t = 400.0f;
    if (L.y < 0.05f) {
        std::fill_n(out_tile,
                     static_cast<size_t>(w) * static_cast<size_t>(h) * 2,
                     uint8_t(255));
        return;
    }
    const float sub_cell = hm.cell / static_cast<float>(ss);
    const float step = sub_cell * 0.5f;
    const int   max_steps = static_cast<int>(max_t / step);
    for (int dz = 0; dz < h; ++dz) {
        uint8_t* row = out_tile + static_cast<size_t>(dz) *
                                   static_cast<size_t>(w) * 2;
        for (int dx = 0; dx < w; ++dx) {
            float wx = hm.origin_x + (static_cast<float>(tx0 + dx) + 0.5f) * sub_cell;
            float wz = hm.origin_z + (static_cast<float>(tz0 + dz) + 0.5f) * sub_cell;
            float h0 = hm.sample_world(wx, wz);
            glm::vec3 p0(wx, h0 + 0.20f, wz);
            bool shadowed = false;
            float t_hit = 0.0f;
            for (int s = 1; s <= max_steps; ++s) {
                float t = static_cast<float>(s) * step;
                glm::vec3 p = p0 + L * t;
                float h_at_p = hm.sample_world(p.x, p.z);
                if (p.y < h_at_p) { shadowed = true; t_hit = t; break; }
                if (p.y - h0 > 100.0f) break;
            }
            row[2 * dx + 0] = shadowed ? 255 : 0;
            row[2 * dx + 1] = shadowed
                ? static_cast<uint8_t>(std::min(t_hit * (255.0f / max_t), 255.0f))
                : 0;
        }
    }
}

std::vector<uint8_t> bake_heightmap_shadow(const Heightmap& hm,
                                           glm::vec3 sun_dir) {
    const int W = hm.width();
    const int H = hm.height();
    std::vector<uint8_t> out(static_cast<size_t>(W) * static_cast<size_t>(H) * 2, 0);
    glm::vec3 L = glm::normalize(sun_dir);

    // Multi-threaded вЂ” split rows across hardware threads. ~1024ВІ cells
    // Г— ~80 marching steps drops from ~1s single-threaded to ~100-150 ms
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
                static_cast<size_t>(z_lo) * static_cast<size_t>(W) * 2;
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

void rebuild_chunk_vertices(VkDevice device, VmaAllocator alloc, VkQueue queue,
                            uint32_t queue_family,
                            const Heightmap& hm, TerrainChunk& c) {
    std::vector<Vertex> verts;
    // Thread c.densify through so the brush rebuild matches the chunk's
    // CURRENT density. Without this, the VBO is regenerated at densify=1
    // even when the chunk's sample_dim was sized for densify>1; the
    // resulting VBO has the wrong number of vertices and gen_chunk_verts
    // bilinear-samples on the wrong source-grid spacing. The next densify
    // dirty-handler tick would eventually reupload at the correct density,
    // but during the brush stroke the chunk briefly renders with broken
    // topology. fake chunks built from default initialisation get
    // densify=1 from the struct default so the previous behaviour is
    // preserved for refresh_terrain_blas's whole-mesh "chunk".
    gen_chunk_verts(hm, c.origin_ix, c.origin_iz, c.sample_dim,
                    verts, c.aabb_min, c.aabb_max, c.densify);
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
