#include "engine/world_props.h"

#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/lsystem.h"

#include <glm/geometric.hpp>
#include <glm/common.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace qlike {
namespace {

using game::LevelMesh;
using game::LevelVertex;

constexpr float kPi = 3.14159265358979f;

// ---- noise ----
float h3(glm::vec3 p) {
    p = glm::fract(p * 0.3183099f + glm::vec3(0.1f));
    p *= 17.0f;
    return glm::fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(glm::vec3 x) {
    glm::vec3 i = glm::floor(x), f = glm::fract(x);
    f = f * f * (3.0f - 2.0f * f);
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    return L(L(L(h3(i), h3(i + glm::vec3(1, 0, 0)), f.x),
               L(h3(i + glm::vec3(0, 1, 0)), h3(i + glm::vec3(1, 1, 0)), f.x), f.y),
             L(L(h3(i + glm::vec3(0, 0, 1)), h3(i + glm::vec3(1, 0, 1)), f.x),
               L(h3(i + glm::vec3(0, 1, 1)), h3(i + glm::vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(glm::vec3 p, int oct) {
    float a = 0.5f, s = 0.0f;
    for (int i = 0; i < oct; ++i) { s += a * vnoise(p); p = p * 2.03f + glm::vec3(5.1f, 1.7f, 9.3f); a *= 0.5f; }
    return s;
}

// ---- icosphere template (shared verts, smooth) ----
struct Ico { std::vector<glm::vec3> v; std::vector<uint32_t> f; };
Ico make_ico(int subdiv) {
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    Ico m;
    m.v = { {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
            {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1} };
    for (auto& p : m.v) p = glm::normalize(p);
    m.f = { 0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
            3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1 };
    for (int s = 0; s < subdiv; ++s) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            auto key = std::minmax(a, b);
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            uint32_t id = uint32_t(m.v.size());
            m.v.push_back(glm::normalize(m.v[a] + m.v[b]));
            mid[key] = id;
            return id;
        };
        std::vector<uint32_t> nf;
        for (size_t i = 0; i < m.f.size(); i += 3) {
            uint32_t a = m.f[i], b = m.f[i + 1], c = m.f[i + 2];
            uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            nf.insert(nf.end(), { a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca });
        }
        m.f = std::move(nf);
    }
    return m;
}

void push_v(LevelMesh& m, const LevelVertex& v) {
    m.verts.push_back(v);
    m.bmin = glm::min(m.bmin, v.p);
    m.bmax = glm::max(m.bmax, v.p);
}

// Displaced blob from the icosphere template. `strata` squashes into
// sedimentary ledges; `ground` (if > -1e8) clamps the underside so the
// rock sits in the terrain instead of floating. sway = (weight, phase).
void add_blob(LevelMesh& m, const Ico& ico, glm::vec3 c, glm::vec3 radii, float rot,
              float seed, float amp, float freq, float strata, float ground,
              glm::vec2 sway_base, float sway_top) {
    const uint32_t base = uint32_t(m.verts.size());
    std::vector<glm::vec3> P(ico.v.size());
    const float cr = std::cos(rot), sr = std::sin(rot);
    for (size_t i = 0; i < ico.v.size(); ++i) {
        glm::vec3 d = ico.v[i];
        float n = fbm(d * freq + glm::vec3(seed, seed * 1.3f, -seed), 4);
        float r = 1.0f + amp * (n - 0.5f) * 2.0f;
        glm::vec3 q = d * r;
        if (strata > 0.0f) {
            float k = 3.5f;
            float yq = std::floor(q.y * k + 0.5f) / k;
            q.y += (yq - q.y) * strata;
        }
        q *= radii;
        q = glm::vec3(cr * q.x - sr * q.z, q.y, sr * q.x + cr * q.z);
        q += c;
        if (q.y < ground) q.y = ground + (q.y - ground) * 0.1f;
        P[i] = q;
    }
    std::vector<glm::vec3> N(P.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < ico.f.size(); i += 3) {
        glm::vec3 fn = glm::cross(P[ico.f[i + 1]] - P[ico.f[i]], P[ico.f[i + 2]] - P[ico.f[i]]);
        N[ico.f[i]] += fn; N[ico.f[i + 1]] += fn; N[ico.f[i + 2]] += fn;
    }
    const float ymin = c.y - radii.y, ymax = c.y + radii.y;
    for (size_t i = 0; i < P.size(); ++i) {
        float hy = glm::clamp((P[i].y - ymin) / std::max(ymax - ymin, 1e-3f), 0.0f, 1.0f);
        float w = sway_base.x + (sway_top - sway_base.x) * hy;
        push_v(m, { P[i], glm::normalize(N[i] + glm::vec3(1e-6f)), w, sway_base.y });
    }
    for (uint32_t idx : ico.f) m.idx.push_back(base + idx);
}

// Tapered cylinder a->b (trunk / branch), radius r0 -> r1.
void add_limb(LevelMesh& m, glm::vec3 a, glm::vec3 b, float r0, float r1, int segs,
              float w0, float w1, float phase) {
    glm::vec3 ax = glm::normalize(b - a);
    glm::vec3 ref = std::abs(ax.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 u = glm::normalize(glm::cross(ax, ref)), v = glm::cross(ax, u);
    const uint32_t base = uint32_t(m.verts.size());
    for (int i = 0; i <= segs; ++i) {
        float t = 2.0f * kPi * float(i % segs) / float(segs);
        glm::vec3 n = u * std::cos(t) + v * std::sin(t);
        push_v(m, { a + n * r0, n, w0, phase });
        push_v(m, { b + n * r1, n, w1, phase });
    }
    for (int i = 0; i < segs; ++i) {
        uint32_t k = base + uint32_t(i) * 2;
        m.idx.insert(m.idx.end(), { k, k + 2, k + 1, k + 1, k + 2, k + 3 });
    }
}

// Jagged cone (pine tier): ring of `segs` points with alternating radius,
// apex above, plus a slightly concave underside.
void add_cone(LevelMesh& m, glm::vec3 c, float r, float h, int segs, float seed,
              float w_rim, float w_apex, float phase) {
    const uint32_t base = uint32_t(m.verts.size());
    glm::vec3 apex = c + glm::vec3(0, h, 0);
    std::vector<glm::vec3> rim(segs);
    for (int i = 0; i < segs; ++i) {
        float t = 2.0f * kPi * float(i) / float(segs) + seed;
        float rr = r * ((i & 1) ? 0.78f : 1.0f) * (0.9f + 0.2f * h3(glm::vec3(seed, float(i), 3.0f)));
        float droop = -h * 0.12f * ((i & 1) ? 0.4f : 1.0f);
        rim[i] = c + glm::vec3(std::cos(t) * rr, droop, std::sin(t) * rr);
    }
    // Outer faces: per-face vertices so tiers read as faceted needles.
    for (int i = 0; i < segs; ++i) {
        glm::vec3 a = rim[i], b = rim[(i + 1) % segs];
        glm::vec3 n = glm::normalize(glm::cross(apex - a, b - a));
        if (n.y < 0.0f) n = -n;
        uint32_t k = uint32_t(m.verts.size());
        push_v(m, { a, n, w_rim, phase });
        push_v(m, { b, n, w_rim, phase });
        push_v(m, { apex, n, w_apex, phase });
        // (a, apex, b) = counter-clockwise seen from outside (front face).
        m.idx.insert(m.idx.end(), { k, k + 2, k + 1 });
    }
    // Underside.
    glm::vec3 under = c + glm::vec3(0, h * 0.05f, 0);
    uint32_t ku = uint32_t(m.verts.size());
    push_v(m, { under, glm::vec3(0, -1, 0), w_rim, phase });
    for (int i = 0; i < segs; ++i) push_v(m, { rim[i], glm::vec3(0, -1, 0), w_rim, phase });
    for (int i = 0; i < segs; ++i)
        m.idx.insert(m.idx.end(), { ku, ku + 1 + uint32_t(i), ku + 1 + uint32_t((i + 1) % segs) });
    (void)base;
}

// Close-range pine tier: two rings (drooping outer rim + raised inner
// ring) so a tier reads as a layer of sagging branches instead of a flat
// cone; every outer point gets its own needle-tip jitter.
void add_cone_layered(LevelMesh& m, glm::vec3 c, float r, float h, int segs, float seed,
                      float w_rim, float w_apex, float phase) {
    glm::vec3 apex = c + glm::vec3(0, h, 0);
    std::vector<glm::vec3> rim(segs), mid(segs);
    for (int i = 0; i < segs; ++i) {
        float t = 2.0f * kPi * float(i) / float(segs) + seed;
        float j = h3(glm::vec3(seed, float(i), 3.0f));
        float rr = r * ((i & 1) ? 0.80f : 1.0f) * (0.86f + 0.28f * j);
        float droop = -h * (0.16f + 0.10f * j) * ((i & 1) ? 0.5f : 1.0f);
        rim[i] = c + glm::vec3(std::cos(t) * rr, droop, std::sin(t) * rr);
        float tm = t + kPi / float(segs);
        float rm = r * 0.52f * (0.9f + 0.2f * h3(glm::vec3(seed, float(i), 7.0f)));
        mid[i] = c + glm::vec3(std::cos(tm) * rm, h * 0.36f, std::sin(tm) * rm);
    }
    auto tri = [&](glm::vec3 a, glm::vec3 b, glm::vec3 d, float wa, float wb, float wd) {
        glm::vec3 n = glm::normalize(glm::cross(d - a, b - a));
        if (n.y < -0.2f) n = -n;
        uint32_t k = uint32_t(m.verts.size());
        push_v(m, { a, n, wa, phase });
        push_v(m, { b, n, wb, phase });
        push_v(m, { d, n, wd, phase });
        m.idx.insert(m.idx.end(), { k, k + 2, k + 1 });
    };
    const float w_mid = 0.5f * (w_rim + w_apex);
    for (int i = 0; i < segs; ++i) {
        int i1 = (i + 1) % segs;
        // rim band (rim i, rim i+1, mid i) + (mid i, rim i+1, mid i+1)
        tri(rim[i], rim[i1], mid[i], w_rim, w_rim, w_mid);
        tri(mid[i], rim[i1], mid[i1], w_mid, w_rim, w_mid);
        // upper band to the apex
        tri(mid[i], mid[i1], apex, w_mid, w_mid, w_apex);
    }
    glm::vec3 under = c + glm::vec3(0, h * 0.05f, 0);
    uint32_t ku = uint32_t(m.verts.size());
    push_v(m, { under, glm::vec3(0, -1, 0), w_rim, phase });
    for (int i = 0; i < segs; ++i) push_v(m, { rim[i], glm::vec3(0, -1, 0), w_rim, phase });
    for (int i = 0; i < segs; ++i)
        m.idx.insert(m.idx.end(), { ku, ku + 1 + uint32_t(i), ku + 1 + uint32_t((i + 1) % segs) });
}

}  // namespace

WorldPropStats build_world_props(const Heightmap& hm, const Road* road,
                                 const WorldPropParams& P,
                                 std::vector<game::LevelMesh>& out,
                                 std::vector<collision::AABB>& cols) {
    WorldPropStats st{};
    if (hm.heights.empty()) return st;
    std::mt19937 rng(uint32_t(P.seed));
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto u = [&]() { return U(rng); };
    auto h_at = [&](float x, float z) { return hm.sample_world(x, z); };
    auto ny_at = [&](float x, float z) {
        float e = 1.5f;
        float dx = h_at(x + e, z) - h_at(x - e, z), dz = h_at(x, z + e) - h_at(x, z - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    auto road_d = [&](float x, float z) {
        if (!road || road->empty()) return 1e9f;
        if (x < road->bb_min.x || x > road->bb_max.x || z < road->bb_min.y || z > road->bb_max.y) return 1e9f;
        return road->distance(x, z);
    };
    const float road_clear = road ? road->half_width + road->shoulder : 0.0f;

    const Ico sphere_lo = make_ico(1), sphere_hi = make_ico(2);

    // Chunked outputs: material -> grid cell -> mesh.
    enum Mat { kRock = 0, kBark, kNeedle, kLeaf, kBush, kMatCount };
    const float kCell = 200.0f;
    std::map<std::tuple<int, int, int>, LevelMesh> chunks;
    auto chunk = [&](Mat mat, glm::vec3 p) -> LevelMesh& {
        int cx = int(std::floor(p.x / kCell)), cz = int(std::floor(p.z / kCell));
        auto key = std::make_tuple(int(mat), cx, cz);
        auto it = chunks.find(key);
        if (it != chunks.end()) return it->second;
        LevelMesh m;
        m.triplanar = true;
        switch (mat) {
        case kRock:   m.tex_albedo = 7; m.tex_normal = 7; m.uv_scale = 0.22f;
                      m.tint = glm::vec3(1.15f, 1.10f, 1.04f); m.fallback = glm::vec3(0.45f); break;
        case kBark:   m.tex_albedo = 7; m.tex_normal = 7; m.uv_scale = 1.4f; m.sway = true;
                      m.tint = glm::vec3(0.85f, 0.66f, 0.50f); m.fallback = glm::vec3(0.30f, 0.20f, 0.12f); break;
        case kNeedle: m.tex_albedo = -1; m.tex_normal = -1; m.uv_scale = 0.9f; m.sway = true;
                      m.tint = glm::vec3(0.075f, 0.16f, 0.08f); m.fallback = m.tint; break;
        case kLeaf:   m.tex_albedo = -1; m.tex_normal = -1; m.uv_scale = 0.7f; m.sway = true;
                      m.tint = glm::vec3(0.16f, 0.28f, 0.07f); m.fallback = m.tint; break;
        case kBush:   m.tex_albedo = -1; m.tex_normal = -1; m.uv_scale = 1.1f; m.sway = true;
                      m.tint = glm::vec3(0.14f, 0.24f, 0.08f); m.fallback = m.tint; break;
        default: break;
        }
        return chunks.emplace(key, std::move(m)).first->second;
    };

    // Occupancy grid (2 m) so props don't intersect each other.
    std::unordered_set<int64_t> occ;
    auto key2 = [](int x, int z) { return (int64_t(x) << 32) ^ int64_t(uint32_t(z)); };
    auto free_at = [&](float x, float z, float r) {
        int r2 = int(std::ceil(r / 2.0f));
        int cx = int(std::floor(x / 2.0f)), cz = int(std::floor(z / 2.0f));
        for (int dz = -r2; dz <= r2; ++dz)
            for (int dx = -r2; dx <= r2; ++dx)
                if (occ.count(key2(cx + dx, cz + dz))) return false;
        return true;
    };
    auto mark = [&](float x, float z, float r) {
        int r2 = int(std::ceil(r / 2.0f));
        int cx = int(std::floor(x / 2.0f)), cz = int(std::floor(z / 2.0f));
        for (int dz = -r2; dz <= r2; ++dz)
            for (int dx = -r2; dx <= r2; ++dx) occ.insert(key2(cx + dx, cz + dz));
    };
    auto site_ok = [&](float x, float z, float clear_extra) {
        if (std::sqrt(x * x + z * z) < P.castle_clear) return false;
        if (road_d(x, z) < road_clear + clear_extra) return false;
        return true;
    };

    // ---------------- Rock formations ----------------
    for (int a = 0; a < 6000 && st.formations < 120; ++a) {
        float x = (u() * 2.0f - 1.0f) * P.half_extent, z = (u() * 2.0f - 1.0f) * P.half_extent;
        if (!site_ok(x, z, 6.0f)) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 1.0f) continue;
        float slope = 1.0f - ny_at(x, z);
        float want = std::clamp(slope * 2.2f, 0.0f, 1.0f);
        want = std::max(want, std::clamp((h - P.rock_band_lo + 15.0f) / 30.0f, 0.0f, 1.0f));
        want *= 0.35f + 0.65f * std::clamp(fbm(glm::vec3(x * 0.01f, 3.0f, z * 0.01f), 2) * 2.0f - 0.4f, 0.0f, 1.0f);
        if (u() > want * 0.5f) continue;
        float size = 2.5f + 4.5f * u() * u();
        if (!free_at(x, z, size * 1.6f)) continue;
        mark(x, z, size * 1.6f);
        int n = 3 + int(u() * 4.0f);
        float strata = (h > P.rock_band_lo - 10.0f || u() < 0.5f) ? 0.55f : 0.0f;
        float rot0 = u() * 2.0f * kPi;
        for (int k = 0; k < n; ++k) {
            float ang = rot0 + float(k) * 2.4f + u() * 0.6f;
            float d = (k == 0) ? 0.0f : size * (0.6f + 0.6f * u());
            float bx = x + std::cos(ang) * d, bz = z + std::sin(ang) * d;
            float s = size * ((k == 0) ? 1.0f : 0.45f + 0.45f * u());
            glm::vec3 rad(s * (0.8f + 0.5f * u()), s * (0.55f + 0.5f * u()), s * (0.8f + 0.5f * u()));
            float gy = h_at(bx, bz);
            glm::vec3 c(bx, gy + rad.y * 0.35f, bz);
            add_blob(chunk(kRock, c), sphere_hi, c, rad, u() * 6.28f, u() * 50.0f, 0.32f, 1.6f,
                     strata, gy - 0.3f, glm::vec2(0.0f), 0.0f);
            glm::vec3 he = rad * 0.72f;
            cols.push_back({ glm::vec3(c.x - he.x, gy - 1.0f, c.z - he.z),
                             glm::vec3(c.x + he.x, c.y + he.y, c.z + he.z) });
        }
        ++st.formations;
    }
    // Lone erratic boulders in the meadows.
    for (int a = 0; a < 4000 && st.boulders < 160; ++a) {
        float x = (u() * 2.0f - 1.0f) * P.half_extent, z = (u() * 2.0f - 1.0f) * P.half_extent;
        if (!site_ok(x, z, 3.0f)) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 0.8f || h > P.tree_line + 30.0f) continue;
        if (ny_at(x, z) < 0.8f) continue;
        float s = 0.9f + 2.0f * u() * u();
        if (!free_at(x, z, s * 1.2f)) continue;
        mark(x, z, s * 1.2f);
        glm::vec3 rad(s * (0.9f + 0.4f * u()), s * (0.6f + 0.3f * u()), s * (0.9f + 0.4f * u()));
        glm::vec3 c(x, h + rad.y * 0.3f, z);
        add_blob(chunk(kRock, c), sphere_hi, c, rad, u() * 6.28f, u() * 50.0f, 0.28f, 1.4f,
                 0.0f, h - 0.2f, glm::vec2(0.0f), 0.0f);
        glm::vec3 he = rad * 0.7f;
        cols.push_back({ glm::vec3(c.x - he.x, h - 1.0f, c.z - he.z),
                         glm::vec3(c.x + he.x, c.y + he.y, c.z + he.z) });
        ++st.boulders;
    }

    // Trees + bushes are GPU-instanced now (place_vegetation).
    for (auto& kv : chunks) out.push_back(std::move(kv.second));
    return st;
}


// =====================================================================
// GPU-instanced vegetation: archetypes with a 4-level LOD chain + placement
// =====================================================================
namespace {

struct PineParams { float H; int tiers; float r0; float seed; float lean; };
struct OakParams  { float H; int branches; float seed; float spread; glm::vec3 top_off;
                    glm::vec2 crown = glm::vec2(1.0f);   // crown scale (xz, y): wide oak / tall narrow / round
                    int clumps = 0; };                  // extra small crown clumps (LOD0 lumpiness)
struct BushParams { int blobs; float seed; float sx = 1.0f, sy = 1.0f; };


// ---------------------------------------------------------------------
// Close-range trees (LOD 0 / 1). The far LODs keep the cheap cone / blob
// silhouettes above; these give the near trees real structure:
//   * broadleaf - branch skeleton grown by SPACE COLONIZATION (Runions,
//     Lane & Prusinkiewicz 2007): attraction points fill the crown
//     envelope, branches grow toward the points they are closest to and
//     consume them, which yields the forking, space-filling habit of a
//     real crown. Branch radii follow the pipe model (da Vinci's rule,
//     r_parent^n = sum r_child^n); foliage is many small leaf clumps on
//     the twig ends, so the crown has depth and gaps instead of being a
//     few big balls.
//   * conifer  - a curved trunk carrying whorls of drooping needle
//     fronds (longer and sagging low down, short and upswept at the
//     top), the habit of a spruce / fir, instead of stacked cones.
// ---------------------------------------------------------------------

// Two-sided triangle (the vegetation pipelines cull back faces). `bias`
// bends both shading normals so the underside is not pitch black.
void tri2(LevelMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, float wa, float wb, float wc,
          float mat, glm::vec3 bias) {
    glm::vec3 n = glm::cross(b - a, c - a);
    float l = glm::length(n);
    if (l < 1e-9f) return;
    n /= l;
    if (n.y < 0.0f) { std::swap(b, c); std::swap(wb, wc); n = -n; }
    const glm::vec3 nt = glm::normalize(n + bias);
    const glm::vec3 nb = glm::normalize(-n * 0.55f + bias);
    const uint32_t k = uint32_t(m.verts.size());
    push_v(m, { a, nt, wa, mat }); push_v(m, { b, nt, wb, mat }); push_v(m, { c, nt, wc, mat });
    push_v(m, { a, nb, wa, mat }); push_v(m, { b, nb, wb, mat }); push_v(m, { c, nb, wc, mat });
    m.idx.insert(m.idx.end(), { k, k + 1, k + 2, k + 3, k + 5, k + 4 });
}

void build_conifer(LevelMesh& m, const PineParams& p, int lod) {
    std::mt19937 r(uint32_t(p.seed * 1000.0f) + 17u);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    auto axis = [&](float y) { return glm::vec3(p.lean * y / p.H, y, 0.0f); };
    auto trunk_r = [&](float y) { return glm::mix(0.30f, 0.03f, std::pow(glm::clamp(y / p.H, 0.0f, 1.0f), 0.8f)); };
    // Trunk.
    const int tsegs = lod == 0 ? 10 : 6, tparts = lod == 0 ? 4 : 2;
    if (lod == 0)
        add_limb(m, glm::vec3(0, -0.3f, 0), glm::vec3(p.lean * 0.05f, 0.7f, 0), 0.52f, 0.31f, tsegs, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < tparts; ++i) {
        float ya = glm::mix(-0.3f, p.H * 0.97f, float(i) / float(tparts));
        float yb = glm::mix(-0.3f, p.H * 0.97f, float(i + 1) / float(tparts));
        add_limb(m, axis(ya), axis(yb), trunk_r(ya), trunk_r(yb), tsegs,
                 0.35f * glm::max(ya, 0.0f) / p.H, 0.35f * yb / p.H, 0.0f);
    }
    (void)r; (void)U;
    // Branching (L-system, monopodial): the trunk apex T puts out a whorl
    // of boughs Q at every node; each bough carries pairs of side
    // branchlets, and every branchlet ends in a flat, drooping needle
    // spray N. Boughs are long and sagging low down, short and upswept
    // toward the top (pitch from the trunk 108 -> 52 deg).
    //   T(w) : w < W -> [ /(a) &(pitch) $ Q(L_w) ] x n   f(dy) /(r) T(w+1)
    //   Q(l) : F [ +N ] [ -N ] F [ +N ] [ -N ] F N
    {
        using namespace lsys;
        const int whorls = lod == 0 ? glm::clamp(int(p.H * 1.05f), 10, 16) : glm::clamp(int(p.H * 0.62f), 6, 9);
        const int nb = lod == 0 ? 7 : 5;
        const float y_lo = p.H * 0.17f, y_hi = p.H * 0.95f;
        Grammar g;
        g.axiom = { { 'f', y_lo }, { 'T', 0.0f } };
        g.rules['T'] = [&](const Module& mm, Rng& rr, Str& o) {
            const int w = int(mm.a + 0.5f);
            if (w >= whorls) return;
            const float f = (float(w) + 0.5f) / float(whorls);               // 0 bottom .. 1 top
            float L = p.r0 * 1.15f * std::pow(1.0f - f, 0.8f) + 0.22f;
            if (f < 0.14f) L *= 0.7f + 2.1f * f;                              // shaded-out lowest boughs
            const float pitch = glm::mix(108.0f, 52.0f, f * f);
            const int cnt = nb + ((lod == 0 && rr.u() < 0.35f) ? 1 : 0);
            for (int b = 0; b < cnt; ++b) {
                o.push_back({ '[' });
                o.push_back({ '/', 360.0f * (float(b) + 0.5f) / float(cnt) + rr.range(-16.0f, 16.0f) });
                o.push_back({ '&', pitch + rr.range(-7.0f, 7.0f) });
                o.push_back({ '$' });
                o.push_back({ 'Q', L * rr.range(0.78f, 1.22f) });
                o.push_back({ ']' });
            }
            o.push_back({ 'f', (y_hi - y_lo) / float(whorls) * rr.range(0.85f, 1.15f) });
            o.push_back({ '/', rr.range(25.0f, 85.0f) });
            o.push_back({ 'T', float(w + 1) });
        };
        g.rules['Q'] = [&](const Module& mm, Rng& rr, Str& o) {
            const float l = mm.a;
            auto pair = [&](float ang, float len) {
                for (int sgn = 0; sgn < 2; ++sgn) {
                    o.push_back({ '[' });
                    o.push_back({ sgn ? '+' : '-', ang + rr.range(-8.0f, 8.0f) });
                    o.push_back({ 'N', len * rr.range(0.85f, 1.15f), lod == 0 ? 1.0f : 1.35f });
                    o.push_back({ ']' });
                }
            };
            o.push_back({ '!', 0.014f + 0.013f * l });
            if (lod == 0) {
                o.push_back({ 'F', l * 0.34f }); pair(58.0f, l * 0.42f);
                o.push_back({ 'N', l * 0.36f, 0.8f });                     // needles on the bough itself
                o.push_back({ 'F', l * 0.30f }); pair(52.0f, l * 0.32f);
                o.push_back({ 'F', l * 0.20f });
                o.push_back({ 'N', l * 0.34f, 1.0f });
            } else {
                o.push_back({ 'F', l * 0.50f }); pair(56.0f, l * 0.46f);
                o.push_back({ 'F', l * 0.22f });
                o.push_back({ 'N', l * 0.44f, 1.35f });
            }
        };
        TurtleParams tp;
        tp.taper = 0.72f;
        tp.bend = 0.13f;                                                      // boughs sag under their own weight
        tp.organs = "N";
        const Shape sh = interpret(g.derive(whorls + 2, uint32_t(p.seed * 1000.0f) + 17u), tp);
        auto lean = [&](glm::vec3 q) { return q + glm::vec3(p.lean * q.y / p.H, 0.0f, 0.0f); };
        auto wt = [&](glm::vec3 q) {
            return 0.12f + 0.45f * glm::clamp(q.y / p.H, 0.0f, 1.0f) +
                   0.4f * glm::clamp(glm::length(glm::vec2(q.x, q.z)) / (p.r0 * 1.3f), 0.0f, 1.0f);
        };
        if (lod == 0)
            for (const auto& s : sh.segs)
                add_limb(m, lean(s.a), lean(s.b), s.r0, s.r1, 3, wt(s.a), wt(s.b), 1.0f);   // needle-clad twigs
        for (const auto& o : sh.organs) {
            const float len = glm::max(o.a, 0.36f);                          // even the top sprays stay visible
            glm::vec3 side = glm::cross(up, o.H);
            side = glm::dot(side, side) > 1e-5f ? glm::normalize(side) : o.L;
            const float w = len * 0.40f * o.b;
            const glm::vec3 p0 = lean(o.pos);
            const glm::vec3 p1 = lean(o.pos + o.H * (len * 0.5f) - up * (0.05f * len));
            const glm::vec3 p2 = lean(o.pos + o.H * len - up * (0.22f * len));
            const glm::vec3 hang = up * (0.38f * w);                          // spray sides hang below the rib
            const glm::vec3 l1 = p1 + side * w - hang, r1 = p1 - side * w - hang;
            const float w0 = wt(o.pos), w1 = glm::min(1.0f, w0 + 0.12f), w2 = glm::min(1.0f, w0 + 0.25f);
            glm::vec3 out(o.pos.x, 0.0f, o.pos.z);
            out = glm::dot(out, out) > 1e-6f ? glm::normalize(out) : glm::vec3(1.0f, 0.0f, 0.0f);
            const glm::vec3 bias = up * 0.35f + out * 0.35f;
            tri2(m, p0, l1, p1, w0, w1, w1, 1.0f, bias);
            tri2(m, p0, p1, r1, w0, w1, w1, 1.0f, bias);
            tri2(m, p1, l1, p2, w1, w1, w2, 1.0f, bias);
            tri2(m, p1, p2, r1, w1, w2, w1, 1.0f, bias);
        }
    }
    // Leader tuft at the very top.
    add_cone(m, axis(p.H * 0.93f), 0.30f, p.H * 0.06f + 0.45f, lod == 0 ? 6 : 5, p.seed, 0.6f, 1.0f, 1.0f);
}

struct SkNode { glm::vec3 p; int parent; float r; int kids; };

// Space colonization inside an ellipsoidal crown envelope. `nodes` holds
// the trunk on entry (root first); grown branch nodes are appended.
void colonize(std::vector<SkNode>& nodes, std::mt19937& rng, glm::vec3 centre, glm::vec3 radii,
              int n_attr, float step) {
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<glm::vec3> attr;
    attr.reserve(size_t(n_attr));
    while (int(attr.size()) < n_attr) {
        glm::vec3 d(U(rng) * 2.0f - 1.0f, U(rng) * 2.0f - 1.0f, U(rng) * 2.0f - 1.0f);
        float l2 = glm::dot(d, d);
        if (l2 > 1.0f || l2 < 0.04f) continue;
        if (d.y < -0.55f) continue;                           // flat-bottomed crown
        // Push points toward the shell: foliage lives where the light is.
        d *= std::pow(l2, -0.18f);
        if (glm::dot(d, d) > 1.0f) d = glm::normalize(d);
        attr.push_back(centre + d * radii);
    }
    const float infl = glm::max(radii.x, radii.y) * 1.35f, kill = step * 0.95f;
    std::vector<glm::vec3> acc;
    std::vector<int> cnt;
    for (int iter = 0; iter < 90 && !attr.empty() && nodes.size() < 520; ++iter) {
        acc.assign(nodes.size(), glm::vec3(0.0f));
        cnt.assign(nodes.size(), 0);
        bool any = false;
        for (const auto& a : attr) {
            int best = -1; float bd = infl * infl;
            for (size_t j = 0; j < nodes.size(); ++j) {
                glm::vec3 d = a - nodes[j].p;
                float d2 = glm::dot(d, d);
                if (d2 < bd) { bd = d2; best = int(j); }
            }
            if (best < 0) continue;
            acc[size_t(best)] += glm::normalize(a - nodes[size_t(best)].p);
            ++cnt[size_t(best)];
            any = true;
        }
        if (!any) break;
        const size_t n0 = nodes.size();
        for (size_t j = 0; j < n0; ++j) {
            if (cnt[j] == 0) continue;
            glm::vec3 dir = acc[j] / float(cnt[j]);
            dir += glm::vec3(0.0f, 0.10f, 0.0f);              // phototropism
            dir += glm::vec3(U(rng) - 0.5f, U(rng) - 0.5f, U(rng) - 0.5f) * 0.25f;
            float dl = glm::length(dir);
            if (dl < 1e-4f) continue;
            glm::vec3 np = nodes[j].p + dir / dl * step;
            // Do not grow a duplicate twig from the same node.
            bool dup = false;
            for (size_t q = n0; q < nodes.size(); ++q)
                if (nodes[q].parent == int(j) && glm::distance(nodes[q].p, np) < step * 0.5f) dup = true;
            for (size_t q = 0; q < n0 && !dup; ++q)
                if (nodes[q].parent == int(j) && glm::distance(nodes[q].p, np) < step * 0.35f) dup = true;
            if (dup) continue;
            nodes.push_back({ np, int(j), 0.0f, 0 });
            ++nodes[j].kids;
        }
        if (nodes.size() == n0) break;
        attr.erase(std::remove_if(attr.begin(), attr.end(), [&](const glm::vec3& a) {
            for (size_t q = n0; q < nodes.size(); ++q)
                if (glm::distance(a, nodes[q].p) < kill) return true;
            return false;
        }), attr.end());
    }
}

void build_broadleaf(LevelMesh& m, const OakParams& p, int lod, const Ico& lo, const Ico& tiny) {
    (void)tiny;
    std::mt19937 r(uint32_t(p.seed * 1000.0f) + 91u);         // same skeleton for every LOD
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const glm::vec3 cs(p.crown.x, p.crown.y, p.crown.x);
    const glm::vec3 top = p.top_off + glm::vec3(0.0f, p.H * 0.62f, 0.0f);
    const glm::vec3 centre = top + glm::vec3(0.0f, 2.4f * cs.y + 0.3f, 0.0f);
    const glm::vec3 radii = glm::vec3(3.0f + p.spread * 0.6f, 2.8f, 3.0f + p.spread * 0.6f) * cs * 0.95f;

    // Trunk nodes: slight S-bend from the root to the crown base.
    std::vector<SkNode> nodes;
    const int tn = 5;
    const float bend = (U(r) - 0.5f) * 0.9f, bend_a = U(r) * 6.2831853f;
    for (int i = 0; i <= tn; ++i) {
        float t = float(i) / float(tn);
        glm::vec3 q = glm::mix(glm::vec3(0.0f, -0.3f, 0.0f), top, t);
        float sb = std::sin(t * 3.14159265f) * bend * 0.35f;
        q += glm::vec3(std::cos(bend_a), 0.0f, std::sin(bend_a)) * sb;
        nodes.push_back({ q, i - 1, 0.0f, i < tn ? 1 : 0 });
    }
    const float step = 0.52f;
    {
        // Both detailed LODs share one skeleton: grow it once per tree seed.
        struct Grown { std::vector<SkNode> nodes; std::mt19937 rng; };
        static std::mutex mu;
        static std::map<uint32_t, Grown> cache;
        const uint32_t key = uint32_t(p.seed * 1000.0f) ^ (uint32_t(p.H * 64.0f) << 12) ^ (uint32_t(p.spread * 64.0f) << 22);
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it == cache.end()) {
            colonize(nodes, r, centre, radii, 420, step);
            cache.emplace(key, Grown{ nodes, r });
        } else {
            nodes = it->second.nodes;
            r = it->second.rng;
        }
    }

    // Pipe-model radii, tips first (children always follow their parent).
    const float n_exp = 2.35f, r_tip = 0.022f;
    std::vector<float> acc(nodes.size(), 0.0f);
    for (size_t i = nodes.size(); i-- > 0;) {
        float ri = nodes[i].kids == 0 ? r_tip : std::pow(acc[i], 1.0f / n_exp);
        nodes[i].r = ri;
        if (nodes[i].parent >= 0) acc[size_t(nodes[i].parent)] += std::pow(ri, n_exp);
    }
    // Scale so the trunk has a believable girth; twigs keep a visible minimum.
    const float want_top = 0.25f + 0.02f * p.H;
    const float sc = want_top / glm::max(nodes[size_t(tn)].r, 1e-3f);
    for (auto& nd : nodes) nd.r = glm::max(nd.r * sc, 0.016f);
    for (int i = 0; i <= tn; ++i)      // taper + root flare down the trunk
        nodes[size_t(i)].r = glm::max(nodes[size_t(i)].r, want_top * (1.0f + 0.75f * (1.0f - float(i) / float(tn))));

    const float y_base = top.y, y_span = glm::max(centre.y + radii.y - y_base, 1.0f);
    auto sway = [&](glm::vec3 q) { return glm::clamp(0.25f + 0.75f * (q.y - y_base) / y_span, 0.0f, 1.0f) * (q.y > y_base ? 1.0f : 0.0f) + (q.y <= y_base ? 0.25f * glm::max(q.y, 0.0f) / glm::max(y_base, 1.0f) : 0.0f); };
    if (lod == 0)
        add_limb(m, glm::vec3(0.0f, -0.3f, 0.0f), glm::vec3(0.0f, 0.8f, 0.0f), nodes[0].r * 1.55f, nodes[0].r * 1.02f, 12, 0.0f, 0.0f, 0.0f);
    const float r_min = lod == 0 ? 0.0f : 0.045f;
    for (size_t i = 1; i < nodes.size(); ++i) {
        const SkNode& nd = nodes[i];
        const SkNode& pa = nodes[size_t(nd.parent)];
        if (nd.r < r_min) continue;
        const float r0 = glm::min(pa.r, nd.r * 1.3f);
        const int segs = nd.r > 0.14f ? (lod == 0 ? 10 : 6) : nd.r > 0.05f ? (lod == 0 ? 6 : 4) : 3;
        add_limb(m, pa.p, nd.p, r0, nd.r, segs, sway(pa.p), sway(nd.p), 0.0f);
    }

    // Foliage: leaf clumps on the twig ends (and along thin twigs).
    std::vector<glm::vec3> spots;
    for (size_t i = size_t(tn) + 1; i < nodes.size(); ++i)
        if (nodes[i].kids == 0 || (nodes[i].r < 0.03f && (i % 3) == 0)) spots.push_back(nodes[i].p);
    const size_t want = lod == 0 ? 90 : 44;
    const size_t stride = std::max<size_t>(1, (spots.size() + want - 1) / want);
    // Fewer clumps -> each must cover more of the envelope.
    const float cover = std::sqrt(float(stride)) * (lod == 0 ? 1.0f : 1.25f);
    // Leafy twigs (L-system), the same way the small plants are grown: a
    // short twig with alternate leaves sticks out of every clump, so the
    // outline of the crown is made of real leaf blades. The clumps behind
    // them stay as the leaf mass.
    //   A(i) : i < n -> F [ +(a) L ] /(180) A(i+1)     i = n -> L
    lsys::Grammar twig;
    const int tw_n = lod == 0 ? 5 : 3;
    twig.axiom = { { '!', 0.011f }, { 'A', 0.0f, 1.0f } };
    twig.rules['A'] = [tw_n](const lsys::Module& mm, lsys::Rng& rr, lsys::Str& o) {
        const int i = int(mm.a + 0.5f);
        const float sz = mm.b;
        if (i >= tw_n) { o.push_back({ 'L', sz * rr.range(0.85f, 1.1f) }); return; }
        o.push_back({ 'F', sz * rr.range(0.45f, 0.62f) });
        o.push_back({ '[' }); o.push_back({ '+', rr.range(48.0f, 68.0f) });
        o.push_back({ 'L', sz * rr.range(0.8f, 1.15f) }); o.push_back({ ']' });
        o.push_back({ '/', 180.0f + rr.range(-25.0f, 25.0f) });
        o.push_back({ '^', rr.range(-8.0f, 8.0f) });
        o.push_back({ 'A', float(i + 1), sz });
    };
    lsys::TurtleParams twp;
    twp.radius = 0.011f; twp.taper = 0.85f;
    twp.bend = 0.5f;                                         // twig tips droop
    twp.organs = "L";
    auto leafy_twig = [&](glm::vec3 from, glm::vec3 dir, float leaf_len, uint32_t seed, float sw) {
        const lsys::Shape sh = lsys::interpret(twig.derive(tw_n + 1, seed), twp);
        // Turtle space (+Y forward) -> world: Y along `dir`.
        const glm::vec3 y = glm::normalize(dir);
        const glm::vec3 x = glm::normalize(glm::cross(std::abs(y.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0), y));
        const glm::vec3 z = glm::cross(x, y);
        auto W = [&](glm::vec3 q) { q *= leaf_len; return from + x * q.x + y * q.y + z * q.z; };
        auto D = [&](glm::vec3 q) { return x * q.x + y * q.y + z * q.z; };
        if (!sh.segs.empty()) {
            const auto& s0 = sh.segs.front(); const auto& s1 = sh.segs.back();
            const glm::vec3 mid = W(sh.segs[sh.segs.size() / 2].a);
            add_limb(m, W(s0.a), mid, 0.012f, 0.008f, 3, sw, sw, 0.0f);
            add_limb(m, mid, W(s1.b), 0.008f, 0.004f, 3, sw, sw, 0.0f);
        }
        const glm::vec3 bias = glm::normalize(dir) * 0.3f + glm::vec3(0.0f, 0.35f, 0.0f);
        for (const auto& o : sh.organs) {
            const float len = o.a * leaf_len;
            const glm::vec3 h = D(o.H), u = D(o.U);
            const glm::vec3 base = W(o.pos);
            const glm::vec3 droop(0.0f, -0.16f * len, 0.0f);
            const glm::vec3 tip = base + h * len + droop;
            const glm::vec3 midp = base + h * (len * 0.45f) + droop * 0.3f;
            const glm::vec3 wv = glm::normalize(glm::cross(u, h)) * (len * 0.27f);
            const glm::vec3 fold = u * (len * 0.06f);
            tri2(m, base, midp + wv + fold, tip, sw, sw, sw, 6.0f, bias);
            tri2(m, base, tip, midp - wv + fold, sw, sw, sw, 6.0f, bias);
        }
    };
    int ci = 0;
    for (size_t i = 0; i < spots.size(); i += stride, ++ci) {
        float s = (0.50f + 0.34f * U(r)) * cover;
        glm::vec3 c = spots[i] + glm::vec3(0.0f, 0.12f * s, 0.0f);
        {
            const glm::vec3 out = glm::normalize(c - centre + glm::vec3(1e-3f));
            const int twigs = lod == 0 ? 2 : 1;
            for (int t = 0; t < twigs; ++t) {
                glm::vec3 d = glm::normalize(out + glm::vec3(U(r) - 0.5f, U(r) - 0.35f, U(r) - 0.5f) * 1.1f);
                leafy_twig(c + d * (s * 0.55f), d, (lod == 0 ? 0.30f : 0.46f) * (0.85f + 0.3f * U(r)),
                           uint32_t(p.seed * 977.0f) + uint32_t(ci * 31 + t * 7), glm::min(1.0f, sway(c) + 0.2f));
            }
        }
        // Flattened, tilted outward from the crown centre like a leafy spray.
        add_blob(m, lo, c, glm::vec3(s, s * (0.72f + 0.2f * U(r)), s), U(r) * 6.28f,
                 p.seed * 3.0f + float(ci) * 1.37f, lod == 0 ? 0.34f : 0.22f, 2.6f, 0.0f, -1e9f,
                 glm::vec2(sway(c) * 0.9f, 2.0f), glm::min(1.0f, sway(c) + 0.15f));
    }
    if (lod == 1) {
        // Mid range: an inner volume closes the see-through gaps cheaply.
        add_blob(m, lo, centre, radii * 0.62f, 0.0f, p.seed * 7.0f, 0.2f, 2.2f, 0.0f, -1e9f,
                 glm::vec2(0.5f, 2.0f), 0.9f);
    }
}

// LOD tables: LOD0 full detail .. LOD3 ~10-30 tris (far silhouette).
void build_pine(LevelMesh& m, const PineParams& p, int lod, const Ico& lo, const Ico& hi) {
    (void)lo; (void)hi;
    if (lod <= 1) { build_conifer(m, p, lod); return; }
    const int   trunk_segs[4] = { 12, 6, 4, 3 };
    const int   cone_segs[4]  = { 18, 10, 6, 5 };
    // LOD0: ~1.5x more, thinner tiers (layered branches); far LODs unchanged.
    const int   tiers         = std::max(1, (lod == 0) ? p.tiers + p.tiers / 2
                                                     : (lod == 1) ? (p.tiers + 1) / 2 + 1
                                                     : (lod == 2) ? 2 : 1);
    glm::vec3 b(0.0f);
    if (lod == 0) {   // root flare: short wide base under the trunk
        add_limb(m, b - glm::vec3(0, 0.3f, 0), b + glm::vec3(p.lean * 0.05f, 0.7f, 0), 0.55f, 0.33f,
                 trunk_segs[lod], 0.0f, 0.0f, 0.0f);
    }
    add_limb(m, b - glm::vec3(0, 0.3f, 0), b + glm::vec3(p.lean, p.H * 0.9f, 0), 0.32f, 0.06f,
             trunk_segs[lod], 0.0f, 0.35f, 0.0f);
    const float y0 = p.H * 0.22f;
    for (int t = 0; t < tiers; ++t) {
        float f01 = tiers > 1 ? float(t) / float(tiers - 1) : 0.0f;
        float r = (tiers == 1) ? p.r0 * 0.95f : (p.r0 - (p.r0 - 0.6f) * f01);
        float th = (tiers == 1) ? (p.H - y0 + 0.5f) : (p.H - y0) / float(tiers) * 1.9f;
        float cy = (tiers == 1) ? y0 : y0 + (p.H - y0 - th * 0.6f) * f01;
        if (lod == 0)
            add_cone_layered(m, glm::vec3(p.lean * cy / p.H, cy, 0.0f), r, th * 0.8f, cone_segs[lod],
                             p.seed + float(t) * 1.7f, 0.15f + 0.6f * f01, 0.3f + 0.7f * f01, 1.0f);
        else
            add_cone(m, glm::vec3(p.lean * cy / p.H, cy, 0.0f), r, th, cone_segs[lod],
                     p.seed + float(t) * 1.7f, 0.15f + 0.6f * f01, 0.3f + 0.7f * f01, 1.0f);
    }
}

void build_oak(LevelMesh& m, const OakParams& p, int lod, const Ico& lo, const Ico& hi, const Ico& tiny,
               const Ico& ultra) {
    if (lod <= 1) { build_broadleaf(m, p, lod, lo, tiny); return; }
    glm::vec3 b(0.0f);
    glm::vec3 top = b + p.top_off + glm::vec3(0, p.H * 0.62f, 0);
    const int limb_segs[4] = { 12, 6, 4, 3 };
    const glm::vec3 cs(p.crown.x, p.crown.y, p.crown.x);   // crown shape scale
    if (lod == 0)   // root flare
        add_limb(m, b - glm::vec3(0, 0.3f, 0), b + glm::vec3(0, 0.8f, 0), 0.70f, 0.44f, limb_segs[lod],
                 0.0f, 0.0f, 0.0f);
    add_limb(m, b - glm::vec3(0, 0.3f, 0), top, 0.42f, 0.24f, limb_segs[lod], 0.0f, 0.25f, 0.0f);
    std::vector<glm::vec3> tips;
    std::mt19937 r(uint32_t(p.seed * 1000.0f));
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    for (int k = 0; k < p.branches; ++k) {
        float ang = p.seed + float(k) * 2.0f * kPi / float(p.branches) + U(r) * 0.5f;
        glm::vec3 tip = top + glm::vec3(std::cos(ang) * p.spread * cs.x, (1.2f + U(r) * 1.4f) * cs.y,
                                        std::sin(ang) * p.spread * cs.x);
        if (lod <= 1) add_limb(m, top, tip, 0.2f, 0.07f, limb_segs[lod] - 1, 0.25f, 0.55f, 0.0f);
        if (lod == 0) {   // second-order twigs off each branch
            for (int t = 0; t < 2; ++t) {
                glm::vec3 a = glm::mix(top, tip, 0.45f + 0.25f * float(t));
                float ta = ang + (t == 0 ? 0.9f : -0.9f) + (U(r) - 0.5f) * 0.4f;
                glm::vec3 e = a + glm::vec3(std::cos(ta) * 1.1f * cs.x, 0.8f + U(r) * 0.6f, std::sin(ta) * 1.1f * cs.x);
                add_limb(m, a, e, 0.08f, 0.03f, 5, 0.4f, 0.7f, 0.0f);
                tips.push_back(e);
            }
        }
        tips.push_back(tip);
    }
    const Ico& crown_ico = (lod == 0) ? ultra : (lod == 1) ? hi : tiny;
    glm::vec3 crown = top + glm::vec3(0, 2.4f * cs.y, 0);
    if (lod >= 2) {
        // Far: one merged canopy volume (covers the tips' extent).
        add_blob(m, crown_ico, crown + glm::vec3(0, 0.3f, 0),
                 glm::vec3(3.0f + p.spread * 0.6f, 2.8f, 3.0f + p.spread * 0.6f) * cs, 0.0f,
                 p.seed * 7.0f, 0.18f, 2.0f, 0.0f, -1e9f, glm::vec2(0.5f, 2.0f), 0.95f);
        return;
    }
    add_blob(m, crown_ico, crown, glm::vec3(3.0f, 2.3f, 3.0f) * cs, 0.0f, p.seed * 7.0f, 0.22f, 2.2f,
             0.0f, -1e9f, glm::vec2(0.5f, 2.0f), 0.95f);
    int ti = 0;
    for (const auto& tp : tips) {
        float s = (1.6f + 0.7f * h3(glm::vec3(p.seed, float(ti), 1.0f))) * (lod == 0 && ti % 3 != 2 ? 0.85f : 1.0f);
        const bool twig = lod == 0 && ti % 3 != 2;   // twig clumps: coarser sphere
        add_blob(m, twig ? hi : crown_ico, tp + glm::vec3(0, 0.4f, 0), glm::vec3(s, s * 0.8f, s), 0.0f,
                 p.seed * 3.0f + float(ti), 0.25f, 2.4f, 0.0f, -1e9f, glm::vec2(0.55f, 2.0f), 1.0f);
        ++ti;
    }
    if (lod == 0) {   // lumpy outer clumps around the crown
        for (int k = 0; k < p.clumps; ++k) {
            float a = p.seed * 1.3f + float(k) * 2.39996f;
            float el = (U(r) - 0.3f) * 1.1f;
            glm::vec3 d(std::cos(a) * std::cos(el), std::sin(el), std::sin(a) * std::cos(el));
            glm::vec3 c = crown + d * glm::vec3(3.0f, 2.3f, 3.0f) * cs * 0.85f;
            float s = 0.9f + 0.6f * U(r);
            add_blob(m, hi, c, glm::vec3(s, s * 0.8f, s), 0.0f, p.seed * 11.0f + float(k), 0.28f, 2.6f,
                     0.0f, -1e9f, glm::vec2(0.6f, 2.0f), 1.0f);
        }
    }
}

// Shrub (L-system, sympodial): several woody stems fan out from the base,
// each forking twice while bending up toward the light; leaf clumps sit
// on the twig tips and along the second-order branches, so a bush shows
// stems and gaps instead of being a pile of balls.
//   axiom  : [ /(137.5 k) &(lean) A(l, 0) ] per stem
//   A(l,o) : o < 2 -> F(l) [side leaf clump] [ /() &() A(l r, o+1) ] x n
//            o = 2 -> T (tip leaf clump)
void build_shrub(LevelMesh& m, const BushParams& p, int lod, const Ico& lo) {
    using namespace lsys;
    Grammar g;
    {
        Rng r{ uint32_t(p.seed * 1000.0f) + 5u };
        const int stems = 4 + (p.blobs > 4 ? 1 : 0);
        for (int k = 0; k < stems; ++k) {
            g.axiom.push_back({ '[' });
            g.axiom.push_back({ '/', 137.5f * float(k) + r.range(-20.0f, 20.0f) });
            g.axiom.push_back({ '&', k == 0 ? r.range(4.0f, 14.0f) : r.range(40.0f, 64.0f) });
            g.axiom.push_back({ '!', 0.034f });
            g.axiom.push_back({ 'A', r.range(0.46f, 0.64f), 0.0f });
            g.axiom.push_back({ ']' });
        }
    }
    g.rules['A'] = [](const Module& mm, Rng& r, Str& o) {
        const float l = mm.a;
        const int ord = int(mm.b + 0.5f);
        if (ord >= 2) { o.push_back({ 'F', l * 0.6f }); o.push_back({ 'T', 1.0f }); return; }
        o.push_back({ 'F', l });
        {   // leaves along the branch as well as at its tips (low ones hide the bare base)
            o.push_back({ '[' }); o.push_back({ '/', r.range(1.0f, 360.0f) }); o.push_back({ '&', r.range(50.0f, 80.0f) });
            o.push_back({ 'f', 0.08f }); o.push_back({ 'T', ord == 0 ? 0.9f : 0.75f }); o.push_back({ ']' });
        }
        const int n = (ord == 0 && r.u() < 0.4f) ? 3 : 2;
        const float roll0 = r.range(1.0f, 360.0f);
        for (int c = 0; c < n; ++c) {
            o.push_back({ '[' });
            o.push_back({ '/', roll0 + 360.0f * float(c) / float(n) + r.range(-25.0f, 25.0f) });
            o.push_back({ '&', r.range(22.0f, 40.0f) });
            o.push_back({ '!', 0.034f * std::pow(0.58f, float(ord + 1)) });
            o.push_back({ 'A', l * r.range(0.62f, 0.80f), float(ord + 1) });
            o.push_back({ ']' });
        }
    };
    TurtleParams tp;
    tp.radius = 0.034f; tp.taper = 0.8f;
    tp.tropism = glm::vec3(0.0f, 1.0f, 0.0f);
    tp.bend = 0.45f;                                         // stems curve up toward the light
    tp.organs = "T";
    const Shape sh = interpret(g.derive(4, uint32_t(p.seed * 1000.0f) + 23u), tp);
    const glm::vec3 sc(p.sx, p.sy, p.sx);
    float top = 0.5f;
    for (const auto& o : sh.organs) top = std::max(top, o.pos.y * p.sy + 0.3f);
    auto sway = [&](glm::vec3 q) { return 0.04f + 0.34f * glm::clamp(q.y / top, 0.0f, 1.0f); };
    for (const auto& s : sh.segs) {
        if (lod == 1 && s.r0 < 0.013f) continue;
        const glm::vec3 a = s.a * sc, b = s.b * sc;
        add_limb(m, a, b, s.r0, s.r1, s.r0 > 0.02f ? (lod == 0 ? 6 : 4) : 3, sway(a), sway(b), 0.0f);
    }
    std::mt19937 rr(uint32_t(p.seed * 1000.0f) + 41u);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    int ci = 0;
    for (const auto& o : sh.organs) {
        const float u1 = U(rr), u2 = U(rr), u3 = U(rr);
        ++ci;
        if (lod == 1 && (ci & 1)) continue;                  // mid range: half the clumps, larger
        const float rad = (0.25f + 0.11f * u1) * o.a * (lod == 0 ? 1.0f : 1.5f);
        const glm::vec3 c = (o.pos + o.H * (rad * 0.45f)) * sc;
        add_blob(m, lo, c, glm::vec3(rad * p.sx, rad * (0.74f + 0.2f * u2) * glm::mix(1.0f, p.sy, 0.5f), rad * p.sx),
                 u3 * 6.28f, p.seed * 5.0f + float(ci) * 1.7f, 0.32f, 2.8f, 0.0f, -0.05f,
                 glm::vec2(sway(c), 3.0f), glm::min(0.45f, sway(c) + 0.08f));
    }
}

void build_bush(LevelMesh& m, const BushParams& p, int lod, const Ico& lo, const Ico& tiny,
                const Ico& hi) {
    if (lod <= 1) { build_shrub(m, p, lod, lo); return; }
    const int n = (lod == 0) ? p.blobs : (lod == 1) ? 2 : 1;
    const Ico& ico = (lod == 0) ? hi : (lod == 1) ? lo : tiny;
    for (int k = 0; k < n; ++k) {
        float ang = p.seed + float(k) * 2.39996f;
        float d = (n == 1) ? 0.0f : (k == 0 ? 0.0f : 0.42f + 0.12f * float(k % 2));
        float r = (n == 1) ? 1.0f : (k == 0 ? 0.8f : 0.6f + 0.1f * float((k * 7) % 3));
        glm::vec3 c(std::cos(ang) * d * p.sx, 0.35f + (k == 0 ? 0.1f : 0.0f), std::sin(ang) * d * p.sx);
        add_blob(m, ico, c, glm::vec3(r * p.sx, r * 0.75f * p.sy, r * p.sx), 0.0f, p.seed * 5.0f + float(k),
                 0.3f, 2.6f, 0.0f, -0.15f, glm::vec2(0.05f, 3.0f), 0.35f);
    }
}

}  // namespace

std::vector<VegArchetype> build_veg_archetypes(int seed) {
    std::vector<VegArchetype> out;
    std::mt19937 rng{ uint32_t(seed) };
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const Ico tiny = make_ico(0), lo = make_ico(1), hi = make_ico(2), ultra = make_ico(3);
    auto finish = [](VegArchetype& a) {
        float rmax = 0.0f, ymax = 0.0f;
        for (const auto& v : a.lods[0].verts) {
            rmax = std::max(rmax, glm::length(glm::vec2(v.p.x, v.p.z)));
            ymax = std::max(ymax, v.p.y);
        }
        a.height = ymax;
        a.radius = std::sqrt(rmax * rmax + ymax * ymax * 0.25f);
    };
    // Pines: 7 variants - young / mature / tall-narrow spruce / broad old
    // pine (height, tier count, base radius).
    for (int v = 0; v < 7; ++v) {
        VegArchetype a; a.species = 0;
        const float kind = float(v % 4);
        float H  = (kind == 0 ? 8.0f : kind == 1 ? 11.0f : kind == 2 ? 15.0f : 12.0f) + 3.0f * U(rng);
        int   tiers = 6 + int(U(rng) * 3.0f) + (kind == 2 ? 2 : 0);
        float r0 = (kind == 2 ? 1.9f : kind == 3 ? 3.1f : 2.4f) + 0.6f * U(rng);
        PineParams p{ H, tiers, r0, U(rng) * 6.28f, (U(rng) - 0.5f) * 0.6f };
        for (int l = 0; l < kVegLods; ++l) build_pine(a.lods[l], p, l, lo, hi);
        finish(a); out.push_back(std::move(a));
    }
    // Broadleaf: 7 variants - wide oak, tall narrow (birch / poplar-like),
    // round ash, young tree; crown shape + branch count + outer clumps.
    for (int v = 0; v < 7; ++v) {
        VegArchetype a; a.species = 1;
        const int kind = v % 4;
        const glm::vec2 crown = kind == 0 ? glm::vec2(1.25f, 0.95f)    // wide oak
                              : kind == 1 ? glm::vec2(0.70f, 1.45f)    // tall narrow
                              : kind == 2 ? glm::vec2(1.00f, 1.05f)    // round
                                          : glm::vec2(0.80f, 0.85f);   // young
        OakParams p{ (kind == 1 ? 9.5f : kind == 3 ? 5.5f : 7.0f) + 3.0f * U(rng),
                     4 + int(U(rng) * 3.5f), U(rng) * 6.28f, (1.8f + 0.9f * U(rng)) * crown.x,
                     glm::vec3((U(rng) - 0.5f) * 0.8f, 0.0f, (U(rng) - 0.5f) * 0.8f),
                     crown, 4 + int(U(rng) * 4.0f) };
        for (int l = 0; l < kVegLods; ++l) build_oak(a.lods[l], p, l, lo, hi, tiny, ultra);
        finish(a); out.push_back(std::move(a));
    }
    // Bushes: 5 variants (low wide / tall / round clusters).
    for (int v = 0; v < 5; ++v) {
        VegArchetype a; a.species = 2;
        BushParams p{ 3 + int(U(rng) * 3.0f), U(rng) * 6.28f,
                      v % 3 == 0 ? 1.35f : v % 3 == 1 ? 0.85f : 1.0f,
                      v % 3 == 0 ? 0.75f : v % 3 == 1 ? 1.45f : 1.0f };
        for (int l = 0; l < kVegLods; ++l) build_bush(a.lods[l], p, l, lo, tiny, hi);
        finish(a); out.push_back(std::move(a));
    }
    for (int v = 0; v < 8; ++v) {                      // rocks (species 3)
        VegArchetype a; a.species = 3;
        float sd = U(rng) * 50.0f;
        float strata = (v % 2 == 0) ? 0.55f : 0.0f;
        float amp = 0.26f + 0.1f * U(rng);
        const Ico* ic[kVegLods] = { &hi, &lo, &tiny, &tiny };
        for (int l = 0; l < kVegLods; ++l)
            add_blob(a.lods[l], *ic[l], glm::vec3(0.0f, 0.35f, 0.0f), glm::vec3(1.0f), 0.0f, sd,
                     l >= 2 ? amp * 0.6f : amp, 1.6f, strata, -0.2f, glm::vec2(0.0f, 4.0f), 0.0f);
        finish(a); out.push_back(std::move(a));
    }
    return out;
}

std::vector<VegInstance> place_vegetation(const Heightmap& hm, const Road* road,
                                          const WorldPropParams& P,
                                          const std::vector<VegArchetype>& arch,
                                          std::vector<collision::AABB>& cols,
                                          WorldPropStats& st) {
    std::vector<VegInstance> out;
    if (hm.heights.empty() || arch.empty()) return out;
    std::vector<int> by_species[4];
    for (size_t i = 0; i < arch.size(); ++i) by_species[arch[i].species].push_back(int(i));
    std::mt19937 rng(uint32_t(P.seed) ^ 0x9E3779B9u);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto u = [&]() { return U(rng); };
    auto h_at = [&](float x, float z) { return hm.sample_world(x, z); };
    auto ny_at = [&](float x, float z) {
        float e = 1.5f;
        float dx = h_at(x + e, z) - h_at(x - e, z), dz = h_at(x, z + e) - h_at(x, z - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    auto road_d = [&](float x, float z) {
        if (!road || road->empty()) return 1e9f;
        if (x < road->bb_min.x || x > road->bb_max.x || z < road->bb_min.y || z > road->bb_max.y) return 1e9f;
        return road->distance(x, z);
    };
    auto excluded = [&](float x, float z) {
        for (const auto& k : P.keep_out)
            if ((x - k.x) * (x - k.x) + (z - k.y) * (z - k.y) < k.z * k.z) return true;
        return false;
    };
    const float road_clear = road ? road->half_width + road->shoulder : 0.0f;
    // Forest density 0..1 (1 = closed forest). Large-scale noise with high
    // contrast (thick woods / open meadows / thin edges) plus hand-placed
    // zones: dense forest on the far (north-west) shore of the castle
    // lake, an open ring around the castle, and big treeless meadows.
    auto blob = [](float x0, float z0, float cx, float cz, float r) {
        float dx = (x0 - cx) / r, dz = (z0 - cz) / r;
        return std::exp(-(dx * dx + dz * dz) * 2.0f);
    };
    auto forest_exact = [&](float x, float z) {
        float n = fbm(glm::vec3(x * 0.0042f, 11.0f, z * 0.0042f), 3) * 0.8f +
                  fbm(glm::vec3(x * 0.017f, 4.0f, z * 0.017f), 2) * 0.35f;
        float d = std::clamp((n - 0.50f) * 6.0f, 0.0f, 1.0f);
        d = std::max(d, std::min(1.0f, blob(x, z, -70.0f, -200.0f, 190.0f) * 1.6f));   // far shore
        d = std::max(d, std::min(1.0f, blob(x, z, -230.0f, -60.0f, 120.0f) * 1.3f));
        float clear = 1.0f - std::clamp((std::sqrt(x * x + z * z) - 75.0f) / 40.0f, 0.0f, 1.0f);
        clear = std::max(clear, blob(x, z, 170.0f, 140.0f, 110.0f) * 1.4f);           // SE meadow
        clear = std::max(clear, blob(x, z, 40.0f, 330.0f, 120.0f) * 1.4f);
        return std::clamp(d * (1.0f - std::min(clear, 1.0f)), 0.0f, 1.0f);
    };
    // Forest density is smooth (>= ~60 m features + 100 m+ blobs), so a
    // 2 m grid baked on all cores and sampled bilinearly matches it while
    // taking the 5-octave noise out of the 3.5 M-attempt placement loops
    // (startup: ~0.9 s -> ~0.2 s).
    const float fcell = 2.0f;
    const float fext = P.veg_half_extent + 8.0f;
    const int fn = int(std::ceil(2.0f * fext / fcell)) + 2;
    std::vector<float> fgrid(size_t(fn) * size_t(fn));
    {
        const unsigned nt = (std::max)(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&, t]() {
                for (int j = int(t); j < fn; j += int(nt))
                    for (int i = 0; i < fn; ++i)
                        fgrid[size_t(j) * size_t(fn) + size_t(i)] =
                            forest_exact(-fext + float(i) * fcell, -fext + float(j) * fcell);
            });
        for (auto& t : th) t.join();
    }
    auto forest = [&](float x, float z) {
        float fx = std::clamp((x + fext) / fcell, 0.0f, float(fn - 1) - 0.001f);
        float fz = std::clamp((z + fext) / fcell, 0.0f, float(fn - 1) - 0.001f);
        int ix = int(fx), iz = int(fz);
        float tx = fx - float(ix), tz = fz - float(iz);
        const float* r0 = &fgrid[size_t(iz) * size_t(fn) + size_t(ix)];
        const float* r1 = r0 + fn;
        return (r0[0] * (1.0f - tx) + r0[1] * tx) * (1.0f - tz) + (r1[0] * (1.0f - tx) + r1[1] * tx) * tz;
    };
    // Dry ground: the trunk AND a 3.5 m ring above water + margin, so
    // shore slopes don't put trees (or their crowns) in the lake.
    auto dry = [&](float x, float z, float margin) {
        if (h_at(x, z) < P.water_level + margin) return false;
        for (int k = 0; k < 6; ++k) {
            float a = float(k) * 1.0472f;
            if (h_at(x + std::cos(a) * 3.5f, z + std::sin(a) * 3.5f) < P.water_level + 0.6f) return false;
        }
        return true;
    };
    // 2.5 m occupancy grid keeps trunks apart.
    std::unordered_set<int64_t> occ;
    auto key2 = [](int x, int z) { return (int64_t(x) << 32) ^ int64_t(uint32_t(z)); };
    auto try_mark = [&](float x, float z, float r) {
        int r2 = int(std::ceil(r / 2.5f));
        int cx = int(std::floor(x / 2.5f)), cz = int(std::floor(z / 2.5f));
        for (int dz = -r2; dz <= r2; ++dz)
            for (int dx = -r2; dx <= r2; ++dx)
                if (occ.count(key2(cx + dx, cz + dz))) return false;
        occ.insert(key2(cx, cz));
        return true;
    };
    const float ext = P.veg_half_extent;
    // Rocks: formations (clusters of 3-7 boulders on slopes / rock band)
    // and lone erratics. Instances of the rock archetypes with a per-
    // instance vertical squash (rot_arch_phase.w).
    auto add_rock = [&](float x, float z, float s, float ysq) {
        const auto& pool = by_species[3];
        int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
        float gy = h_at(x, z);
        VegInstance vi{};
        vi.pos_scale = glm::vec4(x, gy - s * ysq * 0.12f, z, s);
        vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), 0.0f, ysq);
        out.push_back(vi);
        float he = s * 0.72f;
        cols.push_back({ glm::vec3(x - he, gy - 1.0f, z - he),
                         glm::vec3(x + he, gy + s * ysq * 1.1f, z + he) });
    };
    for (int a = 0; a < 60000 && st.formations < P.max_formations; ++a) {
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 6.0f) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 1.0f) continue;
        float slope = 1.0f - ny_at(x, z);
        float want = std::clamp(slope * 2.2f, 0.0f, 1.0f);
        want = std::max(want, std::clamp((h - P.rock_band_lo + 15.0f) / 30.0f, 0.0f, 1.0f));
        want *= 0.35f + 0.65f * std::clamp(fbm(glm::vec3(x * 0.01f, 3.0f, z * 0.01f), 2) * 2.0f - 0.4f, 0.0f, 1.0f);
        if (u() > want * 0.5f) continue;
        float size = 2.5f + 4.5f * u() * u();
        if (!try_mark(x, z, size * 1.6f)) continue;
        int n = 3 + int(u() * 4.0f);
        float rot0 = u() * 6.2831853f;
        for (int k = 0; k < n; ++k) {
            float ang = rot0 + float(k) * 2.4f + u() * 0.6f;
            float d = (k == 0) ? 0.0f : size * (0.6f + 0.6f * u());
            float s = size * ((k == 0) ? 1.0f : 0.45f + 0.45f * u());
            add_rock(x + std::cos(ang) * d, z + std::sin(ang) * d, s, 0.6f + 0.5f * u());
        }
        ++st.formations;
    }
    for (int a = 0; a < 40000 && st.boulders < P.max_boulders; ++a) {
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 3.0f) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 0.8f || h > P.tree_line + 30.0f) continue;
        if (ny_at(x, z) < 0.8f) continue;
        float s = 0.9f + 2.0f * u() * u();
        if (!try_mark(x, z, s * 1.2f)) continue;
        add_rock(x, z, s, 0.6f + 0.3f * u());
        ++st.boulders;
    }
    // Trees. max_trees counts trees only (out already holds the rocks).
    const size_t rocks_placed = out.size();
    for (int a = 0; a < P.tree_attempts && int(out.size() - rocks_placed) < P.max_trees; ++a) {
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 4.5f) continue;
        float h = h_at(x, z);
        if (h < std::max(P.grass_band_lo, P.water_level + 2.0f) || h > P.tree_line) continue;
        if (!dry(x, z, 2.0f)) continue;
        if (ny_at(x, z) < 0.72f) continue;
        float f = forest(x, z);
        float dens = f * f + 0.004f;                  // rare lone trees in meadows
        if (u() > dens) continue;
        bool pine = u() < std::clamp((h - 28.0f) / 30.0f, 0.15f, 0.95f);
        float scale = 0.75f + 0.55f * u();
        if (!try_mark(x, z, (pine ? 1.6f : 2.4f) * scale)) continue;
        const auto& pool = by_species[pine ? 0 : 1];
        int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
        VegInstance vi{};
        vi.pos_scale = glm::vec4(x, h - 0.05f, z, scale);
        vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), u() * 6.2831853f, 0.0f);
        out.push_back(vi);
        float tr = 0.38f * scale;
        cols.push_back({ glm::vec3(x - tr, h - 0.5f, z - tr),
                         glm::vec3(x + tr, h + arch[size_t(ai)].height * scale * 0.6f, z + tr) });
        if (pine) ++st.pines; else ++st.broadleaf;
    }
    // Bushes (no collision).
    int bushes = 0;
    for (int a = 0; a < P.bush_attempts && bushes < P.max_bushes; ++a) {
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear - 6.0f) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 1.5f) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 1.0f || h > P.tree_line + 8.0f) continue;
        if (ny_at(x, z) < 0.75f) continue;
        float f = forest(x, z);
        if (!dry(x, z, 1.2f)) continue;
        // Undergrowth inside forests + thickets along the edges.
        float dens = std::clamp(f * 0.45f + (1.0f - std::abs(f - 0.4f) * 3.0f) * 0.5f, 0.0f, 1.0f) * 0.7f + 0.01f;
        if (u() > dens) continue;
        float scale = 0.6f + 0.9f * u();
        const auto& pool = by_species[2];
        int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
        VegInstance vi{};
        vi.pos_scale = glm::vec4(x, h, z, scale);
        vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), u() * 6.2831853f, 0.0f);
        out.push_back(vi);
        ++bushes;
    }
    st.bushes = bushes;
    return out;
}

} // namespace qlike
