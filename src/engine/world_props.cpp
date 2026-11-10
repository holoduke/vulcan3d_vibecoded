#include "engine/world_props.h"

#include "engine/road.h"
#include "engine/terrain.h"

#include <glm/geometric.hpp>
#include <glm/common.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
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

    // ---------------- Trees ----------------
    auto forest = [&](float x, float z) {
        return fbm(glm::vec3(x * 0.0065f, 11.0f, z * 0.0065f), 3) * 0.75f +
               fbm(glm::vec3(x * 0.021f, 4.0f, z * 0.021f), 2) * 0.45f;
    };
    for (int a = 0; a < 40000 && st.pines + st.broadleaf < 900; ++a) {
        float x = (u() * 2.0f - 1.0f) * P.half_extent, z = (u() * 2.0f - 1.0f) * P.half_extent;
        if (!site_ok(x, z, 4.5f)) continue;
        float h = h_at(x, z);
        if (h < std::max(P.grass_band_lo, P.water_level + 1.5f) || h > P.tree_line) continue;
        float ny = ny_at(x, z);
        if (ny < 0.72f) continue;
        float f = forest(x, z);
        float dens = std::clamp((f - 0.50f) * 4.0f, 0.0f, 1.0f) + 0.015f;   // + lone trees
        if (u() > dens) continue;
        // Pines dominate higher up, broadleaf in the low valleys.
        float pine_p = std::clamp((h - 28.0f) / 30.0f, 0.15f, 0.95f);
        bool pine = u() < pine_p;
        float scale = 0.75f + 0.55f * u();
        float rad = pine ? 2.4f * scale : 3.2f * scale;
        if (!free_at(x, z, rad)) continue;
        mark(x, z, rad * 0.8f);
        float phase = u() * 6.28f;
        float gy = h - 0.3f;
        if (pine) {
            float H = (9.0f + 6.0f * u()) * scale;
            glm::vec3 b(x, gy, z);
            add_limb(chunk(kBark, b), b, b + glm::vec3(0, H * 0.9f, 0), 0.32f * scale, 0.06f, 7,
                     0.0f, 0.35f, phase);
            int tiers = 5 + int(u() * 3.0f);
            float y0 = H * 0.22f;
            for (int t = 0; t < tiers; ++t) {
                float f01 = float(t) / float(tiers - 1);
                float r = (2.6f - 2.0f * f01) * scale * (0.9f + 0.2f * u());
                float th = (H - y0) / float(tiers) * 1.9f;
                float cy = gy + y0 + (H - y0 - th * 0.6f) * f01;
                add_cone(chunk(kNeedle, b), glm::vec3(x, cy, z), r, th, 11, u() * 6.28f,
                         0.15f + 0.6f * f01, 0.3f + 0.7f * f01, phase);
            }
            cols.push_back({ glm::vec3(x - 0.35f, gy, z - 0.35f), glm::vec3(x + 0.35f, gy + H, z + 0.35f) });
            ++st.pines;
        } else {
            float H = (6.5f + 3.5f * u()) * scale;
            glm::vec3 b(x, gy, z);
            glm::vec3 top = b + glm::vec3((u() - 0.5f) * 0.8f, H * 0.62f, (u() - 0.5f) * 0.8f);
            add_limb(chunk(kBark, b), b, top, 0.42f * scale, 0.24f * scale, 8, 0.0f, 0.25f, phase);
            int nb = 3 + int(u() * 2.0f);
            std::vector<glm::vec3> tips;
            for (int k = 0; k < nb; ++k) {
                float ang = phase + float(k) * 2.0f * kPi / float(nb) + u() * 0.5f;
                glm::vec3 tip = top + glm::vec3(std::cos(ang) * 2.2f * scale, (1.2f + u() * 1.4f) * scale,
                                                std::sin(ang) * 2.2f * scale);
                add_limb(chunk(kBark, b), top, tip, 0.2f * scale, 0.07f * scale, 6, 0.25f, 0.55f, phase);
                tips.push_back(tip);
            }
            // Canopy: one crown blob + a lump at each branch tip.
            glm::vec3 crown = top + glm::vec3(0, 2.4f * scale, 0);
            add_blob(chunk(kLeaf, b), sphere_lo, crown, glm::vec3(3.0f, 2.3f, 3.0f) * scale, u() * 6.28f,
                     u() * 50.0f, 0.22f, 2.2f, 0.0f, -1e9f, glm::vec2(0.5f, phase), 0.95f);
            for (const auto& tp : tips) {
                float s = (1.6f + 0.7f * u()) * scale;
                add_blob(chunk(kLeaf, b), sphere_lo, tp + glm::vec3(0, 0.4f * scale, 0),
                         glm::vec3(s, s * 0.8f, s), u() * 6.28f, u() * 50.0f, 0.25f, 2.4f, 0.0f,
                         -1e9f, glm::vec2(0.55f, phase), 1.0f);
            }
            cols.push_back({ glm::vec3(x - 0.45f, gy, z - 0.45f), glm::vec3(x + 0.45f, gy + H * 0.6f, z + 0.45f) });
            ++st.broadleaf;
        }
    }

    // ---------------- Bushes ----------------
    for (int a = 0; a < 20000 && st.bushes < 700; ++a) {
        float x = (u() * 2.0f - 1.0f) * P.half_extent, z = (u() * 2.0f - 1.0f) * P.half_extent;
        if (!site_ok(x, z, 1.5f)) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 1.0f || h > P.tree_line + 8.0f) continue;
        if (ny_at(x, z) < 0.75f) continue;
        float f = forest(x, z);
        float dens = std::clamp(1.0f - std::abs(f - 0.50f) * 6.0f, 0.0f, 1.0f) * 0.6f + 0.03f;  // forest edges
        if (u() > dens) continue;
        float s = 0.6f + 0.8f * u();
        if (!free_at(x, z, s)) continue;
        mark(x, z, s * 0.6f);
        float phase = u() * 6.28f;
        int n = 2 + int(u() * 2.0f);
        for (int k = 0; k < n; ++k) {
            float ang = u() * 6.28f, d = s * 0.5f * u();
            glm::vec3 c(x + std::cos(ang) * d, h + s * 0.35f, z + std::sin(ang) * d);
            float r = s * (0.6f + 0.4f * u());
            add_blob(chunk(kBush, c), sphere_lo, c, glm::vec3(r, r * 0.75f, r), u() * 6.28f, u() * 50.0f,
                     0.3f, 2.6f, 0.0f, h - 0.15f, glm::vec2(0.05f, phase), 0.35f);
        }
        ++st.bushes;
    }

    for (auto& kv : chunks) out.push_back(std::move(kv.second));
    return st;
}

} // namespace qlike
