#include "engine/cave.h"
#include "engine/terrain.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

#undef min
#undef max

namespace qlike {
namespace {

float sstep(float a, float b, float x) {
    float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
float h3(glm::vec3 p) {
    p = glm::fract(p * 0.3183099f + glm::vec3(0.1f));
    p *= 17.0f;
    return glm::fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(glm::vec3 x) {
    glm::vec3 i = glm::floor(x), f = glm::fract(x);
    f = f * f * (3.0f - 2.0f * f);
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    return L(L(L(h3(i), h3(i + glm::vec3(1, 0, 0)), f.x), L(h3(i + glm::vec3(0, 1, 0)), h3(i + glm::vec3(1, 1, 0)), f.x), f.y),
             L(L(h3(i + glm::vec3(0, 0, 1)), h3(i + glm::vec3(1, 0, 1)), f.x), L(h3(i + glm::vec3(0, 1, 1)), h3(i + glm::vec3(1, 1, 1)), f.x), f.y), f.z);
}
// Rock relief: two octaves, -1..1.
float rock(glm::vec3 p) { return (vnoise(p * 0.55f) * 0.65f + vnoise(p * 1.7f + 9.1f) * 0.35f) * 2.0f - 1.0f; }

glm::vec2 xz(glm::vec3 p) { return glm::vec2(p.x, p.z); }

// Per-sample frame in the ground plane: tangent t and side s (to the right).
void frame(const Cave& c, size_t i, glm::vec3& t, glm::vec3& s) {
    const size_t a = i > 0 ? i - 1 : 0, b = std::min(i + 1, c.pts.size() - 1);
    glm::vec2 d = xz(c.pts[b]) - xz(c.pts[a]);
    d = glm::length(d) > 1e-5f ? glm::normalize(d) : glm::vec2(1.0f, 0.0f);
    t = glm::vec3(d.x, 0.0f, d.y);
    s = glm::vec3(-d.y, 0.0f, d.x);
}

void tri(game::LevelMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 na, glm::vec3 nb, glm::vec3 nc) {
    glm::vec3 g = glm::cross(b - a, c - a);
    if (glm::dot(g, g) < 1e-12f) return;
    const uint32_t k = uint32_t(m.verts.size());
    for (const auto& v : { std::pair<glm::vec3, glm::vec3>{a, na}, {b, nb}, {c, nc} }) {
        m.verts.push_back({ v.first, glm::normalize(v.second), 0.0f, 0.0f });
        m.bmin = glm::min(m.bmin, v.first);
        m.bmax = glm::max(m.bmax, v.first);
    }
    // Front = cross(b - a, c - a) along the intended normal (engine winding).
    if (glm::dot(g, na + nb + nc) < 0.0f) m.idx.insert(m.idx.end(), { k, k + 2, k + 1 });
    else                                  m.idx.insert(m.idx.end(), { k, k + 1, k + 2 });
}
void quad(game::LevelMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d,
          glm::vec3 na, glm::vec3 nb, glm::vec3 nc, glm::vec3 nd) {
    tri(m, a, b, c, na, nb, nc);
    tri(m, a, c, d, na, nc, nd);
}

}  // namespace

Cave make_cave(const Heightmap& hm, const Road& road, float water_level, float along) {
    Cave c;
    if (road.empty() || hm.heights.empty()) return c;
    // Road point `along` metres from the gate, its direction and right side.
    size_t i0 = 1;
    float acc = 0.0f;
    for (; i0 + 1 < road.pts.size(); ++i0) {
        acc += glm::length(xz(road.pts[i0]) - xz(road.pts[i0 - 1]));
        if (acc >= along) break;
    }
    const glm::vec2 T = glm::normalize(xz(road.pts[std::min(i0 + 2, road.pts.size() - 1)]) - xz(road.pts[i0 - 1]));
    const glm::vec2 R(-T.y, T.x);                       // right-hand side leaving the castle
    const glm::vec2 mouth = xz(road.pts[i0]) + R * (road.half_width + road.shoulder + 5.5f);

    const float L = 34.0f;                              // passage + chamber
    const int   n_in = int(L) + 1, n_tail = 9;
    c.n_inner = n_in;
    std::vector<float> orig;
    for (int i = 0; i < n_in + n_tail; ++i) {
        const float s = float(i);
        // Straight in from the road with a gentle S-bend (you cannot see
        // the chamber from the mouth).
        const float bend = 4.5f * std::sin(std::min(s, L) / L * 3.14159f) * std::sin(std::min(s, L) / L * 1.5708f);
        const glm::vec2 p = mouth + R * s + T * bend;
        const float chamber = sstep(L - 12.0f, L - 5.0f, s);
        c.pts.push_back(glm::vec3(p.x, 0.0f, p.y));
        c.half_w.push_back(2.4f + 2.5f * chamber);
        c.height.push_back(3.3f + 1.5f * chamber);
        c.cover.push_back(1.0f - sstep(L + 2.0f, L + float(n_tail) - 1.0f, s));
        orig.push_back(hm.sample_world(p.x, p.y));
    }
    // Floor: starts on the ground at the mouth, runs nearly level into the
    // hill, never above the ground and never below the lake.
    const size_t n = c.pts.size();
    std::vector<float> fl(n);
    for (size_t i = 0; i < n; ++i) {
        float f = std::min(orig[0] + 0.015f * float(i), orig[i] - 0.25f * sstep(0.0f, 4.0f, float(i)));
        fl[i] = std::max(f, water_level + 0.9f);
    }
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<float> t = fl;
        for (size_t i = 1; i + 1 < n; ++i) t[i] = 0.25f * fl[i - 1] + 0.5f * fl[i] + 0.25f * fl[i + 1];
        fl = t;
    }
    // Once it has gone down it stays down: the passage descends into the
    // hollow and then runs level under the rising ground (a real roof).
    for (size_t i = 1; i < n; ++i) fl[i] = std::min(fl[i], fl[i - 1]);
    for (size_t i = 0; i < n; ++i) c.pts[i].y = fl[i];
    c.mouth = c.pts[0];
    c.mouth_dir = glm::vec3(R.x, 0.0f, R.y);

    // Outer skin: the hillside as it was before the trench, raised into a
    // rock hump wherever the vault would otherwise poke through it.
    const int K = Cave::kCross;
    c.skin_y.resize(n * K);
    c.skin_u.resize(n * K);
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 t, s;
        frame(c, i, t, s);
        const float W = c.half_w[i] + 4.8f;
        for (int j = 0; j < K; ++j) {
            const float u = W * (2.0f * float(j) / float(K - 1) - 1.0f);
            const glm::vec3 q = c.pts[i] + s * u;
            const float o = hm.sample_world(q.x, q.z);
            const float ell = std::sqrt(std::max(0.0f, 1.0f - (u / W) * (u / W)));
            const float cv = c.cover[i];
            const float arch = c.pts[i].y + (c.height[i] + 1.7f) * ell * cv - (1.0f - cv) * 2.5f;
            float core = std::max(o + glm::mix(-0.7f, 0.5f, cv), arch);
            const float edge = sstep(W, W - 1.8f, std::abs(u));
            core += rock(glm::vec3(q.x, core, q.z)) * 0.45f * edge * cv;
            c.skin_u[i * K + j] = u;
            c.skin_y[i * K + j] = glm::mix(o - 0.8f, core, edge);
        }
    }
    return c;
}

void carve_cave(const Cave& c, Heightmap& hm) {
    if (c.empty() || hm.heights.empty()) return;
    const int W = hm.width();
    glm::vec2 lo(1e9f), hi(-1e9f);
    for (int i = 0; i < c.n_inner; ++i) { lo = glm::min(lo, xz(c.pts[i])); hi = glm::max(hi, xz(c.pts[i])); }
    lo -= glm::vec2(9.0f); hi += glm::vec2(9.0f);
    auto to_ix = [&](float w, float o) { return std::clamp(int((w - o) / hm.cell), 0, W - 1); };
    const int x0 = to_ix(lo.x, hm.origin_x), x1 = to_ix(hi.x, hm.origin_x);
    const int z0 = to_ix(lo.y, hm.origin_z), z1 = to_ix(hi.y, hm.origin_z);
    for (int iz = z0; iz <= z1; ++iz) {
        for (int ix = x0; ix <= x1; ++ix) {
            const glm::vec2 p(hm.origin_x + float(ix) * hm.cell, hm.origin_z + float(iz) * hm.cell);
            float best = 1e9f, floor_y = 0.0f, hw = 0.0f;
            for (int i = 0; i + 1 < c.n_inner; ++i) {
                const glm::vec2 a = xz(c.pts[i]), b = xz(c.pts[i + 1]);
                const glm::vec2 ab = b - a;
                const float t = std::clamp(glm::dot(p - a, ab) / std::max(glm::dot(ab, ab), 1e-6f), 0.0f, 1.0f);
                const float d = glm::length(p - (a + ab * t));
                if (d < best) {
                    best = d;
                    floor_y = glm::mix(c.pts[i].y, c.pts[i + 1].y, t);
                    hw = glm::mix(c.half_w[i], c.half_w[i + 1], t);
                }
            }
            if (best > hw + 1.6f) continue;
            float& h = hm.heights[size_t(iz) * size_t(W) + size_t(ix)];
            const float target = glm::mix(floor_y, h, sstep(hw, hw + 1.6f, best));
            h = std::min(h, target);
        }
    }
}

void append_cave_mesh(game::Level& lv, const Cave& c) {
    if (c.empty()) return;
    game::LevelMesh m;
    m.tex_albedo = 7; m.tex_normal = 7;                 // terrain rock layer
    m.triplanar = true; m.uv_scale = 0.30f;
    m.tint = glm::vec3(0.95f, 0.92f, 0.88f);
    m.fallback = glm::vec3(0.42f, 0.40f, 0.37f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const int K = Cave::kCross;
    const size_t n = c.pts.size();

    // ---- Inner vault (ceiling + upper walls), normals facing in ----
    const int NI = 12;
    auto ring_pt = [&](size_t i, int j, float scale, float adv, glm::vec3& nrm) {
        glm::vec3 t, s;
        frame(c, i, t, s);
        const float th = 3.14159265f * float(j) / float(NI);
        const float a = (c.half_w[i] + 0.9f) * scale, hh = (c.height[i] + 0.45f) * (0.35f + 0.65f * scale);
        const glm::vec3 dir = s * std::cos(th) + up * std::sin(th);
        glm::vec3 p = c.pts[i] + t * adv + s * (a * std::cos(th)) + up * (-0.45f + hh * std::sin(th));
        p += dir * (rock(p * 1.3f) * 0.28f * std::sin(th) * scale);      // rough rock, feet stay put
        nrm = -dir;
        return p;
    };
    struct Ring { size_t i; float scale, adv; };
    std::vector<Ring> rings;
    for (int i = 0; i < c.n_inner; ++i) rings.push_back({ size_t(i), 1.0f, 0.0f });
    {   // rounded end wall behind the chamber
        const size_t e = size_t(c.n_inner - 1);
        const float hw = c.half_w[e];
        rings.push_back({ e, 0.80f, hw * 0.55f });
        rings.push_back({ e, 0.45f, hw * 0.90f });
        rings.push_back({ e, 0.04f, hw * 1.00f });
    }
    for (size_t r = 0; r + 1 < rings.size(); ++r)
        for (int j = 0; j < NI; ++j) {
            glm::vec3 n00, n01, n10, n11;
            const glm::vec3 p00 = ring_pt(rings[r].i, j, rings[r].scale, rings[r].adv, n00);
            const glm::vec3 p01 = ring_pt(rings[r].i, j + 1, rings[r].scale, rings[r].adv, n01);
            const glm::vec3 p10 = ring_pt(rings[r + 1].i, j, rings[r + 1].scale, rings[r + 1].adv, n10);
            const glm::vec3 p11 = ring_pt(rings[r + 1].i, j + 1, rings[r + 1].scale, rings[r + 1].adv, n11);
            quad(m, p00, p01, p11, p10, n00, n01, n11, n10);
        }

    // ---- Outer skin (the restored hillside), smooth normals ----
    auto skin = [&](size_t i, int j) {
        glm::vec3 t, s;
        frame(c, i, t, s);
        glm::vec3 p = c.pts[i] + s * c.skin_u[i * K + j];
        p.y = c.skin_y[i * K + j];
        return p;
    };
    auto skin_n = [&](size_t i, int j) {
        const glm::vec3 a = skin(i > 0 ? i - 1 : i, j), b = skin(std::min(i + 1, n - 1), j);
        const glm::vec3 l = skin(i, std::max(j - 1, 0)), r = skin(i, std::min(j + 1, K - 1));
        glm::vec3 nn = glm::cross(b - a, r - l);
        if (nn.y < 0.0f) nn = -nn;
        return glm::length(nn) > 1e-6f ? glm::normalize(nn) : up;
    };
    for (size_t i = 0; i + 1 < n; ++i)
        for (int j = 0; j + 1 < K; ++j)
            quad(m, skin(i, j), skin(i, j + 1), skin(i + 1, j + 1), skin(i + 1, j),
                 skin_n(i, j), skin_n(i, j + 1), skin_n(i + 1, j + 1), skin_n(i + 1, j));

    // ---- Portal: the rock face around the mouth, between skin and vault ----
    {
        glm::vec3 t, s;
        frame(c, 0, t, s);
        const float W = c.half_w[0] + 4.8f, a = c.half_w[0] + 0.9f, hh = c.height[0] + 0.45f;
        const int NC = 56;
        auto col = [&](int k, glm::vec3& bot, glm::vec3& top) {
            const float u = W * (2.0f * float(k) / float(NC) - 1.0f);
            const float fj = (u / W * 0.5f + 0.5f) * float(K - 1);
            const int j0 = std::clamp(int(fj), 0, K - 2);
            const float y_top = glm::mix(c.skin_y[j0], c.skin_y[j0 + 1], fj - float(j0));
            const float y_arch = std::abs(u) < a ? c.pts[0].y - 0.45f + hh * std::sqrt(1.0f - (u / a) * (u / a))
                                                 : c.pts[0].y - 3.0f;
            const glm::vec3 base = c.pts[0] + s * u;
            bot = glm::vec3(base.x, std::min(y_arch, y_top), base.z);
            top = glm::vec3(base.x, y_top, base.z);
        };
        for (int k = 0; k < NC; ++k) {
            glm::vec3 b0, t0, b1, t1;
            col(k, b0, t0); col(k + 1, b1, t1);
            quad(m, b0, b1, t1, t0, -t, -t, -t, -t);
        }
    }
    lv.meshes.push_back(std::move(m));

    // ---- Colliders: roof slab and the upper walls (the floor and lower
    // walls are terrain). Coarse boxes, enough to stop the player walking
    // through the rock from inside or sinking through the roof. ----
    for (int i = 0; i < c.n_inner; ++i) {
        glm::vec3 t, s;
        frame(c, size_t(i), t, s);
        const glm::vec3 p = c.pts[size_t(i)];
        const float hw = c.half_w[size_t(i)], hh = c.height[size_t(i)];
        const glm::vec3 roof = p + up * (hh + 0.9f);
        lv.aabbs.push_back({ roof - glm::vec3(hw + 1.2f, 0.7f, hw + 1.2f) * glm::vec3(std::abs(s.x) + 0.35f, 1.0f, std::abs(s.z) + 0.35f),
                             roof + glm::vec3(hw + 1.2f, 0.7f, hw + 1.2f) * glm::vec3(std::abs(s.x) + 0.35f, 1.0f, std::abs(s.z) + 0.35f) });
        for (float sg : { -1.0f, 1.0f }) {
            const glm::vec3 w = p + s * (sg * (hw + 1.0f)) + up * (hh * 0.5f + 0.6f);
            lv.aabbs.push_back({ w - glm::vec3(0.7f, hh * 0.5f, 0.7f), w + glm::vec3(0.7f, hh * 0.5f, 0.7f) });
        }
    }
    {
        glm::vec3 t, s;
        const size_t e = size_t(c.n_inner - 1);
        frame(c, e, t, s);
        const glm::vec3 w = c.pts[e] + t * (c.half_w[e] + 1.2f) + up * (c.height[e] * 0.5f);
        lv.aabbs.push_back({ w - glm::vec3(c.half_w[e] + 1.0f, c.height[e] * 0.5f + 0.5f, c.half_w[e] + 1.0f) * glm::vec3(std::abs(s.x) + 0.2f, 1.0f, std::abs(s.z) + 0.2f),
                             w + glm::vec3(c.half_w[e] + 1.0f, c.height[e] * 0.5f + 0.5f, c.half_w[e] + 1.0f) * glm::vec3(std::abs(s.x) + 0.2f, 1.0f, std::abs(s.z) + 0.2f) });
    }
}

Road cave_footprint(const Cave& c) {
    Road r;
    if (c.empty()) return r;
    r.pts = c.pts;
    r.half_width = 7.5f;
    r.shoulder = 0.0f;
    r.blend = 0.0f;
    glm::vec2 lo(1e9f), hi(-1e9f);
    for (const auto& p : c.pts) { lo = glm::min(lo, xz(p)); hi = glm::max(hi, xz(p)); }
    r.bb_min = lo - glm::vec2(12.0f);
    r.bb_max = hi + glm::vec2(12.0f);
    return r;
}

}  // namespace qlike
