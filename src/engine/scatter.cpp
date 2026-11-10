#include "engine/scatter.h"
#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/log.h"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace qlike {
namespace {

float hash31(glm::vec3 p) {
    float h = p.x * 12.9898f + p.y * 78.233f + p.z * 37.719f;
    float s = std::sin(h) * 43758.5453f;
    return s - std::floor(s);
}

// Lumpy rock: icosahedron subdivided twice, vertices displaced by a
// per-direction hash so it reads as an irregular stone, bottom
// flattened so it sits on the ground without floating edges.
void build_rock_mesh_data(std::vector<Vertex>& verts,
                          std::vector<uint32_t>& idx) {
    // Icosahedron base.
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<glm::vec3> pos = {
        {-1,  t, 0}, {1,  t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1,  t}, {0, 1,  t}, {0, -1, -t}, {0, 1, -t},
        { t, 0, -1}, {t, 0,  1}, {-t, 0, -1}, {-t, 0,  1},
    };
    for (auto& p : pos) p = glm::normalize(p);
    std::vector<uint32_t> faces = {
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    };
    // ONE subdivision round (80 tris / 240 verts). Two rounds looked
    // marginally rounder but cost 4x the vertex invocations — at tens of
    // thousands of instances the vertex stage dominated the whole frame
    // (60k x 960 verts = 58M VS invocations, ~100 ms). Clutter reads at
    // a glance; nobody counts facets on a fist-sized rock.
    for (int round = 0; round < 1; ++round) {
        std::vector<glm::vec3> np;
        std::vector<uint32_t>  nf;
        np.reserve(faces.size() * 2);
        for (size_t f = 0; f < faces.size(); f += 3) {
            glm::vec3 a = pos[faces[f]], b = pos[faces[f + 1]],
                      c = pos[faces[f + 2]];
            glm::vec3 ab = glm::normalize(a + b);
            glm::vec3 bc = glm::normalize(b + c);
            glm::vec3 ca = glm::normalize(c + a);
            uint32_t base = uint32_t(np.size());
            np.insert(np.end(), { a, ab, ca,  ab, b, bc,
                                  ca, bc, c,  ab, bc, ca });
            for (int i = 0; i < 12; ++i) nf.push_back(base + i);
        }
        pos = std::move(np);
        faces = std::move(nf);
    }
    // Displace by direction hash (3 octaves of pseudo-noise) and
    // flatten the bottom third.
    for (auto& p : pos) {
        float n = hash31(p * 2.1f) * 0.5f +
                  hash31(p * 4.7f + glm::vec3(13.0f)) * 0.3f +
                  hash31(p * 9.3f + glm::vec3(41.0f)) * 0.2f;
        float r = 0.78f + n * 0.44f;            // 0.78..1.22
        p *= r;
        if (p.y < -0.35f) p.y = -0.35f - (p.y + 0.35f) * 0.15f;
    }
    // Flat-face normals (one normal per triangle, vertices unshared).
    verts.clear();
    idx.clear();
    verts.reserve(faces.size());
    idx.reserve(faces.size());
    for (size_t f = 0; f < faces.size(); f += 3) {
        glm::vec3 a = pos[faces[f]], b = pos[faces[f + 1]],
                  c = pos[faces[f + 2]];
        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        uint32_t base = uint32_t(verts.size());
        verts.push_back({ a, n, glm::vec2(0.0f) });
        verts.push_back({ b, n, glm::vec2(0.0f) });
        verts.push_back({ c, n, glm::vec2(0.0f) });
        idx.insert(idx.end(), { base, base + 1, base + 2 });
    }
}

// ---- Small plant meshes (unit-ish size, y up, base at origin) ----
// uv.x = part id (0 leaf/stem, 1 petal, 2 flower centre), uv.y = sway.
void plant_quad(std::vector<Vertex>& v, std::vector<uint32_t>& ix,
                glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d,
                float part, float s0, float s1) {
    glm::vec3 n = glm::normalize(glm::cross(b - a, d - a));
    uint32_t k = uint32_t(v.size());
    v.push_back({ a, n, { part, s0 } });
    v.push_back({ b, n, { part, s0 } });
    v.push_back({ c, n, { part, s1 } });
    v.push_back({ d, n, { part, s1 } });
    ix.insert(ix.end(), { k, k + 1, k + 2, k, k + 2, k + 3 });
}

// Fern: 9 arching fronds, 10 segments each, folded along the rib, with a
// serrated width profile so each frond reads as rows of leaflets.
void build_fern(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    const int fronds = 9, segs = 10;
    for (int f = 0; f < fronds; ++f) {
        float ang = 6.2831853f * float(f) / float(fronds) + 0.35f * std::sin(float(f) * 2.1f);
        glm::vec3 dir(std::cos(ang), 0.0f, std::sin(ang));
        glm::vec3 side(-dir.z, 0.0f, dir.x);
        float len = 0.80f + 0.30f * std::sin(float(f) * 1.7f);
        float rise = 0.50f + 0.15f * std::cos(float(f) * 2.9f);
        auto rib = [&](float t) {
            float r = t * len;
            return dir * r + glm::vec3(0.0f, std::sin(t * 2.2f) * rise * len - t * t * 0.28f, 0.0f);
        };
        auto width = [&](float t) {
            float env = 0.17f * std::sin(3.14159f * std::min(1.0f, t * 1.05f + 0.04f));
            return env * (0.55f + 0.45f * std::abs(std::sin(t * 3.14159f * 7.0f)));   // leaflets
        };
        for (int s = 0; s < segs; ++s) {
            float t0 = float(s) / segs, t1 = float(s + 1) / segs;
            float w0 = width(t0), w1 = width(t1);
            glm::vec3 p0 = rib(t0), p1 = rib(t1);
            glm::vec3 fold(0.0f, -0.035f, 0.0f);
            plant_quad(v, ix, p0, p0 + side * w0 + fold, p1 + side * w1 + fold, p1, 0.0f, t0, t1);
            plant_quad(v, ix, p0 - side * w0 + fold, p0, p1, p1 - side * w1 + fold, 0.0f, t0, t1);
        }
    }
}

// Tapered tube along `path` (stems), `sides` around, part 0 (leaf colour);
// uv.y = sway weight w0 -> w1.
void plant_tube(std::vector<Vertex>& v, std::vector<uint32_t>& ix, const std::vector<glm::vec3>& path,
                float r0, float r1, int sides, float w0, float w1) {
    const size_t n = path.size();
    if (n < 2) return;
    const uint32_t base = uint32_t(v.size());
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 t = glm::normalize(path[std::min(i + 1, n - 1)] - path[i > 0 ? i - 1 : 0]);
        glm::vec3 ref = std::abs(t.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 u = glm::normalize(glm::cross(t, ref)), w = glm::cross(t, u);
        float f = float(i) / float(n - 1);
        float r = r0 + (r1 - r0) * f;
        for (int s = 0; s < sides; ++s) {
            float a = 6.2831853f * float(s) / float(sides);
            glm::vec3 nrm = u * std::cos(a) + w * std::sin(a);
            v.push_back({ path[i] + nrm * r, nrm, { 0.0f, w0 + (w1 - w0) * f } });
        }
    }
    for (size_t i = 0; i + 1 < n; ++i)
        for (int s = 0; s < sides; ++s) {
            uint32_t a = base + uint32_t(i) * uint32_t(sides) + uint32_t(s);
            uint32_t b = base + uint32_t(i) * uint32_t(sides) + uint32_t((s + 1) % sides);
            uint32_t c = a + uint32_t(sides), d = b + uint32_t(sides);
            ix.insert(ix.end(), { a, b, d, a, d, c });
        }
}

// Curved, folded leaf blade from `b` along `dir` (4 segments).
void plant_leaf(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 b, glm::vec3 dir,
                float len, float wid, float lift, float w_base) {
    glm::vec3 side = glm::normalize(glm::vec3(-dir.z, 0.0f, dir.x));
    const int segs = 4;
    auto rib = [&](float t) {
        return b + dir * (len * t) + glm::vec3(0.0f, std::sin(t * 2.0f) * lift - t * t * lift * 0.6f, 0.0f);
    };
    for (int s = 0; s < segs; ++s) {
        float t0 = float(s) / segs, t1 = float(s + 1) / segs;
        float wa = wid * std::sin(3.14159f * std::min(1.0f, t0 * 0.95f + 0.05f));
        float wb = wid * std::sin(3.14159f * std::min(1.0f, t1 * 0.95f + 0.05f));
        glm::vec3 p0 = rib(t0), p1 = rib(t1), fold(0.0f, -0.008f, 0.0f);
        float s0 = w_base + 0.25f * t0, s1 = w_base + 0.25f * t1;
        plant_quad(v, ix, p0, p0 + side * wa + fold, p1 + side * wb + fold, p1, 0.0f, s0, s1);
        plant_quad(v, ix, p0 - side * wa + fold, p0, p1, p1 - side * wb + fold, 0.0f, s0, s1);
    }
}

// Flower head at hc facing `up`: segmented rounded petals that cup
// progressively along their length, plus a domed centre.
void plant_head(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 hc, glm::vec3 up,
                int petals, float head_r, float petal_cup, float petal_w, float seed) {
    glm::vec3 t1 = glm::normalize(glm::cross(up, std::abs(up.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
    glm::vec3 t2 = glm::cross(up, t1);
    const int segs = 3;
    for (int p = 0; p < petals; ++p) {
        float am = 6.2831853f * (float(p) + 0.5f) / float(petals) + seed;
        float half = 3.14159f / float(petals) * petal_w;
        glm::vec3 em = t1 * std::cos(am) + t2 * std::sin(am);
        glm::vec3 es = t1 * std::cos(am + 1.5708f) + t2 * std::sin(am + 1.5708f);
        auto pt = [&](float t, float sgn) {   // t along the petal, sgn -1..1 across it
            float r = head_r * (0.12f + 0.95f * t);
            float wv = std::sin(3.14159f * std::min(1.0f, 0.25f + 0.85f * t)) * std::tan(half) * r;
            float cup = petal_cup * head_r * t * t;
            return hc + em * r + es * (wv * sgn) + up * (cup - 0.15f * head_r * sgn * sgn * t);
        };
        const uint32_t k = uint32_t(v.size());
        for (int s = 0; s <= segs; ++s) {
            float t = float(s) / segs;
            glm::vec3 a = pt(t, -1.0f), c = pt(t, 0.0f), b = pt(t, 1.0f);
            glm::vec3 nrm = glm::normalize(glm::cross(pt(std::min(1.0f, t + 0.1f), 0.0f) - c, b - a) + up * 0.3f);
            v.push_back({ a, nrm, { 1.0f, 1.0f } });
            v.push_back({ c, nrm, { 1.0f, 1.0f } });
            v.push_back({ b, nrm, { 1.0f, 1.0f } });
        }
        for (int s = 0; s < segs; ++s) {
            uint32_t r0 = k + uint32_t(s) * 3u, r1 = r0 + 3u;
            ix.insert(ix.end(), { r0, r0 + 1, r1 + 1, r0, r1 + 1, r1,
                                  r0 + 1, r0 + 2, r1 + 2, r0 + 1, r1 + 2, r1 + 1 });
        }
    }
    // Domed centre: two rings + top.
    const uint32_t k = uint32_t(v.size());
    const int cs = 10;
    const float r = head_r * 0.22f;
    for (int ring = 0; ring < 2; ++ring) {
        float rr = ring == 0 ? r : r * 0.6f, hh = ring == 0 ? 0.006f : 0.016f;
        for (int i = 0; i < cs; ++i) {
            float a = 6.2831853f * float(i) / float(cs);
            glm::vec3 d = t1 * std::cos(a) + t2 * std::sin(a);
            v.push_back({ hc + d * rr + up * hh, glm::normalize(up + d * 0.5f), { 2.0f, 1.0f } });
        }
    }
    v.push_back({ hc + up * 0.022f, up, { 2.0f, 1.0f } });
    for (int i = 0; i < cs; ++i) {
        uint32_t a = k + uint32_t(i), b = k + uint32_t((i + 1) % cs), c = a + uint32_t(cs), d = b + uint32_t(cs);
        ix.insert(ix.end(), { a, b, d, a, d, c, c, d, k + uint32_t(2 * cs) });
    }
}

// Flower: curved tube stem, curled leaves, `heads` heads (a spike if > 1).
void build_flower(std::vector<Vertex>& v, std::vector<uint32_t>& ix,
                  int petals, float head_r, float petal_cup, float droop, float height,
                  float petal_w = 0.92f, int leaves = 2, int heads = 1) {
    std::vector<glm::vec3> path;
    const int ns = 6;
    for (int i = 0; i <= ns; ++i) {
        float t = float(i) / ns;
        path.push_back(glm::vec3(droop * 0.3f * t * t + 0.012f * std::sin(t * 5.0f), height * t, 0.0f));
    }
    plant_tube(v, ix, path, 0.0065f, 0.0035f, 5, 0.0f, 1.0f);
    for (int k = 0; k < leaves; ++k) {
        float a = float(k) * 2.39996f + 0.7f;
        glm::vec3 d(std::cos(a), 0.0f, std::sin(a));
        float lh = height * (0.10f + 0.16f * float(k % 3));
        plant_leaf(v, ix, glm::vec3(0.0f, lh, 0.0f), d, 0.20f + 0.05f * float(k % 2), 0.035f, 0.10f, 0.15f);
    }
    glm::vec3 hc = path.back();
    glm::vec3 up = glm::normalize(glm::vec3(droop, 1.0f - std::abs(droop) * 0.6f, 0.0f));
    if (heads <= 1) {
        plant_head(v, ix, hc, up, petals, head_r, petal_cup, petal_w, 0.0f);
    } else {   // spike: smaller heads spiralling up the top third of the stem
        for (int h = 0; h < heads; ++h) {
            float t = 0.62f + 0.38f * float(h) / float(heads - 1);
            float a = float(h) * 2.39996f;
            glm::vec3 out(std::cos(a), 0.0f, std::sin(a));
            glm::vec3 c = glm::vec3(0.0f, height * t, 0.0f) + out * (head_r * 0.6f);
            glm::vec3 u = glm::normalize(out * 0.9f + glm::vec3(0.0f, 0.6f, 0.0f));
            plant_head(v, ix, c, u, petals, head_r * (1.1f - 0.4f * t), petal_cup, petal_w, a);
        }
    }
}

// Sort instances by 32 m ground cell and emit one contiguous range per
// cell with a conservative AABB (instance size + raymarch-surface offset).
void sort_into_cells(std::vector<ScatterInstance>& v, uint32_t base,
                     std::vector<ScatterCell>& out) {
    const float kCell = 32.0f;
    auto key = [&](const ScatterInstance& s) {
        int cx = int(std::floor(s.pos_scale.x / kCell)) + 512;
        int cz = int(std::floor(s.pos_scale.z / kCell)) + 512;
        return cz * 1024 + cx;
    };
    std::stable_sort(v.begin(), v.end(),
                     [&](const ScatterInstance& a, const ScatterInstance& b) { return key(a) < key(b); });
    out.clear();
    for (size_t i = 0; i < v.size();) {
        const int k = key(v[i]);
        ScatterCell c{};
        c.mn = glm::vec3(1e9f);
        c.mx = glm::vec3(-1e9f);
        size_t j = i;
        for (; j < v.size() && key(v[j]) == k; ++j) {
            glm::vec3 p(v[j].pos_scale);
            float r = v[j].pos_scale.w * 1.5f + 0.5f;
            c.mn = glm::min(c.mn, p - glm::vec3(r, 1.5f, r));
            c.mx = glm::max(c.mx, p + glm::vec3(r, r * 2.0f + 1.5f, r));
        }
        c.first = base + uint32_t(i);
        c.count = uint32_t(j - i);
        out.push_back(c);
        i = j;
    }
}

} // namespace

ScatterSet build_scatter(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family, const Heightmap& hm,
                         const ScatterParams& params) {
    ScatterSet s{};
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    build_rock_mesh_data(verts, idx);
    s.rock_mesh = create_mesh_from_data(device, alloc, queue, queue_family,
                                        verts.data(), uint32_t(verts.size()),
                                        idx.data(), uint32_t(idx.size()));
    if (hm.heights.empty() || params.max_instances <= 0) return s;

    std::mt19937 rng(uint32_t(params.seed));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int W = hm.width();
    auto h_at = [&](float wx, float wz) { return hm.sample_world(wx, wz); };
    auto n_y_at = [&](float wx, float wz) {
        float e = 1.0f;
        float dx = h_at(wx + e, wz) - h_at(wx - e, wz);
        float dz = h_at(wx, wz + e) - h_at(wx, wz - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    (void)W;

    std::vector<ScatterInstance> inst;
    inst.reserve(size_t(params.max_instances));
    const int attempts = params.max_instances * 4;
    for (int i = 0; i < attempts &&
                    int(inst.size()) < params.max_instances; ++i) {
        float wx = (u01(rng) * 2.0f - 1.0f) * params.half_extent;
        float wz = (u01(rng) * 2.0f - 1.0f) * params.half_extent;
        if (std::abs(wx) < params.keep_out_xz.x &&
            std::abs(wz) < params.keep_out_xz.y) continue;
        if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
            wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
            params.road->distance(wx, wz) <
                params.road->half_width + params.road->shoulder + 0.6f) continue;
        float h = h_at(wx, wz);
        if (h < params.water_level + 0.4f) continue;     // not in/at water
        float ny = n_y_at(wx, wz);
        float slope = 1.0f - ny;

        // Rule weights. Boulders: steep ground or the rock band, plus a
        // sparse scattering everywhere. Pebbles: gentler ground,
        // clustered by a low-frequency hash so they come in patches.
        float rock_w = std::clamp(slope * 2.4f, 0.0f, 1.0f);
        rock_w = (std::max)(rock_w,
                            std::clamp((h - params.rock_band_h) / 25.0f,
                                       0.0f, 1.0f));
        rock_w *= 1.0f - std::clamp((h - params.snow_band_h) / 20.0f,
                                    0.0f, 0.7f);
        float clump = hash31(glm::vec3(std::floor(wx / 9.0f), 7.0f,
                                       std::floor(wz / 9.0f)));
        float pebble_w = (1.0f - slope * 1.5f) * (clump > 0.45f ? 1.0f : 0.15f);

        float r = u01(rng);
        bool boulder = r < rock_w * 0.16f + 0.012f;       // sparse big rocks
        bool pebble  = !boulder && r < 0.55f * std::clamp(pebble_w, 0.0f, 1.0f);
        if (!boulder && !pebble) continue;

        ScatterInstance si{};
        float sc = boulder
            ? 0.15f + u01(rng) * u01(rng) * 0.45f          // 0.15-0.6 m
            : 0.035f + u01(rng) * 0.085f;                  // 0.035-0.12 m
        // Sink proportionally so the flattened base sits in the ground.
        si.pos_scale = glm::vec4(wx, h - sc * 0.18f, wz, sc);
        float tint = 0.8f + u01(rng) * 0.4f;
        // Slight warm/cool per-rock variation around neutral.
        glm::vec3 tc(tint * (0.96f + u01(rng) * 0.08f),
                     tint,
                     tint * (0.94f + u01(rng) * 0.08f));
        si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(),
                                tc.x, tc.y, tc.z);
        inst.push_back(si);
    }

    // ---- Small plants: ferns + three flower species ----
    {
        std::vector<Vertex> pv; std::vector<uint32_t> pi;
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
            pv.clear(); pi.clear();
            switch (sp) {
            case 0: build_fern(pv, pi); break;
            case 1: build_flower(pv, pi, 14, 0.060f, 0.05f, 0.0f, 0.32f, 0.70f, 3); break;   // daisy
            case 2: build_flower(pv, pi, 5, 0.075f, 0.55f, 0.0f, 0.42f, 1.05f, 2); break;    // poppy (cup)
            case 3: build_flower(pv, pi, 6, 0.040f, -0.9f, 0.5f, 0.30f, 0.95f, 2); break;    // bluebell (nodding)
            case 4: build_flower(pv, pi, 16, 0.050f, 0.20f, 0.0f, 0.46f, 0.45f, 3); break;   // cornflower (thin rays)
            case 5: build_flower(pv, pi, 5, 0.035f, 0.70f, 0.0f, 0.24f, 1.10f, 3); break;    // buttercup (small cup)
            case 6: build_flower(pv, pi, 6, 0.028f, 0.35f, 0.0f, 0.62f, 0.95f, 4, 9); break; // lupin spike
            }
            s.plant_mesh[sp] = create_mesh_from_data(device, alloc, queue, queue_family,
                                                     pv.data(), uint32_t(pv.size()),
                                                     pi.data(), uint32_t(pi.size()));
        }
        std::vector<ScatterInstance> per[ScatterSet::kPlantSpecies];
        const float ext = std::min(params.half_extent * 1.3f, 280.0f);
        for (int i = 0; i < 420000; ++i) {
            float wx = (u01(rng) * 2.0f - 1.0f) * ext;
            float wz = (u01(rng) * 2.0f - 1.0f) * ext;
            if (std::abs(wx) < params.keep_out_xz.x && std::abs(wz) < params.keep_out_xz.y) continue;
            if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
                wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
                params.road->distance(wx, wz) < params.road->half_width + 0.4f) continue;
            float h = h_at(wx, wz);
            if (h < params.water_level + 0.6f || h > params.rock_band_h + 5.0f) continue;
            float ny = n_y_at(wx, wz);
            if (ny < 0.8f) continue;
            // Patches: each species owns low-frequency noise regions so
            // flowers come in drifts of one colour, ferns in damp hollows.
            auto patch = [&](float f, float seed) {
                return hash31(glm::vec3(std::floor(wx * f), seed, std::floor(wz * f))) * 0.6f +
                       hash31(glm::vec3(std::floor(wx * f * 3.1f), seed + 9.0f, std::floor(wz * f * 3.1f))) * 0.4f;
            };
            float pf = patch(0.06f, 1.0f), pd = patch(0.05f, 2.0f), pp = patch(0.045f, 3.0f), pb = patch(0.055f, 4.0f);
            float pc = patch(0.05f, 5.0f), pu = patch(0.065f, 6.0f), pl = patch(0.04f, 7.0f);
            int sp = -1;
            float r = u01(rng);
            if (pf > 0.62f && r < 0.22f) sp = 0;
            else if (pd > 0.60f && r < 0.55f) sp = 1;
            else if (pp > 0.64f && r < 0.55f) sp = 2;
            else if (pb > 0.64f && r < 0.55f) sp = 3;
            else if (pc > 0.66f && r < 0.50f) sp = 4;
            else if (pu > 0.63f && r < 0.60f) sp = 5;
            else if (pl > 0.70f && r < 0.40f) sp = 6;
            if (sp < 0) continue;
            ScatterInstance si{};
            float sc = (sp == 0) ? 0.55f + u01(rng) * 0.6f : 1.1f + u01(rng) * 0.6f;
            si.pos_scale = glm::vec4(wx, h - 0.02f, wz, sc);
            glm::vec3 col;
            switch (sp) {
            case 0: col = glm::vec3(0.10f, 0.22f, 0.06f) * (0.8f + 0.4f * u01(rng)); break;
            case 1: col = glm::vec3(0.85f, 0.84f, 0.80f); break;
            case 2: col = glm::vec3(0.75f, 0.06f, 0.03f) * (0.85f + 0.3f * u01(rng)); break;
            case 3: col = glm::vec3(0.22f, 0.20f, 0.62f) * (0.85f + 0.3f * u01(rng)); break;
            case 4: col = glm::vec3(0.16f, 0.26f, 0.85f) * (0.85f + 0.3f * u01(rng)); break;   // cornflower
            case 5: col = glm::vec3(0.95f, 0.76f, 0.06f) * (0.9f + 0.2f * u01(rng)); break;    // buttercup
            default: {   // lupin: pink .. purple drifts
                float k = u01(rng);
                col = glm::mix(glm::vec3(0.80f, 0.30f, 0.55f), glm::vec3(0.45f, 0.22f, 0.75f), k);
                break;
            }
            }
            si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
            per[sp].push_back(si);
        }
        std::vector<ScatterInstance> all;
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
            s.plant_first[sp] = uint32_t(all.size());
            sort_into_cells(per[sp], uint32_t(all.size()), s.plant_cells[sp]);
            s.plant_count[sp] = uint32_t(per[sp].size());
            all.insert(all.end(), per[sp].begin(), per[sp].end());
        }
        if (params.rm_params) {
            s.plant_dy.resize(all.size());
            for (size_t i = 0; i < all.size(); ++i)
                s.plant_dy[i] = raymarch_fine_correction(hm, *params.rm_params, all[i].pos_scale.x,
                                                         all[i].pos_scale.z, params.rm_octaves);
        }
        if (!all.empty()) {
            VkBufferCreateInfo pb{};
            pb.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            pb.size = sizeof(ScatterInstance) * all.size();
            pb.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VmaAllocationCreateInfo pa{};
            pa.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            pa.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo pai{};
            if (vmaCreateBuffer(alloc, &pb, &pa, &s.plant_buffer, &s.plant_alloc, &pai) == VK_SUCCESS) {
                std::memcpy(pai.pMappedData, all.data(), sizeof(ScatterInstance) * all.size());
                s.plant_mapped = pai.pMappedData;
                s.plant_cpu = all;
            }
        }
        log::infof("[scatter] plants: %u ferns, %u daisies, %u poppies, %u bluebells, %u cornflowers, %u buttercups, %u lupins",
                   s.plant_count[0], s.plant_count[1], s.plant_count[2], s.plant_count[3],
                   s.plant_count[4], s.plant_count[5], s.plant_count[6]);
    }

    if (inst.empty()) return s;
    sort_into_cells(inst, 0, s.rock_cells);
    const VkDeviceSize bytes = sizeof(ScatterInstance) *
                               VkDeviceSize(inst.size());
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    if (vmaCreateBuffer(alloc, &bci, &aci, &s.instance_buffer,
                        &s.instance_alloc, &ai) != VK_SUCCESS) {
        log::warn("[scatter] instance buffer alloc failed");
        return s;
    }
    std::memcpy(ai.pMappedData, inst.data(), size_t(bytes));
    s.inst_mapped = ai.pMappedData;
    s.inst_cpu = inst;
    if (params.rm_params) {
        s.inst_dy.resize(inst.size());
        for (size_t i = 0; i < inst.size(); ++i)
            s.inst_dy[i] = raymarch_fine_correction(hm, *params.rm_params, inst[i].pos_scale.x,
                                                    inst[i].pos_scale.z, params.rm_octaves);
    }
    s.instance_count = uint32_t(inst.size());
    log::infof("[scatter] %u clutter instances placed (rock mesh %u tris)",
               s.instance_count, s.rock_mesh.index_count / 3);
    return s;
}

void destroy_scatter(VmaAllocator alloc, ScatterSet& s) {
    destroy_mesh(alloc, s.rock_mesh);
    for (auto& m : s.plant_mesh) if (m.vertex_buffer) destroy_mesh(alloc, m);
    if (s.plant_buffer) {
        vmaDestroyBuffer(alloc, s.plant_buffer, s.plant_alloc);
        s.plant_buffer = VK_NULL_HANDLE;
    }
    if (s.instance_buffer) {
        vmaDestroyBuffer(alloc, s.instance_buffer, s.instance_alloc);
        s.instance_buffer = VK_NULL_HANDLE;
        s.instance_alloc = nullptr;
    }
    s.instance_count = 0;
}

} // namespace qlike

namespace qlike {
void scatter_set_raymarch_surface(ScatterSet& s, bool on) {
    if (s.rm_surface == on) return;
    const float sign = on ? 1.0f : -1.0f;
    auto apply = [&](void* mapped, const std::vector<float>& dy) {
        if (!mapped) return;
        auto* p = static_cast<ScatterInstance*>(mapped);
        for (size_t i = 0; i < dy.size(); ++i) p[i].pos_scale.y += sign * dy[i];
    };
    apply(s.inst_mapped, s.inst_dy);
    apply(s.plant_mapped, s.plant_dy);
    s.rm_surface = on;
}
} // namespace qlike
