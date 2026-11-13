#include "engine/scatter.h"
#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/log.h"
#include "engine/lsystem.h"

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
                float len, float wid, float lift, float w_base, int segs = 8, int across = 4, float round = 0.0f, float teeth = 0.0f) {
    // A smooth curved surface S(t, u): t along the midrib, u across the
    // blade. Shared vertices with analytic (finite-difference) normals, so
    // the leaf shades as one continuous, gently cupped surface instead of
    // a strip of flat facets.
    const glm::vec3 side = glm::normalize(glm::vec3(-dir.z, 0.0f, dir.x));
    // (default 9 x 5 vertices; small, numerous leaves pass fewer.)
    // `round` 0 = lance shape with a pointed tip, 1 = obovate (clover,
    // plantain): widest past the middle, blunt tip.
    auto half_w = [&](float t) {                          // rounded base, widest at 40 %, pointed tip
        float tc = glm::clamp(t, 0.0f, 1.0f);
        float a = std::pow(std::sin(3.14159f * std::pow(tc, 0.75f)), 0.8f);
        float ob = std::sqrt(std::max(0.0f, std::sin(3.14159f * std::pow(tc, 1.6f))));
        a = a + (ob - a) * round;
        // Saw-toothed margin (dandelion, nettle): `teeth` notches along the blade.
        if (teeth > 0.0f) a *= 1.0f - 0.42f * glm::fract(tc * teeth) * (1.0f - tc * 0.5f);
        return wid * (0.06f + 0.94f * a) * (t > 0.999f ? (round > 0.5f ? 0.35f : 0.0f) : 1.0f);
    };
    auto S = [&](float t, float u) {                      // u in -1..1
        glm::vec3 rib = b + dir * (len * t) +
                        glm::vec3(0.0f, std::sin(t * 2.0f) * lift - t * t * lift * 0.6f, 0.0f);
        float w = half_w(t);
        // Cupped along the midrib (edges raised), edges slightly wavy.
        float cup = 0.35f * w * u * u + 0.06f * w * std::sin(t * 9.0f) * u;
        return rib + side * (w * u) + glm::vec3(0.0f, cup, 0.0f);
    };
    const uint32_t base = uint32_t(v.size());
    for (int s = 0; s <= segs; ++s) {
        float t = float(s) / float(segs);
        for (int a = 0; a <= across; ++a) {
            float u = -1.0f + 2.0f * float(a) / float(across);
            const float e = 0.02f;
            glm::vec3 dt = S(std::min(t + e, 1.0f), u) - S(std::max(t - e, 0.0f), u);
            glm::vec3 du = S(std::min(t, 0.97f), std::min(u + 0.1f, 1.0f)) - S(std::min(t, 0.97f), std::max(u - 0.1f, -1.0f));
            glm::vec3 n = glm::cross(dt, du);
            float l = glm::length(n);
            n = l > 1e-9f ? n / l : glm::vec3(0.0f, 1.0f, 0.0f);
            if (n.y < 0.0f) n = -n;
            // uv: part 3 (leaf blade) + the across coordinate in the fraction;
            // sway base in 1/32 steps + the along coordinate in the fraction
            // (decoded by scatter.vert / .frag and clutter_shadow.vert).
            v.push_back({ S(t, u), n, { 3.0f + 0.4f * u, std::floor(w_base * 32.0f) / 32.0f + t * (0.98f / 32.0f) } });
        }
    }
    const uint32_t row = uint32_t(across + 1);
    for (int s = 0; s < segs; ++s)
        for (int a = 0; a < across; ++a) {
            uint32_t i0 = base + uint32_t(s) * row + uint32_t(a), i1 = i0 + 1, i2 = i0 + row, i3 = i2 + 1;
            ix.insert(ix.end(), { i0, i1, i3, i0, i3, i2 });
        }
}

// Flower head at hc facing `up`: segmented rounded petals that cup
// progressively along their length, plus a domed centre.
void plant_head(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 hc, glm::vec3 up,
                int petals, float head_r, float petal_cup, float petal_w, float seed,
                float sway = 1.0f, int segs = 3) {
    glm::vec3 t1 = glm::normalize(glm::cross(up, std::abs(up.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
    glm::vec3 t2 = glm::cross(up, t1);
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
            // Part 4: petal with petal-space coordinates (across in uv.x's
            // fraction, along packed with the sway in uv.y - see plant_leaf).
            const float pk = std::floor(sway * 32.0f) / 32.0f + t * (0.98f / 32.0f);
            v.push_back({ a, nrm, { 4.0f - 0.4f, pk } });
            v.push_back({ c, nrm, { 4.0f, pk } });
            v.push_back({ b, nrm, { 4.0f + 0.4f, pk } });
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
            v.push_back({ hc + d * rr + up * hh, glm::normalize(up + d * 0.5f), { 2.0f, sway } });
        }
    }
    v.push_back({ hc + up * 0.022f, up, { 2.0f, sway } });
    for (int i = 0; i < cs; ++i) {
        uint32_t a = k + uint32_t(i), b = k + uint32_t((i + 1) % cs), c = a + uint32_t(cs), d = b + uint32_t(cs);
        ix.insert(ix.end(), { a, b, d, a, d, c, c, d, k + uint32_t(2 * cs) });
    }
}

// Fern (L-system): a crown of fronds in a 137.5 deg spiral. Each frond is
// a rachis R(i) that puts out a pair of pinnae per node and arches over
// under gravity; pinna length follows the lance-shaped outline of a real
// frond (short at the base, longest a third up, tapering to the tip).
//   axiom : [ /(137.5 k) &(tilt) R(0, len) ] for every frond k
//   R(i,l): i < n -> F(l/n) [ +(a) P(w_i) ] [ -(a) P(w_i) ] R(i+1, l)
//           i = n -> P(tip)
void build_fern(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    using namespace lsys;
    const int fronds = 8, n = 11;
    Grammar g;
    {
        Rng r{ 7u };
        for (int k = 0; k < fronds; ++k) {
            g.axiom.push_back({ '[' });
            g.axiom.push_back({ '/', 137.5f * float(k) + r.range(-12.0f, 12.0f) });
            g.axiom.push_back({ '&', r.range(34.0f, 52.0f) });
            g.axiom.push_back({ 'R', 0.0f, r.range(0.78f, 1.12f) });
            g.axiom.push_back({ ']' });
        }
    }
    g.rules['R'] = [n](const Module& m, Rng& r, Str& o) {
        const int i = int(m.a + 0.5f);
        const float len = m.b;
        if (i >= n) { o.push_back({ 'P', 0.07f * len, 1.0f }); return; }
        const float t = (float(i) + 1.0f) / float(n);
        const float w = len * 0.25f * std::pow(std::sin(3.14159f * (0.10f + 0.88f * t)), 0.85f);
        const float a = 74.0f - 26.0f * t;                    // pinnae sweep forward toward the tip
        o.push_back({ 'F', len / float(n) });
        o.push_back({ '[' }); o.push_back({ '+', a + r.range(-5.0f, 5.0f) }); o.push_back({ 'P', w, t }); o.push_back({ ']' });
        o.push_back({ '[' }); o.push_back({ '-', a + r.range(-5.0f, 5.0f) }); o.push_back({ 'P', w, t }); o.push_back({ ']' });
        o.push_back({ 'R', float(i + 1), len });
    };
    TurtleParams tp;
    tp.radius = 0.007f; tp.taper = 0.9f;
    tp.bend = 1.75f;                                           // fronds arch over
    tp.organs = "P";
    const Shape sh = interpret(g.derive(n + 1, 11u), tp);
    float reach = 1e-3f;
    for (const auto& s : sh.segs) reach = std::max(reach, s.d1);
    // Rachis: a thin ribbon facing up.
    for (const auto& s : sh.segs) {
        glm::vec3 d = glm::normalize(s.b - s.a);
        glm::vec3 side = glm::normalize(glm::cross(d, glm::vec3(0.0f, 1.0f, 0.0f)));
        plant_quad(v, ix, s.a - side * s.r0, s.a + side * s.r0, s.b + side * s.r1, s.b - side * s.r1,
                   0.0f, s.d0 / reach, s.d1 / reach);
    }
    // Pinnae: folded, drooping blades (two triangles each).
    for (const auto& o : sh.organs) {
        const float len = o.a, sw = glm::clamp(o.dist / reach, 0.0f, 1.0f);
        const glm::vec3 droop(0.0f, -0.22f * len, 0.0f);
        const glm::vec3 base = o.pos, tip = o.pos + o.H * len + droop;
        const glm::vec3 mid = o.pos + o.H * (len * 0.42f) + droop * 0.3f;
        // Width runs along the rachis (perpendicular to the pinna, in the frond plane).
        const glm::vec3 w = glm::normalize(glm::cross(o.U, o.H)) * (len * 0.20f + 0.004f);
        const glm::vec3 fold = o.U * (len * 0.05f);
        glm::vec3 n1 = glm::normalize(glm::cross(mid + w - base, tip - base));
        glm::vec3 n2 = glm::normalize(glm::cross(tip - base, mid - w - base));
        if (n1.y < 0.0f) n1 = -n1;
        if (n2.y < 0.0f) n2 = -n2;
        uint32_t k = uint32_t(v.size());
        // Part 3 (leaf blade) with leaf-space coordinates, like plant_leaf.
        const float sk = std::floor(sw * 32.0f) / 32.0f;
        const float t0 = sk, tm = sk + 0.42f * (0.98f / 32.0f), t1 = sk + 0.98f / 32.0f;
        v.push_back({ base, n1, { 3.0f, t0 } }); v.push_back({ mid + w + fold, n1, { 3.4f, tm } }); v.push_back({ tip, n1, { 3.0f, t1 } });
        v.push_back({ base, n2, { 3.0f, t0 } }); v.push_back({ tip, n2, { 3.0f, t1 } }); v.push_back({ mid - w + fold, n2, { 2.6f, tm } });
        ix.insert(ix.end(), { k, k + 1, k + 2, k + 3, k + 4, k + 5 });
    }
}

// Flowering herb (L-system). One grammar covers the species; the numbers
// pick the habit:
//   S(i)  stem apex : i < nodes -> F [leaf] [side branch] /(137.5) S(i+1)
//                     i = nodes -> K (single terminal head) or A(0) (raceme)
//   B     side branch (sympodial): F [leaf] F K - or just a bud
//   A(j)  raceme apex: F [ &(a) F(pedicel) K ] /(roll) A(j+1), flowers
//         shrinking toward the tip (they open from the bottom up)
// Organs: L leaf, M palmate leaf, K flower head (parameter = size).
struct Herb {
    float height = 0.32f;
    int   nodes = 5;            // stem internodes
    int   leaves = 3;           // lowest nodes carrying a leaf
    float leaf_len = 0.20f, leaf_w = 0.035f;
    bool  palmate = false;
    int   branches = 0;         // side branches below the top
    float branch_len = 0.10f;
    int   petals = 8;
    float head_r = 0.05f, cup = 0.1f, petal_w = 0.9f;
    int   raceme = 0;           // flowers along the top of the stem
    float raceme_len = 0.3f;    // fraction of the height
    float raceme_roll = 137.5f; // 137.5 = spiral, ~0 = one-sided
    float pedicel_pitch = 62.0f;
    float lean = 0.0f;          // stem arches over (tropism strength)
    uint32_t seed = 1u;
};

void build_herb(std::vector<Vertex>& v, std::vector<uint32_t>& ix, const Herb& h) {
    using namespace lsys;
    Grammar g;
    g.axiom = { { '!', 0.0068f }, { 'S', 0.0f } };
    const float stem_len = h.height * (h.raceme > 0 ? 1.0f - h.raceme_len : 1.0f);
    g.rules['S'] = [&h, stem_len](const Module& m, Rng& r, Str& o) {
        const int i = int(m.a + 0.5f);
        if (i >= h.nodes) {
            if (h.raceme > 0) o.push_back({ 'A', 0.0f }); else o.push_back({ 'K', 1.0f });
            return;
        }
        o.push_back({ 'F', r.jit(stem_len / float(h.nodes), 0.18f) });
        if (i < h.leaves) {
            const float ll = h.leaf_len * (1.0f - 0.13f * float(i)) * r.range(0.85f, 1.15f);
            o.push_back({ '[' }); o.push_back({ '&', r.range(52.0f, 72.0f) });
            o.push_back({ h.palmate ? 'M' : 'L', ll }); o.push_back({ ']' });
        }
        if (h.branches > 0 && i >= h.nodes - 1 - h.branches && i < h.nodes - 1) {
            o.push_back({ '[' }); o.push_back({ '/', 90.0f }); o.push_back({ '&', r.range(28.0f, 44.0f) });
            o.push_back({ '!', 0.0042f }); o.push_back({ 'B', 1.0f }); o.push_back({ ']' });
        }
        o.push_back({ '/', 137.5f + r.range(-15.0f, 15.0f) });
        o.push_back({ '^', r.range(-5.0f, 5.0f) });           // slight zig-zag between nodes
        o.push_back({ 'S', float(i + 1) });
    };
    g.rules['B'] = [&h](const Module&, Rng& r, Str& o) {
        const float bl = h.branch_len * r.range(0.75f, 1.25f);
        o.push_back({ 'F', bl * 0.5f });
        o.push_back({ '[' }); o.push_back({ '/', 80.0f }); o.push_back({ '&', 60.0f });
        o.push_back({ 'L', h.leaf_len * 0.55f }); o.push_back({ ']' });
        o.push_back({ '^', r.range(8.0f, 20.0f) });           // curves back up toward the light
        o.push_back({ 'F', bl * 0.5f });
        o.push_back({ 'K', r.u() < 0.3f ? 0.42f : r.range(0.7f, 0.9f) });   // bud or smaller flower
    };
    g.rules['A'] = [&h](const Module& m, Rng& r, Str& o) {
        const int j = int(m.a + 0.5f);
        if (j >= h.raceme) { o.push_back({ 'K', 0.34f }); return; }
        const float t = float(j) / float(h.raceme);
        o.push_back({ 'F', h.height * h.raceme_len / float(h.raceme) });
        o.push_back({ '[' }); o.push_back({ '&', h.pedicel_pitch + r.range(-8.0f, 8.0f) });
        o.push_back({ '!', 0.0026f }); o.push_back({ 'F', h.head_r * (0.9f - 0.4f * t) });
        o.push_back({ 'K', 1.0f - 0.55f * t }); o.push_back({ ']' });
        o.push_back({ '/', h.raceme_roll + r.range(-10.0f, 10.0f) });
        o.push_back({ 'A', float(j + 1) });
    };
    TurtleParams tp;
    tp.radius = 0.0068f; tp.taper = 0.93f;
    tp.tropism = glm::normalize(glm::vec3(1.0f, -0.55f, 0.0f));
    tp.bend = h.lean;
    tp.organs = "LMK";
    const Shape sh = interpret(g.derive(h.nodes + h.raceme + 3, h.seed), tp);
    float top = 1e-3f;
    for (const auto& s : sh.segs) top = std::max(top, s.b.y);
    for (const auto& o : sh.organs) top = std::max(top, o.pos.y);
    auto sway = [&](glm::vec3 p) { return glm::clamp(p.y / top, 0.0f, 1.0f); };
    for (const auto& s : sh.segs)
        plant_tube(v, ix, { s.a, s.b }, s.r0, s.r1, s.depth == 0 ? 5 : 4, sway(s.a), sway(s.b));
    for (const auto& o : sh.organs) {
        if (o.s == 'K') {
            const bool main_head = o.a > 0.95f;
            if (o.a < 0.5f) {                                  // closed bud: few narrow petals, strongly cupped
                plant_head(v, ix, o.pos, o.H, 5, h.head_r * 0.42f, 2.6f, 0.9f, o.dist * 31.0f, sway(o.pos), 2);
            } else {
                plant_head(v, ix, o.pos, o.H, h.petals, h.head_r * o.a, h.cup, h.petal_w, o.dist * 31.0f,
                           sway(o.pos), main_head ? 3 : 2);
            }
        } else {
            glm::vec3 d = o.H;
            if (std::abs(d.y) > 0.92f) d = glm::normalize(d + o.U * 0.5f);
            if (o.s == 'L') {
                plant_leaf(v, ix, o.pos, d, o.a, h.leaf_w * (o.a / h.leaf_len), o.a * 0.5f, sway(o.pos));
            } else {                                           // palmate: leaflets fanned around a short stalk
                const glm::vec3 hub = o.pos + d * (o.a * 0.45f);
                plant_tube(v, ix, { o.pos, hub }, 0.0028f, 0.002f, 3, sway(o.pos), sway(hub));
                const glm::vec3 ax = glm::normalize(d + glm::vec3(0.0f, 0.8f, 0.0f));
                glm::vec3 t1 = glm::normalize(glm::cross(ax, glm::vec3(0.31f, 0.0f, 0.95f)));
                glm::vec3 t2 = glm::cross(ax, t1);
                for (int k = 0; k < 7; ++k) {
                    float a = 6.2831853f * float(k) / 7.0f;
                    glm::vec3 ld = glm::normalize(t1 * std::cos(a) + t2 * std::sin(a) + ax * 0.25f);
                    plant_leaf(v, ix, hub, ld, o.a * 0.5f, h.leaf_w * 0.55f, o.a * 0.10f, sway(hub));
                }
            }
        }
    }
}

// Reed / bulrush clump (Typha-like) for the shore. A tuft of long, narrow,
// keeled leaf blades that rise almost vertically and arch over near the
// tip, round flowering stems among them, and the familiar brown cattail
// heads (a velvety cylinder with a thin spike above). Leaves carry
// leaf-space coordinates (part 3, like plant_leaf) so the leaf texture
// and venation run along them; heads are part 5.
void reed_blade(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 base, glm::vec3 lean,
                float len, float wid, float arch, uint32_t seed) {
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 L = glm::normalize(glm::vec3(lean.x, 0.0f, lean.z) + glm::vec3(1e-4f));
    const glm::vec3 side = glm::normalize(glm::cross(up, L));
    const float twist = (float((seed * 2654435761u) >> 8 & 0xFFFF) / 65535.0f - 0.5f) * 1.6f;
    const int segs = 10;
    auto spine = [&](float t) {
        // Straight up most of the way, then bending over toward `lean`.
        float bend = arch * t * t * t;
        return base + up * (len * t * (1.0f - 0.35f * arch * t * t)) + L * (len * (0.10f * t + bend * 0.55f));
    };
    const uint32_t b0 = uint32_t(v.size());
    for (int s = 0; s <= segs; ++s) {
        const float t = float(s) / float(segs);
        const glm::vec3 p = spine(t);
        const glm::vec3 tg = glm::normalize(spine(std::min(t + 0.05f, 1.0f)) - spine(std::max(t - 0.05f, 0.0f)));
        const float tw = twist * t;
        glm::vec3 sd = glm::normalize(side * std::cos(tw) + glm::cross(tg, side) * std::sin(tw));
        const float w = wid * (t < 0.85f ? 1.0f - 0.25f * t : (1.0f - t) / 0.15f * 0.79f);
        glm::vec3 nrm = glm::normalize(glm::cross(sd, tg));
        if (glm::dot(nrm, L) < 0.0f) nrm = -nrm;
        // Keeled (V-shaped) cross-section: the edges sit a little behind the midrib.
        const glm::vec3 keel = -nrm * (w * 0.35f);
        const float pk = t * (0.98f / 32.0f);
        v.push_back({ p - sd * w + keel, nrm, { 3.0f - 0.4f, pk } });
        v.push_back({ p,                 nrm, { 3.0f,        pk } });
        v.push_back({ p + sd * w + keel, nrm, { 3.0f + 0.4f, pk } });
    }
    for (int s = 0; s < segs; ++s) {
        const uint32_t r0 = b0 + uint32_t(s) * 3u, r1 = r0 + 3u;
        ix.insert(ix.end(), { r0, r0 + 1, r1 + 1, r0, r1 + 1, r1, r0 + 1, r0 + 2, r1 + 2, r0 + 1, r1 + 2, r1 + 1 });
    }
}

void build_reed(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    uint32_t seed = 0xBEEDu;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float((seed >> 8) & 0xFFFFu) / 65535.0f; };
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    // Leaves: ~26 blades from a ~0.3 m wide base, outer ones lower and more arched.
    for (int i = 0; i < 26; ++i) {
        const float a = 6.2831853f * rnd();
        const float r = 0.16f * std::sqrt(rnd());
        const glm::vec3 base(std::cos(a) * r, -0.05f, std::sin(a) * r);
        const float outer = r / 0.16f;
        const float len = 1.25f + 0.85f * rnd() - 0.45f * outer;
        const glm::vec3 lean(std::cos(a) + (rnd() - 0.5f) * 0.8f, 0.0f, std::sin(a) + (rnd() - 0.5f) * 0.8f);
        reed_blade(v, ix, base, lean, len, 0.011f + 0.006f * rnd(), 0.25f + 0.65f * outer * rnd(), seed);
    }
    // Flowering stems with cattail heads.
    for (int i = 0; i < 4; ++i) {
        const float a = 6.2831853f * rnd(), r = 0.07f * rnd();
        const glm::vec3 b(std::cos(a) * r, -0.05f, std::sin(a) * r);
        const float H = 1.75f + 0.55f * rnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * (0.05f + 0.10f * rnd());
        std::vector<glm::vec3> path;
        for (int k = 0; k <= 8; ++k) {
            const float t = float(k) / 8.0f;
            path.push_back(b + up * (H * t) + lean * (H * t * t));
        }
        plant_tube(v, ix, path, 0.0065f, 0.0040f, 5, 0.0f, 1.0f);
        // Head: a 16-sided cylinder ~0.20 m long below the stem tip, rounded ends.
        const glm::vec3 hb = path[8] - up * 0.34f + lean * 0.2f, ht = hb + up * 0.20f;
        const float rr = 0.016f + 0.004f * rnd();
        std::vector<glm::vec3> hp = { hb - up * 0.012f, hb, hb + up * 0.06f, ht - up * 0.06f, ht, ht + up * 0.012f };
        const uint32_t h0 = uint32_t(v.size());
        const float rad[6] = { 0.45f, 0.92f, 1.0f, 1.0f, 0.92f, 0.45f };
        const int S = 16;
        for (int k = 0; k < 6; ++k)
            for (int s = 0; s < S; ++s) {
                const float ang = 6.2831853f * float(s) / float(S);
                const glm::vec3 d(std::cos(ang), 0.0f, std::sin(ang));
                const glm::vec3 nrm = glm::normalize(d + up * (k == 0 ? -0.8f : (k == 5 ? 0.8f : 0.0f)));
                v.push_back({ hp[k] + d * (rr * rad[k]), nrm, { 5.0f, 1.0f } });
            }
        for (int k = 0; k < 5; ++k)
            for (int s = 0; s < S; ++s) {
                const uint32_t a0 = h0 + uint32_t(k * S + s), a1 = h0 + uint32_t(k * S + (s + 1) % S);
                ix.insert(ix.end(), { a0, a1, a1 + uint32_t(S), a0, a1 + uint32_t(S), a0 + uint32_t(S) });
            }
        // Thin male spike above the head.
        std::vector<glm::vec3> sp = { ht + up * 0.02f, ht + up * 0.16f };
        plant_tube(v, ix, sp, 0.0030f, 0.0012f, 4, 1.0f, 1.0f);
    }
}

// ---- Ground cover (the low layer open-world games fill their ground
// with: clover, plantain, dandelion, nettle, foxglove, thistle, fungi).
// Kept low-poly per leaf - they are scattered by the ten thousand. ----

// Surface of revolution around `axis` through `base`: profile (r, y)
// pairs bottom to top, `part` written to every vertex (uv.y = 0: no sway).
void plant_lathe(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 base, glm::vec3 axis,
                 const std::vector<glm::vec2>& prof, int sides, float part) {
    const glm::vec3 up = glm::normalize(axis);
    const glm::vec3 t1 = glm::normalize(glm::cross(up, std::abs(up.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
    const glm::vec3 t2 = glm::cross(up, t1);
    const uint32_t b0 = uint32_t(v.size());
    const size_t n = prof.size();
    for (size_t i = 0; i < n; ++i) {
        // Profile tangent -> outward normal (dr, dy) rotated by -90 deg.
        glm::vec2 d = prof[std::min(i + 1, n - 1)] - prof[i > 0 ? i - 1 : 0];
        glm::vec2 pn = glm::normalize(glm::vec2(d.y, -d.x) + glm::vec2(1e-6f));
        for (int s = 0; s < sides; ++s) {
            const float a = 6.2831853f * float(s) / float(sides);
            const glm::vec3 radial = t1 * std::cos(a) + t2 * std::sin(a);
            v.push_back({ base + radial * prof[i].x + up * prof[i].y,
                          glm::normalize(radial * pn.x + up * pn.y), { part, 0.0f } });
        }
    }
    for (size_t i = 0; i + 1 < n; ++i)
        for (int s = 0; s < sides; ++s) {
            const uint32_t a = b0 + uint32_t(i) * uint32_t(sides) + uint32_t(s);
            const uint32_t b = b0 + uint32_t(i) * uint32_t(sides) + uint32_t((s + 1) % sides);
            ix.insert(ix.end(), { a, b, b + uint32_t(sides), a, b + uint32_t(sides), a + uint32_t(sides) });
        }
}

// Brown seed spike (plantain): a thin cylinder in part 5 (the cattail
// head material) from `a` to `b`.
void seed_spike(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 a, glm::vec3 b, float r) {
    const glm::vec3 ax = b - a;
    const float L = glm::length(ax);
    plant_lathe(v, ix, a, ax, { { r * 0.5f, 0.0f }, { r, L * 0.15f }, { r, L * 0.75f }, { r * 0.4f, L } }, 6, 5.0f);
    // plant_lathe writes sway 0; spikes sit high on a flexible stalk.
    for (size_t i = v.size() - 24; i < v.size(); ++i) v[i].uv.y = 1.0f;
}

uint32_t g_seed = 1u;
float grnd() { g_seed = g_seed * 1664525u + 1013904223u; return float((g_seed >> 8) & 0xFFFFu) / 65535.0f; }

// White clover (Trifolium repens): a creeping patch of trifoliate leaves
// on petioles of different lengths, a few round white flower heads.
void build_clover(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0xC10Fu;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < 11; ++i) {
        const float a = 6.2831853f * grnd(), r = 0.20f * std::sqrt(grnd());
        const glm::vec3 b(std::cos(a) * r, -0.01f, std::sin(a) * r);
        const float hgt = 0.035f + 0.075f * grnd();
        const glm::vec3 top = b + up * hgt + glm::vec3(grnd() - 0.5f, 0.0f, grnd() - 0.5f) * 0.03f;
        plant_tube(v, ix, { b, top }, 0.0016f, 0.0013f, 3, 0.0f, 0.6f);
        const float roll = 6.2831853f * grnd(), size = 0.018f + 0.012f * grnd();
        for (int k = 0; k < 3; ++k) {
            const float la = roll + 2.0944f * float(k);
            const glm::vec3 d = glm::normalize(glm::vec3(std::cos(la), 0.12f, std::sin(la)));
            plant_leaf(v, ix, top, d, size, size * 0.55f, size * 0.12f, 0.6f, 3, 2, 1.0f);
        }
    }
    for (int i = 0; i < 3; ++i) {                       // flower heads on taller stalks
        const float a = 6.2831853f * grnd(), r = 0.14f * std::sqrt(grnd());
        const glm::vec3 b(std::cos(a) * r, -0.01f, std::sin(a) * r);
        const glm::vec3 top = b + up * (0.10f + 0.05f * grnd());
        plant_tube(v, ix, { b, top }, 0.0016f, 0.0012f, 3, 0.0f, 1.0f);
        plant_head(v, ix, top, up, 14, 0.011f, 2.4f, 0.75f, grnd() * 6.0f, 1.0f, 2);
    }
}

// Greater plantain (Plantago major): a flat rosette of broad, ribbed,
// spoon-shaped leaves pressed to the ground and slim brown seed spikes.
void build_plantain(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0x91A7u;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const int n = 8;
    for (int i = 0; i < n; ++i) {
        const float a = 6.2831853f * (float(i) + 0.3f * grnd()) / float(n) + float(i % 2) * 0.3f;
        const glm::vec3 d = glm::normalize(glm::vec3(std::cos(a), 0.22f + 0.15f * grnd(), std::sin(a)));
        const float len = 0.13f + 0.07f * grnd();
        plant_leaf(v, ix, glm::vec3(0.0f, 0.005f, 0.0f), d, len, len * 0.36f, len * 0.08f, 0.2f, 5, 2, 0.85f);
    }
    for (int i = 0; i < 3; ++i) {
        const float a = 6.2831853f * grnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * (0.02f + 0.03f * grnd());
        const float H = 0.16f + 0.10f * grnd();
        const glm::vec3 b(0.0f, 0.0f, 0.0f), m = b + up * (H * 0.55f) + lean, t = b + up * H + lean * 1.8f;
        plant_tube(v, ix, { b, m }, 0.0018f, 0.0015f, 3, 0.0f, 0.6f);
        seed_spike(v, ix, m, t, 0.0035f);
    }
}

// Dandelion: a rosette of deeply toothed leaves, hollow stalks with
// yellow heads (many strap florets) and a closed bud.
void build_dandelion(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0xDA4Du;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < 8; ++i) {
        const float a = 6.2831853f * (float(i) + 0.4f * grnd()) / 8.0f;
        const glm::vec3 d = glm::normalize(glm::vec3(std::cos(a), 0.30f + 0.2f * grnd(), std::sin(a)));
        const float len = 0.12f + 0.06f * grnd();
        plant_leaf(v, ix, glm::vec3(0.0f, 0.004f, 0.0f), d, len, len * 0.17f, len * 0.10f, 0.25f, 10, 2, 0.0f, 4.0f);
    }
    for (int i = 0; i < 2; ++i) {
        const float a = 6.2831853f * grnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * 0.03f;
        const float H = 0.14f + 0.10f * grnd();
        const glm::vec3 m = up * (H * 0.5f) + lean * 0.4f, t = up * H + lean;
        plant_tube(v, ix, { glm::vec3(0.0f), m, t }, 0.0024f, 0.0020f, 4, 0.0f, 1.0f);
        const glm::vec3 face = glm::normalize(up + lean * 6.0f);
        if (i == 0) plant_head(v, ix, t, face, 24, 0.020f, 0.35f, 0.55f, grnd() * 6.0f, 1.0f, 2);
        else        plant_head(v, ix, t, face, 6, 0.010f, 2.6f, 0.9f, grnd() * 6.0f, 1.0f, 2);   // bud
    }
}

// Stinging nettle: a few upright square-ish stems with opposite,
// decussate pairs of heart-shaped, serrated leaves angled down, smaller
// toward the tip. Grows in clumps along forest edges.
void build_nettle(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0x2E77u;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    for (int s = 0; s < 4; ++s) {
        const float a = 6.2831853f * grnd(), r = 0.10f * std::sqrt(grnd());
        const glm::vec3 b(std::cos(a) * r, -0.02f, std::sin(a) * r);
        const float H = 0.45f + 0.35f * grnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * (0.04f + 0.05f * grnd());
        auto P = [&](float t) { return b + up * (H * t) + lean * (t * t); };
        std::vector<glm::vec3> path;
        for (int k = 0; k <= 6; ++k) path.push_back(P(float(k) / 6.0f));
        plant_tube(v, ix, path, 0.0035f, 0.0016f, 4, 0.0f, 1.0f);
        const int nodes = 6;
        float roll = grnd() * 3.14159f;
        for (int k = 1; k <= nodes; ++k) {
            const float t = float(k) / float(nodes + 1);
            const glm::vec3 p = P(t);
            const float len = (0.10f - 0.055f * t) * (0.85f + 0.3f * grnd());
            for (int side = 0; side < 2; ++side) {
                const float la = roll + 3.14159f * float(side);
                const glm::vec3 d = glm::normalize(glm::vec3(std::cos(la), -0.15f - 0.25f * t, std::sin(la)));
                plant_leaf(v, ix, p, d, len, len * 0.38f, len * 0.05f, t, 8, 2, 0.35f, 5.0f);
            }
            roll += 1.5708f;                              // decussate
        }
    }
}

// Mushrooms: a small cluster of capped fungi (stem + gills part 6, cap
// part 7 = instance colour). Rigid - no sway.
void build_mushrooms(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0xF0461u;
    const int n = 5;
    for (int i = 0; i < n; ++i) {
        const float a = 6.2831853f * grnd(), r = i == 0 ? 0.0f : 0.05f + 0.08f * grnd();
        const glm::vec3 b(std::cos(a) * r, -0.01f, std::sin(a) * r);
        const float s = (i == 0 ? 1.25f : 0.55f + 0.6f * grnd());
        const glm::vec3 ax = glm::normalize(glm::vec3(std::cos(a) * 0.18f * r / 0.13f, 1.0f, std::sin(a) * 0.18f * r / 0.13f));
        const float sh = 0.055f * s * (0.8f + 0.4f * grnd());   // stem height
        const float cr = 0.026f * s * (0.8f + 0.5f * grnd());   // cap radius
        // Stem: slightly bulbous foot, tapering up.
        plant_lathe(v, ix, b, ax, { { 0.0f, -0.004f }, { 0.0075f * s, 0.0f }, { 0.0068f * s, sh * 0.4f },
                                    { 0.0056f * s, sh } }, 8, 6.0f);
        // Gills: the cap underside, from the stem out to the rim.
        const glm::vec3 c = b + ax * sh;
        plant_lathe(v, ix, c, ax, { { 0.0056f * s, 0.0f }, { cr * 0.6f, -0.002f }, { cr, -0.004f } }, 12, 6.0f);
        // Cap: rim curled under, domed top (flatter on older ones).
        const float dome = 0.55f + 0.35f * grnd();
        plant_lathe(v, ix, c, ax, { { cr, -0.0042f }, { cr * 1.03f, 0.0f }, { cr * 0.93f, cr * 0.32f * dome },
                                    { cr * 0.68f, cr * 0.62f * dome }, { cr * 0.32f, cr * 0.80f * dome },
                                    { 0.0f, cr * 0.84f * dome } }, 12, 7.0f);
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
            params.road->distance(wx, wz) + 1.6f * road_edge_jitter(wx, wz) <
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
            // { height, nodes, leaves, leaf_len, leaf_w, palmate, branches, branch_len,
            //   petals, head_r, cup, petal_w, raceme, raceme_len, raceme_roll, pedicel_pitch, lean, seed }
            case 1: build_herb(pv, pi, { 0.34f, 5, 3, 0.17f, 0.030f, false, 2, 0.11f, 14, 0.058f, 0.05f, 0.70f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 3u }); break;   // daisy: branched, several heads
            case 2: build_herb(pv, pi, { 0.44f, 5, 3, 0.20f, 0.040f, false, 1, 0.13f, 5, 0.075f, 0.55f, 1.05f, 0, 0.3f, 137.5f, 62.0f, 0.25f, 5u }); break;   // poppy: cup + a bud
            case 3: build_herb(pv, pi, { 0.34f, 3, 3, 0.24f, 0.022f, false, 0, 0.10f, 6, 0.036f, -0.9f, 0.95f, 6, 0.50f, 12.0f, 118.0f, 2.6f, 7u }); break;   // bluebell: one-sided nodding raceme
            case 4: build_herb(pv, pi, { 0.48f, 6, 4, 0.16f, 0.018f, false, 2, 0.14f, 16, 0.050f, 0.20f, 0.45f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 9u }); break;   // cornflower: thin rays, branched
            case 5: build_herb(pv, pi, { 0.26f, 4, 3, 0.13f, 0.045f, false, 3, 0.09f, 5, 0.034f, 0.70f, 1.10f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 11u }); break;  // buttercup: loose cyme of small cups
            case 7: build_reed(pv, pi); break;
            case 8:  build_clover(pv, pi); break;
            case 9:  build_plantain(pv, pi); break;
            case 10: build_dandelion(pv, pi); break;
            case 11: build_nettle(pv, pi); break;
            // foxglove: tall one-sided spike of drooping bells; thistle: branched, purple tufts
            case 12: build_herb(pv, pi, { 0.95f, 4, 4, 0.24f, 0.060f, false, 0, 0.10f, 6, 0.030f, -1.2f, 1.0f, 14, 0.55f, 22.0f, 105.0f, 0.35f, 17u }); break;
            case 13: build_herb(pv, pi, { 0.62f, 6, 5, 0.17f, 0.030f, false, 2, 0.12f, 30, 0.026f, 2.4f, 0.35f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 19u }); break;
            case 14: build_mushrooms(pv, pi); break;
            case 6: build_herb(pv, pi, { 0.66f, 4, 4, 0.20f, 0.040f, true, 0, 0.10f, 5, 0.030f, 0.45f, 0.95f, 12, 0.48f, 137.5f, 58.0f, 0.0f, 13u }); break;  // lupin: palmate leaves, dense spike
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
                params.road->distance(wx, wz) + road_edge_jitter(wx, wz) < params.road->half_width + 0.4f) continue;
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
            float r = u01(rng) / std::max(params.flower_density, 0.01f);
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
        // Reed beds along the shore: dense clumps on the wet margin, from a
        // little below the waterline (standing in the shallows) to just above.
        {
            int reeds = 0;
            for (int i = 0; i < 600000 && reeds < 9000; ++i) {
                float wx = (u01(rng) * 2.0f - 1.0f) * ext;
                float wz = (u01(rng) * 2.0f - 1.0f) * ext;
                if (std::abs(wx) < params.keep_out_xz.x && std::abs(wz) < params.keep_out_xz.y) continue;
                if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
                    wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
                    params.road->distance(wx, wz) < params.road->half_width + params.road->shoulder + 1.0f) continue;
                float h = h_at(wx, wz);
                if (h < params.water_level - 0.35f || h > params.water_level + 0.9f) continue;
                if (n_y_at(wx, wz) < 0.72f) continue;
                // Beds, not a uniform fringe: low-frequency patches plus density
                // falling off with height above the water.
                float bed = hash31(glm::vec3(std::floor(wx * 0.11f), 31.0f, std::floor(wz * 0.11f))) * 0.6f +
                            hash31(glm::vec3(std::floor(wx * 0.37f), 37.0f, std::floor(wz * 0.37f))) * 0.4f;
                float wet = 1.0f - glm::smoothstep(params.water_level + 0.1f, params.water_level + 0.9f, h);
                if (bed < 0.42f || u01(rng) > wet * glm::clamp(params.flower_density, 0.0f, 2.0f)) continue;
                ScatterInstance si{};
                si.pos_scale = glm::vec4(wx, h - 0.03f, wz, 0.75f + 0.55f * u01(rng));
                glm::vec3 col = glm::mix(glm::vec3(0.32f, 0.38f, 0.16f), glm::vec3(0.42f, 0.40f, 0.20f), u01(rng));
                si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
                per[7].push_back(si);
                ++reeds;
            }
        }

        // ---- Ground cover. Ecology, the way open-world games layer it:
        // meadows get clover drifts, plantain and dandelions (plantain also
        // crowds trampled path edges); forest edges nettle clumps and
        // foxgloves; the shaded forest floor mushroom clusters; thistles
        // stand alone in rough grass. `cover` = how much tree crown is
        // overhead (0 open .. 1 under a canopy), from a 4 m grid. ----
        {
            const float gd = glm::clamp(params.ground_density, 0.0f, 3.0f);
            const int G = int(std::ceil(2.0f * ext / 4.0f));
            std::vector<float> cover(size_t(G) * size_t(G), 0.0f);
            if (params.trees) {
                for (const glm::vec3& t : *params.trees) {
                    const float R = t.z * 1.35f + 2.0f;
                    const int x0 = std::max(0, int((t.x - R + ext) / 4.0f)), x1 = std::min(G - 1, int((t.x + R + ext) / 4.0f));
                    const int z0 = std::max(0, int((t.y - R + ext) / 4.0f)), z1 = std::min(G - 1, int((t.y + R + ext) / 4.0f));
                    for (int gz = z0; gz <= z1; ++gz)
                        for (int gx = x0; gx <= x1; ++gx) {
                            const float cx = -ext + (float(gx) + 0.5f) * 4.0f, cz = -ext + (float(gz) + 0.5f) * 4.0f;
                            const float d = std::sqrt((cx - t.x) * (cx - t.x) + (cz - t.y) * (cz - t.y));
                            float& c = cover[size_t(gz) * size_t(G) + size_t(gx)];
                            c = std::min(1.0f, c + std::max(0.0f, 1.0f - d / R));
                        }
                }
            }
            auto cover_at = [&](float x, float z) {
                const float fx = (x + ext) / 4.0f - 0.5f, fz = (z + ext) / 4.0f - 0.5f;
                const int ix0 = std::clamp(int(std::floor(fx)), 0, G - 2), iz0 = std::clamp(int(std::floor(fz)), 0, G - 2);
                const float tx = glm::clamp(fx - float(ix0), 0.0f, 1.0f), tz = glm::clamp(fz - float(iz0), 0.0f, 1.0f);
                auto C = [&](int a, int b) { return cover[size_t(b) * size_t(G) + size_t(a)]; };
                return glm::mix(glm::mix(C(ix0, iz0), C(ix0 + 1, iz0), tx), glm::mix(C(ix0, iz0 + 1), C(ix0 + 1, iz0 + 1), tx), tz);
            };
            auto gpatch = [&](float x, float z, float f, float seed) {
                return hash31(glm::vec3(std::floor(x * f), seed, std::floor(z * f))) * 0.6f +
                       hash31(glm::vec3(std::floor(x * f * 3.1f), seed + 9.0f, std::floor(z * f * 3.1f))) * 0.4f;
            };
            const int tries = int(520000.0f * gd);
            for (int i = 0; i < tries; ++i) {
                const float wx = (u01(rng) * 2.0f - 1.0f) * ext;
                const float wz = (u01(rng) * 2.0f - 1.0f) * ext;
                if (std::abs(wx) < params.keep_out_xz.x && std::abs(wz) < params.keep_out_xz.y) continue;
                float road_d = 1e9f;
                if (params.road && wx > params.road->bb_min.x - 4.0f && wx < params.road->bb_max.x + 4.0f &&
                    wz > params.road->bb_min.y - 4.0f && wz < params.road->bb_max.y + 4.0f)
                    road_d = params.road->distance(wx, wz) + road_edge_jitter(wx, wz) - params.road->half_width;
                if (road_d < 0.35f) continue;                                      // never on the paving
                const float h = h_at(wx, wz);
                if (h < params.water_level + 0.5f || h > params.rock_band_h + 3.0f) continue;
                if (n_y_at(wx, wz) < 0.80f) continue;
                const float cv = cover_at(wx, wz);
                const float edge = std::max(0.0f, 1.0f - std::abs(cv - 0.35f) / 0.35f);   // forest margin, peaks at cover 0.35
                const float verge = 1.0f - glm::smoothstep(0.4f, 3.0f, road_d);           // trampled path edge
                const float r = u01(rng);
                int sp = -1;
                // Weighted pick (probability per try; the try count carries
                // the density slider): the weights encode where each grows.
                float w[7];
                w[0] = (cv < 0.30f && gpatch(wx, wz, 0.07f, 21.0f) > 0.45f) ? 0.40f : 0.0f;                 // clover drifts
                w[1] = (gpatch(wx, wz, 0.09f, 22.0f) > 0.35f ? 0.06f : 0.015f) * (1.0f - cv) + 0.30f * verge;  // plantain (+ path edges)
                w[2] = (cv < 0.35f && gpatch(wx, wz, 0.06f, 23.0f) > 0.30f) ? 0.05f : 0.0f;               // dandelion
                w[3] = gpatch(wx, wz, 0.12f, 24.0f) > 0.45f ? 0.07f * edge : 0.0f;                        // nettle clumps
                w[4] = 0.014f * edge;                                                                   // foxglove
                w[5] = cv < 0.2f ? 0.0035f : 0.0f;                                                      // lone thistles
                w[6] = (cv > 0.45f && gpatch(wx, wz, 0.15f, 25.0f) > 0.58f) ? 0.035f * cv : 0.0f;          // fungi
                float acc = 0.0f;
                for (int k = 0; k < 7; ++k) { acc += w[k]; if (r < acc) { sp = 8 + k; break; } }
                if (sp < 0) continue;
                ScatterInstance si{};
                float sc = 0.85f + 0.4f * u01(rng);
                if (sp == 8) sc = 0.8f + 0.5f * u01(rng);
                if (sp == 11 || sp == 12) sc = 0.8f + 0.45f * u01(rng);
                si.pos_scale = glm::vec4(wx, h - (sp == 14 ? 0.01f : 0.015f), wz, sc);
                glm::vec3 col;
                switch (sp) {
                case 8:  col = u01(rng) < 0.8f ? glm::vec3(0.88f, 0.86f, 0.80f) : glm::vec3(0.85f, 0.55f, 0.65f); break;   // white / pink clover
                case 10: col = glm::vec3(0.98f, 0.74f, 0.04f) * (0.9f + 0.15f * u01(rng)); break;
                case 12: col = glm::mix(glm::vec3(0.80f, 0.32f, 0.62f), glm::vec3(0.62f, 0.25f, 0.70f), u01(rng)); break;
                case 13: col = glm::vec3(0.58f, 0.22f, 0.62f) * (0.85f + 0.3f * u01(rng)); break;
                case 14: {
                    const float k = u01(rng);
                    col = k < 0.08f ? glm::vec3(0.72f, 0.10f, 0.05f)                        // the odd red one
                        : glm::mix(glm::vec3(0.50f, 0.33f, 0.18f), glm::vec3(0.68f, 0.56f, 0.40f), u01(rng));
                    break;
                }
                default: col = glm::vec3(0.6f, 0.6f, 0.6f); break;                          // leaf-only: tint unused
                }
                si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
                per[sp].push_back(si);
            }
        }

        // Lift for the raymarched ground relief (raymarch_ground_relief):
        // packed as whole turns of the Y rotation (4 mm steps), which the
        // vertex shaders read back with floor(rot / 2 pi) and fade with
        // distance exactly like the relief itself (10 - 26 m). Without it the
        // relief (~0.17 m on average) swallowed clover and plantain close by.
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp)
            for (ScatterInstance& si : per[sp]) {
                const float x = si.pos_scale.x, z = si.pos_scale.z;
                float d = raymarch_ground_relief(x, z, h_at(x, z), params.water_level);
                if (params.road && x > params.road->bb_min.x - 2.0f && x < params.road->bb_max.x + 2.0f &&
                    z > params.road->bb_min.y - 2.0f && z < params.road->bb_max.y + 2.0f)
                    d *= glm::smoothstep(params.road->half_width + 0.1f, params.road->half_width + 0.9f,
                                         params.road->distance(x, z));
                const float q = std::clamp(std::floor(d / 0.004f + 0.5f), 0.0f, 120.0f);
                si.rot_tint.x = std::fmod(si.rot_tint.x, glm::two_pi<float>()) + glm::two_pi<float>() * q;
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
        log::infof("[scatter] plants: %u ferns, %u daisies, %u poppies, %u bluebells, %u cornflowers, %u buttercups, %u lupins, %u reeds",
                   s.plant_count[0], s.plant_count[1], s.plant_count[2], s.plant_count[3],
                   s.plant_count[4], s.plant_count[5], s.plant_count[6], s.plant_count[7]);
        log::infof("[scatter] ground: %u clover, %u plantain, %u dandelion, %u nettle, %u foxglove, %u thistle, %u fungi",
                   s.plant_count[8], s.plant_count[9], s.plant_count[10], s.plant_count[11],
                   s.plant_count[12], s.plant_count[13], s.plant_count[14]);
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
