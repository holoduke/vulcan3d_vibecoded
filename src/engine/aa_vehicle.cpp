#include "engine/aa_vehicle.h"

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace qlike {
namespace {

using game::LevelMesh;
using game::LevelVertex;
constexpr float kPi = 3.14159265358979f;

void push(LevelMesh& m, const LevelVertex& v) {
    m.verts.push_back(v);
    m.bmin = glm::min(m.bmin, v.p);
    m.bmax = glm::max(m.bmax, v.p);
}
// Planar UV (metres) from the dominant normal axis.
glm::vec2 puv(glm::vec3 p, glm::vec3 n) {
    glm::vec3 a = glm::abs(n);
    if (a.x >= a.y && a.x >= a.z) return { p.z, -p.y };
    if (a.y >= a.z) return { p.x, p.z };
    return { p.x, -p.y };
}
// Flat polygon (3 or 4 points) with winding fixed against `n`.
void poly(LevelMesh& m, std::initializer_list<glm::vec3> pts, glm::vec3 n) {
    std::vector<glm::vec3> p(pts);
    glm::vec3 g = glm::cross(p[1] - p[0], p[2] - p[0]);
    if (glm::dot(g, g) < 1e-12f && p.size() == 4) g = glm::cross(p[2] - p[0], p[3] - p[0]);
    if (glm::dot(g, g) < 1e-12f) return;
    if (glm::dot(n, n) < 1e-12f) n = g;
    n = glm::normalize(n);
    const bool flip = glm::dot(g, n) < 0.0f;
    uint32_t b = uint32_t(m.verts.size());
    for (const auto& q : p) { glm::vec2 uv = puv(q, n); push(m, { q, n, uv.x, uv.y }); }
    auto tri = [&](uint32_t i, uint32_t j, uint32_t k) {
        if (flip) m.idx.insert(m.idx.end(), { b + i, b + k, b + j });
        else      m.idx.insert(m.idx.end(), { b + i, b + j, b + k });
    };
    tri(0, 1, 2);
    if (p.size() == 4) tri(0, 2, 3);
}

// Chamfered box (bevel b) transformed by xf (rigid + translation).
void bbox(LevelMesh& m, const glm::mat4& xf, glm::vec3 h, float b) {
    b = std::min(b, std::min(h.x, std::min(h.y, h.z)) * 0.45f);
    auto P = [&](int ax, glm::vec3 s) {
        glm::vec3 q = s * (h - glm::vec3(b));
        q[ax] = s[ax] * h[ax];
        return glm::vec3(xf * glm::vec4(q, 1.0f));
    };
    const glm::mat3 R(xf);
    auto N = [&](glm::vec3 n) { return glm::normalize(R * n); };
    // Faces.
    for (int ax = 0; ax < 3; ++ax)
        for (float sgn : { -1.0f, 1.0f }) {
            int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
            glm::vec3 c[4];
            const float s1[4] = { -1, 1, 1, -1 }, s2[4] = { -1, -1, 1, 1 };
            for (int k = 0; k < 4; ++k) {
                glm::vec3 s(0.0f); s[ax] = sgn; s[a1] = s1[k]; s[a2] = s2[k];
                c[k] = P(ax, s);
            }
            glm::vec3 n(0.0f); n[ax] = sgn;
            poly(m, { c[0], c[1], c[2], c[3] }, N(n));
        }
    // Edge chamfers.
    for (int ax = 0; ax < 3; ++ax) {
        int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;   // edge runs along ax
        for (float s1 : { -1.0f, 1.0f })
            for (float s2 : { -1.0f, 1.0f }) {
                glm::vec3 lo(0.0f), hi(0.0f);
                lo[ax] = -1; hi[ax] = 1; lo[a1] = hi[a1] = s1; lo[a2] = hi[a2] = s2;
                glm::vec3 n(0.0f); n[a1] = s1; n[a2] = s2;
                poly(m, { P(a1, lo), P(a1, hi), P(a2, hi), P(a2, lo) }, N(n));
            }
    }
    // Corners.
    for (float sx : { -1.0f, 1.0f })
        for (float sy : { -1.0f, 1.0f })
            for (float sz : { -1.0f, 1.0f }) {
                glm::vec3 s(sx, sy, sz);
                poly(m, { P(0, s), P(1, s), P(2, s) }, N(s));
            }
}
// Hexahedron from 8 corners: 0-3 bottom (-z,-x),(+x),(+x,+z),(-x,+z); 4-7 top same order.
void hexa(LevelMesh& m, const glm::mat4& xf, const glm::vec3 c[8]) {
    glm::vec3 w[8];
    glm::vec3 ctr(0.0f);
    for (int i = 0; i < 8; ++i) { w[i] = glm::vec3(xf * glm::vec4(c[i], 1.0f)); ctr += w[i]; }
    ctr /= 8.0f;
    const int f[6][4] = { {0,1,2,3}, {4,5,6,7}, {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7} };
    for (const auto& q : f) {
        glm::vec3 fc = (w[q[0]] + w[q[1]] + w[q[2]] + w[q[3]]) * 0.25f;
        glm::vec3 n = glm::cross(w[q[1]] - w[q[0]], w[q[2]] - w[q[0]]);
        if (glm::dot(n, n) < 1e-10f) n = glm::cross(w[q[2]] - w[q[0]], w[q[3]] - w[q[0]]);
        if (glm::dot(n, fc - ctr) < 0.0f) n = -n;
        poly(m, { w[q[0]], w[q[1]], w[q[2]], w[q[3]] }, n);
    }
}
// Cylinder along local +Z from z0 to z1 (smooth sides, flat caps).
void cyl(LevelMesh& m, const glm::mat4& xf, float r, float z0, float z1, int segs, bool caps = true) {
    const glm::mat3 R(xf);
    uint32_t b = uint32_t(m.verts.size());
    for (int i = 0; i <= segs; ++i) {
        float a = 2.0f * kPi * float(i) / float(segs);
        glm::vec3 d(std::cos(a), std::sin(a), 0.0f);
        glm::vec3 n = glm::normalize(R * d);
        float u = a * r;
        push(m, { glm::vec3(xf * glm::vec4(d * r + glm::vec3(0, 0, z0), 1)), n, u, z0 });
        push(m, { glm::vec3(xf * glm::vec4(d * r + glm::vec3(0, 0, z1), 1)), n, u, z1 });
    }
    for (int i = 0; i < segs; ++i) {
        uint32_t k = b + uint32_t(i) * 2;
        // outward winding: (k, k+2, k+1) is CCW seen from outside for +Z axis
        m.idx.insert(m.idx.end(), { k, k + 1, k + 2, k + 1, k + 3, k + 2 });
    }
    // Fix winding against the radial normal of the first quad.
    {
        size_t q = m.idx.size() - size_t(segs) * 6;
        glm::vec3 p0 = m.verts[m.idx[q]].p, p1 = m.verts[m.idx[q + 1]].p, p2 = m.verts[m.idx[q + 2]].p;
        if (glm::dot(glm::cross(p1 - p0, p2 - p0), m.verts[m.idx[q]].n) < 0.0f)
            for (size_t t = q; t < m.idx.size(); t += 3) std::swap(m.idx[t + 1], m.idx[t + 2]);
    }
    if (caps) {
        for (float z : { z0, z1 }) {
            glm::vec3 n = glm::normalize(R * glm::vec3(0, 0, z == z0 ? -1.0f : 1.0f));
            for (int i = 0; i < segs; ++i) {
                float a0 = 2.0f * kPi * float(i) / float(segs), a1 = 2.0f * kPi * float(i + 1) / float(segs);
                poly(m, { glm::vec3(xf * glm::vec4(0, 0, z, 1)),
                          glm::vec3(xf * glm::vec4(std::cos(a0) * r, std::sin(a0) * r, z, 1)),
                          glm::vec3(xf * glm::vec4(std::cos(a1) * r, std::sin(a1) * r, z, 1)) }, n);
            }
        }
    }
}
glm::mat4 T(glm::vec3 t) { return glm::translate(glm::mat4(1.0f), t); }
glm::mat4 RY(float a) { return glm::rotate(glm::mat4(1.0f), a, glm::vec3(0, 1, 0)); }
glm::mat4 RX(float a) { return glm::rotate(glm::mat4(1.0f), a, glm::vec3(1, 0, 0)); }
// Rotate local +Z onto +X (wheel axles).
glm::mat4 ZtoX() { return RY(kPi * 0.5f); }

LevelMesh mat_metal() {
    LevelMesh m; m.tex_albedo = 3; m.tex_normal = 3; m.uv_scale = 0.6f;
    m.tint = glm::vec3(0.26f, 0.33f, 0.20f); m.fallback = glm::vec3(0.20f, 0.26f, 0.15f);
    return m;
}
LevelMesh mat_dark() {
    LevelMesh m; m.tex_albedo = 3; m.tex_normal = 3; m.uv_scale = 1.2f;
    m.tint = glm::vec3(0.16f, 0.16f, 0.15f); m.fallback = glm::vec3(0.08f);
    return m;
}
LevelMesh mat_rubber() {
    LevelMesh m; m.tint = glm::vec3(0.035f, 0.035f, 0.035f); m.fallback = m.tint;
    return m;
}

// Tracked hull shared by the gun and missile vehicles.
void build_hull(AaModel& M) {
    LevelMesh hm = mat_metal(), hd = mat_dark(), hr = mat_rubber();
    const glm::mat4 I(1.0f);
    {   // lower hull with sloped front / rear
        glm::vec3 c[8] = { {-1.30f, 0.42f, -2.95f}, {1.30f, 0.42f, -2.95f}, {1.30f, 0.42f, 2.85f}, {-1.30f, 0.42f, 2.85f},
                           {-1.45f, 1.28f, -3.40f}, {1.45f, 1.28f, -3.40f}, {1.45f, 1.28f, 3.55f}, {-1.45f, 1.28f, 3.55f} };
        hexa(hm, I, c);
    }
    {   // upper glacis + deck
        glm::vec3 c[8] = { {-1.45f, 1.28f, 1.70f}, {1.45f, 1.28f, 1.70f}, {1.45f, 1.28f, 3.55f}, {-1.45f, 1.28f, 3.55f},
                           {-1.38f, 1.62f, 1.70f}, {1.38f, 1.62f, 1.70f}, {1.38f, 1.62f, 2.45f}, {-1.38f, 1.62f, 2.45f} };
        hexa(hm, I, c);
    }
    bbox(hm, T({0.0f, 1.44f, -0.85f}), {1.42f, 0.18f, 2.55f}, 0.06f);       // rear deck
    bbox(hm, T({0.0f, 1.28f, -3.45f}), {1.35f, 0.25f, 0.10f}, 0.04f);        // rear plate
    for (float s : { -1.0f, 1.0f }) {
        bbox(hm, T({s * 1.62f, 1.31f, 0.08f}), {0.30f, 0.025f, 3.55f}, 0.012f); // fenders
        bbox(hm, T({s * 1.91f, 1.16f, 0.15f}), {0.025f, 0.15f, 3.15f}, 0.01f);  // side skirt
        bbox(hd, T({s * 0.62f, 1.40f, 3.48f}), {0.08f, 0.07f, 0.06f}, 0.02f);   // headlight housing
        cyl(hd, T({s * 0.62f, 1.40f, 3.54f}), 0.055f, 0.0f, 0.03f, 12);        // lens
        bbox(hd, T({s * 0.95f, 0.70f, 3.58f}), {0.08f, 0.10f, 0.06f}, 0.02f);   // tow hooks
        bbox(hd, T({s * 0.95f, 0.70f, -3.50f}), {0.08f, 0.10f, 0.06f}, 0.02f);
        bbox(hm, T({s * 1.62f, 1.40f, -2.75f}), {0.22f, 0.08f, 0.45f}, 0.03f);  // stowage bins
        cyl(hd, T({s * 0.85f, 1.16f, -3.52f}), 0.09f, -0.08f, 0.04f, 12);     // exhausts
    }
    for (int i = 0; i < 6; ++i)                                                // engine grilles
        bbox(hd, T({-0.55f + 0.22f * float(i), 1.635f, -2.15f}), {0.07f, 0.012f, 0.65f}, 0.01f);
    for (int i = 0; i < 2; ++i)                                                // jerry cans
        bbox(hm, T({-0.35f + 0.7f * float(i), 1.20f, -3.62f}), {0.17f, 0.22f, 0.07f}, 0.03f);

    // Running gear: road wheels, sprocket, idler, return rollers, track.
    for (float s : { -1.0f, 1.0f }) {
        const float x = s * 1.57f;
        for (int w = 0; w < 6; ++w) {
            float z = -2.45f + 0.98f * float(w);
            cyl(hr, T({x, 0.47f, z}) * ZtoX(), 0.33f, -0.24f, 0.24f, 20);
            cyl(hd, T({x, 0.47f, z}) * ZtoX(), 0.22f, -0.26f, 0.26f, 16);     // hubs
        }
        cyl(hd, T({x, 0.66f, 3.02f}) * ZtoX(), 0.33f, -0.22f, 0.22f, 18);     // sprocket
        for (int t = 0; t < 10; ++t) {                                          // sprocket teeth
            float a = 2.0f * kPi * float(t) / 10.0f;
            bbox(hd, T({x, 0.66f + std::sin(a) * 0.37f, 3.02f + std::cos(a) * 0.37f}) * RX(-a),
                 {0.2f, 0.035f, 0.05f}, 0.01f);
        }
        cyl(hd, T({x, 0.66f, -3.02f}) * ZtoX(), 0.30f, -0.22f, 0.22f, 18);    // idler
        for (int r = 0; r < 3; ++r)
            cyl(hd, T({x, 1.02f, -1.6f + 1.6f * float(r)}) * ZtoX(), 0.09f, -0.12f, 0.12f, 10);
        // Track loop: straight runs + arcs around sprocket / idler.
        std::vector<glm::vec3> path;   // (y, z, tangent angle)
        const float R = 0.53f, yc = 0.61f, zf = 3.02f, zb = -3.02f;
        auto add_seg = [&](glm::vec2 a, glm::vec2 b) {
            float L = glm::length(b - a); int n = std::max(1, int(L / 0.16f));
            for (int i = 0; i < n; ++i) {
                glm::vec2 p = a + (b - a) * (float(i) + 0.5f) / float(n);
                path.push_back({ p.x, p.y, std::atan2(b.x - a.x, b.y - a.y) });
            }
        };
        add_seg({yc - R, zb}, {yc - R, zf});
        for (int i = 0; i < 10; ++i) {
            float a = -kPi * 0.5f + kPi * (float(i) + 0.5f) / 10.0f;
            path.push_back({ yc + std::sin(a) * R, zf + std::cos(a) * R, a + kPi * 0.5f });
        }
        add_seg({yc + R, zf}, {yc + R, zb});
        for (int i = 0; i < 10; ++i) {
            float a = kPi * 0.5f + kPi * (float(i) + 0.5f) / 10.0f;
            path.push_back({ yc + std::sin(a) * R, zb + std::cos(a) * R, a + kPi * 0.5f });
        }
        for (const auto& p : path) {
            glm::mat4 xf = T({x, p.x, p.y}) * RX(-p.z);
            bbox(hr, xf, {0.27f, 0.035f, 0.068f}, 0.012f);                      // link
            bbox(hd, xf * T({0.0f, 0.0f, 0.0f}), {0.06f, 0.045f, 0.03f}, 0.008f); // guide horn
        }
    }
    M.meshes[AaModel::kHull] = { hm, hd, hr };
}

}  // namespace

AaModel build_aa_model() {
    AaModel M;
    build_hull(M);
    const glm::mat4 I(1.0f);

    // ================= TURRET =================
    M.turret_pivot = { 0.0f, 1.62f, -0.30f };
    LevelMesh tm = mat_metal(), td = mat_dark();
    {
        glm::vec3 c[8] = { {-1.20f, 0.0f, -1.55f}, {1.20f, 0.0f, -1.55f}, {1.20f, 0.0f, 1.55f}, {-1.20f, 0.0f, 1.55f},
                           {-1.00f, 0.95f, -1.35f}, {1.00f, 0.95f, -1.35f}, {1.00f, 0.95f, 1.05f}, {-1.00f, 0.95f, 1.05f} };
        hexa(tm, I, c);
    }
    {   // sloped front cheek
        glm::vec3 c[8] = { {-1.10f, 0.05f, 1.55f}, {1.10f, 0.05f, 1.55f}, {1.10f, 0.05f, 1.95f}, {-1.10f, 0.05f, 1.95f},
                           {-0.95f, 0.85f, 1.05f}, {0.95f, 0.85f, 1.05f}, {0.95f, 0.55f, 1.65f}, {-0.95f, 0.55f, 1.65f} };
        hexa(tm, I, c);
    }
    bbox(tm, T({0.0f, 0.55f, -1.85f}), {0.95f, 0.36f, 0.40f}, 0.06f);            // bustle
    cyl(tm, T({0.0f, -0.05f, 0.0f}) * RX(-kPi * 0.5f), 1.30f, 0.0f, 0.12f, 32);   // turret ring
    cyl(tm, T({0.42f, 0.95f, -0.35f}) * RX(-kPi * 0.5f), 0.36f, 0.0f, 0.24f, 20); // cupola
    cyl(td, T({0.42f, 1.19f, -0.35f}) * RX(-kPi * 0.5f), 0.30f, 0.0f, 0.05f, 20); // hatch
    for (int i = 0; i < 6; ++i) {                                                  // vision blocks
        float a = 2.0f * kPi * float(i) / 6.0f;
        bbox(td, T({0.42f + std::cos(a) * 0.37f, 1.08f, -0.35f + std::sin(a) * 0.37f}) * RY(-a),
             {0.03f, 0.04f, 0.07f}, 0.01f);
    }
    bbox(tm, T({-0.45f, 1.0f, 0.15f}), {0.32f, 0.05f, 0.35f}, 0.03f);              // loader hatch
    // Tracking radar (front dish).
    cyl(td, T({0.0f, 0.62f, 1.82f}), 0.42f, 0.0f, 0.18f, 28);
    cyl(tm, T({0.0f, 0.62f, 2.00f}), 0.06f, 0.0f, 0.22f, 10);
    // Radar mast.
    cyl(td, T({0.0f, 0.90f, -1.80f}) * RX(-kPi * 0.5f), 0.10f, 0.0f, 0.55f, 12);
    // Smoke dischargers.
    for (float s : { -1.0f, 1.0f })
        for (int k = 0; k < 3; ++k)
            cyl(td, T({s * 0.80f, 0.55f + 0.14f * float(k), -1.55f}) * RY(s * 2.4f), 0.06f, 0.0f, 0.28f, 10);
    // Antennas.
    cyl(td, T({-0.75f, 0.95f, -1.75f}) * RX(-kPi * 0.5f), 0.012f, 0.0f, 2.2f, 6, false);
    cyl(td, T({0.75f, 0.95f, -1.75f}) * RX(-kPi * 0.5f), 0.012f, 0.0f, 1.6f, 6, false);
    M.meshes[AaModel::kTurret] = { tm, td };

    // ================= GUNS (pitch) =================
    M.gun_pivot = { 0.0f, 0.55f, 0.35f };
    LevelMesh gm = mat_metal(), gd = mat_dark();
    for (float s : { -1.0f, 1.0f }) {
        const float x = s * 1.52f;
        bbox(gm, T({x, 0.0f, 0.25f}), {0.24f, 0.34f, 1.15f}, 0.05f);           // gun housing
        bbox(gm, T({x - s * 0.30f, 0.0f, 0.0f}), {0.10f, 0.18f, 0.25f}, 0.03f); // trunnion arm
        bbox(gd, T({x + s * 0.24f, -0.05f, 0.1f}), {0.03f, 0.22f, 0.6f}, 0.01f); // ammo chute
        cyl(gd, T({x, 0.05f, 1.35f}), 0.11f, 0.0f, 0.95f, 16);                 // cooling jacket
        cyl(gd, T({x, 0.05f, 1.35f}), 0.06f, 0.95f, 3.55f, 14);                // barrel
        cyl(gd, T({x, 0.05f, 4.50f}), 0.10f, 0.0f, 0.28f, 14);                 // muzzle brake
        for (int k = 0; k < 4; ++k)                                              // brake ports
            bbox(gd, T({x, 0.05f, 4.57f + 0.06f * float(k)}), {0.115f, 0.02f, 0.012f}, 0.004f);
        M.muzzle[s < 0 ? 0 : 1] = { x, 0.05f, 4.80f };
    }
    M.meshes[AaModel::kGuns] = { gm, gd };

    // ================= SEARCH RADAR (spins) =================
    M.radar_pivot = { 0.0f, 1.48f, -1.80f };
    LevelMesh rm = mat_metal(), rd = mat_dark();
    bbox(rm, RX(-0.25f), {0.85f, 0.32f, 0.05f}, 0.03f);                          // antenna panel
    bbox(rd, T({0.0f, -0.05f, 0.08f}), {0.12f, 0.08f, 0.1f}, 0.02f);             // feed
    cyl(rd, RX(-kPi * 0.5f) * T({0.0f, 0.0f, -0.12f}), 0.08f, 0.0f, 0.12f, 10);  // rotor
    M.meshes[AaModel::kRadar] = { rm, rd };
    return M;
}

namespace {

// Lathe along local +Z: rings (z, radius), smooth sides.
void lathe_z(LevelMesh& m, const glm::mat4& xf, std::initializer_list<glm::vec2> prof, int segs) {
    std::vector<glm::vec2> p(prof);
    for (size_t i = 0; i + 1 < p.size(); ++i) {
        const float z0 = p[i].x, r0 = p[i].y, z1 = p[i + 1].x, r1 = p[i + 1].y;
        const float dz = z1 - z0, dr = r1 - r0;
        // Outward normal of the cone segment: (dz, -dr) in (radial, z).
        glm::vec2 nrz = glm::normalize(glm::vec2(dz, -dr) + glm::vec2(1e-6f, 0.0f));
        const glm::mat3 R(xf);
        uint32_t b = uint32_t(m.verts.size());
        for (int s = 0; s <= segs; ++s) {
            float a = 2.0f * kPi * float(s) / float(segs);
            glm::vec3 d(std::cos(a), std::sin(a), 0.0f);
            glm::vec3 n = glm::normalize(R * (d * nrz.x + glm::vec3(0, 0, nrz.y)));
            push(m, { glm::vec3(xf * glm::vec4(d * r0 + glm::vec3(0, 0, z0), 1)), n, a * 0.2f, z0 });
            push(m, { glm::vec3(xf * glm::vec4(d * r1 + glm::vec3(0, 0, z1), 1)), n, a * 0.2f, z1 });
        }
        for (int s = 0; s < segs; ++s) {
            uint32_t k = b + uint32_t(s) * 2;
            glm::vec3 p0 = m.verts[k].p, p1 = m.verts[k + 1].p, p2 = m.verts[k + 2].p, p3 = m.verts[k + 3].p;
            glm::vec3 g = glm::cross(p1 - p0, p2 - p0);
            if (glm::dot(g, g) < 1e-14f) g = glm::cross(p3 - p1, p2 - p1);
            if (glm::dot(g, m.verts[k].n) >= 0.0f) m.idx.insert(m.idx.end(), { k, k + 1, k + 2, k + 1, k + 3, k + 2 });
            else                                   m.idx.insert(m.idx.end(), { k, k + 2, k + 1, k + 1, k + 2, k + 3 });
        }
    }
}

LevelMesh mat_missile() {
    LevelMesh m; m.tint = glm::vec3(0.62f, 0.64f, 0.60f); m.fallback = m.tint;
    return m;
}

}  // namespace

// Medium-range SAM launcher in the style of a tracked TELAR: the same
// hull, a turntable with a nose radome (fire-control radar), and an
// elevating launch frame carrying four missiles side by side.
//   turret = turntable + radome (yaw)
//   guns   = launch frame (pitch about the rear trunnion)
//   radar  = small rotating antenna
AaModel build_sam_model() {
    AaModel M;
    build_hull(M);
    const glm::mat4 I(1.0f);

    // ---- Turntable ----
    M.turret_pivot = { 0.0f, 1.62f, -0.35f };
    LevelMesh tm = mat_metal(), td = mat_dark();
    cyl(tm, T({0.0f, -0.05f, 0.0f}) * RX(-kPi * 0.5f), 1.38f, 0.0f, 0.16f, 36);     // ring
    {   // platform deck, tapered toward the front
        glm::vec3 c[8] = { {-1.42f, 0.11f, -2.55f}, {1.42f, 0.11f, -2.55f}, {1.05f, 0.11f, 2.35f}, {-1.05f, 0.11f, 2.35f},
                           {-1.36f, 0.42f, -2.50f}, {1.36f, 0.42f, -2.50f}, {0.98f, 0.42f, 2.25f}, {-0.98f, 0.42f, 2.25f} };
        hexa(tm, I, c);
    }
    // Radome: drum with a domed face, on a short neck at the front.
    bbox(tm, T({0.0f, 0.62f, 1.55f}), {0.55f, 0.22f, 0.45f}, 0.06f);
    lathe_z(tm, T({0.0f, 1.05f, 1.30f}),
            { {0.0f, 0.0f}, {0.0f, 0.74f}, {0.95f, 0.74f}, {1.20f, 0.66f}, {1.38f, 0.48f}, {1.48f, 0.24f}, {1.52f, 0.0f} }, 32);
    // Equipment cabinets beside / behind the radome.
    for (float s : { -1.0f, 1.0f }) {
        bbox(tm, T({s * 1.02f, 0.72f, -0.20f}), {0.30f, 0.30f, 1.20f}, 0.05f);
        bbox(td, T({s * 1.33f, 0.74f, -0.20f}), {0.012f, 0.22f, 0.50f}, 0.006f);    // louvre panel
        bbox(tm, T({s * 1.02f, 0.60f, -1.95f}), {0.30f, 0.18f, 0.45f}, 0.04f);
        cyl(td, T({s * 0.95f, 1.02f, 0.60f}) * RX(-kPi * 0.5f), 0.012f, 0.0f, 1.9f, 6, false);   // whip antennas
    }
    // Trunnion towers for the launch frame (rear).
    for (float s : { -1.0f, 1.0f })
        bbox(tm, T({s * 0.62f, 0.78f, -1.55f}), {0.11f, 0.36f, 0.30f}, 0.03f);
    cyl(td, T({0.0f, 1.02f, -1.55f}) * ZtoX(), 0.09f, -0.78f, 0.78f, 14);             // trunnion axle
    M.meshes[AaModel::kTurret] = { tm, td };

    // ---- Launch frame (pitch) ----
    M.gun_pivot = { 0.0f, 1.02f, -1.55f };
    LevelMesh gm = mat_metal(), gd = mat_dark();
    const float rail_x[4] = { -1.02f, -0.34f, 0.34f, 1.02f };
    // Spine box + cross beams; each missile sits on its own launch rail.
    bbox(gm, T({0.0f, 0.02f, 0.95f}), {0.36f, 0.14f, 1.75f}, 0.04f);
    for (float z : { -0.35f, 0.9f, 2.3f })
        bbox(gm, T({0.0f, 0.08f, z}), {1.18f, 0.06f, 0.09f}, 0.02f);
    for (int k = 0; k < 4; ++k) {
        bbox(gd, T({rail_x[k], 0.20f, 1.35f}), {0.05f, 0.05f, 2.55f}, 0.012f);       // rail
        bbox(gd, T({rail_x[k], 0.14f, -0.95f}), {0.09f, 0.10f, 0.12f}, 0.02f);       // rear latch
        bbox(gd, T({rail_x[k], 0.14f, 2.60f}), {0.07f, 0.08f, 0.10f}, 0.02f);        // front shoe
        M.rail[k] = { rail_x[k], 0.47f, -1.05f };                                       // missile tail centre
    }
    M.meshes[AaModel::kGuns] = { gm, gd };

    // ---- Small rotating antenna on the cabinet roof ----
    M.radar_pivot = { -1.02f, 1.10f, -0.9f };
    LevelMesh rm = mat_metal(), rd = mat_dark();
    bbox(rm, RX(-0.15f) * T({0.0f, 0.12f, 0.0f}), {0.32f, 0.10f, 0.03f}, 0.015f);
    cyl(rd, RX(-kPi * 0.5f) * T({0.0f, 0.0f, -0.08f}), 0.05f, 0.0f, 0.12f, 10);
    M.meshes[AaModel::kRadar] = { rm, rd };

    // ---- Missile (tail at origin, nose toward +Z; 5.5 m) ----
    LevelMesh mm = mat_missile(), md = mat_dark();
    lathe_z(mm, I, { {0.0f, 0.0f}, {0.0f, 0.15f}, {0.25f, 0.20f}, {3.9f, 0.20f}, {4.1f, 0.17f}, {4.75f, 0.17f},
                     {5.15f, 0.12f}, {5.40f, 0.055f}, {5.50f, 0.0f} }, 16);
    for (int k = 0; k < 4; ++k) {
        glm::mat4 r = glm::rotate(I, kPi * 0.25f + kPi * 0.5f * float(k), glm::vec3(0, 0, 1));
        {   // long-chord wing
            glm::vec3 c[8] = { {0.19f, -0.012f, 1.25f}, {0.19f, 0.012f, 1.25f}, {0.19f, 0.012f, 3.35f}, {0.19f, -0.012f, 3.35f},
                               {0.36f, -0.008f, 1.25f}, {0.36f, 0.008f, 1.25f}, {0.36f, 0.008f, 2.75f}, {0.36f, -0.008f, 2.75f} };
            hexa(mm, r, c);
        }
        {   // tail control fin
            glm::vec3 c[8] = { {0.19f, -0.012f, 0.08f}, {0.19f, 0.012f, 0.08f}, {0.19f, 0.012f, 0.62f}, {0.19f, -0.012f, 0.62f},
                               {0.43f, -0.008f, 0.08f}, {0.43f, 0.008f, 0.08f}, {0.43f, 0.008f, 0.42f}, {0.43f, -0.008f, 0.42f} };
            hexa(md, r, c);
        }
    }
    M.missile = { mm, md };
    return M;
}

} // namespace qlike
