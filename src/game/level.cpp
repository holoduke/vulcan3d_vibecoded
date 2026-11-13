#include "game/level.h"

#include <glm/vec2.hpp>
#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace qlike::game {

namespace {
// World-space inflation applied to SPOM-wall brush AABBs to cover the
// shell-mapping extrusion done by cube.vert (kShellThickness = 0.04 m
// along each face normal, gated to SPOM wall materials). Frustum culls
// using the raw size would clip wall edges at screen-side as the
// extruded geometry crosses the frustum plane a few cm earlier than
// the un-extruded box. Match the cube.vert constant exactly + a tiny
// safety margin for the bumpy brick silhouette extension.
//
// Applied ONLY to Level::render_aabbs. Level::aabbs stays tight so the
// player collision boxes match the un-extruded wall -- inflating the
// collision AABB would push the player back 5 cm from every wall,
// which reads as "I can't walk up to the wall."
constexpr float kShellInflation = 0.05f;

// SPOM wall materials -- mirrors height_idx_for_albedo() in cube.frag.
// Keep in sync; the extrusion gate in cube.vert uses the same list.
bool is_spom_wall_albedo(int tex_albedo) {
    return tex_albedo == 1 || tex_albedo == 4;
}

// Tight, physics-accurate AABB. Used for collision.
collision::AABB to_aabb(const Brush& b) {
    glm::vec3 half = b.size * 0.5f;
    return { b.center - half, b.center + half };
}

// Render-cull AABB. Inflated for SPOM wall brushes so the frustum cull
// doesn't clip the cube.vert shell extrusion at the screen edge.
collision::AABB to_render_aabb(const Brush& b) {
    glm::vec3 half = b.size * 0.5f;
    if (is_spom_wall_albedo(b.tex_albedo)) {
        // Walls extrude along their face normal only, but cheaper to
        // pad all 3 axes than to track which faces are walls vs floors
        // here -- the world-space cost is 5 cm on a 22 m wall
        // (< 0.25 % overshoot, no measurable frustum-cull regression
        // in the castle scene).
        half += glm::vec3(kShellInflation);
    }
    return { b.center - half, b.center + half };
}

// Sentinel for "no separate fallback" — substitutes `rgb` so brushes added
// without an explicit fallback look the same in textured and untextured mode
// (lanterns, lamps, anything whose `rgb` is already the desired color).
constexpr glm::vec3 kFallbackUseRgb(-1.0f, -1.0f, -1.0f);

void add(Level& lv, glm::vec3 center, glm::vec3 size, glm::vec3 rgb,
         int tex_albedo = -1, int tex_normal = -1, float uv_scale = 1.0f,
         glm::vec3 fallback = kFallbackUseRgb) {
    glm::vec3 fb = (fallback == kFallbackUseRgb) ? rgb : fallback;
    Brush b{ center, size, glm::vec4(rgb, 1.0f), glm::vec3(0.0f), false,
             tex_albedo, tex_normal, uv_scale,
             glm::vec4(fb, 1.0f) };
    lv.brushes.push_back(b);
    lv.aabbs.push_back(to_aabb(b));
    lv.render_aabbs.push_back(to_render_aabb(b));
}

void add_lamp(Level& lv, glm::vec3 center, glm::vec3 size, glm::vec3 rgb,
              glm::vec3 emissive) {
    // Full-emissive shading bypasses albedo/normal entirely — set fallback
    // to rgb so the box looks identical in both texture modes.
    Brush b{ center, size, glm::vec4(rgb, 1.0f), emissive, true,
             -1, -1, 1.0f, glm::vec4(rgb, 1.0f) };
    lv.brushes.push_back(b);
    lv.aabbs.push_back(to_aabb(b));
    lv.render_aabbs.push_back(to_render_aabb(b));
}

void add_lantern(Level& lv, glm::vec3 base) {
    // Post.
    add(lv, base + glm::vec3(0.0f, 1.4f, 0.0f),
        glm::vec3(0.18f, 2.8f, 0.18f), glm::vec3(0.18f, 0.16f, 0.14f));
    // Cap below the lamp (small dark cube).
    add(lv, base + glm::vec3(0.0f, 2.95f, 0.0f),
        glm::vec3(0.40f, 0.10f, 0.40f), glm::vec3(0.10f, 0.09f, 0.08f));
    // Lamp head: warm emissive.
    add_lamp(lv, base + glm::vec3(0.0f, 3.30f, 0.0f),
             glm::vec3(0.55f, 0.55f, 0.55f),
             glm::vec3(1.00f, 0.78f, 0.40f),
             glm::vec3(2.5f, 1.8f, 0.7f));
}

// ---------------------------------------------------------------------
// Mesh builders for the non-box castle parts. Everything is built in
// castle-local space (y = 0 at the courtyard floor) and lifted with the
// brushes in init_world. UVs are in texture repeats (tile_m metres per
// repeat), V pointing DOWN the world so NormalGL's green = world-up.
// ---------------------------------------------------------------------
constexpr float kPi = 3.14159265358979f;

void mesh_push(LevelMesh& m, const LevelVertex& v) {
    m.verts.push_back(v);
    m.bmin = glm::min(m.bmin, v.p);
    m.bmax = glm::max(m.bmax, v.p);
}
void mesh_quad(LevelMesh& m, const LevelVertex& a, const LevelVertex& b,
               const LevelVertex& c, const LevelVertex& d) {
    // The world pipelines cull back faces with the engine's cube winding
    // (front = cross(b - a, c - a) pointing outward). Builders pass the
    // intended outward normal per vertex; flip the winding whenever the
    // quad's geometric normal disagrees, so every builder is cull-safe.
    glm::vec3 g = glm::cross(b.p - a.p, c.p - a.p);
    if (glm::dot(g, g) < 1e-12f) g = glm::cross(c.p - a.p, d.p - a.p);
    const bool flip = glm::dot(g, a.n + b.n + c.n + d.n) < 0.0f;
    uint32_t base = static_cast<uint32_t>(m.verts.size());
    mesh_push(m, a); mesh_push(m, b); mesh_push(m, c); mesh_push(m, d);
    if (flip) m.idx.insert(m.idx.end(), { base, base + 2, base + 1, base, base + 3, base + 2 });
    else      m.idx.insert(m.idx.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
}

// Revolve a (radius, y) profile around the vertical axis through `c`.
// Hard edges between profile segments (flat along the profile, smooth
// around it). The number of texture repeats around is rounded to an
// integer so the seam wraps cleanly.
void lathe(LevelMesh& m, glm::vec3 c, const std::vector<glm::vec2>& prof,
           int segs, float tile_m) {
    float max_r = 0.0f;
    for (const auto& p : prof) max_r = (std::max)(max_r, p.x);
    const float tiles_around =
        (std::max)(1.0f, std::round(2.0f * kPi * max_r / tile_m));
    float s = 0.0f;
    for (size_t j = 0; j + 1 < prof.size(); ++j) {
        glm::vec2 p0 = prof[j], p1 = prof[j + 1];
        glm::vec2 d = p1 - p0;
        float len = glm::length(d);
        if (len < 1e-5f) continue;
        glm::vec2 n2 = glm::vec2(d.y, -d.x) / len;   // outward in (r, y)
        for (int i = 0; i < segs; ++i) {
            float a0 = 2.0f * kPi * float(i) / float(segs);
            float a1 = 2.0f * kPi * float(i + 1) / float(segs);
            auto vtx = [&](float a, glm::vec2 p, float sv, float ui) {
                glm::vec2 dir(std::cos(a), std::sin(a));
                return LevelVertex{
                    c + glm::vec3(dir.x * p.x, p.y, dir.y * p.x),
                    glm::normalize(glm::vec3(dir.x * n2.x, n2.y, dir.y * n2.x)),
                    ui, -sv / tile_m };
            };
            float u0 = tiles_around * float(i) / float(segs);
            float u1 = tiles_around * float(i + 1) / float(segs);
            mesh_quad(m, vtx(a0, p0, s, u0), vtx(a1, p0, s, u1),
                         vtx(a1, p1, s + len, u1), vtx(a0, p1, s + len, u0));
        }
        s += len;
    }
}

// Solid ring sector (merlon / corbel on a round tower).
void ring_sector(LevelMesh& m, glm::vec3 c, float r_in, float r_out,
                 float a0, float a1, float y0, float y1, int segs, float tile_m) {
    auto P = [&](float a, float r, float y) {
        return c + glm::vec3(std::cos(a) * r, y, std::sin(a) * r);
    };
    const glm::vec3 up(0, 1, 0);
    for (int i = 0; i < segs; ++i) {
        float b0 = a0 + (a1 - a0) * float(i) / float(segs);
        float b1 = a0 + (a1 - a0) * float(i + 1) / float(segs);
        glm::vec3 n0(std::cos(b0), 0, std::sin(b0)), n1(std::cos(b1), 0, std::sin(b1));
        float u0 = b0 * r_out / tile_m, u1 = b1 * r_out / tile_m;
        float v0 = -y0 / tile_m, v1 = -y1 / tile_m;
        mesh_quad(m, {P(b0, r_out, y0), n0, u0, v0}, {P(b1, r_out, y0), n1, u1, v0},
                     {P(b1, r_out, y1), n1, u1, v1}, {P(b0, r_out, y1), n0, u0, v1});
        mesh_quad(m, {P(b1, r_in, y0), -n1, u1, v0}, {P(b0, r_in, y0), -n0, u0, v0},
                     {P(b0, r_in, y1), -n0, u0, v1}, {P(b1, r_in, y1), -n1, u1, v1});
        float ri = r_in / tile_m, ro = r_out / tile_m;
        mesh_quad(m, {P(b0, r_in, y1), up, u0, ri}, {P(b0, r_out, y1), up, u0, ro},
                     {P(b1, r_out, y1), up, u1, ro}, {P(b1, r_in, y1), up, u1, ri});
        mesh_quad(m, {P(b1, r_in, y0), -up, u1, ri}, {P(b1, r_out, y0), -up, u1, ro},
                     {P(b0, r_out, y0), -up, u0, ro}, {P(b0, r_in, y0), -up, u0, ri});
    }
    auto cap = [&](float a, float side) {
        glm::vec3 n = glm::vec3(-std::sin(a), 0, std::cos(a)) * side;
        float ri = r_in / tile_m, ro = r_out / tile_m;
        mesh_quad(m, {P(a, r_in, y0), n, ri, -y0 / tile_m}, {P(a, r_out, y0), n, ro, -y0 / tile_m},
                     {P(a, r_out, y1), n, ro, -y1 / tile_m}, {P(a, r_in, y1), n, ri, -y1 / tile_m});
    };
    cap(a0, -1.0f);
    cap(a1, +1.0f);
}

// Crenellated ring: `count` merlons evenly spaced, `duty` = merlon share.
void merlon_ring(LevelMesh& m, glm::vec3 c, float r_in, float r_out,
                 float y0, float y1, int count, float duty, float tile_m) {
    for (int k = 0; k < count; ++k) {
        float a = 2.0f * kPi * float(k) / float(count);
        float w = 2.0f * kPi / float(count) * duty;
        ring_sector(m, c, r_in, r_out, a - w * 0.5f, a + w * 0.5f, y0, y1, 3, tile_m);
    }
}

// Corbels under a parapet: small radial brackets.
void corbel_ring(LevelMesh& m, glm::vec3 c, float r_in, float r_out,
                 float y0, float y1, int count, float tile_m) {
    for (int k = 0; k < count; ++k) {
        float a = 2.0f * kPi * (float(k) + 0.5f) / float(count);
        float w = 0.30f / r_out;
        ring_sector(m, c, r_in, r_out, a - w * 0.5f, a + w * 0.5f, y0, y1, 1, tile_m);
    }
}

// Round-arched gate passage: the part of the rectangular opening
// [-R, R] x [ys, ys + R] OUTSIDE the semicircle of radius R, extruded
// z0..z1, plus the curved soffit (vault underside).
void arch_infill(LevelMesh& m, float ys, float R, float z0, float z1, float tile_m) {
    const int N = 20;
    auto rect_pt = [&](float a) {
        float ca = std::cos(a), sa = std::sin(a);
        float t = 1e9f;
        if (std::abs(ca) > 1e-4f) t = (std::min)(t, R / std::abs(ca));
        if (sa > 1e-4f)           t = (std::min)(t, R / sa);
        return glm::vec2(ca * t, ys + sa * t);
    };
    auto V = [&](glm::vec2 q, float z, glm::vec3 n) {
        return LevelVertex{ glm::vec3(q.x, q.y, z), n, q.x / tile_m, -q.y / tile_m };
    };
    const glm::vec3 fz(0, 0, 1);
    float arc = 0.0f;
    for (int i = 0; i < N; ++i) {
        float a0 = kPi * float(i) / float(N), a1 = kPi * float(i + 1) / float(N);
        glm::vec2 c0(R * std::cos(a0), ys + R * std::sin(a0));
        glm::vec2 c1(R * std::cos(a1), ys + R * std::sin(a1));
        glm::vec2 p0 = rect_pt(a0), p1 = rect_pt(a1);
        mesh_quad(m, V(c0, z1, fz), V(p0, z1, fz), V(p1, z1, fz), V(c1, z1, fz));
        mesh_quad(m, V(c1, z0, -fz), V(p1, z0, -fz), V(p0, z0, -fz), V(c0, z0, -fz));
        glm::vec3 n0(-std::cos(a0), -std::sin(a0), 0), n1(-std::cos(a1), -std::sin(a1), 0);
        float seg = glm::length(c1 - c0);
        mesh_quad(m, {glm::vec3(c0, z0), n0, z0 / tile_m, -arc / tile_m},
                     {glm::vec3(c0, z1), n0, z1 / tile_m, -arc / tile_m},
                     {glm::vec3(c1, z1), n1, z1 / tile_m, -(arc + seg) / tile_m},
                     {glm::vec3(c1, z0), n1, z0 / tile_m, -(arc + seg) / tile_m});
        arc += seg;
    }
}

// Raised voussoir band framing the arch on a wall face (XY plane at z0,
// protruding to z1).
void arch_band(LevelMesh& m, float ys, float r_in, float r_out,
               float z0, float z1, float tile_m) {
    const int N = 20;
    auto V = [&](glm::vec3 p, glm::vec3 n) {
        return LevelVertex{ p, n, p.x / tile_m, -p.y / tile_m };
    };
    const glm::vec3 o(0, ys, 0), fz(0, 0, 1), Z0(0, 0, z0), Z1(0, 0, z1);
    for (int i = 0; i < N; ++i) {
        float a0 = kPi * float(i) / float(N), a1 = kPi * float(i + 1) / float(N);
        glm::vec3 d0(std::cos(a0), std::sin(a0), 0), d1(std::cos(a1), std::sin(a1), 0);
        mesh_quad(m, V(o + d0 * r_in + Z1, fz), V(o + d0 * r_out + Z1, fz),
                     V(o + d1 * r_out + Z1, fz), V(o + d1 * r_in + Z1, fz));
        mesh_quad(m, V(o + d0 * r_out + Z0, d0), V(o + d1 * r_out + Z0, d1),
                     V(o + d1 * r_out + Z1, d1), V(o + d0 * r_out + Z1, d0));
        mesh_quad(m, V(o + d1 * r_in + Z0, -d1), V(o + d0 * r_in + Z0, -d0),
                     V(o + d0 * r_in + Z1, -d0), V(o + d1 * r_in + Z1, -d1));
    }
}

// Four-sided hip (pyramid) roof over a rectangle, with eave underside.
void hip_roof(LevelMesh& m, glm::vec3 c, float hx, float hz, float y0,
              float apex_h, float tile_m) {
    glm::vec3 apex = c + glm::vec3(0, y0 + apex_h, 0);
    glm::vec3 k[4] = { c + glm::vec3(-hx, y0, hz), c + glm::vec3(hx, y0, hz),
                       c + glm::vec3(hx, y0, -hz), c + glm::vec3(-hx, y0, -hz) };
    for (int f = 0; f < 4; ++f) {
        glm::vec3 a = k[f], b = k[(f + 1) % 4];
        glm::vec3 n = glm::normalize(glm::cross(b - a, apex - a));
        if (n.y < 0.0f) n = -n;
        glm::vec3 e = glm::normalize(b - a);
        float slant = glm::length(apex - (a + b) * 0.5f);
        auto V = [&](glm::vec3 p, float vs) {
            return LevelVertex{ p, n, glm::dot(p - a, e) / tile_m, -vs / tile_m };
        };
        mesh_quad(m, V(a, 0.0f), V(b, 0.0f), V(apex, slant), V(apex, slant));
    }
    const glm::vec3 dn(0, -1, 0);
    mesh_quad(m, {k[3], dn, 0, 0}, {k[2], dn, 1, 0}, {k[1], dn, 1, 1}, {k[0], dn, 0, 1});
}

// Player collision for a round tower: octagon-ish union of 3 boxes.
void tower_colliders(std::vector<collision::AABB>& out, glm::vec3 c, float r,
                     float y0, float y1) {
    const glm::vec3 ext[3] = { {r * 0.92f, 0, r * 0.38f}, {r * 0.38f, 0, r * 0.92f},
                               {r * 0.71f, 0, r * 0.71f} };
    for (const auto& e : ext) {
        out.push_back({ glm::vec3(c.x - e.x, c.y + y0, c.z - e.z),
                             glm::vec3(c.x + e.x, c.y + y1, c.z + e.z) });
    }
}
}

Level make_arena(float r, float wh) {
    Level lv;

    // Texture indices match the engine's load order:
    //   0 = Ground054, 1 = Bricks067, 2 = Wood048,
    //   3 = Metal042A, 4 = PaintedBricks001 (keep walls),
    //   5 = Tiles130 (courtyard floor), 6 = Tiles074 (keep interior floor)
    constexpr int kGround = 0;
    constexpr int kBrick  = 1;
    constexpr int kPaint  = 4;
    constexpr int kTile1  = 5;     // courtyard floor
    constexpr int kTile2  = 6;     // keep interior floor

    // For each brush we pass two colors:
    //   `tint`    — multiplies the texture (near-white so the texture's hue
    //               dominates when textures are on)
    //   `proto`   — the saturated prototype color used when textures are off
    //
    // Without `proto`, switching textures off would leave near-white walls
    // everywhere, with no distinguishing detail to mask RT GI noise — the
    // user sees that as "stripes / weird texture / flicker on the wall."
    auto T = [](float r, float g, float b) { return glm::vec3(r, g, b); };

    // Floor + arena perimeter walls removed — replaced by the procedural
    // terrain heightmap (see src/engine/terrain.cpp). The castle below
    // sits on a flattened plateau in the heightmap so the brushes still
    // align at y=0..keep_h.
    (void)r; (void)wh; (void)kGround;

    // -----------------------------------------------------------------
    // Castle (centered at origin, entrance facing +Z).
    //   * 1.6 m thick, 7 m curtain walls with a walkable top, a plinth
    //     course along the outer faces and real crenellations.
    //   * Four ROUND corner towers (mesh): battered base, string course,
    //     corbelled parapet, merlon ring, conical tile roof.
    //   * Gatehouse: deep block with a round-arched passage (mesh arch +
    //     voussoir band), flanked by two round towers with overhanging
    //     conical roofs. The gate doors (props.cpp) sit at z = 11.
    //   * Keep: 8 x 8 m, 11 m tall, hip roof, four corbelled bartizans.
    // Straight walls stay brushes (box relief gives them real masonry
    // depth); everything curved is a LevelMesh.
    // -----------------------------------------------------------------
    {
        const float cr     = 11.0f;  // wall centre line half-extent
        const float wh     = 7.0f;   // wall-walk height
        const float wt     = 1.6f;   // wall thickness
        const float base_y = -1.5f;  // walls run below grade (uneven ground)
        const float ent_w  = 3.2f;   // gate opening width
        const float ent_h  = 3.6f;   // gate opening height (rect part)

        const auto castle_tint  = T(0.95f, 0.92f, 0.86f);  // warm sandstone
        const auto castle_proto = T(0.55f, 0.50f, 0.42f);
        const float wall_uv  = 0.22f;          // ~4.5 m texture repeat (~55 cm stones)
        const float cren_uv  = 0.30f;
        const float stone_tile = 1.0f / wall_uv;
        const auto stair_tint   = T(0.92f, 0.85f, 0.75f);
        const auto stair_proto  = T(0.50f, 0.35f, 0.20f);
        const auto crenel_proto = T(0.45f, 0.42f, 0.35f);
        constexpr int kRoof = 13;              // engine kTexRoof

        auto wall = [&](glm::vec3 c, glm::vec3 s) {
            add(lv, c, s, castle_tint, kBrick, kBrick, wall_uv, castle_proto);
        };
        auto merlon = [&](glm::vec3 c, glm::vec3 s) {
            add(lv, c, s, castle_tint, kBrick, kBrick, cren_uv, crenel_proto);
        };

        // Tower layout (needed to trim walls / merlons).
        const float tw_r   = 3.2f;                       // corner tower radius
        const float gt_r   = 2.4f;                       // gatehouse tower radius
        const float gt_x   = 4.4f;                       // gatehouse tower |x| (clear of the passage)
        const float gt_z   = cr + 0.9f;                  // gatehouse tower z
        const float gh_z0  = cr - wt * 0.5f;             // gatehouse block inner face
        const float gh_z1  = cr + 2.0f;                  // gatehouse block outer face
        const float gh_h   = 9.0f;

        // ---- Curtain walls ----
        const float wy = (wh + base_y) * 0.5f, wsy = wh - base_y;
        wall(glm::vec3(0.0f, wy, -cr), glm::vec3(2.0f * cr, wsy, wt));   // south
        wall(glm::vec3( cr, wy, 0.0f), glm::vec3(wt, wsy, 2.0f * cr));   // east
        wall(glm::vec3(-cr, wy, 0.0f), glm::vec3(wt, wsy, 2.0f * cr));   // west
        {   // north, either side of the gatehouse
            float x0 = gt_x, x1 = cr;
            float w = x1 - x0, cx = (x0 + x1) * 0.5f;
            wall(glm::vec3(-cx, wy, cr), glm::vec3(w, wsy, wt));
            wall(glm::vec3( cx, wy, cr), glm::vec3(w, wsy, wt));
        }
        // Plinth course: 0.45 m skirt along every outer face.
        {
            const float ph = 0.8f - base_y, py = (0.8f + base_y) * 0.5f, pd = 0.45f;
            const float po = cr + wt * 0.5f + pd * 0.5f;
            wall(glm::vec3(0.0f, py, -po), glm::vec3(2.0f * cr, ph, pd));
            wall(glm::vec3( po, py, 0.0f), glm::vec3(pd, ph, 2.0f * cr));
            wall(glm::vec3(-po, py, 0.0f), glm::vec3(pd, ph, 2.0f * cr));
            float x0 = gt_x, x1 = cr, cx = (x0 + x1) * 0.5f;
            wall(glm::vec3(-cx, py, po), glm::vec3(x1 - x0, ph, pd));
            wall(glm::vec3( cx, py, po), glm::vec3(x1 - x0, ph, pd));
        }
        // Crenellations on the outer edge of the wall-walk.
        {
            const float m_w = 0.95f, m_h = 1.15f, m_d = 0.5f, pitch = 1.65f;
            const float my = wh + m_h * 0.5f;
            const float edge = cr + wt * 0.5f - m_d * 0.5f;
            const float lim = cr - tw_r - 0.35f;          // stop at the towers
            auto run = [&](bool along_x, float fixed, float sign, float skip_lo) {
                for (float t = -lim + m_w * 0.5f; t <= lim - m_w * 0.5f + 1e-3f; t += pitch) {
                    if (std::abs(t) < skip_lo) continue;
                    glm::vec3 c = along_x ? glm::vec3(t, my, fixed * sign)
                                          : glm::vec3(fixed * sign, my, t);
                    glm::vec3 s = along_x ? glm::vec3(m_w, m_h, m_d)
                                          : glm::vec3(m_d, m_h, m_w);
                    merlon(c, s);
                }
            };
            run(true,  edge, -1.0f, 0.0f);                 // south
            run(false, edge, +1.0f, 0.0f);                 // east
            run(false, edge, -1.0f, 0.0f);                 // west
            run(true,  edge, +1.0f, gt_x + gt_r + 0.4f);   // north (gatehouse gap)
        }

        // ---- Gatehouse block with arched passage ----
        {
            const float d = gh_z1 - gh_z0, cz = (gh_z0 + gh_z1) * 0.5f;
            const float gy = (gh_h + base_y) * 0.5f, gsy = gh_h - base_y;
            const float side_w = gt_x - ent_w * 0.5f;
            const float side_x = ent_w * 0.5f + side_w * 0.5f;
            wall(glm::vec3(-side_x, gy, cz), glm::vec3(side_w, gsy, d));
            wall(glm::vec3( side_x, gy, cz), glm::vec3(side_w, gsy, d));
            wall(glm::vec3(0.0f, (gh_h + ent_h) * 0.5f, cz),
                 glm::vec3(ent_w, gh_h - ent_h, d));                  // over the passage
            // Battlement on the gatehouse front + back edges.
            for (float x : { -1.1f, 1.1f }) {
                merlon(glm::vec3(x, gh_h + 0.55f, gh_z1 - 0.25f), glm::vec3(0.95f, 1.1f, 0.5f));
                merlon(glm::vec3(x, gh_h + 0.55f, gh_z0 + 0.25f), glm::vec3(0.95f, 1.1f, 0.5f));
            }
            // Paved passage floor (the courtyard slab stops at the wall).
            add(lv, glm::vec3(0.0f, -0.20f, cz - 0.2f), glm::vec3(ent_w, 0.5f, d + 0.4f),
                T(0.96f, 0.94f, 0.90f), kTile1, kTile1, 0.4f, T(0.45f, 0.42f, 0.38f));
        }

        // ---- Mesh parts ----
        LevelMesh stone;  stone.tex_albedo = kBrick; stone.tex_normal = kBrick;
        stone.tint = castle_tint; stone.fallback = castle_proto;
        LevelMesh roof;   roof.tex_albedo = kRoof; roof.tex_normal = kRoof;
        roof.tint = T(0.92f, 0.86f, 0.82f); roof.fallback = T(0.55f, 0.25f, 0.18f);
        std::vector<collision::AABB> mesh_cols;
        const float roof_tile = 1.7f;

        // Corner towers.
        const glm::vec3 corners[4] = {
            { cr, 0.0f,  cr}, {-cr, 0.0f,  cr}, { cr, 0.0f, -cr}, {-cr, 0.0f, -cr} };
        for (const glm::vec3& c : corners) {
            const float r = tw_r, top = 12.0f;
            lathe(stone, c, {
                { r + 0.55f, -4.0f }, { r + 0.55f, 0.0f },           // buried footing
                { r, 2.6f },                                          // battered base
                { r, 6.8f }, { r + 0.14f, 6.8f }, { r + 0.14f, 7.1f },// string course
                { r, 7.1f }, { r, top - 0.6f },
                { r + 0.45f, top },                                   // machicolation
                { r + 0.45f, top + 0.55f }, { r - 0.4f, top + 0.55f },// parapet top
                { r - 0.4f, top + 0.2f }, { 0.0f, top + 0.2f },       // tower top
            }, 64, stone_tile);
            corbel_ring(stone, c, r - 0.05f, r + 0.42f, top - 1.15f, top - 0.45f, 28, stone_tile);
            merlon_ring(stone, c, r - 0.4f, r + 0.45f, top + 0.55f, top + 1.6f, 12, 0.55f, stone_tile);
            lathe(roof, c, { { r - 0.25f, top + 0.4f }, { 0.0f, top + 7.2f } }, 64, roof_tile);
            tower_colliders(mesh_cols, c, r + 0.2f, base_y, top + 1.6f);
        }

        // Gatehouse towers (overhanging conical roofs).
        for (float sx : { -1.0f, 1.0f }) {
            const glm::vec3 c(sx * gt_x, 0.0f, gt_z);
            const float r = gt_r, top = 10.4f;
            lathe(stone, c, {
                { r + 0.45f, -3.0f }, { r + 0.45f, 0.0f }, { r, 2.2f },
                { r, top - 0.7f }, { r + 0.35f, top - 0.2f }, { r + 0.35f, top },
                { 0.0f, top },
            }, 56, stone_tile);
            corbel_ring(stone, c, r - 0.05f, r + 0.33f, top - 1.25f, top - 0.6f, 22, stone_tile);
            lathe(roof, c, {
                { r + 0.35f, top - 0.1f }, { r + 0.75f, top - 0.1f },   // eave underside
                { 0.0f, top + 6.0f } }, 56, roof_tile);
            tower_colliders(mesh_cols, c, r + 0.2f, base_y, top);
        }

        // Arched gate passage (front half, outside the doors at z = 11)
        // + raised voussoir band on the gatehouse face.
        arch_infill(stone, ent_h - ent_w * 0.5f, ent_w * 0.5f, cr + 0.25f, gh_z1, stone_tile);
        arch_band(stone, ent_h - ent_w * 0.5f, ent_w * 0.5f, ent_w * 0.5f + 0.55f,
                  gh_z1, gh_z1 + 0.14f, stone_tile);

        // ---- Keep (donjon) ----
        const float kp_cx = 0.0f, kp_cz = -4.0f;
        const float kp_s = 8.0f, kp_wt = 1.0f, kp_h = 11.0f;
        const float door_w = 1.4f, door_h = 2.6f;
        {
            const float ky = (kp_h - 0.5f) * 0.5f, ksy = kp_h + 0.5f;
            const float h2 = kp_s * 0.5f;
            wall(glm::vec3(kp_cx, ky, kp_cz - h2 + kp_wt * 0.5f), glm::vec3(kp_s, ksy, kp_wt)); // S
            wall(glm::vec3(kp_cx + h2 - kp_wt * 0.5f, ky, kp_cz), glm::vec3(kp_wt, ksy, kp_s - 2.0f * kp_wt)); // E
            wall(glm::vec3(kp_cx - h2 + kp_wt * 0.5f, ky, kp_cz), glm::vec3(kp_wt, ksy, kp_s - 2.0f * kp_wt)); // W
            const float seg_w = (kp_s - door_w) * 0.5f;
            const float seg_x = (door_w + seg_w) * 0.5f;
            const float nz = kp_cz + h2 - kp_wt * 0.5f;
            wall(glm::vec3(kp_cx - seg_x, ky, nz), glm::vec3(seg_w, ksy, kp_wt));
            wall(glm::vec3(kp_cx + seg_x, ky, nz), glm::vec3(seg_w, ksy, kp_wt));
            wall(glm::vec3(kp_cx, (kp_h + door_h) * 0.5f, nz),
                 glm::vec3(door_w, kp_h - door_h, kp_wt));                  // over the door
            // Ceiling slab (the hip roof sits on it).
            wall(glm::vec3(kp_cx, kp_h - 0.15f, kp_cz), glm::vec3(kp_s - 2.0f * kp_wt, 0.3f, kp_s - 2.0f * kp_wt));
            // Interior floor.
            add(lv, glm::vec3(kp_cx, -0.17f, kp_cz),
                glm::vec3(kp_s - 2.0f * kp_wt, 0.5f, kp_s - 2.0f * kp_wt),
                T(0.96f, 0.94f, 0.90f), kTile2, kTile2, 0.5f, T(0.55f, 0.48f, 0.40f));
            hip_roof(roof, glm::vec3(kp_cx, 0.0f, kp_cz), h2 + 0.45f, h2 + 0.45f,
                     kp_h, 4.6f, roof_tile);
            // Corbelled corner bartizans with their own cone roofs.
            for (float sx : { -1.0f, 1.0f }) {
                for (float sz : { -1.0f, 1.0f }) {
                    glm::vec3 c(kp_cx + sx * (h2 + 0.15f), 0.0f, kp_cz + sz * (h2 + 0.15f));
                    lathe(stone, c, {
                        { 0.15f, kp_h - 4.2f }, { 1.05f, kp_h - 2.6f },
                        { 1.05f, kp_h + 1.0f }, { 0.0f, kp_h + 1.0f } }, 40, stone_tile);
                    lathe(roof, c, { { 1.0f, kp_h + 0.9f }, { 1.3f, kp_h + 0.9f },
                                     { 0.0f, kp_h + 3.9f } }, 40, roof_tile);
                }
            }
        }

        // ---- Stairway to the wall-walk (east wall, inside) ----
        {
            const int   n_steps = 17;                    // 0.41 m risers
            const float step_h  = wh / float(n_steps);
            const float step_d  = 0.5f;
            const float stair_w = 2.0f;
            const float stair_x = cr - wt * 0.5f - stair_w * 0.5f;
            const float z0 = -(float(n_steps) * 0.5f) * step_d;
            for (int i = 0; i < n_steps; ++i) {
                float h = step_h * float(i + 1);
                add(lv, glm::vec3(stair_x, (h + base_y) * 0.5f, z0 + float(i) * step_d),
                    glm::vec3(stair_w, h - base_y, step_d),
                    stair_tint, kBrick, kBrick, wall_uv, stair_proto);
            }
        }

        // ---- Courtyard floor (inside the curtain walls) ----
        {
            const float in_ext = (cr - wt * 0.5f) * 2.0f + 0.1f;
            add(lv, glm::vec3(0.0f, -0.20f, 0.0f), glm::vec3(in_ext, 0.5f, in_ext),
                T(0.96f, 0.94f, 0.90f), kTile1, kTile1, 0.4f, T(0.45f, 0.42f, 0.38f));
        }

        // Lanterns: inside the gate, and either side of the keep door.
        add_lantern(lv, glm::vec3( ent_w * 0.5f + 1.2f, 0.0f, gh_z0 - 1.2f));
        add_lantern(lv, glm::vec3(-ent_w * 0.5f - 1.2f, 0.0f, gh_z0 - 1.2f));
        add_lantern(lv, glm::vec3( door_w * 0.5f + 0.8f, 0.0f, kp_cz + kp_s * 0.5f + 0.8f));
        add_lantern(lv, glm::vec3(-door_w * 0.5f - 0.8f, 0.0f, kp_cz + kp_s * 0.5f + 0.8f));

        lv.meshes.push_back(std::move(stone));
        lv.meshes.push_back(std::move(roof));
        // Mesh colliders go AFTER every brush AABB (see Level::meshes).
        for (const auto& a : mesh_cols) lv.aabbs.push_back(a);
    }

    // Outer arena lanterns removed along with the floor + perimeter walls
    // — they used to anchor the corners of the brick arena. With the
    // open terrain there is nothing to corner.
    return lv;
}

// Free-standing outpost towers scattered over the foothills (world space,
// `base` = ground point under the centre). Styles:
//   0 watchtower  - tall round tower, machicolated parapet, conical roof
//   1 beacon      - slender, open crenellated top with a raised fire basket
//   2 ruin        - broken shell: wall segments of uneven height, no roof
//   3 keep        - squat drum with a second, narrower storey and a roof
void append_outpost_tower(Level& lv, glm::vec3 base, int style, uint32_t seed) {
    constexpr int kBrick = 1, kRoof = 13;
    const float stone_tile = 1.0f / 0.22f, roof_tile = 1.7f;
    LevelMesh stone; stone.tex_albedo = kBrick; stone.tex_normal = kBrick;
    stone.tint = glm::vec3(0.86f, 0.84f, 0.80f); stone.fallback = glm::vec3(0.55f, 0.50f, 0.42f);
    LevelMesh roof; roof.tex_albedo = kRoof; roof.tex_normal = kRoof;
    roof.tint = glm::vec3(0.92f, 0.86f, 0.82f); roof.fallback = glm::vec3(0.55f, 0.25f, 0.18f);
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return float((seed >> 8) & 0xFFFFu) / 65535.0f;
    };
    std::vector<collision::AABB> cols;
    const glm::vec3 c = base;
    if (style == 0) {
        const float r = 3.0f + rnd() * 0.6f, top = 15.0f + rnd() * 4.0f;
        lathe(stone, c, {
            { r + 0.9f, -6.0f }, { r + 0.9f, 0.0f }, { r, 3.4f },             // footing + batter
            { r, top * 0.5f }, { r + 0.14f, top * 0.5f }, { r + 0.14f, top * 0.5f + 0.3f },
            { r, top * 0.5f + 0.3f }, { r, top - 0.7f },
            { r + 0.5f, top }, { r + 0.5f, top + 0.55f }, { r - 0.4f, top + 0.55f },
            { r - 0.4f, top + 0.2f }, { 0.0f, top + 0.2f },
        }, 48, stone_tile);
        corbel_ring(stone, c, r - 0.05f, r + 0.46f, top - 1.2f, top - 0.45f, 22, stone_tile);
        merlon_ring(stone, c, r - 0.4f, r + 0.5f, top + 0.55f, top + 1.6f, 10, 0.55f, stone_tile);
        lathe(roof, c, { { r - 0.3f, top + 0.4f }, { r + 0.05f, top + 1.7f }, { 0.0f, top + 8.0f } }, 48, roof_tile);
        tower_colliders(cols, c, r + 0.2f, -1.0f, top + 1.6f);
    } else if (style == 1) {
        const float r = 2.1f + rnd() * 0.3f, top = 20.0f + rnd() * 5.0f;
        lathe(stone, c, {
            { r + 1.1f, -6.0f }, { r + 1.1f, 0.0f }, { r + 0.25f, 4.5f }, { r, top - 0.9f },
            { r + 0.65f, top }, { r + 0.65f, top + 0.5f }, { r + 0.25f, top + 0.5f },
            { r + 0.25f, top + 0.15f }, { 0.0f, top + 0.15f },
        }, 40, stone_tile);
        corbel_ring(stone, c, r - 0.05f, r + 0.6f, top - 1.3f, top - 0.5f, 16, stone_tile);
        merlon_ring(stone, c, r + 0.25f, r + 0.65f, top + 0.5f, top + 1.5f, 8, 0.5f, stone_tile);
        // Fire basket on a short pillar.
        lathe(stone, c, { { 0.35f, top + 0.15f }, { 0.35f, top + 1.3f }, { 0.75f, top + 1.6f },
                          { 0.75f, top + 1.95f }, { 0.55f, top + 1.95f }, { 0.0f, top + 1.75f } }, 16, stone_tile);
        tower_colliders(cols, c, r + 0.4f, -1.0f, top + 1.5f);
    } else if (style == 2) {
        const float r = 3.6f + rnd() * 0.8f, top = 7.0f + rnd() * 3.0f;
        // Solid plinth, then a hollow broken shell above it.
        lathe(stone, c, { { r + 0.8f, -6.0f }, { r + 0.8f, 0.0f }, { r, 2.2f }, { r, 2.6f }, { 0.0f, 2.6f } },
              40, stone_tile);
        const int segs = 14;
        float h = top;
        for (int k = 0; k < segs; ++k) {
            // Random-walk the wall height; one side collapsed almost to the plinth.
            h = glm::clamp(h + (rnd() - 0.5f) * 4.0f, 3.0f, top + 2.5f);
            float hk = (k >= segs - 4) ? 2.9f + rnd() * 1.4f : h;
            float a0 = 2.0f * kPi * float(k) / float(segs), a1 = 2.0f * kPi * float(k + 1) / float(segs);
            ring_sector(stone, c, r - 1.0f, r, a0, a1 + 0.002f, 2.55f, hk, 3, stone_tile);
        }
        // Fallen blocks around the foot.
        for (int k = 0; k < 7; ++k) {
            float a = rnd() * 2.0f * kPi, d = r + 1.0f + rnd() * 3.5f;
            glm::vec3 bc = c + glm::vec3(std::cos(a) * d, -0.2f, std::sin(a) * d);
            float s = 0.45f + rnd() * 0.5f;
            lathe(stone, bc, { { 0.0f, 0.0f }, { s, 0.0f }, { s * 0.9f, s * 0.9f }, { 0.0f, s * 0.9f } }, 5, stone_tile);
        }
        tower_colliders(cols, c, r + 0.1f, -1.0f, 2.6f);   // the solid plinth only: the shell is open
    } else {
        const float r = 4.6f + rnd() * 0.6f, mid = 8.0f + rnd() * 1.5f, r2 = r - 1.6f, top = mid + 6.5f + rnd() * 2.0f;
        lathe(stone, c, {
            { r + 1.0f, -6.0f }, { r + 1.0f, 0.0f }, { r, 2.8f }, { r, mid - 0.6f },
            { r + 0.45f, mid }, { r + 0.45f, mid + 0.5f }, { r - 0.35f, mid + 0.5f },
            { r - 0.35f, mid + 0.15f }, { r2, mid + 0.15f },                       // wall-walk
            { r2, top - 0.5f }, { r2 + 0.35f, top }, { r2 + 0.35f, top + 0.2f }, { 0.0f, top + 0.2f },
        }, 56, stone_tile);
        corbel_ring(stone, c, r - 0.05f, r + 0.42f, mid - 1.2f, mid - 0.45f, 30, stone_tile);
        merlon_ring(stone, c, r - 0.35f, r + 0.45f, mid + 0.5f, mid + 1.5f, 14, 0.55f, stone_tile);
        lathe(roof, c, { { r2 + 0.35f, top + 0.1f }, { r2 + 0.8f, top + 0.1f }, { 0.0f, top + 5.5f } }, 48, roof_tile);
        tower_colliders(cols, c, r + 0.2f, -1.0f, mid + 1.5f);
    }
    lv.meshes.push_back(std::move(stone));
    if (!roof.idx.empty()) lv.meshes.push_back(std::move(roof));
    for (const auto& a : cols) lv.aabbs.push_back(a);
}


// ---------------------------------------------------------------------
// Road lamp post: a hewn timber post with a braced arm and a hanging iron
// lantern (world space; `base` on the ground, the arm reaches along
// `arm_dir`, a horizontal unit vector, toward the road).
// ---------------------------------------------------------------------
namespace {

// Tapered prism along a -> b with `segs` sides (4 = square beam, more =
// round). UVs: u around, v along (texture repeats, tile_m metres each).
void lp_beam(LevelMesh& m, glm::vec3 a, glm::vec3 b, float r0, float r1, int segs,
             glm::vec3 ref, float tile_m, bool caps, float twist = 0.0f) {
    const glm::vec3 ax = glm::normalize(b - a);
    glm::vec3 u = glm::normalize(ref - ax * glm::dot(ref, ax));
    const glm::vec3 w = glm::cross(ax, u);
    const float len = glm::length(b - a);
    const float k = segs == 4 ? 1.41421356f : 1.0f;      // square: r is the half side
    auto dir = [&](int i, float t) {
        const float ang = 2.0f * kPi * (float(i) + (segs == 4 ? 0.5f : 0.0f)) / float(segs) + twist * t;
        return u * std::cos(ang) + w * std::sin(ang);
    };
    for (int i = 0; i < segs; ++i) {
        const glm::vec3 d0a = dir(i, 0.0f), d1a = dir(i + 1, 0.0f), d0b = dir(i, 1.0f), d1b = dir(i + 1, 1.0f);
        const glm::vec3 n = glm::normalize(d0a + d1a);
        const glm::vec3 n0 = segs <= 6 ? n : d0a, n1 = segs <= 6 ? n : d1a;      // flat faces on beams
        const float u0 = float(i) * 2.0f * r0 * k / tile_m, u1 = u0 + 2.0f * r0 * k / tile_m;
        mesh_quad(m, { a + d0a * (r0 * k), n0, u0, 0.0f }, { a + d1a * (r0 * k), n1, u1, 0.0f },
                     { b + d1b * (r1 * k), n1, u1, -len / tile_m }, { b + d0b * (r1 * k), n0, u0, -len / tile_m });
    }
    if (caps)
        for (int i = 1; i + 1 < segs; ++i) {
            mesh_quad(m, { b + dir(0, 1.0f) * (r1 * k), ax, 0.0f, 0.0f }, { b + dir(i, 1.0f) * (r1 * k), ax, 0.1f, 0.0f },
                         { b + dir(i + 1, 1.0f) * (r1 * k), ax, 0.1f, 0.1f }, { b + dir(i + 1, 1.0f) * (r1 * k), ax, 0.1f, 0.1f });
            mesh_quad(m, { a + dir(0, 0.0f) * (r0 * k), -ax, 0.0f, 0.0f }, { a + dir(i, 0.0f) * (r0 * k), -ax, 0.1f, 0.0f },
                         { a + dir(i + 1, 0.0f) * (r0 * k), -ax, 0.1f, 0.1f }, { a + dir(i + 1, 0.0f) * (r0 * k), -ax, 0.1f, 0.1f });
        }
}

}  // namespace

glm::vec3 append_lamp_post(Level& lv, glm::vec3 base, glm::vec3 arm_dir, uint32_t seed) {
    constexpr int kWood = 17, kMetal = 3;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float((seed >> 8) & 0xFFFFu) / 65535.0f; };
    LevelMesh wood; wood.tex_albedo = kWood; wood.tex_normal = kWood;
    wood.tint = glm::vec3(0.95f, 0.90f, 0.84f); wood.fallback = glm::vec3(0.30f, 0.21f, 0.13f);
    LevelMesh iron; iron.tex_albedo = kMetal; iron.tex_normal = kMetal;
    iron.tint = glm::vec3(0.16f, 0.15f, 0.14f); iron.fallback = glm::vec3(0.09f, 0.09f, 0.09f);
    LevelMesh glass;                                     // warm, self-lit panes + flame
    glass.tint = glass.fallback = glm::vec3(1.0f, 0.74f, 0.36f);
    glass.emissive = glm::vec3(3.2f, 2.0f, 0.75f);
    glass.full_emissive = true;

    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 A = glm::normalize(glm::vec3(arm_dir.x, 0.0f, arm_dir.z));
    const glm::vec3 S(-A.z, 0.0f, A.x);
    const float H = 3.55f + 0.25f * rnd();
    const glm::vec3 lean = (A * (rnd() - 0.5f) + S * (rnd() - 0.5f)) * 0.10f;   // posts are never plumb
    const glm::vec3 foot = base - up * 0.7f;
    const glm::vec3 top = base + up * H + lean;
    const float wt = 0.55f;                              // wood texture: metres per repeat

    // Post: hewn octagon, slightly tapered and twisted, with a chamfered top.
    lp_beam(wood, foot, top - up * 0.12f, 0.105f, 0.085f, 8, A, wt, false, 0.25f);
    lp_beam(wood, top - up * 0.12f, top, 0.085f, 0.05f, 8, A, wt, true, 0.0f);
    // Stone-packed footing collar.
    lp_beam(wood, base - up * 0.05f, base + up * 0.16f, 0.15f, 0.125f, 8, A, wt, true);
    // Arm (square beam) through the post, and a diagonal brace under it.
    const glm::vec3 arm_c = base + up * (H - 0.42f) + lean * 0.88f;
    const glm::vec3 arm_a = arm_c - A * 0.22f, arm_b = arm_c + A * 0.98f + up * 0.03f;
    lp_beam(wood, arm_a, arm_b, 0.052f, 0.045f, 4, up, wt, true);
    lp_beam(wood, base + up * (H - 1.18f) + lean * 0.67f + A * 0.07f, arm_c + A * 0.66f - up * 0.04f,
            0.036f, 0.036f, 4, S, wt, true);
    // Iron: straps round the joint, pegs, and the hook plate at the arm end.
    lp_beam(iron, arm_c - up * 0.075f, arm_c + up * 0.075f, 0.097f, 0.097f, 8, A, 0.3f, false);
    lp_beam(iron, arm_c + A * 0.62f - up * 0.062f, arm_c + A * 0.70f - up * 0.062f, 0.058f, 0.058f, 4, up, 0.3f, true);
    const glm::vec3 hook = arm_b - A * 0.10f;
    lp_beam(iron, hook - up * 0.06f - A * 0.03f, hook - up * 0.06f + A * 0.03f, 0.056f, 0.056f, 4, up, 0.3f, true);
    // Chain: four alternating links.
    glm::vec3 c = hook - up * 0.055f;
    for (int i = 0; i < 4; ++i) {
        const glm::vec3 side = (i & 1) ? A : S;
        lp_beam(iron, c - side * 0.012f, c - up * 0.055f - side * 0.012f, 0.0045f, 0.0045f, 4, side, 0.2f, false);
        lp_beam(iron, c + side * 0.012f, c - up * 0.055f + side * 0.012f, 0.0045f, 0.0045f, 4, side, 0.2f, false);
        c -= up * 0.05f;
    }
    // Lantern: ring, pyramid roof with a finial, four corner bars, base
    // tray, glass panes and a flame.
    const float lw = 0.105f, lh = 0.27f;                 // half width, body height
    const glm::vec3 lt = c - up * 0.10f;                 // top of the body
    const glm::vec3 lb = lt - up * lh;
    lp_beam(iron, c, lt + up * 0.12f, 0.008f, 0.012f, 6, A, 0.2f, false);
    lp_beam(iron, lt + up * 0.11f, lt + up * 0.135f, 0.022f, 0.016f, 6, A, 0.2f, true);
    lp_beam(iron, lt, lt + up * 0.115f, lw * 1.28f, 0.02f, 4, A, 0.3f, true);             // roof
    lp_beam(iron, lt - up * 0.018f, lt, lw * 1.12f, lw * 1.28f, 4, A, 0.3f, true);        // eave
    lp_beam(iron, lb - up * 0.03f, lb, lw * 0.80f, lw * 1.10f, 4, A, 0.3f, true);         // tray
    lp_beam(iron, lb - up * 0.05f, lb - up * 0.03f, 0.02f, lw * 0.80f, 4, A, 0.3f, true);
    for (int k = 0; k < 4; ++k) {
        const glm::vec3 d = (A * ((k & 1) ? 1.0f : -1.0f) + S * ((k & 2) ? 1.0f : -1.0f)) * lw;
        lp_beam(iron, lb + d, lt + d, 0.0085f, 0.0085f, 4, A, 0.2f, false);
        // mid rail on each side
        const glm::vec3 e = (k & 1) ? A : S, f = (k & 1) ? S : A;
        const glm::vec3 mc = lb + up * (lh * 0.5f) + e * (((k & 2) ? 1.0f : -1.0f) * lw);
        lp_beam(iron, mc - f * lw, mc + f * lw, 0.005f, 0.005f, 4, up, 0.2f, false);
    }
    lp_beam(glass, lb + up * 0.004f, lt - up * 0.004f, lw * 0.93f, lw * 0.93f, 4, A, 1.0f, false);   // panes
    lp_beam(glass, lb + up * 0.02f, lb + up * 0.13f, 0.022f, 0.004f, 6, A, 1.0f, true);              // flame

    // All posts share three meshes (wood / iron / glass): 3 draws and 3
    // BLAS geometries for the whole road instead of 3 per post.
    auto merge = [&](LevelMesh&& m, auto same) {
        for (LevelMesh& e : lv.meshes) {
            if (!same(e)) continue;
            const uint32_t off = uint32_t(e.verts.size());
            e.verts.insert(e.verts.end(), m.verts.begin(), m.verts.end());
            for (uint32_t i : m.idx) e.idx.push_back(i + off);
            e.bmin = glm::min(e.bmin, m.bmin); e.bmax = glm::max(e.bmax, m.bmax);
            return;
        }
        lv.meshes.push_back(std::move(m));
    };
    merge(std::move(wood),  [](const LevelMesh& e) { return e.tex_albedo == kWood && !e.full_emissive; });
    merge(std::move(iron),  [](const LevelMesh& e) { return e.tex_albedo == kMetal && e.tint == glm::vec3(0.16f, 0.15f, 0.14f); });
    merge(std::move(glass), [](const LevelMesh& e) { return e.full_emissive && e.emissive == glm::vec3(3.2f, 2.0f, 0.75f); });
    lv.aabbs.push_back({ base - glm::vec3(0.13f, 0.3f, 0.13f), base + glm::vec3(0.13f, H, 0.13f) });
    return lb + up * (lh * 0.5f);
}

} // namespace qlike::game

