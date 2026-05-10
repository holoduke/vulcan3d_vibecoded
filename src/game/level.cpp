#include "game/level.h"

#include <glm/vec2.hpp>

namespace qlike::game {

namespace {
collision::AABB to_aabb(const Brush& b) {
    return { b.center - b.size * 0.5f, b.center + b.size * 0.5f };
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
}

void add_lamp(Level& lv, glm::vec3 center, glm::vec3 size, glm::vec3 rgb,
              glm::vec3 emissive) {
    // Full-emissive shading bypasses albedo/normal entirely — set fallback
    // to rgb so the box looks identical in both texture modes.
    Brush b{ center, size, glm::vec4(rgb, 1.0f), emissive, true,
             -1, -1, 1.0f, glm::vec4(rgb, 1.0f) };
    lv.brushes.push_back(b);
    lv.aabbs.push_back(to_aabb(b));
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
}

Level make_arena(float r, float wh) {
    Level lv;

    // Texture indices match the engine's load order:
    //   0 = Ground054, 1 = Bricks067, 2 = Wood048,
    //   3 = Metal042A, 4 = PaintedPlaster017
    constexpr int kGround = 0;
    constexpr int kBrick  = 1;
    constexpr int kPaint  = 4;

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
    // Castle (centered at origin). 22×22 m footprint, 5 m walls, four
    // corner towers (2.4 m square × 10 m tall), a north-facing entrance,
    // a stairway up to the wall walkway, full-perimeter wall-top walk,
    // crenelations along all 4 walls, and an inner keep with a doorway.
    // -----------------------------------------------------------------
    {
        const float cr      = 11.0f;  // half-extent of outer wall (22 m wide)
        const float wh      = 5.0f;   // wall height
        const float wt      = 0.6f;   // wall thickness
        const float ent_w   = 3.2f;   // entrance opening width (north)
        const float arch_h  = 1.4f;   // arch lintel height (sits above the gap)

        const auto castle_tint  = T(0.95f, 0.92f, 0.86f);  // warm sandstone
        const auto castle_proto = T(0.55f, 0.50f, 0.42f);
        const auto stair_tint   = T(0.92f, 0.85f, 0.75f);
        const auto stair_proto  = T(0.50f, 0.35f, 0.20f);
        const auto crenel_proto = T(0.45f, 0.42f, 0.35f);

        // Helper: add a continuous 4-side wall ring with one wall split
        // around an entrance gap. North wall (player-facing) gets the gap.
        auto add_wall = [&](float cx, float cz, float sx, float sz) {
            add(lv, glm::vec3(cx, wh * 0.5f, cz), glm::vec3(sx, wh, sz),
                castle_tint, kBrick, kBrick, 0.35f, castle_proto);
        };
        add_wall(0.0f, -cr, 2.0f * cr, wt);                 // south
        add_wall( cr, 0.0f, wt, 2.0f * cr);                 // east
        add_wall(-cr, 0.0f, wt, 2.0f * cr);                 // west
        // North split:
        const float seg_w = (2.0f * cr - ent_w) * 0.5f;
        const float seg_x = (cr + ent_w * 0.5f) * 0.5f;
        add_wall(-seg_x,  cr, seg_w, wt);
        add_wall( seg_x,  cr, seg_w, wt);
        // Arch lintel above the entrance: top of the gap, still lets the
        // player walk through (clearance = wh - arch_h ≈ 3.6 m).
        add(lv, glm::vec3(0.0f, wh - arch_h * 0.5f,  cr),
            glm::vec3(ent_w, arch_h, wt),
            castle_tint, kBrick, kBrick, 0.35f, castle_proto);

        // Crenelations along all 4 walls — small merlons every ~1.4 m on
        // each wall's outer edge, sitting on top of the wall.
        const float merlon_s   = 0.45f;
        const float merlon_h   = 0.7f;
        const float merlon_gap = 1.4f;   // step between merlons (centre-to-centre)
        const float wall_top_y = wh + merlon_h * 0.5f;
        // Span along a wall, in the wall's tangent dimension. crenelations
        // skip the corner tower zones since towers carry their own tops.
        const float corner_keep = 1.5f;  // metres from each corner to skip
        auto add_merlons_along_x = [&](float cz_outer) {
            for (float x = -cr + corner_keep + 0.6f;
                 x <  cr - corner_keep;
                 x += merlon_gap) {
                add(lv, glm::vec3(x, wall_top_y, cz_outer),
                    glm::vec3(merlon_s, merlon_h, merlon_s),
                    castle_tint, kBrick, kBrick, 0.5f, crenel_proto);
            }
        };
        auto add_merlons_along_z = [&](float cx_outer) {
            for (float z = -cr + corner_keep + 0.6f;
                 z <  cr - corner_keep;
                 z += merlon_gap) {
                add(lv, glm::vec3(cx_outer, wall_top_y, z),
                    glm::vec3(merlon_s, merlon_h, merlon_s),
                    castle_tint, kBrick, kBrick, 0.5f, crenel_proto);
            }
        };
        add_merlons_along_x(-(cr - wt * 0.5f));  // south outer edge (z = -cr + wt/2)
        add_merlons_along_x( (cr - wt * 0.5f));  // north outer edge — stops short of entrance gap automatically
        add_merlons_along_z(-(cr - wt * 0.5f));  // west outer edge
        add_merlons_along_z( (cr - wt * 0.5f));  // east outer edge

        // Four corner towers — taller than the walls so they read as proper
        // towers, with crenelated tops.
        const float tw_s = 2.4f;
        const float tw_h = 10.0f;
        glm::vec3 tower_centers[4] = {
            glm::vec3( cr,  tw_h * 0.5f,  cr),  // NE
            glm::vec3(-cr,  tw_h * 0.5f,  cr),  // NW
            glm::vec3( cr,  tw_h * 0.5f, -cr),  // SE
            glm::vec3(-cr,  tw_h * 0.5f, -cr),  // SW
        };
        for (int i = 0; i < 4; ++i) {
            add(lv, tower_centers[i],
                glm::vec3(tw_s, tw_h, tw_s),
                castle_tint, kBrick, kBrick, 0.35f, castle_proto);
            // Crenelations around the tower top: 8 merlons (corners + mid-
            // edges) form a battlement ring.
            const float c_s = 0.4f;
            const float c_h = 0.7f;
            const float c_y = tw_h + c_h * 0.5f;
            const float c_o = (tw_s - c_s) * 0.5f;
            glm::vec3 base = tower_centers[i];
            base.y = c_y;
            // 4 corners + 4 mid-edges
            const glm::vec2 offs[8] = {
                glm::vec2(-1,-1), glm::vec2(-1, 1),
                glm::vec2( 1,-1), glm::vec2( 1, 1),
                glm::vec2(-1, 0), glm::vec2( 1, 0),
                glm::vec2( 0,-1), glm::vec2( 0, 1),
            };
            for (int k = 0; k < 8; ++k) {
                const glm::vec2& o = offs[k];
                add(lv, base + glm::vec3(o.x * c_o, 0.0f, o.y * c_o),
                    glm::vec3(c_s, c_h, c_s),
                    castle_tint, kBrick, kBrick, 0.5f, crenel_proto);
            }
        }

        // Wall-top walkway: a thin platform along the inside of all four
        // walls, forming a square ring at y = wh. Built from 4 strips,
        // slightly inset so it doesn't conflict with the merlons.
        const float walk_w = 1.2f;
        const float walk_in = wt + walk_w * 0.5f;
        const float walk_y  = wh + 0.05f;
        // North + south strips (full inside length minus tower bases)
        add(lv, glm::vec3(0.0f, walk_y,  cr - walk_in),
            glm::vec3(2.0f * cr - 2.0f * tw_s, 0.1f, walk_w),
            stair_tint, kBrick, kBrick, 0.3f, stair_proto);
        add(lv, glm::vec3(0.0f, walk_y, -cr + walk_in),
            glm::vec3(2.0f * cr - 2.0f * tw_s, 0.1f, walk_w),
            stair_tint, kBrick, kBrick, 0.3f, stair_proto);
        // East + west strips
        add(lv, glm::vec3( cr - walk_in, walk_y, 0.0f),
            glm::vec3(walk_w, 0.1f, 2.0f * cr - 2.0f * tw_s),
            stair_tint, kBrick, kBrick, 0.3f, stair_proto);
        add(lv, glm::vec3(-cr + walk_in, walk_y, 0.0f),
            glm::vec3(walk_w, 0.1f, 2.0f * cr - 2.0f * tw_s),
            stair_tint, kBrick, kBrick, 0.3f, stair_proto);

        // Stairway up to the walkway. East wall inside, climbing south to
        // north. step_h stays under collision::kStepHeight (0.45) so the
        // player walks up smoothly without jumping.
        const int   n_steps = 13;
        const float step_h  = wh / float(n_steps);
        const float step_d  = 0.55f;
        const float stair_w = 2.4f;
        const float stair_x = cr - wt - walk_w - 0.6f - stair_w * 0.5f;
        const float stair_z0 = -(float(n_steps) * 0.5f) * step_d;
        for (int i = 0; i < n_steps; ++i) {
            float h = step_h * float(i + 1);
            float z = stair_z0 + float(i) * step_d;
            add(lv, glm::vec3(stair_x, h * 0.5f, z),
                glm::vec3(stair_w, h, step_d),
                stair_tint, kBrick, kBrick, 0.3f, stair_proto);
        }

        // Inner keep / house — 6×6 m footprint × 4 m tall, centered in the
        // back half of the courtyard. Has a doorway on the north face so
        // the player can step inside.
        const float kp_cx = 0.0f;
        const float kp_cz = -4.5f;
        const float kp_sx = 6.0f;
        const float kp_sz = 6.0f;
        const float kp_h  = 4.0f;
        const float door_w = 1.2f;
        const float door_h = 2.4f;
        const auto  keep_tint  = T(0.92f, 0.84f, 0.72f);
        const auto  keep_proto = T(0.55f, 0.40f, 0.25f);
        // Three solid walls + the north wall split around a door gap.
        add(lv, glm::vec3(kp_cx, kp_h * 0.5f, kp_cz - kp_sz * 0.5f + wt * 0.5f),
            glm::vec3(kp_sx, kp_h, wt),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);                  // south
        add(lv, glm::vec3(kp_cx + kp_sx * 0.5f - wt * 0.5f, kp_h * 0.5f, kp_cz),
            glm::vec3(wt, kp_h, kp_sz),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);                  // east
        add(lv, glm::vec3(kp_cx - kp_sx * 0.5f + wt * 0.5f, kp_h * 0.5f, kp_cz),
            glm::vec3(wt, kp_h, kp_sz),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);                  // west
        // North wall split for the door.
        const float kp_seg_w = (kp_sx - door_w) * 0.5f;
        const float kp_seg_x = (kp_sx * 0.5f + door_w * 0.5f) * 0.5f;
        add(lv, glm::vec3(kp_cx - kp_seg_x, kp_h * 0.5f, kp_cz + kp_sz * 0.5f - wt * 0.5f),
            glm::vec3(kp_seg_w, kp_h, wt),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);
        add(lv, glm::vec3(kp_cx + kp_seg_x, kp_h * 0.5f, kp_cz + kp_sz * 0.5f - wt * 0.5f),
            glm::vec3(kp_seg_w, kp_h, wt),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);
        // Lintel above keep door — sits between door_h and kp_h, fills
        // the gap above the door so a clean rectangle door is visible.
        const float lintel_h = kp_h - door_h;
        add(lv, glm::vec3(kp_cx, door_h + lintel_h * 0.5f,
                          kp_cz + kp_sz * 0.5f - wt * 0.5f),
            glm::vec3(door_w, lintel_h, wt),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);
        // Roof slab so the keep is visually capped.
        add(lv, glm::vec3(kp_cx, kp_h + 0.05f, kp_cz),
            glm::vec3(kp_sx, 0.1f, kp_sz),
            keep_tint, kPaint, kPaint, 1.0f, keep_proto);

        // Lanterns: pair flanking the entrance + pair near the keep door,
        // and one inside the keep so the interior isn't pitch-black.
        add_lantern(lv, glm::vec3( ent_w * 0.5f + 0.9f, 0.0f, cr - wt - 1.0f));
        add_lantern(lv, glm::vec3(-ent_w * 0.5f - 0.9f, 0.0f, cr - wt - 1.0f));
        add_lantern(lv, glm::vec3( door_w * 0.5f + 0.6f, 0.0f, kp_cz + kp_sz * 0.5f + 0.6f));
        add_lantern(lv, glm::vec3(-door_w * 0.5f - 0.6f, 0.0f, kp_cz + kp_sz * 0.5f + 0.6f));
    }

    // Outer arena lanterns removed along with the floor + perimeter walls
    // — they used to anchor the corners of the brick arena. With the
    // open terrain there is nothing to corner.
    return lv;
}

} // namespace qlike::game
