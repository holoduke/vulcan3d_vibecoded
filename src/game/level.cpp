#include "game/level.h"

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

    // Floor — Ground054 with mid-density tiling (~2m per tile).
    add(lv, glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(2.0f * r, 1.0f, 2.0f * r),
        T(0.85f, 0.85f, 0.85f), kGround, kGround, 0.5f,
        T(0.35f, 0.35f, 0.40f));

    // Walls (4 sides), top of wall at y = wh — brick.
    const float wt = 1.0f;
    const float cy = wh * 0.5f;
    add(lv, glm::vec3( r + wt * 0.5f, cy, 0.0f), glm::vec3(wt, wh, 2.0f * r),
        T(0.95f, 0.85f, 0.80f), kBrick, kBrick, 0.45f,
        T(0.55f, 0.30f, 0.30f));
    add(lv, glm::vec3(-r - wt * 0.5f, cy, 0.0f), glm::vec3(wt, wh, 2.0f * r),
        T(0.85f, 0.95f, 0.85f), kBrick, kBrick, 0.45f,
        T(0.30f, 0.55f, 0.30f));
    add(lv, glm::vec3(0.0f, cy,  r + wt * 0.5f), glm::vec3(2.0f * r + 2.0f * wt, wh, wt),
        T(0.85f, 0.85f, 0.95f), kBrick, kBrick, 0.45f,
        T(0.30f, 0.30f, 0.55f));
    add(lv, glm::vec3(0.0f, cy, -r - wt * 0.5f), glm::vec3(2.0f * r + 2.0f * wt, wh, wt),
        T(0.95f, 0.95f, 0.85f), kBrick, kBrick, 0.45f,
        T(0.55f, 0.55f, 0.30f));

    // A few obstacle pillars in the outer arena (between castle and outer
    // walls), useful as cover and for testing RT shadows / GI off objects
    // away from the castle.
    add(lv, glm::vec3( 18.0f, 1.0f,   0.0f), glm::vec3(2.0f, 2.0f, 2.0f),
        T(0.55f, 0.75f, 0.95f), kPaint, kPaint, 1.0f,
        T(0.20f, 0.55f, 0.85f));
    add(lv, glm::vec3(-18.0f, 1.0f,   0.0f), glm::vec3(2.0f, 2.0f, 2.0f),
        T(0.95f, 0.65f, 0.75f), kPaint, kPaint, 1.0f,
        T(0.85f, 0.30f, 0.55f));
    add(lv, glm::vec3(  0.0f, 0.5f, -22.0f), glm::vec3(4.0f, 1.0f, 1.0f),
        T(0.85f, 0.85f, 0.65f), kPaint, kPaint, 1.0f,
        T(0.70f, 0.70f, 0.30f));

    // -----------------------------------------------------------------
    // Castle (centered at origin). 18×18 m footprint, 4.5 m walls, two
    // square towers flanking the south-facing entrance, a stairway up to
    // the wall walkway, and a small inner keep.
    // -----------------------------------------------------------------
    {
        const float cr      = 9.0f;   // half-extent of outer wall (18 m wide)
        const float wh      = 4.5f;   // wall height
        const float wt      = 0.6f;   // wall thickness
        const float ent_w   = 3.0f;   // south-entrance opening width
        const float arch_h  = 1.2f;   // arch lintel height (sits above the gap)

        const auto castle_tint  = T(0.95f, 0.92f, 0.86f);  // warm sandstone
        const auto castle_proto = T(0.55f, 0.50f, 0.42f);
        const auto stair_tint   = T(0.92f, 0.85f, 0.75f);
        const auto stair_proto  = T(0.50f, 0.35f, 0.20f);

        // South wall (continuous; back of castle, away from spawn).
        add(lv, glm::vec3(0.0f, wh * 0.5f, -cr),
            glm::vec3(2.0f * cr, wh, wt),
            castle_tint, kBrick, kBrick, 0.7f, castle_proto);
        // East and west walls (continuous).
        add(lv, glm::vec3( cr, wh * 0.5f, 0.0f),
            glm::vec3(wt, wh, 2.0f * cr),
            castle_tint, kBrick, kBrick, 0.7f, castle_proto);
        add(lv, glm::vec3(-cr, wh * 0.5f, 0.0f),
            glm::vec3(wt, wh, 2.0f * cr),
            castle_tint, kBrick, kBrick, 0.7f, castle_proto);

        // North wall (player-facing) — split around the entrance gap.
        const float seg_w = (2.0f * cr - ent_w) * 0.5f;
        const float seg_x = (cr + ent_w * 0.5f) * 0.5f;
        add(lv, glm::vec3(-seg_x, wh * 0.5f,  cr),
            glm::vec3(seg_w, wh, wt),
            castle_tint, kBrick, kBrick, 0.7f, castle_proto);
        add(lv, glm::vec3( seg_x, wh * 0.5f,  cr),
            glm::vec3(seg_w, wh, wt),
            castle_tint, kBrick, kBrick, 0.7f, castle_proto);
        // Arch lintel above the entrance: top of the gap, still lets the
        // player walk through (clearance ≈ wh - arch_h = 3.3 m).
        add(lv, glm::vec3(0.0f, wh - arch_h * 0.5f,  cr),
            glm::vec3(ent_w, arch_h, wt),
            castle_tint, kBrick, kBrick, 0.7f, castle_proto);

        // Two flanking towers at the north (player-facing) corners. Twice
        // as tall as the walls so they read as proper towers, with a small
        // crenelation ring at the top.
        const float tw_s = 2.4f;          // tower side length
        const float tw_h = 9.0f;          // tower height
        glm::vec3 tower_centers[2] = {
            glm::vec3( cr,  tw_h * 0.5f,  cr),
            glm::vec3(-cr,  tw_h * 0.5f,  cr),
        };
        for (int i = 0; i < 2; ++i) {
            // Solid tower body. Slightly different proto color per side so
            // they're visually distinguishable.
            glm::vec3 proto = (i == 0) ? T(0.50f, 0.40f, 0.30f)
                                       : T(0.40f, 0.40f, 0.45f);
            add(lv, tower_centers[i],
                glm::vec3(tw_s, tw_h, tw_s),
                castle_tint, kBrick, kBrick, 0.7f, proto);
            // Crenelations: 4 small caps at the corners of the tower top.
            const float c_s = 0.35f;
            const float c_h = 0.6f;
            const float c_y = tw_h + c_h * 0.5f;
            const float c_o = (tw_s - c_s) * 0.5f;
            glm::vec3 base = tower_centers[i];
            base.y = c_y;
            for (int dx = -1; dx <= 1; dx += 2) {
                for (int dz = -1; dz <= 1; dz += 2) {
                    add(lv, base + glm::vec3(dx * c_o, 0.0f, dz * c_o),
                        glm::vec3(c_s, c_h, c_s),
                        castle_tint, kBrick, kBrick, 1.0f, proto);
                }
            }
        }

        // Stairway up to the wall-walkway. Sits along the inside of the
        // east wall, climbing south-to-north. n_steps × step_h reaches wh,
        // and step_h stays under collision::kStepHeight (0.45) so the
        // player walks up smoothly without jumping.
        const int   n_steps = 12;
        const float step_h  = wh / float(n_steps);   // 0.375 m per step
        const float step_d  = 0.55f;
        const float stair_w = 2.4f;
        const float stair_x = cr - wt - 0.5f - stair_w * 0.5f;  // hugs east wall
        const float stair_z0 = -(float(n_steps) * 0.5f) * step_d;  // centered
        for (int i = 0; i < n_steps; ++i) {
            float h = step_h * float(i + 1);
            float z = stair_z0 + float(i) * step_d;
            add(lv, glm::vec3(stair_x, h * 0.5f, z),
                glm::vec3(stair_w, h, step_d),
                stair_tint, kBrick, kBrick, 0.6f, stair_proto);
        }

        // Wall walkway — thin platform on top of the south wall (the back
        // wall, away from the entrance), reachable from the top of the
        // stairs by walking along the inside-east-wall edge at wh height.
        // Connects via a short ledge on the east wall too so the path
        // top-of-stairs → east ledge → south walkway works.
        const float walk_w = 1.2f;
        add(lv, glm::vec3(0.0f, wh + 0.05f, -cr + wt + walk_w * 0.5f),
            glm::vec3(2.0f * cr - 2.0f * wt, 0.1f, walk_w),
            stair_tint, kBrick, kBrick, 0.6f, stair_proto);
        add(lv, glm::vec3(cr - wt - walk_w * 0.5f, wh + 0.05f, 0.0f),
            glm::vec3(walk_w, 0.1f, 2.0f * cr - 2.0f * wt),
            stair_tint, kBrick, kBrick, 0.6f, stair_proto);

        // Small inner keep — 4×4 m, 6 m tall in the back of the courtyard.
        // No door (decorative). Gives the courtyard a landmark and breaks
        // line of sight from the entrance to the back walkway.
        add(lv, glm::vec3(0.0f, 3.0f, -4.0f),
            glm::vec3(4.0f, 6.0f, 4.0f),
            T(0.90f, 0.78f, 0.65f), kPaint, kPaint, 1.0f,
            T(0.55f, 0.40f, 0.25f));

        // Lanterns flanking the inside of the entrance and along the back
        // give the courtyard atmospheric warm pools of light at dusk.
        add_lantern(lv, glm::vec3( ent_w * 0.5f + 0.7f, 0.0f, cr - wt - 1.0f));
        add_lantern(lv, glm::vec3(-ent_w * 0.5f - 0.7f, 0.0f, cr - wt - 1.0f));
        add_lantern(lv, glm::vec3( 3.0f, 0.0f, -7.0f));
        add_lantern(lv, glm::vec3(-3.0f, 0.0f, -7.0f));
    }

    // Outer arena lantern posts, kept further out so the castle doesn't
    // overlap them. Two at each "edge" of the arena, off the castle axis.
    add_lantern(lv, glm::vec3( 18.0f, 0.0f,  18.0f));
    add_lantern(lv, glm::vec3(-18.0f, 0.0f,  18.0f));
    add_lantern(lv, glm::vec3( 18.0f, 0.0f, -18.0f));
    add_lantern(lv, glm::vec3(-18.0f, 0.0f, -18.0f));

    return lv;
}

} // namespace qlike::game
