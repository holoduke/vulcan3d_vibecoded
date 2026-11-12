#pragma once

#include <glm/vec3.hpp>
#include <cstdint>
#include <glm/vec4.hpp>
#include <vector>

#include "game/collision.h"

namespace qlike::game {

struct Brush {
    glm::vec3 center;
    glm::vec3 size;   // full size on each axis
    glm::vec4 color;  // texture-tint (multiplies the sampled texture)
    glm::vec3 emissive = glm::vec3(0.0f);  // rgb light output (lantern lamps)
    bool full_emissive = false;            // skip diffuse lighting entirely
    // Texture indices into the engine's bound array (-1 = no texture).
    // uv_scale multiplies the per-vertex UV — bigger walls want bigger numbers
    // so the brick/stone repeats at a sensible scale rather than stretches.
    int   tex_albedo = -1;
    int   tex_normal = -1;
    float uv_scale   = 1.0f;
    // Color used when textures are globally disabled. With textures on,
    // `color` is treated as a near-white tint and the texture supplies the
    // actual hue. With textures off there's no texture to tint, so a
    // near-white color reads as boring/glowing flat white. `fallback_color`
    // is the saturated "prototype look" — distinct per-wall hues etc.
    glm::vec4 fallback_color = glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);
};

// Non-box castle geometry (round towers, conical roofs, the gate arch).
// Same vertex layout as the engine's Vertex (pos, normal, uv) so it can be
// uploaded as-is. UVs are pre-scaled to texture repeats (uv_scale = 1).
struct LevelVertex {
    glm::vec3 p;
    glm::vec3 n;
    float u, v;
};
struct LevelMesh {
    std::vector<LevelVertex> verts;
    std::vector<uint32_t>    idx;
    int       tex_albedo = -1;
    int       tex_normal = -1;
    glm::vec3 tint{1.0f};
    glm::vec3 fallback{0.6f};
    glm::vec3 bmin{1e9f}, bmax{-1e9f};
    // World props: world-space triplanar texturing (uv_scale = repeats
    // per metre) instead of vertex UVs; the UVs then carry wind sway
    // (u = sway weight, v = phase) when `sway` is set.
    // GPU vegetation: index of the first leaf-card triangle (cards follow the
    // branches, two triangles each) and their foliage-atlas cell. The ray
    // tracing BLAS puts the cards in a separate, alpha-tested geometry.
    uint32_t card_index_start = 0xFFFFFFFFu;
    int      card_cell = 0;
    glm::vec3 emissive{0.0f};      // self-lit meshes (lantern glass)
    bool  full_emissive = false;
    bool  triplanar = false;
    bool  sway      = false;
    float uv_scale  = 1.0f;
};

struct Level {
    std::vector<Brush> brushes;
    // Mesh parts (castle towers / roofs / arch). Their player collision
    // boxes are appended to `aabbs` AFTER the brush entries (index
    // >= brushes.size()); render_aabbs stays brush-aligned.
    std::vector<LevelMesh> meshes;
    std::vector<collision::AABB> aabbs;  // mirrored from brushes for collision
    // Render-cull AABBs. Identical to `aabbs` for most brushes, but
    // SPOM-wall brushes carry a small per-axis inflation to cover the
    // shell-mapping extrusion done in cube.vert (kShellThickness =
    // 0.04 m along each face normal). MUST NOT be used for physics --
    // inflating the collision box would push the player off the wall.
    std::vector<collision::AABB> render_aabbs;
};

// Built-in starter level: floor + four walls + a few obstacle cubes.
Level make_arena(float radius = 30.0f, float wall_height = 8.0f);

// Appends a free-standing stone tower in WORLD space (base = ground point):
// style 0 watchtower, 1 beacon, 2 ruin, 3 two-storey keep. Adds its mesh
// parts and player colliders to the level.
void append_outpost_tower(Level& lv, glm::vec3 base, int style, uint32_t seed);
// Timber lamp post with a hanging lantern; returns the lantern's centre.
glm::vec3 append_lamp_post(Level& lv, glm::vec3 base, glm::vec3 arm_dir, uint32_t seed);

} // namespace qlike::game
