#pragma once

#include <glm/vec3.hpp>
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

struct Level {
    std::vector<Brush> brushes;
    std::vector<collision::AABB> aabbs;  // mirrored from brushes for collision
};

// Built-in starter level: floor + four walls + a few obstacle cubes.
Level make_arena(float radius = 30.0f, float wall_height = 8.0f);

} // namespace qlike::game
