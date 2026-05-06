#pragma once

#include <glm/vec3.hpp>
#include <span>

namespace qlike::collision {

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

bool aabb_overlap(const AABB& a, const AABB& b);
AABB aabb_translate(const AABB& a, glm::vec3 d);

struct SweepResult {
    bool hit;
    float t;          // contact time in [0,1] along the displacement
    glm::vec3 normal; // surface normal of the static box at point of contact
};

// Swept AABB-vs-AABB. Treats `moving` as displaced by `displacement` over t in [0,1]
// and finds the first time it touches `fixed`. Initial overlap is reported as
// {hit=false, t=1, normal=0} — the caller should resolve penetration separately.
SweepResult sweep_aabb(const AABB& moving, glm::vec3 displacement, const AABB& fixed);

struct MoveResult {
    glm::vec3 position;
    glm::vec3 velocity;
    bool grounded;
    // Vertical distance the auto step-up logic boosted us this frame, in
    // metres. 0 when no step occurred. The renderer subtracts a decaying
    // copy of this from the eye height so a flight of stairs reads as a
    // smooth ramp rather than a per-step zigsaw.
    float step_amount;
};

// Slide-move: given a player AABB at `position`, attempt to move by `velocity * dt`
// against a list of AABBs, sliding along contact planes. Up to `max_iterations`
// slide iterations.
//
// `static_count` is the number of leading entries in `world` that are STATIC
// (level brushes / stairs). Auto step-up is only permitted over static
// obstacles — stepping onto a dynamic crate would let the player walk
// through pushable boxes. The default of npos treats every entry as static
// (back-compat for the test suite).
MoveResult slide_move(const AABB& player, glm::vec3 position, glm::vec3 velocity,
                      std::span<const AABB> world, float dt,
                      int max_iterations = 4,
                      size_t static_count = static_cast<size_t>(-1));

} // namespace qlike::collision
