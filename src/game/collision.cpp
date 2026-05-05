#include "game/collision.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace qlike::collision {

namespace {
constexpr float kEpsilon = 1e-4f;
}

bool aabb_overlap(const AABB& a, const AABB& b) {
    // Strict: exact face-touching counts as NOT overlapping. Required so the
    // player can rest on the floor without being treated as penetrating it.
    return a.min.x < b.max.x && a.max.x > b.min.x &&
           a.min.y < b.max.y && a.max.y > b.min.y &&
           a.min.z < b.max.z && a.max.z > b.min.z;
}

AABB aabb_translate(const AABB& a, glm::vec3 d) {
    return AABB{ a.min + d, a.max + d };
}

SweepResult sweep_aabb(const AABB& moving, glm::vec3 displacement, const AABB& fixed) {
    if (aabb_overlap(moving, fixed)) {
        return { false, 1.0f, glm::vec3(0.0f) };
    }

    glm::vec3 entry(0.0f);
    glm::vec3 exit(0.0f);

    for (int i = 0; i < 3; ++i) {
        if (std::abs(displacement[i]) < std::numeric_limits<float>::epsilon()) {
            if (moving.max[i] < fixed.min[i] || moving.min[i] > fixed.max[i]) {
                return { false, 1.0f, glm::vec3(0.0f) };
            }
            entry[i] = -std::numeric_limits<float>::infinity();
            exit[i]  =  std::numeric_limits<float>::infinity();
        } else if (displacement[i] > 0.0f) {
            entry[i] = (fixed.min[i] - moving.max[i]) / displacement[i];
            exit[i]  = (fixed.max[i] - moving.min[i]) / displacement[i];
        } else {
            entry[i] = (fixed.max[i] - moving.min[i]) / displacement[i];
            exit[i]  = (fixed.min[i] - moving.max[i]) / displacement[i];
        }
    }

    float entry_time = std::max({ entry.x, entry.y, entry.z });
    float exit_time  = std::min({ exit.x,  exit.y,  exit.z });

    if (entry_time > exit_time || entry_time < 0.0f || entry_time > 1.0f) {
        return { false, 1.0f, glm::vec3(0.0f) };
    }

    glm::vec3 normal(0.0f);
    if (entry.x >= entry.y && entry.x >= entry.z) {
        normal.x = displacement.x > 0.0f ? -1.0f : 1.0f;
    } else if (entry.y >= entry.x && entry.y >= entry.z) {
        normal.y = displacement.y > 0.0f ? -1.0f : 1.0f;
    } else {
        normal.z = displacement.z > 0.0f ? -1.0f : 1.0f;
    }

    return { true, entry_time, normal };
}

namespace {
// Push `pos` out of any world brush it currently penetrates, along the axis
// of smallest penetration. Repeats up to `max_passes` times to escape corners.
glm::vec3 depenetrate(const AABB& player, glm::vec3 pos,
                      std::span<const AABB> world, int max_passes = 4) {
    auto box_at = [&](glm::vec3 p) {
        return aabb_translate(player, p - 0.5f * (player.min + player.max));
    };
    for (int pass = 0; pass < max_passes; ++pass) {
        bool moved = false;
        AABB box = box_at(pos);
        for (const AABB& w : world) {
            int   min_axis = -1;
            float min_pen  = std::numeric_limits<float>::infinity();
            float min_dir  = 0.0f;
            bool  overlap  = true;
            for (int i = 0; i < 3; ++i) {
                float push_pos = w.max[i] - box.min[i];  // distance to clear in +i
                float push_neg = box.max[i] - w.min[i];  // distance to clear in -i
                if (push_pos <= 0.0f || push_neg <= 0.0f) { overlap = false; break; }
                float pen = std::min(push_pos, push_neg);
                if (pen < min_pen) {
                    min_pen  = pen;
                    min_axis = i;
                    min_dir  = (push_pos < push_neg) ? 1.0f : -1.0f;
                }
            }
            if (overlap && min_axis >= 0) {
                pos[min_axis] += min_dir * (min_pen + kEpsilon);
                box = box_at(pos);
                moved = true;
            }
        }
        if (!moved) break;
    }
    return pos;
}
}

MoveResult slide_move(const AABB& player, glm::vec3 position, glm::vec3 velocity,
                      std::span<const AABB> world, float dt, int max_iterations) {
    // Maximum height of a vertical step the player can climb without jumping.
    // Walking into a riser ≤ kStepHeight: the player is automatically lifted
    // onto the step's top surface. Walking into a wall ≥ kStepHeight: blocked
    // as before. 0.45 m matches a generous flight of stairs (~30 cm steps
    // with margin) without letting the player walk up boxes meant as cover.
    constexpr float kStepHeight = 0.45f;

    auto box_at = [&](glm::vec3 p) {
        return aabb_translate(player, p - 0.5f * (player.min + player.max));
    };
    auto pos_overlaps = [&](glm::vec3 p) -> bool {
        AABB pb = box_at(p);
        for (const AABB& w : world) if (aabb_overlap(pb, w)) return true;
        return false;
    };
    auto best_sweep = [&](AABB swept_box, glm::vec3 displacement) {
        float t = 1.0f; glm::vec3 n(0.0f);
        for (const AABB& w : world) {
            SweepResult r = sweep_aabb(swept_box, displacement, w);
            if (r.hit && r.t < t) { t = r.t; n = r.normal; }
        }
        return std::pair<float, glm::vec3>(t, n);
    };

    position = depenetrate(player, position, world);
    glm::vec3 remaining = velocity * dt;
    glm::vec3 pos = position;
    bool grounded = false;

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (glm::dot(remaining, remaining) < kEpsilon * kEpsilon) break;

        AABB swept_box = box_at(pos);
        auto [earliest_t, hit_normal] = best_sweep(swept_box, remaining);
        bool any_hit = earliest_t < 1.0f;

        if (!any_hit) {
            // Free travel — no skin offset.
            pos += remaining;
            remaining = glm::vec3(0.0f);
            break;
        }

        // Step-up: if we hit a vertical wall (mostly horizontal normal) and
        // there's clear space above the player when lifted by kStepHeight,
        // try the same horizontal move from the lifted position. If that
        // clears the obstacle, accept the step and drop the player back
        // down onto the new surface. Lets a real flight of stairs work
        // without per-step jumping.
        if (std::abs(hit_normal.y) < 0.5f) {
            glm::vec3 horiz_disp(remaining.x, 0.0f, remaining.z);
            glm::vec3 lifted_pos = pos + glm::vec3(0.0f, kStepHeight, 0.0f);
            if (!pos_overlaps(lifted_pos)) {
                AABB lifted_box = box_at(lifted_pos);
                auto step = best_sweep(lifted_box, horiz_disp);
                float step_t = step.first;
                if (step_t > earliest_t + kEpsilon) {
                    // Move horizontally from the lifted position…
                    glm::vec3 stepped = lifted_pos +
                        horiz_disp * std::max(0.0f, step_t - kEpsilon);
                    // …then snap straight down by up to kStepHeight to land
                    // on the step's top. drop_t is the fraction of
                    // kStepHeight we travel before hitting something.
                    AABB stepped_box = box_at(stepped);
                    auto drop = best_sweep(stepped_box,
                                            glm::vec3(0.0f, -kStepHeight, 0.0f));
                    float drop_t = drop.first;
                    pos = stepped + glm::vec3(
                        0.0f, -kStepHeight * std::max(0.0f, drop_t - kEpsilon),
                        0.0f);
                    grounded = true;
                    remaining = glm::vec3(0.0f);
                    break;
                }
            }
        }

        // Move up to (but not into) the contact, leaving a small skin distance.
        float move_t = std::max(0.0f, earliest_t - kEpsilon);
        pos += remaining * move_t;

        if (hit_normal.y > 0.7f) grounded = true;

        // Slide: project remaining displacement onto the contact plane.
        glm::vec3 leftover = remaining * (1.0f - earliest_t);
        float into_plane = glm::dot(leftover, hit_normal);
        leftover -= hit_normal * into_plane;
        remaining = leftover;

        // Cancel velocity component into the surface so subsequent frames don't
        // re-accumulate into it.
        float vel_into = glm::dot(velocity, hit_normal);
        if (vel_into < 0.0f) velocity -= hit_normal * vel_into;
    }

    return { pos, velocity, grounded };
}

} // namespace qlike::collision
