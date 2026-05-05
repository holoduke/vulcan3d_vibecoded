#pragma once

#include <glm/vec3.hpp>

namespace qlike::physics {

// Default tunings — Quake3-ish, Y-up. Distances are in "engine units" (~roughly inches).
struct MoveParams {
    float ground_friction = 6.0f;
    float ground_accel    = 10.0f;
    float air_accel       = 10.0f;
    // Strafe-jump enabler: in air, wishspeed is capped at this value before applying
    // accelerate(). The asymmetry between "capped wishspeed" (used as the speed bound)
    // and full wishspeed (used in the per-tick add) is what allows speed to accumulate.
    float air_wish_cap    = 30.0f;
    float stop_speed      = 100.0f;
    float max_walk_speed  = 320.0f;
    float gravity         = 800.0f;
    float jump_speed      = 270.0f;
};

// Apply ground friction to the horizontal (x,z) component. y is preserved.
// Velocities below `snap_threshold` (m/s) get clamped to zero — this is Quake's
// "stop drifting" behavior. Default 1.0 matches Quake's "1 unit/s"; for a
// meters-scale world pass something like 0.05.
glm::vec3 apply_friction(glm::vec3 vel, float friction, float stop_speed, float dt,
                         float snap_threshold = 1.0f);

// Quake "PM_Accelerate". Adds at most (wishspeed - currentspeed_along_wishdir) per call,
// scaled by accel*dt*wishspeed. wishdir must be normalized.
glm::vec3 accelerate(glm::vec3 vel, glm::vec3 wishdir, float wishspeed,
                     float accel, float dt);

// Quake "PM_AirAccelerate". Same as accelerate but with wishspeed capped to wish_cap
// for the bound check, while the full wishspeed is still used in the rate term.
glm::vec3 air_accelerate(glm::vec3 vel, glm::vec3 wishdir, float wishspeed,
                         float accel, float wish_cap, float dt);

// Apply gravity to the y component (Y-up world).
glm::vec3 apply_gravity(glm::vec3 vel, float gravity, float dt);

} // namespace qlike::physics
