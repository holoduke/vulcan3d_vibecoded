#include "game/physics.h"

#include <algorithm>
#include <cmath>

namespace qlike::physics {

glm::vec3 apply_friction(glm::vec3 vel, float friction, float stop_speed, float dt,
                         float snap_threshold) {
    float horiz_sq = vel.x * vel.x + vel.z * vel.z;
    if (horiz_sq < snap_threshold * snap_threshold) {
        return glm::vec3(0.0f, vel.y, 0.0f);
    }
    float speed = std::sqrt(horiz_sq);
    float control = speed < stop_speed ? stop_speed : speed;
    float drop = control * friction * dt;
    float newspeed = std::max(0.0f, speed - drop);
    float scale = newspeed / speed;
    return glm::vec3(vel.x * scale, vel.y, vel.z * scale);
}

glm::vec3 accelerate(glm::vec3 vel, glm::vec3 wishdir, float wishspeed,
                     float accel, float dt) {
    float current = vel.x * wishdir.x + vel.y * wishdir.y + vel.z * wishdir.z;
    float add = wishspeed - current;
    if (add <= 0.0f) return vel;
    float accel_speed = accel * dt * wishspeed;
    if (accel_speed > add) accel_speed = add;
    return glm::vec3(vel.x + accel_speed * wishdir.x,
                     vel.y + accel_speed * wishdir.y,
                     vel.z + accel_speed * wishdir.z);
}

glm::vec3 air_accelerate(glm::vec3 vel, glm::vec3 wishdir, float wishspeed,
                         float accel, float wish_cap, float dt) {
    float capped = std::min(wishspeed, wish_cap);
    // Quake3 PM_AirAccelerate caps wishspeed THEN calls regular accelerate with that
    // capped value. The strafe-jump trick falls out because along-wishdir current speed
    // is what's checked against the (low) cap, while the existing horizontal velocity
    // can be near-perpendicular to wishdir and thus dot-product-tiny.
    return accelerate(vel, wishdir, capped, accel, dt);
}

glm::vec3 apply_gravity(glm::vec3 vel, float gravity, float dt) {
    return glm::vec3(vel.x, vel.y - gravity * dt, vel.z);
}

} // namespace qlike::physics
