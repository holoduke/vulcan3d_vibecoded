#include "game/player.h"

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace qlike::game {

namespace {
constexpr float kHalfPi   = 1.57079632679f;
constexpr float kPitchCap = kHalfPi - 0.01f;
}

glm::vec3 Player::forward() const {
    float cp = std::cos(pitch);
    return { std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp };
}

glm::vec3 Player::right() const {
    return { std::cos(yaw), 0.0f, std::sin(yaw) };
}

glm::mat4 Player::view_matrix() const {
    glm::vec3 eye = eye_position();
    return glm::lookAt(eye, eye + forward(), glm::vec3(0.0f, 1.0f, 0.0f));
}

void update_player(Player& p, const PlayerInput& in, const Level& world, float dt) {
    update_player(p, in, std::span<const collision::AABB>(world.aabbs), dt);
}

void update_player(Player& p, const PlayerInput& in,
                   std::span<const collision::AABB> world_aabbs, float dt) {
    // Mouse-look has been moved to the render loop (so it runs at frame rate
    // for low latency, decoupled from fixed-timestep physics). Calls passing
    // mouse_dx/dy here would still apply, but the engine now passes 0.
    if (in.mouse_dx != 0.0f || in.mouse_dy != 0.0f) {
        p.yaw   += in.mouse_dx * Player::kMouseSensitivity;
        p.pitch -= in.mouse_dy * Player::kMouseSensitivity;
        p.pitch = std::clamp(p.pitch, -kPitchCap, kPitchCap);
    }

    // Smooth crouch lerp at ~8/sec (≈125 ms full transition). Decoupled from
    // hitbox to avoid the "uncrouch into a low ceiling" trap.
    {
        float target = in.crawl ? 1.0f : 0.0f;
        float k = std::min(1.0f, 8.0f * dt);
        p.crouch_factor += (target - p.crouch_factor) * k;
        p.crouch_factor = std::clamp(p.crouch_factor, 0.0f, 1.0f);
    }

    // Build wish direction from input + horizontal axes (no pitch).
    glm::vec3 fwd_h(std::sin(p.yaw), 0.0f, -std::cos(p.yaw));
    glm::vec3 right_h(std::cos(p.yaw), 0.0f, std::sin(p.yaw));

    glm::vec3 wish(0.0f);
    if (in.fwd)   wish += fwd_h;
    if (in.back)  wish -= fwd_h;
    if (in.right) wish += right_h;
    if (in.left)  wish -= right_h;

    float wish_len_sq = glm::dot(wish, wish);
    glm::vec3 wishdir(0.0f);
    float wishspeed = 0.0f;
    if (wish_len_sq > 1e-6f) {
        float len = std::sqrt(wish_len_sq);
        wishdir = wish / len;
        // Sprint wins over crawl if both keys are held. Air movement keeps
        // the base air_wish_cap regardless — sprint/crawl are ground-only.
        float speed_mult = 1.0f;
        if      (in.sprint) speed_mult = 1.5f;
        else if (in.crawl)  speed_mult = 0.4f;
        wishspeed = kPlayerMove.max_walk_speed * speed_mult;
    }

    // Apply ground friction or air drag style behavior, then accelerate.
    if (p.on_ground) {
        // 0.05 m/s snap threshold (matches Quake's "1 unit/s" for our meters world).
        p.velocity = physics::apply_friction(p.velocity,
            kPlayerMove.ground_friction, kPlayerMove.stop_speed, dt, 0.05f);
        p.velocity = physics::accelerate(p.velocity, wishdir, wishspeed,
            kPlayerMove.ground_accel, dt);
    } else {
        p.velocity = physics::air_accelerate(p.velocity, wishdir, wishspeed,
            kPlayerMove.air_accel, kPlayerMove.air_wish_cap, dt);
    }

    // Gravity (per-tick override from PlayerInput so the menu can drive it).
    p.velocity = physics::apply_gravity(p.velocity, in.gravity, dt);

    // Jump: instantaneous upward kick, only when grounded.
    if (in.jump && p.on_ground) {
        p.velocity.y = kPlayerMove.jump_speed;
        p.on_ground = false;
    }

    auto move = collision::slide_move(p.shape(), p.position, p.velocity, world_aabbs, dt);
    p.position = move.position;
    p.velocity = move.velocity;
    p.on_ground = move.grounded;

    // Step-up smoothing — the renderer subtracts step_smooth_offset from the
    // eye y, so an instant physics jump up a stair reads as a smooth ramp.
    // Add this tick's step amount, then decay exponentially. Half-life
    // ~70 ms (decay rate 10/s) feels natural without lagging perceptibly.
    p.step_smooth_offset += move.step_amount;
    p.step_smooth_offset *= std::max(0.0f, 1.0f - 10.0f * dt);
    // Cap so a chain of steps within one tick can't accumulate runaway.
    p.step_smooth_offset = std::clamp(p.step_smooth_offset, -0.6f, 0.6f);
}

} // namespace qlike::game
