#include "game/physics.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/geometric.hpp>

using qlike::physics::accelerate;
using qlike::physics::air_accelerate;
using qlike::physics::apply_friction;
using qlike::physics::apply_gravity;
using Catch::Approx;

namespace {
constexpr float kDt = 1.0f / 60.0f;
}

TEST_CASE("apply_friction reduces horizontal speed") {
    glm::vec3 v(200.0f, 0.0f, 0.0f);
    glm::vec3 r = apply_friction(v, 6.0f, 100.0f, kDt);
    REQUIRE(r.x < v.x);
    REQUIRE(r.x > 0.0f);
    REQUIRE(r.y == Approx(0.0f));
    REQUIRE(r.z == Approx(0.0f));
}

TEST_CASE("apply_friction snaps to zero below 1 unit/s") {
    glm::vec3 v(0.5f, 100.0f, 0.0f);
    glm::vec3 r = apply_friction(v, 6.0f, 100.0f, kDt);
    REQUIRE(r.x == Approx(0.0f));
    REQUIRE(r.z == Approx(0.0f));
    REQUIRE(r.y == Approx(100.0f));  // y untouched
}

TEST_CASE("apply_friction preserves y component") {
    glm::vec3 v(200.0f, 50.0f, 0.0f);
    glm::vec3 r = apply_friction(v, 6.0f, 100.0f, kDt);
    REQUIRE(r.y == Approx(50.0f));
}

TEST_CASE("apply_friction uses stop_speed floor for control") {
    // At low speeds the drop is computed against stop_speed, not actual speed.
    glm::vec3 fast(200.0f, 0.0f, 0.0f);
    glm::vec3 slow(50.0f, 0.0f, 0.0f);
    glm::vec3 rf = apply_friction(fast, 6.0f, 100.0f, kDt);
    glm::vec3 rs = apply_friction(slow, 6.0f, 100.0f, kDt);
    // Slow loses an absolute drop computed from stop_speed (100), not 50.
    float drop_slow = 50.0f - rs.x;
    float expected_slow_drop = 100.0f * 6.0f * kDt;
    REQUIRE(drop_slow == Approx(expected_slow_drop).margin(0.01f));
    // Fast loses an absolute drop computed from current speed (200).
    float drop_fast = 200.0f - rf.x;
    float expected_fast_drop = 200.0f * 6.0f * kDt;
    REQUIRE(drop_fast == Approx(expected_fast_drop).margin(0.01f));
}

TEST_CASE("accelerate adds speed in wishdir from rest") {
    glm::vec3 v(0.0f);
    glm::vec3 wishdir(1.0f, 0.0f, 0.0f);
    glm::vec3 r = accelerate(v, wishdir, 320.0f, 10.0f, kDt);
    // accel_speed = 10 * (1/60) * 320 = 53.33
    REQUIRE(r.x == Approx(53.333f).margin(0.01f));
    REQUIRE(r.y == Approx(0.0f));
    REQUIRE(r.z == Approx(0.0f));
}

TEST_CASE("accelerate caps at wishspeed along wishdir") {
    // Already moving at wishspeed; one more tick must not exceed it.
    glm::vec3 v(320.0f, 0.0f, 0.0f);
    glm::vec3 wishdir(1.0f, 0.0f, 0.0f);
    glm::vec3 r = accelerate(v, wishdir, 320.0f, 10.0f, kDt);
    REQUIRE(r.x == Approx(320.0f));
}

TEST_CASE("accelerate is no-op if already past wishspeed") {
    glm::vec3 v(500.0f, 0.0f, 0.0f);
    glm::vec3 wishdir(1.0f, 0.0f, 0.0f);
    glm::vec3 r = accelerate(v, wishdir, 320.0f, 10.0f, kDt);
    REQUIRE(r.x == Approx(500.0f));
}

TEST_CASE("accelerate cannot move you backward") {
    glm::vec3 v(100.0f, 0.0f, 0.0f);
    glm::vec3 wishdir(-1.0f, 0.0f, 0.0f);
    glm::vec3 r = accelerate(v, wishdir, 320.0f, 10.0f, kDt);
    // current along wishdir = -100, add = 320 - (-100) = 420
    // accel_speed = 10*dt*320 = 53.33, less than add
    // so r = v + 53.33 * wishdir = (100 - 53.33, 0, 0)
    REQUIRE(r.x == Approx(46.667f).margin(0.01f));
}

TEST_CASE("air_accelerate caps wishspeed at wish_cap") {
    glm::vec3 v(0.0f);
    glm::vec3 wishdir(1.0f, 0.0f, 0.0f);
    glm::vec3 r = air_accelerate(v, wishdir, 320.0f, 10.0f, 30.0f, kDt);
    // capped wishspeed = 30. add = 30 - 0 = 30.
    // accel_speed = 10 * dt * 30 = 5.0
    REQUIRE(r.x == Approx(5.0f).margin(0.01f));
}

TEST_CASE("air_accelerate enables strafe-jump speed gain perpendicular to motion") {
    // Player flying forward at 320, strafing sideways. Wishdir is at 45deg.
    // Component along wishdir = 320 * cos(45) = ~226. Capped wishspeed = 30.
    // So along wishdir, current (226) is > capped (30) -> no add along wishdir.
    // BUT: the trick happens when the wishdir is more sideways than forward, so
    // current along wishdir is small enough that capped speed allows acceleration.
    // Here we verify the simpler case: if wishdir is purely sideways, add WILL apply.
    glm::vec3 v(320.0f, 0.0f, 0.0f);          // moving forward
    glm::vec3 wishdir(0.0f, 0.0f, 1.0f);      // wish to strafe sideways
    glm::vec3 r = air_accelerate(v, wishdir, 320.0f, 10.0f, 30.0f, kDt);
    REQUIRE(r.x == Approx(320.0f));            // forward speed preserved
    REQUIRE(r.z > 0.0f);                       // gained sideways speed
    // Combined speed > 320 -> faster than max ground speed. That's strafe-jumping.
    REQUIRE(glm::length(r) > 320.0f);
}

TEST_CASE("apply_gravity only affects y") {
    glm::vec3 v(100.0f, 0.0f, 50.0f);
    glm::vec3 r = apply_gravity(v, 800.0f, kDt);
    REQUIRE(r.x == Approx(100.0f));
    REQUIRE(r.z == Approx(50.0f));
    REQUIRE(r.y == Approx(-800.0f * kDt));
}
