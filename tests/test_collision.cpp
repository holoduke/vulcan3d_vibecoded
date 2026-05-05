#include "game/collision.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>

using qlike::collision::AABB;
using qlike::collision::aabb_overlap;
using qlike::collision::sweep_aabb;
using qlike::collision::slide_move;
using Catch::Approx;

namespace {
AABB unit(glm::vec3 center) {
    return AABB{ center - glm::vec3(0.5f), center + glm::vec3(0.5f) };
}
}

TEST_CASE("aabb_overlap detects intersecting boxes") {
    AABB a{ glm::vec3(0,0,0), glm::vec3(1,1,1) };
    AABB b{ glm::vec3(0.5f,0.5f,0.5f), glm::vec3(2,2,2) };
    REQUIRE(aabb_overlap(a, b));
}

TEST_CASE("aabb_overlap rejects disjoint boxes") {
    AABB a{ glm::vec3(0), glm::vec3(1) };
    AABB b{ glm::vec3(2), glm::vec3(3) };
    REQUIRE_FALSE(aabb_overlap(a, b));
}

TEST_CASE("aabb_overlap rejects touching (strict overlap)") {
    AABB a{ glm::vec3(0), glm::vec3(1) };
    AABB b{ glm::vec3(1, 0, 0), glm::vec3(2, 1, 1) };
    REQUIRE_FALSE(aabb_overlap(a, b));
}

TEST_CASE("slide_move depenetrates if spawned inside a wall") {
    AABB player{ glm::vec3(-0.5f), glm::vec3(0.5f) };
    glm::vec3 pos(0.0f, 0.0f, 0.0f);          // CENTER of player
    glm::vec3 vel(0.0f);                       // no input motion
    std::array<AABB, 1> world{
        AABB{ glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 0.2f, 1.0f) }
    };
    auto r = slide_move(player, pos, vel, world, 1.0f / 60.0f);
    // Player should have been pushed up so its bottom (pos.y - 0.5) >= top of brush (0.2).
    REQUIRE(r.position.y - 0.5f >= 0.2f - 1e-3f);
}

TEST_CASE("sweep_aabb hits a box directly in front") {
    AABB moving{ glm::vec3(0), glm::vec3(1) };
    AABB fixed{ glm::vec3(3, 0, 0), glm::vec3(4, 1, 1) };
    glm::vec3 disp(4.0f, 0.0f, 0.0f);
    auto r = sweep_aabb(moving, disp, fixed);
    REQUIRE(r.hit);
    REQUIRE(r.t == Approx(0.5f));  // 2 units to gap, displacement 4 -> t=0.5
    REQUIRE(r.normal.x == Approx(-1.0f));
    REQUIRE(r.normal.y == Approx(0.0f));
    REQUIRE(r.normal.z == Approx(0.0f));
}

TEST_CASE("sweep_aabb misses when path goes past") {
    AABB moving{ glm::vec3(0), glm::vec3(1) };
    AABB fixed{ glm::vec3(3, 5, 0), glm::vec3(4, 6, 1) };
    glm::vec3 disp(4.0f, 0.0f, 0.0f);
    auto r = sweep_aabb(moving, disp, fixed);
    REQUIRE_FALSE(r.hit);
}

TEST_CASE("sweep_aabb misses when displacement points away") {
    AABB moving{ glm::vec3(0), glm::vec3(1) };
    AABB fixed{ glm::vec3(3, 0, 0), glm::vec3(4, 1, 1) };
    glm::vec3 disp(-4.0f, 0.0f, 0.0f);
    auto r = sweep_aabb(moving, disp, fixed);
    REQUIRE_FALSE(r.hit);
}

TEST_CASE("sweep_aabb returns hit with correct y-down normal when falling") {
    AABB moving{ glm::vec3(0, 5, 0), glm::vec3(1, 6, 1) };
    AABB floor{ glm::vec3(-10, 0, -10), glm::vec3(10, 1, 10) };
    glm::vec3 disp(0.0f, -10.0f, 0.0f);  // falling
    auto r = sweep_aabb(moving, disp, floor);
    REQUIRE(r.hit);
    REQUIRE(r.normal.y == Approx(1.0f));  // floor pushes up against the falling box
}

TEST_CASE("sweep_aabb reports no hit for initially overlapping boxes") {
    AABB moving{ glm::vec3(0), glm::vec3(2) };
    AABB fixed{ glm::vec3(1), glm::vec3(3) };
    auto r = sweep_aabb(moving, glm::vec3(1, 0, 0), fixed);
    REQUIRE_FALSE(r.hit);
}

TEST_CASE("slide_move in empty world moves freely") {
    AABB player = unit(glm::vec3(0));
    glm::vec3 pos(0, 5, 0);
    glm::vec3 vel(100, 0, 0);
    std::array<AABB, 0> world{};
    auto r = slide_move(player, pos, vel, world, 1.0f / 60.0f);
    REQUIRE(r.position.x == Approx(pos.x + vel.x / 60.0f));
    REQUIRE(r.position.y == Approx(pos.y));
    REQUIRE_FALSE(r.grounded);
}

TEST_CASE("slide_move stops at a wall") {
    AABB player = unit(glm::vec3(0));
    glm::vec3 pos(0, 5, 0);
    glm::vec3 vel(1000, 0, 0);  // very fast into wall
    std::array<AABB, 1> world{
        AABB{ glm::vec3(2, 0, -10), glm::vec3(3, 10, 10) }
    };
    auto r = slide_move(player, pos, vel, world, 1.0f / 60.0f);
    // Player should be stopped before x=2 (its right edge can't pass x=2).
    REQUIRE(r.position.x + 0.5f <= 2.0f + 1e-3f);
}

TEST_CASE("slide_move sets grounded when landing on floor") {
    AABB player = unit(glm::vec3(0));
    glm::vec3 pos(0, 1.0f, 0);
    glm::vec3 vel(0, -200, 0);  // falling
    std::array<AABB, 1> world{
        AABB{ glm::vec3(-10, -1, -10), glm::vec3(10, 0, 10) }  // floor at y=0
    };
    auto r = slide_move(player, pos, vel, world, 1.0f / 60.0f);
    REQUIRE(r.grounded);
}

TEST_CASE("slide_move slides along a wall when moving diagonally into it") {
    AABB player = unit(glm::vec3(0));
    glm::vec3 pos(0, 5, 0);
    // Velocity has both forward and sideways components; wall blocks +x.
    glm::vec3 vel(500, 0, 100);
    std::array<AABB, 1> world{
        AABB{ glm::vec3(2, 0, -10), glm::vec3(3, 10, 10) }
    };
    auto r = slide_move(player, pos, vel, world, 1.0f / 60.0f);
    // x velocity should have been zeroed after hitting the wall, z should remain.
    REQUIRE(r.velocity.x == Approx(0.0f).margin(1e-3f));
    REQUIRE(r.velocity.z == Approx(100.0f));
    // Position advanced in z (slid along the wall).
    REQUIRE(r.position.z > 0.0f);
}
