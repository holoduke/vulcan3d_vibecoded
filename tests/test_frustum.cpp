#include "engine/frustum.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/gtc/matrix_transform.hpp>

using qlike::Frustum;
using qlike::extract_frustum;
using qlike::aabb_visible;
using Catch::Approx;

namespace {
glm::mat4 std_view_proj() {
    // Eye at +Z 10, looking at origin, FOV 80°, aspect 1, near 0.1, far 100.
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 10.0f),
                                 glm::vec3(0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(80.0f), 1.0f, 0.1f, 100.0f);
    proj[1][1] *= -1.0f;  // Vulkan Y-flip — same convention as the engine
    return proj * view;
}
}

TEST_CASE("frustum extracts six normalized planes") {
    Frustum f = extract_frustum(std_view_proj());
    for (int i = 0; i < 6; ++i) {
        float n = std::sqrt(f.planes[i].x * f.planes[i].x +
                            f.planes[i].y * f.planes[i].y +
                            f.planes[i].z * f.planes[i].z);
        REQUIRE(n == Approx(1.0f).margin(1e-3f));
    }
}

TEST_CASE("aabb at origin is visible from a +Z camera") {
    Frustum f = extract_frustum(std_view_proj());
    REQUIRE(aabb_visible(f, glm::vec3(-0.5f), glm::vec3(0.5f)));
}

TEST_CASE("aabb behind the camera is culled") {
    Frustum f = extract_frustum(std_view_proj());
    // Camera is at +Z 10 looking at origin, so anything significantly past
    // +Z 10 (behind it) must fail the near-plane test.
    REQUIRE_FALSE(aabb_visible(f,
                               glm::vec3(-1.0f, -1.0f, 30.0f),
                               glm::vec3( 1.0f,  1.0f, 35.0f)));
}

TEST_CASE("aabb far past the far plane is culled") {
    Frustum f = extract_frustum(std_view_proj());
    REQUIRE_FALSE(aabb_visible(f,
                               glm::vec3(-1.0f, -1.0f, -200.0f),
                               glm::vec3( 1.0f,  1.0f, -150.0f)));
}

TEST_CASE("aabb fully outside one side of the frustum is culled") {
    Frustum f = extract_frustum(std_view_proj());
    // Way to the right; camera is looking down -Z, so a box centered at x=200
    // is entirely outside the right plane no matter the depth.
    REQUIRE_FALSE(aabb_visible(f,
                               glm::vec3(200.0f, -1.0f, -5.0f),
                               glm::vec3(220.0f,  1.0f,  5.0f)));
}

TEST_CASE("aabb straddling a plane is still visible") {
    Frustum f = extract_frustum(std_view_proj());
    // A wide AABB that pokes outside the frustum on one side but still has
    // part inside should be reported visible.
    REQUIRE(aabb_visible(f,
                         glm::vec3(-50.0f, -1.0f, -5.0f),
                         glm::vec3(  0.5f,  1.0f,  5.0f)));
}
