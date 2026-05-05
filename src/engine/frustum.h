#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace qlike {

struct Frustum {
    glm::vec4 planes[6];  // ax + by + cz + d ≥ 0 means "inside"
};

// Extract the 6 frustum planes from a view-projection matrix (Vulkan clip
// space: x ∈ [-w, w], y ∈ [-w, w], z ∈ [0, w]).
Frustum extract_frustum(const glm::mat4& view_proj);

// Conservative AABB-vs-frustum test: returns true unless the AABB lies on the
// fully-outside side of any single plane.
bool aabb_visible(const Frustum& f, const glm::vec3& mn, const glm::vec3& mx);

} // namespace qlike
