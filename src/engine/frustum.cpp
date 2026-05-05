#include "engine/frustum.h"

#include <glm/geometric.hpp>

namespace qlike {

Frustum extract_frustum(const glm::mat4& vp) {
    glm::mat4 t = glm::transpose(vp);
    Frustum f;
    f.planes[0] = t[3] + t[0];   // left   (x ≥ -w)
    f.planes[1] = t[3] - t[0];   // right  (x ≤  w)
    f.planes[2] = t[3] + t[1];   // bottom (y ≥ -w)
    f.planes[3] = t[3] - t[1];   // top    (y ≤  w)
    f.planes[4] = t[2];          // near   (z ≥  0)  — Vulkan 0..w depth
    f.planes[5] = t[3] - t[2];   // far    (z ≤  w)
    for (int i = 0; i < 6; ++i) {
        float len = glm::length(glm::vec3(f.planes[i]));
        if (len > 1e-6f) f.planes[i] /= len;
    }
    return f;
}

bool aabb_visible(const Frustum& f, const glm::vec3& mn, const glm::vec3& mx) {
    for (int i = 0; i < 6; ++i) {
        const glm::vec4& p = f.planes[i];
        glm::vec3 c(p.x > 0 ? mx.x : mn.x,
                    p.y > 0 ? mx.y : mn.y,
                    p.z > 0 ? mx.z : mn.z);
        if (p.x * c.x + p.y * c.y + p.z * c.z + p.w < 0.0f) return false;
    }
    return true;
}

} // namespace qlike
