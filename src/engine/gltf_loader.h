#pragma once

#include "engine/mesh.h"

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <string>
#include <vector>

namespace qlike {

// One glTF primitive flattened into raw data ready for upload + BLAS build.
// Node-hierarchy transforms are pre-baked into vertex positions / normals so
// every primitive can be placed in viewmodel space by a single root offset.
struct GltfPrimitive {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    glm::vec4 base_color = glm::vec4(0.6f, 0.6f, 0.65f, 1.0f);
};

struct GltfModel {
    std::vector<GltfPrimitive> primitives;
    glm::vec3 aabb_min{ 0.0f };
    glm::vec3 aabb_max{ 0.0f };
};

// Load a glTF or GLB file. Returns an empty model on file-not-found or parse
// failure (logs the reason). Pre-bakes node hierarchy into vertex positions
// using triangulated POSITION + NORMAL accessors. Other vertex attributes
// (UV, tangent, joint weights) are dropped — the engine doesn't use them yet.
GltfModel load_gltf(const std::string& path);

} // namespace qlike
