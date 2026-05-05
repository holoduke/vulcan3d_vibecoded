#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

namespace qlike {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Mesh {
    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_alloc = nullptr;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VmaAllocation index_alloc = nullptr;
    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
};

void destroy_mesh(VmaAllocator alloc, Mesh& mesh);

// Build a unit cube mesh (extents -0.5..+0.5) with per-face flat normals.
// Uses 24 vertices (4 per face for sharp normals) and 36 indices.
Mesh create_cube_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                      uint32_t queue_family);

// Cylinder mesh aligned with the Y axis (matching Jolt's CylinderShape
// convention), unit radius, half-length 1 (so total length 2). Instances
// scale and rotate it. Smooth side normals, flat caps.
Mesh create_cylinder_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                          uint32_t queue_family, int segments = 16);

// Upload an arbitrary indexed mesh (e.g. from a glTF or procedural builder).
// Buffers are device-local and wear the AS-input usage bits so they can also
// be referenced by a BLAS build.
Mesh create_mesh_from_data(VkDevice device, VmaAllocator alloc, VkQueue queue,
                           uint32_t queue_family,
                           const Vertex* vertices, uint32_t vertex_count,
                           const uint32_t* indices, uint32_t index_count);

} // namespace qlike
