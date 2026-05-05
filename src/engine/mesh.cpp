#include "engine/mesh.h"

#include "engine/log.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qlike {

namespace {

void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        log::errorf("mesh vk call failed: %s (%d)", what, static_cast<int>(r));
        throw std::runtime_error(std::string("mesh vk error: ") + what);
    }
}

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
};

Buffer make_buffer(VmaAllocator alloc, VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   VmaMemoryUsage mem_usage,
                   VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .size = size, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = mem_usage;
    aci.flags = flags;
    Buffer b{};
    vk_check(vmaCreateBuffer(alloc, &bci, &aci, &b.handle, &b.alloc, nullptr),
             "vmaCreateBuffer");
    return b;
}

void copy_to_host_buffer(VmaAllocator alloc, VmaAllocation a,
                         const void* data, size_t bytes) {
    void* mapped = nullptr;
    vk_check(vmaMapMemory(alloc, a, &mapped), "vmaMapMemory");
    std::memcpy(mapped, data, bytes);
    vmaUnmapMemory(alloc, a);
}

void copy_buffer_immediate(VkDevice device, VkQueue queue, uint32_t qf,
                           VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = qf,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(device, &pci, nullptr, &pool), "vkCreateCommandPool tx");

    VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &ai, &cmd), "vkAllocateCommandBuffers tx");

    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer tx");
    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer tx");

    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = nullptr,
        .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr, .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr,
    };
    vk_check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit tx");
    vk_check(vkQueueWaitIdle(queue), "vkQueueWaitIdle tx");

    vkDestroyCommandPool(device, pool, nullptr);
}

} // namespace

// Stage on host-visible, copy to device-local. Adds the AS-input usage bit so
// the same buffers can be referenced by VK_KHR_acceleration_structure builds.
Mesh create_mesh_from_data(VkDevice device, VmaAllocator alloc, VkQueue queue,
                           uint32_t queue_family,
                           const Vertex* verts, uint32_t vert_count,
                           const uint32_t* indices, uint32_t idx_count) {
    const VkDeviceSize vbo_size = sizeof(Vertex) * vert_count;
    const VkDeviceSize ibo_size = sizeof(uint32_t) * idx_count;

    Buffer vbo_stage = make_buffer(alloc, vbo_size,
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VMA_MEMORY_USAGE_CPU_ONLY);
    copy_to_host_buffer(alloc, vbo_stage.alloc, verts, static_cast<size_t>(vbo_size));

    Buffer ibo_stage = make_buffer(alloc, ibo_size,
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VMA_MEMORY_USAGE_CPU_ONLY);
    copy_to_host_buffer(alloc, ibo_stage.alloc, indices, static_cast<size_t>(ibo_size));

    constexpr VkBufferUsageFlags kRtFlags =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    Buffer vbo = make_buffer(alloc, vbo_size,
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | kRtFlags,
                             VMA_MEMORY_USAGE_GPU_ONLY);
    Buffer ibo = make_buffer(alloc, ibo_size,
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | kRtFlags,
                             VMA_MEMORY_USAGE_GPU_ONLY);

    copy_buffer_immediate(device, queue, queue_family,
                          vbo_stage.handle, vbo.handle, vbo_size);
    copy_buffer_immediate(device, queue, queue_family,
                          ibo_stage.handle, ibo.handle, ibo_size);

    vmaDestroyBuffer(alloc, vbo_stage.handle, vbo_stage.alloc);
    vmaDestroyBuffer(alloc, ibo_stage.handle, ibo_stage.alloc);

    Mesh m{};
    m.vertex_buffer = vbo.handle;
    m.vertex_alloc = vbo.alloc;
    m.index_buffer = ibo.handle;
    m.index_alloc = ibo.alloc;
    m.vertex_count = vert_count;
    m.index_count = idx_count;
    return m;
}

void destroy_mesh(VmaAllocator alloc, Mesh& m) {
    if (m.vertex_buffer) vmaDestroyBuffer(alloc, m.vertex_buffer, m.vertex_alloc);
    if (m.index_buffer)  vmaDestroyBuffer(alloc, m.index_buffer,  m.index_alloc);
    m = {};
}

Mesh create_cube_mesh(VkDevice device, VmaAllocator alloc,
                      VkQueue queue, uint32_t queue_family) {
    // Unit cube, flat-shaded (one normal per face, so 4 unique verts per face).
    // UVs: each face is its own [0,1]² square with (0,0) at the lower-left
    // when looking at the face from outside the cube.
    constexpr float h = 0.5f;
    const Vertex verts[] = {
        // +X face (looking from +X toward origin, U along -Z, V along +Y)
        { { h,-h,-h}, { 1, 0, 0}, {1.0f, 0.0f} }, { { h, h,-h}, { 1, 0, 0}, {1.0f, 1.0f} },
        { { h, h, h}, { 1, 0, 0}, {0.0f, 1.0f} }, { { h,-h, h}, { 1, 0, 0}, {0.0f, 0.0f} },
        // -X face
        { {-h,-h, h}, {-1, 0, 0}, {1.0f, 0.0f} }, { {-h, h, h}, {-1, 0, 0}, {1.0f, 1.0f} },
        { {-h, h,-h}, {-1, 0, 0}, {0.0f, 1.0f} }, { {-h,-h,-h}, {-1, 0, 0}, {0.0f, 0.0f} },
        // +Y face (top, looking down: U along +X, V along +Z)
        { {-h, h,-h}, { 0, 1, 0}, {0.0f, 0.0f} }, { {-h, h, h}, { 0, 1, 0}, {0.0f, 1.0f} },
        { { h, h, h}, { 0, 1, 0}, {1.0f, 1.0f} }, { { h, h,-h}, { 0, 1, 0}, {1.0f, 0.0f} },
        // -Y face (bottom)
        { {-h,-h, h}, { 0,-1, 0}, {0.0f, 1.0f} }, { {-h,-h,-h}, { 0,-1, 0}, {0.0f, 0.0f} },
        { { h,-h,-h}, { 0,-1, 0}, {1.0f, 0.0f} }, { { h,-h, h}, { 0,-1, 0}, {1.0f, 1.0f} },
        // +Z face (looking from +Z back, U along +X, V along +Y)
        { {-h,-h, h}, { 0, 0, 1}, {0.0f, 0.0f} }, { { h,-h, h}, { 0, 0, 1}, {1.0f, 0.0f} },
        { { h, h, h}, { 0, 0, 1}, {1.0f, 1.0f} }, { {-h, h, h}, { 0, 0, 1}, {0.0f, 1.0f} },
        // -Z face
        { { h,-h,-h}, { 0, 0,-1}, {0.0f, 0.0f} }, { {-h,-h,-h}, { 0, 0,-1}, {1.0f, 0.0f} },
        { {-h, h,-h}, { 0, 0,-1}, {1.0f, 1.0f} }, { { h, h,-h}, { 0, 0,-1}, {0.0f, 1.0f} },
    };
    constexpr uint32_t kVtxCount = sizeof(verts) / sizeof(verts[0]);

    const uint32_t indices[] = {
        0, 1, 2,    0, 2, 3,
        4, 5, 6,    4, 6, 7,
        8, 9,10,    8,10,11,
       12,13,14,   12,14,15,
       16,17,18,   16,18,19,
       20,21,22,   20,22,23,
    };
    constexpr uint32_t kIdxCount = sizeof(indices) / sizeof(indices[0]);

    Mesh m = create_mesh_from_data(device, alloc, queue, queue_family,
                         verts, kVtxCount, indices, kIdxCount);
    log::infof("cube mesh created: %u verts, %u indices", kVtxCount, kIdxCount);
    return m;
}

Mesh create_cylinder_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                          uint32_t queue_family, int segments) {
    if (segments < 3) segments = 3;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float r  = 1.0f;
    constexpr float hh = 1.0f;

    std::vector<Vertex> verts;
    verts.reserve(segments * 2 + (segments + 1) * 2);
    std::vector<uint32_t> indices;
    indices.reserve(segments * 6 + segments * 6);

    // Side: smooth radial normals, top + bottom ring.
    for (int i = 0; i < segments; ++i) {
        float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
        float cs = std::cos(a), sn = std::sin(a);
        glm::vec3 n(cs, 0.0f, sn);
        float u = static_cast<float>(i) / static_cast<float>(segments);
        verts.push_back({ glm::vec3(r * cs,  hh, r * sn), n, glm::vec2(u, 1.0f) });
        verts.push_back({ glm::vec3(r * cs, -hh, r * sn), n, glm::vec2(u, 0.0f) });
    }
    for (int i = 0; i < segments; ++i) {
        int j = (i + 1) % segments;
        uint32_t a = static_cast<uint32_t>(i) * 2 + 0;  // top i
        uint32_t b = static_cast<uint32_t>(i) * 2 + 1;  // bot i
        uint32_t c = static_cast<uint32_t>(j) * 2 + 1;  // bot j
        uint32_t d = static_cast<uint32_t>(j) * 2 + 0;  // top j
        indices.push_back(a); indices.push_back(b); indices.push_back(c);
        indices.push_back(a); indices.push_back(c); indices.push_back(d);
    }

    // Top cap fan (normal +Y). Verts duplicated so the cap is flat-shaded.
    // UVs: planar (XZ → UV), centred at 0.5,0.5 with radius 0.5.
    uint32_t top_center = static_cast<uint32_t>(verts.size());
    verts.push_back({ glm::vec3(0.0f, hh, 0.0f), glm::vec3(0, 1, 0),
                      glm::vec2(0.5f, 0.5f) });
    uint32_t top_first = static_cast<uint32_t>(verts.size());
    for (int i = 0; i < segments; ++i) {
        float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
        float cs = std::cos(a), sn = std::sin(a);
        verts.push_back({ glm::vec3(r * cs, hh, r * sn),
                          glm::vec3(0, 1, 0),
                          glm::vec2(0.5f + 0.5f * cs, 0.5f + 0.5f * sn) });
    }
    for (int i = 0; i < segments; ++i) {
        uint32_t a = top_first + static_cast<uint32_t>(i);
        uint32_t b = top_first + static_cast<uint32_t>((i + 1) % segments);
        indices.push_back(top_center); indices.push_back(b); indices.push_back(a);
    }

    // Bottom cap fan (normal -Y).
    uint32_t bot_center = static_cast<uint32_t>(verts.size());
    verts.push_back({ glm::vec3(0.0f, -hh, 0.0f), glm::vec3(0, -1, 0),
                      glm::vec2(0.5f, 0.5f) });
    uint32_t bot_first = static_cast<uint32_t>(verts.size());
    for (int i = 0; i < segments; ++i) {
        float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
        float cs = std::cos(a), sn = std::sin(a);
        verts.push_back({ glm::vec3(r * cs, -hh, r * sn),
                          glm::vec3(0, -1, 0),
                          glm::vec2(0.5f + 0.5f * cs, 0.5f - 0.5f * sn) });
    }
    for (int i = 0; i < segments; ++i) {
        uint32_t a = bot_first + static_cast<uint32_t>(i);
        uint32_t b = bot_first + static_cast<uint32_t>((i + 1) % segments);
        indices.push_back(bot_center); indices.push_back(a); indices.push_back(b);
    }

    Mesh m = create_mesh_from_data(device, alloc, queue, queue_family,
                         verts.data(), static_cast<uint32_t>(verts.size()),
                         indices.data(), static_cast<uint32_t>(indices.size()));
    log::infof("cylinder mesh created: %u verts, %u indices (%d segments)",
               m.vertex_count, m.index_count, segments);
    return m;
}

} // namespace qlike
