#include "engine/gltf_loader.h"

#include "engine/log.h"

// tinygltf is header-only. We instantiate the parser implementation here and
// suppress its image-decoder paths since we don't render textures yet — that
// drops the entire stb_image / stb_image_write include surface.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include <tiny_gltf.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>

namespace qlike {

namespace {

glm::mat4 node_transform(const tinygltf::Node& n) {
    if (n.matrix.size() == 16) {
        glm::mat4 m(1.0f);
        for (int i = 0; i < 16; ++i) {
            glm::value_ptr(m)[i] = static_cast<float>(n.matrix[i]);
        }
        return m;
    }
    glm::vec3 t(0.0f), s(1.0f);
    glm::quat r(1, 0, 0, 0);
    if (n.translation.size() == 3) {
        t = glm::vec3(static_cast<float>(n.translation[0]),
                      static_cast<float>(n.translation[1]),
                      static_cast<float>(n.translation[2]));
    }
    if (n.scale.size() == 3) {
        s = glm::vec3(static_cast<float>(n.scale[0]),
                      static_cast<float>(n.scale[1]),
                      static_cast<float>(n.scale[2]));
    }
    if (n.rotation.size() == 4) {
        r = glm::quat(static_cast<float>(n.rotation[3]),
                      static_cast<float>(n.rotation[0]),
                      static_cast<float>(n.rotation[1]),
                      static_cast<float>(n.rotation[2]));
    }
    return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) *
           glm::scale(glm::mat4(1.0f), s);
}

const float* accessor_floats(const tinygltf::Model& m, int idx,
                             size_t& count, size_t& stride_floats) {
    const auto& acc = m.accessors[idx];
    const auto& view = m.bufferViews[acc.bufferView];
    const auto& buf = m.buffers[view.buffer];
    int comps = tinygltf::GetNumComponentsInType(acc.type);
    count = acc.count;
    size_t element_size = comps * sizeof(float);
    size_t stride_bytes = view.byteStride ? view.byteStride : element_size;
    stride_floats = stride_bytes / sizeof(float);
    return reinterpret_cast<const float*>(
        buf.data.data() + view.byteOffset + acc.byteOffset);
}

void process_primitive(const tinygltf::Model& model,
                       const tinygltf::Primitive& prim,
                       const glm::mat4& xform,
                       GltfPrimitive& out) {
    auto pos_it = prim.attributes.find("POSITION");
    if (pos_it == prim.attributes.end()) return;
    int pos_acc = pos_it->second;

    size_t pos_count = 0, pos_stride = 0;
    const float* pos_data = accessor_floats(model, pos_acc, pos_count, pos_stride);

    const float* nrm_data = nullptr;
    size_t nrm_stride = 0, nrm_count = 0;
    auto nrm_it = prim.attributes.find("NORMAL");
    if (nrm_it != prim.attributes.end()) {
        nrm_data = accessor_floats(model, nrm_it->second, nrm_count, nrm_stride);
    }
    const float* uv_data = nullptr;
    size_t uv_stride = 0, uv_count = 0;
    auto uv_it = prim.attributes.find("TEXCOORD_0");
    if (uv_it != prim.attributes.end()) {
        uv_data = accessor_floats(model, uv_it->second, uv_count, uv_stride);
    }

    const glm::mat3 normal_mat = glm::mat3(glm::transpose(glm::inverse(xform)));

    out.vertices.reserve(out.vertices.size() + pos_count);
    uint32_t base = static_cast<uint32_t>(out.vertices.size());
    for (size_t i = 0; i < pos_count; ++i) {
        glm::vec4 p(pos_data[i * pos_stride + 0],
                    pos_data[i * pos_stride + 1],
                    pos_data[i * pos_stride + 2], 1.0f);
        glm::vec3 wp(xform * p);
        glm::vec3 n(0.0f, 1.0f, 0.0f);
        if (nrm_data && i < nrm_count) {
            n = glm::vec3(nrm_data[i * nrm_stride + 0],
                          nrm_data[i * nrm_stride + 1],
                          nrm_data[i * nrm_stride + 2]);
            n = glm::normalize(normal_mat * n);
        }
        glm::vec2 uv(0.0f);
        if (uv_data && i < uv_count) {
            uv = glm::vec2(uv_data[i * uv_stride + 0],
                           uv_data[i * uv_stride + 1]);
        }
        out.vertices.push_back({ wp, n, uv });
    }

    // Indices: convert whatever component type to uint32_t.
    if (prim.indices < 0) {
        // Non-indexed: synthesize a sequential index list.
        for (size_t i = 0; i < pos_count; ++i) {
            out.indices.push_back(base + static_cast<uint32_t>(i));
        }
        return;
    }
    const auto& acc = model.accessors[prim.indices];
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[view.buffer];
    const unsigned char* base_ptr = buf.data.data() + view.byteOffset + acc.byteOffset;
    out.indices.reserve(out.indices.size() + acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        uint32_t idx = 0;
        switch (acc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                idx = base_ptr[i];
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                uint16_t v;
                std::memcpy(&v, base_ptr + i * sizeof(uint16_t), sizeof(uint16_t));
                idx = v;
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                uint32_t v;
                std::memcpy(&v, base_ptr + i * sizeof(uint32_t), sizeof(uint32_t));
                idx = v;
                break;
            }
            default:
                idx = 0;
                break;
        }
        out.indices.push_back(base + idx);
    }
}

void walk_node(const tinygltf::Model& model, int node_idx,
               const glm::mat4& parent, std::vector<GltfPrimitive>& out) {
    const auto& node = model.nodes[node_idx];
    glm::mat4 xform = parent * node_transform(node);
    if (node.mesh >= 0) {
        const auto& mesh = model.meshes[node.mesh];
        for (const auto& prim : mesh.primitives) {
            GltfPrimitive p{};
            // Material base color factor, if any.
            if (prim.material >= 0 &&
                static_cast<size_t>(prim.material) < model.materials.size()) {
                const auto& mat = model.materials[prim.material];
                if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
                    p.base_color = glm::vec4(
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[0]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[1]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[2]),
                        static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[3]));
                }
            }
            process_primitive(model, prim, xform, p);
            if (!p.vertices.empty() && !p.indices.empty()) {
                out.push_back(std::move(p));
            }
        }
    }
    for (int child : node.children) walk_node(model, child, xform, out);
}

} // namespace

GltfModel load_gltf(const std::string& path) {
    GltfModel out{};
    if (!std::filesystem::exists(path)) {
        log::infof("[gltf] no asset at %s — using procedural viewmodel", path.c_str());
        return out;
    }

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = false;
    bool is_glb = path.size() >= 4 &&
                  (path.compare(path.size() - 4, 4, ".glb") == 0 ||
                   path.compare(path.size() - 4, 4, ".GLB") == 0);
    if (is_glb) {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }
    if (!warn.empty()) log::infof("[gltf] %s: %s", path.c_str(), warn.c_str());
    if (!ok) {
        log::errorf("[gltf] failed to load %s: %s", path.c_str(), err.c_str());
        return out;
    }

    int scene_idx = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (model.scenes.empty()) {
        log::errorf("[gltf] %s has no scenes", path.c_str());
        return out;
    }
    const auto& scene = model.scenes[scene_idx];
    for (int root : scene.nodes) {
        walk_node(model, root, glm::mat4(1.0f), out.primitives);
    }

    // AABB across all primitives — lets the caller normalize/scale to fit.
    glm::vec3 mn(std::numeric_limits<float>::max());
    glm::vec3 mx(std::numeric_limits<float>::lowest());
    for (const auto& p : out.primitives) {
        for (const auto& v : p.vertices) {
            mn = glm::min(mn, v.position);
            mx = glm::max(mx, v.position);
        }
    }
    out.aabb_min = mn;
    out.aabb_max = mx;

    size_t total_v = 0, total_i = 0;
    for (const auto& p : out.primitives) {
        total_v += p.vertices.size();
        total_i += p.indices.size();
    }
    log::infof("[gltf] loaded %s: %zu primitives, %zu verts, %zu indices, "
               "extent=(%.2f,%.2f,%.2f)",
               path.c_str(), out.primitives.size(), total_v, total_i,
               out.aabb_max.x - out.aabb_min.x,
               out.aabb_max.y - out.aabb_min.y,
               out.aabb_max.z - out.aabb_min.z);
    return out;
}

} // namespace qlike
