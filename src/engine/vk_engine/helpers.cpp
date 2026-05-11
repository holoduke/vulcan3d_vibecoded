// Implementation of the shared statics declared in
// src/engine/vk_engine/internal.h. Everything in this TU is engine-wide:
// validation messenger callback, the RT extension function-pointer table,
// the per-draw push-constant + per-frame UBO layouts, and a handful of
// small math / RNG helpers reached for by multiple sections of the engine.

#include "engine/vk_engine/internal.h"

#include <cmath>
#include <vector>

namespace qlike {

// 240 bytes — past the Vulkan 128-byte minimum guarantee but well within the
// 256-byte limit every modern desktop GPU exposes. If a target device caps
// at 128, this layout would need to spill into a small per-draw UBO.
static_assert(sizeof(PushConstants) == 256, "push constant layout");
// 352 → 368 (terrain_extra) → 400 (water_params + water_color)
//     → 432 (water_color_shallow + water_shore) → 448 (fog_band)
//     → 464 (terrain_rt_extra) → 512 (grass_color_top + bottom + ground)
//     → 528 (grass_color_ground_far) → 544 (grass_shadow_params).
static_assert(sizeof(SceneUBO) == 544, "scene ubo layout");

RtFuncs g_rt;

void load_rt_functions(VkDevice device) {
    g_rt.create_as = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    g_rt.destroy_as = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    g_rt.cmd_build_as = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    g_rt.get_as_device_addr = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    g_rt.get_as_build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    g_rt.cmd_write_as_props = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(
        vkGetDeviceProcAddr(device, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
    g_rt.cmd_copy_as = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCmdCopyAccelerationStructureKHR"));
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    const char* msg = data && data->pMessage ? data->pMessage : "(no message)";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        g_validation_error_count.fetch_add(1, std::memory_order_relaxed);
        log::errorf("[vk] %s", msg);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        g_validation_warning_count.fetch_add(1, std::memory_order_relaxed);
        log::warnf( "[vk] %s", msg);
    } else {
        log::infof( "[vk] %s", msg);
    }
    return VK_FALSE;
}

// Spark "temperature" gradient. `t` is 0 = cooled / dim, 1 = freshly emitted /
// hot orange-yellow. Peak intensity is intentionally below pure white so a
// cluster of sparks at frame-0 doesn't read as a "white blob" at the impact
// point — earlier versions used (3.2, 2.9, 2.3) which bloomed into a flash.
glm::vec3 spark_blackbody(float t) {
    t = glm::clamp(t, 0.0f, 1.0f);
    glm::vec3 c0(0.25f, 0.04f, 0.005f);   // dark red ember
    glm::vec3 c1(1.20f, 0.45f, 0.08f);    // orange
    glm::vec3 c2(1.80f, 1.20f, 0.30f);    // yellow-hot
    glm::vec3 c3(2.10f, 1.80f, 0.90f);    // peak — warm white, not pure white
    if (t < 0.33f)      return glm::mix(c0, c1, t / 0.33f);
    else if (t < 0.66f) return glm::mix(c1, c2, (t - 0.33f) / 0.33f);
    else                return glm::mix(c2, c3, (t - 0.66f) / 0.34f);
}

// Rotate the local +Y axis to point along `dir`, then translate to `pos`.
// Used to align the cylinder bullet mesh to its current velocity vector each
// frame — keeps the visual following the actual (gravity-bent) trajectory.
glm::mat4 align_local_y_to(glm::vec3 pos, glm::vec3 dir) {
    glm::vec3 from(0.0f, 1.0f, 0.0f);
    float dotp = glm::dot(from, dir);
    glm::quat q;
    if (dotp > 0.99999f) {
        q = glm::quat(1, 0, 0, 0);
    } else if (dotp < -0.99999f) {
        q = glm::angleAxis(3.14159265f, glm::vec3(1, 0, 0));
    } else {
        glm::vec3 axis = glm::normalize(glm::cross(from, dir));
        q = glm::angleAxis(std::acos(dotp), axis);
    }
    return glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(q);
}

// Smoothstep snap-back + half-sine spring-forward used by the viewmodel
// recoil animation. `t_remaining` decays from `duration` → 0; at duration
// the return is 0 (fully recovered), at fire (t_remaining=duration) we're
// in the snap-back portion.
float recoil_kick(float t_remaining, float duration, float stroke) {
    if (t_remaining <= 0.0f || duration <= 0.0f) return 0.0f;
    float t = 1.0f - (t_remaining / duration);   // 0 at fire → 1 at end
    if (t < 0.30f) {
        // Snap back: smooth ramp to full stroke at t=0.30.
        float k = t / 0.30f;
        return stroke * k * k * (3.0f - 2.0f * k);
    }
    // Spring forward: half-sine from full back to 0 between t=0.30 and t=1.0.
    float k = (t - 0.30f) / 0.70f;
    return stroke * std::cos(k * 1.57079632f);
}

// xorshift32 — fast deterministic RNG, plenty for box positions / spark dirs.
uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}
float frand(uint32_t& s) { return (xorshift32(s) >> 8) * (1.0f / 16777216.0f); }
float frand_range(uint32_t& s, float a, float b) { return a + frand(s) * (b - a); }

VkDeviceAddress buffer_device_address(VkDevice device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo i{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &i);
}

VkDeviceAddress as_device_address(VkDevice device, VkAccelerationStructureKHR as) {
    VkAccelerationStructureDeviceAddressInfoKHR i{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .pNext = nullptr,
        .accelerationStructure = as,
    };
    return g_rt.get_as_device_addr(device, &i);
}

// Write all scene descriptor bindings once. Buffer / TLAS / texture handles
// don't change after init, so this is one-shot — not per-frame.
//
// Texture views are passed in as kTextureCount-sized arrays (binding 3 =
// albedos, binding 4 = normals); a single shared sampler is used for both.
void write_scene_descriptors_once(
    VkDevice device, VkDescriptorSet set,
    VkBuffer ubo, VkAccelerationStructureKHR tlas, VkBuffer materials,
    VkBuffer prev_transforms,
    const VkImageView* albedo_views, const VkImageView* normal_views,
    uint32_t tex_count, VkSampler tex_sampler,
    const VkImageView* spom_height_views, uint32_t spom_count,
    VkImageView grass_mask_view,
    VkImageView fog_wisp_view) {

    VkDescriptorBufferInfo ubo_bi{ ubo, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo mat_bi{ materials, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo prev_bi{ prev_transforms, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSetAccelerationStructureKHR as_info{};
    as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_info.accelerationStructureCount = 1;
    as_info.pAccelerationStructures = &tlas;

    std::vector<VkDescriptorImageInfo> alb_infos(tex_count);
    std::vector<VkDescriptorImageInfo> nrm_infos(tex_count);
    for (uint32_t i = 0; i < tex_count; ++i) {
        alb_infos[i].sampler = tex_sampler;
        alb_infos[i].imageView = albedo_views[i];
        alb_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        nrm_infos[i].sampler = tex_sampler;
        nrm_infos[i].imageView = normal_views[i];
        nrm_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // SPOM displacement-map array at binding 12. Each slot maps to a
    // material that runs parallax in cube.frag; missing views fall back to
    // the slot-1 albedo so the descriptor is always valid (cube.frag
    // gates SPOM separately on texture index, so a fallback is never
    // sampled in practice — defensive only).
    std::vector<VkDescriptorImageInfo> spom_infos(spom_count);
    for (uint32_t i = 0; i < spom_count; ++i) {
        spom_infos[i].sampler = tex_sampler;
        spom_infos[i].imageView = spom_height_views[i] ? spom_height_views[i]
                                                       : albedo_views[1];
        spom_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkWriteDescriptorSet w[7]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &ubo_bi;

    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].pNext = &as_info;
    w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[2].dstSet = set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &mat_bi;

    w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[3].dstSet = set; w[3].dstBinding = 3; w[3].descriptorCount = tex_count;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[3].pImageInfo = alb_infos.data();

    w[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[4].dstSet = set; w[4].dstBinding = 4; w[4].descriptorCount = tex_count;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[4].pImageInfo = nrm_infos.data();

    w[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[5].dstSet = set; w[5].dstBinding = 5; w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[5].pBufferInfo = &prev_bi;

    w[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[6].dstSet = set; w[6].dstBinding = 12; w[6].descriptorCount = spom_count;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[6].pImageInfo = spom_infos.data();

    // Binding 13: grass-density mask. Falls back to slot-0 albedo if
    // the mask hasn't been baked yet (defensive — bake runs at level
    // load before this descriptor write, but the fallback keeps the
    // descriptor valid).
    VkDescriptorImageInfo grass_mask_bi{};
    grass_mask_bi.sampler = tex_sampler;
    grass_mask_bi.imageView = grass_mask_view ? grass_mask_view : albedo_views[0];
    grass_mask_bi.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w8{};
    w8.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w8.dstSet = set; w8.dstBinding = 13; w8.descriptorCount = 1;
    w8.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w8.pImageInfo = &grass_mask_bi;

    // Binding 14: fog wisp pattern (R8). Same defensive fallback to
    // slot-0 albedo if the bake hasn't run yet.
    VkDescriptorImageInfo fog_wisp_bi{};
    fog_wisp_bi.sampler = tex_sampler;
    fog_wisp_bi.imageView = fog_wisp_view ? fog_wisp_view : albedo_views[0];
    fog_wisp_bi.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w9{};
    w9.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w9.dstSet = set; w9.dstBinding = 14; w9.descriptorCount = 1;
    w9.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w9.pImageInfo = &fog_wisp_bi;

    VkWriteDescriptorSet writes[9] = { w[0], w[1], w[2], w[3],
                                        w[4], w[5], w[6], w8, w9 };
    vkUpdateDescriptorSets(device, 9, writes, 0, nullptr);
}

} // namespace qlike
