// Scene descriptor set + UBO update. The TLAS / materials / prev_transforms
// SSBOs and per-texture sampler arrays all live behind one descriptor set
// (scene_desc_set_) bound by every world raster pipeline. Once init_rt()
// produces the TLAS handle, helpers.cpp's write_scene_descriptors_once()
// fills the bindings; from then on update_scene_ubo() is the only per-frame
// descriptor work (memcpy into a host-mapped UBO buffer).

#include "engine/vk_engine/internal.h"

#include <cmath>
#include <cstring>

namespace qlike {

void VulkanEngine::init_descriptors() {
    // Scene UBO buffer.
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(SceneUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &scene_ubo_buffer_, &scene_ubo_alloc_, nullptr),
                 "scene ubo buffer");
    }

    // Descriptor pool: 1 UBO + 1 AS + 2 SSBO (materials, prev_transforms) +
    // 2 sampler arrays (albedo[N], normal[N]).
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        // +1 for the heightmap shadow texture at binding 6.
        // +1 for the sun shadow map at binding 7 (sampler2DShadow).
        // +1 for the raw heightmap texture at binding 8 (R32_SFLOAT) —
        // sampled by terrain_raymarch.frag so the procedural raymarched
        // terrain follows the gameplay heightmap shape.
        // +3 for the low-res raymarch upscale targets (bindings 9..11):
        // color, motion vector, depth — sampled by the compose pass
        // when terrain_raymarch_scale < 1.0.
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTextureCount * 2 + 6 },
    };
    VkDescriptorPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 4,
        .pPoolSizes = sizes,
    };
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &scene_desc_pool_),
             "scene desc pool");

    // Bindings: 0=UBO, 1=TLAS, 2=materials, 3=albedo[N], 4=normal[N],
    //           5=prev_transforms, 6=heightmap shadow, 7=sun shadow map,
    //           8=raw heightmap (R32_SFLOAT) for terrain_raymarch.frag,
    //           9=LR raymarch color, 10=LR raymarch motion-vec, 11=LR
    //              raymarch depth — read by the compose pass when
    //              terrain_raymarch_scale < 1.
    VkDescriptorSetLayoutBinding bindings[12]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = kTextureCount;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = kTextureCount;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 6: pre-baked heightmap sun-shadow R8 texture. Sampled by
    // grass.vert AND cube.frag — the latter uses it as a distant-terrain
    // shadow fallback so the BLAS-vs-LOD-raster mismatch can't produce
    // false-hit shadow rays on far ridges.
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 7: sun shadow map (single-cascade D32). Sampled by grass.vert
    // as a sampler2DShadow for hardware PCF + comparison. Replaces the
    // heightmap-bake path going forward (binding 6 stays for fallback).
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 8: raw heightmap as R32_SFLOAT, sampled by
    // terrain_raymarch.frag so the procedural ray-marched terrain
    // follows the gameplay heightmap shape exactly (castle / cubes
    // sit correctly on the visible surface).
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 9..11: low-res raymarch upscale targets, sampled by the
    // compose pass when terrain_raymarch_scale < 1. Color and motion
    // are HDR / SFLOAT; depth is R32 sampled as a regular texture so
    // the compose can write gl_FragDepth = upscaled depth.
    for (int b = 9; b <= 11; ++b) {
        bindings[b].binding = static_cast<uint32_t>(b);
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .bindingCount = 12, .pBindings = bindings,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                         &scene_desc_set_layout_),
             "scene desc set layout");

    VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = scene_desc_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &scene_desc_set_layout_,
    };
    vk_check(vkAllocateDescriptorSets(device_, &ai, &scene_desc_set_),
             "scene desc set alloc");
    log::info("descriptors created");
}

void VulkanEngine::destroy_descriptors() {
    if (scene_desc_set_layout_) vkDestroyDescriptorSetLayout(device_, scene_desc_set_layout_, nullptr);
    if (scene_desc_pool_)       vkDestroyDescriptorPool(device_, scene_desc_pool_, nullptr);
    if (scene_ubo_buffer_)      vmaDestroyBuffer(allocator_, scene_ubo_buffer_, scene_ubo_alloc_);
    scene_desc_set_layout_ = VK_NULL_HANDLE;
    scene_desc_pool_ = VK_NULL_HANDLE;
    scene_desc_set_ = VK_NULL_HANDLE;
    scene_ubo_buffer_ = VK_NULL_HANDLE;
    scene_ubo_alloc_ = nullptr;
}

void VulkanEngine::update_scene_ubo() {
    // Per-frame: only update the UBO contents (memcpy to mapped buffer).
    // Descriptor set bindings are written once after init_rt().
    SceneUBO data{};
    float pitch = glm::radians(rt_.sun_pitch_deg);
    float yaw   = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::cos(pitch) * std::sin(yaw),
                  std::sin(pitch),
                  std::cos(pitch) * std::cos(yaw));
    sun = glm::normalize(sun);
    data.sun_direction = glm::vec4(sun, 0.0f);
    data.sun_color     = glm::vec4(rt_.sun_color, rt_.sun_intensity);
    data.ambient       = glm::vec4(rt_.ground_ambient, 1.0f);
    data.sky_color     = glm::vec4(rt_.sky_color, 1.0f);
    data.rt_flags = glm::ivec4(rt_.shadow_enabled ? 1 : 0,
                               rt_.shadow_samples,
                               rt_.ao_samples,
                               static_cast<int>(frame_number_));
    data.rt_params = glm::vec4(rt_.shadow_softness,
                               rt_.ao_radius,
                               rt_.ambient_strength,
                               rt_.shadow_strength);
    data.rt_flags2 = glm::ivec4(rt_.gi_samples,
                                rt_.reflections_enabled ? 1 : 0,
                                rt_.gi_bounces,
                                rt_.ao_mode);  // 0=off, 1=fast, 2=RTAO
    data.rt_params2 = glm::vec4(rt_.gi_strength,
                                rt_.gi_radius,
                                rt_.reflection_strength,
                                rt_.shadow_curve);
    glm::vec3 eye = player_.eye_position();
    data.camera_pos = glm::vec4(eye, 0.0f);
    // Distance LOD: 15 m = full samples (gameplay-relevant range — shadows
    // and AO are most visible up close); past 50 m fades to 1 sample so the
    // skybox-distance cubes don't waste a full ray budget per pixel.
    // .z carries gi_shadow_max_bounce — see RtSettings; cast to int in
    // cube.frag where it gates the per-bounce sun shadow ray.
    // .z = gi_shadow_max_bounce (cube.frag casts to int).
    // .w = ao_floor — see RtSettings; cube.frag remaps raw AO to
    //      [ao_floor, 1.0] so corner darkness doesn't compound.
    data.rt_lod = glm::vec4(15.0f, 50.0f,
                             static_cast<float>(rt_.gi_shadow_max_bounce),
                             rt_.ao_floor);
    {
        // cube.frag renders at render_extent_ (so does its motion-vec output);
        // gl_FragCoord.xy / viewport.zw is in render-resolution UV space.
        float w = static_cast<float>(render_extent_.width);
        float h = static_cast<float>(render_extent_.height);
        data.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
    }
    {
        // Muzzle flash — origin a short distance ahead of the eye so it
        // doesn't get trapped inside the player capsule. Intensity falls
        // linearly from 1 at fire to 0 over kMuzzleFlashDuration so the
        // pop is sharp and brief. Color is warm and quite hot — at intensity
        // 1.0 a single shot lights nearby walls visibly through RT shadows.
        if (muzzle_flash_timer_ > 0.0f) {
            float t = std::min(1.0f, muzzle_flash_timer_ / kMuzzleFlashDuration);
            glm::vec3 fwd = player_.forward();
            glm::vec3 pos = player_.eye_position() + fwd * 0.35f;
            float intensity = 14.0f * t;       // strong HDR pop
            data.muzzle_pos   = glm::vec4(pos, intensity);
            data.muzzle_color = glm::vec4(1.00f, 0.85f, 0.55f, 6.0f);  // 6 m radius
        } else {
            data.muzzle_pos   = glm::vec4(0.0f);
            data.muzzle_color = glm::vec4(0.0f);
        }
    }
    data.terrain_params = glm::vec4(rt_.terrain_fog_strength,
                                    rt_.terrain_wrap_strength,
                                    rt_.terrain_detail_strength,
                                    rt_.terrain_shadow_softness_scale);
    data.terrain_h_low  = glm::vec4(rt_.terrain_h_sand_grass_start,
                                    rt_.terrain_h_sand_grass_end,
                                    rt_.terrain_h_grass_dirt_start,
                                    rt_.terrain_h_grass_dirt_end);
    data.terrain_h_high = glm::vec4(rt_.terrain_h_dirt_rock_start,
                                    rt_.terrain_h_dirt_rock_end,
                                    rt_.terrain_h_rock_snow_start,
                                    rt_.terrain_h_rock_snow_end);
    data.grass_extra = glm::vec4(rt_.grass_height_scale,
                                 rt_.grass_alpha_cutoff,
                                 rt_.grass_slope_n_min,
                                 rt_.grass_distance_density);
    data.grass_extra2 = glm::vec4(rt_.grass_alt_min,
                                  rt_.grass_alt_max,
                                  rt_.shadow_map_world_half,
                                  static_cast<float>(rt_.terrain_debug_mode));
    data.light_vp = sun_shadow_light_vp_;
    data.terrain_extra = glm::vec4(rt_.terrain_shading_contrast,
                                    rt_.shadow_near_mult,
                                    0.0f, 0.0f);

    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, scene_ubo_alloc_, &ai);
    if (ai.pMappedData) std::memcpy(ai.pMappedData, &data, sizeof(data));
    // Defensive flush — no-op on coherent host memory but required by spec
    // for non-coherent allocations (e.g. some non-NVIDIA drivers).
    vmaFlushAllocation(allocator_, scene_ubo_alloc_, 0, sizeof(data));
}

} // namespace qlike
