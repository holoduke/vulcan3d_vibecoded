// Physically based sky + aerial perspective (Hillaire 2020, "A Scalable
// and Production Ready Sky and Atmosphere Rendering Technique").
//
//   transmittance LUT  256x64   - once (atmosphere parameters are static)
//   multi-scatter LUT  32x32    - once
//   sky-view LUT       192x108  - every frame (sun / camera height)
//   aerial perspective 32^3     - every frame (camera frustum froxels)
//
// One compute pipeline (shaders/atmosphere.comp, pass in a push constant);
// every image stays in VK_IMAGE_LAYOUT_GENERAL so the same images are
// storage targets here and sampled textures in compose.frag.

#include "engine/vk_engine/internal.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace qlike {

namespace {
constexpr uint32_t kTransW = 256, kTransH = 64;
constexpr uint32_t kMsW = 32, kMsH = 32;
constexpr uint32_t kSkyW = 192, kSkyH = 108;
constexpr uint32_t kApN = 32;
constexpr float kBottomRadiusKm = 6360.0f;

struct AtmoPC {
    glm::vec4 sun;     // xyz dir toward sun, w = pass
    glm::vec4 cam;     // xyz planet-space camera (km), w = aerial scale
    glm::vec4 eye;     // xyz world camera (m), w = AP max distance (km)
    glm::mat4 inv_vp;
};
}  // namespace

void VulkanEngine::init_atmosphere() {
    auto make = [&](AtmoImage& im, VkImageType type, uint32_t w, uint32_t h, uint32_t d, const char* name) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = type;
        ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = { w, h, d };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &im.image, &im.alloc, nullptr), name);
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = im.image;
        vci.viewType = (type == VK_IMAGE_TYPE_3D) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im.view), name);
    };
    make(atmo_trans_, VK_IMAGE_TYPE_2D, kTransW, kTransH, 1, "atmo transmittance");
    make(atmo_ms_,    VK_IMAGE_TYPE_2D, kMsW, kMsH, 1, "atmo multiscatter");
    make(atmo_sky_,   VK_IMAGE_TYPE_2D, kSkyW, kSkyH, 1, "atmo skyview");
    make(atmo_ap_,    VK_IMAGE_TYPE_3D, kApN, kApN, kApN, "atmo aerial");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (AtmoImage* im : { &atmo_trans_, &atmo_ms_, &atmo_sky_, &atmo_ap_ })
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // Sky-view LUT is read by the raymarched terrain BEFORE this frame's
        // atmosphere dispatch: start it at zero, not undefined memory.
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
        VkClearColorValue z{};
        VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, atmo_sky_.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
        vkCmdPipelineBarrier2(cb, &di);
    });
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &atmo_sampler_), "atmo sampler");
    }
    {
        VkDescriptorSetLayoutBinding b[6]{};
        for (int i = 0; i < 6; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i < 2) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 6; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &atmo_set_layout_), "atmo set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &atmo_pool_), "atmo pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = atmo_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &atmo_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &atmo_set_), "atmo set");

        VkDescriptorImageInfo ii[6] = {
            { atmo_sampler_, atmo_trans_.view, VK_IMAGE_LAYOUT_GENERAL },
            { atmo_sampler_, atmo_ms_.view,    VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_trans_.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_ms_.view,    VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_sky_.view,   VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_ap_.view,    VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet w[6]{};
        for (int i = 0; i < 6; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = atmo_set_;
            w[i].dstBinding = uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = b[i].descriptorType;
            w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(device_, 6, w, 0, nullptr);
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmoPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &atmo_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &atmo_pipeline_layout_), "atmo layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/atmosphere.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = atmo_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &atmo_pipeline_),
                 "atmo pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    // Scene set binding 30: the sky-view LUT for raymarched terrain / water.
    if (scene_desc_set_) {
        VkDescriptorImageInfo si{ atmo_sampler_, atmo_sky_.view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 30; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &si;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    atmo_luts_ready_ = false;
    log::info("[atmosphere] Hillaire 2020 sky: LUTs + aerial perspective ready");
    init_clouds();
}

// ---------------------------------------------------------------------
// Volumetric clouds (shaders/clouds.comp)
// ---------------------------------------------------------------------
namespace {
constexpr uint32_t kCloudW = 960, kCloudH = 540;
constexpr uint32_t kShapeN = 128, kDetailN = 32;
struct CloudPC {
    glm::mat4 inv_vp;
    glm::vec4 eye;      // xyz camera (m), w time (s)
    glm::vec4 sun;      // xyz toward sun, w mode
    glm::vec4 params;   // coverage, density, wind m/s, sun intensity (engine * pi)
    glm::vec4 params2;  // frame, camera planet radius (km)
};
}  // namespace

void VulkanEngine::init_clouds() {
    auto make = [&](AtmoImage& im, VkImageType type, VkFormat fmt, uint32_t w, uint32_t h, uint32_t d,
                    const char* name) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = type;
        ici.format = fmt;
        ici.extent = { w, h, d };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &im.image, &im.alloc, nullptr), name);
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = im.image;
        vci.viewType = (type == VK_IMAGE_TYPE_3D) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im.view), name);
    };
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &cloud_sampler_), "cloud sampler");
    }
    make(cloud_img_,    VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, kCloudW, kCloudH, 1, "cloud layer");
    make(cloud_shape_,  VK_IMAGE_TYPE_3D, VK_FORMAT_R16_SFLOAT, kShapeN, kShapeN, kShapeN, "cloud shape noise");
    make(cloud_detail_, VK_IMAGE_TYPE_3D, VK_FORMAT_R16_SFLOAT, kDetailN, kDetailN, kDetailN, "cloud detail noise");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (AtmoImage* im : { &cloud_img_, &cloud_shape_, &cloud_detail_ })
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkClearColorValue z{};
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
        vkCmdClearColorImage(cb, cloud_img_.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
        vkCmdPipelineBarrier2(cb, &di);
    });
    {
        VkDescriptorSetLayoutBinding b[8]{};
        for (int i = 0; i < 8; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i < 5) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 8; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &cloud_set_layout_), "cloud set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          3 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &cloud_pool_), "cloud pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = cloud_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &cloud_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &cloud_set_), "cloud set");
        // Binding 2 (scene depth) is written every dispatch (view changes on resize).
        VkDescriptorImageInfo ii[8] = {
            { atmo_sampler_, atmo_trans_.view,   VK_IMAGE_LAYOUT_GENERAL },
            { atmo_sampler_, atmo_sky_.view,     VK_IMAGE_LAYOUT_GENERAL },
            { atmo_sampler_, VK_NULL_HANDLE,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { cloud_sampler_, cloud_shape_.view, VK_IMAGE_LAYOUT_GENERAL },
            { cloud_sampler_, cloud_detail_.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_img_.view,    VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_shape_.view,  VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_detail_.view, VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet w[7]{};
        int n = 0;
        for (int i = 0; i < 8; ++i) {
            if (i == 2) continue;
            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = cloud_set_;
            w[n].dstBinding = uint32_t(i);
            w[n].descriptorCount = 1;
            w[n].descriptorType = b[i].descriptorType;
            w[n].pImageInfo = &ii[i];
            ++n;
        }
        vkUpdateDescriptorSets(device_, uint32_t(n), w, 0, nullptr);
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CloudPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &cloud_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &cloud_pipeline_layout_), "cloud layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/clouds.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = cloud_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &cloud_pipeline_),
                 "cloud pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    cloud_noise_ready_ = false;
    log::info("[clouds] volumetric cloud layer ready (960x540, 128^3 + 32^3 noise)");
    init_fog();
}

void VulkanEngine::destroy_clouds() {
    destroy_fog();
    if (cloud_sampler_) vkDestroySampler(device_, cloud_sampler_, nullptr);
    cloud_sampler_ = VK_NULL_HANDLE;
    if (cloud_pipeline_) vkDestroyPipeline(device_, cloud_pipeline_, nullptr);
    if (cloud_pipeline_layout_) vkDestroyPipelineLayout(device_, cloud_pipeline_layout_, nullptr);
    if (cloud_pool_) vkDestroyDescriptorPool(device_, cloud_pool_, nullptr);
    if (cloud_set_layout_) vkDestroyDescriptorSetLayout(device_, cloud_set_layout_, nullptr);
    for (AtmoImage* im : { &cloud_img_, &cloud_shape_, &cloud_detail_ }) {
        if (im->view) vkDestroyImageView(device_, im->view, nullptr);
        if (im->image) vmaDestroyImage(allocator_, im->image, im->alloc);
        *im = {};
    }
    cloud_pipeline_ = VK_NULL_HANDLE; cloud_pipeline_layout_ = VK_NULL_HANDLE;
    cloud_pool_ = VK_NULL_HANDLE; cloud_set_layout_ = VK_NULL_HANDLE; cloud_set_ = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_clouds(VkCommandBuffer cmd) {
    if (!cloud_pipeline_ || !depth_view_) return;
    {
        VkDescriptorImageInfo di{ atmo_sampler_, depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = cloud_set_; w.dstBinding = 2; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &di;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    const FrameView& fv = current_frame_view_;
    float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
    CloudPC pc{};
    pc.inv_vp = fv.inv_vp;
    pc.eye = glm::vec4(fv.eye_pos, static_cast<float>(glfwGetTime()));
    pc.params = glm::vec4(glm::clamp(rt_.cloud_coverage, 0.0f, 1.0f), std::max(0.0f, rt_.cloud_density),
                          rt_.cloud_speed, rt_.atmosphere_intensity * 3.14159265f);
    pc.params2 = glm::vec4(float(frame_number_ % 64), kBottomRadiusKm + std::max(fv.eye_pos.y, 1.0f) * 0.001f,
                           0.0f, 0.0f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloud_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloud_pipeline_layout_, 0, 1, &cloud_set_, 0, nullptr);
    auto barrier = [&]() {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
        VkDependencyInfo dinfo{}; dinfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dinfo.memoryBarrierCount = 1; dinfo.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dinfo);
    };
    auto run = [&](float mode, uint32_t gx, uint32_t gy, uint32_t gz) {
        pc.sun = glm::vec4(sun, mode);
        vkCmdPushConstants(cmd, cloud_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, gx, gy, gz);
        barrier();
    };
    if (!cloud_noise_ready_) {
        run(0.0f, kShapeN / 8, kShapeN / 8, kShapeN);
        run(1.0f, kDetailN / 8, kDetailN / 8, kDetailN);
        cloud_noise_ready_ = true;
    }
    run(2.0f, (kCloudW + 7) / 8, (kCloudH + 7) / 8, 1);
}

void VulkanEngine::destroy_atmosphere() {
    destroy_clouds();
    if (atmo_pipeline_) vkDestroyPipeline(device_, atmo_pipeline_, nullptr);
    if (atmo_pipeline_layout_) vkDestroyPipelineLayout(device_, atmo_pipeline_layout_, nullptr);
    if (atmo_pool_) vkDestroyDescriptorPool(device_, atmo_pool_, nullptr);
    if (atmo_set_layout_) vkDestroyDescriptorSetLayout(device_, atmo_set_layout_, nullptr);
    if (atmo_sampler_) vkDestroySampler(device_, atmo_sampler_, nullptr);
    for (AtmoImage* im : { &atmo_trans_, &atmo_ms_, &atmo_sky_, &atmo_ap_ }) {
        if (im->view) vkDestroyImageView(device_, im->view, nullptr);
        if (im->image) vmaDestroyImage(allocator_, im->image, im->alloc);
        *im = {};
    }
    atmo_pipeline_ = VK_NULL_HANDLE; atmo_pipeline_layout_ = VK_NULL_HANDLE;
    atmo_pool_ = VK_NULL_HANDLE; atmo_set_layout_ = VK_NULL_HANDLE;
    atmo_sampler_ = VK_NULL_HANDLE; atmo_set_ = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_atmosphere(VkCommandBuffer cmd) {
    if (!atmo_pipeline_) return;
    const FrameView& fv = current_frame_view_;
    float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
    AtmoPC pc{};
    // World Y is metres above sea level; planet frame == world axes.
    const float h_km = std::max(fv.eye_pos.y, 1.0f) * 0.001f;
    pc.cam = glm::vec4(0.0f, kBottomRadiusKm + h_km, 0.0f, std::max(0.1f, rt_.aerial_scale));
    pc.eye = glm::vec4(fv.eye_pos, std::max(0.5f, rt_.aerial_max_km));
    pc.inv_vp = fv.inv_vp;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atmo_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atmo_pipeline_layout_, 0, 1, &atmo_set_, 0, nullptr);
    auto barrier = [&]() {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    };
    auto run = [&](float pass, uint32_t gx, uint32_t gy) {
        pc.sun = glm::vec4(sun, pass);
        vkCmdPushConstants(cmd, atmo_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, gx, gy, 1);
        barrier();
    };
    if (!atmo_luts_ready_) {
        run(0.0f, (kTransW + 7) / 8, (kTransH + 7) / 8);
        run(1.0f, (kMsW + 7) / 8, (kMsH + 7) / 8);
        atmo_luts_ready_ = true;
    }
    run(2.0f, (kSkyW + 7) / 8, (kSkyH + 7) / 8);
    run(3.0f, (kApN + 7) / 8, (kApN + 7) / 8);
    if (rt_.clouds_enabled) dispatch_clouds(cmd);
}


// ---------------------------------------------------------------------
// Froxel volumetric fog (shaders/froxel_fog.comp)
// ---------------------------------------------------------------------
namespace {
constexpr uint32_t kFogX = 160, kFogY = 90, kFogZ = 64;
struct FogPC {
    glm::mat4 inv_vp;
    glm::vec4 eye;      // xyz camera (m), w time
    glm::vec4 sun;      // xyz toward sun, w pass
    glm::vec4 sun_col;  // rgb (engine E * pi), w ground reference height (m)
    glm::vec4 amb;      // rgb ambient, w frame
    glm::vec4 params;   // density, height falloff, anisotropy, max distance
};
}  // namespace

void VulkanEngine::init_fog() {
    for (AtmoImage* im : { &fog_inject_, &fog_integ_ }) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_3D;
        ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = { kFogX, kFogY, kFogZ };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &im->image, &im->alloc, nullptr), "fog volume");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = im->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im->view), "fog volume view");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        for (AtmoImage* im : { &fog_inject_, &fog_integ_ })
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdPipelineBarrier2(cb, &di);
        // Integrated volume = (L 0, T 1) until the first dispatch: no fog.
        VkClearColorValue clr{ { 0.0f, 0.0f, 0.0f, 1.0f } };
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, fog_integ_.image, VK_IMAGE_LAYOUT_GENERAL, &clr, 1, &r);
        vkCmdPipelineBarrier2(cb, &di);
    });
    const VkDescriptorType types[5] = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR };
    {
        VkDescriptorSetLayoutBinding b[5]{};
        for (int i = 0; i < 5; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = types[i];
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 5; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &fog_set_layout_), "fog set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 3; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &fog_pool_), "fog pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = fog_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &fog_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &fog_set_), "fog set");
        VkDescriptorImageInfo ii[4] = {
            { VK_NULL_HANDLE, fog_inject_.view, VK_IMAGE_LAYOUT_GENERAL },
            { atmo_sampler_, fog_inject_.view,  VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, fog_integ_.view,  VK_IMAGE_LAYOUT_GENERAL },
            { cloud_sampler_, cloud_detail_.view, VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet w[4]{};
        for (int i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = fog_set_;
            w[i].dstBinding = uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = types[i];
            w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FogPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &fog_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &fog_pipeline_layout_), "fog layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/froxel_fog.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = fog_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &fog_pipeline_),
                 "fog pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    fog_tlas_bound_ = VK_NULL_HANDLE;
    log::info("[fog] froxel volumetric fog ready (160x90x64, RT sun visibility)");
}

void VulkanEngine::destroy_fog() {
    if (fog_pipeline_) vkDestroyPipeline(device_, fog_pipeline_, nullptr);
    if (fog_pipeline_layout_) vkDestroyPipelineLayout(device_, fog_pipeline_layout_, nullptr);
    if (fog_pool_) vkDestroyDescriptorPool(device_, fog_pool_, nullptr);
    if (fog_set_layout_) vkDestroyDescriptorSetLayout(device_, fog_set_layout_, nullptr);
    for (AtmoImage* im : { &fog_inject_, &fog_integ_ }) {
        if (im->view) vkDestroyImageView(device_, im->view, nullptr);
        if (im->image) vmaDestroyImage(allocator_, im->image, im->alloc);
        *im = {};
    }
    fog_pipeline_ = VK_NULL_HANDLE; fog_pipeline_layout_ = VK_NULL_HANDLE;
    fog_pool_ = VK_NULL_HANDLE; fog_set_layout_ = VK_NULL_HANDLE; fog_set_ = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_fog(VkCommandBuffer cmd) {
    if (!fog_pipeline_ || tlas_ == VK_NULL_HANDLE) return;
    if (fog_tlas_bound_ != tlas_) {   // the TLAS handle changes when it is reallocated
        VkWriteDescriptorSetAccelerationStructureKHR as{};
        as.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as.accelerationStructureCount = 1;
        as.pAccelerationStructures = &tlas_;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.pNext = &as;
        w.dstSet = fog_set_; w.dstBinding = 4; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        fog_tlas_bound_ = tlas_;
    }
    const FrameView& fv = current_frame_view_;
    float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
    FogPC pc{};
    pc.inv_vp = fv.inv_vp;
    pc.eye = glm::vec4(fv.eye_pos, static_cast<float>(glfwGetTime()));
    pc.sun_col = glm::vec4(last_sun_rgb_ * rt_.sun_intensity * 3.14159265f, rt_.water_level + 1.0f);
    pc.amb = glm::vec4(last_sky_rgb_ * rt_.ambient_strength * 0.8f, float(frame_number_ % 64));
    pc.params = glm::vec4(std::max(0.0f, rt_.fog_density), std::max(0.0f, rt_.fog_height_falloff),
                          glm::clamp(rt_.fog_anisotropy, -0.9f, 0.95f), std::max(20.0f, rt_.fog_max_dist));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_pipeline_layout_, 0, 1, &fog_set_, 0, nullptr);
    auto barrier = [&]() {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    };
    pc.sun = glm::vec4(sun, 0.0f);
    vkCmdPushConstants(cmd, fog_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (kFogX + 7) / 8, (kFogY + 7) / 8, kFogZ);
    barrier();
    pc.sun.w = 1.0f;
    vkCmdPushConstants(cmd, fog_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (kFogX + 7) / 8, (kFogY + 7) / 8, 1);
    barrier();
}

} // namespace qlike
