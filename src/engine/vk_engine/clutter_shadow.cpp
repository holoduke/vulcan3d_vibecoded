// Small-object sun shadows: flowers, ferns and scatter rocks rendered
// (depth only) into a dedicated 2048^2 map covering 64 m around the
// camera. Too many instances for the ray-tracing TLAS, and the main sun
// shadow map also contains the terrain (a terrain receiver would need a
// large depth bias that erases 30 cm flower shadows). Sampled by
// deferred_lighting.frag (binding 14) for terrain receivers.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"

#include <algorithm>
#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#ifndef QLIKE_SHADER_DIR
#define QLIKE_SHADER_DIR "shaders"
#endif

namespace qlike {

namespace {
constexpr uint32_t kClutterDim  = 2048;
constexpr float    kClutterHalf = 32.0f;   // metres
constexpr float    kClutterDepth = 60.0f;  // +- along the sun axis
}

void VulkanEngine::init_clutter_shadow() {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_D32_SFLOAT;
    ici.extent = { kClutterDim, kClutterDim, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &clutter_shadow_image_, &clutter_shadow_alloc_, nullptr),
             "clutter shadow image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = clutter_shadow_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_shadow_view_), "clutter shadow view");
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &clutter_shadow_sampler_), "clutter shadow sampler");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image_aspect(cb, clutter_shadow_image_, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_ASPECT_DEPTH_BIT);
    });

    VkPushConstantRange pc{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) + 2 * sizeof(glm::vec4) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &clutter_shadow_layout_), "clutter layout");
    clutter_shadow_vert_ = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/clutter_shadow.vert.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = clutter_shadow_vert_;
    cfg.frag = VK_NULL_HANDLE;
    cfg.layout = clutter_shadow_layout_;
    cfg.color_attachment_count = 0;
    cfg.depth_format = VK_FORMAT_D32_SFLOAT;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS;
    VkVertexInputBindingDescription b0{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputBindingDescription b1{ 1, sizeof(ScatterInstance), VK_VERTEX_INPUT_RATE_INSTANCE };
    cfg.vbindings = { b0, b1 };
    VkVertexInputAttributeDescription a0{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, position)) };
    VkVertexInputAttributeDescription a2{ 2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(Vertex, uv)) };
    VkVertexInputAttributeDescription a3{ 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    VkVertexInputAttributeDescription a4{ 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
    cfg.vattrs = { a0, a2, a3, a4 };
    clutter_shadow_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    {
        VkDescriptorImageInfo ii{ clutter_shadow_sampler_, clutter_shadow_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 28; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    clutter_shadow_vp_ = glm::mat4(0.0f);
    log::info("[clutter shadow] 2048^2 map, 64 m around the camera");
}

void VulkanEngine::destroy_clutter_shadow() {
    if (clutter_shadow_pipeline_) vkDestroyPipeline(device_, clutter_shadow_pipeline_, nullptr);
    if (clutter_shadow_layout_) vkDestroyPipelineLayout(device_, clutter_shadow_layout_, nullptr);
    if (clutter_shadow_vert_) vkDestroyShaderModule(device_, clutter_shadow_vert_, nullptr);
    if (clutter_shadow_view_) vkDestroyImageView(device_, clutter_shadow_view_, nullptr);
    if (clutter_shadow_sampler_) vkDestroySampler(device_, clutter_shadow_sampler_, nullptr);
    clutter_shadow_sampler_ = VK_NULL_HANDLE;
    if (clutter_shadow_image_) vmaDestroyImage(allocator_, clutter_shadow_image_, clutter_shadow_alloc_);
    clutter_shadow_pipeline_ = VK_NULL_HANDLE; clutter_shadow_layout_ = VK_NULL_HANDLE;
    clutter_shadow_vert_ = VK_NULL_HANDLE; clutter_shadow_view_ = VK_NULL_HANDLE;
    clutter_shadow_image_ = VK_NULL_HANDLE;
}

// Computed in update_scene_ubo() (before any pass) so the deferred pass
// AND the raymarched terrain (via the scene UBO) use the same matrix the
// map is rendered with this frame.
static const bool kNoClutter = std::getenv("QLIKE_NO_CLUTTER_SHADOW") != nullptr;   // A/B test hook
void VulkanEngine::update_clutter_shadow_vp() {
    if (!clutter_shadow_pipeline_ || scatter_.instance_count == 0 || kNoClutter) {
        clutter_shadow_vp_ = glm::mat4(0.0f);
        return;
    }
    // Light VP: ortho box around the camera along the sun, texel-snapped.
    float p_rad = glm::radians(rt_.sun_pitch_deg), y_rad = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun = glm::normalize(glm::vec3(std::sin(y_rad) * std::cos(p_rad), std::sin(p_rad),
                                             std::cos(y_rad) * std::cos(p_rad)));
    glm::vec3 up_ref = std::abs(sun.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    glm::vec3 rx = glm::normalize(glm::cross(up_ref, sun)), ry = glm::normalize(glm::cross(sun, rx));
    glm::vec3 cam = current_frame_view_.eye_pos;
    const float texel = 2.0f * kClutterHalf / float(kClutterDim);
    float u = std::round(glm::dot(cam, rx) / texel) * texel;
    float v = std::round(glm::dot(cam, ry) / texel) * texel;
    glm::vec3 c = rx * u + ry * v + sun * glm::dot(cam, sun);
    glm::mat4 view = glm::lookAt(c + sun * kClutterDepth, c, ry);
    glm::mat4 proj = glm::ortho(-kClutterHalf, kClutterHalf, -kClutterHalf, kClutterHalf, 0.0f, 2.0f * kClutterDepth);
    proj[1][1] *= -1.0f;
    clutter_shadow_vp_ = proj * view;
}

void VulkanEngine::render_clutter_shadow(VkCommandBuffer cmd) {
    if (clutter_shadow_vp_[3][3] < 0.5f) return;
    const glm::vec3 cam = current_frame_view_.eye_pos;

    vkinit::transition_image_aspect(cmd, clutter_shadow_image_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkRenderingAttachmentInfo da{};
    da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    da.imageView = clutter_shadow_view_;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    da.clearValue.depthStencil = { 1.0f, 0 };
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { { 0, 0 }, { kClutterDim, kClutterDim } };
    ri.layerCount = 1;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{ 0.0f, 0.0f, float(kClutterDim), float(kClutterDim), 0.0f, 1.0f };
    VkRect2D sc{ { 0, 0 }, { kClutterDim, kClutterDim } };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, clutter_shadow_pipeline_);
    // p: .x wind time, .y plant (sway) flag, .z colour-pass draw distance -
    // the same fade / sway as scatter.vert so shadows match what is drawn.
    struct { glm::mat4 vp; glm::vec4 cam; glm::vec4 p; } pc{ clutter_shadow_vp_, glm::vec4(cam, kClutterHalf * 1.42f),
        glm::vec4(static_cast<float>(frame_number_) * 0.016f, 0.0f,
                  std::max(60.0f, rt_.grass_distance * 1.6f), 0.0f) };
    vkCmdPushConstants(cmd, clutter_shadow_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    VkDeviceSize offs[2] = { 0, 0 };
    // Only cells overlapping the shadow box around the camera (XZ).
    const float box_r = kClutterHalf * 1.42f;
    auto in_box = [&](const ScatterCell& c) {
        float qx = std::clamp(cam.x, c.mn.x, c.mx.x), qz = std::clamp(cam.z, c.mn.z, c.mx.z);
        return (qx - cam.x) * (qx - cam.x) + (qz - cam.z) * (qz - cam.z) < box_r * box_r;
    };
    {
        VkBuffer b[2] = { scatter_.rock_mesh.vertex_buffer, scatter_.instance_buffer };
        vkCmdBindVertexBuffers(cmd, 0, 2, b, offs);
        vkCmdBindIndexBuffer(cmd, scatter_.rock_mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        scatter_draw_cells(cmd, scatter_.rock_mesh.index_count, scatter_.rock_cells, in_box);
    }
    if (scatter_.plant_buffer) {
        pc.p.y = 1.0f;
        pc.p.z = std::max(45.0f, rt_.grass_distance * 0.9f);
        vkCmdPushConstants(cmd, clutter_shadow_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
            if (!scatter_.plant_count[sp]) continue;
            VkBuffer b[2] = { scatter_.plant_mesh[sp].vertex_buffer, scatter_.plant_buffer };
            vkCmdBindVertexBuffers(cmd, 0, 2, b, offs);
            vkCmdBindIndexBuffer(cmd, scatter_.plant_mesh[sp].index_buffer, 0, VK_INDEX_TYPE_UINT32);
            scatter_draw_cells(cmd, scatter_.plant_mesh[sp].index_count, scatter_.plant_cells[sp], in_box);
        }
    }
    vkCmdEndRendering(cmd);
    vkinit::transition_image_aspect(cmd, clutter_shadow_image_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
}

} // namespace qlike
