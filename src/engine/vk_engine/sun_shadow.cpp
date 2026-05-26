// Sun shadow map pass — single-cascade ortho frustum, depth-only render
// of casters from the light's POV. Sampled by cube.frag (binding 7) and
// grass.vert for soft shadows; the heightmap bake handles distant
// terrain that falls outside the ortho box.
//
// Pipeline + resources + light-VP update + render-pass all collected
// here so the cross-file scatter (was: setup.cpp + world.cpp) becomes
// one TU. Methods stay as VulkanEngine members.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"
#include "engine/frustum.h"

namespace qlike {

void VulkanEngine::init_sun_shadow_pipeline() {
    std::string sd = QLIKE_SHADER_DIR;
    sun_shadow_vert_module_ = vkpipe::load_shader_module(device_, sd + "/shadow.vert.spv");

    // Dedicated pipeline layout — no descriptor sets. Shadow.vert only
    // reads `pc.mvp` (light_view_proj × model already baked on the host);
    // forcing the cube pipeline_layout_ here meant render_sun_shadow_pass
    // had to vkCmdBindDescriptorSets the entire scene set (TLAS + texture
    // arrays + materials SSBO) for code that consumes none of it.
    // 16-byte PC range — only the light-clip mvp matrix. The shadow
    // shader doesn't read model / prev_mvp / color / etc., so pushing
    // the full 256-byte cube PushConstants per draw was wasted host
    // bandwidth (hundreds of KB/frame across all caster draws).
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset = 0;
        pc.size = sizeof(glm::mat4);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 0;
        plci.pSetLayouts = nullptr;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &sun_shadow_pipeline_layout_),
                 "sun shadow pipeline layout");
    }

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = sun_shadow_vert_module_;
    cfg.frag = VK_NULL_HANDLE;          // depth-only
    cfg.layout = sun_shadow_pipeline_layout_;
    cfg.color_attachment_count = 0;
    cfg.depth_format = VK_FORMAT_D32_SFLOAT;
    // CULL_NONE on shadow casters: at a ground-contact line (cube on
    // terrain) the bottom face is back-facing the light and equals the
    // ground depth, so CULL_FRONT would let the ground test as lit at
    // the contact. Recording both faces with positive depth bias gives
    // a sun-facing caster depth at the contact line and the existing
    // bias still controls self-acne.
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS;
    cfg.depth_bias_enable = true;        // dynamic slope/constant bias

    VkVertexInputBindingDescription vb{};
    vb.binding = 0;
    vb.stride = sizeof(Vertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    cfg.vbindings.push_back(vb);
    VkVertexInputAttributeDescription a0{};
    a0.location = 0; a0.binding = 0;
    a0.format = VK_FORMAT_R32G32B32_SFLOAT;
    a0.offset = offsetof(Vertex, position);
    cfg.vattrs = { a0 };

    sun_shadow_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::info("sun shadow pipeline built");
}

void VulkanEngine::destroy_sun_shadow_pipeline() {
    if (sun_shadow_pipeline_) {
        vkDestroyPipeline(device_, sun_shadow_pipeline_, nullptr);
        sun_shadow_pipeline_ = VK_NULL_HANDLE;
    }
    if (sun_shadow_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, sun_shadow_pipeline_layout_, nullptr);
        sun_shadow_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (sun_shadow_vert_module_) {
        vkDestroyShaderModule(device_, sun_shadow_vert_module_, nullptr);
        sun_shadow_vert_module_ = VK_NULL_HANDLE;
    }
}

// Light frustum: ortho box centred on the player. Half-width covers the
// grass distance + slack. Depth along the sun axis is large so far-off
// mountain ridges still register as casters even when the receiver
// (grass) is right under the camera. Texel-snapped each frame to avoid
// crawling shadow edges as the player walks.
void VulkanEngine::init_sun_shadow_resources() {
    // Snap requested resolution to the allowed list. Default 1024 if a
    // settings file from before this slider existed lacks the field.
    int requested = rt_.shadow_map_resolution;
    int snapped = kShadowMapResolutions[1];  // 1024
    int best_diff = std::abs(requested - snapped);
    for (int v : kShadowMapResolutions) {
        int d = std::abs(requested - v);
        if (d < best_diff) { best_diff = d; snapped = v; }
    }
    sun_shadow_dim_ = snapped;
    rt_.shadow_map_resolution = snapped;  // write-back so the UI shows the snapped value
    const uint32_t W = static_cast<uint32_t>(sun_shadow_dim_);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_D32_SFLOAT;
    ici.extent = { W, W, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &sun_shadow_image_, &sun_shadow_alloc_, nullptr),
             "sun shadow image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = sun_shadow_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr, &sun_shadow_view_),
             "sun shadow view");

    // Comparison sampler — sampler2DShadow returns the comparison result
    // (0..1 with PCF) instead of raw depth. LINEAR + 2x2 hardware PCF gives
    // soft penumbra essentially for free.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;  // out-of-bounds = lit
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    vk_check(vkCreateSampler(device_, &si, nullptr, &sun_shadow_sampler_),
             "sun shadow sampler");

    // Initial transition into SHADER_READ_ONLY so render_sun_shadow_pass's
    // first transition (READ_ONLY → DEPTH_ATTACHMENT_OPTIMAL) is
    // symmetric with subsequent frames.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image_aspect(cb, sun_shadow_image_,
                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         VK_IMAGE_ASPECT_DEPTH_BIT);
    });

    // Descriptor binding 7 only needs writing once — the image handle
    // doesn't change for the lifetime of the engine.
    VkDescriptorImageInfo dii{};
    dii.sampler = sun_shadow_sampler_;
    dii.imageView = sun_shadow_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 7;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void VulkanEngine::destroy_sun_shadow_resources() {
    if (sun_shadow_view_) {
        vkDestroyImageView(device_, sun_shadow_view_, nullptr);
        sun_shadow_view_ = VK_NULL_HANDLE;
    }
    if (sun_shadow_image_) {
        vmaDestroyImage(allocator_, sun_shadow_image_, sun_shadow_alloc_);
        sun_shadow_image_ = VK_NULL_HANDLE;
        sun_shadow_alloc_ = nullptr;
    }
    if (sun_shadow_sampler_) {
        vkDestroySampler(device_, sun_shadow_sampler_, nullptr);
        sun_shadow_sampler_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::update_sun_shadow_light_vp() {
    // Sun direction from the user settings (matches cube.frag and the
    // heightmap bake). Light position floats along the sun axis above the
    // player so the ortho frustum's depth range covers nearby grass and
    // distant mountain ridges that need to cast on it.
    float p_rad = glm::radians(rt_.sun_pitch_deg);
    float y_rad = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun_dir = glm::normalize(glm::vec3(
        std::sin(y_rad) * std::cos(p_rad),
        std::sin(p_rad),
        std::cos(y_rad) * std::cos(p_rad)));

    // Half-extent of the ortho box, slider-controlled.
    const float half = std::max(rt_.shadow_map_world_half, 30.0f);
    // Sun-axis depth: large so the shadow camera can sit far back along
    // the sun direction and still capture distant cliff peaks.
    const float depth_back  = 600.0f;
    const float depth_front = 400.0f;

    glm::vec3 cam = player_.eye_position();

    // Texel-grid snap: round the light-space view-plane origin to the
    // nearest shadow texel so distant edges don't crawl as the player
    // walks. Pick world axes spanning the light-view plane (any two axes
    // perpendicular to sun_dir).
    glm::vec3 up_ref = std::abs(sun_dir.y) > 0.99f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right_axis = glm::normalize(glm::cross(up_ref, sun_dir));
    glm::vec3 up_axis    = glm::normalize(glm::cross(sun_dir, right_axis));
    float texel = (2.0f * half) / static_cast<float>(sun_shadow_dim_);
    float u = glm::dot(cam, right_axis);
    float v = glm::dot(cam, up_axis);
    u = std::round(u / texel) * texel;
    v = std::round(v / texel) * texel;
    // Reconstruct snapped centre (component along sun_dir doesn't matter
    // for an ortho frustum, but keep player-y so the box is positioned
    // sensibly).
    glm::vec3 snapped_centre = right_axis * u + up_axis * v +
                                sun_dir * glm::dot(cam, sun_dir);

    glm::vec3 light_eye = snapped_centre + sun_dir * depth_back;
    glm::mat4 view = glm::lookAt(light_eye, snapped_centre, up_axis);
    glm::mat4 proj = glm::ortho(-half, half, -half, half,
                                 0.0f, depth_back + depth_front);
    proj[1][1] *= -1.0f;  // match Vulkan-y-flipped convention used elsewhere
    sun_shadow_light_vp_ = proj * view;
}

void VulkanEngine::render_sun_shadow_pass(VkCommandBuffer cmd) {
    // Transition into DEPTH_ATTACHMENT_OPTIMAL for the depth-only render.
    vkinit::transition_image_aspect(cmd, sun_shadow_image_,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);

    VkClearValue clear_depth{};
    clear_depth.depthStencil.depth = 1.0f;

    VkRenderingAttachmentInfo depth_att{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = sun_shadow_view_,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear_depth,
    };
    VkExtent2D ext{ static_cast<uint32_t>(sun_shadow_dim_),
                    static_cast<uint32_t>(sun_shadow_dim_) };
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, ext },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 0, .pColorAttachments = nullptr,
        .pDepthAttachment = &depth_att, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width = static_cast<float>(sun_shadow_dim_);
    vp_state.height = static_cast<float>(sun_shadow_dim_);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, ext };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sun_shadow_pipeline_);
    // No descriptor-set bind — shadow.vert consumes only push constants.

    // CULL_NONE makes the front (sun-facing) face write the correct
    // caster depth, so there's no need for positive caster bias to
    // separate caster from receiver. Slope-scaled bias was the
    // direct cause of the contact-line gap: on near-vertical caster
    // faces (cube sides, steep terrain) max(|dz/dx|, |dz/dy|) blows
    // up and the slope factor multiplies that into a multi-metre
    // push, detaching the shadow from the base. Keep only a tiny
    // constant offset to absorb depth-encoding noise.
    vkCmdSetDepthBias(cmd, /*const*/ 0.5f, /*clamp*/ 0.0f, /*slope*/ 0.0f);

    glm::mat4 vp = sun_shadow_light_vp_;
    Frustum light_frustum = extract_frustum(vp);

    auto push_shadow = [&](const glm::mat4& model) {
        glm::mat4 mvp = vp * model;
        vkCmdPushConstants(cmd, sun_shadow_pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(glm::mat4), &mvp);
    };

    // Static brushes (castle, towers). Light-frustum cull: only brushes
    // whose AABB intersects the ortho box need rasterising. Cuts ~80%
    // of brush draws when the player isn't inside the keep — was the
    // dominant cost behind the 14-second TDR repro.
    VkDeviceSize cube_off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &cube_off);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& a = world_.aabbs[i];
        if (!aabb_visible(light_frustum, a.min, a.max)) continue;
        push_shadow(static_brush_models_[i]);   // pre-baked, see init_world
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }

    // Dyn-props — same cube BLAS is the cube mesh; cylinders use their
    // own mesh. Light-frustum cull mirrors the brushes path.
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(light_frustum, dr.aabb_min, dr.aabb_max)) continue;
        glm::mat4 model = dr.world * glm::scale(glm::mat4(1.0f),
                                                 dyn_props_[i].full_size);
        push_shadow(model);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }

    // Terrain chunks — frustum-culled, distance-LOD'd. Distance metric is
    // camera-to-chunk (not light-to-chunk) so terrain casters share the
    // same LOD selection as the main render — keeps the cost bounded
    // and matches the resolution the user actually sees.
    if (!terrain_chunks_.chunks.empty()) {
        glm::vec3 cam = player_.eye_position();
        // Shared LOD thresholds with the depth + colour passes -- a
        // mismatch here would silently break the depth pre-pass's
        // LESS_OR_EQUAL test. Build the scaled threshold array once
        // and reuse for every chunk.
        float thresh[kTerrainLodCount - 1];
        for (int i = 0; i < kTerrainLodCount - 1; ++i)
            thresh[i] = rt_.terrain_lod_distance[i] * rt_.terrain_lod_scale;
        for (const auto& c : terrain_chunks_.chunks) {
            if (!aabb_visible(light_frustum, c.aabb_min, c.aabb_max)) continue;
            int lod = pick_terrain_lod(c, cam, thresh, kTerrainLodCount - 1);
            VkDeviceSize toff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &c.mesh.vertex_buffer, &toff);
            VkBuffer ibo = (lod == 0) ? c.mesh.index_buffer : c.ibo_lod[lod - 1];
            vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
            push_shadow(glm::mat4(1.0f));
            vkCmdDrawIndexed(cmd, c.index_count_lod[lod], 1, 0, 0, 0);
        }
    }

    vkCmdEndRendering(cmd);

    // Transition back to SHADER_READ_ONLY so grass.vert can sample it.
    vkinit::transition_image_aspect(cmd, sun_shadow_image_,
                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);
}

} // namespace qlike
