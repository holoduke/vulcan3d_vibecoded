// Phase 3 of the deferred-render migration: a fullscreen lighting pass
// that reads the G-buffer + depth + the existing scene UBO/TLAS and
// writes lit colour to staging_color_image_. compose.frag picks
// scene_color vs staging_color at composite-time via the
// deferred_lighting_active_ runtime toggle — default OFF, so the
// existing forward path is byte-identical until Phase 4 promotes the
// deferred output to the canonical scene_color writer.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

#include <string>

namespace qlike {

// Mirrors the layout of the shader's PC block (std430 packed under push
// constant rules). Total 192 B — under the 256 B threshold all desktop
// GPUs expose.
struct DeferredLightingPC {
    glm::mat4 inv_vp;
    glm::vec4 light_pos[4];   // .xyz pos, .w radius (m); .w == 0 disables
    glm::vec4 light_col[4];   // .rgb colour, .a intensity multiplier
};
static_assert(sizeof(DeferredLightingPC) == 192,
              "DeferredLightingPC layout mismatch — check std140 expectations");


void VulkanEngine::init_deferred_lighting() {
    // Descriptor set layout: scene UBO + TLAS + gbuffer0 + gbuffer1 +
    // depth + sun_shadow. Six bindings, all fragment-stage. Reuses the
    // same binding numbers the shader declares.
    {
        // Bindings 0..8. Binding 8 is u_svgf_gi — cube.frag's raw GI
        // output (pre-denoise). Sampling it lets the deferred shader
        // reuse cube.frag's material-aware multi-bounce GI instead of
        // its own 4-ray approximation, closing most of the remaining
        // SAD gap on indoor / shadow-side surfaces. Image is in GENERAL
        // layout when cube.frag writes via imageStore — sampler2D reads
        // are allowed from GENERAL.
        VkDescriptorSetLayoutBinding b[9]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        for (int i = 2; i < 9; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .bindingCount = 9, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &dlci, nullptr,
                                              &deferred_lighting_desc_layout_),
                 "deferred lighting desc layout");
    }

    // Pipeline layout. Push constant = inverse-VP (64 B) + 4 light_pos
    // vec4 (64 B) + 4 light_col vec4 (64 B) = 192 B total. Vulkan
    // guarantees 128 B minimum; modern desktop GPUs all expose ≥ 256 B,
    // so 192 is safe. (No fallback path — if the device can't take it
    // the pipeline create will report it.)
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = sizeof(DeferredLightingPC);
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &deferred_lighting_desc_layout_,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &deferred_lighting_pipeline_layout_),
                 "deferred lighting pipeline layout");
    }

    // Descriptor pool + set.
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     7 },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 3, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr,
                                         &deferred_lighting_desc_pool_),
                 "deferred lighting desc pool");
        VkDescriptorSetAllocateInfo dsai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = deferred_lighting_desc_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &deferred_lighting_desc_layout_,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dsai,
                                           &deferred_lighting_desc_set_),
                 "deferred lighting alloc desc set");
    }

    // Pipeline.
    {
        std::string sd = QLIKE_SHADER_DIR;
        deferred_lighting_vert_ = vkpipe::load_shader_module(
            device_, sd + "/deferred_lighting.vert.spv");
        deferred_lighting_frag_ = vkpipe::load_shader_module(
            device_, sd + "/deferred_lighting.frag.spv");

        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = deferred_lighting_vert_;
        cfg.frag = deferred_lighting_frag_;
        cfg.layout = deferred_lighting_pipeline_layout_;
        cfg.color_formats = { scene_color_format_ };  // writes staging
        cfg.depth_format  = VK_FORMAT_UNDEFINED;
        cfg.depth_test    = false;
        cfg.depth_write   = false;
        cfg.cull          = VK_CULL_MODE_NONE;
        cfg.vbindings.clear();
        cfg.vattrs.clear();
        deferred_lighting_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    log::info("[deferred] lighting pass initialised "
              "(staging buffer + pipeline ready; default = deferred ON)");
}

void VulkanEngine::destroy_deferred_lighting() {
    if (deferred_lighting_pipeline_) {
        vkDestroyPipeline(device_, deferred_lighting_pipeline_, nullptr);
        deferred_lighting_pipeline_ = VK_NULL_HANDLE;
    }
    if (deferred_lighting_pipeline_layout_) {
        vkDestroyPipelineLayout(device_,
                                deferred_lighting_pipeline_layout_, nullptr);
        deferred_lighting_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (deferred_lighting_vert_) {
        vkDestroyShaderModule(device_, deferred_lighting_vert_, nullptr);
        deferred_lighting_vert_ = VK_NULL_HANDLE;
    }
    if (deferred_lighting_frag_) {
        vkDestroyShaderModule(device_, deferred_lighting_frag_, nullptr);
        deferred_lighting_frag_ = VK_NULL_HANDLE;
    }
    if (deferred_lighting_desc_pool_) {
        vkDestroyDescriptorPool(device_,
                                deferred_lighting_desc_pool_, nullptr);
        deferred_lighting_desc_pool_ = VK_NULL_HANDLE;
        deferred_lighting_desc_set_ = VK_NULL_HANDLE;
    }
    if (deferred_lighting_desc_layout_) {
        vkDestroyDescriptorSetLayout(device_,
                                     deferred_lighting_desc_layout_, nullptr);
        deferred_lighting_desc_layout_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::render_deferred_lighting(VkCommandBuffer cmd) {
    if (!deferred_lighting_pipeline_) return;
    if (!deferred_lighting_active_) return;     // toggle off — skip pass entirely

    // Refresh descriptor set every frame (cheap — 6 writes). Could be
    // pinned at staging-target recreate; doing it every frame keeps
    // resize / swap-chain paths trivially correct.
    {
        VkDescriptorBufferInfo scene_bi{ scene_ubo_buffer_, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSetAccelerationStructureKHR tlas_w{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr,
            .accelerationStructureCount = 1, .pAccelerationStructures = &tlas_,
        };
        VkDescriptorImageInfo g0_ii{ linear_sampler_, gbuffer0_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo g1_ii{ linear_sampler_, gbuffer1_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo dep_ii{ linear_sampler_, depth_view_,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo ss_ii{ linear_sampler_, sun_shadow_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo sc_ii{ linear_sampler_, scene_color_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // Terrain shadow bake — falls back to gbuffer1 view if unset
        // (e.g. on terrain-less scenes); the shader gates the sample on
        // material_id == 3 anyway, so the bound texture is never read
        // off-terrain. linear_sampler_ matches cube.frag's 5x5 PCF.
        VkImageView terr_v = terrain_shadow_view_ ? terrain_shadow_view_
                                                   : gbuffer1_view_;
        VkDescriptorImageInfo ts_ii{ linear_sampler_, terr_v,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // SVGF GI image — cube.frag writes raw GI via imageStore in
        // VK_IMAGE_LAYOUT_GENERAL. Sampling a GENERAL-layout image as
        // sampler2D is permitted by the spec when the image was created
        // with both STORAGE and SAMPLED usage bits, which init_svgf_targets
        // does (setup.cpp ~1498). Falls back to gbuffer1 view if missing
        // so the descriptor stays valid for early-init crashes.
        VkImageView svgf_v = svgf_gi_view_ ? svgf_gi_view_ : gbuffer1_view_;
        VkDescriptorImageInfo svgf_ii{ linear_sampler_, svgf_v,
                                       VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[9]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = deferred_lighting_desc_set_; w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &scene_bi;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].pNext = &tlas_w;
        w[1].dstSet = deferred_lighting_desc_set_; w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        for (int i = 2; i < 9; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = deferred_lighting_desc_set_; w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[2].pImageInfo = &g0_ii;
        w[3].pImageInfo = &g1_ii;
        w[4].pImageInfo = &dep_ii;
        w[5].pImageInfo = &ss_ii;
        w[6].pImageInfo = &sc_ii;
        w[7].pImageInfo = &ts_ii;
        w[8].pImageInfo = &svgf_ii;
        vkUpdateDescriptorSets(device_, 9, w, 0, nullptr);
    }

    // Transition staging_color into COLOR_ATTACHMENT layout.
    vkinit::transition_image(cmd, staging_color_image_,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkClearValue clear{};
    clear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkRenderingAttachmentInfo color_att{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = staging_color_view_,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear,
    };
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, render_extent_ },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 1, .pColorAttachments = &color_att,
        .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{};
    vp.x = 0.0f; vp.y = 0.0f;
    vp.width  = static_cast<float>(render_extent_.width);
    vp.height = static_cast<float>(render_extent_.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                       deferred_lighting_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             deferred_lighting_pipeline_layout_, 0, 1,
                             &deferred_lighting_desc_set_, 0, nullptr);
    // Fill the push-constant struct: inverse-VP + active point lights.
    // The muzzle flash takes slot 0 when firing; lanterns + future
    // dynamic lights can take 1..3 once the engine collects them.
    DeferredLightingPC pcd{};
    pcd.inv_vp = current_frame_view_.inv_vp;
    // Slot 0: muzzle flash. The scene UBO already has muzzle_pos + colour;
    // mirror them here so the deferred path responds to the same trigger.
    if (muzzle_flash_timer_ > 0.0f) {
        const float t = std::min(1.0f, muzzle_flash_timer_ / kMuzzleFlashDuration);
        const float intensity = t * t * 12.0f;
        const glm::vec3 muzzle = player_.eye_position() + player_.forward() * 0.6f;
        pcd.light_pos[0] = glm::vec4(muzzle, 6.0f);   // 6 m radius
        pcd.light_col[0] = glm::vec4(1.00f, 0.85f, 0.55f, intensity);
    }
    vkCmdPushConstants(cmd, deferred_lighting_pipeline_layout_,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pcd), &pcd);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // Leave staging in TRANSFER_SRC so the caller can blit it back into
    // scene_color (the canonical input to SVGF/TAA/compose). Earlier
    // versions transitioned to SHADER_READ_ONLY which silently produced
    // garbage in the downstream blit on NVIDIA — vkCmdBlitImage requires
    // a TRANSFER_SRC_OPTIMAL (or GENERAL) source layout.
    vkinit::transition_image(cmd, staging_color_image_,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

} // namespace qlike
