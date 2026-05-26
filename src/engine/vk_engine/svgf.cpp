// SVGF deep denoiser (4a-svgf-deep) — variance moments + 3-pass a-trous.
//
// Builds on top of the SVGF Session 2 foundation (storage images at
// bindings 19/20/21 in scene_desc): this TU adds two new passes and
// the two new variance-moments storage targets (bindings 22/23).
//
//   svgf_moments.frag  — reprojects previous-frame luminance moments
//                        through the existing motion_vec image and EMA-
//                        updates with the current scene_color's luma.
//                        Writes the new slot. One full-screen pass per
//                        frame.
//
//   svgf_atrous.frag   — 5x5 edge-stop spatial filter. Runs 3 times per
//                        frame at strides 1, 2, 4 (the classic SVGF
//                        a-trous cascade). Inputs: a colour source, the
//                        scene depth (for the depth edge-stop and a
//                        derived geometric normal), and the moments
//                        slot just written. Outputs: filtered colour.
//
// Frame graph (when rt_.svgf_enabled):
//   world raster --(scene_color/depth/motion in SHADER_READ_ONLY)-->
//     svgf_moments (writes moments[cur])
//       svgf_atrous stride 1: scene_color -> atrous[0]
//       svgf_atrous stride 2: atrous[0]   -> atrous[1]
//       svgf_atrous stride 4: atrous[1]   -> scene_color  (write back)
//     -- TAA runs as before, but on a denoised scene_color --
//
// When rt_.svgf_enabled is false, none of the passes here are
// dispatched and TAA's existing single-pass a-trous covers the
// spatial reconstruction. The pipelines + descriptor sets stand up at
// init time either way so the toggle is a per-frame branch with zero
// allocation cost.

#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"

#include <cstring>
#include <string>

namespace qlike {

// ---- UBO layouts -----------------------------------------------------------
// Kept in this TU so the C++ side and the GLSL declarations stay in
// lock-step; bumping a field requires updating both halves at the same
// time.

struct SvgfMomentsUBO {
    glm::vec4 viewport;  // x: w, y: h, z: 1/w, w: 1/h
    glm::vec4 params;    // x: alpha_min, y: alpha_disocclusion,
                         // z: history_valid, w: 0
};

struct SvgfAtrousUBO {
    glm::vec4 viewport;       // x: w, y: h, z: 1/w, w: 1/h
    glm::vec4 params;         // x: stride, y: sigma_l, z: sigma_z, w: sigma_n
};

// Per-slice UBO stride. Worst-case minUniformBufferOffsetAlignment is
// 256 B on Turing/Ampere NVIDIA cards; padding each slice up keeps the
// vkCmdBindDescriptorSets offset legal across vendors. Plain UBOs (not
// DYNAMIC) bake the offset into the descriptor at write time, so this
// padding turns into a per-slice memcpy at the right address rather
// than a runtime offset shift.
inline constexpr VkDeviceSize kSvgfAtrousUBOStride = 256u;
static_assert(sizeof(SvgfAtrousUBO) <= kSvgfAtrousUBOStride,
              "SvgfAtrousUBO must fit in one aligned slice");

// ============================================================================
// Init / destroy
// ============================================================================

void VulkanEngine::init_svgf_passes() {
    std::string sd = QLIKE_SHADER_DIR;
    svgf_moments_frag_module_ = vkpipe::load_shader_module(
        device_, sd + "/svgf_moments.frag.spv");
    svgf_atrous_frag_module_  = vkpipe::load_shader_module(
        device_, sd + "/svgf_atrous.frag.spv");

    // ---------- moments pass ----------
    //
    // Set layout:
    //   0 = current scene_color (combined image sampler)
    //   1 = current depth       (combined image sampler)
    //   2 = motion_vec          (combined image sampler)
    //   3 = prev moments        (combined image sampler -- sampled, not
    //                            storage, so we can do a bilinear-ish
    //                            reproject; we still use texelFetch in
    //                            the shader because nearest gives
    //                            cleanest moments accumulation, but the
    //                            sampler binding is universally
    //                            supported and matches the colour
    //                            sampler binding type)
    //   4 = output moments slot (storage image -- written via imageStore)
    //   5 = UBO
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(SvgfMomentsUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &svgf_moments_ubo_buffer_,
                                 &svgf_moments_ubo_alloc_, nullptr),
                 "svgf moments ubo");
    }
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2u * 4u },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          2u * 1u },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2u * 1u },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = 2u, .poolSizeCount = 3, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr,
                                         &svgf_moments_desc_pool_),
                 "svgf moments desc pool");
    }
    {
        VkDescriptorSetLayoutBinding b[6]{};
        for (int i = 0; i < 4; ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        b[4].binding = 4;
        b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[4].descriptorCount = 1;
        b[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[5].binding = 5;
        b[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[5].descriptorCount = 1;
        b[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .bindingCount = 6, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                              &svgf_moments_set_layout_),
                 "svgf moments set layout");
    }
    {
        VkDescriptorSetLayout layouts[2] = {
            svgf_moments_set_layout_, svgf_moments_set_layout_,
        };
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = svgf_moments_desc_pool_,
            .descriptorSetCount = 2,
            .pSetLayouts = layouts,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai,
                                          svgf_moments_desc_sets_),
                 "svgf moments set alloc");
    }
    {
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &svgf_moments_set_layout_,
            .pushConstantRangeCount = 0, .pPushConstantRanges = nullptr,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &svgf_moments_pipeline_layout_),
                 "svgf moments pipeline layout");
    }
    {
        // The moments fragment shader writes via imageStore (the
        // STORAGE_IMAGE bound at descriptor 4). It still has to be a
        // graphics pipeline because dynamic-rendering requires at
        // least one colour attachment at vkCmdBeginRendering -- so the
        // shader returns a dummy colour into a throwaway attachment
        // we cycle through anyway (svgf_atrous_image_[0], which the
        // first a-trous pass overwrites immediately).
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = taa_vert_module_;     // reuse fullscreen-tri vert
        cfg.frag = svgf_moments_frag_module_;
        cfg.layout = svgf_moments_pipeline_layout_;
        cfg.color_format = scene_color_format_;
        cfg.depth_format = VK_FORMAT_UNDEFINED;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false;
        cfg.depth_write = false;
        svgf_moments_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    // ---------- a-trous pass ----------
    //
    // Set layout (6 sets total -- 2 parities x 3 chained passes):
    //   0 = colour input  (combined image sampler)
    //   1 = depth         (combined image sampler)
    //   2 = moments       (combined image sampler -- the freshly-written
    //                       slot from svgf_moments)
    //   3 = UBO (stride + sigmas + viewport)
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            // 3 strides per frame x 2 parities = 6 slices. Each slice
            // padded to kSvgfAtrousUBOStride so the
            // VkDescriptorBufferInfo::offset honours the device's
            // minUniformBufferOffsetAlignment (typically 256 B on
            // NVIDIA, 64 on AMD).
            .size = kSvgfAtrousUBOStride * 6u,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &svgf_atrous_ubo_buffer_,
                                 &svgf_atrous_ubo_alloc_, nullptr),
                 "svgf atrous ubo");
    }
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6u * 3u },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         6u * 1u },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = 6u, .poolSizeCount = 2, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr,
                                         &svgf_atrous_desc_pool_),
                 "svgf atrous desc pool");
    }
    {
        VkDescriptorSetLayoutBinding b[4]{};
        for (int i = 0; i < 3; ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        b[3].binding = 3;
        b[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[3].descriptorCount = 1;
        b[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .bindingCount = 4, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                              &svgf_atrous_set_layout_),
                 "svgf atrous set layout");
    }
    {
        VkDescriptorSetLayout layouts[6];
        for (int i = 0; i < 6; ++i) layouts[i] = svgf_atrous_set_layout_;
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = svgf_atrous_desc_pool_,
            .descriptorSetCount = 6,
            .pSetLayouts = layouts,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai,
                                          svgf_atrous_desc_sets_),
                 "svgf atrous set alloc");
    }
    {
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &svgf_atrous_set_layout_,
            .pushConstantRangeCount = 0, .pPushConstantRanges = nullptr,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &svgf_atrous_pipeline_layout_),
                 "svgf atrous pipeline layout");
    }
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = taa_vert_module_;
        cfg.frag = svgf_atrous_frag_module_;
        cfg.layout = svgf_atrous_pipeline_layout_;
        cfg.color_format = scene_color_format_;
        cfg.depth_format = VK_FORMAT_UNDEFINED;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false;
        cfg.depth_write = false;
        svgf_atrous_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    recreate_svgf_pass_targets();
    log::info("SVGF deep denoiser pipelines initialised "
              "(moments + 3-pass a-trous)");
}

void VulkanEngine::destroy_svgf_passes() {
    if (svgf_atrous_pipeline_) {
        vkDestroyPipeline(device_, svgf_atrous_pipeline_, nullptr);
        svgf_atrous_pipeline_ = VK_NULL_HANDLE;
    }
    if (svgf_atrous_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, svgf_atrous_pipeline_layout_, nullptr);
        svgf_atrous_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (svgf_atrous_frag_module_) {
        vkDestroyShaderModule(device_, svgf_atrous_frag_module_, nullptr);
        svgf_atrous_frag_module_ = VK_NULL_HANDLE;
    }
    if (svgf_atrous_desc_pool_) {
        vkDestroyDescriptorPool(device_, svgf_atrous_desc_pool_, nullptr);
        svgf_atrous_desc_pool_ = VK_NULL_HANDLE;
    }
    if (svgf_atrous_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, svgf_atrous_set_layout_, nullptr);
        svgf_atrous_set_layout_ = VK_NULL_HANDLE;
    }
    if (svgf_atrous_ubo_buffer_) {
        vmaDestroyBuffer(allocator_, svgf_atrous_ubo_buffer_,
                         svgf_atrous_ubo_alloc_);
        svgf_atrous_ubo_buffer_ = VK_NULL_HANDLE;
        svgf_atrous_ubo_alloc_  = nullptr;
    }
    if (svgf_moments_pipeline_) {
        vkDestroyPipeline(device_, svgf_moments_pipeline_, nullptr);
        svgf_moments_pipeline_ = VK_NULL_HANDLE;
    }
    if (svgf_moments_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, svgf_moments_pipeline_layout_, nullptr);
        svgf_moments_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (svgf_moments_frag_module_) {
        vkDestroyShaderModule(device_, svgf_moments_frag_module_, nullptr);
        svgf_moments_frag_module_ = VK_NULL_HANDLE;
    }
    if (svgf_moments_desc_pool_) {
        vkDestroyDescriptorPool(device_, svgf_moments_desc_pool_, nullptr);
        svgf_moments_desc_pool_ = VK_NULL_HANDLE;
    }
    if (svgf_moments_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, svgf_moments_set_layout_, nullptr);
        svgf_moments_set_layout_ = VK_NULL_HANDLE;
    }
    if (svgf_moments_ubo_buffer_) {
        vmaDestroyBuffer(allocator_, svgf_moments_ubo_buffer_,
                         svgf_moments_ubo_alloc_);
        svgf_moments_ubo_buffer_ = VK_NULL_HANDLE;
        svgf_moments_ubo_alloc_  = nullptr;
    }
}

// ---- Descriptor (re)wires --------------------------------------------------
// Called from init_svgf_passes (one-shot after pipelines stand up) and
// from recreate_svgf_targets (after a destroy+realloc that swapped the
// image handles backing scene_color / depth / motion_vec / moments /
// atrous). All views referenced here must be live -- callers gate by
// checking scene_color_view_ etc. before invoking.

void VulkanEngine::recreate_svgf_pass_targets() {
    if (!svgf_moments_set_layout_ || !svgf_atrous_set_layout_) return;
    if (!linear_sampler_ || !scene_color_view_ || !depth_view_ ||
        !motion_vec_view_) {
        return;
    }
    if (!svgf_moments_view_[0] || !svgf_atrous_view_[0]) return;

    // ---- moments pass -----------------------------------------------------
    // Per frame parity, the input is the OTHER moments slot (the one
    // written F-1) and the output is the slot for THIS frame.
    for (int parity = 0; parity < 2; ++parity) {
        const int cur_slot  = parity;
        const int prev_slot = 1 - parity;
        VkDescriptorImageInfo i_color  { linear_sampler_, scene_color_view_,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth  { linear_sampler_, depth_view_,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_motion { linear_sampler_, motion_vec_view_,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // Prev moments: sample via the same linear_sampler_. The image
        // lives in GENERAL so we bind that layout (no transition cost
        // -- GENERAL is shader-readable, just suboptimal for cache).
        // Spec allows sampling a STORAGE_IMAGE-usage image because we
        // also added SAMPLED usage at allocation time.
        VkDescriptorImageInfo i_prev_m { linear_sampler_,
                                          svgf_moments_view_[prev_slot],
                                          VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo i_cur_m  { VK_NULL_HANDLE,
                                          svgf_moments_view_[cur_slot],
                                          VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo i_ubo { svgf_moments_ubo_buffer_, 0,
                                        VK_WHOLE_SIZE };

        VkWriteDescriptorSet w[6]{};
        for (int i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = svgf_moments_desc_sets_[parity];
            w[i].dstBinding = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[0].pImageInfo = &i_color;
        w[1].pImageInfo = &i_depth;
        w[2].pImageInfo = &i_motion;
        w[3].pImageInfo = &i_prev_m;
        w[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[4].dstSet = svgf_moments_desc_sets_[parity];
        w[4].dstBinding = 4;
        w[4].descriptorCount = 1;
        w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[4].pImageInfo = &i_cur_m;
        w[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[5].dstSet = svgf_moments_desc_sets_[parity];
        w[5].dstBinding = 5;
        w[5].descriptorCount = 1;
        w[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[5].pBufferInfo = &i_ubo;
        vkUpdateDescriptorSets(device_, 6, w, 0, nullptr);
    }

    // ---- a-trous pass -----------------------------------------------------
    // 6 sets = 2 parities x 3 passes. The colour source per pass is:
    //   pass 0: scene_color (the world pass's output, in SHADER_READ_ONLY
    //                         by the time SVGF runs)
    //   pass 1: atrous_view_[0]
    //   pass 2: atrous_view_[1]
    // The moments slot read by all 3 passes is the SAME (the one
    // svgf_moments just wrote). Index = parity * 3 + pass.
    for (int parity = 0; parity < 2; ++parity) {
        const int cur_m_slot = parity;
        for (int pass = 0; pass < 3; ++pass) {
            const int set_idx = parity * 3 + pass;
            VkImageView color_src;
            switch (pass) {
                case 0: color_src = scene_color_view_;        break;
                case 1: color_src = svgf_atrous_view_[0];     break;
                default:color_src = svgf_atrous_view_[1];     break;
            }
            VkDescriptorImageInfo i_color   { linear_sampler_, color_src,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo i_depth   { linear_sampler_, depth_view_,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo i_moments { linear_sampler_,
                                              svgf_moments_view_[cur_m_slot],
                                              VK_IMAGE_LAYOUT_GENERAL };
            // Per-pass UBO sub-region. Each pass within a parity reads
            // its own SvgfAtrousUBO slice; pre-computed offset baked
            // into pBufferInfo means we never re-bind/update the
            // descriptor mid-frame, only the UBO contents flip on
            // resolution / sigma changes.
            VkDescriptorBufferInfo i_ubo {
                svgf_atrous_ubo_buffer_,
                static_cast<VkDeviceSize>(set_idx) * kSvgfAtrousUBOStride,
                sizeof(SvgfAtrousUBO),
            };

            VkWriteDescriptorSet w[4]{};
            for (int i = 0; i < 3; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet = svgf_atrous_desc_sets_[set_idx];
                w[i].dstBinding = static_cast<uint32_t>(i);
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            }
            w[0].pImageInfo = &i_color;
            w[1].pImageInfo = &i_depth;
            w[2].pImageInfo = &i_moments;
            w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[3].dstSet = svgf_atrous_desc_sets_[set_idx];
            w[3].dstBinding = 3;
            w[3].descriptorCount = 1;
            w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[3].pBufferInfo = &i_ubo;
            vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
        }
    }
}

// ============================================================================
// Per-frame render: moments update + 3-pass a-trous
// ============================================================================
//
// Caller must have transitioned scene_color, depth, motion_vec to
// SHADER_READ_ONLY (TAA's pre-existing post-world batch barrier covers
// scene_color + motion_vec + depth, so render_svgf slots in right
// after that without an extra transition). All SVGF storage images
// stay in GENERAL for their entire lifetime; the a-trous targets stay
// in SHADER_READ_ONLY between frames and are flipped to
// COLOR_ATTACHMENT for the duration of each pass that writes them.

void VulkanEngine::render_svgf(VkCommandBuffer cmd, uint32_t frame_parity) {
    if (!rt_.svgf_enabled) return;
    if (!svgf_moments_pipeline_ || !svgf_atrous_pipeline_) return;

    const uint32_t parity = frame_parity & 1u;
    const float w = static_cast<float>(render_extent_.width);
    const float h = static_cast<float>(render_extent_.height);

    // ---- write UBOs ------------------------------------------------------
    {
        SvgfMomentsUBO m{};
        m.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
        // alpha_min: lower bound on the EMA blend (alpha = 1 means take
        // ONLY the current sample; alpha = 0.1 keeps 90% history). 0.1
        // is the SVGF reference value -- gives ~10-frame temporal
        // window, which lines up with how long ReSTIR holds a
        // reservoir before disocclusion-reset.
        // alpha_disocclusion: alpha to fall back to when motion-vec
        // reprojection lands off-screen or the depth mismatch flags a
        // new surface. 1.0 (take the fresh sample only) so the moments
        // instantly track the new geometry.
        // history_valid: gate; first frame at engine start has
        // prev_view_proj_valid_ == false, then it pins to true.
        m.params = glm::vec4(0.1f, 1.0f,
                              prev_view_proj_valid_ ? 1.0f : 0.0f, 0.0f);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, svgf_moments_ubo_alloc_, &ai);
        if (ai.pMappedData) std::memcpy(ai.pMappedData, &m, sizeof(m));
    }
    {
        // sigma_l: luminance edge-stop width, scaled INSIDE the shader
        // by sqrt(variance) so noisy pixels get a softer luma weight
        // while stable surfaces clamp tight. 4.0 is the SVGF default.
        // sigma_z: depth edge-stop width (in clip-Z units). 1.0 means
        // ~1% depth difference at unity z-gradient is the falloff
        // point; the shader scales by max(0.01, |dz/dxy| * stride)
        // so distant pixels (large dz) tolerate larger absolute z
        // differences. The 0.01 floor stops planar surfaces (dz~=0)
        // from rejecting all neighbours.
        // sigma_n: normal edge-stop sharpness. Larger = tighter
        // tolerance for normal misalignment. 128 is the SVGF default
        // (max(0, dot(N_p, N_q))^128 -- already 0 at ~7 degrees).
        const float kSigmaL = 4.0f;
        const float kSigmaZ = 1.0f;
        const float kSigmaN = 128.0f;
        const int strides[3] = { 1, 2, 4 };
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, svgf_atrous_ubo_alloc_, &ai);
        if (ai.pMappedData) {
            // Write each slice at its aligned offset (per-slice stride
            // = kSvgfAtrousUBOStride, sized for the worst-case
            // minUniformBufferOffsetAlignment so the matching
            // VkDescriptorBufferInfo::offset stays legal).
            for (int p = 0; p < 2; ++p) {
                for (int s = 0; s < 3; ++s) {
                    SvgfAtrousUBO u{};
                    u.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
                    u.params = glm::vec4(static_cast<float>(strides[s]),
                                         kSigmaL, kSigmaZ, kSigmaN);
                    char* dst = static_cast<char*>(ai.pMappedData) +
                                static_cast<size_t>(p * 3 + s) *
                                    kSvgfAtrousUBOStride;
                    std::memcpy(dst, &u, sizeof(u));
                }
            }
        }
    }

    // ---- moments pass: 1 fullscreen tri ----------------------------------
    //
    // The fragment shader's real output is via imageStore into
    // svgf_moments_image_[parity] (binding 4). The dummy colour
    // attachment exists only because dynamic-rendering requires at
    // least one attachment; we reuse svgf_atrous_image_[0] for it
    // (about to be overwritten by the first a-trous pass anyway). The
    // image flips into COLOR_ATTACHMENT for the moments pass, then
    // back into COLOR_ATTACHMENT (no-op) for the a-trous pass, then
    // SHADER_READ_ONLY after.
    {
        vkinit::transition_image(cmd, svgf_atrous_image_[0],
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkClearValue clear_dummy{};
        clear_dummy.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        auto color = vkinit::color_attachment_info(
            svgf_atrous_view_[0], &clear_dummy,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0, 0}, render_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &color,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{};
        vp.x = 0.0f; vp.y = 0.0f;
        vp.width = w; vp.height = h;
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, render_extent_ };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          svgf_moments_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                svgf_moments_pipeline_layout_, 0, 1,
                                &svgf_moments_desc_sets_[parity], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        // Two barriers in one batched vkCmdPipelineBarrier:
        //   1. SHADER_WRITE -> SHADER_READ on the moments storage
        //      image so pass 0 (a-trous, samples this image at
        //      binding 2) sees the imageStore result.
        //   2. COLOR_ATTACHMENT_WRITE -> COLOR_ATTACHMENT_WRITE on
        //      svgf_atrous_image_[0] so pass 0's render-pass write
        //      strictly orders after the moments pass's dummy write
        //      (Vulkan requires explicit WAW sync between back-to-
        //      back render passes targeting the same image, even at
        //      the same layout).
        VkImageMemoryBarrier b[2]{};
        b[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].image = svgf_moments_image_[parity];
        b[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b[1].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[1].image = svgf_atrous_image_[0];
        b[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 2, b);
    }

    // ---- a-trous chain: 3 fragment passes --------------------------------
    //
    // Pass 0 (stride 1): scene_color -> atrous[0]
    // Pass 1 (stride 2): atrous[0]   -> atrous[1]
    // Pass 2 (stride 4): atrous[1]   -> scene_color  (writes back)
    //
    // Layout flips per pass:
    //   pre  pass 0: atrous[0] -> COLOR_ATTACHMENT (already set above
    //                              for the moments dummy write). The
    //                              moments pass leaves it in that
    //                              layout because we never flipped it
    //                              back; pass 0 reuses the layout.
    //   post pass 0: atrous[0] -> SHADER_READ_ONLY
    //   pre  pass 1: atrous[1] -> COLOR_ATTACHMENT
    //   post pass 1: atrous[1] -> SHADER_READ_ONLY
    //   pre  pass 2: scene_color -> COLOR_ATTACHMENT
    //   post pass 2: scene_color -> SHADER_READ_ONLY  (TAA expects this)

    auto run_pass = [&](int pass_idx) {
        const int set_idx = parity * 3 + pass_idx;
        VkImageView dst_view;
        VkImage     dst_image;
        switch (pass_idx) {
            case 0: dst_view = svgf_atrous_view_[0];
                    dst_image = svgf_atrous_image_[0]; break;
            case 1: dst_view = svgf_atrous_view_[1];
                    dst_image = svgf_atrous_image_[1]; break;
            default:dst_view = scene_color_view_;
                    dst_image = scene_color_image_;    break;
        }
        if (pass_idx == 1) {
            vkinit::transition_image(cmd, svgf_atrous_image_[1],
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        } else if (pass_idx == 2) {
            vkinit::transition_image(cmd, scene_color_image_,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
        // pass 0 dst (atrous[0]) is already in COLOR_ATTACHMENT from
        // the moments dummy-write block above -- no transition needed.

        VkClearValue cl{};
        cl.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        auto color = vkinit::color_attachment_info(
            dst_view, &cl,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr, .flags = 0,
            .renderArea = { {0, 0}, render_extent_ },
            .layerCount = 1, .viewMask = 0,
            .colorAttachmentCount = 1, .pColorAttachments = &color,
            .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{};
        vp.x = 0.0f; vp.y = 0.0f;
        vp.width = w; vp.height = h;
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, render_extent_ };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          svgf_atrous_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                svgf_atrous_pipeline_layout_, 0, 1,
                                &svgf_atrous_desc_sets_[set_idx],
                                0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        // dst -> SHADER_READ_ONLY so the next pass (or TAA, after
        // pass 2) can sample it.
        vkinit::transition_image(cmd, dst_image,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    run_pass(0);
    run_pass(1);
    run_pass(2);
}

} // namespace qlike
