// SSDM — Screen-Space Displacement Mapping (Lobel 2008).
//
// Phase 1: dedicated render pass after the main world pass that draws the
// SPOM-wall brushes only and writes a 2D screen-space displacement vector
// per visible pixel into `ssdm_disp_image_` (RG16F at render_extent_).
// Vector encodes "this pixel's brick TOP would project to here" — Phase 4's
// compose pass uses pyramid B's barycenter to remap scene_color.
//
// Architecture: a SEPARATE pass (not a 3rd attachment on the main pass) so
// existing pipelines, blend states and fragment shaders stay untouched.
// Costs an extra over-draw of the visible SPOM walls (~50 brushes in the
// castle scene), runs at render_extent_, depth-tests against the main
// pass's depth buffer (LESS_OR_EQUAL, no writes) so we only paint pixels
// where the wall actually rasterised.

#include "internal.h"

#include "engine/vk_engine.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/mesh.h"
#include "engine/log.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstdint>
#include <string>

using namespace qlike;

namespace {

bool brush_is_spom_wall(int tex_albedo) {
    // Mirrors height_idx_for_albedo() in cube.frag and is_spom_wall_albedo
    // in level.cpp. Keep in lock-step or the SSDM pass and cube.frag's
    // SPOM gating will diverge and produce dropouts.
    return tex_albedo == 1 || tex_albedo == 4;
}

}  // namespace

void VulkanEngine::init_ssdm() {
    std::string sd = QLIKE_SHADER_DIR;
    ssdm_disp_frag_module_ =
        vkpipe::load_shader_module(device_, sd + "/ssdm_disp.frag.spv");

    // ---- Phase 1 pipeline: dedicated displacement-vector raster pass ----
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert            = vert_module_;
        cfg.frag            = ssdm_disp_frag_module_;
        cfg.layout          = pipeline_layout_;
        cfg.color_formats   = { ssdm_disp_format_ };
        cfg.depth_format    = depth_format_;
        cfg.depth_test      = true;
        cfg.depth_write     = false;
        cfg.depth_compare   = VK_COMPARE_OP_LESS_OR_EQUAL;
        cfg.alpha_blend_color0_only = false;

        VkVertexInputBindingDescription vb{};
        vb.binding   = 0;
        vb.stride    = sizeof(Vertex);
        vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        cfg.vbindings.push_back(vb);

        VkVertexInputAttributeDescription a0{ 0, 0,
            VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) };
        VkVertexInputAttributeDescription a1{ 1, 0,
            VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };
        VkVertexInputAttributeDescription a2{ 2, 0,
            VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) };
        cfg.vattrs = { a0, a1, a2 };

        ssdm_disp_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    // ---- Phase 3 pipeline: barycenter refinement (fullscreen-tri frag) ----
    {
        ssdm_refine_comp_module_ =
            vkpipe::load_shader_module(device_, sd + "/ssdm_refine.frag.spv");

        // Set layout: 2 sampled images (pyrA + pyrB), no storage (output is
        // a color attachment, not an image2D). One descriptor set is
        // shared across all kSsdmMipsMax iterations -- mip selection is
        // done via textureLod() in the shader, driven by push constants.
        VkDescriptorSetLayoutBinding b[2]{};
        for (int i = 0; i < 2; ++i) {
            b[i].binding         = static_cast<uint32_t>(i);
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &dlci, nullptr,
                                              &ssdm_refine_set_layout_),
                 "ssdm refine desc layout");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset     = 0;
        pc.size       = 8 * sizeof(uint32_t);  // 2 uvec2 + 4 uint
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &ssdm_refine_set_layout_,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &ssdm_refine_pipeline_layout_),
                 "ssdm refine pipeline layout");

        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert            = taa_vert_module_;   // fullscreen triangle
        cfg.frag            = ssdm_refine_comp_module_;
        cfg.layout          = ssdm_refine_pipeline_layout_;
        cfg.color_formats   = { ssdm_disp_format_ };  // RG16F
        cfg.depth_format    = VK_FORMAT_UNDEFINED;     // no depth
        cfg.depth_test      = false;
        cfg.depth_write     = false;
        cfg.alpha_blend_color0_only = false;
        // CRITICAL: taa.vert's fullscreen triangle is back-facing under
        // VK_FRONT_FACE_COUNTER_CLOCKWISE (the PipelineBuilder default).
        // Default cull = BACK_BIT → entire triangle culled, image stays
        // unwritten. TAA itself uses the same workaround.
        cfg.cull = VK_CULL_MODE_NONE;
        ssdm_refine_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

        // Descriptor pool: 1 set is enough (shared across all mip iterations
        // because the per-mip view is selected via render-pass color
        // attachment, not via descriptor binding).
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dpci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1, .pPoolSizes = &ps,
        };
        vk_check(vkCreateDescriptorPool(device_, &dpci, nullptr,
                                         &ssdm_refine_desc_pool_),
                 "ssdm refine desc pool");

        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = ssdm_refine_desc_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &ssdm_refine_set_layout_,
        };
        // We only use slot 0; the kSsdmMipsMax-sized array is leftover from
        // an earlier per-mip-set design (kept so future changes don't have
        // to reshape the field). Slot 0 holds the shared "all mips of pyrA
        // + all mips of pyrB" view bindings.
        vk_check(vkAllocateDescriptorSets(device_, &dai,
                                           &ssdm_refine_desc_sets_[0]),
                 "ssdm refine alloc desc set");
    }

    // ---- Phase 4 pipeline: compose remap ----
    {
        ssdm_compose_frag_module_ =
            vkpipe::load_shader_module(device_, sd + "/ssdm_compose.frag.spv");

        // Set layout: 2 sampled images -- scene_color + pyrB.
        VkDescriptorSetLayoutBinding b[2]{};
        for (int i = 0; i < 2; ++i) {
            b[i].binding         = static_cast<uint32_t>(i);
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &dlci, nullptr,
                                              &ssdm_compose_set_layout_),
                 "ssdm compose desc layout");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset     = 0;
        pc.size       = 4 * sizeof(uint32_t);  // uvec2 + uvec2 pad
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &ssdm_compose_set_layout_,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &ssdm_compose_pipeline_layout_),
                 "ssdm compose pipeline layout");

        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert            = taa_vert_module_;
        cfg.frag            = ssdm_compose_frag_module_;
        cfg.layout          = ssdm_compose_pipeline_layout_;
        cfg.color_formats   = { scene_color_format_ };
        cfg.depth_format    = VK_FORMAT_UNDEFINED;
        cfg.depth_test      = false;
        cfg.depth_write     = false;
        cfg.alpha_blend_color0_only = false;
        cfg.cull            = VK_CULL_MODE_NONE;  // see refine pipeline note
        ssdm_compose_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dpci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1, .pPoolSizes = &ps,
        };
        vk_check(vkCreateDescriptorPool(device_, &dpci, nullptr,
                                         &ssdm_compose_desc_pool_),
                 "ssdm compose desc pool");

        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = ssdm_compose_desc_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &ssdm_compose_set_layout_,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai,
                                           &ssdm_compose_desc_set_),
                 "ssdm compose alloc set");
    }

    // ---- Shell silhouette pipeline ----
    // Runs as a SEPARATE render pass AFTER the post-world depth transition
    // so shell.frag can sample the depth buffer and discard pixels where
    // the main pass already drew a wall (= shell only fills silhouette
    // extension band past brush edges, not over the wall itself).
    //
    // Pipeline layout: 2 sets -- scene_desc_set (set 0, for cube textures
    // + SPOM heights + scene UBO) and shell_depth_set (set 1, for the
    // depth-buffer sampler). No depth attachment in the render pass; depth
    // is read-only via the sampler. depth_test = false, depth_write =
    // false (no attachment available).
    {
        shell_vert_module_ =
            vkpipe::load_shader_module(device_, sd + "/shell.vert.spv");
        shell_frag_module_ =
            vkpipe::load_shader_module(device_, sd + "/shell.frag.spv");

        // Set 1: depth sampler.
        VkDescriptorSetLayoutBinding db{};
        db.binding         = 0;
        db.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        db.descriptorCount = 1;
        db.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1, .pBindings = &db,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &dlci, nullptr,
                                              &shell_depth_set_layout_),
                 "shell depth desc layout");

        VkDescriptorSetLayout layouts[2] = {
            scene_desc_set_layout_, shell_depth_set_layout_
        };
        VkPushConstantRange pcr{};
        pcr.stageFlags = kPushConstantStages;
        pcr.offset     = 0;
        pcr.size       = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 2, .pSetLayouts = layouts,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
        };
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                         &shell_pipeline_layout_),
                 "shell pipeline layout");

        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1, .pPoolSizes = &ps,
        };
        vk_check(vkCreateDescriptorPool(device_, &dpci, nullptr,
                                         &shell_depth_desc_pool_),
                 "shell depth desc pool");
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = shell_depth_desc_pool_,
            .descriptorSetCount = 1, .pSetLayouts = &shell_depth_set_layout_,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai,
                                           &shell_depth_desc_set_),
                 "shell depth alloc desc set");

        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert            = shell_vert_module_;
        cfg.frag            = shell_frag_module_;
        cfg.layout          = shell_pipeline_layout_;
        cfg.color_formats   = { scene_color_format_, motion_vec_format_ };
        cfg.depth_format    = VK_FORMAT_UNDEFINED;  // no depth attachment
        cfg.depth_test      = false;
        cfg.depth_write     = false;
        cfg.alpha_blend_color0_only = false;

        VkVertexInputBindingDescription vb{};
        vb.binding   = 0;
        vb.stride    = sizeof(Vertex);
        vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        cfg.vbindings.push_back(vb);
        VkVertexInputAttributeDescription a0{ 0, 0,
            VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) };
        VkVertexInputAttributeDescription a1{ 1, 0,
            VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };
        VkVertexInputAttributeDescription a2{ 2, 0,
            VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) };
        cfg.vattrs = { a0, a1, a2 };

        shell_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }

    recreate_ssdm_targets();
    log::info("SSDM init: disp + refine + compose + shell pipelines ready");
}

void VulkanEngine::recreate_ssdm_targets() {
    // Destroy views first (image is referenced by them).
    if (ssdm_disp_view_) {
        vkDestroyImageView(device_, ssdm_disp_view_, nullptr);
        ssdm_disp_view_ = VK_NULL_HANDLE;
    }
    if (ssdm_disp_pyramid_view_) {
        vkDestroyImageView(device_, ssdm_disp_pyramid_view_, nullptr);
        ssdm_disp_pyramid_view_ = VK_NULL_HANDLE;
    }
    if (ssdm_disp_image_) {
        vmaDestroyImage(allocator_, ssdm_disp_image_, ssdm_disp_alloc_);
        ssdm_disp_image_ = VK_NULL_HANDLE;
        ssdm_disp_alloc_ = nullptr;
    }

    VkImageCreateInfo ici{
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = ssdm_disp_format_,
        .extent                = { render_extent_.width, render_extent_.height, 1 },
        // kSsdmMipsMax levels: level 0 = Phase 1 raster output, levels 1..N-1
        // = box-averaged from level below via vkCmdBlitImage(LINEAR) in
        // Phase 2's build_ssdm_pyramid_a().
        .mipLevels             = kSsdmMipsMax,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        // COLOR  -> Phase 1 render pass writes level 0
        // SAMPLED -> Phase 3 compute samples all mips
        // STORAGE -> Phase 3 compute writes to pyramid B (reused layout
        //            for both pyramids; same flags keep one alloc path)
        // TRANSFER_{SRC,DST} -> Phase 2's mip-build blit chain
        .usage                 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_STORAGE_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &ssdm_disp_image_,
                            &ssdm_disp_alloc_, nullptr),
             "ssdm disp image");

    // Level-0 view: the Phase 1 render pass binds this as its single
    // color attachment. Must be a 1-mip-level view so the rasteriser
    // writes exclusively to mip 0, leaving 1..N-1 untouched until the
    // Phase 2 blit chain populates them.
    VkImageViewCreateInfo vci{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .image            = ssdm_disp_image_,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = ssdm_disp_format_,
        .components       = {},
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &ssdm_disp_view_),
             "ssdm disp view");

    // All-mip view: Phase 3's sampling lookups need access to every mip
    // (the barycenter iteration samples pyramid A at the level matching
    // the current iteration's resolution).
    VkImageViewCreateInfo vci_pyr = vci;
    vci_pyr.subresourceRange.levelCount = kSsdmMipsMax;
    vk_check(vkCreateImageView(device_, &vci_pyr, nullptr,
                               &ssdm_disp_pyramid_view_),
             "ssdm disp pyramid view");

    // ---- Pyramid B (Phase 3 output, Phase 4 input) ----
    if (ssdm_pyrb_pyramid_view_) {
        vkDestroyImageView(device_, ssdm_pyrb_pyramid_view_, nullptr);
        ssdm_pyrb_pyramid_view_ = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < kSsdmMipsMax; ++i) {
        if (ssdm_pyrb_mip_views_[i]) {
            vkDestroyImageView(device_, ssdm_pyrb_mip_views_[i], nullptr);
            ssdm_pyrb_mip_views_[i] = VK_NULL_HANDLE;
        }
    }
    if (ssdm_pyrb_image_) {
        vmaDestroyImage(allocator_, ssdm_pyrb_image_, ssdm_pyrb_alloc_);
        ssdm_pyrb_image_ = VK_NULL_HANDLE;
        ssdm_pyrb_alloc_ = nullptr;
    }

    {
        VkImageCreateInfo ici_b = ici;
        // pyrB needs COLOR_ATTACHMENT (Phase 3 writes per mip as color out)
        // and SAMPLED (Phase 3 reads prev mip + Phase 4 reads level 0).
        // No transfer flags required -- pyrB is fully rebuilt every frame.
        ici_b.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT;
        vk_check(vmaCreateImage(allocator_, &ici_b, &aci, &ssdm_pyrb_image_,
                                &ssdm_pyrb_alloc_, nullptr),
                 "ssdm pyrB image");

        VkImageViewCreateInfo vci_b_pyr = vci;
        vci_b_pyr.image                       = ssdm_pyrb_image_;
        vci_b_pyr.subresourceRange.levelCount = kSsdmMipsMax;
        vk_check(vkCreateImageView(device_, &vci_b_pyr, nullptr,
                                    &ssdm_pyrb_pyramid_view_),
                 "ssdm pyrB pyramid view");

        for (uint32_t i = 0; i < kSsdmMipsMax; ++i) {
            VkImageViewCreateInfo vci_b_mip = vci;
            vci_b_mip.image                          = ssdm_pyrb_image_;
            vci_b_mip.subresourceRange.baseMipLevel  = i;
            vci_b_mip.subresourceRange.levelCount    = 1;
            vk_check(vkCreateImageView(device_, &vci_b_mip, nullptr,
                                        &ssdm_pyrb_mip_views_[i]),
                     "ssdm pyrB mip view");
        }
    }

    // Pre-transition ALL mips to SHADER_READ_ONLY_OPTIMAL. Phase 1's
    // render pass flips level 0 to COLOR_ATTACHMENT explicitly and
    // back; Phase 2 flips levels 1..N-1 to TRANSFER_DST and back.
    // Phase 3 flips per-mip pyrB to COLOR_ATTACHMENT and back.
    // Both pyramids expect SHADER_READ as the steady-state idle layout.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier barriers[2]{};
        barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].srcAccessMask       = 0;
        barriers[0].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image               = ssdm_disp_image_;
        barriers[0].subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT,
                                             0, kSsdmMipsMax, 0, 1 };
        barriers[1] = barriers[0];
        barriers[1].image               = ssdm_pyrb_image_;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, barriers);
    });

    // Wire descriptor set (pyrA at binding 0, pyrB-prev at binding 1).
    // Both are all-mip sampler views; the shader picks the mip via
    // textureLod(). Linear sampler reused from TAA.
    if (ssdm_refine_desc_sets_[0] && linear_sampler_) {
        VkDescriptorImageInfo i_pyrA{ linear_sampler_,
                                      ssdm_disp_pyramid_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_pyrB{ linear_sampler_,
                                      ssdm_pyrb_pyramid_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = ssdm_refine_desc_sets_[0];
            w[i].dstBinding      = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[0].pImageInfo = &i_pyrA;
        w[1].pImageInfo = &i_pyrB;
        vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    }

    // ---- Phase 4: SSDM-remap output image ----
    if (ssdm_remap_view_) {
        vkDestroyImageView(device_, ssdm_remap_view_, nullptr);
        ssdm_remap_view_ = VK_NULL_HANDLE;
    }
    if (ssdm_remap_image_) {
        vmaDestroyImage(allocator_, ssdm_remap_image_, ssdm_remap_alloc_);
        ssdm_remap_image_ = VK_NULL_HANDLE;
        ssdm_remap_alloc_ = nullptr;
    }
    {
        VkImageCreateInfo ici_r = ici;
        ici_r.format = scene_color_format_;
        ici_r.mipLevels = 1;
        ici_r.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        vk_check(vmaCreateImage(allocator_, &ici_r, &aci, &ssdm_remap_image_,
                                &ssdm_remap_alloc_, nullptr),
                 "ssdm remap image");

        VkImageViewCreateInfo vci_r = vci;
        vci_r.image  = ssdm_remap_image_;
        vci_r.format = scene_color_format_;
        vci_r.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci_r, nullptr,
                                    &ssdm_remap_view_),
                 "ssdm remap view");

        // Initial transition: UNDEFINED → COLOR_ATTACHMENT (compose_ssdm
        // begins by rendering into it, so it expects that layout each frame).
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            VkImageMemoryBarrier b{};
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image            = ssdm_remap_image_;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        });
    }

    // Wire Phase 4 desc set: scene_color (binding 0) + pyrB (binding 1).
    if (ssdm_compose_desc_set_ && linear_sampler_ &&
        scene_color_view_ && ssdm_pyrb_pyramid_view_) {
        VkDescriptorImageInfo i_scene{ linear_sampler_, scene_color_view_,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_pyrB{ linear_sampler_,
                                      ssdm_pyrb_pyramid_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = ssdm_compose_desc_set_;
            w[i].dstBinding      = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[0].pImageInfo = &i_scene;
        w[1].pImageInfo = &i_pyrB;
        vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    }

    // Wire shell depth sampler: bind the depth view at set 1, binding 0.
    // depth_view_ aspect mask is DEPTH; the sampler reads the .r channel.
    if (shell_depth_desc_set_ && linear_sampler_ && depth_view_) {
        VkDescriptorImageInfo i_depth{ linear_sampler_, depth_view_,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = shell_depth_desc_set_;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &i_depth;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
}

void VulkanEngine::destroy_ssdm() {
    if (ssdm_disp_pipeline_) {
        vkDestroyPipeline(device_, ssdm_disp_pipeline_, nullptr);
        ssdm_disp_pipeline_ = VK_NULL_HANDLE;
    }
    if (ssdm_refine_pipeline_) {
        vkDestroyPipeline(device_, ssdm_refine_pipeline_, nullptr);
        ssdm_refine_pipeline_ = VK_NULL_HANDLE;
    }
    if (ssdm_refine_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, ssdm_refine_pipeline_layout_, nullptr);
        ssdm_refine_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (ssdm_refine_desc_pool_) {
        vkDestroyDescriptorPool(device_, ssdm_refine_desc_pool_, nullptr);
        ssdm_refine_desc_pool_ = VK_NULL_HANDLE;
    }
    if (ssdm_refine_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, ssdm_refine_set_layout_, nullptr);
        ssdm_refine_set_layout_ = VK_NULL_HANDLE;
    }
    if (ssdm_disp_frag_module_) {
        vkDestroyShaderModule(device_, ssdm_disp_frag_module_, nullptr);
        ssdm_disp_frag_module_ = VK_NULL_HANDLE;
    }
    if (ssdm_refine_comp_module_) {
        vkDestroyShaderModule(device_, ssdm_refine_comp_module_, nullptr);
        ssdm_refine_comp_module_ = VK_NULL_HANDLE;
    }
    if (ssdm_disp_view_) {
        vkDestroyImageView(device_, ssdm_disp_view_, nullptr);
        ssdm_disp_view_ = VK_NULL_HANDLE;
    }
    if (ssdm_disp_pyramid_view_) {
        vkDestroyImageView(device_, ssdm_disp_pyramid_view_, nullptr);
        ssdm_disp_pyramid_view_ = VK_NULL_HANDLE;
    }
    if (ssdm_disp_image_) {
        vmaDestroyImage(allocator_, ssdm_disp_image_, ssdm_disp_alloc_);
        ssdm_disp_image_ = VK_NULL_HANDLE;
        ssdm_disp_alloc_ = nullptr;
    }
    if (ssdm_pyrb_pyramid_view_) {
        vkDestroyImageView(device_, ssdm_pyrb_pyramid_view_, nullptr);
        ssdm_pyrb_pyramid_view_ = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < kSsdmMipsMax; ++i) {
        if (ssdm_pyrb_mip_views_[i]) {
            vkDestroyImageView(device_, ssdm_pyrb_mip_views_[i], nullptr);
            ssdm_pyrb_mip_views_[i] = VK_NULL_HANDLE;
        }
    }
    if (ssdm_pyrb_image_) {
        vmaDestroyImage(allocator_, ssdm_pyrb_image_, ssdm_pyrb_alloc_);
        ssdm_pyrb_image_ = VK_NULL_HANDLE;
        ssdm_pyrb_alloc_ = nullptr;
    }
    if (ssdm_compose_pipeline_) {
        vkDestroyPipeline(device_, ssdm_compose_pipeline_, nullptr);
        ssdm_compose_pipeline_ = VK_NULL_HANDLE;
    }
    if (ssdm_compose_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, ssdm_compose_pipeline_layout_, nullptr);
        ssdm_compose_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (ssdm_compose_desc_pool_) {
        vkDestroyDescriptorPool(device_, ssdm_compose_desc_pool_, nullptr);
        ssdm_compose_desc_pool_ = VK_NULL_HANDLE;
    }
    if (ssdm_compose_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, ssdm_compose_set_layout_, nullptr);
        ssdm_compose_set_layout_ = VK_NULL_HANDLE;
    }
    if (ssdm_compose_frag_module_) {
        vkDestroyShaderModule(device_, ssdm_compose_frag_module_, nullptr);
        ssdm_compose_frag_module_ = VK_NULL_HANDLE;
    }
    if (ssdm_remap_view_) {
        vkDestroyImageView(device_, ssdm_remap_view_, nullptr);
        ssdm_remap_view_ = VK_NULL_HANDLE;
    }
    if (ssdm_remap_image_) {
        vmaDestroyImage(allocator_, ssdm_remap_image_, ssdm_remap_alloc_);
        ssdm_remap_image_ = VK_NULL_HANDLE;
        ssdm_remap_alloc_ = nullptr;
    }
    if (shell_pipeline_) {
        vkDestroyPipeline(device_, shell_pipeline_, nullptr);
        shell_pipeline_ = VK_NULL_HANDLE;
    }
    if (shell_pipeline_layout_) {
        vkDestroyPipelineLayout(device_, shell_pipeline_layout_, nullptr);
        shell_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (shell_depth_desc_pool_) {
        vkDestroyDescriptorPool(device_, shell_depth_desc_pool_, nullptr);
        shell_depth_desc_pool_ = VK_NULL_HANDLE;
    }
    if (shell_depth_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, shell_depth_set_layout_, nullptr);
        shell_depth_set_layout_ = VK_NULL_HANDLE;
    }
    if (shell_vert_module_) {
        vkDestroyShaderModule(device_, shell_vert_module_, nullptr);
        shell_vert_module_ = VK_NULL_HANDLE;
    }
    if (shell_frag_module_) {
        vkDestroyShaderModule(device_, shell_frag_module_, nullptr);
        shell_frag_module_ = VK_NULL_HANDLE;
    }
}

// Shell silhouette pass — runs as its OWN render pass after the post-world
// depth transition (depth in SHADER_READ_ONLY, scene_color still in
// COLOR_ATTACHMENT). Color attachments: scene_color + motion_vec with
// LOAD/STORE so the main pass output is preserved underneath. No depth
// attachment — depth is sampled via shell_depth_desc_set_ and the
// fragment shader discards pixels where the main pass already drew a
// wall (vs pixels where sky is behind, where shell can extend bricks).
void VulkanEngine::render_shell_pass(VkCommandBuffer cmd) {
    if (!shell_pipeline_) return;

    auto color_att = vkinit::color_attachment_info(scene_color_view_,
                                                   nullptr,
                                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    auto motion_att = vkinit::color_attachment_info(motion_vec_view_,
                                                    nullptr,
                                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    color_att.loadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    motion_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    motion_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo color_atts[2] = { color_att, motion_att };

    VkRenderingInfo info{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .flags                = 0,
        .renderArea           = { {0, 0}, render_extent_ },
        .layerCount           = 1,
        .viewMask             = 0,
        .colorAttachmentCount = 2,
        .pColorAttachments    = color_atts,
        .pDepthAttachment     = nullptr,
        .pStencilAttachment   = nullptr,
    };
    vkCmdBeginRendering(cmd, &info);

    VkViewport vp_state{ 0.0f, 0.0f,
                         static_cast<float>(render_extent_.width),
                         static_cast<float>(render_extent_.height),
                         0.0f, 1.0f };
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shell_pipeline_);
    VkDescriptorSet sets[2] = { scene_desc_set_, shell_depth_desc_set_ };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            shell_pipeline_layout_, 0, 2, sets, 0, nullptr);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const glm::mat4& vp      = last_view_proj_;
    const Frustum&   frustum = current_frame_view_.frustum;
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        if (!brush_is_spom_wall(b.tex_albedo)) continue;
        const auto& a = world_.render_aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) continue;
        const glm::mat4& model = static_brush_models_[i];

        PushConstants pc{};
        pc.mvp        = vp * model;
        pc.model      = model;
        pc.prev_mvp   = pc.mvp;
        pc.color      = b.color;
        pc.emissive   = glm::vec4(0.0f);
        pc.tex_params = glm::vec4(static_cast<float>(b.tex_albedo),
                                  static_cast<float>(b.tex_normal),
                                  b.uv_scale,
                                  /*object_space=*/0.0f);
        vkCmdPushConstants(cmd, shell_pipeline_layout_, kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);
}

// Phase 4: SSDM compose remap. Reads scene_color (SHADER_READ_ONLY,
// transitioned by the post-world batch in vk_engine.cpp) + pyrB level 0,
// writes the remapped color into ssdm_remap_image_, then blits back
// into scene_color so downstream consumers (TAA, FSR, SVGF, compose)
// pick up the displaced result transparently.
void VulkanEngine::compose_ssdm(VkCommandBuffer cmd) {
    if (!ssdm_compose_pipeline_ || !ssdm_remap_image_) return;

    // Render into ssdm_remap (already in COLOR_ATTACHMENT layout: first
    // frame from init_one_time_submit, subsequent frames from the end-of-
    // function blit-source-to-COLOR_ATTACHMENT cycle below).
    VkClearValue clear{};
    clear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkRenderingAttachmentInfo att{
        .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext              = nullptr,
        .imageView          = ssdm_remap_view_,
        .imageLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode        = VK_RESOLVE_MODE_NONE,
        .resolveImageView   = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp             = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue         = clear,
    };
    VkRenderingInfo info{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .flags                = 0,
        .renderArea           = { {0, 0}, render_extent_ },
        .layerCount           = 1,
        .viewMask             = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &att,
        .pDepthAttachment     = nullptr,
        .pStencilAttachment   = nullptr,
    };
    vkCmdBeginRendering(cmd, &info);

    VkViewport vp{ 0.0f, 0.0f,
                   static_cast<float>(render_extent_.width),
                   static_cast<float>(render_extent_.height),
                   0.0f, 1.0f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      ssdm_compose_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            ssdm_compose_pipeline_layout_,
                            0, 1, &ssdm_compose_desc_set_, 0, nullptr);

    uint32_t pc[4] = { render_extent_.width, render_extent_.height, 0, 0 };
    vkCmdPushConstants(cmd, ssdm_compose_pipeline_layout_,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // Now blit ssdm_remap -> scene_color so downstream sees the result.
    // Transitions:
    //   ssdm_remap   : COLOR_ATTACHMENT -> TRANSFER_SRC
    //   scene_color  : SHADER_READ      -> TRANSFER_DST
    {
        VkImageMemoryBarrier b[2]{};
        b[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b[0].oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b[0].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].image            = ssdm_remap_image_;
        b[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b[1] = b[0];
        b[1].srcAccessMask    = VK_ACCESS_SHADER_READ_BIT;
        b[1].dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[1].oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b[1].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[1].image            = scene_color_image_;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, b);
    }

    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = { 0, 0, 0 };
    blit.srcOffsets[1] = { static_cast<int32_t>(render_extent_.width),
                            static_cast<int32_t>(render_extent_.height), 1 };
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[0] = blit.srcOffsets[0];
    blit.dstOffsets[1] = blit.srcOffsets[1];
    vkCmdBlitImage(cmd,
                   ssdm_remap_image_,  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   scene_color_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_NEAREST);

    // Restore layouts:
    //   ssdm_remap   : TRANSFER_SRC  -> COLOR_ATTACHMENT (next frame's render)
    //   scene_color  : TRANSFER_DST  -> SHADER_READ      (downstream consumers)
    {
        VkImageMemoryBarrier b[2]{};
        b[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b[0].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b[0].newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].image            = ssdm_remap_image_;
        b[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b[1] = b[0];
        b[1].srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[1].dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
        b[1].oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[1].newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b[1].image            = scene_color_image_;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, b);
    }
}

// Phase 3: iterative barycenter refinement. Reads from pyramid A (already
// in SHADER_READ_ONLY after Phase 2's blit chain), writes to pyramid B
// per-mip via a fullscreen-triangle render pass. Loops top→0; at the top
// level pyrB_prev is unused (gated by push constant). Between iterations
// a barrier flips the just-written mip to SHADER_READ so the next
// iteration can sample it via the all-mip pyrB sampler view.
void VulkanEngine::refine_ssdm_pyramid_b(VkCommandBuffer cmd) {
    if (!ssdm_refine_pipeline_ || !ssdm_pyrb_image_) return;

    struct PC {
        uint32_t base_extent[2];
        uint32_t level_extent[2];
        uint32_t level;
        uint32_t is_top_level;
        uint32_t _pad0;
        uint32_t _pad1;
    };
    static_assert(sizeof(PC) == 8 * sizeof(uint32_t), "ssdm refine pc size");

    // Iterate from coarsest (kSsdmMipsMax - 1) down to 0.
    for (int32_t lvl = static_cast<int32_t>(kSsdmMipsMax) - 1; lvl >= 0; --lvl) {
        uint32_t mip_w = std::max(1u, render_extent_.width  >> lvl);
        uint32_t mip_h = std::max(1u, render_extent_.height >> lvl);

        // Transition this mip of pyrB from SHADER_READ to COLOR_ATTACHMENT.
        // Other mips stay in SHADER_READ for the sampler.
        {
            VkImageMemoryBarrier b{};
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image            = ssdm_pyrb_image_;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                   static_cast<uint32_t>(lvl), 1, 0, 1 };
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        }

        VkClearValue clear{};
        clear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkRenderingAttachmentInfo att{
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = ssdm_pyrb_mip_views_[lvl],
            .imageLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode        = VK_RESOLVE_MODE_NONE,
            .resolveImageView   = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp             = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue         = clear,
        };
        VkRenderingInfo info{
            .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext                = nullptr,
            .flags                = 0,
            .renderArea           = { {0, 0}, { mip_w, mip_h } },
            .layerCount           = 1,
            .viewMask             = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &att,
            .pDepthAttachment     = nullptr,
            .pStencilAttachment   = nullptr,
        };
        vkCmdBeginRendering(cmd, &info);

        VkViewport vp{ 0.0f, 0.0f,
                       static_cast<float>(mip_w),
                       static_cast<float>(mip_h),
                       0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, { mip_w, mip_h } };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ssdm_refine_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                ssdm_refine_pipeline_layout_,
                                0, 1, &ssdm_refine_desc_sets_[0],
                                0, nullptr);

        PC pc{};
        pc.base_extent[0]  = render_extent_.width;
        pc.base_extent[1]  = render_extent_.height;
        pc.level_extent[0] = mip_w;
        pc.level_extent[1] = mip_h;
        pc.level           = static_cast<uint32_t>(lvl);
        pc.is_top_level    = (lvl == static_cast<int32_t>(kSsdmMipsMax) - 1)
                                ? 1u : 0u;
        vkCmdPushConstants(cmd, ssdm_refine_pipeline_layout_,
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        // Flip this mip back to SHADER_READ so the NEXT iteration (the
        // finer level below) can sample it as pyrB_prev.
        {
            VkImageMemoryBarrier b{};
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image            = ssdm_pyrb_image_;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                   static_cast<uint32_t>(lvl), 1, 0, 1 };
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        }
    }
}

// Phase 2: build mip pyramid A. After Phase 1's render pass writes level 0
// of ssdm_disp_image_, blit each mip into the next via LINEAR filter --
// vkCmdBlitImage with LINEAR averages 2x2 source texels into 1 destination
// texel, matching the paper's "average 4 texels into 1 upper-level texel"
// exactly. No compute shader needed.
//
// Layout state on entry: level 0 in COLOR_ATTACHMENT_OPTIMAL (just written),
// levels 1..N-1 in SHADER_READ_ONLY (idle state from prev frame or initial
// transition). On exit: all levels in SHADER_READ_ONLY for Phase 3 reads.
void VulkanEngine::build_ssdm_pyramid_a(VkCommandBuffer cmd) {
    // Level 0: COLOR_ATTACHMENT -> TRANSFER_SRC (blit source for mip 0->1).
    // Levels 1..N-1: SHADER_READ -> TRANSFER_DST (blit destination).
    {
        VkImageMemoryBarrier b[2]{};
        b[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b[0].oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b[0].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].image            = ssdm_disp_image_;
        b[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b[1] = b[0];
        b[1].srcAccessMask    = VK_ACCESS_SHADER_READ_BIT;
        b[1].dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[1].oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b[1].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                  1, kSsdmMipsMax - 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, b);
    }

    int32_t w = static_cast<int32_t>(render_extent_.width);
    int32_t h = static_cast<int32_t>(render_extent_.height);
    for (uint32_t lvl = 1; lvl < kSsdmMipsMax; ++lvl) {
        int32_t next_w = w > 1 ? w / 2 : 1;
        int32_t next_h = h > 1 ? h / 2 : 1;
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel   = lvl - 1;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { w, h, 1 };
        blit.dstSubresource = blit.srcSubresource;
        blit.dstSubresource.mipLevel = lvl;
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { next_w, next_h, 1 };
        vkCmdBlitImage(cmd,
                       ssdm_disp_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       ssdm_disp_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);
        // The destination level we just wrote becomes the next iteration's
        // source. Flip its layout TRANSFER_DST -> TRANSFER_SRC. Skip on the
        // final level (no further blit consumes it).
        if (lvl + 1 < kSsdmMipsMax) {
            VkImageMemoryBarrier b{};
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image            = ssdm_disp_image_;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, lvl, 1, 0, 1 };
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        }
        w = next_w;
        h = next_h;
    }

    // Final transition: all mips -> SHADER_READ_ONLY for Phase 3+ readers.
    // Levels 0..N-2 are in TRANSFER_SRC, the last level is in TRANSFER_DST.
    // Use one barrier covering everything (allowed since both source layouts
    // converge to the same destination layout).
    {
        VkImageMemoryBarrier b[2]{};
        b[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b[0].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b[0].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].image            = ssdm_disp_image_;
        b[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                  0, kSsdmMipsMax - 1, 0, 1 };
        b[1] = b[0];
        b[1].srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[1].oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,
                                  kSsdmMipsMax - 1, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, b);
    }
}

void VulkanEngine::render_world_ssdm_disp(VkCommandBuffer cmd) {
    if (!ssdm_disp_pipeline_ || !ssdm_disp_view_) return;

    // SHADER_READ_ONLY -> COLOR_ATTACHMENT. Phase 1 always overwrites the
    // full image (loadOp = CLEAR) so the previous layout's contents are
    // unimportant; only the image semantic state matters.
    vkinit::transition_image(cmd, ssdm_disp_image_,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkClearValue clear_disp{};
    clear_disp.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    auto disp_att = vkinit::color_attachment_info(
        ssdm_disp_view_, &clear_disp, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Depth-test against the main pass's depth buffer with LOAD/STORE.
    // Depth was just written by the main pass; we read it but don't modify
    // it, matching the cube color pass's depth_compare = LESS_OR_EQUAL so
    // our SSDM fragments only paint where the wall actually rasterised.
    VkRenderingAttachmentInfo depth_att{
        .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext              = nullptr,
        .imageView          = depth_view_,
        .imageLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode        = VK_RESOLVE_MODE_NONE,
        .resolveImageView   = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp             = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue         = {},
    };

    VkRenderingInfo rendering{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .flags                = 0,
        .renderArea           = { {0, 0}, render_extent_ },
        .layerCount           = 1,
        .viewMask             = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &disp_att,
        .pDepthAttachment     = &depth_att,
        .pStencilAttachment   = nullptr,
    };
    vkCmdBeginRendering(cmd, &rendering);

    VkViewport vp_state{};
    vp_state.x        = 0.0f;
    vp_state.y        = 0.0f;
    vp_state.width    = static_cast<float>(render_extent_.width);
    vp_state.height   = static_cast<float>(render_extent_.height);
    vp_state.minDepth = 0.0f;
    vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);

    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_disp_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &scene_desc_set_,
                            0, nullptr);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const glm::mat4& vp = last_view_proj_;
    const Frustum&   frustum = current_frame_view_.frustum;
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        if (!brush_is_spom_wall(b.tex_albedo)) continue;
        const auto& a = world_.render_aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) continue;
        const glm::mat4& model = static_brush_models_[i];

        PushConstants pc{};
        pc.mvp        = vp * model;
        pc.model      = model;
        pc.prev_mvp   = pc.mvp;  // SSDM disp pass doesn't write motion vec
        pc.color      = b.color;
        pc.emissive   = glm::vec4(0.0f);
        pc.tex_params = glm::vec4(static_cast<float>(b.tex_albedo),
                                  static_cast<float>(b.tex_normal),
                                  b.uv_scale,
                                  /*object_space=*/0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    // Phase 2 immediately consumes level 0 -- it expects level 0 in
    // COLOR_ATTACHMENT layout (just rendered to) and handles the
    // transition chain to TRANSFER_SRC/DST and finally SHADER_READ
    // for all mips. Don't transition here.
    build_ssdm_pyramid_a(cmd);
    // Phase 3: iterative barycenter refinement (pyramid A → pyramid B).
    // Leaves all pyrB mips in SHADER_READ for Phase 4 / future sampling.
    refine_ssdm_pyramid_b(cmd);
}
