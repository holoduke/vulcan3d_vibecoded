#include "engine/vk_pipelines.h"

#include "engine/log.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace qlike::vkpipe {

// Set by the engine via set_pipeline_cache() once it's loaded the disk blob.
// Stays VK_NULL_HANDLE until then; vkCreateGraphicsPipelines accepts null and
// just doesn't cache, which is the correct fallback if the cache file is
// missing or corrupt.
static VkPipelineCache g_pipeline_cache = VK_NULL_HANDLE;

VkShaderModule load_shader_module(VkDevice device, const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("can't open shader: " + path);
    std::streamsize sz = f.tellg();
    f.seekg(0);
    std::vector<char> bytes(static_cast<size_t>(sz));
    f.read(bytes.data(), sz);

    VkShaderModuleCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .codeSize = static_cast<size_t>(sz),
        .pCode = reinterpret_cast<const uint32_t*>(bytes.data()),
    };
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateShaderModule failed for " + path);
    }
    log::infof("loaded shader: %s (%lld bytes)", path.c_str(), static_cast<long long>(sz));
    return mod;
}

VkPipeline build_graphics_pipeline(VkDevice device, const GraphicsPipelineConfig& cfg) {
    VkPipelineShaderStageCreateInfo stages[4]{};
    uint32_t stage_count = 0;
    stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[stage_count].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[stage_count].module = cfg.vert;
    stages[stage_count].pName = "main";
    ++stage_count;
    const bool use_tess = cfg.tesc && cfg.tese && cfg.patch_control_points > 0;
    if (use_tess) {
        stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        stages[stage_count].module = cfg.tesc;
        stages[stage_count].pName = "main";
        ++stage_count;
        stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        stages[stage_count].module = cfg.tese;
        stages[stage_count].pName = "main";
        ++stage_count;
    }
    // Fragment stage is optional — depth-only shadow passes drop it so
    // the rasteriser writes only depth and no color.
    if (cfg.frag) {
        stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[stage_count].module = cfg.frag;
        stages[stage_count].pName = "main";
        ++stage_count;
    }

    VkPipelineVertexInputStateCreateInfo vi{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .vertexBindingDescriptionCount = static_cast<uint32_t>(cfg.vbindings.size()),
        .pVertexBindingDescriptions = cfg.vbindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(cfg.vattrs.size()),
        .pVertexAttributeDescriptions = cfg.vattrs.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo ia{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .topology = use_tess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
                             : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineTessellationStateCreateInfo tess{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .patchControlPoints = cfg.patch_control_points,
    };

    VkPipelineViewportStateCreateInfo vp{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .viewportCount = 1, .pViewports = nullptr,
        .scissorCount = 1, .pScissors = nullptr,
    };

    VkPipelineRasterizationStateCreateInfo rs{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = cfg.polygon_mode,
        .cullMode = cfg.cull,
        .frontFace = cfg.front_face,
        .depthBiasEnable = cfg.depth_bias_enable ? VK_TRUE : VK_FALSE,
        .depthBiasConstantFactor = 0, .depthBiasClamp = 0, .depthBiasSlopeFactor = 0,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo ms{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f, .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE, .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineDepthStencilStateCreateInfo ds{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .depthTestEnable = cfg.depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = cfg.depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp = cfg.depth_compare,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {}, .back = {},
        .minDepthBounds = 0.0f, .maxDepthBounds = 1.0f,
    };

    // Resolve color attachment count: prefer the multi-format `color_formats`
    // vector when populated (used by cube.frag's color + motion-vec output);
    // fall back to the single-format `color_attachment_count` shorthand.
    const uint32_t resolved_color_count =
        cfg.color_formats.empty()
            ? static_cast<uint32_t>(cfg.color_attachment_count)
            : static_cast<uint32_t>(cfg.color_formats.size());

    // One blend state per color attachment. additive_blend applies uniformly
    // to all attachments. alpha_blend_color0_only enables SRC_ALPHA blending
    // on attachment 0 only (used by the grass raymarch pipeline so blade
    // edges blend with terrain while motion-vec attachment 1 stays opaque).
    std::vector<VkPipelineColorBlendAttachmentState> blends(resolved_color_count);
    for (uint32_t ai = 0; ai < resolved_color_count; ++ai) {
        VkPipelineColorBlendAttachmentState& blend = blends[ai];
        bool alpha_blend = cfg.alpha_blend_color0_only && ai == 0;
        blend.blendEnable = (cfg.additive_blend || alpha_blend) ? VK_TRUE : VK_FALSE;
        if (cfg.additive_blend) {
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        } else if (alpha_blend) {
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        }
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }

    VkPipelineColorBlendStateCreateInfo cb{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .logicOpEnable = VK_FALSE, .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = resolved_color_count,
        .pAttachments = resolved_color_count > 0 ? blends.data() : nullptr,
        .blendConstants = {0,0,0,0},
    };

    VkDynamicState dyn_states[3] = { VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dyn{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .dynamicStateCount = cfg.depth_bias_enable ? 3u : 2u,
        .pDynamicStates = dyn_states,
    };

    const VkFormat* color_format_ptr = nullptr;
    if (!cfg.color_formats.empty()) {
        color_format_ptr = cfg.color_formats.data();
    } else if (cfg.color_attachment_count > 0) {
        color_format_ptr = &cfg.color_format;
    }
    VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = resolved_color_count,
        .pColorAttachmentFormats = color_format_ptr,
        .depthAttachmentFormat = cfg.depth_format,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    // Optional attachment-based VRS state. Chained AFTER rendering so the
    // pNext walk hits it; combiner ops force the per-fragment rate to come
    // from the bound attachment image (REPLACE on op[1] discards the
    // pipeline/primitive rates from op[0]'s KEEP). fragmentSize is the
    // pipeline-rate slot we set to 1×1; combinerOps[0] = KEEP just passes
    // that through. combinerOps[1] = REPLACE overrides with attachment.
    VkPipelineFragmentShadingRateStateCreateInfoKHR vrs_state{};
    if (cfg.enable_vrs) {
        vrs_state.sType = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR;
        vrs_state.fragmentSize = { 1, 1 };
        vrs_state.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
        vrs_state.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;
        vrs_state.pNext = rendering.pNext;
        rendering.pNext = &vrs_state;
    }

    VkGraphicsPipelineCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering, .flags = 0,
        .stageCount = stage_count, .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pTessellationState = use_tess ? &tess : nullptr,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = &ds,
        .pColorBlendState = &cb,
        .pDynamicState = &dyn,
        .layout = cfg.layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, g_pipeline_cache, 1, &pci, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateGraphicsPipelines failed");
    }
    return pipeline;
}

void set_pipeline_cache(VkPipelineCache cache) {
    g_pipeline_cache = cache;
}

} // namespace qlike::vkpipe
