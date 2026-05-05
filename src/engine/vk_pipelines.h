#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace qlike::vkpipe {

VkShaderModule load_shader_module(VkDevice device, const std::string& spv_path);

struct GraphicsPipelineConfig {
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    // Single-attachment shorthand — used by every pipeline that has exactly
    // one color attachment. For multi-attachment output (e.g. cube.frag
    // writing scene_color at location 0 and motion vector at location 1),
    // populate `color_formats` instead; if non-empty it overrides this.
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    std::vector<VkFormat> color_formats;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    std::vector<VkVertexInputBindingDescription> vbindings;
    std::vector<VkVertexInputAttributeDescription> vattrs;
    VkCullModeFlags cull = VK_CULL_MODE_BACK_BIT;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_test = true;
    bool depth_write = true;
    // LESS_OR_EQUAL is the safe default — same behaviour as LESS for geometry
    // drawn once, but lets a depth pre-pass + color pass coexist (the color
    // pass's fragment must pass against the *equal* depth value written
    // earlier).
    VkCompareOp depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    int color_attachment_count = 1;

    // Per-attachment additive blend — used by the bloom upsample pass to
    // accumulate into the larger mip. Applied to ALL color attachments when
    // set; if you need mixed blend modes (one additive, others not) this
    // field needs to grow into a per-attachment vector — no caller needs
    // that today.
    bool additive_blend = false;
};

VkPipeline build_graphics_pipeline(VkDevice device, const GraphicsPipelineConfig& cfg);

// Optional persistent pipeline cache. When set via set_pipeline_cache(), every
// build_graphics_pipeline() call after that hands the cache to Vulkan, so
// equivalent pipelines compile in milliseconds instead of seconds. The engine
// is expected to load the cache from disk at startup, set it here, and write
// the data back at shutdown.
void set_pipeline_cache(VkPipelineCache cache);

} // namespace qlike::vkpipe
