// GPU-driven vegetation + rocks ("open world" scale).
//
//   * Archetypes (pine / broadleaf / bush / rock variants) each carry a
//     4-level LOD chain, packed into ONE vertex + index buffer.
//   * Every instance on the whole 2 km map lives in a GPU buffer. Each
//     frame veg_cull.comp frustum-culls all of them, picks a LOD from the
//     projected size (dithered per instance), drops sub-pixel ones and
//     compacts survivors into per-(archetype, LOD) buckets, bumping the
//     matching VkDrawIndexedIndirectCommand. One vkCmdDrawIndexedIndirect
//     then draws everything - depth prepass and colour pass.
//   * Ray-traced shadows / GI: the instances nearest the camera are added
//     to the TLAS as instances of a per-archetype BLAS (LOD1 geometry).
//
// Cost scales with what's on screen, not with how much is placed: tens of
// thousands of trees cost a few hundred K triangles because everything
// past ~60 m is 20-100 triangles and anything below a pixel is skipped.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/physics_world.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef QLIKE_SHADER_DIR
#define QLIKE_SHADER_DIR "shaders"
#endif

namespace qlike {

namespace {
VkBuffer make_buffer(VmaAllocator a, VkDeviceSize size, VkBufferUsageFlags usage,
                     bool host, VmaAllocation& alloc, void** mapped) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = std::max<VkDeviceSize>(size, 16);
    bci.usage = usage;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (host) aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocationInfo ai{};
    vk_check(vmaCreateBuffer(a, &bci, &aci, &b, &alloc, &ai), "veg buffer");
    if (mapped) *mapped = ai.pMappedData;
    return b;
}
}  // namespace

void VulkanEngine::init_vegetation_data(const Heightmap& hm) {
    WorldPropParams wp{};
    wp.water_level  = rt_.water_level;
    wp.tree_density = glm::clamp(rt_.forest_density, 0.0f, 2.0f);
    wp.bush_density = glm::clamp(rt_.bush_density, 0.0f, 2.0f);
    wp.max_trees    = int(float(wp.max_trees) * std::max(1.0f, wp.tree_density));
    wp.max_bushes   = int(float(wp.max_bushes) * std::max(1.0f, wp.bush_density));
    wp.rock_band_lo = rt_.terrain_h_dirt_rock_start;
    for (const auto& a : aa_) wp.keep_out.push_back(glm::vec3(a.pos.x, a.pos.z, 9.0f));
    for (const auto& o : outposts_) wp.keep_out.push_back(glm::vec3(o.x, o.z, 10.0f));
    for (const auto& l : lamp_lights_) wp.keep_out.push_back(glm::vec3(l.x, l.z, 2.5f));
    const auto tv0 = std::chrono::steady_clock::now();
    veg_arch_ = build_veg_archetypes(0xF0E57);
    const auto tv1 = std::chrono::steady_clock::now();
    std::vector<collision::AABB> cols;
    WorldPropStats st{};
    veg_inst_cpu_ = place_vegetation(hm, &castle_road_, wp, veg_arch_, cols, st);
    log::infof("[init] veg archetypes %.0f ms, placement %.0f ms",
               std::chrono::duration<double, std::milli>(tv1 - tv0).count(),
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv1).count());
    veg_col_begin_ = world_.aabbs.size();
    veg_col_count_ = cols.size();
    for (const auto& a : cols) world_.aabbs.push_back(a);

    // Merge every archetype's LODs into one VB/IB; build the indirect
    // template (one command per archetype x LOD).
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    veg_cmd_template_.clear();
    veg_lod_ranges_.clear();
    std::vector<uint32_t> per_arch(veg_arch_.size(), 0);
    for (const auto& vi : veg_inst_cpu_) per_arch[size_t(vi.rot_arch_phase.y + 0.5f)]++;
    uint32_t vis_base = 0;
    size_t tris_lod[kVegLods]{};
    veg_has_imp_.assign(veg_arch_.size(), 0);
    for (size_t a = 0; a < veg_arch_.size(); ++a) {
        // LOD 4 = impostor card for trees (procedural silhouette in
        // cube.frag): one quad, local x = u * half width, y = v * height;
        // the normal slot carries (u, v, kind) and uv.y = 5 (material).
        game::LevelMesh imp;
        const int sp = veg_arch_[a].species;
        if (sp <= 1) {
            float rmax = 0.0f, ymax = 0.0f;
            for (const auto& v : veg_arch_[a].lods[0].verts) {
                rmax = std::max(rmax, std::max(std::abs(v.p.x), std::abs(v.p.z)));
                ymax = std::max(ymax, v.p.y);
            }
            const float kind = (sp == 0) ? 1.0f : 2.0f;
            const float us[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, vs[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
            for (int k = 0; k < 4; ++k)
                imp.verts.push_back({ glm::vec3(us[k] * rmax, vs[k] * ymax, 0.0f),
                                      glm::vec3(us[k], vs[k], kind), 0.0f, 5.0f });
            imp.idx = { 0, 1, 2, 0, 2, 3 };
            veg_has_imp_[a] = 1;
        }
        for (int l = 0; l < kVegDrawLods; ++l) {
            const auto& m = (l < kVegLods) ? veg_arch_[a].lods[l] : imp;
            VkDrawIndexedIndirectCommand c{};
            c.indexCount = uint32_t(m.idx.size());
            c.instanceCount = 0;
            c.firstIndex = uint32_t(idx.size());
            c.vertexOffset = int32_t(verts.size());
            c.firstInstance = vis_base;
            vis_base += per_arch[a];
            veg_cmd_template_.push_back(c);
            veg_lod_ranges_.push_back({ uint32_t(verts.size()), uint32_t(m.verts.size()),
                                        uint32_t(idx.size()), uint32_t(m.idx.size()),
                                        m.card_index_start, m.card_cell });
            for (const auto& v : m.verts) verts.push_back({ v.p, v.n, glm::vec2(v.u, v.v) });
            idx.insert(idx.end(), m.idx.begin(), m.idx.end());
            if (l < kVegLods) tris_lod[l] += m.idx.size() / 3;
        }
    }
    veg_mesh_ = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                      verts.data(), uint32_t(verts.size()),
                                      idx.data(), uint32_t(idx.size()));
    veg_count_ = uint32_t(veg_inst_cpu_.size());
    veg_vis_capacity_ = std::max(vis_base, 1u);

    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    void* mp = nullptr;
    veg_inst_buf_ = make_buffer(allocator_, sizeof(VegInstance) * std::max<size_t>(veg_count_, 1), ssbo,
                                true, veg_inst_alloc_, &mp);
    if (mp && veg_count_) std::memcpy(mp, veg_inst_cpu_.data(), sizeof(VegInstance) * veg_count_);
    veg_inst_mapped_ = mp;   // blast_vegetation() zeroes destroyed instances here
    vmaFlushAllocation(allocator_, veg_inst_alloc_, 0, VK_WHOLE_SIZE);
    veg_vis_buf_ = make_buffer(allocator_, sizeof(uint32_t) * veg_vis_capacity_, ssbo, false,
                               veg_vis_alloc_, nullptr);
    veg_fade_buf_ = make_buffer(allocator_, sizeof(float) * veg_vis_capacity_, ssbo, false,
                                veg_fade_alloc_, nullptr);
    create_veg_hiz();
    veg_ind_buf_ = make_buffer(allocator_, sizeof(VkDrawIndexedIndirectCommand) * veg_cmd_template_.size(),
                               ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               false, veg_ind_alloc_, nullptr);
    std::vector<glm::vec4> ainfo;
    int rock_variant = 0;
    for (size_t a = 0; a < veg_arch_.size(); ++a) {
        const bool rock = veg_arch_[a].species == 3;
        // Rocks: .z = -(variant + 1) (the ray-marched surface derives its
        // shape parameters from it); radius covers the inflated hull.
        ainfo.push_back(glm::vec4(veg_arch_[a].radius * (rock ? 1.45f : 1.0f), veg_arch_[a].height * 0.5f,
                                  rock ? -float(++rock_variant) : float(veg_has_imp_[a]),
                                  float(veg_arch_[a].species)));
    }
    veg_arch_buf_ = make_buffer(allocator_, sizeof(glm::vec4) * ainfo.size(), ssbo, true, veg_arch_alloc_, &mp);
    if (mp) std::memcpy(mp, ainfo.data(), sizeof(glm::vec4) * ainfo.size());
    vmaFlushAllocation(allocator_, veg_arch_alloc_, 0, VK_WHOLE_SIZE);

    // Descriptor set (set 1): instances, visible list, indirect, arch info.
    VkDescriptorSetLayoutBinding b[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        b[i].binding = i;
        b[i].descriptorType = (i == 4) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 6; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &veg_set_layout_), "veg set layout");
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
                                   { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &veg_pool_), "veg pool");
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = veg_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &veg_set_layout_;
    vk_check(vkAllocateDescriptorSets(device_, &dai, &veg_set_), "veg set");
    VkDescriptorBufferInfo bi[5] = {
        { veg_inst_buf_, 0, VK_WHOLE_SIZE }, { veg_vis_buf_, 0, VK_WHOLE_SIZE },
        { veg_ind_buf_, 0, VK_WHOLE_SIZE },  { veg_arch_buf_, 0, VK_WHOLE_SIZE },
        { veg_fade_buf_, 0, VK_WHOLE_SIZE } };
    const uint32_t bmap[5] = { 0, 1, 2, 3, 5 };
    VkWriteDescriptorSet w[4 + 1]{};
    for (uint32_t i = 0; i < 5; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = veg_set_; w[i].dstBinding = bmap[i]; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 5, w, 0, nullptr);
    write_veg_hiz_binding();

    log::infof("[veg] %u instances (%d pines, %d broadleaf, %d bushes, %d rock formations, "
               "%d boulders), %zu archetypes; LOD tris per archetype avg %zu/%zu/%zu/%zu",
               veg_count_, st.pines, st.broadleaf, st.bushes, st.formations, st.boulders,
               veg_arch_.size(), tris_lod[0] / veg_arch_.size(), tris_lod[1] / veg_arch_.size(),
               tris_lod[2] / veg_arch_.size(), tris_lod[3] / veg_arch_.size());
}

void VulkanEngine::init_vegetation_pipelines() {
    if (!veg_set_layout_) return;
    const std::string sd = QLIKE_SHADER_DIR;
    veg_vert_module_ = vkpipe::load_shader_module(device_, sd + "/veg.vert.spv");
    veg_cull_module_ = vkpipe::load_shader_module(device_, sd + "/veg_cull.comp.spv");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    pc.size = sizeof(PushConstants);
    VkDescriptorSetLayout sets[2] = { scene_desc_set_layout_, veg_set_layout_ };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 2; plci.pSetLayouts = sets;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &veg_layout_), "veg layout");

    VkPushConstantRange cpc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 208 };
    VkPipelineLayoutCreateInfo cplci = plci;
    cplci.pPushConstantRanges = &cpc;
    vk_check(vkCreatePipelineLayout(device_, &cplci, nullptr, &veg_cull_layout_), "veg cull layout");
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = veg_cull_module_;
    cci.stage.pName = "main";
    cci.layout = veg_cull_layout_;
    vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &veg_cull_pipeline_),
             "veg cull pipeline");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = veg_vert_module_;
    cfg.frag = frag_module_;
    cfg.layout = veg_layout_;
    cfg.color_formats = { scene_color_format_, motion_vec_format_, gbuffer0_format_, gbuffer1_format_ };
    cfg.depth_format = depth_format_;
    VkVertexInputBindingDescription vb{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    cfg.vbindings = { vb };
    VkVertexInputAttributeDescription a0{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, position)) };
    VkVertexInputAttributeDescription a1{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, normal)) };
    VkVertexInputAttributeDescription a2{ 2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(Vertex, uv)) };
    cfg.vattrs = { a0, a1, a2 };
    cfg.cull = VK_CULL_MODE_NONE;       // leaf cards are seen from both sides
    veg_color_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    if (frag_gbuf_module_) {
        auto g = cfg; g.frag = frag_gbuf_module_;
        veg_gbuf_pipeline_ = vkpipe::build_graphics_pipeline(device_, g);
    }
    auto d = cfg;
    veg_depth_frag_module_ = vkpipe::load_shader_module(device_, sd + "/veg_depth.frag.spv");
    d.frag = veg_depth_frag_module_;
    d.color_formats.clear();
    d.color_attachment_count = 0;
    d.depth_compare = VK_COMPARE_OP_LESS;
    veg_depth_pipeline_ = vkpipe::build_graphics_pipeline(device_, d);
    log::info("[veg] cull + draw pipelines built");
}

void VulkanEngine::cull_vegetation(VkCommandBuffer cmd) {
    if (!veg_cull_pipeline_ || veg_count_ == 0) return;
    // Previous frame's indirect / vertex reads must be done before reset.
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
    vkCmdUpdateBuffer(cmd, veg_ind_buf_, 0,
                      sizeof(VkDrawIndexedIndirectCommand) * veg_cmd_template_.size(),
                      veg_cmd_template_.data());
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);

    const FrameView& fv = current_frame_view_;
    build_veg_hiz(cmd);
    struct { glm::vec4 planes[6]; glm::vec4 cam; glm::vec4 params; glm::mat4 prev_vp; glm::vec4 hiz; } pc{};
    for (int i = 0; i < 6; ++i) pc.planes[i] = fv.frustum.planes[i];
    // Detail is chosen for the OUTPUT resolution (the upscaler reconstructs
    // to it), not the internal render resolution.
    const float ppr = float(swapchain_extent_.height) / (2.0f * std::tan(glm::radians(80.0f) * 0.5f));
    pc.cam = glm::vec4(fv.eye_pos, ppr);
    pc.params = glm::vec4(float(veg_count_), 1450.0f, std::max(0.25f, rt_.veg_lod_bias),
                          (veg_hiz_ready_ && !std::getenv("QLIKE_NO_HIZ")) ? 1.0f : 0.0f);
    {
        // Camera cut (teleport, respawn, big snap turn): last frame's Hi-Z
        // no longer describes this view and would cull visible trees for a
        // frame - skip the occlusion test once.
        const glm::vec3 fwd = -glm::vec3(fv.inv_view[2]);
        const bool cut = glm::distance(fv.eye_pos, veg_prev_eye_) > 8.0f ||
                         glm::dot(fwd, veg_prev_fwd_) < 0.9f;
        if (cut) pc.params.w = 0.0f;
        veg_prev_eye_ = fv.eye_pos;
        veg_prev_fwd_ = fwd;
    }
    pc.prev_vp = veg_prev_vp_;
    pc.hiz = glm::vec4(float(veg_hiz_w_), float(veg_hiz_h_), float(veg_hiz_mips_), 0.0f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, veg_cull_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, veg_cull_layout_, 1, 1, &veg_set_, 0, nullptr);
    vkCmdPushConstants(cmd, veg_cull_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (veg_count_ + 127) / 128, 1, 1);
    veg_prev_vp_ = fv.vp;          // next frame's Hi-Z = this frame's prepass

    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void VulkanEngine::draw_vegetation(VkCommandBuffer cmd, bool depth_only) {
    if (!veg_color_pipeline_ || veg_count_ == 0) return;
    const FrameView& fv = current_frame_view_;
    VkPipeline p = depth_only ? veg_depth_pipeline_
                 : (deferred_owns_lighting() && veg_gbuf_pipeline_) ? veg_gbuf_pipeline_
                 : veg_color_pipeline_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    VkDescriptorSet sets[2] = { scene_desc_set_, veg_set_ };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, veg_layout_, 0, 2, sets, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &veg_mesh_.vertex_buffer, &off);
    vkCmdBindIndexBuffer(cmd, veg_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    PushConstants pc{};
    pc.mvp = fv.vp;
    pc.model = glm::mat4(1.0f);
    pc.prev_mvp = prev_view_proj_valid_ ? prev_view_proj_ : fv.vp;
    pc.color = glm::vec4(1.0f);
    pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
    pc.grass_params = glm::vec4(1.0f, 1.0f, 1.0f, -1.0f);   // veg draw, mesh mode, triplanar
    vkCmdPushConstants(cmd, veg_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    vkCmdDrawIndexedIndirect(cmd, veg_ind_buf_, 0, uint32_t(veg_cmd_template_.size()),
                             sizeof(VkDrawIndexedIndirectCommand));
    // Restore set 0 on the world layout for the callers' following draws.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1,
                            &scene_desc_set_, 0, nullptr);
}

void VulkanEngine::build_veg_blas() {
    if (veg_arch_.empty() || !veg_mesh_.vertex_buffer) return;
    const uint32_t shared_families[2] = { graphics_queue_family_, compute_queue_family_ };
    const VkDeviceAddress vbo = buffer_device_address(device_, veg_mesh_.vertex_buffer);
    const VkDeviceAddress ibo = buffer_device_address(device_, veg_mesh_.index_buffer);
    // One RT shape per mesh LOD, so the shape a shadow ray meets is the
    // one that is drawn (see collect_veg_rt_instances).
    static constexpr int kRtLod[kVegRtLevels] = { 0, 1, 2, 3 };
    veg_blas_.resize(veg_arch_.size() * kVegRtLevels);
    for (size_t q = 0; q < veg_blas_.size(); ++q) {
        const size_t a = q / kVegRtLevels;
        const auto& r = veg_lod_ranges_[a * kVegDrawLods + kRtLod[q % kVegRtLevels]];
        VkAccelerationStructureGeometryTrianglesDataKHR tri{};
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = vbo + VkDeviceAddress(r.first_vertex) * sizeof(Vertex);
        tri.vertexStride = sizeof(Vertex);
        tri.maxVertex = r.vertex_count - 1;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = ibo + VkDeviceAddress(r.first_index) * sizeof(uint32_t);
        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tri;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        // Leaf cards go in a second, NON-opaque geometry: shadow rays then
        // run the leaf cut-out test on them (see any_hit_m in the shaders)
        // instead of treating every card as a solid rectangle.
        VkAccelerationStructureGeometryKHR geoms[2] = { geom, geom };
        VkAccelerationStructureBuildRangeInfoKHR ranges[2]{};
        uint32_t prim_counts[2] = { r.index_count / 3, 0 };
        uint32_t ngeo = 1;
        if (r.card_start != 0xFFFFFFFFu && r.card_start > 0 && r.card_start < r.index_count) {
            prim_counts[0] = r.card_start / 3;
            prim_counts[1] = (r.index_count - r.card_start) / 3;
            geoms[1].flags = 0;
            ranges[1].primitiveOffset = r.card_start * uint32_t(sizeof(uint32_t));
            ngeo = 2;
        }
        ranges[0].primitiveCount = prim_counts[0];
        ranges[1].primitiveCount = prim_counts[1];
        bgi.geometryCount = ngeo; bgi.pGeometries = geoms;
        uint32_t prims = r.index_count / 3;
        (void)prims;

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        g_rt.get_as_build_sizes(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, prim_counts, &sizes);
        auto& B = veg_blas_[q];
        {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizes.accelerationStructureSize;
            bci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bci.sharingMode = compute_queue_distinct_ ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
            bci.queueFamilyIndexCount = compute_queue_distinct_ ? 2u : 0u;
            bci.pQueueFamilyIndices = compute_queue_distinct_ ? shared_families : nullptr;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &B.buf, &B.alloc, nullptr), "veg blas buf");
        }
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = B.buf; ci.size = sizes.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vk_check(g_rt.create_as(device_, &ci, nullptr, &B.as), "veg blas");
        bgi.dstAccelerationStructure = B.as;
        VmaAllocation salloc = nullptr;
        VkBuffer scratch = make_buffer(allocator_, sizes.buildScratchSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       false, salloc, nullptr);
        bgi.scratchData.deviceAddress = buffer_device_address(device_, scratch);
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            (void)range;
            const VkAccelerationStructureBuildRangeInfoKHR* pr = ranges;
            g_rt.cmd_build_as(cb, 1, &bgi, &pr);
        });
        vmaDestroyBuffer(allocator_, scratch, salloc);
        B.addr = as_device_address(device_, B.as);
    }
}

void VulkanEngine::destroy_vegetation() {
    for (auto& B : veg_blas_) {
        if (B.as) g_rt.destroy_as(device_, B.as, nullptr);
        if (B.buf) vmaDestroyBuffer(allocator_, B.buf, B.alloc);
    }
    veg_blas_.clear();
    for (VkPipeline* p : { &veg_cull_pipeline_, &veg_color_pipeline_, &veg_gbuf_pipeline_, &veg_depth_pipeline_ })
        if (*p) { vkDestroyPipeline(device_, *p, nullptr); *p = VK_NULL_HANDLE; }
    if (veg_layout_) vkDestroyPipelineLayout(device_, veg_layout_, nullptr);
    if (veg_cull_layout_) vkDestroyPipelineLayout(device_, veg_cull_layout_, nullptr);
    veg_layout_ = veg_cull_layout_ = VK_NULL_HANDLE;
    destroy_veg_hiz();
    destroy_veg_hiz_pipeline();
    if (veg_fade_buf_) { vmaDestroyBuffer(allocator_, veg_fade_buf_, veg_fade_alloc_); veg_fade_buf_ = VK_NULL_HANDLE; }
    for (VkShaderModule* m : { &veg_vert_module_, &veg_cull_module_, &veg_depth_frag_module_, &veg_hiz_module_ })
        if (*m) { vkDestroyShaderModule(device_, *m, nullptr); *m = VK_NULL_HANDLE; }
    if (veg_pool_) vkDestroyDescriptorPool(device_, veg_pool_, nullptr);
    if (veg_set_layout_) vkDestroyDescriptorSetLayout(device_, veg_set_layout_, nullptr);
    veg_pool_ = VK_NULL_HANDLE; veg_set_layout_ = VK_NULL_HANDLE;
    auto kill = [&](VkBuffer& b, VmaAllocation& a) { if (b) vmaDestroyBuffer(allocator_, b, a); b = VK_NULL_HANDLE; };
    kill(veg_inst_buf_, veg_inst_alloc_);
    veg_inst_mapped_ = nullptr;
    kill(veg_vis_buf_, veg_vis_alloc_);
    kill(veg_ind_buf_, veg_ind_alloc_);
    kill(veg_arch_buf_, veg_arch_alloc_);
    if (veg_mesh_.vertex_buffer) destroy_mesh(allocator_, veg_mesh_);
}

// TLAS: the instances nearest the camera (RT shadows / AO / GI).
void VulkanEngine::collect_veg_rt_instances(glm::vec3 eye, uint32_t cap,
                                            std::vector<uint32_t>& out) const {
    out.clear();
    if (veg_blas_.empty()) return;
    const float R = 650.0f;   // trees cast RT shadows out to here
    std::vector<std::pair<float, uint32_t>> cand;
    for (uint32_t i = 0; i < veg_count_; ++i) {
        const auto& v = veg_inst_cpu_[i];
        if (v.pos_scale.w <= 0.0f) continue;   // destroyed (blast_vegetation)
        float dx = v.pos_scale.x - eye.x, dz = v.pos_scale.z - eye.z;
        float d2 = dx * dx + dz * dz;
        if (d2 < R * R) cand.push_back({ d2, i });
    }
    if (cand.size() > cap) {
        std::nth_element(cand.begin(), cand.begin() + cap, cand.end());
        cand.resize(cap);
    }
    // Encode the RT shape level in the top 2 bits. It follows the LOD the
    // rasteriser picks (same pixel-radius thresholds as veg_cull.comp):
    // with a finer shape in the TLAS than on screen, the extra leaf clumps
    // of the fine model shadowed the coarse crown in black patches.
    const float ppr = float(swapchain_extent_.height) / (2.0f * std::tan(glm::radians(80.0f) * 0.5f)) *
                      std::max(0.25f, rt_.veg_lod_bias);
    for (const auto& c : cand) {
        const auto& v = veg_inst_cpu_[c.second];
        const size_t a = size_t(v.rot_arch_phase.y + 0.5f);
        const float ysq = v.rot_arch_phase.w > 0.0f ? v.rot_arch_phase.w : 1.0f;
        // Exactly veg_cull.comp's measure: 3D distance to the crown centre
        // and the same per-instance dither, so the shadow shape is the
        // bucket that is drawn (horizontal distance picked a finer one
        // looking down from hills, the missing dither one off at edges).
        const float hh = a < veg_arch_.size() ? veg_arch_[a].height * 0.5f : 0.0f;
        const float r = (a < veg_arch_.size() ? veg_arch_[a].radius : 1.0f) * v.pos_scale.w * std::max(ysq, 1.0f) + 0.6f;
        const glm::vec3 cc = glm::vec3(v.pos_scale) + glm::vec3(0.0f, hh * v.pos_scale.w * ysq, 0.0f);
        const float dither = glm::fract(v.rot_arch_phase.z * 2.7182818f) * 0.25f + 0.88f;
        const float px = r / std::max(glm::distance(cc, eye), 0.1f) * ppr * dither;
        uint32_t lvl = px > 85.0f ? 0u : px > 40.0f ? 1u : px > 16.0f ? 2u : 3u;
        if (a < veg_arch_.size() && veg_arch_[a].species <= 2) {     // trees / bushes: same detail curve as veg_cull.comp
            const float detail = std::clamp(std::pow(px / (veg_arch_[a].species == 2 ? 25.0f : 60.0f), 1.2f), 0.03f, 1.0f);
            lvl = detail > 0.5f ? 0u : detail > 0.22f ? 1u : detail > 0.09f ? 2u : 3u;
        }
        out.push_back(c.second | (lvl << 30));
    }
}


// ---------------------------------------------------------------------
// Hi-Z (max-depth pyramid) for vegetation occlusion culling. Built at the
// start of each frame from the PREVIOUS frame's prepass depth copy
// (binding-27 image); veg_cull.comp reprojects with that frame's vp.
// ---------------------------------------------------------------------
void VulkanEngine::create_veg_hiz() {
    if (!depth_copy_image_ || !depth_copy_view_) return;
    veg_hiz_w_ = std::max(1u, (render_extent_.width + 1) / 2);
    veg_hiz_h_ = std::max(1u, (render_extent_.height + 1) / 2);
    veg_hiz_mips_ = 1;
    for (uint32_t m = std::max(veg_hiz_w_, veg_hiz_h_); m > 1; m >>= 1) ++veg_hiz_mips_;
    veg_hiz_mips_ = std::min(veg_hiz_mips_, uint32_t(kVegHizMaxMips));
    veg_hiz_src_view_ = depth_copy_view_;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32_SFLOAT;
    ici.extent = { veg_hiz_w_, veg_hiz_h_, 1 };
    ici.mipLevels = veg_hiz_mips_; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &veg_hiz_image_, &veg_hiz_alloc_, nullptr), "hiz image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = veg_hiz_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, veg_hiz_mips_, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &veg_hiz_view_), "hiz view");
    for (uint32_t m = 0; m < veg_hiz_mips_; ++m) {
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &veg_hiz_mip_views_[m]), "hiz mip view");
    }
    if (!veg_hiz_sampler_) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = VK_LOD_CLAMP_NONE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &veg_hiz_sampler_), "hiz sampler");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image = veg_hiz_image_;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, veg_hiz_mips_, 0, 1 };
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cb, &dep);
    });

    // Build pipeline (once) + per-mip descriptor sets.
    if (!veg_hiz_pipeline_) {
        VkDescriptorSetLayoutBinding hb[3]{};
        hb[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        hb[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        hb[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3; lci.pBindings = hb;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &veg_hiz_set_layout_), "hiz set layout");
        VkPushConstantRange pr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &veg_hiz_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &veg_hiz_layout_), "hiz layout");
        veg_hiz_module_ = vkpipe::load_shader_module(device_, std::string(QLIKE_SHADER_DIR) + "/hiz_build.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = veg_hiz_module_;
        cci.stage.pName = "main";
        cci.layout = veg_hiz_layout_;
        vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &veg_hiz_pipeline_), "hiz pipeline");
    }
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, uint32_t(kVegHizMaxMips) },
                                   { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, uint32_t(2 * kVegHizMaxMips) } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = kVegHizMaxMips; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &veg_hiz_pool_), "hiz pool");
    for (uint32_t m = 0; m < veg_hiz_mips_; ++m) {
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = veg_hiz_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &veg_hiz_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &veg_hiz_sets_[m]), "hiz set");
        VkDescriptorImageInfo di0{ depth_copy_sampler_, depth_copy_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo di1{ VK_NULL_HANDLE, veg_hiz_mip_views_[m > 0 ? m - 1 : 0], VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo di2{ VK_NULL_HANDLE, veg_hiz_mip_views_[m], VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[3]{};
        for (int k = 0; k < 3; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = veg_hiz_sets_[m]; w[k].dstBinding = uint32_t(k); w[k].descriptorCount = 1;
        }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &di0;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[1].pImageInfo = &di1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[2].pImageInfo = &di2;
        vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    }
    veg_hiz_ready_ = false;
}

void VulkanEngine::write_veg_hiz_binding() {
    if (!veg_set_ || !veg_hiz_view_) return;
    VkDescriptorImageInfo di{ veg_hiz_sampler_, veg_hiz_view_, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = veg_set_; w.dstBinding = 4; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &di;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void VulkanEngine::destroy_veg_hiz() {
    for (auto& v : veg_hiz_mip_views_) if (v) { vkDestroyImageView(device_, v, nullptr); v = VK_NULL_HANDLE; }
    if (veg_hiz_view_) vkDestroyImageView(device_, veg_hiz_view_, nullptr);
    if (veg_hiz_image_) vmaDestroyImage(allocator_, veg_hiz_image_, veg_hiz_alloc_);
    if (veg_hiz_pool_) vkDestroyDescriptorPool(device_, veg_hiz_pool_, nullptr);
    veg_hiz_view_ = VK_NULL_HANDLE; veg_hiz_image_ = VK_NULL_HANDLE; veg_hiz_pool_ = VK_NULL_HANDLE;
    veg_hiz_ready_ = false;
}

void VulkanEngine::destroy_veg_hiz_pipeline() {
    if (veg_hiz_pipeline_) vkDestroyPipeline(device_, veg_hiz_pipeline_, nullptr);
    if (veg_hiz_layout_) vkDestroyPipelineLayout(device_, veg_hiz_layout_, nullptr);
    if (veg_hiz_set_layout_) vkDestroyDescriptorSetLayout(device_, veg_hiz_set_layout_, nullptr);
    if (veg_hiz_sampler_) vkDestroySampler(device_, veg_hiz_sampler_, nullptr);
    veg_hiz_pipeline_ = VK_NULL_HANDLE; veg_hiz_layout_ = VK_NULL_HANDLE;
    veg_hiz_set_layout_ = VK_NULL_HANDLE; veg_hiz_sampler_ = VK_NULL_HANDLE;
}

void VulkanEngine::build_veg_hiz(VkCommandBuffer cmd) {
    if (!depth_copy_image_) return;
    // Resize / depth-copy recreation -> rebuild the pyramid resources.
    if (veg_hiz_src_view_ != depth_copy_view_ ||
        veg_hiz_w_ != std::max(1u, (render_extent_.width + 1) / 2) ||
        veg_hiz_h_ != std::max(1u, (render_extent_.height + 1) / 2) || !veg_hiz_image_) {
        vkDeviceWaitIdle(device_);
        destroy_veg_hiz();
        create_veg_hiz();
        write_veg_hiz_binding();
        veg_hiz_have_prev_ = false;
    }
    if (!veg_hiz_pipeline_ || !depth_copy_layout_valid_ || !veg_hiz_have_prev_) {
        veg_hiz_have_prev_ = depth_copy_layout_valid_;   // valid from next frame on
        veg_hiz_ready_ = false;
        return;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, veg_hiz_pipeline_);
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);   // previous frame's cull read the pyramid
    uint32_t sw = render_extent_.width, sh = render_extent_.height;
    for (uint32_t m = 0; m < veg_hiz_mips_; ++m) {
        uint32_t dw = std::max(1u, veg_hiz_w_ >> m), dh = std::max(1u, veg_hiz_h_ >> m);
        int32_t info[4] = { m == 0 ? 1 : 0, int32_t(sw), int32_t(sh), 0 };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, veg_hiz_layout_, 0, 1,
                                &veg_hiz_sets_[m], 0, nullptr);
        vkCmdPushConstants(cmd, veg_hiz_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, info);
        vkCmdDispatch(cmd, (dw + 7) / 8, (dh + 7) / 8, 1);
        vkCmdPipelineBarrier2(cmd, &dep);
        sw = dw; sh = dh;
    }
    veg_hiz_ready_ = true;
}

// Blast damage to the forest. Every tree / bush whose base lies inside the
// blast radius is destroyed: hidden (instance scale 0 in the CPU copy and
// the host-mapped GPU buffer, so culling, impostors and the RT list drop
// it), its trunk collider removed from the Jolt compound, and its trunk
// thrown out as wood splinters - pushed away from the blast, harder the
// closer the tree was, some burning / smoking. The foliage simply goes up
// in the explosion (no green debris). Rocks are left alone.
void VulkanEngine::blast_vegetation(glm::vec3 pos, float radius) {
    if (veg_inst_cpu_.empty() || radius <= 0.5f) return;
    auto& R = spacejet_rng_state_;
    std::vector<glm::vec2> trunks;
    int n_trees = 0;
    static uint32_t next_id = 0x40000000u;
    for (size_t i = 0; i < veg_inst_cpu_.size(); ++i) {
        VegInstance& v = veg_inst_cpu_[i];
        const float s = v.pos_scale.w;
        if (s <= 0.0f) continue;
        const glm::vec3 p(v.pos_scale);
        const float dx = p.x - pos.x, dz = p.z - pos.z;
        const float d2 = dx * dx + dz * dz;
        const int a = int(v.rot_arch_phase.y + 0.5f);
        if (a < 0 || size_t(a) >= veg_arch_.size()) continue;
        const int sp = veg_arch_[size_t(a)].species;
        if (sp == 3) continue;                                    // rocks survive
        const float r_eff = radius * (sp == 2 ? 1.15f : 1.0f);   // bushes go a bit further out
        if (d2 > r_eff * r_eff || std::abs(p.y - pos.y) > 25.0f) continue;

        v.pos_scale.w = 0.0f;
        if (veg_inst_mapped_)
            static_cast<VegInstance*>(veg_inst_mapped_)[i].pos_scale.w = 0.0f;
        trunks.push_back(glm::vec2(p.x, p.z));
        ++n_trees;

        // Splinters: the trunk breaks into pieces along its height.
        const float d = std::sqrt(d2);
        const float h = veg_arch_[size_t(a)].height * s * (v.rot_arch_phase.w > 0.0f ? v.rot_arch_phase.w : 1.0f);
        const float closeness = 1.0f - glm::clamp(d / r_eff, 0.0f, 1.0f);
        glm::vec3 away = glm::vec3(dx, 0.0f, dz);
        away = glm::length(away) > 0.1f ? glm::normalize(away) : glm::vec3(1, 0, 0);
        const int pieces = sp == 2 ? 2 : 4 + int(xorshift32(R) % 4u);
        for (int k = 0; k < pieces; ++k) {
            JetDebris w{};
            w.chunk = -1;
            w.pos = w.prev_pos = p + glm::vec3(0.0f, h * frand_range(R, 0.05f, 0.55f), 0.0f);
            glm::vec3 dir = glm::normalize(away + glm::vec3(frand_range(R, -0.5f, 0.5f),
                                                            frand_range(R, 0.4f, 1.1f),
                                                            frand_range(R, -0.5f, 0.5f)));
            w.vel = dir * frand_range(R, 12.0f, 38.0f) * (0.45f + closeness);
            glm::vec3 ax(frand_range(R, -1, 1), frand_range(R, -1, 1), frand_range(R, -1, 1));
            w.spin_axis = glm::normalize(ax + glm::vec3(1e-3f));
            w.spin_rate = frand_range(R, 3.0f, 10.0f);
            w.rot = w.prev_rot = glm::mat3(glm::rotate(glm::mat4(1.0f), frand_range(R, 0.0f, 6.28f), w.spin_axis));
            const float girth = (sp == 2 ? 0.10f : 0.18f) * std::max(s, 0.5f);
            w.scale = 1.0f;
            w.dims = glm::vec3(girth * frand_range(R, 0.6f, 1.2f), girth * frand_range(R, 0.5f, 1.0f),
                               (sp == 2 ? 0.5f : 1.0f) * frand_range(R, 0.7f, 2.6f));
            w.color = glm::vec3(0.30f, 0.21f, 0.13f) * frand_range(R, 0.6f, 1.1f);   // raw / charred wood
            w.vt = frand_range(R, 30.0f, 55.0f);
            w.burning = (xorshift32(R) % 4u) == 0u;
            w.id = next_id++;
            jet_debris_.push_back(std::move(w));
        }
    }
    if (n_trees == 0) return;
    if (jet_debris_.size() > 400) jet_debris_.erase(jet_debris_.begin(), jet_debris_.begin() + (jet_debris_.size() - 400));
    size_t cols = physics_ ? physics_->remove_compound_boxes_at(trunks.data(), trunks.size(), 0.8f) : 0;
    veg_rt_valid_ = false;          // re-pick the RT tree set (destroyed ones drop out)
    log::infof("[veg] blast at (%.0f, %.0f, %.0f) r %.0f m: %d trees/bushes destroyed, %zu colliders removed",
               pos.x, pos.y, pos.z, radius, n_trees, cols);
}

} // namespace qlike
