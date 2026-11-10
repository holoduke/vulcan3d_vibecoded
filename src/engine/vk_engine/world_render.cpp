// Per-frame frame-view computation + the raster render passes: depth
// pre-pass, half-rate shadow pass, raymarched terrain/grass compose, and
// the main render_world color pass that emits every cube/cylinder draw.
// Split out of the old monolithic world.cpp (maintainability pass).

#include "engine/vk_engine/internal.h"
#include "engine/gltf_loader.h"
#include "engine/grass.h"
#include "engine/skybox.h"
#include "engine/terrain.h"
#include "engine/texture.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <utility>
#include <vector>

namespace qlike {

// Per-frame camera state shared across the three render passes
// (terrain_raymarch_lr, world depth pre-pass, world color pass). draw()
// calls this once before any of them; each pass reads the cached
// `current_frame_view_`. Bit-identical between passes so the LESS_OR_EQUAL
// depth test survives.
VulkanEngine::FrameView VulkanEngine::compute_frame_view() {
    FrameView fv{};
    float aspect = static_cast<float>(render_extent_.width) /
                   static_cast<float>(render_extent_.height);
    constexpr float kFixedDt = 1.0f / 120.0f;
    float alpha = physics_accumulator_ / kFixedDt;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    fv.render_pos = glm::mix(prev_player_position_, player_.position, alpha);

    glm::vec3 saved = player_.position;
    player_.position = fv.render_pos;
    fv.view = player_.view_matrix();
    // Explosion camera shake: small, fast, decaying rotational jitter
    // (sum of incommensurate sines = smooth noise, no RNG state).
    if (cam_shake_ > 1e-4f) {
        float tt = static_cast<float>(glfwGetTime());
        float a = cam_shake_;
        float rx = a * 0.020f * (std::sin(tt * 37.0f) + 0.6f * std::sin(tt * 71.0f + 1.3f));
        float ry = a * 0.016f * (std::sin(tt * 43.0f + 2.1f) + 0.5f * std::sin(tt * 89.0f));
        float rz = a * 0.010f * std::sin(tt * 29.0f + 0.7f);
        glm::mat4 shake = glm::rotate(glm::mat4(1.0f), rx, glm::vec3(1, 0, 0)) *
                          glm::rotate(glm::mat4(1.0f), ry, glm::vec3(0, 1, 0)) *
                          glm::rotate(glm::mat4(1.0f), rz, glm::vec3(0, 0, 1));
        fv.view = shake * fv.view;
    }
    // Capture eye position with the lerped body pos still active so
    // raymarched passes (camera_pos in scene UBO) match the smoothed
    // view matrix. Without this the raw physics-tick eye is ahead of
    // the lerped camera and grass/terrain visibly lag rasterised geo.
    fv.eye_pos = player_.eye_position();
    player_.position = saved;
    fv.proj = glm::perspective(glm::radians(80.0f), aspect, 0.05f, 1500.0f);
    fv.proj[1][1] *= -1.0f;
    if (rt_.taa_jitter_enabled || rt_.fsr2_enabled) {
        auto halton = [](int b, int i) {
            float f = 1.0f, r = 0.0f;
            while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
            return r;
        };
        // FSR2 path: phase length scales with the upscale ratio so the
        // jitter fully covers each output pixel over `phase_count`
        // frames. Default formula: ceil(8 * (display / render)^2).
        // For 1.0x = 8 samples, 0.67x = 18, 0.5x = 32.
        // TAA path: keeps the original 16-sample loop.
        int phase_count;
        float strength;
        if (rt_.fsr2_enabled) {
            float ratio = static_cast<float>(swapchain_extent_.width) /
                          static_cast<float>(std::max(1u, render_extent_.width));
            phase_count = static_cast<int>(std::ceil(8.0f * ratio * ratio));
            phase_count = std::max(8, std::min(64, phase_count));
            strength    = 1.0f;   // full sub-pixel for FSR2
        } else {
            phase_count = 16;
            strength    = rt_.taa_jitter_strength;
        }
        int idx = static_cast<int>(frame_number_ % phase_count) + 1;
        float jx = (halton(2, idx) - 0.5f) * 2.0f * strength /
                   static_cast<float>(render_extent_.width);
        float jy = (halton(3, idx) - 0.5f) * 2.0f * strength /
                   static_cast<float>(render_extent_.height);
        fv.proj[2][0] += jx;
        fv.proj[2][1] += jy;
        // Stash for fsr2 dispatch (Phase 3 needs it for the
        // jitter-cancel reproject).
        fv.jitter = glm::vec2(jx, jy);
    }
    fv.vp = fv.proj * fv.view;
    fv.inv_vp = glm::inverse(fv.vp);
    // Hoisted once-per-frame: inv_view (draw_viewmodel) + frustum (3
    // render passes used to each call extract_frustum on the same vp).
    fv.inv_view = glm::inverse(fv.view);
    fv.frustum = extract_frustum(fv.vp);
    return fv;
}

void VulkanEngine::render_terrain_raymarch_lr(VkCommandBuffer cmd) {
    const FrameView& fv = current_frame_view_;
    const glm::mat4& vp = fv.vp;

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(tr_lr_extent_.width);
    vp_state.height = static_cast<float>(tr_lr_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, tr_lr_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                       terrain_raymarch_lr_pipeline_ ? terrain_raymarch_lr_pipeline_
                                                     : terrain_raymarch_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeline_layout_, 0, 1, &scene_desc_set_,
                             0, nullptr);

    PushConstants pc{};
    pc.mvp      = vp;
    pc.model    = fv.inv_vp;
    pc.prev_mvp = prev_view_proj_valid_ ? prev_view_proj_ : vp;
    // Plateau half-extent must mirror the full-res raymarch path
    // (line ~2800) вЂ” otherwise the LR pass paints a much wider
    // plateau than the full-res one, and switching scale below 100%
    // makes the area around the castle visibly jump up to plateau
    // height. Was 28.0 (pre-shrink default), full-res was already 11.5.
    pc.color    = glm::vec4(0.0f, 0.0f, 11.5f, 22.0f);
    // emissive layout for the raymarch shader:
    //   x = step factor (0.15..0.8) вЂ” march step relative to SDF gap
    //   y = lod_near_m  вЂ” ray distance at which the FBM LOD ramp starts
    //   z = lod_far_m   вЂ” ray distance at which the FBM LOD ramp ends
    //   w = lod_min_oct вЂ” minimum octave count at far end of ramp
    pc.emissive = glm::vec4(
        std::clamp(rt_.terrain_raymarch_step_factor, 0.15f, 0.8f),
        std::max(1.0f, rt_.terrain_raymarch_lod_near_m),
        std::max(rt_.terrain_raymarch_lod_near_m + 1.0f,
                 rt_.terrain_raymarch_lod_far_m),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
    pc.tex_params = glm::vec4(
        static_cast<float>(std::clamp(rt_.terrain_raymarch_max_steps, 60, 300)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_shadow_steps, 16, 96)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_octaves, 4, 24)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_normal_octaves, 4, 32)));
    // grass_params slot repurposed for raymarch knobs:
    //   x = volumetric fog strength (0 = off)
    //   y = relaxation cone-stepping flag (>0.5 = on)
    //   z = fog god-ray self-shadow flag (>0.5 = on)
    //   w = water RT-reflection flag (>0.5 = march FBM in refl dir)
    pc.grass_params = glm::vec4(
        std::max(0.0f, rt_.terrain_raymarch_fog_strength),
        rt_.terrain_raymarch_relaxation ? 1.0f : 0.0f,
        rt_.terrain_raymarch_fog_godrays ? 1.0f : 0.0f,
        rt_.water_rt_reflections ? 1.0f : 0.0f);
    vkCmdPushConstants(cmd, pipeline_layout_,
                       kPushConstantStages,
                       0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void VulkanEngine::render_terrain_raymarch_compose(VkCommandBuffer cmd) {
    // Fullscreen-tri compose pass РІР‚вЂќ sits at the START of the main
    // world color pass. Samples LR raymarch outputs and writes
    // upscaled color/motion + gl_FragDepth. The pipeline's depth_test
    // = LESS_OR_EQUAL means cube/castle/dyn-prop fragments drawn
    // afterwards by the rest of render_world() correctly occlude the
    // terrain where they're closer.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                       tr_compose_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeline_layout_, 0, 1, &scene_desc_set_,
                             0, nullptr);
    // pc.color = (sharpen, LR width, LR height, 0). The shader uses
    // .x for the CAS-style adaptive sharpen amount and .yz to size
    // the kernel's neighbour offsets at the LR resolution (so the
    // 4-tap kernel matches actual data, not the upscaled grid).
    PushConstants pc{};
    pc.color = glm::vec4(rt_.terrain_raymarch_sharpen,
                         static_cast<float>(tr_lr_extent_.width),
                         static_cast<float>(tr_lr_extent_.height),
                         0.0f);
    vkCmdPushConstants(cmd, pipeline_layout_,
                       kPushConstantStages,
                       0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void VulkanEngine::render_grass_raymarch(VkCommandBuffer cmd) {
    // Fullscreen-tri pass that walks the SDF blade field. Drawn into the
    // same color/motion/depth attachments as the cube draws, AFTER them,
    // so cube/castle/dyn-prop depth (already in depth_image_) correctly
    // occludes the marched grass.
    const FrameView& fv = current_frame_view_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grass_rm_pipeline_);
    // descriptor set is already bound by render_world's earlier draws.

    PushConstants pc{};
    pc.mvp     = fv.vp;
    pc.model   = fv.inv_vp;          // shader uses model as inv(view_proj)
    pc.prev_mvp = prev_view_proj_valid_ ? prev_view_proj_ : fv.vp;
    pc.color   = glm::vec4(rt_.grass_raymarch_distance,
                            rt_.grass_wind,
                            static_cast<float>(frame_number_) * 0.016f,
                            rt_.grass_cutoff_soft_dist);
    // emissive.{y,z,w} = LOD ramp params (shared with terrain raymarch).
    // .x stays 0 вЂ” the grass shader doesn't read a step factor.
    pc.emissive = glm::vec4(
        0.0f,
        std::max(1.0f, rt_.terrain_raymarch_lod_near_m),
        std::max(rt_.terrain_raymarch_lod_near_m + 1.0f,
                 rt_.terrain_raymarch_lod_far_m),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
    pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
    // .w = blade height scale. The SDF blade is ~1.7 units tall
    // intrinsically; grass_raymarch.frag multiplies by .w * 0.21 so the
    // default slider (в‰€1.0) gives ~0.25-0.37 m blades вЂ” matching the
    // instanced raster grass instead of the old 1.1-1.7 m dark spears.
    pc.grass_params = glm::vec4(rt_.grass_distance, rt_.grass_wind,
                                 static_cast<float>(frame_number_) * 0.016f,
                                 rt_.grass_height_scale);
    vkCmdPushConstants(cmd, pipeline_layout_,
                       kPushConstantStages,
                       0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void VulkanEngine::render_world_depth_pass(VkCommandBuffer cmd) {
    // Cached FrameView built once at top of draw(); bit-identical with
    // the color pass so depth-EQUAL/LESS_OR_EQUAL stays correct.
    const FrameView& fv = current_frame_view_;
    const glm::mat4& vp = fv.vp;
    // Frustum hoisted into FrameView: was extract_frustum(vp) per pass
    // (3 callers all on the same vp); now shared.
    const Frustum& frustum = fv.frustum;

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(render_extent_.width);
    vp_state.height = static_cast<float>(render_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    auto push_depth = [&](const glm::mat4& model) {
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    // Skip SPOM brushes in the depth pre-pass. Their cube.frag color
    // path may emit alpha = 0 at silhouette cavities (so terrain /
    // earlier draws show through the bumpy brick edge); but if the
    // pre-pass writes the wall's full geometric depth here, the later
    // terrain compose's gl_FragDepth test fails at silhouette pixels
    // в†’ terrain pixel discarded в†’ scene_color stays at sky clear
    // colour в†’ user sees sky through the cavity gap (the symptom that
    // looks like a "squared edge" with sky leak). Letting terrain
    // compose win the depth race for those pixels fixes it; we lose
    // early-Z on castle stone walls only.
    // Only the SPOM WALL brushes (albedo 1, 4) need to skip the depth
    // pre-pass вЂ” that's how their silhouette cavity shows what's
    // behind. Floors (albedo 5, 6) are SPOM-textured but face UP, so
    // their silhouette never points at terrain. Skipping them here
    // caused TAA flicker: cube color pass and terrain compose race
    // at the 5 cm floor-vs-plateau separation, and sub-pixel jitter
    // flips the winner every frame в†’ "flashing through the floor".
    auto is_spom_albedo = [](int a) {
        return a == 1 || a == 4;
    };
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.render_aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) continue;
        if (is_spom_albedo(b.tex_albedo)) continue;
        push_depth(static_brush_models_[i]);    // pre-baked, see init_world
    }
    // Castle mesh parts (no discard in mesh mode -> safe in the prepass).
    if (!castle_meshes_.empty()) {
        draw_castle_meshes(cmd, vp, vp, frustum, true, VK_NULL_HANDLE);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    if (!aa_.empty()) {
        draw_aa(cmd, vp, vp, frustum, true, VK_NULL_HANDLE);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    // GPU-instanced vegetation + rocks (indirect, culled on the GPU).
    if (veg_count_ > 0) {
        draw_vegetation(cmd, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) continue;
        // dr.model = dr.world * scale(full_size), baked in
        // rebuild_dyn_render_cache (was per-draw glm::scale here).
        push_depth(dr.model);
    }
    // Terrain: quadtree CLOD, single instanced draw. The pre-pass and
    // colour pass share terrain.vert + the same instance list + the same
    // push constants, so the primed depth is bit-identical to the colour
    // pass and LESS_OR_EQUAL never rejects.
    if (terrain_clod_cam_count_ > 0 && !rt_.terrain_raymarch_enabled &&
        terrain_clod_patch_.vertex_buffer) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          terrain_depth_pipeline_);
        VkBuffer vbufs[2] = { terrain_clod_patch_.vertex_buffer,
                              terrain_clod_inst_buf_ };
        VkDeviceSize voffs[2] = { 0, terrain_clod_cam_offset_ };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
        vkCmdBindIndexBuffer(cmd, terrain_clod_patch_.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        pc.mvp = vp;
        pc.model = glm::mat4(1.0f);
        pc.emissive = terrain_clod_hm_params();
        pc.tex_params = glm::vec4(rt_.terrain_pom_strength, 0.0f, 1.0f, 2.0f);
        pc.grass_params = glm::vec4(rt_.terrain_sand_ripple_scale,
                                    rt_.terrain_grass_line_scale,
                                    rt_.terrain_disp_amp,
                                    rt_.terrain_disp_smooth_mip);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, terrain_clod_patch_.index_count,
                         terrain_clod_cam_count_, 0, 0, 0);
        // Re-bind the caller's depth pipeline for any post-terrain draws.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline_);
    }
}

// Half-rate sun-shadow producer pass (roadmap item #4, Phase 2). Re-uses
// the depth pre-pass geometry walk (brushes + dyn props) at half the
// render extent. Skips terrain (terrain has its own hybrid bake/grass-
// shadow-map path that the consumer in cube.frag will continue to
// honour). Pipeline writes a single R8 occlusion value via one RT
// shadow ray per half-res pixel into shadow_lr_image_.
void VulkanEngine::render_world_shadow_lr_pass(VkCommandBuffer cmd) {
    const FrameView& fv = current_frame_view_;
    const glm::mat4& vp = fv.vp;
    // Shared FrameView frustum (same vp as the depth + color passes).
    const Frustum& frustum = fv.frustum;

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(shadow_lr_extent_.width);
    vp_state.height = static_cast<float>(shadow_lr_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);
    VkRect2D scissor{ {0, 0}, shadow_lr_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_lr_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    auto push = [&](const glm::mat4& model) {
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& a = world_.render_aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) continue;
        push(static_brush_models_[i]);
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) continue;
        // dr.model = dr.world * scale(full_size), baked once per frame.
        push(dr.model);
    }
}

void VulkanEngine::render_world(VkCommandBuffer cmd) {
    // Cached FrameView built once at top of draw().
    const FrameView& fv = current_frame_view_;
    last_view_proj_ = fv.vp;
    last_inv_view_proj_ = fv.inv_vp;
    const glm::mat4& vp = last_view_proj_;
    // Shared FrameView frustum -- same vp as depth + shadow_lr passes.
    const Frustum& frustum = fv.frustum;

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(render_extent_.width);
    vp_state.height = static_cast<float>(render_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);

    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const bool tex_on = rt_.textures_enabled;
    // prev_view_proj_ is captured at the END of each frame from the previous
    // frame's last_view_proj_; on frame 0 it's identity (which produces a
    // single-frame motion-vector glitch Р Р†Р вЂљРІР‚Сњ TAA's "history_valid" gate handles
    // the invalid-history first frame, so this doesn't surface visually).
    const glm::mat4 prev_vp = prev_view_proj_;
    auto draw_brush = [&](const glm::mat4& model, const glm::mat4& prev_model,
                          const glm::vec4& color,
                          const glm::vec3& emissive, bool full_emissive,
                          int tex_albedo, int tex_normal, float uv_scale,
                          bool object_space, int face_mask = -1) {
        PushConstants pc{};
        // Static brushes: .z = exposed-face bitmask, .w = 1 -> cube.frag box
        // relief (real depth at exposed edges, flat at brush-brush seams).
        if (face_mask >= 0)
            pc.grass_params = glm::vec4(0.0f, 0.0f, static_cast<float>(face_mask), 1.0f);
        pc.mvp = vp * model;
        pc.model = model;
        pc.prev_mvp = prev_vp * prev_model;
        pc.color = color;
        pc.emissive = glm::vec4(emissive, full_emissive ? 1.0f : 0.0f);
        if (tex_on) {
            pc.tex_params = glm::vec4(static_cast<float>(tex_albedo),
                                      static_cast<float>(tex_normal),
                                      uv_scale, object_space ? 1.0f : 0.0f);
        } else {
            pc.tex_params = glm::vec4(-1.0f, -1.0f, uv_scale, 0.0f);
        }
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    if (brush_face_mask_.size() != world_.brushes.size()) compute_brush_face_masks();
    prof_mark(cmd, "w_begin");
    int culled_static = 0, culled_dyn = 0, drawn_static = 0, drawn_dyn = 0;
    // Terrain pass Р Р†Р вЂљРІР‚Сњ chunked. Each visible chunk is drawn at full LOD
    // when within `near_lod` metres of the camera, otherwise half LOD.
    // The merged-static-BLAS-style "single big draw" is reserved for the
    // RT path; for raster we want LOD + frustum cull per chunk so the
    // 2km terrain stays cheap from any viewpoint.
    // Mesh (CDLOD) terrain draws first when the raymarcher is off, so its
    // depth is in the buffer before the (optional) water-only raymarch
    // pass below вЂ” hills then correctly occlude the water plane.
    if (!rt_.terrain_raymarch_enabled && terrain_clod_cam_count_ > 0 &&
        terrain_clod_patch_.vertex_buffer) {
        // Quadtree CLOD terrain: ONE instanced draw. terrain.vert fetches
        // height + normal from the heightmap texture per instance node;
        // push constants are identical to the depth pre-pass so the
        // LESS_OR_EQUAL prime always matches.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            (rt_.terrain_wireframe && terrain_wire_pipeline_)
                ? terrain_wire_pipeline_
                : ((terrain_gbuf_pipeline_ && deferred_owns_lighting())
                       ? terrain_gbuf_pipeline_ : terrain_pipeline_));
        VkBuffer vbufs[2] = { terrain_clod_patch_.vertex_buffer,
                              terrain_clod_inst_buf_ };
        VkDeviceSize voffs[2] = { 0, terrain_clod_cam_offset_ };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
        vkCmdBindIndexBuffer(cmd, terrain_clod_patch_.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        pc.mvp = vp;
        pc.model = glm::mat4(1.0f);
        pc.prev_mvp = prev_vp;
        // .emissive = heightmap mapping (origin XZ, 1/cell, 1/(dim+1)).
        pc.emissive = terrain_clod_hm_params();
        // .x = terrain parallax-occlusion strength, .w = 2 в†’ is_terrain.
        pc.tex_params = glm::vec4(rt_.terrain_pom_strength, 0.0f, 1.0f, 2.0f);
        // .xy = shore ripple scales (fragment), .zw = rocky detail
        // displacement amp / mip (vertex) вЂ” MUST match the depth pre-pass.
        pc.grass_params = glm::vec4(rt_.terrain_sand_ripple_scale,
                                    rt_.terrain_grass_line_scale,
                                    rt_.terrain_disp_amp,
                                    rt_.terrain_disp_smooth_mip);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, terrain_clod_patch_.index_count,
                         terrain_clod_cam_count_, 0, 0, 0);
        // Switch back to the cube pipeline + cube mesh for the brush loop.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    prof_mark(cmd, "w_terrain_mesh");

    // Raymarch pass. Runs for the full FBM terrain (raymarch enabled) OR,
    // when the mesh terrain is active, as a water-only pass: the shader
    // sees terrain_local_info.x > 0.5, skips the terrain march entirely,
    // and only intersects + shades the water plane (all FBM water types,
    // foam, showthrough, reflections). Its per-pixel gl_FragDepth is
    // hardware depth-tested against the mesh depth written just above, so
    // hills occlude the water exactly like in the FBM path.
    if (rt_.terrain_raymarch_enabled
        && terrain_raymarch_pipeline_ != VK_NULL_HANDLE
        && !tr_use_lowres()) {
        // Procedural FBM ray-marched terrain РІР‚вЂќ fullscreen draw, frag
        // writes hit-point depth so rasterised cubes / castle / dyn-
        // props (which already wrote depth earlier in this pass) still
        // occlude correctly. Misses (sky) discard, leaving compose's
        // sky pass to fill those pixels.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_raymarch_pipeline_);
        PushConstants pc{};
        pc.mvp = vp;
        // Reuse cached inverse from FrameView (was glm::inverse(vp) each
        // time -- same matrix as fv.inv_vp computed once in compute_frame_view).
        pc.model = fv.inv_vp;          // frag reconstructs world ray from NDC
        pc.prev_mvp = prev_vp;
        // .xy = plateau centre, .z = plateau half-extent, .w = plateau
        // height. Shader's terrainM blends FBM toward the plateau
        // height inside the castle footprint so brushes / dyn-props
        // visibly sit on the procedural surface. Values mirror the
        // gameplay heightmap's plateau in init_world (11.5 m half-ext,
        // 22 m height, centred at origin вЂ” tight pad just under the
        // castle so the surrounding terrain stays fully procedural).
        pc.color = glm::vec4(0.0f, 0.0f, 11.5f, 22.0f);
        pc.emissive = glm::vec4(0.0f);
        // tex_params for the raymarch shader carries quality knobs:
        //   .x = max ray-march steps      (60..300)
        //   .y = shadow ray steps         (16..96)
        //   .z = FBM octaves for march    (4..9)
        //   .w = FBM octaves for normals  (4..16)
        // Lower = faster, higher = smoother / longer reach.
        pc.tex_params = glm::vec4(
            static_cast<float>(std::clamp(rt_.terrain_raymarch_max_steps, 60, 300)),
            static_cast<float>(std::clamp(rt_.terrain_raymarch_shadow_steps, 16, 96)),
            static_cast<float>(std::clamp(rt_.terrain_raymarch_octaves, 4, 24)),
            static_cast<float>(std::clamp(rt_.terrain_raymarch_normal_octaves, 4, 32)));
        // emissive layout вЂ” mirrors render_terrain_raymarch_lr:
        //   x = step factor (0.15..0.8)
        //   y = lod_near_m  вЂ” ray-t at which FBM LOD ramp starts
        //   z = lod_far_m   вЂ” ray-t at which FBM LOD ramp ends
        //   w = lod_min_oct вЂ” minimum octave count at far end
        pc.emissive = glm::vec4(
            std::clamp(rt_.terrain_raymarch_step_factor, 0.15f, 0.8f),
            std::max(1.0f, rt_.terrain_raymarch_lod_near_m),
            std::max(rt_.terrain_raymarch_lod_near_m + 1.0f,
                     rt_.terrain_raymarch_lod_far_m),
            static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
        // grass_params slot РІР‚вЂќ fog strength + flags (mirrors LR pass).
        pc.grass_params = glm::vec4(
            std::max(0.0f, rt_.terrain_raymarch_fog_strength),
            rt_.terrain_raymarch_relaxation ? 1.0f : 0.0f,
            rt_.terrain_raymarch_fog_godrays ? 1.0f : 0.0f,
            rt_.water_rt_reflections ? 1.0f : 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        // 3 verts, 1 instance РІР‚вЂќ gl_VertexIndex 0..2 covers the screen.
        vkCmdDraw(cmd, 3, 1, 0, 0);
        // Restore cube pipeline + cube mesh for the grass / dust /
        // viewmodel passes that follow.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    prof_mark(cmd, "w_terrain_raymarch");
    // Voxel buildings rendered BEFORE the rasterised water plane so the
    // water's alpha-blend pass sees the submerged voxel pixels in the
    // scene_color/depth buffer and blends over them. Without this swap,
    // voxel pixels written after the water draw replace the water tint
    // entirely вЂ” a submerged tower reads as fully opaque dry stone.
    render_voxels(cmd);
    // Re-bind cube pipeline + mesh: voxel pass swapped them out.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &scene_desc_set_,
                            0, nullptr);
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    // Castle gate panels вЂ” drawn after the static brushes (so they sit at
    // the entrance wall) but before the water raster so submerged panels
    // would also blend correctly. Same cube pipeline, just an extra two
    // draws with per-panel hinge-rotated model matrices.
    draw_doors(cmd, vp, prev_vp);
    prof_mark(cmd, "w_voxels_doors");

    // Mesh-terrain mode + water: draw the REAL rasterised water plane
    // (verbatim ocean shading in water.frag) instead of the fullscreen
    // analytic water-only pass. Depth is owned by the plane geometry, so
    // the terrainв†”water silhouette is a pixel-exact hardware depth test
    // вЂ” the previous analytic-vs-rasterised mismatch (edge gaps / seam
    // lines) is gone by construction.
    if (rt_.water_enabled && !rt_.terrain_raymarch_enabled
        && terrain_water_pipeline_ != VK_NULL_HANDLE
        && water_plane_mesh_.index_count > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          terrain_water_pipeline_);
        PushConstants pc{};
        pc.mvp = vp;
        // Reuse cached inverse (was a per-frame glm::inverse(vp) here).
        pc.model = fv.inv_vp;          // frag reconstructs the view ray
        pc.prev_mvp = prev_vp;
        // .x = water level в†’ water.vert lifts the plane to it; the frag
        // still reads scene.water_params.y for the shaded surface.
        // .y = clarity depth (m) в†’ Beer's-law absorption in water.frag.
        // .z = shoreline softness (m of depth the water fades in over).
        // .w = foam opacity (vs the see-through water it sits on).
        pc.color = glm::vec4(rt_.water_level,
                             std::max(0.1f, rt_.water_clarity_depth),
                             std::max(0.02f, rt_.water_shore_softness),
                             glm::clamp(rt_.water_foam_opacity, 0.0f, 1.0f));
        pc.emissive = glm::vec4(0.0f);
        pc.tex_params = glm::vec4(0.0f);
        // Fog strength + flags + water RT-reflection toggle (mirrors the
        // raymarch water-only push so reflections/fog match exactly).
        pc.grass_params = glm::vec4(
            std::max(0.0f, rt_.terrain_raymarch_fog_strength),
            rt_.terrain_raymarch_relaxation ? 1.0f : 0.0f,
            rt_.terrain_raymarch_fog_godrays ? 1.0f : 0.0f,
            rt_.water_rt_reflections ? 1.0f : 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        // Push the water plane slightly farther in NDC than coplanar
        // terrain so depth-test ties go to the terrain. Constant 16 +
        // slope 1.5 is generous enough to swamp perspective-interp
        // precision differences between the 2-triangle water plane
        // and the tessellated CDLOD terrain at the shoreline. The
        // bias is in depth-buffer units (D32 float here) so the
        // effective world-space push is sub-mm at near depth and a
        // few cm at 1 km вЂ” invisible to the eye, decisive for the
        // depth test.
        vkCmdSetDepthBias(cmd, /*const*/ 16.0f, /*clamp*/ 0.0f,
                          /*slope*/ 1.5f);
        VkDeviceSize woff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &water_plane_mesh_.vertex_buffer, &woff);
        vkCmdBindIndexBuffer(cmd, water_plane_mesh_.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, water_plane_mesh_.index_count, 1, 0, 0, 0);
        // Restore cube pipeline + mesh for the passes that follow.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    prof_mark(cmd, "w_water");
    // Spacejet РІР‚вЂќ hovering decoration on the plateau. One draw per
    // glTF primitive. Animated via sin(time) for the bob and a
    // slow yaw rotation. Raster-only; no physics, no BLAS.
    if (!spacejet_meshes_.empty() && !spacejet_flights_.empty()) {
        for (const auto& f : spacejet_flights_) {
            if (f.dead) continue;
            // Full 3D attitude from the AI's fwd/up basis. The asset's
            // nose points -Z, so local Z = -fwd, Y = up, X = fwd x up.
            auto build_model = [&](glm::vec3 pos, glm::vec3 fwd, glm::vec3 up) {
                glm::vec3 z = -glm::normalize(fwd);
                glm::vec3 x = glm::normalize(glm::cross(up, z));
                glm::vec3 y = glm::cross(z, x);
                glm::mat4 basis(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f),
                                glm::vec4(z, 0.0f), glm::vec4(pos, 1.0f));
                return basis *
                       glm::scale    (glm::mat4(1.0f),
                                       glm::vec3(spacejet_base_scale_ * f.scale)) *
                       glm::translate(glm::mat4(1.0f), spacejet_centre_offset_);
            };
            glm::mat4 model      = build_model(f.pos,      f.fwd,      f.up);
            glm::mat4 prev_model = build_model(f.prev_pos, f.prev_fwd, f.prev_up);

            // Red hit-flash: emissive bumps to bright red for the
            // hit_flash_t window (~0.2 s) so the player sees damage
            // register. Destroying flights also flash continuously
            // until they're removed at the end of the fade.
            const float flash = std::max(f.destroying ? 0.6f : 0.0f,
                                          glm::clamp(f.hit_flash_t / 0.20f,
                                                     0.0f, 1.0f));
            const glm::vec4 hit_emissive(2.8f * flash, 0.05f * flash,
                                          0.05f * flash, 0.0f);
            // Tint the base colour reddish while flashing too вЂ” emissive
            // alone reads as glow but the body still looks neutral; the
            // colour mix sells it as "the plane is on fire."
            const glm::vec4 hit_color = glm::mix(glm::vec4(1.0f),
                                                  glm::vec4(1.6f, 0.4f, 0.4f, 1.0f),
                                                  flash);
            for (const auto& sm : spacejet_meshes_) {
                VkDeviceSize off = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &sm.mesh.vertex_buffer, &off);
                vkCmdBindIndexBuffer(cmd, sm.mesh.index_buffer, 0,
                                      VK_INDEX_TYPE_UINT32);
                PushConstants pc{};
                pc.mvp      = vp * model;
                pc.model    = model;
                pc.prev_mvp = prev_vp * prev_model;
                pc.color    = sm.base_color * hit_color;
                pc.emissive = hit_emissive;
                pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
                vkCmdPushConstants(cmd, pipeline_layout_,
                                   kPushConstantStages,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, sm.mesh.index_count, 1, 0, 0, 0);
            }
        }
        // Restore cube mesh binding for the grass / dust passes that
        // assume the cube vertex/index buffer is current.
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    prof_mark(cmd, "w_jets");
    // Grass Р Р†Р вЂљРІР‚Сњ instanced billboard blades. Single draw call covers
    // every blade; the grass.vert collapses out-of-range blades to
    // a degenerate triangle (NaN clip space) so no fragment work runs
    // for them. Skipped entirely when disabled in Settings or when
    // build_grass produced an empty set.
    // Raymarched grass takes over the entire grass slot when enabled вЂ”
    // skip the rasterised draw to avoid double-rendering.
    // Raymarched grass is now deferred until AFTER static brushes +
    // dyn-props are drawn, so the alpha blend reads the castle/cube
    // colours instead of the still-cleared sky. See `render_grass_
    // raymarch` call further down.
    if (rt_.grass_enabled && grass_.instance_count > 0 &&
        grass_pipeline_ != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grass_pipeline_);
        // descriptor set is the same scene set, already bound at the
        // top of render_world; no need to rebind.
        VkBuffer  bufs[2]    = { grass_.blade_mesh.vertex_buffer,
                                  grass_.instance_buffer };
        VkDeviceSize offs[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, bufs, offs);
        vkCmdBindIndexBuffer(cmd, grass_.blade_mesh.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        // Grass push constants Р Р†Р вЂљРІР‚Сњ only grass_params is read by grass.vert,
        // the rest are dummies. Time is just the frame number scaled to
        // a sane wind period.
        PushConstants gpc{};
        gpc.mvp = vp;
        gpc.model = glm::mat4(1.0f);
        gpc.prev_mvp = prev_vp;
        gpc.color = glm::vec4(1.0f);
        gpc.emissive = glm::vec4(0.0f);
        gpc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
        float t = static_cast<float>(frame_number_) * 0.016f;
        gpc.grass_params = glm::vec4(rt_.grass_distance, rt_.grass_wind, t, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &gpc);
        // Density slider goes 0..4. We map (density / 4.0) onto the
        // [0, total_placed] range so density=1.0 gives 25% of the
        // placed blades (the original visual default after the 2M
        // bump) and density=4.0 unleashes the whole pile. Halton
        // placement order means any prefix is a uniformly sparser
        // sampling of the field.
        const float frac = std::clamp(rt_.grass_density, 0.0f, 4.0f) * 0.25f;
        // Per-cell draws: skip cells beyond grass distance (XZ distance to
        // the cell AABB is <= the shader's 3D distance, so this is
        // conservative), outside the altitude band grass.vert fades over,
        // or outside the view frustum. The per-cell prefix keeps the
        // density slider's uniform-thinning behaviour.
        const float max_d = rt_.grass_distance + 5.0f;
        const glm::vec2 cam_xz(fv.eye_pos.x, fv.eye_pos.z);
        const float alt_lo = rt_.grass_alt_min - 2.0f;
        const float alt_hi = std::max(rt_.grass_alt_max, rt_.grass_alt_min + 4.0f) + 2.0f;
        for (const auto& c : grass_.cells) {
            uint32_t n = static_cast<uint32_t>(std::ceil(static_cast<float>(c.count) * frac));
            if (n == 0) continue;
            glm::vec2 q = glm::clamp(cam_xz, glm::vec2(c.aabb_min.x, c.aabb_min.z),
                                     glm::vec2(c.aabb_max.x, c.aabb_max.z));
            if (glm::length(q - cam_xz) > max_d) continue;
            if (c.aabb_max.y < alt_lo || c.aabb_min.y > alt_hi) continue;
            if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
            vkCmdDrawIndexed(cmd, grass_.blade_mesh.index_count,
                             std::min(n, c.count), 0, 0, c.first);
        }
        // Restore the world pipeline + cube mesh bindings for the
        // brush loop below.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    prof_mark(cmd, "w_jets_grass");
    // Ground clutter (AAA Phase 3): one instanced draw of the lumpy-rock
    // mesh. Independent of which grass path is active вЂ” clutter is what
    // breaks the bare-heightfield look between blades.
    if (scatter_.instance_count > 0 && scatter_pipeline_ != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scatter_pipeline_);
        VkBuffer  sbufs[2]    = { scatter_.rock_mesh.vertex_buffer,
                                  scatter_.instance_buffer };
        VkDeviceSize soffs[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, sbufs, soffs);
        vkCmdBindIndexBuffer(cmd, scatter_.rock_mesh.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        PushConstants spc{};
        spc.mvp = vp;
        spc.model = glm::mat4(1.0f);
        spc.prev_mvp = prev_vp;
        // .x = clutter draw distance вЂ” reuse the grass distance slider
        // so near-field density follows the user's grass preference.
        spc.grass_params = glm::vec4(std::max(60.0f, rt_.grass_distance * 1.6f),
                                     0.0f, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                           0, sizeof(PushConstants), &spc);
        vkCmdDrawIndexed(cmd, scatter_.rock_mesh.index_count,
                         scatter_.instance_count, 0, 0, 0);
        // Flowers + ferns: per-species mesh, instance ranges in one buffer.
        if (scatter_.plant_buffer && scatter_plant_pipeline_) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scatter_plant_pipeline_);
            PushConstants ppc = spc;
            ppc.grass_params = glm::vec4(std::max(45.0f, rt_.grass_distance * 0.9f), 1.0f,
                                         static_cast<float>(frame_number_) * 0.016f, 0.0f);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                               0, sizeof(PushConstants), &ppc);
            for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
                if (scatter_.plant_count[sp] == 0) continue;
                VkBuffer pb[2] = { scatter_.plant_mesh[sp].vertex_buffer, scatter_.plant_buffer };
                VkDeviceSize po[2] = { 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 2, pb, po);
                vkCmdBindIndexBuffer(cmd, scatter_.plant_mesh[sp].index_buffer, 0,
                                     VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, scatter_.plant_mesh[sp].index_count,
                                 scatter_.plant_count[sp], 0, 0, scatter_.plant_first[sp]);
            }
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    prof_mark(cmd, "w_scatter");
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.render_aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) { ++culled_static; continue; }
        const glm::mat4& model = static_brush_models_[i];   // pre-baked, see init_world
        glm::vec4 brush_base = tex_on ? b.color : b.fallback_color;
        // Static brushes don't move Р Р†Р вЂљРІР‚Сњ prev_model = current.
        draw_brush(model, model, brush_base, b.emissive, b.full_emissive,
                   b.tex_albedo, b.tex_normal, b.uv_scale, false,
                   i < brush_face_mask_.size() ? brush_face_mask_[i] : -1);
        ++drawn_static;
    }
    if (!castle_meshes_.empty()) {
        draw_castle_meshes(cmd, vp, prev_vp, frustum, false, VK_NULL_HANDLE);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    if (!aa_.empty()) {
        draw_aa(cmd, vp, prev_vp, frustum, false, VK_NULL_HANDLE);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    if (veg_count_ > 0) {
        draw_vegetation(cmd, false);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) { ++culled_dyn; continue; }
        // dr.model / dr.prev_model: world * scale(full_size) baked once
        // per frame in rebuild_dyn_render_cache. Was per-draw glm::scale
        // + 2 mat4 muls here (and again in depth/shadow/TLAS sites).
        glm::vec4 dyn_base = tex_on
            ? dyn_props_[i].color
            : dyn_props_[i].fallback_color;
        draw_brush(dr.model, dr.prev_model, dyn_base, glm::vec3(0.0f), false,
                   dyn_props_[i].tex_albedo, dyn_props_[i].tex_normal,
                   dyn_props_[i].uv_scale, /*object_space=*/true);
        ++drawn_dyn;
    }

    prof_mark(cmd, "w_brushes_dyn");
    // SPOM silhouette shell pass DISABLED. Iterations explored:
    //   1. cube.vert per-face extrusion (offset bug on walls)
    //   2. Paper SSDM screen-space remap (paper assumes outward bumps;
    //      our inward-recessed bricks give barely-visible extension)
    //   3. Shell mesh, normal-push only (corner gaps show sky)
    //   4. Shell mesh + lateral push (corners fill but shell overdraws
    //      walls with simpler NВ·L lighting в†’ "weird shadows")
    // Real dramatic silhouette extension requires either stencil-based
    // masking (touches every depth-aware pipeline) or re-authored
    // outward-bump height maps. Leaving render_shell_pass / shell
    // pipeline in the codebase so re-enabling is one-line.
    // render_shell_pass(cmd);

    // Bullet-impact decals Р Р†Р вЂљРІР‚Сњ drawn AFTER static + dyn brushes so they sit
    // on top of the wall textures, BEFORE sparks so the spark glow
    // overlaps the scorch mark visually. Uses the still-bound cube mesh.
    if (!decals_.empty()) {
        draw_decals(cmd, vp);
    }

    // Hit sparks (raster path).
    if (physics_ && !particles_.empty()) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        draw_spark_trails(cmd, vp);
    }
    // Shadow debug overlay (gated by rt_.shadow_debug_overlay).
    draw_shadow_debug(cmd, vp);

    // Laser bolts + tracer streaks are drawn as glow ribbons in the FX
    // pass (fx.cpp, after deferred lighting). Falling bombs: dark body
    // with a lighter nose band, aligned to the velocity.
    // Flag poles (cloth itself is drawn in the FX pass).
    if (!flags_.empty()) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        for (const auto& f : flags_) {
            auto push = [&](glm::vec3 c, float r, float hl, glm::vec4 col) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), c) *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(r, hl, r));
                PushConstants pc{};
                pc.mvp = vp * model;
                pc.model = model;
                pc.prev_mvp = prev_vp * model;
                pc.color = col;
                pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f);
                vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
            };
            push(f.pole_base + glm::vec3(0.0f, f.pole_h * 0.5f, 0.0f), 0.045f, f.pole_h * 0.5f,
                 glm::vec4(0.22f, 0.16f, 0.10f, 1.0f));                 // wooden pole
            push(f.pole_base + glm::vec3(0.0f, f.pole_h + 0.06f, 0.0f), 0.08f, 0.06f,
                 glm::vec4(0.75f, 0.55f, 0.15f, 1.0f));                 // gilt finial
        }
    }
    if (!bombs_.empty()) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        for (const auto& b : bombs_) {
            glm::vec3 d = glm::normalize(b.vel + glm::vec3(1e-4f));
            auto push = [&](glm::vec3 c, float r, float hl, glm::vec4 col) {
                glm::mat4 model = align_local_y_to(c, d) *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(r, hl, r));
                PushConstants pc{};
                pc.mvp = vp * model;
                pc.model = model;
                pc.prev_mvp = prev_vp * model;   // camera motion (was zero -> TAA smear)
                pc.color = col;
                pc.emissive = glm::vec4(0.0f);
                pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f);
                vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
            };
            push(b.pos, 0.22f, 0.85f, glm::vec4(0.16f, 0.17f, 0.15f, 1.0f));
            push(b.pos + d * 0.95f, 0.16f, 0.12f, glm::vec4(0.55f, 0.45f, 0.12f, 1.0f));
            push(b.pos - d * 0.95f, 0.30f, 0.08f, glm::vec4(0.12f, 0.12f, 0.12f, 1.0f));
        }
    }

    prof_mark(cmd, "w_fx");
    // Raymarched grass вЂ” deferred to here so the alpha blend reads
    // the actual castle/cube/dyn-prop colours instead of the still-
    // cleared sky background. Was running before the brush loops,
    // which made translucent blades in front of the castle show sky
    // colour through them.
    if (rt_.grass_enabled && rt_.grass_raymarch_enabled &&
        grass_rm_pipeline_ != VK_NULL_HANDLE) {
        render_grass_raymarch(cmd);
        // Restore the world cube pipeline + cube mesh for the
        // viewmodel draw immediately below.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0,
                              VK_INDEX_TYPE_UINT32);
    }

    // Viewmodel: rendered last so it overdraws world geometry only when its
    // depth wins. Its depth values are tiny (close to camera), so it sits in
    // front of everything in practice.
    // fv.inv_view is cached in compute_frame_view (was a per-frame
    // glm::inverse(fv.view) here -- a full 4x4 mat inverse is ~80 fmuls).
    prof_mark(cmd, "w_grass_rm");
    draw_viewmodel(cmd, vp, fv.inv_view);
    prof_mark(cmd, "w_viewmodel");

    // (Voxel buildings were drawn earlier, before the water raster, so
    // the water plane's alpha blend correctly tints submerged voxel
    // pixels. See the comment + call near the start of render_world's
    // water block.)

    last_draw_static_ = drawn_static;
    last_draw_dyn_    = drawn_dyn;
    last_culled_      = culled_static + culled_dyn;
}

} // namespace qlike

namespace qlike {

void VulkanEngine::compute_brush_face_masks() {
    const auto& bs = world_.brushes;
    brush_face_mask_.assign(bs.size(), 0);
    constexpr float kTouch = 0.02f;   // faces within 2 cm count as touching
    constexpr float kCovered = 0.6f;  // >= 60 % of the face area covered -> hidden
    int exposed_total = 0;
    for (size_t i = 0; i < bs.size(); ++i) {
        const glm::vec3 lo = bs[i].center - bs[i].size * 0.5f;
        const glm::vec3 hi = bs[i].center + bs[i].size * 0.5f;
        int mask = 0;
        for (int a = 0; a < 3; ++a) {
            const int u = (a + 1) % 3, v = (a + 2) % 3;
            const float face_area = bs[i].size[u] * bs[i].size[v];
            for (int s = 0; s < 2; ++s) {
                const float plane = s ? hi[a] : lo[a];
                float covered = 0.0f;
                for (size_t j = 0; j < bs.size() && face_area > 0.0f; ++j) {
                    if (j == i) continue;
                    const glm::vec3 jlo = bs[j].center - bs[j].size * 0.5f;
                    const glm::vec3 jhi = bs[j].center + bs[j].size * 0.5f;
                    // j must reach across the face plane and extend past it
                    // on the outside by more than the touch tolerance.
                    const bool spans = s ? (jlo[a] <= plane + kTouch && jhi[a] >= plane + kTouch)
                                         : (jhi[a] >= plane - kTouch && jlo[a] <= plane - kTouch);
                    if (!spans) continue;
                    const float ou = std::min(hi[u], jhi[u]) - std::max(lo[u], jlo[u]);
                    const float ov = std::min(hi[v], jhi[v]) - std::max(lo[v], jlo[v]);
                    if (ou > 0.0f && ov > 0.0f) covered += ou * ov;
                }
                if (covered < kCovered * face_area) {
                    mask |= 1 << (a * 2 + s);
                    ++exposed_total;
                }
            }
        }
        brush_face_mask_[i] = mask;
    }
    log::infof("[spom] box relief: %d/%zu brush faces exposed", exposed_total,
               bs.size() * 6);
}

} // namespace qlike
