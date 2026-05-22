// Scene descriptor set + UBO update. The TLAS / materials / prev_transforms
// SSBOs and per-texture sampler arrays all live behind one descriptor set
// (scene_desc_set_) bound by every world raster pipeline. Once init_rt()
// produces the TLAS handle, helpers.cpp's write_scene_descriptors_once()
// fills the bindings; from then on update_scene_ubo() is the only per-frame
// descriptor work (memcpy into a host-mapped UBO buffer).

#include "engine/vk_engine/internal.h"

#include <cmath>
#include <cstring>

namespace qlike {

void VulkanEngine::init_descriptors() {
    // Scene UBO buffer.
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(SceneUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &scene_ubo_buffer_, &scene_ubo_alloc_, nullptr),
                 "scene ubo buffer");
    }

    // Descriptor pool: 1 UBO + 1 AS + 2 SSBO (materials, prev_transforms) +
    // 2 sampler arrays (albedo[N], normal[N]).
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        // 2 originals (materials, prev_transforms) + 2 ReSTIR reservoirs.
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        // SVGF GI denoiser storage images: 1 raw gi (binding 19) +
        // 2 history ping-pong (bindings 20, 21).
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        // +1 for the heightmap shadow texture at binding 6.
        // +1 for the sun shadow map at binding 7 (sampler2DShadow).
        // +1 for the raw heightmap texture at binding 8 (R32_SFLOAT) —
        // sampled by terrain_raymarch.frag so the procedural raymarched
        // terrain follows the gameplay heightmap shape.
        // +3 for the low-res raymarch upscale targets (bindings 9..11):
        // color, motion vector, depth — sampled by the compose pass
        // when terrain_raymarch_scale < 1.0.
        // +kSpomMaterialCount for the SPOM height-map array (binding 12).
        // +1 for the grass-density mask R8 texture (binding 13).
        // +1 for the fog wisp R8 texture (binding 14).
        // +1 (now 9) for the FULL terrain-height R32F texture at
        // binding 17 — the water path samples the real mesh surface.
        // +1 (now 10) for the half-rate shadow R8 image at binding 18
        // — cube.frag's bilateral upsample consumer (roadmap item #4
        // Phase 3) samples it in place of the inline PCSS trace.
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            kTextureCount * 2 + 10 + kSpomMaterialCount },
    };
    VkDescriptorPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 5,
        .pPoolSizes = sizes,
    };
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &scene_desc_pool_),
             "scene desc pool");

    // Bindings: 0=UBO, 1=TLAS, 2=materials, 3=albedo[N], 4=normal[N],
    //           5=prev_transforms, 6=heightmap shadow, 7=sun shadow map,
    //           8=raw heightmap (R32_SFLOAT) for terrain_raymarch.frag,
    //           9=LR raymarch color, 10=LR raymarch motion-vec, 11=LR
    //              raymarch depth — read by the compose pass when
    //              terrain_raymarch_scale < 1.
    //          15=ReSTIR reservoir SSBO (read prev), 16=write cur.
    //              Owned by restir.cpp; cube.frag's GI loop reads/writes.
    VkDescriptorSetLayoutBinding bindings[22]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = kTextureCount;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = kTextureCount;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 6: pre-baked heightmap sun-shadow R8 texture. Sampled by
    // grass.vert AND cube.frag — the latter uses it as a distant-terrain
    // shadow fallback so the BLAS-vs-LOD-raster mismatch can't produce
    // false-hit shadow rays on far ridges.
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 7: sun shadow map (single-cascade D32). Sampled by grass.vert
    // as a sampler2DShadow for hardware PCF + comparison. Replaces the
    // heightmap-bake path going forward (binding 6 stays for fallback).
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 8: raw heightmap as R32_SFLOAT, sampled by
    // terrain_raymarch.frag so the procedural ray-marched terrain
    // follows the gameplay heightmap shape exactly (castle / cubes
    // sit correctly on the visible surface).
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 9..11: low-res raymarch upscale targets, sampled by the
    // compose pass when terrain_raymarch_scale < 1. Color and motion
    // are HDR / SFLOAT; depth is R32 sampled as a regular texture so
    // the compose can write gl_FragDepth = upscaled depth.
    for (int b = 9; b <= 11; ++b) {
        bindings[b].binding = static_cast<uint32_t>(b);
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    // Binding 12: SPOM displacement map array. cube.frag samples one entry
    // per parallax-using material (currently 4: castle walls, keep walls,
    // courtyard floor, keep interior floor). Array index = SPOM material
    // ID returned by cube.frag::height_idx_for_albedo().
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[12].descriptorCount = kSpomMaterialCount;
    bindings[12].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 13: grass-density mask (R8). Both the raymarched grass
    // pass and the terrain raymarch's getMaterial sample this for
    // eligibility — replaces the 9-cell noised() storm and ensures
    // grass blades and green ground tint always agree.
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 14: fog wisp pattern (R8 256², REPEAT). One textureLod()
    // replaces the 3-octave noise2 calls in terrain_raymarch.frag's
    // fog density + godray probe loops.
    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 15/16: ReSTIR GI reservoir SSBOs (ping-pong). Session 2
    // ships the bindings + writes; session 3 (temporal) starts reading
    // binding 15. Layout matches the GLSL Reservoir struct in cube.frag.
    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 17: FULL baked terrain height (R32_SFLOAT, dim+1²) — the
    // exact array the CDLOD mesh + physics use. terrain_raymarch.frag's
    // water path samples this in mesh-terrain mode so shore /
    // showthrough / reflection / self-shadow follow the real mesh
    // surface with zero procedural FBM evaluation.
    bindings[17].binding = 17;
    bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[17].descriptorCount = 1;
    bindings[17].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 18: half-rate sun-shadow R8 image (roadmap item #4
    // Phase 3). Produced by render_world_shadow_lr_pass at
    // render_extent_/2; consumed by cube.frag's bilateral upsample
    // for brush/dyn surfaces when rt_.half_rate_shadows is on. Written
    // by init_shadow_lr_pipeline and recreate_shadow_lr.
    bindings[18].binding = 18;
    bindings[18].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[18].descriptorCount = 1;
    bindings[18].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 19: SVGF GI denoiser storage image. cube.frag writes raw
    // per-pixel GI irradiance via imageStore; sessions 2+ (in
    // docs/svgf_plan.md) add the temporal-accumulation and à-trous
    // passes that read it. R16G16B16A16F at render_extent_.
    bindings[19].binding = 19;
    bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[19].descriptorCount = 1;
    bindings[19].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 20 / 21: SVGF history ping-pong pair (Session 2). cube.frag
    // selects read vs write by frame parity (scene.rt_flags.w & 1) —
    // same race-free model as the ReSTIR ring buffer.
    bindings[20].binding = 20;
    bindings[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[20].descriptorCount = 1;
    bindings[20].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[21].binding = 21;
    bindings[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[21].descriptorCount = 1;
    bindings[21].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .bindingCount = 22, .pBindings = bindings,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                         &scene_desc_set_layout_),
             "scene desc set layout");

    VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = scene_desc_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &scene_desc_set_layout_,
    };
    vk_check(vkAllocateDescriptorSets(device_, &ai, &scene_desc_set_),
             "scene desc set alloc");
    log::info("descriptors created");
}

void VulkanEngine::destroy_descriptors() {
    if (scene_desc_set_layout_) vkDestroyDescriptorSetLayout(device_, scene_desc_set_layout_, nullptr);
    if (scene_desc_pool_)       vkDestroyDescriptorPool(device_, scene_desc_pool_, nullptr);
    if (scene_ubo_buffer_)      vmaDestroyBuffer(allocator_, scene_ubo_buffer_, scene_ubo_alloc_);
    scene_desc_set_layout_ = VK_NULL_HANDLE;
    scene_desc_pool_ = VK_NULL_HANDLE;
    scene_desc_set_ = VK_NULL_HANDLE;
    scene_ubo_buffer_ = VK_NULL_HANDLE;
    scene_ubo_alloc_ = nullptr;
}

void VulkanEngine::update_scene_ubo() {
    // Per-frame: only update the UBO contents (memcpy to mapped buffer).
    // Descriptor set bindings are written once after init_rt().
    SceneUBO data{};
    float pitch = glm::radians(rt_.sun_pitch_deg);
    float yaw   = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::cos(pitch) * std::sin(yaw),
                  std::sin(pitch),
                  std::cos(pitch) * std::cos(yaw));
    sun = glm::normalize(sun);
    data.sun_direction = glm::vec4(sun, 0.0f);
    // Auto golden-hour tint. When the sun is low (pitch ≤ 20° elevation),
    // smoothly bias the sun colour toward warm orange and the sky toward
    // pinkish-amber — the standard sunset/sunrise look. Drives every
    // shader that reads scene.sun_color / scene.sky_color (cube,
    // terrain_raymarch, grass, water) from a single uniform source.
    // Toggle via rt_.auto_golden_hour; user's manual sun_color / sky_color
    // are still the basis, just modulated by altitude when enabled.
    glm::vec3 sun_rgb = rt_.sun_color;
    glm::vec3 sky_rgb = rt_.sky_color;
    if (rt_.auto_golden_hour) {
        float alt = sun.y;                       // sin(pitch)
        // 0 at zenith (alt≥0.6, ~37°), 1 at horizon (alt≤0.10, ~6°).
        float t = glm::clamp((0.60f - alt) / 0.50f, 0.0f, 1.0f);
        // Warm sunset tint: orange-amber at horizon, peach at ~15°.
        // No brightness boost — just shift the spectrum. The previous
        // 1.35x lift over-cranked the sun and washed everything out.
        glm::vec3 warm     = glm::vec3(1.00f, 0.55f, 0.25f);
        glm::vec3 sky_warm = glm::vec3(0.95f, 0.50f, 0.40f);
        sun_rgb = glm::mix(sun_rgb, sun_rgb * warm, t);
        sky_rgb = glm::mix(sky_rgb, sky_warm,       t * 0.85f);
    }
    data.sun_color     = glm::vec4(sun_rgb, rt_.sun_intensity);
    data.ambient       = glm::vec4(rt_.ground_ambient, rt_.terrain_ao_punch);
    data.sky_color     = glm::vec4(sky_rgb, 1.0f);
    data.rt_flags = glm::ivec4(rt_.shadow_enabled ? 1 : 0,
                               rt_.shadow_samples,
                               rt_.ao_samples,
                               static_cast<int>(frame_number_));
    data.rt_params = glm::vec4(rt_.shadow_softness,
                               rt_.ao_radius,
                               rt_.ambient_strength,
                               rt_.shadow_strength);
    // ReSTIR-lite: when on, hand the shader gi_samples / 4 (rounded down,
    // floored at 1 if gi_samples > 0). The 4x reduction is the visible
    // perf win for session 1; sessions 2+ replace this with the actual
    // reservoir-resampled algorithm — see docs/restir_plan.md.
    int gi_samples_eff = rt_.gi_samples;
    if (rt_.gi_restir_enabled && gi_samples_eff > 0) {
        gi_samples_eff = std::max(1, gi_samples_eff / 4);
    }
    data.rt_flags2 = glm::ivec4(gi_samples_eff,
                                rt_.reflections_enabled ? 1 : 0,
                                rt_.gi_bounces,
                                rt_.ao_mode);  // 0=off, 1=fast, 2=RTAO
    // ReSTIR session 3+4 (temporal + spatial reservoir blend) was
    // quarantined after shipping (visible wobble). Session 5 fixed both
    // root causes and re-enables it (see the restir_params line below).
    //
    // The session-1 gi_samples / 4 reduction also stays in effect when
    // rt_.gi_restir_enabled (see gi_samples_eff above): ReSTIR converges
    // the quartered sample count back to full quality via temporal +
    // spatial reservoir reuse.
    //
    // restir_params packing:
    // .x: ReSTIR enabled (rt_.gi_restir_enabled). Consumed ONLY by the
    //     two reservoir gates in cube.frag — verified no other shader
    //     reads restir_params.x, so this is independent of FSR.
    // .y: M_max (sample-count cap; bounds temporal variance/lag).
    // .zw: jitter UV delta (cur−prev) for FSR3 motion-vec jitter
    //      cancellation in cube.frag and terrain_raymarch.frag.
    //      Only non-zero when FSR is on (TAA expects jittered motion
    //      vec, would freeze if subtracted unconditionally).
    glm::vec2 jdelta_uv(0.0f, 0.0f);
    if (rt_.fsr2_enabled) {
        jdelta_uv = (current_frame_view_.jitter - prev_jitter_) * 0.5f;
    }
    // .x re-enabled in ReSTIR session 5. It is consumed ONLY by the two
    // temporal/spatial reservoir gates in cube.frag (verified — no other
    // shader reads restir_params.x), so driving it from the user toggle
    // is independent of FSR. The session 3+4 quarantine root causes are
    // now fixed: (a) the single-buffer aliasing race is gone — the
    // reservoir SSBO is a 3-region ring indexed by frame%3 (race-free
    // under kFrameOverlap=2; see restir.cpp), and (b) disocclusion is
    // depth-aware — cube.frag carries prev cam-distance in the reservoir
    // pad slot and rejects reprojected/neighbour samples whose surface
    // is too far in normal OR depth. .z is no longer the (jitter-
    // polluted) normal threshold; cube.frag uses fixed constants.
    float restir_on = rt_.gi_restir_enabled ? 1.0f : 0.0f;
    data.restir_params = glm::vec4(restir_on, 32.0f, jdelta_uv.x, jdelta_uv.y);
    // .x = SPOM bump-depth. .yzw = procedural ground splat knobs
    // (free slots — no UBO layout change): y = material strength,
    // z = metres per material repeat, w = detail-normal strength.
    data.spom_params   = glm::vec4(std::max(0.0f, rt_.spom_strength),
                                    glm::clamp(rt_.ground_mat_strength, 0.0f, 1.0f),
                                    std::max(0.25f, rt_.ground_mat_tile_m),
                                    glm::clamp(rt_.ground_mat_normal, 0.0f, 1.0f));
    // .x = water-only flag for the terrain_raymarch pass: 1.0 when
    // the mesh terrain is the active ground (raymarch terrain OFF) but
    // water is enabled — the pass then skips the terrain march and
    // only renders the water surface on top of the rasterised mesh.
    // .z = half-rate-shadow toggle for cube.frag (Phase 3 consumer).
    //      When 1, brush/dyn surfaces skip the inline blocker+PCSS
    //      block and bilinear-sample u_shadow_lr (binding 18) instead.
    // .w = SVGF GI denoiser enable (Session 2). When >0.5 cube.frag
    //      EMA-blends shade_radiance with the reprojected history
    //      sample from binding 20/21 (ping-pong by frame parity).
    data.terrain_local_info = glm::vec4(
        (!rt_.terrain_raymarch_enabled && rt_.water_enabled) ? 1.0f : 0.0f,
        terrain_height_max_ + 5.0f,   // .y = mesh max height (air early-out)
        rt_.half_rate_shadows ? 1.0f : 0.0f,
        rt_.svgf_enabled ? 1.0f : 0.0f);
    // Copy the baked 32×32 hi-Z max-cell grid into the UBO (1024
    // floats → 256 vec4). The grid is static after level load (baked
    // in init_world from terrain_data_.heights); when it isn't ready
    // yet, write a high constant so the shader's cell-skip never
    // wrongly skips real terrain on the first frames.
    if (terrain_max_grid_ready_) {
        std::memcpy(data.terrain_max_grid, terrain_max_grid_.data(),
                    sizeof(float) * 1024);
    } else {
        for (int i = 0; i < 256; ++i)
            data.terrain_max_grid[i] = glm::vec4(1.0e9f);
    }
    data.rt_params2 = glm::vec4(rt_.gi_strength,
                                rt_.gi_radius,
                                rt_.reflection_strength,
                                rt_.shadow_curve);
    // Use the lerped eye captured by compute_frame_view, NOT the raw
    // physics-tick player_.eye_position(). Mismatch caused raymarched
    // grass/terrain to render one frame behind rasterised geometry —
    // their ray origin came from the raw eye while their projection
    // matrix used the smoothed render_pos.
    glm::vec3 eye = current_frame_view_.eye_pos;
    data.camera_pos = glm::vec4(eye, 0.0f);
    // Distance LOD: 15 m = full samples (gameplay-relevant range — shadows
    // and AO are most visible up close); past 50 m fades to 1 sample so the
    // skybox-distance cubes don't waste a full ray budget per pixel.
    // .z carries gi_shadow_max_bounce — see RtSettings; cast to int in
    // cube.frag where it gates the per-bounce sun shadow ray.
    // .z = gi_shadow_max_bounce (cube.frag casts to int).
    // .w = ao_floor — see RtSettings; cube.frag remaps raw AO to
    //      [ao_floor, 1.0] so corner darkness doesn't compound.
    data.rt_lod = glm::vec4(15.0f, 50.0f,
                             static_cast<float>(rt_.gi_shadow_max_bounce),
                             rt_.ao_floor);
    {
        // cube.frag renders at render_extent_ (so does its motion-vec output);
        // gl_FragCoord.xy / viewport.zw is in render-resolution UV space.
        float w = static_cast<float>(render_extent_.width);
        float h = static_cast<float>(render_extent_.height);
        data.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
    }
    {
        // Muzzle flash — origin a short distance ahead of the eye so it
        // doesn't get trapped inside the player capsule. Intensity falls
        // linearly from 1 at fire to 0 over kMuzzleFlashDuration so the
        // pop is sharp and brief. Color is warm and quite hot — at intensity
        // 1.0 a single shot lights nearby walls visibly through RT shadows.
        if (muzzle_flash_timer_ > 0.0f) {
            float t = std::min(1.0f, muzzle_flash_timer_ / kMuzzleFlashDuration);
            glm::vec3 fwd = player_.forward();
            glm::vec3 pos = player_.eye_position() + fwd * 0.35f;
            float intensity = 14.0f * t;       // strong HDR pop
            data.muzzle_pos   = glm::vec4(pos, intensity);
            data.muzzle_color = glm::vec4(1.00f, 0.85f, 0.55f, 6.0f);  // 6 m radius
        } else {
            data.muzzle_pos   = glm::vec4(0.0f);
            data.muzzle_color = glm::vec4(0.0f);
        }
    }
    data.terrain_params = glm::vec4(rt_.terrain_fog_strength,
                                    rt_.terrain_wrap_strength,
                                    rt_.terrain_detail_strength,
                                    rt_.terrain_shadow_softness_scale);
    data.terrain_h_low  = glm::vec4(rt_.terrain_h_sand_grass_start,
                                    rt_.terrain_h_sand_grass_end,
                                    rt_.terrain_h_grass_dirt_start,
                                    rt_.terrain_h_grass_dirt_end);
    data.terrain_h_high = glm::vec4(rt_.terrain_h_dirt_rock_start,
                                    rt_.terrain_h_dirt_rock_end,
                                    rt_.terrain_h_rock_snow_start,
                                    rt_.terrain_h_rock_snow_end);
    data.grass_extra = glm::vec4(rt_.grass_height_scale,
                                 rt_.grass_alpha_cutoff,
                                 rt_.grass_slope_n_min,
                                 rt_.grass_distance_density);
    data.grass_extra2 = glm::vec4(rt_.grass_alt_min,
                                  rt_.grass_alt_max,
                                  rt_.shadow_map_world_half,
                                  static_cast<float>(rt_.terrain_debug_mode));
    data.light_vp = sun_shadow_light_vp_;
    data.terrain_extra = glm::vec4(rt_.terrain_shading_contrast,
                                    rt_.shadow_near_mult,
                                    rt_.gi_softener,
                                    rt_.gi_debug_viz ? 1.0f : 0.0f);
    // Ocean / water plane params. Time is derived from frame number
    // at a nominal 60 Hz — exact wall-clock isn't needed for waves
    // and avoids threading a real timer here.
    float water_t = static_cast<float>(frame_number_) * (1.0f / 60.0f);
    data.water_params = glm::vec4(rt_.water_enabled ? 1.0f : 0.0f,
                                   rt_.water_level,
                                   rt_.water_wave_strength,
                                   water_t);
    // water_color_shallow.w doubles as the wave-frequency multiplier
    // (was unused). Read on the shader side as wave_scale.
    data.water_color_shallow.w = std::max(0.05f, rt_.water_wave_scale);
    // water_color.w doubles as the "shadows enabled" flag (0 / 1).
    data.water_color  = glm::vec4(rt_.water_color,
                                   rt_.water_shadows_enabled ? 1.0f : 0.0f);
    data.water_color_shallow = glm::vec4(rt_.water_color_shallow, 0.0f);
    // water_shore.w doubles as the transparency / underwater
    // showthrough strength (0 = opaque, 1 = shallow shows terrain).
    data.water_shore = glm::vec4(rt_.water_shore_blend,
                                  rt_.water_shore_noise,
                                  rt_.water_tlas_reflections ? 1.0f : 0.0f,
                                  rt_.water_transparency);
    data.water_foam_color  = glm::vec4(rt_.water_foam_color,
                                        rt_.water_foam_strength);
    // Water style + river-style knobs packed into water_foam_params.yzw:
    //   y = water_style (int, 0 = default, 1 = river)
    //   z = water_river_speed
    //   w = water_river_normal_str
    // (water_river_extinct_mix lives in water_color_shallow.w — that slot
    //  was wave_scale; we keep both by repurposing only when style >= 1.)
    // foam_params.w is repurposed across styles:
    //   river: river normal_str
    //   lake : lake bump_strength
    float foam_w_slot = (rt_.water_style == 2)
        ? std::max(0.0f, rt_.water_lake_bump_strength)
        : std::max(0.0f, rt_.water_river_normal_str);
    data.water_foam_params = glm::vec4(std::max(0.05f, rt_.water_foam_width),
                                        static_cast<float>(rt_.water_style),
                                        std::max(0.0f, rt_.water_river_speed),
                                        foam_w_slot);
    data.fog_band = glm::vec4(rt_.terrain_raymarch_fog_y_start,
                               rt_.terrain_raymarch_fog_y_top,
                               rt_.terrain_raymarch_fog_noise,
                               rt_.terrain_rt_lod_distance);
    data.terrain_rt_extra = glm::vec4(
        static_cast<float>(rt_.terrain_pcss_samples_cap),
        static_cast<float>(rt_.terrain_gi_samples_cap),
        rt_.terrain_ao_final_strength,
        static_cast<float>(rt_.terrain_gi_bounces_cap));
    data.grass_color_top    = glm::vec4(rt_.grass_color_top,
                                         rt_.grass_density);
    data.grass_color_bottom = glm::vec4(rt_.grass_color_bottom,
                                         rt_.grass_base_ao_floor);
    data.grass_color_ground = glm::vec4(rt_.grass_color_ground,
                                         rt_.grass_ground_tint_strength);
    data.grass_color_ground_far = glm::vec4(rt_.grass_color_ground_far,
                                              std::max(50.0f, rt_.grass_ground_tint_far_distance));
    data.grass_shadow_params = glm::vec4(
        rt_.grass_shadow_strength,
        static_cast<float>(rt_.grass_shadow_samples),
        rt_.grass_shadow_max_dist,
        0.0f);
    // Shoreline grass tint — see SceneUBO comment in internal.h.
    data.grass_shore_color = glm::vec4(rt_.grass_shore_color,
                                        rt_.grass_shore_strength);
    data.grass_shore_params = glm::vec4(std::max(0.05f, rt_.grass_shore_distance),
                                         0.0f, 0.0f, 0.0f);
    // Shoreline TERRAIN tint — same packing.
    data.terrain_shore_color = glm::vec4(rt_.terrain_shore_color,
                                          rt_.terrain_shore_strength);
    data.terrain_shore_params = glm::vec4(std::max(0.05f, rt_.terrain_shore_distance),
                                           0.0f, 0.0f, 0.0f);
    // Distance fog — strength=0 disables in-shader.
    data.distance_fog_color = glm::vec4(rt_.distance_fog_color,
                                         std::max(0.0f, std::min(1.0f, rt_.distance_fog_strength)));
    data.distance_fog_params = glm::vec4(std::max(0.0f, rt_.distance_fog_density),
                                          std::max(0.0f, rt_.distance_fog_start),
                                          std::max(0.0f, rt_.distance_fog_height),
                                          std::max(0.0f, std::min(1.0f, rt_.distance_fog_max)));
    data.terrain_shore_general_color = glm::vec4(rt_.terrain_shore_general_color,
                                                  rt_.terrain_shore_general_strength);
    data.terrain_shore_general_params = glm::vec4(std::max(0.05f, rt_.terrain_shore_general_distance),
                                                   0.0f, 0.0f, 0.0f);
    data.terrain_sand_color = glm::vec4(rt_.terrain_sand_color, 0.0f);
    {
        const float deg2rad = 3.14159265358979f / 180.0f;
        if (rt_.water_style == 2) {
            // Lake style: pack lake knobs into the same UBO slots.
            //   .x = unused, .y = lake time speed,
            //   .z = lake uv scale, .w = unused
            data.water_river_extra = glm::vec4(
                0.0f,
                std::max(0.0f, rt_.water_lake_time_speed),
                std::max(0.05f, rt_.water_lake_uv_scale),
                0.0f);
            // extinct.w holds the lake's bump-fade distance / 100
            // (shader multiplies by 100 to get metres).
            data.water_river_extinct = glm::vec4(
                rt_.water_river_extinct_color,  // unused by lake shader
                std::max(0.05f, rt_.water_lake_bump_dist) * 0.01f);
        } else {
            data.water_river_extra = glm::vec4(
                rt_.water_river_flow_angle * deg2rad,
                std::max(0.0f, rt_.water_river_time_speed),
                std::max(0.05f, rt_.water_river_detail),
                std::max(0.0f, rt_.water_river_foam_amount));
            data.water_river_extinct = glm::vec4(
                rt_.water_river_extinct_color,
                std::max(0.0f, rt_.water_river_extinct_density));
        }
    }

    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, scene_ubo_alloc_, &ai);
    if (ai.pMappedData) std::memcpy(ai.pMappedData, &data, sizeof(data));
    // Defensive flush — no-op on coherent host memory but required by spec
    // for non-coherent allocations (e.g. some non-NVIDIA drivers).
    vmaFlushAllocation(allocator_, scene_ubo_alloc_, 0, sizeof(data));
}

} // namespace qlike
