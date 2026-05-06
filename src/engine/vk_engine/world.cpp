// World-side state: level + meshes + skybox + texture array + viewmodel +
// dyn-prop spawning + the per-frame `render_world` raster pass that drives
// every cube/cylinder draw the engine emits. Lives here because all of these
// touch `world_`, `dyn_props_`, `physics_`, the TLAS instance cache (via
// rebuild_dyn_render_cache), and the same pipeline_layout_/scene_desc_set_.

#include "engine/vk_engine/internal.h"
#include "engine/gltf_loader.h"
#include "engine/skybox.h"
#include "engine/terrain.h"
#include "engine/texture.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace qlike {

void VulkanEngine::init_world() {
    cube_mesh_ = create_cube_mesh(device_, allocator_, graphics_queue_, graphics_queue_family_);
    cylinder_mesh_ = create_cylinder_mesh(device_, allocator_,
                                          graphics_queue_, graphics_queue_family_);
    world_ = game::make_arena();
    log::infof("world: %u brushes", static_cast<unsigned>(world_.brushes.size()));

    // Procedural terrain: generated first so the castle can be lifted to
    // sit on the plateau height. Phase 1 is non-streaming — one big mesh
    // (~1km²). See docs/terrain_plan.md for streaming/LOD/sculpt phases.
    {
        HeightmapParams hp{};
        hp.dim = 256;
        hp.cell_size = 4.0f;          // 4m per cell × 256 = 1024m square
        hp.height_scale = 90.0f;
        hp.plateau_extent = glm::vec2(20.0f, 20.0f);
        hp.plateau_height = 14.0f;     // castle base sits at this Y
        hp.plateau_blend  = 16.0f;
        Heightmap hm = generate_heightmap(hp);
        // Cache for collision + Phase 4 sculpt access.
        terrain_data_.dim = hm.dim;
        terrain_data_.cell = hm.cell;
        terrain_data_.origin_x = hm.origin_x;
        terrain_data_.origin_z = hm.origin_z;
        terrain_data_.heights = hm.heights;
        terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                           graphics_queue_, graphics_queue_family_, hm);
        // Lift every level brush by plateau_height so the castle's y=0
        // baseline lands on the plateau surface. Cheaper than rewriting
        // level.cpp — the brush AABBs / world matrices update through the
        // same vector.
        for (auto& b : world_.brushes) b.center.y += hp.plateau_height;
        for (auto& a : world_.aabbs)   { a.min.y += hp.plateau_height;
                                          a.max.y += hp.plateau_height; }
        // Lift player's default spawn point above the new ground.
        player_.position.y += hp.plateau_height;
    }

    physics_ = std::make_unique<PhysicsWorld>();
    // Batch-add the level brushes — Jolt's AddBodiesPrepare/Finalize takes
    // the broadphase mutex once for the whole pile instead of per body.
    // For 150+ static brushes (the castle) this is the difference between
    // ~30 ms and ~3 ms at level load.
    std::vector<PhysicsWorld::StaticBox> sboxes;
    sboxes.reserve(world_.brushes.size());
    for (const auto& b : world_.brushes) {
        sboxes.push_back({b.center, b.size * 0.5f});
    }
    physics_->add_static_boxes(sboxes.data(), sboxes.size());
    // Heightfield collision for terrain. Player + crates collide cleanly
    // against the bumpy ground — far cheaper than tessellated brushes.
    if (!terrain_data_.heights.empty()) {
        physics_->add_static_heightfield(
            terrain_data_.heights.data(), terrain_data_.dim,
            glm::vec2(terrain_data_.origin_x, terrain_data_.origin_z),
            terrain_data_.cell);
    }

    log::infof("Jolt: %d static bodies; dynamic boxes will spawn over time "
               "(max %d, every %.2fs)",
               physics_->body_count(),
               kMaxDynProps, kSpawnInterval);
}

void VulkanEngine::init_skybox() {
    // EXR first — keeps real sun radiance so the bloom pass naturally produces
    // a glaring sun disc. Falls back to the LDR JPG if EXR is missing or
    // tinyexr can't decode it. Engine is invoked from build/ on Windows but
    // from the project root in CI, so probe both.
    static const char* kSkyboxCandidates[] = {
        "assets/sky/MorningSkyHDRI002B_4K_HDR.exr",
        "../assets/sky/MorningSkyHDRI002B_4K_HDR.exr",
        "../../assets/sky/MorningSkyHDRI002B_4K_HDR.exr",
        "assets/sky/MorningSkyHDRI002B_4K_TONEMAPPED.jpg",
        "../assets/sky/MorningSkyHDRI002B_4K_TONEMAPPED.jpg",
        "../../assets/sky/MorningSkyHDRI002B_4K_TONEMAPPED.jpg",
    };
    for (const char* path : kSkyboxCandidates) {
        skybox_ = load_skybox(device_, allocator_, graphics_queue_,
                              graphics_queue_family_, path);
        if (skybox_.image != VK_NULL_HANDLE) break;
    }
    if (skybox_.image == VK_NULL_HANDLE) {
        log::info("[skybox] no asset loaded — creating 1x1 placeholder so the "
                  "compose descriptor stays bound");
        // Build a 1x1 black image manually (cheap fallback).
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_SRGB;
        ici.extent = { 1, 1, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci,
                                 &skybox_.image, &skybox_.alloc, nullptr),
                 "skybox dummy image");

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = skybox_.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0; vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0; vci.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device_, &vci, nullptr, &skybox_.view),
                 "skybox dummy view");

        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &skybox_.sampler),
                 "skybox dummy sampler");

        skybox_.width = 1; skybox_.height = 1;
        skybox_.sun_direction = glm::vec3(0.45f, 0.7f, 0.55f);

        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, skybox_.image,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    } else {
        // Real asset loaded — sync the directional-light angles to the
        // brightest pixel in the panorama so cast shadows match the visible
        // sun.
        glm::vec3 d = glm::normalize(skybox_.sun_direction);
        float pitch_deg = glm::degrees(std::asin(glm::clamp(d.y, -1.0f, 1.0f)));
        float yaw_deg = glm::degrees(std::atan2(d.x, d.z));
        rt_.sun_pitch_deg = pitch_deg;
        rt_.sun_yaw_deg = yaw_deg;
        log::infof("[skybox] sun synced to UI: pitch=%.1f° yaw=%.1f°",
                   pitch_deg, yaw_deg);
    }
}

void VulkanEngine::destroy_skybox_resources() {
    destroy_skybox(device_, allocator_, skybox_);
}

void VulkanEngine::init_textures() {
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.anisotropyEnable = VK_FALSE;
        // VK_LOD_CLAMP_NONE lets the sampler pick from the entire mip chain
        // texture.cpp generates — eliminates far-distance moiré and halves
        // average sample bandwidth on receding floors/walls.
        si.maxLod = VK_LOD_CLAMP_NONE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &texture_sampler_),
                 "texture sampler");
    }

    auto probe = [&](const std::string& tail, VkFormat fmt) -> TextureSlot {
        const std::string roots[] = {
            "assets/tex/" + tail,
            "../assets/tex/" + tail,
            "../../assets/tex/" + tail,
        };
        for (const std::string& p : roots) {
            Texture2D r = upload_texture_from_file(
                device_, allocator_, graphics_queue_,
                graphics_queue_family_, p, fmt);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                return s;
            }
        }
        log::warnf("[texture] %s missing on every probe path", tail.c_str());
        return {};
    };

    struct Spec {
        const char* name;
        const char* color_jpg;
        const char* normal_jpg;
    };
    // Index → category. Indices ALSO match what level.cpp / spawn_random_box
    // pass into Brush::tex_albedo, so reordering here means reordering there.
    const Spec specs[kTextureCount] = {
        { "Ground054",         "Ground054/Ground054_2K-JPG_Color.jpg",
                               "Ground054/Ground054_2K-JPG_NormalGL.jpg" },
        { "Bricks067",         "Bricks067/Bricks067_2K-JPG_Color.jpg",
                               "Bricks067/Bricks067_2K-JPG_NormalGL.jpg" },
        { "Wood048",           "Wood048/Wood048_2K-JPG_Color.jpg",
                               "Wood048/Wood048_2K-JPG_NormalGL.jpg" },
        { "Metal042A",         "Metal042A/Metal042A_2K-JPG_Color.jpg",
                               "Metal042A/Metal042A_2K-JPG_NormalGL.jpg" },
        { "PaintedPlaster017", "PaintedPlaster017/PaintedPlaster017_2K-JPG_Color.jpg",
                               "PaintedPlaster017/PaintedPlaster017_2K-JPG_NormalGL.jpg" },
    };

    for (int i = 0; i < kTextureCount; ++i) {
        // Albedos go through an sRGB sampler so the GPU does the gamma decode
        // automatically; normals stay UNORM (linear-space, already in [0,1]).
        albedo_textures_[i] = probe(specs[i].color_jpg,  VK_FORMAT_R8G8B8A8_SRGB);
        normal_textures_[i] = probe(specs[i].normal_jpg, VK_FORMAT_R8G8B8A8_UNORM);
    }
}

void VulkanEngine::destroy_textures() {
    auto kill = [&](TextureSlot& s) {
        if (s.view)  vkDestroyImageView(device_, s.view, nullptr);
        if (s.image) vmaDestroyImage(allocator_, s.image, s.alloc);
        s = {};
    };
    for (int i = 0; i < kTextureCount; ++i) {
        kill(albedo_textures_[i]);
        kill(normal_textures_[i]);
    }
    if (texture_sampler_) {
        vkDestroySampler(device_, texture_sampler_, nullptr);
        texture_sampler_ = VK_NULL_HANDLE;
    }
}

void VulkanEngine::init_viewmodel() {
    // Procedural placeholder gun: a barrel (cylinder) + body (box) + grip
    // (box). Camera-space coords: +X right, +Y up, -Z forward.
    auto rotX = [](float deg) {
        return glm::rotate(glm::mat4(1.0f), glm::radians(deg),
                           glm::vec3(1.0f, 0.0f, 0.0f));
    };
    auto T = [](glm::vec3 t) { return glm::translate(glm::mat4(1.0f), t); };
    auto S = [](glm::vec3 s) { return glm::scale(glm::mat4(1.0f), s); };

    const glm::vec4 metal(0.32f, 0.34f, 0.40f, 1.0f);
    const glm::vec4 dark (0.16f, 0.17f, 0.20f, 1.0f);
    const glm::vec4 grip (0.18f, 0.13f, 0.10f, 1.0f);

    constexpr float gx = 0.18f;
    constexpr float gy = -0.18f;
    constexpr float body_hz = 0.16f;
    constexpr float body_zc = -0.32f;
    constexpr float barrel_hl = 0.12f;
    constexpr float barrel_zc = body_zc - body_hz - barrel_hl;

    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy, body_zc)) * S(glm::vec3(0.045f, 0.045f, body_hz)),
        metal,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cylinder,
        T(glm::vec3(gx, gy, barrel_zc)) * rotX(90.0f) *
            S(glm::vec3(0.040f, barrel_hl, 0.040f)),
        dark,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy + 0.050f, body_zc)) *
            S(glm::vec3(0.018f, 0.005f, body_hz * 0.55f)),
        dark,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy + 0.040f, barrel_zc - barrel_hl + 0.01f)) *
            S(glm::vec3(0.006f, 0.012f, 0.006f)),
        dark,
    });
    viewmodel_proc_parts_.push_back({
        ViewmodelPart::Kind::Cube,
        T(glm::vec3(gx, gy - 0.10f, body_zc + 0.06f)) *
            S(glm::vec3(0.038f, 0.075f, 0.045f)),
        grip,
    });

    GltfModel gltf = load_gltf("assets/gun.glb");
    if (!gltf.primitives.empty()) {
        glm::vec3 center = (gltf.aabb_min + gltf.aabb_max) * 0.5f;
        glm::vec3 extent = gltf.aabb_max - gltf.aabb_min;
        float max_side = std::max({extent.x, extent.y, extent.z, 1e-3f});
        float scale = 0.30f / max_side;

        viewmodel_root_offset_ =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.16f, -0.20f, -0.35f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(scale)) *
            glm::translate(glm::mat4(1.0f), -center);

        for (const auto& prim : gltf.primitives) {
            ViewmodelMesh vm{};
            vm.mesh = create_mesh_from_data(device_, allocator_, graphics_queue_,
                                            graphics_queue_family_,
                                            prim.vertices.data(),
                                            static_cast<uint32_t>(prim.vertices.size()),
                                            prim.indices.data(),
                                            static_cast<uint32_t>(prim.indices.size()));
            vm.base_color = prim.base_color;
            viewmodel_gltf_.push_back(std::move(vm));
        }
        log::infof("viewmodel: glTF loaded with %zu primitives (placeholder hidden)",
                   viewmodel_gltf_.size());
        viewmodel_proc_parts_.clear();
    } else {
        log::info("viewmodel: using procedural placeholder "
                  "(drop assets/gun.glb to override)");
    }
}

void VulkanEngine::destroy_viewmodel() {
    for (auto& vm : viewmodel_gltf_) destroy_mesh(allocator_, vm.mesh);
    viewmodel_gltf_.clear();
    viewmodel_proc_parts_.clear();
}

void VulkanEngine::draw_viewmodel(VkCommandBuffer cmd, const glm::mat4& vp,
                                  const glm::mat4& view_inv) {
    auto draw_part = [&](const Mesh& m, const glm::mat4& model,
                         const glm::vec4& color, const glm::vec3& emissive,
                         bool full_emissive) {
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, m.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        // Viewmodel is camera-attached — prev_mvp = mvp gives zero motion,
        // which is what we want for SVGF (the gun shouldn't smear when the
        // camera turns even though world points "moved").
        pc.prev_mvp = pc.mvp;
        pc.color = color;
        pc.emissive = glm::vec4(emissive, full_emissive ? 1.0f : 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, m.index_count, 1, 0, 0, 0);
    };

    // Recoil: shift the entire viewmodel back along camera +Z (toward the
    // camera) by the kick curve. Camera looks down -Z, so positive +Z = back.
    float kick = recoil_kick(recoil_timer_, kRecoilDuration, kRecoilStroke);
    glm::mat4 recoil_offset = glm::translate(glm::mat4(1.0f),
                                             glm::vec3(0.0f, 0.0f, kick));
    glm::mat4 view_inv_recoil = view_inv * recoil_offset;

    if (!viewmodel_gltf_.empty()) {
        glm::mat4 root = view_inv_recoil * viewmodel_root_offset_;
        for (const auto& vm : viewmodel_gltf_) {
            draw_part(vm.mesh, root, vm.base_color, glm::vec3(0.0f), false);
        }
        return;
    }

    for (const auto& p : viewmodel_proc_parts_) {
        glm::mat4 world = view_inv_recoil * p.local;
        const Mesh& mesh = (p.kind == ViewmodelPart::Kind::Cube)
                              ? cube_mesh_ : cylinder_mesh_;
        draw_part(mesh, world, p.color, glm::vec3(0.0f), false);
    }

    // Muzzle flash: a bright, short-lived emissive blob just past the muzzle
    // tip. Bloom downstream halos it dramatically. Anchored so the
    // back face of the (scale-animated) cube always sits past the
    // barrel tip — without this, the small fade-in/out cube floats
    // forward of the tip because scale shrinks but center stays fixed.
    if (muzzle_flash_timer_ > 0.0f) {
        const float barrel_tip_z = -0.72f;  // matches barrel_zc - barrel_hl in init
        float t = std::min(1.0f, muzzle_flash_timer_ / kMuzzleFlashDuration);
        float intensity = t * t;
        float core_size = 0.04f * (0.6f + 0.4f * t);   // full cube size
        // Center the cube so its back face (toward camera) is exactly at
        // the barrel tip independent of size: center_z = tip - half_size.
        float core_z = barrel_tip_z - core_size * 0.5f;
        glm::vec3 muzzle(0.18f, -0.18f, core_z);
        glm::mat4 base = glm::translate(glm::mat4(1.0f), muzzle);

        glm::mat4 core = view_inv_recoil * base *
                         glm::scale(glm::mat4(1.0f), glm::vec3(core_size));
        glm::vec3 core_emi = glm::vec3(8.0f, 6.5f, 3.0f) * intensity;
        draw_part(cube_mesh_, core, glm::vec4(1.0f, 0.85f, 0.45f, 1.0f),
                  core_emi, true);

        glm::mat4 rotX90 = glm::rotate(glm::mat4(1.0f),
                                       glm::radians(90.0f),
                                       glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 tongue = view_inv_recoil * base *
                           glm::translate(glm::mat4(1.0f),
                                          glm::vec3(0.0f, 0.0f, -0.06f * t)) *
                           rotX90 *
                           glm::scale(glm::mat4(1.0f),
                                      glm::vec3(0.025f, 0.06f * t, 0.025f));
        glm::vec3 tongue_emi = glm::vec3(6.0f, 4.5f, 1.6f) * intensity;
        draw_part(cylinder_mesh_, tongue, glm::vec4(1.0f, 0.78f, 0.30f, 1.0f),
                  tongue_emi, true);
    }
}

void VulkanEngine::rebuild_dyn_render_cache() {
    // Resize without reset so we can reuse cached transforms for sleeping
    // bodies.
    dyn_render_cache_.resize(dyn_props_.size());
    if (!physics_) return;
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        DynRender& dr = dyn_render_cache_[i];
        const uint32_t body = dyn_props_[i].body_id;
        const auto handle = dyn_props_[i].jolt_handle;
        const bool slot_matches = dr.valid && dr.body_id == body;
        // Reuse cached transform if the slot still maps to the same body AND
        // Jolt has put it to sleep.
        if (slot_matches && !physics_->is_body_active_h(handle)) {
            // Sleeping → no motion this frame. Drag prev_world forward.
            dr.prev_world = dr.world;
            continue;
        }
        // Capture the prior world before we overwrite it.
        glm::mat4 captured_prev = slot_matches ? dr.world : glm::mat4(1.0f);
        if (!physics_->get_body_world_matrix_h(handle, dr.world)) {
            dr.valid = false;
            dr.body_id = 0;
            continue;
        }
        if (!slot_matches) captured_prev = dr.world;
        dr.prev_world = captured_prev;
        glm::vec3 he = dyn_props_[i].full_size * 0.5f;
        glm::vec3 mn(std::numeric_limits<float>::max());
        glm::vec3 mx(std::numeric_limits<float>::lowest());
        for (int j = 0; j < 8; ++j) {
            glm::vec4 c((j & 1) ? he.x : -he.x,
                        (j & 2) ? he.y : -he.y,
                        (j & 4) ? he.z : -he.z, 1.0f);
            glm::vec3 wc(dr.world * c);
            mn = glm::min(mn, wc);
            mx = glm::max(mx, wc);
        }
        dr.valid = true;
        dr.body_id = body;
        dr.aabb_min = mn;
        dr.aabb_max = mx;
    }
}

void VulkanEngine::rebuild_tick_aabbs() {
    // Static level + every dynamic box's world-space AABB envelope. With
    // rotated boxes we use the AABB of the rotated cube (over-approximation)
    // — the player's collision is still solid against tilted falling crates.
    tick_aabbs_.clear();
    tick_aabbs_.reserve(world_.aabbs.size() + dyn_props_.size());
    for (const auto& a : world_.aabbs) tick_aabbs_.push_back(a);
    if (!physics_) return;
    if (dyn_tick_aabb_cache_.size() != dyn_props_.size()) {
        dyn_tick_aabb_cache_.resize(dyn_props_.size());
    }
    for (size_t idx = 0; idx < dyn_props_.size(); ++idx) {
        const auto& dp = dyn_props_[idx];
        DynTickAabb& slot = dyn_tick_aabb_cache_[idx];
        // Slot identity check: the cached entry is only valid if it was
        // populated for THIS body. FIFO eviction (front-erase + push_back)
        // shifts every other slot down by 1, so a slot's body identity can
        // change without dyn_props_.size() changing. Without this guard the
        // stale cached AABB describes a now-removed box and the player
        // walks through the new occupant.
        const bool slot_matches = slot.body_id == dp.body_id;
        const bool active = physics_->is_body_active_h(dp.jolt_handle);
        if (slot_matches && !active) {
            // Sleeping body in a slot we already populated → AABB hasn't
            // moved since last compute, reuse.
            tick_aabbs_.push_back(slot.aabb);
            continue;
        }
        glm::mat4 m;
        if (!physics_->get_body_world_matrix_h(dp.jolt_handle, m)) {
            slot.body_id = 0;   // mark invalid
            continue;
        }
        glm::vec3 he = dp.full_size * 0.5f;
        glm::vec3 mn(std::numeric_limits<float>::max());
        glm::vec3 mx(std::numeric_limits<float>::lowest());
        for (int i = 0; i < 8; ++i) {
            glm::vec4 c((i & 1) ? he.x : -he.x,
                        (i & 2) ? he.y : -he.y,
                        (i & 4) ? he.z : -he.z, 1.0f);
            glm::vec3 wc(m * c);
            mn = glm::min(mn, wc);
            mx = glm::max(mx, wc);
        }
        slot.aabb = {mn, mx};
        slot.body_id = dp.body_id;
        tick_aabbs_.push_back(slot.aabb);
    }
}

void VulkanEngine::spawn_random_box() {
    if (!physics_) return;

    glm::vec3 pos(frand_range(spawn_rng_state_, -16.0f, 16.0f),
                  frand_range(spawn_rng_state_, 13.0f, 18.0f),
                  frand_range(spawn_rng_state_, -16.0f, 16.0f));

    glm::vec3 euler(frand_range(spawn_rng_state_, -1.2f, 1.2f),
                    frand_range(spawn_rng_state_, -1.2f, 1.2f),
                    frand_range(spawn_rng_state_, -1.2f, 1.2f));

    float side = frand_range(spawn_rng_state_, 0.30f, 0.75f);
    glm::vec3 he(side, side * frand_range(spawn_rng_state_, 0.7f, 1.3f), side);

    static const glm::vec3 palette[] = {
        {1.00f, 0.40f, 0.40f}, {0.40f, 1.00f, 0.40f}, {0.40f, 0.55f, 1.00f},
        {1.00f, 1.00f, 0.30f}, {1.00f, 0.50f, 1.00f}, {0.40f, 1.00f, 1.00f},
        {1.00f, 0.70f, 0.30f}, {0.70f, 0.30f, 0.85f}, {0.85f, 0.85f, 0.85f},
    };
    constexpr int kPaletteCount = sizeof(palette) / sizeof(palette[0]);
    glm::vec3 col = palette[xorshift32(spawn_rng_state_) % kPaletteCount];

    uint32_t id = physics_->add_dynamic_box(pos, he, euler);
    if (id == 0) return;
    DynamicProp p{};
    p.body_id = id;
    p.jolt_handle = physics_->handle_of(id);
    p.full_size = he * 2.0f;
    p.fallback_color = glm::vec4(col, 1.0f);
    int tex = 1 + static_cast<int>(xorshift32(spawn_rng_state_) % 4u);
    p.tex_albedo = tex;
    p.tex_normal = tex;
    p.uv_scale = 1.5f;
    p.color = glm::vec4(glm::mix(glm::vec3(0.9f), col, 0.4f), 1.0f);
    dyn_props_.push_back(p);

    // FIFO eviction: drop the oldest while we exceed the cap.
    while (static_cast<int>(dyn_props_.size()) > kMaxDynProps) {
        physics_->remove_body(dyn_props_.front().body_id);
        dyn_props_.erase(dyn_props_.begin());
    }
}

void VulkanEngine::render_world_depth_pass(VkCommandBuffer cmd) {
    // Mirror render_world()'s view setup. We don't try to share state across
    // the two passes because that would require threading FrameView through
    // both, and the saving is in the noise vs cube.frag's inline-RT cost
    // which the prepass exists to skip.
    float aspect = static_cast<float>(render_extent_.width) /
                   static_cast<float>(render_extent_.height);

    constexpr float kFixedDt = 1.0f / 120.0f;
    float alpha = physics_accumulator_ / kFixedDt;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    glm::vec3 render_pos = glm::mix(prev_player_position_, player_.position, alpha);

    glm::vec3 saved = player_.position;
    player_.position = render_pos;
    glm::mat4 view = player_.view_matrix();
    player_.position = saved;
    glm::mat4 proj = glm::perspective(glm::radians(80.0f), aspect, 0.05f, 500.0f);
    proj[1][1] *= -1.0f;

    // Match the color-pass jitter exactly so depth-prepass and color-pass
    // depth values are bit-identical (LESS_OR_EQUAL passes the equal-depth
    // pixel; if jitter mismatched, the color pass would mostly fail the test
    // and the screen would go black).
    if (rt_.taa_jitter_enabled) {
        auto halton = [](int b, int i) {
            float f = 1.0f, r = 0.0f;
            while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
            return r;
        };
        int idx = static_cast<int>(frame_number_ % 16) + 1;
        float jx = (halton(2, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.width);
        float jy = (halton(3, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.height);
        proj[2][0] += jx;
        proj[2][1] += jy;
    }

    glm::mat4 vp = proj * view;
    Frustum frustum = extract_frustum(vp);

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
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) continue;
        glm::mat4 model = glm::translate(glm::mat4(1.0f), b.center) *
                          glm::scale(glm::mat4(1.0f), b.size);
        push_depth(model);
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) continue;
        glm::mat4 model = dr.world * glm::scale(glm::mat4(1.0f), dyn_props_[i].full_size);
        push_depth(model);
    }
    // Terrain: vertices are already in world space, model = identity.
    if (terrain_mesh_.index_count > 0) {
        VkDeviceSize toff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &terrain_mesh_.vertex_buffer, &toff);
        vkCmdBindIndexBuffer(cmd, terrain_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        pc.mvp = vp;
        pc.model = glm::mat4(1.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, terrain_mesh_.index_count, 1, 0, 0, 0);
    }
}

void VulkanEngine::render_world(VkCommandBuffer cmd) {
    float aspect = static_cast<float>(render_extent_.width) /
                   static_cast<float>(render_extent_.height);

    // Render at the interpolated position between the last two physics ticks,
    // so visuals are smooth even when tick rate != render rate.
    constexpr float kFixedDt = 1.0f / 120.0f;
    float alpha = physics_accumulator_ / kFixedDt;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    glm::vec3 render_pos = glm::mix(prev_player_position_, player_.position, alpha);

    glm::vec3 saved = player_.position;
    player_.position = render_pos;
    glm::mat4 view = player_.view_matrix();
    player_.position = saved;
    glm::mat4 proj = glm::perspective(glm::radians(80.0f), aspect, 0.05f, 500.0f);
    proj[1][1] *= -1.0f;

    // Sub-pixel Halton jitter — TAA integrates these offsets into a super-
    // sampled-looking result across ~16 frames.
    if (rt_.taa_jitter_enabled) {
        auto halton = [](int b, int i) {
            float f = 1.0f, r = 0.0f;
            while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
            return r;
        };
        int idx = static_cast<int>(frame_number_ % 16) + 1;
        float jx = (halton(2, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.width);
        float jy = (halton(3, idx) - 0.5f) * 2.0f * rt_.taa_jitter_strength /
                   static_cast<float>(render_extent_.height);
        proj[2][0] += jx;
        proj[2][1] += jy;
    }

    last_view_proj_ = proj * view;
    glm::mat4 vp = last_view_proj_;
    Frustum frustum = extract_frustum(vp);

    VkViewport vp_state{};
    vp_state.x = 0.0f; vp_state.y = 0.0f;
    vp_state.width  = static_cast<float>(render_extent_.width);
    vp_state.height = static_cast<float>(render_extent_.height);
    vp_state.minDepth = 0.0f; vp_state.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp_state);

    VkRect2D scissor{ {0, 0}, render_extent_ };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const bool tex_on = rt_.textures_enabled;
    // prev_view_proj_ is captured at the END of each frame from the previous
    // frame's last_view_proj_; on frame 0 it's identity (which produces a
    // single-frame motion-vector glitch — TAA's "history_valid" gate handles
    // the invalid-history first frame, so this doesn't surface visually).
    const glm::mat4 prev_vp = prev_view_proj_;
    auto draw_brush = [&](const glm::mat4& model, const glm::mat4& prev_model,
                          const glm::vec4& color,
                          const glm::vec3& emissive, bool full_emissive,
                          int tex_albedo, int tex_normal, float uv_scale,
                          bool object_space) {
        PushConstants pc{};
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
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    int culled_static = 0, culled_dyn = 0, drawn_static = 0, drawn_dyn = 0;
    // Terrain pass — bound first because the brush+dyn loop below assumes
    // the cube mesh is bound. Switch back to cube after.
    if (terrain_mesh_.index_count > 0) {
        VkDeviceSize toff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &terrain_mesh_.vertex_buffer, &toff);
        vkCmdBindIndexBuffer(cmd, terrain_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        pc.mvp = vp;
        pc.model = glm::mat4(1.0f);
        pc.prev_mvp = prev_vp;
        pc.color = glm::vec4(terrain_color_, 1.0f);
        pc.emissive = glm::vec4(0.0f);
        pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, terrain_mesh_.index_count, 1, 0, 0, 0);
        // Rebind cube for the brush loop.
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) { ++culled_static; continue; }
        glm::mat4 model = glm::translate(glm::mat4(1.0f), b.center) *
                          glm::scale(glm::mat4(1.0f), b.size);
        glm::vec4 brush_base = tex_on ? b.color : b.fallback_color;
        // Static brushes don't move — prev_model = current.
        draw_brush(model, model, brush_base, b.emissive, b.full_emissive,
                   b.tex_albedo, b.tex_normal, b.uv_scale, false);
        ++drawn_static;
    }
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) { ++culled_dyn; continue; }
        const glm::mat4 scale_m = glm::scale(glm::mat4(1.0f), dyn_props_[i].full_size);
        glm::mat4 model = dr.world * scale_m;
        glm::mat4 prev_model = dr.prev_world * scale_m;
        glm::vec4 dyn_base = tex_on
            ? dyn_props_[i].color
            : dyn_props_[i].fallback_color;
        draw_brush(model, prev_model, dyn_base, glm::vec3(0.0f), false,
                   dyn_props_[i].tex_albedo, dyn_props_[i].tex_normal,
                   dyn_props_[i].uv_scale, /*object_space=*/true);
        ++drawn_dyn;
    }

    // Bullet-impact decals — drawn AFTER static + dyn brushes so they sit
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

    // Projectiles: switch to cylinder mesh, draw each in-flight bullet.
    if (!projectiles_.empty() && physics_) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        for (const auto& p : projectiles_) {
            glm::mat4 world;
            if (!physics_->get_body_world_matrix(p.body_id, world)) continue;
            glm::vec3 pos(world[3]);
            glm::vec3 vel = physics_->get_linear_velocity(p.body_id);
            glm::vec3 dir = glm::length(vel) > 1e-3f ? glm::normalize(vel) : p.initial_dir;
            glm::mat4 model = align_local_y_to(pos, dir) *
                              glm::scale(glm::mat4(1.0f),
                                         glm::vec3(p.radius, p.half_length, p.radius));
            PushConstants pc{};
            pc.mvp = vp * model;
            pc.model = model;
            // Projectiles don't track a prev_pose yet — zero-motion approx.
            pc.prev_mvp = pc.mvp;
            pc.color = p.color;
            pc.emissive = glm::vec4(p.emissive, 1.0f);  // full_emissive
            vkCmdPushConstants(cmd, pipeline_layout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
        }
    }

    // Viewmodel: rendered last so it overdraws world geometry only when its
    // depth wins. Its depth values are tiny (close to camera), so it sits in
    // front of everything in practice.
    draw_viewmodel(cmd, vp, glm::inverse(view));

    last_draw_static_ = drawn_static;
    last_draw_dyn_    = drawn_dyn;
    last_culled_      = culled_static + culled_dyn;
}

} // namespace qlike
