// Scene props and dynamic objects: the first-person viewmodel, the
// spacejet flyover system, the swinging castle doors, random falling-box
// spawning, and the per-frame dynamic-prop render/TLAS upkeep caches.
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

void VulkanEngine::init_viewmodel() {
    // Procedural placeholder gun: a barrel (cylinder) + body (box) + grip
    // (box). Camera-space coords: +X right, +Y up, -Z forward.
    auto rotX = [](float deg) {
        return glm::rotate(glm::mat4(1.0f), glm::radians(deg),
                           glm::vec3(1.0f, 0.0f, 0.0f));
    };
    auto T = [](glm::vec3 t) { return glm::translate(glm::mat4(1.0f), t); };
    auto S = [](glm::vec3 s) { return glm::scale(glm::mat4(1.0f), s); };

    // Dark-grey palette вЂ” gunmetal body, near-black sights/barrel,
    // very dark grip. Cooler than the previous bluish metal so the
    // gun reads as black-grey in the lower-right of the screen.
    const glm::vec4 metal(0.18f, 0.18f, 0.18f, 1.0f);
    const glm::vec4 dark (0.08f, 0.08f, 0.08f, 1.0f);
    const glm::vec4 grip (0.11f, 0.11f, 0.11f, 1.0f);

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

    // Try the user-droppable single-file path first, then the multi-file
    // scene layout (matches the spacejet asset). Either format works.
    GltfModel gltf = load_gltf("assets/gun.glb");
    if (gltf.primitives.empty()) {
        gltf = load_gltf("assets/gun/scene.gltf");
    }
    if (!gltf.primitives.empty()) {
        glm::vec3 center = (gltf.aabb_min + gltf.aabb_max) * 0.5f;
        glm::vec3 extent = gltf.aabb_max - gltf.aabb_min;
        float max_side = std::max({extent.x, extent.y, extent.z, 1e-3f});
        // 0.55m max side reads better at first-person scale than the
        // earlier 0.30m. The 180В° Y-rotation flips the gun's nose to
        // point forward (camera -Z); without it the assets/gun model
        // points back at the player.
        // m254 asset orientation walk-through: bare в†’ pointed UP, pitch
        // -90В°X в†’ pointed LEFT on its side. The asset's barrel axis
        // turned out to be -X (not +X as previously assumed) вЂ” yaw
        // +90В° around Y mapped that to +Z (toward the player) and the
        // gun pointed backwards. Yaw -90В° flips it the other way so
        // the barrel ends up along -Z (camera forward).
        float scale = 0.85f / max_side;
        glm::mat4 rot_yaw = glm::rotate(glm::mat4(1.0f),
                                         glm::radians(-90.0f),
                                         glm::vec3(0.0f, 1.0f, 0.0f));

        // Z = -0.22 (was -0.35) pulls the gun back toward the camera
        // a bit so a slightly larger model still fits comfortably in
        // the lower-right corner of the frame.
        viewmodel_root_offset_ =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.16f, -0.20f, -0.22f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(scale)) *
            rot_yaw *
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

void VulkanEngine::init_spacejet() {
    GltfModel gltf = load_gltf("assets/spacejet/scene.gltf");
    if (gltf.primitives.empty()) {
        log::info("spacejet: no asset at assets/spacejet/scene.gltf РІР‚вЂќ skipping");
        return;
    }
    // Normalise to a sensible visual size РІР‚вЂќ scale so the asset's
    // longest dimension is ~6 m. Per-flight scale jitter multiplies
    // this base value.
    glm::vec3 center = (gltf.aabb_min + gltf.aabb_max) * 0.5f;
    glm::vec3 extent = gltf.aabb_max - gltf.aabb_min;
    float max_side = std::max({extent.x, extent.y, extent.z, 1e-3f});
    spacejet_centre_offset_ = -center;
    spacejet_base_scale_    = 6.0f / max_side;

    spacejet_meshes_.reserve(gltf.primitives.size());
    for (const auto& prim : gltf.primitives) {
        SpacejetMesh sm{};
        sm.mesh = create_mesh_from_data(device_, allocator_, graphics_queue_,
                                         graphics_queue_family_,
                                         prim.vertices.data(),
                                         static_cast<uint32_t>(prim.vertices.size()),
                                         prim.indices.data(),
                                         static_cast<uint32_t>(prim.indices.size()));
        sm.base_color = prim.base_color;
        spacejet_meshes_.push_back(std::move(sm));
    }
    log::infof("spacejet: loaded %zu primitives, base_scale=%.3f РІР‚вЂќ flyover system armed",
               spacejet_meshes_.size(), spacejet_base_scale_);
}

void VulkanEngine::destroy_spacejet() {
    for (auto& sm : spacejet_meshes_) destroy_mesh(allocator_, sm.mesh);
    spacejet_meshes_.clear();
    spacejet_flights_.clear();
}

void VulkanEngine::tick_spacejet(float dt) {
    // Decay the top-center damage HUD ttl regardless of whether any
    // flights exist (we want it to fade out cleanly after the last
    // plane in a wave is destroyed and the list is emptied).
    if (last_target_hud_ttl_ > 0.0f) last_target_hud_ttl_ -= dt;
    if (kill_flash_t_       > 0.0f) kill_flash_t_       -= dt;
    if (spacejet_meshes_.empty()) return;

    // Per-tick: integrate heading (turn_rate * dt), then position
    // along the new heading. Despawn when ttl elapses.
    constexpr float kDestroyFade  = 0.6f;   // seconds before removal
    constexpr float kPostCastleMax = 20.0f; // seconds after passing castle
    for (auto& f : spacejet_flights_) {
        f.prev_pos     = f.pos;
        f.prev_heading = f.heading;
        // Destroying flights pitch down + decelerate so the explosion
        // burst reads as "the plane was killed". 0.6 s window matches
        // the spark TTL so the wreckage disappears as sparks fade.
        if (f.destroying) {
            f.destroy_t += dt;
            f.speed     *= std::max(0.0f, 1.0f - dt * 1.5f);
            f.pos.y     -= dt * 14.0f * f.destroy_t;     // accelerating fall
        }
        f.heading += f.turn_rate * dt;
        glm::vec3 fwd{ std::sin(f.heading), 0.0f, std::cos(f.heading) };
        glm::vec3 prev = f.pos;
        f.pos += fwd * (f.speed * dt);
        f.t   += dt;
        if (f.hit_flash_t > 0.0f) f.hit_flash_t -= dt;
        // Detect castle pass: dot(pos, fwd) goes from negative (before
        // origin) to positive (past origin). The castle sits at world
        // origin. Once detected, start counting post-castle seconds
        // toward the 20 s hard removal.
        if (f.post_castle_t < 0.0f) {
            float dot_prev = glm::dot(prev,  fwd);
            float dot_cur  = glm::dot(f.pos, fwd);
            if (dot_prev <= 0.0f && dot_cur > 0.0f) f.post_castle_t = 0.0f;
        } else {
            f.post_castle_t += dt;
        }
    }
    spacejet_flights_.erase(
        std::remove_if(spacejet_flights_.begin(), spacejet_flights_.end(),
                       [kDestroyFade, kPostCastleMax](const SpacejetFlight& f){
                           return f.t >= f.ttl ||
                                  (f.destroying && f.destroy_t >= kDestroyFade) ||
                                  (f.post_castle_t >= kPostCastleMax);
                       }),
        spacejet_flights_.end());

    // Spawn cadence -- the sky is busy now. Was 4-12 s between waves
    // with 55% solo flights; now 1.5-4 s and the distribution is
    // biased toward multi-jet formations.
    spacejet_spawn_timer_ -= dt;
    if (spacejet_spawn_timer_ > 0.0f) return;
    spacejet_spawn_timer_ = frand_range(spacejet_rng_state_, 1.5f, 4.0f);

    // Wave size: pairs / trios dominate, 5-jet formations frequent.
    uint32_t r = xorshift32(spacejet_rng_state_) % 100u;
    int wave = (r < 15) ? 1 :          //  15% solo
               (r < 45) ? 2 :          //  30% pair
               (r < 75) ? 3 :          //  30% trio
               (r < 90) ? 4 : 5;       //  15% quad, 10% five

    // Pick a flight heading. 0..2РџР‚, 0 = +Z. Spawn point is 1.4 km
    // behind the castle (i.e. opposite the heading direction) so the
    // jet flies inbound, over, and out the other side.
    float heading = frand(spacejet_rng_state_) * 6.28318530718f;
    glm::vec3 fwd{ std::sin(heading), 0.0f, std::cos(heading) };
    glm::vec3 right{ fwd.z, 0.0f, -fwd.x };

    constexpr float kTrackLen = 2800.0f;
    // Lateral jitter dropped 80 m в†’ 25 m so every wave actually
    // crosses (or grazes) the castle silhouette at midflight. 80 m
    // could put flights at the edge of view; 25 m guarantees the
    // jets pass within the castle footprint plus a small jitter
    // for visual variety.
    constexpr float kSideJit  = 25.0f;
    float lateral  = frand_range(spacejet_rng_state_, -kSideJit, kSideJit);
    // Altitude lowered (was 150-190 m). Still clear of FBM peaks at
    // 120 m by a comfortable margin, but close enough that the jets
    // read as a real combat threat instead of a distant flyover.
    float altitude = frand_range(spacejet_rng_state_, 95.0f, 130.0f);
    float speed    = frand_range(spacejet_rng_state_,  70.0f, 130.0f);

    // Random gentle turn rate. Most waves go straight; some bank
    // through a wide arc for variety. Cap at ~6 deg/s (0.1 rad/s)
    // so the curve is visible but not aerobatic. ~30% chance of any
    // turn at all.
    float turn_rate = 0.0f;
    if ((xorshift32(spacejet_rng_state_) & 0xFFu) < 80u) {
        turn_rate = frand_range(spacejet_rng_state_, -0.10f, 0.10f);
    }

    glm::vec3 wave_origin = -fwd * (kTrackLen * 0.5f) +
                            right * lateral +
                            glm::vec3(0.0f, altitude, 0.0f);

    for (int i = 0; i < wave; ++i) {
        float side  = (wave == 1) ? 0.0f
                                  : (float(i) - float(wave - 1) * 0.5f) * 12.0f;
        float along = (wave == 1) ? 0.0f
                                  : float(i) * 6.0f;
        SpacejetFlight f{};
        f.pos          = wave_origin + right * side - fwd * along;
        f.heading      = heading;
        f.prev_heading = heading;
        f.turn_rate    = turn_rate;
        f.speed        = speed + frand_range(spacejet_rng_state_, -8.0f, 8.0f);
        f.scale        = 1.0f + frand_range(spacejet_rng_state_, -0.2f, 0.4f);
        f.t            = 0.0f;
        f.ttl          = kTrackLen / std::max(40.0f, f.speed) + 2.0f;
        f.prev_pos     = f.pos;
        spacejet_flights_.push_back(f);
    }
    log::infof("[spacejet] wave of %d launched (heading=%.1f deg, alt=%.0f, "
               "spd=%.0f, turn=%.2f rad/s)",
               wave, glm::degrees(heading), altitude, speed, turn_rate);
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
        // Viewmodel is camera-attached Р Р†Р вЂљРІР‚Сњ prev_mvp = mvp gives zero motion,
        // which is what we want for SVGF (the gun shouldn't smear when the
        // camera turns even though world points "moved").
        pc.prev_mvp = pc.mvp;
        pc.color = color;
        pc.emissive = glm::vec4(emissive, full_emissive ? 1.0f : 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
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
    // barrel tip Р Р†Р вЂљРІР‚Сњ without this, the small fade-in/out cube floats
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
            // Sleeping Р Р†РІР‚В РІР‚в„ў no motion this frame. Drag prev_world forward.
            dr.prev_world = dr.world;
            // Sleeping bodies: prev_model trails the cached model so
            // motion-vec emit stays zero. (model itself is unchanged.)
            dr.prev_model = dr.model;
            continue;
        }
        // Capture the prior world AND model before we overwrite them.
        glm::mat4 captured_prev = slot_matches ? dr.world : glm::mat4(1.0f);
        glm::mat4 captured_prev_model = slot_matches ? dr.model : glm::mat4(1.0f);
        if (!physics_->get_body_world_matrix_h(handle, dr.world)) {
            dr.valid = false;
            dr.body_id = 0;
            continue;
        }
        if (!slot_matches) captured_prev = dr.world;
        dr.prev_world = captured_prev;
        // World-space AABB of an axis-aligned local box under an affine
        // transform. Shared helper -- exact for any non-shearing transform
        // (which is all Jolt rigid bodies produce).
        world_aabb_of_box(dr.world, dyn_props_[i].full_size * 0.5f,
                          dr.aabb_min, dr.aabb_max);
        dr.valid   = true;
        dr.body_id = body;
        // Bake `world * scale(full_size)` once per frame. Was done
        // per-draw in 4 sites (depth pass, shadow_lr pass, color pass,
        // TLAS writer); ~60 boxes * 4 sites = ~240 mat4 muls/frame
        // collapsed to ~60. The scale matrix itself is a 9-mul build
        // we now do once per body instead of per-pass.
        const glm::mat4 scale_m = glm::scale(glm::mat4(1.0f),
                                              dyn_props_[i].full_size);
        dr.model      = dr.world * scale_m;
        // First-frame-of-slot guard mirrors captured_prev above: keep
        // motion-vec emit zero by holding prev_model == model.
        dr.prev_model = slot_matches ? captured_prev_model : dr.model;
    }
}

void VulkanEngine::rebuild_tick_aabbs() {
    // Static level + every dynamic box's world-space AABB envelope. With
    // rotated boxes we use the AABB of the rotated cube (over-approximation)
    // Р Р†Р вЂљРІР‚Сњ the player's collision is still solid against tilted falling crates.
    tick_aabbs_.clear();
    tick_aabbs_.reserve(world_.aabbs.size() + voxel_collision_aabbs_.size() +
                        dyn_props_.size());
    for (const auto& a : world_.aabbs) tick_aabbs_.push_back(a);
    // Voxel-tower solid boxes count as STATIC geometry (auto step-up onto
    // carved ledges works). Must stay in the static prefix вЂ” the caller
    // passes static_count = world_.aabbs.size() + voxel_collision_aabbs_.size().
    for (const auto& a : voxel_collision_aabbs_) tick_aabbs_.push_back(a);
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
            // Sleeping body in a slot we already populated Р Р†РІР‚В РІР‚в„ў AABB hasn't
            // moved since last compute, reuse.
            tick_aabbs_.push_back(slot.aabb);
            continue;
        }
        glm::mat4 m;
        if (!physics_->get_body_world_matrix_h(dp.jolt_handle, m)) {
            slot.body_id = 0;   // mark invalid
            continue;
        }
        glm::vec3 mn, mx;
        world_aabb_of_box(m, dp.full_size * 0.5f, mn, mx);
        slot.aabb = {mn, mx};
        slot.body_id = dp.body_id;
        tick_aabbs_.push_back(slot.aabb);
    }
}

void VulkanEngine::init_doors() {
    // North entrance gap is at z = +11 (castle outer wall half-extent =
    // 11 m), centered on x = 0, gap width 3.2 m, clearance ~3.6 m
    // (set in game/level.cpp). Two panels meet at x = 0 when closed; each
    // hinges on its outer edge so opening swings the inner edges outward
    // away from the courtyard, leaving a clear archway for the player to
    // walk through.
    const float plateau_y       = 22.0f;
    const float ent_half        = 1.6f;
    const float gate_z          = 11.0f;
    const float panel_w_half    = 0.78f;   // 0.78 Г— 2 = 1.56 m; sub-2 cm gap
                                            // at meet line so the closed
                                            // panels don't z-fight in the
                                            // middle
    const float panel_h_half    = 1.70f;   // 3.4 m tall вЂ” fits in 3.6 m gap
    const float panel_t_half    = 0.06f;   // 12 cm thick wooden plank
    const float y_center        = plateau_y + panel_h_half + 0.05f;

    auto make_door = [&](float hinge_x, float closed_dx, float open_a) {
        Door d{};
        d.hinge_world   = glm::vec3(hinge_x, y_center, gate_z);
        d.closed_offset = glm::vec3(closed_dx, 0.0f, 0.0f);
        d.half_size     = glm::vec3(panel_w_half, panel_h_half, panel_t_half);
        d.open_angle    = open_a;
        d.angle         = 0.0f;
        return d;
    };
    // Left panel: hinge on -X side, swings outward (toward +Z) by ~85В°.
    doors_.push_back(make_door(-ent_half, +panel_w_half + 0.01f, +1.48f));
    // Right panel: hinge on +X side, swings outward by -85В°.
    doors_.push_back(make_door(+ent_half, -panel_w_half - 0.01f, -1.48f));
}

void VulkanEngine::update_doors(float dt) {
    if (doors_.empty()) return;
    // Trigger distance is measured from the player's eye to the midpoint
    // between the two hinges. 5 m gives the doors time to swing fully
    // open at a normal walking pace before the player reaches the gap.
    glm::vec3 mid(0.0f);
    for (const auto& d : doors_) mid += d.hinge_world;
    mid /= float(doors_.size());

    const float dist = glm::length(player_.eye_position() - mid);
    constexpr float kOpenRadius = 5.0f;
    constexpr float kCloseDelay = 3.0f;     // seconds after leaving range
    if (dist < kOpenRadius) door_far_time_ = 0.0f;
    else                    door_far_time_ += dt;
    const bool want_open = door_far_time_ < kCloseDelay;

    const float lerp_k = std::min(1.0f, 4.0f * dt);   // ~250 ms time constant
    for (auto& d : doors_) {
        const float target = want_open ? d.open_angle : 0.0f;
        d.angle += (target - d.angle) * lerp_k;
    }
}

void VulkanEngine::draw_doors(VkCommandBuffer cmd, const glm::mat4& vp,
                              const glm::mat4& prev_vp) {
    if (doors_.empty()) return;
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &off);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    for (const auto& d : doors_) {
        // Hinge-relative rotation: translate the local cube origin to the
        // panel centre via closed_offset, rotate around the hinge, then
        // place the whole thing in world. cube mesh is unit-cube
        // (-0.5..+0.5), so scale by full panel dimensions (2 Г— half).
        const glm::mat4 model =
            glm::translate(glm::mat4(1.0f), d.hinge_world) *
            glm::rotate(glm::mat4(1.0f), d.angle, glm::vec3(0, 1, 0)) *
            glm::translate(glm::mat4(1.0f), d.closed_offset) *
            glm::scale(glm::mat4(1.0f), d.half_size * 2.0f);
        PushConstants pc{};
        pc.mvp        = vp * model;
        pc.model      = model;
        pc.prev_mvp   = prev_vp * model;
        pc.color      = glm::vec4(0.55f, 0.33f, 0.16f, 1.0f);   // wood tint
        pc.emissive   = glm::vec4(0.0f);
        // Wood048 lives at albedo index 2 (see init_textures specs[]).
        // tex_params.x/y = albedo+normal slot, .z = uv scale, .w = 0 в†’
        // world-space triplanar so the planks read consistently across
        // both panels.
        pc.tex_params = glm::vec4(2.0f, 2.0f, 0.7f, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    }
}

void VulkanEngine::spawn_random_box() {
    if (!physics_) return;

    // Spawn around the castle: XY/Z spread covers the keep + courtyard;
    // height seeds well above the highest tower so crates fall onto the
    // walkway / merlons / outside the perimeter.
    //   - terrain plateau ~y22, castle interior ~y22..27, towers ~y32
    //   - spawn 38..50m so crates have airtime to scatter realistically.
    const float plateau_y = terrain_data_.heights.empty() ? 0.0f : 22.0f;
    glm::vec3 pos(frand_range(spawn_rng_state_, -16.0f, 16.0f),
                  plateau_y + frand_range(spawn_rng_state_, 16.0f, 28.0f),
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

} // namespace qlike
