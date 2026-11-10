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

// ---------------------------------------------------------------------
// Jet AI.
//
// Layered the way flight-game bots usually are:
//   * decision layer  - per-leader state machine picking the next
//                       maneuver (orbit pass, bomb run, loop, reversal,
//                       egress) with weighted randomness so sorties don't
//                       repeat the same pattern;
//   * maneuver layer  - each mode outputs a desired flight direction
//                       (pursuit of a waypoint / circle tangent / loop
//                       pitch program);
//   * flight core     - turns the velocity toward the desired direction
//                       under a g-limited turn rate, banks so lift (up)
//                       points along lateral accel + gravity (correct
//                       coordinated turns and inverted loop tops), trades
//                       speed for altitude, and enforces terrain clearance.
// Wingmen fly formation slots in their leader's frame and copy its bomb
// release. Shot-down jets tumble into the ground and explode.
// ---------------------------------------------------------------------
namespace {
constexpr float kG = 9.81f;

glm::vec3 rotate_toward(const glm::vec3& from, const glm::vec3& to, float max_angle) {
    float c = glm::clamp(glm::dot(from, to), -1.0f, 1.0f);
    float ang = std::acos(c);
    if (ang <= max_angle || ang < 1e-5f) return to;
    glm::vec3 axis = glm::cross(from, to);
    float al = glm::length(axis);
    if (al < 1e-5f) {   // opposite: pick any perpendicular (prefer up)
        axis = glm::cross(from, glm::vec3(0, 1, 0));
        if (glm::length(axis) < 1e-5f) axis = glm::vec3(1, 0, 0);
        al = glm::length(axis);
    }
    axis /= al;
    // Rodrigues
    return glm::normalize(from * std::cos(max_angle) +
                          glm::cross(axis, from) * std::sin(max_angle) +
                          axis * glm::dot(axis, from) * (1.0f - std::cos(max_angle)));
}
glm::vec3 rotate_axis(const glm::vec3& v, const glm::vec3& axis, float a) {
    return v * std::cos(a) + glm::cross(axis, v) * std::sin(a) +
           axis * glm::dot(axis, v) * (1.0f - std::cos(a));
}
}  // namespace

void VulkanEngine::tick_spacejet(float dt) {
    if (last_target_hud_ttl_ > 0.0f) last_target_hud_ttl_ -= dt;
    if (kill_flash_t_       > 0.0f) kill_flash_t_       -= dt;
    if (spacejet_meshes_.empty() || dt <= 0.0f) return;
    dt = std::min(dt, 0.05f);

    auto& R = spacejet_rng_state_;
    auto ground_h = [&](float x, float z) {
        return std::max(sample_terrain_height(x, z), rt_.water_level);
    };

    // Leader decision: pick the next maneuver.
    auto choose_next = [&](SpacejetFlight& f) {
        f.mode_t = 0.0f;
        ++f.sorties;
        if (f.sorties > f.sortie_max) {
            f.mode = JetMode::Egress;
            glm::vec3 h(f.fwd.x, 0.0f, f.fwd.z);
            if (glm::length(h) < 1e-3f) h = glm::vec3(1, 0, 0);
            f.target = glm::normalize(h) * 3000.0f;
            return;
        }
        float agl = f.pos.y - ground_h(f.pos.x, f.pos.z);
        float r = frand(R);
        if (r < 0.36f) {
            f.mode = JetMode::BombRun;
            f.bomb_dropped = false;
            // Aim point: 60..500 m from the castle, avoiding the keep itself.
            float a = frand(R) * 6.2831853f;
            float d = frand_range(R, 60.0f, 500.0f);
            f.target = glm::vec3(std::cos(a) * d, 0.0f, std::sin(a) * d);
            f.target.y = ground_h(f.target.x, f.target.z);
        } else if (r < 0.60f) {
            f.mode = JetMode::Orbit;
            f.orbit_r = frand_range(R, 260.0f, 520.0f);
            // Orbit in the direction we're already turning toward.
            glm::vec3 rad = f.pos; rad.y = 0.0f;
            glm::vec3 tang = glm::cross(glm::vec3(0, 1, 0), glm::normalize(rad + glm::vec3(1e-3f)));
            f.orbit_dir = glm::dot(tang, f.fwd) >= 0.0f ? 1.0f : -1.0f;
            f.target.x = frand_range(R, 14.0f, 24.0f);     // orbit duration
        } else if (r < 0.80f && agl > 85.0f && agl < 320.0f) {
            f.mode = JetMode::Loop;
            f.loop_done = 0.0f;
            f.loop_rate = 6.2831853f / frand_range(R, 7.0f, 9.0f);   // ~8 g, r ~150 m
        } else {
            // Reversal: fly on briefly, then a hard climbing 180 back
            // over the castle.
            f.mode = JetMode::Turn;
            glm::vec3 back(-f.fwd.x, 0.0f, -f.fwd.z);
            f.target = glm::normalize(back + glm::vec3(1e-3f, 0.0f, 0.0f));
            f.orbit_dir = (xorshift32(R) & 1u) ? 1.0f : -1.0f;
        }
    };

    // ---- Spawn waves ----
    int alive = 0;
    for (const auto& f : spacejet_flights_) if (!f.dead && !f.destroying) ++alive;
    spacejet_spawn_timer_ -= dt;
    if (spacejet_spawn_timer_ <= 0.0f && alive < 14) {
        spacejet_spawn_timer_ = frand_range(R, 3.5f, 8.0f);
        int wave = 2 + int(xorshift32(R) % 3u);              // 2..4 jets
        float bearing = frand(R) * 6.2831853f;
        glm::vec3 from(std::cos(bearing), 0.0f, std::sin(bearing));
        SpacejetFlight lead{};
        lead.cruise_alt = frand_range(R, 110.0f, 170.0f);
        lead.pos = from * 1350.0f;
        lead.pos.y = ground_h(lead.pos.x, lead.pos.z) + lead.cruise_alt;
        lead.target = glm::vec3(frand_range(R, -150.0f, 150.0f), 0.0f,
                                frand_range(R, -150.0f, 150.0f));
        lead.fwd = glm::normalize(glm::vec3(lead.target.x, 0, lead.target.z) - glm::vec3(lead.pos.x, 0, lead.pos.z));
        lead.up = glm::vec3(0, 1, 0);
        lead.speed = frand_range(R, 105.0f, 125.0f);
        lead.scale = 1.0f + frand_range(R, -0.1f, 0.3f);
        lead.sortie_max = 3 + int(xorshift32(R) % 3u);
        lead.id = ++spacejet_next_id_;
        lead.prev_pos = lead.pos; lead.prev_fwd = lead.fwd; lead.prev_up = lead.up;
        spacejet_flights_.push_back(lead);
        const glm::vec3 slots[3] = { {-20.0f, -2.0f, -16.0f}, {20.0f, -3.0f, -16.0f}, {0.0f, -5.0f, -34.0f} };
        for (int i = 1; i < wave; ++i) {
            SpacejetFlight w = lead;
            w.trail.clear();
            w.id = ++spacejet_next_id_;
            w.leader_id = lead.id;
            w.slot = slots[i - 1];
            glm::vec3 right = glm::normalize(glm::cross(lead.fwd, lead.up));
            w.pos = lead.pos + right * w.slot.x + lead.up * w.slot.y + lead.fwd * w.slot.z;
            w.prev_pos = w.pos;
            w.scale = 1.0f + frand_range(R, -0.1f, 0.3f);
            spacejet_flights_.push_back(w);
        }
        log::infof("[jets] wave of %d inbound from %.0f deg, alt %.0f m",
                   wave, glm::degrees(bearing), lead.cruise_alt);
    }

    auto find_by_id = [&](uint32_t id) -> SpacejetFlight* {
        if (id == 0) return nullptr;
        for (auto& o : spacejet_flights_)
            if (o.id == id && !o.dead && !o.destroying) return &o;
        return nullptr;
    };

    for (auto& f : spacejet_flights_) {
        f.prev_pos = f.pos; f.prev_fwd = f.fwd; f.prev_up = f.up;
        f.t += dt;
        f.mode_t += dt;
        if (f.hit_flash_t > 0.0f) f.hit_flash_t -= dt;

        // Trail sampling + ageing (also for dead jets until it fades).
        const float kTrailLife = 6.0f;
        while (!f.trail.empty() && f.t - f.trail.front().t > kTrailLife)
            f.trail.erase(f.trail.begin());
        if (f.dead) continue;

        const float half_len = 3.0f * f.scale;   // asset normalised to 6 m
        if (f.destroying) {
            // Shot down: nose drops, spins, accelerates into the ground.
            f.mode = JetMode::Crash;
            f.destroy_t += dt;
            glm::vec3 right = glm::normalize(glm::cross(f.fwd, f.up));
            f.fwd = glm::normalize(rotate_axis(f.fwd, right, -0.55f * dt));
            f.up  = glm::normalize(rotate_axis(f.up, f.fwd, 2.4f * dt));
            f.up  = glm::normalize(f.up - f.fwd * glm::dot(f.up, f.fwd));
            f.speed = std::min(f.speed + kG * 1.2f * dt, 190.0f);
            f.pos += f.fwd * f.speed * dt + glm::vec3(0.0f, -kG * f.destroy_t * dt, 0.0f);
            float gh = ground_h(f.pos.x, f.pos.z);
            if (f.pos.y <= gh + 1.0f || f.destroy_t > 25.0f) {
                f.pos.y = std::max(f.pos.y, gh);
                spawn_explosion(f.pos, 1.3f, true);
                f.dead = true;
            }
        } else {
            SpacejetFlight* lead = find_by_id(f.leader_id);
            if (f.leader_id != 0 && !lead) {
                f.leader_id = 0;                 // promoted to leader
                f.mode = JetMode::Ingress;
                f.target = glm::vec3(0.0f);
                f.mode_t = 0.0f;
            }
            glm::vec3 desired = f.fwd;
            bool hold_alt = true;
            float max_g = 6.0f;
            float speed_cmd = 115.0f;
            if (lead) {
                // ---- Wingman: fly the slot, lead it slightly ----
                glm::vec3 lr = glm::normalize(glm::cross(lead->fwd, lead->up));
                glm::vec3 slot_w = lead->pos + lr * f.slot.x + lead->up * f.slot.y +
                                   lead->fwd * f.slot.z;
                glm::vec3 aim = slot_w + lead->fwd * 45.0f - f.pos;
                desired = glm::normalize(aim);
                float along = glm::dot(slot_w - f.pos, lead->fwd);
                speed_cmd = lead->speed + glm::clamp(along * 0.8f, -25.0f, 25.0f);
                max_g = 8.0f;
                hold_alt = false;
                if (lead->mode == JetMode::Loop) {
                    if (f.mode != JetMode::Loop) { f.mode = JetMode::Loop; f.loop_rate = lead->loop_rate; }
                }
                else if (f.mode == JetMode::Loop) f.mode = JetMode::Ingress;
                // Copy the leader's bomb release.
                if (lead->bomb_dropped && !f.bomb_dropped && lead->mode == JetMode::Turn) {
                    f.bomb_dropped = true;
                    Bomb b{};
                    b.pos = f.pos - f.up * 1.5f;
                    b.vel = f.fwd * f.speed;
                    b.prev_pos = b.pos;
                    bombs_.push_back(b);
                }
                if (!lead->bomb_dropped) f.bomb_dropped = false;
            } else {
                // ---- Leader maneuvers ----
                switch (f.mode) {
                case JetMode::Ingress: {
                    glm::vec3 to = f.target - f.pos; to.y = 0.0f;
                    desired = glm::normalize(to + glm::vec3(1e-3f, 0, 0));
                    if (glm::length(to) < 450.0f) choose_next(f);
                    break;
                }
                case JetMode::Orbit: {
                    glm::vec3 rad(f.pos.x, 0.0f, f.pos.z);
                    float d = glm::length(rad);
                    glm::vec3 ru = d > 1.0f ? rad / d : glm::vec3(1, 0, 0);
                    glm::vec3 tang = glm::cross(glm::vec3(0, 1, 0), ru) * f.orbit_dir;
                    desired = glm::normalize(tang - ru * glm::clamp((d - f.orbit_r) / f.orbit_r * 2.0f, -1.0f, 1.0f));
                    if (f.mode_t > f.target.x) choose_next(f);
                    break;
                }
                case JetMode::BombRun: {
                    glm::vec3 to = f.target - f.pos; to.y = 0.0f;
                    float hd = glm::length(to);
                    desired = to / std::max(hd, 1e-3f);
                    // Fly the run lower: ~100 m above the aim point.
                    f.cruise_alt = 100.0f;
                    float h = std::max(f.pos.y - f.target.y, 1.0f);
                    // Ballistic impact prediction for a release NOW: the bomb
                    // keeps the jet's velocity (incl. dive), falls h metres.
                    // Light drag (fx.cpp) shortens range ~4 %.
                    glm::vec3 v0 = f.fwd * f.speed;
                    float vy = v0.y;
                    float t_fall = (vy + std::sqrt(vy * vy + 2.0f * kG * h)) / kG;
                    glm::vec3 impact = f.pos + glm::vec3(v0.x, 0.0f, v0.z) * (t_fall * 0.96f);
                    glm::vec3 hf = glm::normalize(glm::vec3(f.fwd.x, 0, f.fwd.z) + glm::vec3(1e-4f));
                    glm::vec3 miss = f.target - impact; miss.y = 0.0f;
                    float along = glm::dot(miss, hf);                 // >0: impact short
                    float cross_err = glm::length(miss - hf * along);
                    if (!f.bomb_dropped && along <= 0.0f && along > -60.0f &&
                        cross_err < 35.0f && h < 190.0f) {
                        f.bomb_dropped = true;
                        Bomb b{};
                        b.pos = f.pos - f.up * 1.5f;
                        b.vel = f.fwd * f.speed;
                        b.prev_pos = b.pos;
                        bombs_.push_back(b);
                        log::infof("[jets] bomb away -> (%.0f, %.0f)", f.target.x, f.target.z);
                        // Pull up and reverse.
                        f.mode = JetMode::Turn;
                        f.mode_t = 0.0f;
                        glm::vec3 back(-f.fwd.x, 0.0f, -f.fwd.z);
                        f.target = glm::normalize(back + glm::vec3(1e-3f, 0, 0));
                        f.cruise_alt = frand_range(R, 120.0f, 170.0f);
                        f.orbit_dir = (xorshift32(R) & 1u) ? 1.0f : -1.0f;
                    } else if (f.mode_t > 45.0f) {
                        choose_next(f);
                    }
                    break;
                }
                case JetMode::Loop:
                    hold_alt = false;
                    break;
                case JetMode::Turn: {
                    // First 2.5 s: climb out straight, then a hard
                    // banked climbing reversal toward `target` heading.
                    if (f.mode_t < 2.5f) {
                        // Fixed ~19 deg climb relative to the HORIZON (was
                        // relative to the nose -> compounding pull-up).
                        glm::vec3 fh = glm::normalize(glm::vec3(f.fwd.x, 0, f.fwd.z) + glm::vec3(1e-4f));
                        desired = glm::normalize(fh + glm::vec3(0, 0.35f, 0));
                        hold_alt = false;
                    } else {
                        // Level banked reversal to the chosen side (the
                        // shortest-arc 180 went up and over - an Immelmann
                        // that left jets 500+ m up).
                        glm::vec3 fh = glm::normalize(glm::vec3(f.fwd.x, 0, f.fwd.z) + glm::vec3(1e-4f));
                        glm::vec3 lat = glm::cross(glm::vec3(0, 1, 0), fh) * f.orbit_dir;
                        desired = glm::normalize(fh * 0.15f + lat);
                        max_g = 6.5f;
                        if (glm::dot(glm::normalize(glm::vec3(f.fwd.x, 0, f.fwd.z) + glm::vec3(1e-4f)),
                                     f.target) > 0.96f) {
                            f.mode = JetMode::Ingress;    // back over the castle
                            f.mode_t = 0.0f;
                            f.target = glm::vec3(frand_range(R, -120.0f, 120.0f), 0.0f,
                                                 frand_range(R, -120.0f, 120.0f));
                        }
                    }
                    if (f.mode_t > 30.0f) choose_next(f);
                    break;
                }
                case JetMode::Egress: {
                    glm::vec3 to = f.target - f.pos; to.y = 0.0f;
                    desired = glm::normalize(to + glm::vec3(1e-3f, 0, 0));
                    glm::vec2 hp(f.pos.x, f.pos.z);
                    if (glm::length(hp) > 1700.0f) { f.dead = true; f.trail.clear(); }
                    break;
                }
                default: break;
                }
            }

            const float gh_here  = ground_h(f.pos.x, f.pos.z);
            if (f.mode == JetMode::Loop) {
                // Pitch program: constant-rate pull through 360 deg about
                // the wing axis (lift toward the loop centre).
                glm::vec3 right = glm::normalize(glm::cross(f.fwd, f.up));
                float da = f.loop_rate * dt;
                glm::vec3 nf = glm::normalize(rotate_axis(f.fwd, right, da));
                f.up  = glm::normalize(rotate_axis(f.up, right, da));
                f.turn_g = f.loop_rate * f.speed / kG;
                f.fwd = nf;
                f.loop_done += da;
                if (!lead && (f.loop_done >= 6.2831853f || f.pos.y < gh_here + 45.0f)) {
                    // level out (re-erect up) and carry on
                    f.up = glm::vec3(0, 1, 0);
                    choose_next(f);
                }
            } else {
                // ---- Altitude hold + terrain avoidance ----
                if (hold_alt) {
                    float want = gh_here + f.cruise_alt;
                    desired.y += glm::clamp((want - f.pos.y) / 160.0f, -0.35f, 0.35f);
                }
                glm::vec3 ahead = f.pos + f.fwd * f.speed * 3.0f;
                float gh_ahead = std::max(ground_h(ahead.x, ahead.z),
                                          ground_h((f.pos.x + ahead.x) * 0.5f,
                                                   (f.pos.z + ahead.z) * 0.5f));
                float clear = std::min(f.pos.y - gh_here, ahead.y - gh_ahead);
                if (clear < 70.0f)
                    desired.y += glm::clamp((70.0f - clear) / 35.0f, 0.0f, 1.5f);
                desired.y = glm::clamp(desired.y, -0.55f, 0.85f);
                desired = glm::normalize(desired);

                // ---- Flight core: g-limited turn + coordinated bank ----
                float max_rate = max_g * kG / std::max(f.speed, 30.0f);
                glm::vec3 nf = rotate_toward(f.fwd, desired, max_rate * dt);
                glm::vec3 acc = (nf - f.fwd) / dt * f.speed;          // lateral accel
                f.turn_g = glm::length(acc) / kG;
                glm::vec3 up_t = acc + glm::vec3(0.0f, kG, 0.0f);     // lift vector
                f.fwd = nf;
                up_t -= f.fwd * glm::dot(up_t, f.fwd);
                if (glm::length(up_t) > 1e-3f) up_t = glm::normalize(up_t);
                else up_t = f.up;
                float k = 1.0f - std::exp(-dt * 3.5f);                // roll response
                f.up = glm::normalize(glm::mix(f.up, up_t, k));
                f.up = glm::normalize(f.up - f.fwd * glm::dot(f.up, f.fwd));
            }
            // Energy: climbing bleeds speed, diving gains; throttle
            // pulls back toward the commanded speed.
            f.speed += (-kG * f.fwd.y * 0.55f + (speed_cmd - f.speed) * 0.35f) * dt;
            f.speed = glm::clamp(f.speed, 70.0f, 185.0f);
            f.pos += f.fwd * f.speed * dt;
            if (f.pos.y < gh_here + 25.0f) f.pos.y = gh_here + 25.0f;   // hard floor
            if (f.t > 300.0f) { f.dead = true; }
        }

        // Exhaust trail sample.
        f.trail_acc += dt;
        if (!f.dead && f.trail_acc >= 0.04f) {
            f.trail_acc = 0.0f;
            f.trail.push_back({ f.pos - f.fwd * (half_len * 0.9f), f.t, f.turn_g });
        }
        (void)half_len;
    }

    // Periodic AI trace (every 10 s) - cheap, helps tune behaviour.
    static float ai_log_t = 0.0f;
    ai_log_t += dt;
    if (ai_log_t > 10.0f) {
        ai_log_t = 0.0f;
        static const char* kModeName[] = { "ingress", "orbit", "bombrun", "loop", "turn", "egress", "crash" };
        for (const auto& f : spacejet_flights_) {
            if (f.dead) continue;
            log::infof("[jets] id %u %s%s pos (%.0f, %.0f, %.0f) agl %.0f spd %.0f g %.1f",
                       f.id, f.leader_id ? "wing " : "lead ", kModeName[int(f.mode)],
                       f.pos.x, f.pos.y, f.pos.z,
                       f.pos.y - std::max(sample_terrain_height(f.pos.x, f.pos.z), rt_.water_level),
                       f.speed, f.turn_g);
        }
    }
    spacejet_flights_.erase(
        std::remove_if(spacejet_flights_.begin(), spacejet_flights_.end(),
                       [](const SpacejetFlight& f) { return f.dead && f.trail.empty(); }),
        spacejet_flights_.end());
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
        // No texture: albedo/normal index -1 (was left at 0, so the gun
        // was triplanar-tinted with the ground texture), uv scale 1.
        pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
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
    // Panels now swing INWARD (toward the courtyard): the gatehouse has a
    // round-arched front passage that an outward swing would clip into.
    // Left panel: hinge on -X side, swings outward (toward +Z) by ~85В°.
    doors_.push_back(make_door(-ent_half, +panel_w_half + 0.01f, -1.48f));
    // Right panel: hinge on +X side, swings outward by -85В°.
    doors_.push_back(make_door(+ent_half, -panel_w_half - 0.01f, +1.48f));
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
