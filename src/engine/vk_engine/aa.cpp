// Anti-aircraft vehicles: procedural SPAAG model (aa_vehicle.cpp), target
// tracking with lead + drop compensation, slew-rate-limited turret and
// gun elevation, burst fire alternating barrels, ballistic red tracer
// rounds (drawn in fx.cpp) that can damage and down the jets.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/audio.h"
#include "engine/log.h"
#include "engine/terrain.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace qlike {

namespace {
constexpr float kRoundSpeed = 720.0f;    // m/s
constexpr float kG          = 9.81f;
constexpr float kRange      = 1150.0f;
glm::mat4 T(glm::vec3 t) { return glm::translate(glm::mat4(1.0f), t); }
glm::mat4 RY(float a) { return glm::rotate(glm::mat4(1.0f), a, glm::vec3(0, 1, 0)); }
glm::mat4 RX(float a) { return glm::rotate(glm::mat4(1.0f), a, glm::vec3(1, 0, 0)); }
float wrap_pi(float a) {
    while (a > 3.14159265f) a -= 6.2831853f;
    while (a < -3.14159265f) a += 6.2831853f;
    return a;
}
}  // namespace

void VulkanEngine::place_aa_vehicles(const Heightmap& hm) {
    aa_.clear();
    uint32_t rng = 0xA11A7u;
    auto h_at = [&](float x, float z) { return hm.sample_world(x, z); };
    const float want[6] = { 0.6f, 1.7f, 2.8f, 3.9f, 5.0f, 5.9f };   // spread around the castle
    for (float ang0 : want) {
        for (int tries = 0; tries < 400; ++tries) {
            float a = ang0 + frand_range(rng, -0.45f, 0.45f);
            float d = frand_range(rng, 48.0f, 105.0f);   // open meadow ring round the castle
            float x = std::cos(a) * d, z = std::sin(a) * d;
            float h = h_at(x, z);
            if (h < rt_.water_level + 1.2f) continue;
            if (!castle_road_.empty() && castle_road_.distance(x, z) < 9.0f) continue;
            // Flat enough under the whole hull footprint.
            float hmin = h, hmax = h;
            for (int k = 0; k < 8; ++k) {
                float aa = float(k) * 0.785f;
                float hh = h_at(x + std::cos(aa) * 3.6f, z + std::sin(aa) * 3.6f);
                hmin = std::min(hmin, hh); hmax = std::max(hmax, hh);
            }
            if (hmax - hmin > 0.9f) continue;
            bool clash = false;
            for (const auto& o : aa_) if (glm::distance(glm::vec2(o.pos.x, o.pos.z), glm::vec2(x, z)) < 40.0f) clash = true;
            if (clash) continue;
            AaVehicle v{};
            v.pos = glm::vec3(x, (hmin + h) * 0.5f, z);
            v.body_yaw = frand_range(rng, 0.0f, 6.2831853f);
            v.turret_yaw = frand_range(rng, -3.0f, 3.0f);
            v.gun_pitch = 0.3f;
            v.radar_spin = frand_range(rng, 0.0f, 6.28f);
            aa_.push_back(v);
            // Hull collision (axis-aligned bound of the rotated hull).
            float c = std::abs(std::cos(v.body_yaw)), s = std::abs(std::sin(v.body_yaw));
            glm::vec3 he(c * 1.7f + s * 3.5f, 1.4f, s * 1.7f + c * 3.5f);
            world_.aabbs.push_back({ v.pos + glm::vec3(-he.x, -0.3f, -he.z),
                                     v.pos + glm::vec3(he.x, 2.6f, he.z) });
            break;
        }
    }
    for (const auto& v : aa_) log::infof("[aa] vehicle at (%.0f, %.1f, %.0f)", v.pos.x, v.pos.y, v.pos.z);
    log::infof("[aa] %zu anti-aircraft vehicles placed", aa_.size());
}

void VulkanEngine::init_aa_meshes() {
    if (aa_.empty()) return;
    AaModel M = build_aa_model();
    aa_model_pivots_[0] = M.turret_pivot;
    aa_model_pivots_[1] = M.gun_pivot;
    aa_model_pivots_[2] = M.radar_pivot;
    aa_muzzle_[0] = M.muzzle[0];
    aa_muzzle_[1] = M.muzzle[1];
    size_t tris = 0;
    for (int p = 0; p < AaModel::kPartCount; ++p) {
        for (const auto& lm : M.meshes[p]) {
            if (lm.idx.empty()) continue;
            CastleMeshGpu g{};
            g.mesh = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                           reinterpret_cast<const Vertex*>(lm.verts.data()),
                                           uint32_t(lm.verts.size()), lm.idx.data(), uint32_t(lm.idx.size()));
            g.tex_albedo = lm.tex_albedo; g.tex_normal = lm.tex_normal;
            g.tint = lm.tint; g.fallback = lm.fallback;
            g.bmin = lm.bmin; g.bmax = lm.bmax;
            g.uv_scale = lm.uv_scale;
            aa_parts_.push_back(g);
            aa_part_kind_.push_back(p);
            tris += lm.idx.size() / 3;
        }
    }
    log::infof("[aa] vehicle model: %zu mesh parts, %zu tris", aa_parts_.size(), tris);
}

void VulkanEngine::destroy_aa() {
    for (auto& g : aa_parts_) {
        if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
        if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
        if (g.mesh.vertex_buffer) destroy_mesh(allocator_, g.mesh);
    }
    aa_parts_.clear();
    aa_part_kind_.clear();
}

void VulkanEngine::aa_part_models(const AaVehicle& v, bool prev, glm::mat4 out[4]) const {
    const float by = v.body_yaw;
    const float ty = prev ? v.prev_turret_yaw : v.turret_yaw;
    const float gp = prev ? v.prev_gun_pitch : v.gun_pitch;
    const float rs = prev ? v.prev_radar_spin : v.radar_spin;
    out[0] = T(v.pos) * RY(by);
    out[1] = out[0] * T(aa_model_pivots_[0]) * RY(ty);
    out[2] = out[1] * T(aa_model_pivots_[1]) * RX(-gp);
    out[3] = out[1] * T(aa_model_pivots_[2]) * RY(rs);
}

void VulkanEngine::tick_aa(float dt) {
    if (aa_.empty()) return;
    if (dt <= 0.0f) {
        for (auto& v : aa_) {   // frozen: zero motion vectors
            v.prev_turret_yaw = v.turret_yaw;
            v.prev_gun_pitch = v.gun_pitch;
            v.prev_radar_spin = v.radar_spin;
        }
        return;
    }
    tick_aa_sounds(dt);
    dt = std::min(dt, 0.05f);
    uint32_t& R = aa_rng_;

    for (auto& v : aa_) {
        v.prev_turret_yaw = v.turret_yaw;
        v.prev_gun_pitch = v.gun_pitch;
        v.prev_radar_spin = v.radar_spin;
        v.radar_spin += dt * 4.2f;
        if (v.flash_t > 0.0f) v.flash_t -= dt;

        // Target: nearest live jet in range above the horizon.
        const SpacejetFlight* tgt = nullptr;
        float best = kRange;
        for (const auto& f : spacejet_flights_) {
            if (f.dead || f.destroying) continue;
            float d = glm::distance(f.pos, v.pos);
            if (d < best && f.pos.y > v.pos.y + 15.0f) { best = d; tgt = &f; }
        }
        glm::mat4 m[4];
        aa_part_models(v, false, m);
        const glm::vec3 pivot = glm::vec3(m[2][3]);
        float want_yaw = v.turret_yaw, want_pitch = 0.12f;
        if (tgt) {
            // Lead + drop: iterate time of flight twice.
            glm::vec3 vel = tgt->fwd * tgt->speed;
            glm::vec3 aim = tgt->pos;
            for (int it = 0; it < 3; ++it) {
                float tof = glm::distance(aim, pivot) / kRoundSpeed;
                aim = tgt->pos + vel * tof + glm::vec3(0.0f, 0.5f * kG * tof * tof, 0.0f);
            }
            glm::vec3 d = aim - pivot;
            float wyaw = std::atan2(d.x, d.z);
            want_yaw = wrap_pi(wyaw - v.body_yaw);
            want_pitch = std::atan2(d.y, glm::length(glm::vec2(d.x, d.z)));
        } else {
            want_yaw = v.turret_yaw + std::sin(v.radar_spin * 0.15f) * 0.2f;   // idle scan
        }
        want_pitch = glm::clamp(want_pitch, -0.05f, 1.40f);
        float dy = wrap_pi(want_yaw - v.turret_yaw);
        v.turret_yaw = wrap_pi(v.turret_yaw + glm::clamp(dy, -1.7f * dt, 1.7f * dt));
        float dp = want_pitch - v.gun_pitch;
        v.gun_pitch += glm::clamp(dp, -1.2f * dt, 1.2f * dt);

        // Fire control: bursts when on target.
        v.cooldown -= dt;
        const bool on_target = tgt && std::abs(dy) < 0.035f && std::abs(dp) < 0.035f;
        // A burst is one or two plays of the recorded burst (12 rounds over
        // kAaClipShots s), so the gun stops when the recording's shots do.
        const float kAaClipShots = 0.96f;
        if (on_target && v.cooldown <= 0.0f && v.burst <= 0.0f) {
            v.burst = kAaClipShots * (frand(R) < 0.45f ? 2.0f : 1.0f);
            if (audio_) play_aa_burst_audio(glm::vec3(v.pos) + glm::vec3(0.0f, 2.0f, 0.0f));
        }
        if (v.burst > 0.0f) {
            const float burst_before = v.burst;
            v.burst -= dt;
            if (audio_ && burst_before > kAaClipShots && v.burst <= kAaClipShots)
                play_aa_burst_audio(glm::vec3(v.pos) + glm::vec3(0.0f, 2.0f, 0.0f));
            v.fire_acc += dt;
            // 12.5 rds/s (alternating barrels). The sound is the recording
            // itself (play_aa_burst_audio), not triggered per round.
            const float interval = 0.080f;
            aa_part_models(v, false, m);
            while (v.fire_acc >= interval) {
                v.fire_acc -= interval;
                int b = v.barrel; v.barrel ^= 1;
                glm::vec3 mz = glm::vec3(m[2] * glm::vec4(aa_muzzle_[b], 1.0f));
                glm::vec3 dir = glm::normalize(glm::mat3(m[2]) * glm::vec3(0, 0, 1));
                // Dispersion ~4 mrad.
                glm::vec3 rt = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0)));
                glm::vec3 up = glm::cross(rt, dir);
                dir = glm::normalize(dir + rt * frand_range(R, -0.010f, 0.010f) +
                                     up * frand_range(R, -0.010f, 0.010f));
                AaRound r{};
                r.pos = mz; r.prev = mz; r.vel = dir * kRoundSpeed;
                aa_rounds_.push_back(r);
                v.flash_t = 0.05f;
                v.flash_barrel = b;
            }
            if (v.burst <= 0.0f) {
                v.cooldown = frand_range(R, 0.8f, 2.2f); v.fire_acc = 0.0f;
            }
        }
    }

    // Rounds: ballistic, hit test against jets (swept sphere).
    for (auto& r : aa_rounds_) {
        r.prev = r.pos;
        r.t += dt;
        r.vel.y -= kG * dt;
        r.pos += r.vel * dt;
        if (r.t > 2.6f) { r.dead = true; continue; }
        float gh = std::max(sample_terrain_height(r.pos.x, r.pos.z), rt_.water_level);
        if (r.pos.y < gh) { r.dead = true; continue; }
        glm::vec3 seg = r.pos - r.prev;
        float sl2 = glm::dot(seg, seg);
        for (auto& f : spacejet_flights_) {
            if (f.dead || f.destroying) continue;
            float rad = 3.6f * f.scale;
            float tt = sl2 > 1e-6f ? glm::clamp(glm::dot(f.pos - r.prev, seg) / sl2, 0.0f, 1.0f) : 0.0f;
            glm::vec3 q = r.prev + seg * tt;
            if (glm::distance(q, f.pos) > rad) continue;
            r.dead = true;
            f.hit_flash_t = 0.12f;
            if (++f.aa_hits % 8 == 0) spawn_hit_particles(q, -glm::normalize(seg), glm::normalize(seg));
            if (f.aa_hits >= 32) {          // ~same time-to-kill at 12.5 rds/s
                f.hp = 0;
                f.destroying = true;
                f.destroy_t = 0.0f;
                spawn_explosion(f.pos, 0.55f, false);
                log::infof("[aa] jet %u shot down", f.id);
            }
            break;
        }
    }
    aa_rounds_.erase(std::remove_if(aa_rounds_.begin(), aa_rounds_.end(),
                                    [](const AaRound& r) { return r.dead; }),
                     aa_rounds_.end());
}

void VulkanEngine::draw_aa(VkCommandBuffer cmd, const glm::mat4& vp, const glm::mat4& prev_vp,
                           const Frustum& fr, bool depth_only, VkPipelineLayout shadow_layout) {
    if (aa_.empty() || aa_parts_.empty()) return;
    const bool tex_on = rt_.textures_enabled;
    VkDeviceSize off = 0;
    for (const auto& v : aa_) {
        if (!aabb_visible(fr, v.pos - glm::vec3(8.0f), v.pos + glm::vec3(8.0f))) continue;
        glm::mat4 m[4], pm[4];
        aa_part_models(v, false, m);
        aa_part_models(v, true, pm);
        for (size_t i = 0; i < aa_parts_.size(); ++i) {
            const auto& g = aa_parts_[i];
            const int p = aa_part_kind_[i];
            vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off);
            vkCmdBindIndexBuffer(cmd, g.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            if (shadow_layout) {
                glm::mat4 mvp = vp * m[p];
                vkCmdPushConstants(cmd, shadow_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
            } else {
                PushConstants pc{};
                pc.mvp = vp * m[p];
                pc.model = m[p];
                pc.prev_mvp = prev_vp * pm[p];
                pc.grass_params = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);   // static-mesh mode, vertex UVs
                if (!depth_only) {
                    pc.color = glm::vec4(tex_on ? g.tint : g.fallback, 1.0f);
                    pc.tex_params = (tex_on && g.tex_albedo >= 0)
                        ? glm::vec4(float(g.tex_albedo), float(g.tex_normal), g.uv_scale, 0.0f)
                        : glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
                }
                vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
            }
            vkCmdDrawIndexed(cmd, g.mesh.index_count, 1, 0, 0, 0);
        }
    }
}

} // namespace qlike
