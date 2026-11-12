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
#include <cstdlib>

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
    auto add_vehicle = [&](float x, float z, float y, float yaw, int kind = 0) {
        AaVehicle v{};
        v.kind = kind;
        v.pos = glm::vec3(x, y, z);
        v.body_yaw = yaw;
        v.turret_yaw = frand_range(rng, -3.0f, 3.0f);
        v.gun_pitch = kind == 1 ? 0.12f : 0.3f;
        v.cooldown = kind == 1 ? frand_range(rng, 3.0f, 9.0f) : 0.0f;
        v.radar_spin = frand_range(rng, 0.0f, 6.28f);
        aa_.push_back(v);
        // Hull collision (axis-aligned bound of the rotated hull).
        float c = std::abs(std::cos(v.body_yaw)), s = std::abs(std::sin(v.body_yaw));
        glm::vec3 he(c * 1.7f + s * 3.5f, 1.4f, s * 1.7f + c * 3.5f);
        world_.aabbs.push_back({ v.pos + glm::vec3(-he.x, -0.3f, -he.z),
                                 v.pos + glm::vec3(he.x, 2.6f, he.z) });
    };
    auto footprint = [&](float x, float z, float& hmin, float& hmax) {
        hmin = hmax = h_at(x, z);
        for (int k = 0; k < 8; ++k) {
            float aa = float(k) * 0.785f;
            float hh = h_at(x + std::cos(aa) * 3.6f, z + std::sin(aa) * 3.6f);
            hmin = std::min(hmin, hh); hmax = std::max(hmax, hh);
        }
    };
    // One vehicle parked on the verge of the castle road, just outside the
    // gate, hull along the road: the flattest spot 8-11 m off the centreline
    // within the first stretch of road.
    if (!castle_road_.empty()) {
        const auto& P = castle_road_.pts;
        float best = 1e9f; glm::vec3 bp(0.0f); float byaw = 0.0f;
        float arc = 0.0f;
        for (size_t i = 1; i + 1 < P.size(); ++i) {
            arc += glm::distance(P[i], P[i - 1]);
            if (arc < 16.0f) continue;
            if (arc > 75.0f) break;
            glm::vec2 t = glm::normalize(glm::vec2(P[i + 1].x - P[i - 1].x, P[i + 1].z - P[i - 1].z));
            glm::vec2 n(-t.y, t.x);
            for (float side : { -1.0f, 1.0f }) {
                for (float off : { 8.0f, 9.5f, 11.0f }) {
                    float x = P[i].x + n.x * side * off, z = P[i].z + n.y * side * off;
                    if (std::max(std::abs(x), std::abs(z)) < 22.0f) continue;   // clear of the walls
                    float hmin, hmax; footprint(x, z, hmin, hmax);
                    if (hmin < rt_.water_level + 0.8f) continue;
                    float score = (hmax - hmin) + std::abs(hmax - P[i].y) * 0.5f + off * 0.02f;
                    if (score < best) {
                        best = score; bp = glm::vec3(x, (hmin + hmax) * 0.5f, z);
                        byaw = std::atan2(t.x, t.y);
                    }
                }
            }
        }
        if (best < 1.6f) add_vehicle(bp.x, bp.z, bp.y, byaw);
    }
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
            add_vehicle(x, z, (hmin + h) * 0.5f, frand_range(rng, 0.0f, 6.2831853f));
            break;
        }
    }
    // Missile launchers: a few further out, on open flat ground.
    const float sam_ang[3] = { 1.15f, 3.25f, 5.35f };
    for (float ang0 : sam_ang) {
        for (int tries = 0; tries < 600; ++tries) {
            float a = ang0 + frand_range(rng, -0.6f, 0.6f);
            float d = frand_range(rng, 110.0f, 210.0f);
            float x = std::cos(a) * d, z = std::sin(a) * d;
            float hmin, hmax; footprint(x, z, hmin, hmax);
            if (hmin < rt_.water_level + 1.2f || hmax - hmin > 0.9f) continue;
            if (!castle_road_.empty() && castle_road_.distance(x, z) < 9.0f) continue;
            bool clash = false;
            for (const auto& o : aa_) if (glm::distance(glm::vec2(o.pos.x, o.pos.z), glm::vec2(x, z)) < 40.0f) clash = true;
            if (clash) continue;
            add_vehicle(x, z, (hmin + hmax) * 0.5f, frand_range(rng, 0.0f, 6.2831853f), 1);
            break;
        }
    }
    for (const auto& v : aa_) log::infof("[aa] %s at (%.0f, %.1f, %.0f)", v.kind == 1 ? "missile launcher" : "gun vehicle", v.pos.x, v.pos.y, v.pos.z);
    log::infof("[aa] %zu anti-aircraft vehicles placed", aa_.size());
}

void VulkanEngine::init_aa_meshes() {
    if (aa_.empty()) return;
    size_t tris = 0;
    auto upload = [&](const game::LevelMesh& lm) {
        CastleMeshGpu g{};
        g.mesh = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                       reinterpret_cast<const Vertex*>(lm.verts.data()),
                                       uint32_t(lm.verts.size()), lm.idx.data(), uint32_t(lm.idx.size()));
        g.tex_albedo = lm.tex_albedo; g.tex_normal = lm.tex_normal;
        g.tint = lm.tint; g.fallback = lm.fallback;
        g.bmin = lm.bmin; g.bmax = lm.bmax;
        g.uv_scale = lm.uv_scale;
        tris += lm.idx.size() / 3;
        return g;
    };
    for (int vt = 0; vt < 2; ++vt) {
        AaModel M = vt == 0 ? build_aa_model() : build_sam_model();
        aa_model_pivots_[vt][0] = M.turret_pivot;
        aa_model_pivots_[vt][1] = M.gun_pivot;
        aa_model_pivots_[vt][2] = M.radar_pivot;
        if (vt == 0) { aa_muzzle_[0] = M.muzzle[0]; aa_muzzle_[1] = M.muzzle[1]; }
        else for (int k = 0; k < 4; ++k) sam_rail_[k] = M.rail[k];
        for (int p = 0; p < AaModel::kPartCount; ++p) {
            for (const auto& lm : M.meshes[p]) {
                if (lm.idx.empty()) continue;
                aa_parts_.push_back(upload(lm));
                aa_part_kind_.push_back(p);
                aa_part_vtype_.push_back(vt);
            }
        }
        for (const auto& lm : M.missile)
            if (!lm.idx.empty()) sam_missile_parts_.push_back(upload(lm));
    }
    log::infof("[aa] vehicle models: %zu mesh parts, %zu tris", aa_parts_.size() + sam_missile_parts_.size(), tris);
}

void VulkanEngine::destroy_aa() {
    for (auto* list : { &aa_parts_, &sam_missile_parts_ }) {
        for (auto& g : *list) {
            if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
            if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
            if (g.mesh.vertex_buffer) destroy_mesh(allocator_, g.mesh);
        }
        list->clear();
    }
    aa_part_kind_.clear();
    aa_part_vtype_.clear();
}

void VulkanEngine::aa_part_models(const AaVehicle& v, bool prev, glm::mat4 out[4]) const {
    const float by = v.body_yaw;
    const float ty = prev ? v.prev_turret_yaw : v.turret_yaw;
    const float gp = prev ? v.prev_gun_pitch : v.gun_pitch;
    const float rs = prev ? v.prev_radar_spin : v.radar_spin;
    out[0] = T(v.pos) * RY(by);
    out[1] = out[0] * T(aa_model_pivots_[v.kind][0]) * RY(ty);
    out[2] = out[1] * T(aa_model_pivots_[v.kind][1]) * RX(-gp);
    out[3] = out[1] * T(aa_model_pivots_[v.kind][2]) * RY(rs);
}

void VulkanEngine::tick_aa(float dt) {
    if (aa_.empty()) return;
    if (dt <= 0.0f) {
        for (auto& v : aa_) {   // frozen: zero motion vectors
            v.prev_turret_yaw = v.turret_yaw;
            v.prev_gun_pitch = v.gun_pitch;
            v.prev_radar_spin = v.radar_spin;
        }
        for (auto& s : sam_missiles_) { s.prev_pos = s.pos; s.prev_dir = s.dir; }
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
        if (v.kind == 1) continue;   // missile launchers: tick_sam

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

        // Fire control: bursts when on target. A burst is a random number
        // of rounds; the sound is the recorded 12-round burst, entered late
        // for the odd rounds (so 5 rounds = the recording's last 5 shots +
        // its natural tail) and replayed whole for every further 12.
        v.cooldown -= dt;
        const bool on_target = tgt && std::abs(dy) < 0.035f && std::abs(dp) < 0.035f;
        const float interval = 0.080f;     // 12.5 rds/s (alternating barrels)
        const int   kClipRounds = 12;
        const glm::vec3 snd_pos = glm::vec3(v.pos) + glm::vec3(0.0f, 2.0f, 0.0f);
        if (on_target && v.cooldown <= 0.0f && v.rounds_left <= 0) {
            // Mostly short taps, sometimes a long hose.
            const float u = frand(R);
            v.rounds_left = u < 0.35f ? int(frand_range(R, 3.0f, 8.99f))
                          : u < 0.80f ? int(frand_range(R, 9.0f, 18.99f))
                                      : int(frand_range(R, 19.0f, 34.99f));
            v.fire_acc = interval;                      // first round now
            const int first = (v.rounds_left - 1) % kClipRounds + 1;
            v.snd_next = float(first) * interval;
            if (audio_) play_aa_burst_audio(snd_pos, float(kClipRounds - first) * interval);
        }
        if (v.rounds_left > 0) {
            v.fire_acc += dt;
            v.snd_next -= dt;
            if (v.snd_next <= 0.0f && v.rounds_left > 1) {
                v.snd_next += float(kClipRounds) * interval;
                if (audio_) play_aa_burst_audio(snd_pos, 0.0f);
            }
            aa_part_models(v, false, m);
            while (v.fire_acc >= interval && v.rounds_left > 0) {
                v.fire_acc -= interval;
                --v.rounds_left;
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
            if (v.rounds_left <= 0) {
                v.cooldown = frand(R) < 0.3f ? frand_range(R, 0.25f, 0.7f) : frand_range(R, 0.9f, 3.2f);
                v.fire_acc = 0.0f;
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
    tick_sam(dt);
}

// ---------------------------------------------------------------------
// Surface-to-air missiles
// ---------------------------------------------------------------------
namespace {
constexpr float kSamRange     = 2300.0f;
constexpr float kSamMaxSpeed  = 540.0f;   // m/s after the boost
constexpr float kSamAccel     = 270.0f;   // m/s^2 boost
constexpr float kSamBurn      = 7.0f;     // s of motor (glow + thick smoke)
constexpr float kSamTurn      = 1.9f;     // rad/s at full speed
constexpr float kSamFuse      = 11.0f;    // proximity fuse radius (m, x jet scale)
constexpr float kSamLife      = 16.0f;
}  // namespace

glm::mat4 VulkanEngine::sam_missile_model(glm::vec3 pos, glm::vec3 dir) const {
    glm::vec3 z = glm::normalize(dir);
    glm::vec3 ref = std::abs(z.y) > 0.98f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 x = glm::normalize(glm::cross(ref, z));
    glm::vec3 y = glm::cross(z, x);
    return glm::mat4(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f), glm::vec4(z, 0.0f), glm::vec4(pos, 1.0f));
}

void VulkanEngine::tick_sam(float dt) {
    uint32_t& R = aa_rng_;
    auto find_jet = [&](uint32_t id) -> SpacejetFlight* {
        for (auto& f : spacejet_flights_)
            if (f.id == id && !f.dead && !f.destroying) return &f;
        return nullptr;
    };

    // ---- Launchers: acquire, slew, lock, launch ----
    for (auto& v : aa_) {
        if (v.kind != 1) continue;
        if (v.rails == 0u) {
            v.reload_t -= dt;
            if (v.reload_t <= 0.0f) v.rails = 0xFu;
        }
        v.cooldown -= dt;
        // Test hook: QLIKE_TEST_SAM=1 makes the launchers fire as fast as they can lock.
        static const bool kTestSam = std::getenv("QLIKE_TEST_SAM") != nullptr;
        if (kTestSam) v.cooldown = std::min(v.cooldown, 0.5f);
        // Nearest live jet that no missile is already flying at.
        const SpacejetFlight* tgt = nullptr;
        float best = kSamRange;
        for (const auto& f : spacejet_flights_) {
            if (f.dead || f.destroying) continue;
            if (f.pos.y < v.pos.y + 30.0f) continue;
            bool engaged = false;
            for (const auto& s : sam_missiles_) if (!s.dead && s.target_id == f.id) engaged = true;
            for (const auto& o : aa_)
                if (&o != &v && o.kind == 1 && o.lock_id == f.id && o.lock_t > 0.0f) engaged = true;
            if (engaged) continue;
            float d = glm::distance(f.pos, v.pos);
            if (d < best) { best = d; tgt = &f; }
        }
        glm::mat4 m[4];
        aa_part_models(v, false, m);
        const glm::vec3 pivot = glm::vec3(m[2][3]);
        float want_yaw = v.turret_yaw, want_pitch = 0.12f;      // stowed: frame nearly flat
        if (tgt && v.rails != 0u) {
            // Point the frame ahead of the target and well above it: the
            // missile climbs out and the guidance does the rest.
            glm::vec3 aim = tgt->pos + tgt->fwd * tgt->speed * (best / kSamMaxSpeed);
            glm::vec3 d = aim - pivot;
            want_yaw = wrap_pi(std::atan2(d.x, d.z) - v.body_yaw);
            want_pitch = glm::clamp(std::atan2(d.y, glm::length(glm::vec2(d.x, d.z))) + 0.12f, 0.35f, 1.15f);
        }
        float dy = wrap_pi(want_yaw - v.turret_yaw);
        v.turret_yaw = wrap_pi(v.turret_yaw + glm::clamp(dy, -0.9f * dt, 0.9f * dt));
        float dp = want_pitch - v.gun_pitch;
        v.gun_pitch += glm::clamp(dp, -0.45f * dt, 0.45f * dt);

        if (!tgt || v.rails == 0u || v.cooldown > 0.0f) { v.lock_t = 0.0f; v.lock_id = 0xFFFFFFFFu; continue; }
        if (v.lock_id != tgt->id) { v.lock_id = tgt->id; v.lock_t = 0.0f; }
        if (std::abs(dy) < 0.10f && std::abs(dp) < 0.10f) v.lock_t += dt;
        if (v.lock_t < 2.6f) continue;

        // Launch from a random loaded rail.
        int loaded[4], nl = 0;
        for (int k = 0; k < 4; ++k) if (v.rails & (1u << k)) loaded[nl++] = k;
        const int rail = loaded[std::min(nl - 1, int(frand(R) * float(nl)))];
        v.rails &= ~(1u << rail);
        aa_part_models(v, false, m);
        SamMissile s{};
        s.pos = glm::vec3(m[2] * glm::vec4(sam_rail_[rail], 1.0f));
        s.dir = glm::normalize(glm::mat3(m[2]) * glm::vec3(0, 0, 1));
        s.prev_pos = s.pos; s.prev_dir = s.dir;
        s.speed = 20.0f;
        s.target_id = tgt->id;
        s.will_decoy = frand(R) < 0.38f;
        sam_missiles_.push_back(std::move(s));
        v.cooldown = frand_range(R, 14.0f, 28.0f);
        v.lock_t = 0.0f; v.lock_id = 0xFFFFFFFFu;
        if (v.rails == 0u) v.reload_t = 40.0f;
        queue_distant_sound(int(ClipID::SamLaunch), glm::vec3(m[2][3]), 0.0f, 2.2f, false);
        log::infof("[aa] missile away at jet %u (%.0f m)", tgt->id, best);
    }

    // ---- Flares ----
    // Test hook: QLIKE_TEST_FLARE=1 dumps a flare string ahead of the camera.
    static const bool kTestFlare = std::getenv("QLIKE_TEST_FLARE") != nullptr;
    if (kTestFlare && flares_.empty()) {
        const glm::vec3 fw(std::sin(player_.yaw), 0.0f, -std::cos(player_.yaw));
        const glm::vec3 rt(-fw.z, 0.0f, fw.x);
        for (int k = 0; k < 6; ++k) {
            Flare fl{};
            fl.pos = player_.position + fw * 120.0f + glm::vec3(0.0f, 70.0f, 0.0f);
            fl.vel = rt * (90.0f - float(k) * 9.0f) + fw * ((k & 1 ? 1.0f : -1.0f) * frand_range(R, 14.0f, 30.0f)) -
                     glm::vec3(0.0f, frand_range(R, 8.0f, 22.0f), 0.0f);
            fl.life = frand_range(R, 2.8f, 4.2f);
            fl.id = flare_next_id_++;
            flares_.push_back(fl);
        }
    }
    for (auto& fl : flares_) {
        fl.t += dt;
        fl.vel.y -= 9.81f * 0.55f * dt;                       // light, draggy
        fl.vel *= 1.0f / (1.0f + 0.9f * dt);
        fl.pos += fl.vel * dt;
    }
    flares_.erase(std::remove_if(flares_.begin(), flares_.end(),
                                 [](const Flare& f) { return f.t > f.life; }), flares_.end());

    // ---- Missiles ----
    auto prune_trail = [](SamMissile& s) {
        size_t keep = 0;
        while (keep < s.trail.size() && s.t - s.trail[keep].w > 8.0f) ++keep;
        if (keep) s.trail.erase(s.trail.begin(), s.trail.begin() + std::ptrdiff_t(keep));
    };
    for (auto& s : sam_missiles_) {
        s.prev_pos = s.pos; s.prev_dir = s.dir;
        s.t += dt;
        // Detonated: only its smoke trail lives on until it has thinned out.
        if (s.dead) { prune_trail(s); continue; }
        s.speed = std::min(kSamMaxSpeed, s.speed + kSamAccel * dt);
        SpacejetFlight* tgt = find_jet(s.target_id);
        // Countermeasures: with the missile closing, the jet dumps a string
        // of flares; some seekers transfer their lock to one of them.
        if (tgt && !s.flared && glm::distance(tgt->pos, s.pos) < 480.0f) {
            s.flared = true;
            glm::vec3 right = glm::normalize(glm::cross(tgt->fwd, tgt->up));
            for (int k = 0; k < 6; ++k) {
                Flare fl{};
                fl.pos = tgt->pos - tgt->fwd * 3.0f;
                fl.vel = tgt->fwd * (tgt->speed * 0.55f) + right * ((k & 1 ? 1.0f : -1.0f) * frand_range(R, 14.0f, 30.0f)) -
                         tgt->up * frand_range(R, 8.0f, 22.0f) - tgt->fwd * (float(k) * 9.0f);
                fl.life = frand_range(R, 2.8f, 4.2f);
                fl.id = flare_next_id_++;
                flares_.push_back(fl);
                if (s.will_decoy && k == 2) s.flare_id = fl.id;
            }
            if (s.flare_id) { s.target_id = 0xFFFFFFFFu; tgt = nullptr; }
        }
        const Flare* chase = nullptr;
        if (s.flare_id) for (const auto& fl : flares_) if (fl.id == s.flare_id) chase = &fl;
        if (chase) {
            glm::vec3 to = chase->pos + chase->vel * 0.15f - s.pos;
            float dist = glm::length(to);
            glm::vec3 want = to / std::max(dist, 1e-3f);
            float ang = std::acos(glm::clamp(glm::dot(s.dir, want), -1.0f, 1.0f));
            float lim = kSamTurn * dt;
            s.dir = ang <= lim ? want : glm::normalize(glm::mix(s.dir, want, lim / ang));
            if (dist < 14.0f) { s.dead = true; spawn_explosion(s.pos + s.dir * 5.5f, 0.45f, false); continue; }
        } else if (tgt) {
            if (s.t > 0.35f) {          // clear the rail first
                // Lead pursuit: aim at where the jet will be at intercept.
                glm::vec3 vel = tgt->fwd * tgt->speed;
                glm::vec3 aim = tgt->pos;
                for (int it = 0; it < 3; ++it)
                    aim = tgt->pos + vel * (glm::distance(aim, s.pos) / std::max(s.speed, 150.0f));
                glm::vec3 want = glm::normalize(aim - s.pos);
                float ang = std::acos(glm::clamp(glm::dot(s.dir, want), -1.0f, 1.0f));
                float lim = kSamTurn * glm::clamp(s.speed / kSamMaxSpeed, 0.35f, 1.0f) * dt;
                if (ang <= lim) s.dir = want;
                else s.dir = glm::normalize(glm::mix(s.dir, want, lim / ang));
            }
        } else {
            // Target gone: coast, nose slowly dropping, then self-destruct.
            s.lost_t += dt;
            s.dir = glm::normalize(s.dir + glm::vec3(0.0f, -0.12f * dt, 0.0f));
        }
        s.pos += s.dir * (s.speed * dt);

        // Smoke trail samples at the nozzle.
        s.trail_acc += dt;
        if (s.trail_acc >= 0.03f) { s.trail_acc = 0.0f; s.trail.push_back(glm::vec4(s.pos, s.t)); }
        prune_trail(s);

        // Proximity fuse (closest approach along this step).
        if (tgt) {
            glm::vec3 seg = s.pos - s.prev_pos;
            glm::vec3 rel = seg - tgt->fwd * (tgt->speed * dt);     // motion relative to the jet
            float l2 = glm::dot(rel, rel);
            float tt = l2 > 1e-6f ? glm::clamp(glm::dot(tgt->prev_pos - s.prev_pos, rel) / l2, 0.0f, 1.0f) : 0.0f;
            glm::vec3 nose = s.prev_pos + seg * tt + s.dir * 5.5f;
            glm::vec3 jet  = glm::mix(tgt->prev_pos, tgt->pos, tt);
            if (glm::distance(nose, jet) < kSamFuse * tgt->scale) {
                s.dead = true;
                spawn_explosion(glm::mix(nose, jet, 0.5f), 0.6f, false);
                tgt->hit_flash_t = 0.15f;
                tgt->hp = 0;
                tgt->destroying = true;
                tgt->destroy_t = 0.0f;
                log::infof("[aa] jet %u hit by missile", tgt->id);
                continue;
            }
        }
        const float gh = std::max(sample_terrain_height(s.pos.x, s.pos.z), rt_.water_level);
        if (s.pos.y < gh) {
            s.dead = true;
            spawn_explosion(glm::vec3(s.pos.x, gh, s.pos.z), 0.5f, true);
        } else if (s.lost_t > 1.3f || s.t > kSamLife) {
            s.dead = true;
            spawn_explosion(s.pos, 0.35f, false);
        }
    }
    sam_missiles_.erase(std::remove_if(sam_missiles_.begin(), sam_missiles_.end(),
                                       [](const SamMissile& s) { return s.dead && s.trail.empty(); }),
                        sam_missiles_.end());
}

void VulkanEngine::draw_aa(VkCommandBuffer cmd, const glm::mat4& vp, const glm::mat4& prev_vp,
                           const Frustum& fr, bool depth_only, VkPipelineLayout shadow_layout) {
    if (aa_.empty() || aa_parts_.empty()) return;
    const bool tex_on = rt_.textures_enabled;
    VkDeviceSize off = 0;
    auto draw_part = [&](const CastleMeshGpu& g, const glm::mat4& model, const glm::mat4& prev_model) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, g.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        if (shadow_layout) {
            glm::mat4 mvp = vp * model;
            vkCmdPushConstants(cmd, shadow_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
        } else {
            PushConstants pc{};
            pc.mvp = vp * model;
            pc.model = model;
            pc.prev_mvp = prev_vp * prev_model;
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
    };
    for (const auto& v : aa_) {
        if (!aabb_visible(fr, v.pos - glm::vec3(9.0f), v.pos + glm::vec3(9.0f))) continue;
        glm::mat4 m[4], pm[4];
        aa_part_models(v, false, m);
        aa_part_models(v, true, pm);
        for (size_t i = 0; i < aa_parts_.size(); ++i) {
            if (aa_part_vtype_[i] != v.kind) continue;
            const int p = aa_part_kind_[i];
            draw_part(aa_parts_[i], m[p], pm[p]);
        }
        if (v.kind == 1) {      // missiles still on their rails
            for (int k = 0; k < 4; ++k) {
                if (!(v.rails & (1u << k))) continue;
                const glm::mat4 t = T(sam_rail_[k]);
                for (const auto& g : sam_missile_parts_) draw_part(g, m[2] * t, pm[2] * t);
            }
        }
    }
    for (const auto& s : sam_missiles_) {
        if (s.dead) continue;
        if (!aabb_visible(fr, s.pos - glm::vec3(6.0f), s.pos + glm::vec3(6.0f))) continue;
        const glm::mat4 model = sam_missile_model(s.pos, s.dir);
        const glm::mat4 prev  = sam_missile_model(s.prev_pos, s.prev_dir);
        for (const auto& g : sam_missile_parts_) draw_part(g, model, prev);
    }
}

} // namespace qlike
