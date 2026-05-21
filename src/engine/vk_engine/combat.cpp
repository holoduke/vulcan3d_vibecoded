// Combat: projectile spawn / update / impact, hit-spark particle physics +
// trail draw, hitscan ray test, and the "player walking pushes boxes"
// horizontal-impulse loop. All depend on `physics_` (Jolt) being live.

#include "engine/vk_engine/internal.h"
#include "engine/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace qlike {

void VulkanEngine::apply_player_pushes(glm::vec3 pre_velocity) {
    if (!physics_) return;
    // We push horizontally only; vertical pushes from walking would lift boxes
    // when the player walks "up against" them, which feels wrong.
    glm::vec3 horiz_vel(pre_velocity.x, 0.0f, pre_velocity.z);
    if (glm::dot(horiz_vel, horiz_vel) < 0.04f) return;  // below 0.2 m/s — skip

    // Player-AABB in world space, slightly inflated so we still register a
    // push when slide_move has parked the player a few cm short of the box.
    constexpr float kSkin = 0.06f;
    glm::vec3 phe = game::Player::kHalfExtents + glm::vec3(kSkin);
    glm::vec3 p_min = player_.position - phe;
    glm::vec3 p_max = player_.position + phe;

    // Effective player "mass" for the impulse equation. Real mass would be
    // ~70 kg; we use a higher figure so pushes feel meaty (a 1 m³ crate at
    // 1000 kg gets a noticeable shove rather than a polite nudge).
    constexpr float kPlayerMass = 320.0f;

    // Use the cached per-body AABB built by rebuild_tick_aabbs this tick —
    // saves a Jolt body-mutex acquire + 8-corner mat-vec per dyn prop.
    // Falls back to a fresh query if the cache slot is for a different body
    // or isn't populated yet.
    const bool have_cache = dyn_tick_aabb_cache_.size() == dyn_props_.size();
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const auto& dp = dyn_props_[i];
        glm::vec3 b_min, b_max;
        if (have_cache && dyn_tick_aabb_cache_[i].body_id == dp.body_id) {
            b_min = dyn_tick_aabb_cache_[i].aabb.min;
            b_max = dyn_tick_aabb_cache_[i].aabb.max;
        } else {
            glm::mat4 m;
            if (!physics_->get_body_world_matrix(dp.body_id, m)) continue;
            glm::vec3 he = dp.full_size * 0.5f;
            b_min = glm::vec3(std::numeric_limits<float>::max());
            b_max = glm::vec3(std::numeric_limits<float>::lowest());
            for (int j = 0; j < 8; ++j) {
                glm::vec4 c((j & 1) ? he.x : -he.x,
                            (j & 2) ? he.y : -he.y,
                            (j & 4) ? he.z : -he.z, 1.0f);
                glm::vec3 wc(m * c);
                b_min = glm::min(b_min, wc);
                b_max = glm::max(b_max, wc);
            }
        }

        bool overlap = p_max.x > b_min.x && p_min.x < b_max.x &&
                       p_max.y > b_min.y && p_min.y < b_max.y &&
                       p_max.z > b_min.z && p_min.z < b_max.z;
        if (!overlap) continue;

        // Direction from player toward the box center, projected to the
        // horizontal plane (no boost-jumping by walking into stuff).
        glm::vec3 box_center = (b_min + b_max) * 0.5f;
        glm::vec3 dir(box_center.x - player_.position.x,
                      0.0f,
                      box_center.z - player_.position.z);
        float d_len = glm::length(dir);
        if (d_len < 1e-3f) continue;
        dir /= d_len;

        // Speed at which the player WAS heading into the box.
        float into_box = glm::dot(horiz_vel, dir);
        if (into_box <= 0.1f) continue;

        // Impulse = effective_player_mass * speed_into_box. Jolt then divides
        // by the body's mass internally, so heavier crates accelerate less
        // for the same impulse — bigger boxes are stiffer.
        glm::vec3 impulse = dir * (into_box * kPlayerMass);
        physics_->apply_impulse_h(dp.jolt_handle, impulse);
    }
}

void VulkanEngine::fire_projectile(glm::vec3 origin, glm::vec3 direction) {
    if (!physics_) return;
    glm::vec3 dir = glm::normalize(direction);

    // Orientation: rotate the local +Y axis (cylinder mesh + Jolt cylinder both
    // align along Y) onto the firing direction so the bullet visually points
    // along its motion.
    glm::quat orient;
    {
        glm::vec3 from(0.0f, 1.0f, 0.0f);
        float dotp = glm::dot(from, dir);
        if (dotp > 0.99999f) {
            orient = glm::quat(1, 0, 0, 0);
        } else if (dotp < -0.99999f) {
            orient = glm::angleAxis(3.14159265f, glm::vec3(1, 0, 0));
        } else {
            glm::vec3 axis = glm::normalize(glm::cross(from, dir));
            orient = glm::angleAxis(std::acos(dotp), axis);
        }
    }

    // Muzzle: a short distance ahead of the eye so the bullet is immediately
    // visible and doesn't spawn inside the player capsule.
    glm::vec3 spawn = origin + dir * 0.5f;

    float speed = std::max(10.0f, game_.bullet_speed);
    uint32_t id = physics_->add_dynamic_cylinder_ccd(
        spawn, kProjectileRad, kProjectileHalf, orient,
        dir * speed, std::max(0.1f, game_.bullet_mass));
    if (id == 0) return;

    muzzle_flash_timer_ = kMuzzleFlashDuration;
    recoil_timer_       = kRecoilDuration;
    if (audio_) {
        // Shot is "local" — happens at the camera, mostly bypasses 3D
        // attenuation. Light pitch jitter so rapid fire doesn't sound
        // mechanical.
        audio_->play_local(ClipID::Shot, 0.7f, 0.05f, 0.06f);
    }

    Projectile p{};
    p.body_id = id;
    p.jolt_handle = physics_->handle_of(id);
    p.radius = kProjectileRad;
    p.half_length = kProjectileHalf;
    p.ttl = kProjectileTtl;
    p.initial_speed = speed;
    p.initial_dir = dir;
    p.color = glm::vec4(1.0f, 0.78f, 0.30f, 1.0f);   // warm tracer
    // Stay just below the bloom threshold so the bullet reads as a hot line
    // rather than a star-shaped halo.
    p.emissive = glm::vec3(0.95f, 0.70f, 0.25f);
    projectiles_.push_back(p);
}

void VulkanEngine::update_projectiles(float dt) {
    if (!physics_) return;
    for (auto it = projectiles_.begin(); it != projectiles_.end(); ) {
        it->ttl -= dt;
        bool drop = it->ttl <= 0.0f;
        bool was_impact = false;
        glm::vec3 hit_pos(0.0f);

        if (!drop) {
            glm::mat4 m;
            if (!physics_->get_body_world_matrix_h(it->jolt_handle, m)) {
                drop = true;
            } else if (m[3].y < -100.0f) {
                drop = true;     // fell out of the world
            } else {
                // Impact detection. After Jolt's step the bullet's velocity
                // tells us what happened. The previous heuristic only
                // checked the speed-along-fire-axis — that meant a shallow
                // glance off a wall (which keeps most of its speed and
                // deflects only ~10°) wasn't registered as an impact and
                // the bullet kept ricocheting. New rule: any noticeable
                // direction change kills the bullet. Only a near-zero
                // deflection (≤ 6° = `dot ≥ ~0.994`) lets it continue —
                // i.e. only super grazing hits ricochet.
                glm::vec3 v = physics_->get_linear_velocity_h(it->jolt_handle);
                float speed = glm::length(v);
                float min_speed = std::max(20.0f, it->initial_speed * 0.5f);
                bool dir_change = false;
                if (speed > 0.5f) {
                    float align = glm::dot(v / speed, it->initial_dir);
                    // cos(6°) ≈ 0.9945 — anything more deflected impacts.
                    if (align < 0.9945f) dir_change = true;
                }
                if (dir_change || speed < min_speed) {
                    drop = true;
                    was_impact = true;
                    hit_pos = glm::vec3(m[3]);
                }
            }
        }

        if (drop) {
            if (was_impact) {
                // Use the bullet's post-impact velocity as the spark spray
                // axis — that's the actual reflection direction.
                glm::vec3 post_vel = physics_->get_linear_velocity_h(it->jolt_handle);
                glm::vec3 reflect = glm::length(post_vel) > 0.5f
                                         ? glm::normalize(post_vel)
                                         : -it->initial_dir;
                glm::vec3 saved_dir = it->initial_dir;
                uint32_t  saved_body = it->body_id;

                // Order matters: remove the bullet body BEFORE the decal
                // raycast so the ray doesn't hit the cylinder itself
                // (that's what made decals float in midair). Then place
                // the decal BEFORE spawning sparks so the ray doesn't
                // hit the freshly-spawned spark spheres either.
                physics_->remove_body(saved_body);
                spawn_impact_decal(hit_pos, saved_dir);
                spawn_hit_particles(hit_pos, reflect, saved_dir);
                if (audio_) {
                    audio_->play_at(ClipID::Impact, hit_pos, 0.8f, 0.10f, 0.08f);
                }
            } else {
                physics_->remove_body(it->body_id);
            }
            it = projectiles_.erase(it);
        } else {
            ++it;
        }
    }
}

void VulkanEngine::spawn_hit_particles(glm::vec3 pos, glm::vec3 reflect_dir,
                                       glm::vec3 incoming_dir) {
    if (!physics_) return;
    if (static_cast<int>(particles_.size()) >= kMaxParticles) return;

    // Spray axis = reflection direction. Build an orthonormal basis so we
    // can sample the cone around it.
    glm::vec3 fwd = glm::length(reflect_dir) > 1e-3f
                        ? glm::normalize(reflect_dir)
                        : glm::vec3(0, 1, 0);
    glm::vec3 ref = std::abs(fwd.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 right = glm::normalize(glm::cross(ref, fwd));
    glm::vec3 up    = glm::cross(fwd, right);

    glm::vec3 incoming = glm::length(incoming_dir) > 1e-3f
                             ? glm::normalize(incoming_dir)
                             : -fwd;
    glm::vec3 back_to_shooter = -incoming;
    float head_on = glm::clamp(glm::dot(fwd, back_to_shooter), 0.0f, 1.0f);

    const float scale = std::max(0.05f, game_.spark_scale);
    int count = std::max(1, static_cast<int>(kParticlesPerHit * scale + 0.5f));
    for (int i = 0; i < count; ++i) {
        if (static_cast<int>(particles_.size()) >= kMaxParticles) break;

        bool wide = frand(spawn_rng_state_) < 0.15f;
        float cone_lo = wide ? -0.2f : 0.30f;
        float u = frand(spawn_rng_state_);
        float v = frand(spawn_rng_state_);
        float cone_cos = cone_lo + (1.0f - cone_lo) * (1.0f - u);
        float ang = 6.28318531f * v;
        float r = std::sqrt(std::max(0.0f, 1.0f - cone_cos * cone_cos));
        glm::vec3 dir = glm::normalize(fwd * cone_cos +
                                       right * (std::cos(ang) * r) +
                                       up    * (std::sin(ang) * r));

        float aligned = glm::clamp(cone_cos, 0.0f, 1.0f);
        float backward = glm::clamp(glm::dot(dir, back_to_shooter), 0.0f, 1.0f);
        float speed_envelope = (1.0f - 0.6f * head_on) *
                               (0.55f + 0.45f * aligned) *
                               (1.0f - 0.35f * backward);
        float jitter = frand_range(spawn_rng_state_, 0.85f, 1.15f);
        // ~4% of sparks get a "screamer" boost — flies way harder, gives
        // each impact a couple of long streakers among the cluster of
        // short hops. Looks like sparks ricocheting off something extra
        // hard. Bonus also kicks the upward y-velocity.
        bool screamer = frand(spawn_rng_state_) < 0.04f;
        float boost = screamer
                       ? frand_range(spawn_rng_state_, 3.0f, 5.0f)
                       : 1.0f;
        float speed = frand_range(spawn_rng_state_, 4.5f, 11.0f) *
                      speed_envelope * jitter * scale * boost;

        glm::vec3 vel = dir * speed;
        vel.y += frand_range(spawn_rng_state_, 0.6f, 2.2f) *
                 speed_envelope * scale * boost;

        float spawn_dist = frand_range(spawn_rng_state_, 0.05f, 0.18f);
        glm::vec3 spawn = pos + dir * spawn_dist;

        // Low restitution: sparks land with a small hop and stop, instead
        // of bouncing all over the room. Friction also bumped a little so
        // they don't slide on the floor.
        uint32_t id = physics_->add_dynamic_sphere(
            spawn, kParticleColRad, vel, /*mass*/ 0.02f,
            /*restitution*/ 0.20f, /*friction*/ 0.45f);
        if (id == 0) continue;

        Particle pa{};
        pa.body_id = id;
        pa.jolt_handle = physics_->handle_of(id);
        float ttl_jitter = frand_range(spawn_rng_state_, 0.75f, 1.25f);
        pa.ttl_max = kParticleTtl * ttl_jitter;
        pa.ttl = pa.ttl_max;
        pa.vis_radius = kParticleVisRad;
        pa.vis_base_half = kParticleVisBase;
        particles_.push_back(pa);
    }
}

void VulkanEngine::draw_decals(VkCommandBuffer cmd, const glm::mat4& vp) {
    if (decals_.empty()) return;
    // Bind cylinder mesh — its 16-segment side gives a near-round disc
    // when scaled to a thin Y. (The previous cube path showed obvious
    // square edges on close-up scorch marks.)
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &off);
    vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    for (const auto& d : decals_) {
        float life = 1.0f - (d.age / d.ttl);
        if (life <= 0.0f) continue;

        glm::vec3 world_pos    = d.pos;
        glm::vec3 world_normal = d.normal;
        if (d.parent_body_id != 0) {
            // Parented to a dyn body: transform local-frame pos+normal by
            // the body's CURRENT world matrix so the decal travels with
            // the crate. If the body has gone (despawn / FIFO eviction),
            // skip drawing — update_decals will reap it next tick.
            glm::mat4 body_world;
            if (!physics_->get_body_world_matrix_h(d.parent_handle, body_world)) {
                continue;
            }
            world_pos    = glm::vec3(body_world * glm::vec4(d.pos, 1.0f));
            world_normal = glm::vec3(body_world * glm::vec4(d.normal, 0.0f));
        }

        glm::mat4 model = align_local_y_to(world_pos, world_normal) *
                          glm::scale(glm::mat4(1.0f),
                                     glm::vec3(d.size, 0.002f, d.size));
        // Lighter base color — opaque pipeline can't blend with the wall
        // beneath, so a brighter scorch reads as more "tinted/transparent"
        // than the previous near-black 0.04*life. Fades with life.
        glm::vec3 col(0.18f * life);
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        pc.prev_mvp = pc.mvp;
        pc.color = glm::vec4(col, 1.0f);
        pc.emissive = glm::vec4(0.0f);
        pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
    }
}

void VulkanEngine::draw_spark_trails(VkCommandBuffer cmd, const glm::mat4& vp) {
    if (!physics_) return;

    auto draw_segment = [&](glm::vec3 a, glm::vec3 b, float radius,
                            const glm::vec3& emissive) {
        glm::vec3 mid = (a + b) * 0.5f;
        glm::vec3 d = b - a;
        float len = glm::length(d);
        if (len < 1e-4f) return;
        glm::vec3 dir = d / len;
        glm::mat4 model = align_local_y_to(mid, dir) *
                          glm::scale(glm::mat4(1.0f),
                                     glm::vec3(radius, len * 0.5f, radius));
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        // Spark trails are sub-pixel emissive cylinders — zero-motion approx.
        pc.prev_mvp = pc.mvp;
        pc.color = glm::vec4(glm::min(emissive, glm::vec3(1.0f)), 1.0f);
        pc.emissive = glm::vec4(emissive, 1.0f);    // full_emissive
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
    };

    for (auto& pa : particles_) {
        // Reuse the world matrix update_particles already cached this frame.
        // Falling back to a Jolt query here would re-acquire the body mutex
        // for every particle every frame — the dominant CPU cost when sparks
        // are dense.
        if (!pa.valid_world) continue;
        glm::vec3 pos(pa.world[3]);

        // Push current position into the history ring (newest at end).
        if (pa.trail_count < kSparkTrailLen) {
            pa.trail[pa.trail_count++] = pos;
        } else {
            for (int i = 0; i < kSparkTrailLen - 1; ++i) {
                pa.trail[i] = pa.trail[i + 1];
            }
            pa.trail[kSparkTrailLen - 1] = pos;
        }

        if (pa.trail_count < 2) continue;
        const float life_t = pa.ttl_max > 0.0f ? (pa.ttl / pa.ttl_max) : 0.0f;
        const float thin_factor = 0.7f;
        const float spark_bloom = std::max(0.0f, rt_.spark_bloom);
        // Scale visual radius by sqrt(spark_bloom). Bloom halo width is
        // limited by source spatial footprint — a 1-pixel emissive vanishes
        // by mip 2, so cranking emissive alone only brightens the center,
        // never widens the halo. Growing the cylinder gives the bloom
        // prefilter a 4–6 px source at high settings, which survives 3–4
        // mip levels and produces the wide soft glow the user expected.
        // sqrt keeps the slider feel linear (5x bloom ≈ 2.2x width).
        const float radius = pa.vis_radius * thin_factor *
                             std::max(1.0f, std::sqrt(spark_bloom));

        for (int i = 0; i < pa.trail_count - 1; ++i) {
            float u = static_cast<float>(i) / static_cast<float>(kSparkTrailLen - 1);
            float temp = std::max(0.0f, life_t * (0.30f + 0.70f * u));
            glm::vec3 emi = spark_blackbody(temp) * spark_bloom;
            draw_segment(pa.trail[i], pa.trail[i + 1], radius, emi);
        }
    }
}

void VulkanEngine::draw_shadow_debug(VkCommandBuffer cmd, const glm::mat4& vp) {
    if (!rt_.shadow_debug_overlay) return;

    // Re-use the cylinder mesh as a "line": thin radius, scaled along Y.
    auto draw_line = [&](glm::vec3 a, glm::vec3 b, glm::vec3 emissive,
                          float radius = 0.06f) {
        glm::vec3 mid = (a + b) * 0.5f;
        glm::vec3 d = b - a;
        float len = glm::length(d);
        if (len < 1e-3f) return;
        glm::vec3 dir = d / len;
        glm::mat4 model = align_local_y_to(mid, dir) *
                          glm::scale(glm::mat4(1.0f),
                                     glm::vec3(radius, len * 0.5f, radius));
        PushConstants pc{};
        pc.mvp = vp * model;
        pc.model = model;
        pc.prev_mvp = pc.mvp;
        pc.color = glm::vec4(glm::min(emissive, glm::vec3(1.0f)), 1.0f);
        pc.emissive = glm::vec4(emissive, 1.0f);  // .a > 0 → skip lighting in cube.frag
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
    };

    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &off);
    vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    // -------- Shadow ortho frustum (8 corners → 12 edges) --------
    // Recover world-space corners by inverting light_vp and projecting
    // the 8 NDC cube corners. Vulkan NDC: x,y ∈ [-1, 1], z ∈ [0, 1].
    glm::mat4 inv_light = glm::inverse(sun_shadow_light_vp_);
    glm::vec3 corners[8];
    int idx = 0;
    for (int z = 0; z <= 1; ++z) {
        for (int y = -1; y <= 1; y += 2) {
            for (int x = -1; x <= 1; x += 2) {
                glm::vec4 c = inv_light * glm::vec4(static_cast<float>(x),
                                                     static_cast<float>(y),
                                                     static_cast<float>(z),
                                                     1.0f);
                corners[idx++] = glm::vec3(c) / c.w;
            }
        }
    }
    // Edge index pairs for a [-1,1]³ cube ordered (x_low/high, y_low/high,
    // z_low/high): bottom 4, top 4, vertical 4.
    const int kEdges[12][2] = {
        {0,1}, {1,3}, {3,2}, {2,0},   // near plane (z=0)
        {4,5}, {5,7}, {7,6}, {6,4},   // far plane (z=1)
        {0,4}, {1,5}, {2,6}, {3,7},   // connecting verticals
    };
    const glm::vec3 kFrustumColor(2.0f, 2.0f, 0.0f);  // bright yellow
    for (const auto& e : kEdges) {
        draw_line(corners[e[0]], corners[e[1]], kFrustumColor, 0.08f);
    }

    // -------- Cross-fade boundary circle (bake / shadow-map seam) ----
    // Approximate the smoothstep boundary as a horizontal circle around
    // the player at height = camera Y, radius = shadow_map_world_half.
    // Visualises where grass switches from shadow map → bake.
    glm::vec3 cam = player_.eye_position();
    const float kBoundaryR = std::max(rt_.shadow_map_world_half, 30.0f);
    const int kSegs = 48;
    const glm::vec3 kBoundaryColor(0.4f, 0.4f, 3.0f);  // bright blue
    glm::vec3 prev = cam + glm::vec3(kBoundaryR, 0.0f, 0.0f);
    for (int i = 1; i <= kSegs; ++i) {
        float a = (static_cast<float>(i) / kSegs) * 6.2831853f;
        glm::vec3 cur = cam + glm::vec3(std::cos(a) * kBoundaryR,
                                          0.0f,
                                          std::sin(a) * kBoundaryR);
        draw_line(prev, cur, kBoundaryColor, 0.10f);
        prev = cur;
    }

    // -------- Sun direction line (shadow camera "look") --------------
    // Anchor at the player; length matches the light eye distance.
    float p_rad = glm::radians(rt_.sun_pitch_deg);
    float y_rad = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun_dir = glm::normalize(glm::vec3(
        std::sin(y_rad) * std::cos(p_rad), std::sin(p_rad),
        std::cos(y_rad) * std::cos(p_rad)));
    draw_line(cam, cam + sun_dir * 20.0f,
              glm::vec3(3.0f, 1.5f, 0.0f),  // orange — sunwards
              0.12f);

    // -------- Heightmap bake world bounds (outermost reach) ----------
    // The bake covers [-side/2, +side/2] in X and Z. Drawn as a light
    // grey square at terrain Y so the user can see how far the bake
    // extends compared to the shadow map.
    if (terrain_data_.dim > 0) {
        float side = static_cast<float>(terrain_data_.dim) * terrain_data_.cell;
        float h = 0.5f;
        glm::vec3 a(-side * 0.5f, h, -side * 0.5f);
        glm::vec3 b( side * 0.5f, h, -side * 0.5f);
        glm::vec3 c( side * 0.5f, h,  side * 0.5f);
        glm::vec3 d(-side * 0.5f, h,  side * 0.5f);
        const glm::vec3 kBakeColor(1.5f, 1.5f, 1.5f);
        draw_line(a, b, kBakeColor, 0.15f);
        draw_line(b, c, kBakeColor, 0.15f);
        draw_line(c, d, kBakeColor, 0.15f);
        draw_line(d, a, kBakeColor, 0.15f);
    }
}

void VulkanEngine::spawn_impact_decal(glm::vec3 hit_pos, glm::vec3 incoming_dir) {
    if (!physics_) return;
    glm::vec3 dir = glm::length(incoming_dir) > 1e-3f
                        ? glm::normalize(incoming_dir)
                        : glm::vec3(0, 0, 1);
    glm::vec3 from = hit_pos - dir * 0.5f;
    auto rh = physics_->raycast(from, dir, 1.0f);
    glm::vec3 normal = rh.hit ? rh.normal : -dir;
    glm::vec3 pos    = rh.hit ? rh.position : hit_pos;
    pos += normal * 0.005f;     // anti-z-fight skin

    Decal d{};
    d.size = frand_range(spawn_rng_state_, 0.04f, 0.07f);
    d.ttl  = kDecalTtl;
    if (rh.hit && rh.dynamic && rh.body_id != 0) {
        // Hit a dyn box — parent the decal to it. Store pos/normal in
        // the body's LOCAL frame so render-time multiplies by current
        // world transform and the decal moves with the crate.
        glm::mat4 body_world;
        if (physics_->get_body_world_matrix(rh.body_id, body_world)) {
            glm::mat4 inv = glm::inverse(body_world);
            d.pos      = glm::vec3(inv * glm::vec4(pos, 1.0f));
            d.normal   = glm::vec3(inv * glm::vec4(normal, 0.0f));
            d.parent_body_id = rh.body_id;
            d.parent_handle  = physics_->handle_of(rh.body_id);
        } else {
            d.pos = pos; d.normal = normal;
        }
    } else {
        d.pos    = pos;
        d.normal = normal;
    }
    decals_.push_back(d);
    while (static_cast<int>(decals_.size()) > kMaxDecals) {
        decals_.erase(decals_.begin());
    }
}

void VulkanEngine::update_decals(float dt) {
    for (auto& d : decals_) d.age += dt;
    // Drop expired decals + decals whose parent body has been removed
    // (FIFO eviction of the dyn prop they were stuck to).
    decals_.erase(
        std::remove_if(decals_.begin(), decals_.end(),
                        [this](const Decal& d) {
                            if (d.age >= d.ttl) return true;
                            if (d.parent_body_id != 0 && physics_) {
                                glm::mat4 m;
                                if (!physics_->get_body_world_matrix_h(d.parent_handle, m)) {
                                    return true;
                                }
                            }
                            return false;
                        }),
        decals_.end());
}

void VulkanEngine::update_particles(float dt) {
    if (!physics_) return;
    for (auto it = particles_.begin(); it != particles_.end(); ) {
        it->ttl -= dt;
        bool drop = it->ttl <= 0.0f;
        // Cache the body's world matrix once per frame. rebuild_tlas (now
        // skips particles for TLAS) and draw_spark_trails both read from
        // it->world afterward instead of re-querying Jolt — saves two body-
        // mutex locks per particle per frame.
        it->valid_world = false;
        if (!drop) {
            if (!physics_->get_body_world_matrix_h(it->jolt_handle, it->world)) {
                drop = true;
            } else if (it->world[3].y < -100.0f) {
                drop = true;
            } else {
                it->valid_world = true;
            }
        }
        if (drop) {
            physics_->remove_body(it->body_id);
            it = particles_.erase(it);
        } else {
            ++it;
        }
    }
}

void VulkanEngine::try_fire_hitscan(glm::vec3 origin, glm::vec3 direction) {
    if (!physics_) return;
    auto hit = physics_->raycast(origin, glm::normalize(direction), 200.0f);
    if (!hit.hit) {
        log::infof("[shoot] miss origin=(%.2f,%.2f,%.2f) dir=(%.2f,%.2f,%.2f)",
                   origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
        return;
    }
    if (hit.dynamic && hit.body_id != 0) {
        // ~3 kg·m/s impulse along ray + a bit upward for satisfying flips.
        glm::vec3 imp = direction * 6.0f + glm::vec3(0.0f, 2.5f, 0.0f);
        physics_->apply_impulse(hit.body_id, imp);
        ++score_;
        log::infof("[shoot] HIT dyn body=%u dist=%.2f pos=(%.2f,%.2f,%.2f) score=%d",
                   hit.body_id, hit.distance,
                   hit.position.x, hit.position.y, hit.position.z, score_);
    } else {
        log::infof("[shoot] static dist=%.2f pos=(%.2f,%.2f,%.2f)",
                   hit.distance, hit.position.x, hit.position.y, hit.position.z);
    }
}

} // namespace qlike
