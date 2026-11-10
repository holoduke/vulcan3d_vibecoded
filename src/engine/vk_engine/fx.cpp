// FX: bombs, layered explosions, contrails and laser beams.
//
// Explosions follow the usual AAA layering (flash -> fireball -> smoke ->
// ground dust -> debris), each layer short-lived and timed:
//   * flash        - additive radial billboard, ~0.3 s, + the dynamic
//                    point light (descriptors.cpp) + camera shake;
//   * fireball     - raymarched noise volume with blackbody emission that
//                    cools into soot (fx_volume.frag), ~3 s;
//   * smoke        - 5-7 billowing puffs (own velocity, drag, buoyancy,
//                    wind drift), churning noise, fray out over ~15 s;
//   * dust ring    - flattened ground-hugging volume racing outward, ~9 s;
//   * debris       - physics sparks (spawn_hit_particles) in an upward
//                    45-degree cone (ground bursts reflect upward), plus
//                    a scorch decal.
// Contrails / laser / tracers are CPU-built camera-facing ribbons drawn in
// the same pass (fx_ribbon.frag), premultiplied-alpha blended into the
// lit HDR scene after deferred lighting, so they never go through the
// G-buffer (the old opaque emissive cylinders showed dark TAA/deferred
// silhouettes around their edges).

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/audio.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef QLIKE_SHADER_DIR
#define QLIKE_SHADER_DIR "shaders"
#endif

namespace qlike {

namespace {
constexpr float kFxG = 9.81f;
}

// ---------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------
void VulkanEngine::spawn_explosion(glm::vec3 pos, float size, bool ground) {
    Explosion e{};
    e.pos = pos;
    e.size = size;
    e.ground = ground;
    e.seed = frand_range(spacejet_rng_state_, 0.0f, 100.0f);
    explosions_.push_back(e);
    if (explosions_.size() > 12) explosions_.erase(explosions_.begin());

    // Debris: sparks in an upward cone (ground bursts) or a sphere (air).
    const int bursts = ground ? 7 : 5;
    for (int i = 0; i < bursts; ++i) {
        float a = frand(spacejet_rng_state_) * 6.2831853f;
        float el = ground ? frand_range(spacejet_rng_state_, 0.5f, 1.3f)
                          : frand_range(spacejet_rng_state_, -1.2f, 1.2f);
        glm::vec3 d(std::cos(a) * std::cos(el), std::sin(el), std::sin(a) * std::cos(el));
        spawn_hit_particles(pos + glm::vec3(0.0f, 1.0f, 0.0f), glm::normalize(d), -d);
    }
    if (ground) spawn_impact_decal(pos + glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0, -1, 0));

    const float dist = glm::distance(pos, player_.eye_position());
    cam_shake_ = std::max(cam_shake_, size * glm::clamp(1.0f - dist / 700.0f, 0.0f, 1.0f) * 1.6f);
    if (dist < 400.0f) {
        kill_flash_t_ = std::max(kill_flash_t_, kKillFlashDuration *
                                 glm::clamp(1.0f - dist / 400.0f, 0.1f, 0.8f));
    }
    if (audio_) {
        float vol = 2.2f * size;
        audio_->play_at(ClipID::Impact, pos, vol, -0.45f, 0.10f);
        audio_->play_at(ClipID::Impact, pos, vol * 0.8f, -0.30f, 0.10f);
        audio_->play_at(ClipID::Impact, pos, vol * 0.6f, 0.05f, 0.10f);
    }
    log::infof("[fx] explosion at (%.0f, %.0f, %.0f) size %.2f%s",
               pos.x, pos.y, pos.z, size, ground ? " ground" : " air");
}

// ---------------------------------------------------------------------
// Flags: position-based cloth (Verlet + distance constraints, Jakobsen
// 2001 / Mueller PBD) pinned along the hoist edge to a pole. Forces:
// gravity, and per-triangle aerodynamic drag/lift from the RELATIVE wind
// (F ~ area * (n . v_rel) * n), which is what makes real flags ripple and
// stream. Wind = slowly veering base direction + gusts.
// ---------------------------------------------------------------------
void VulkanEngine::init_flags() {
    flags_.clear();
    // Castle-local tower tops (level.cpp) + the 22 m plateau lift. Not
    // every tower gets a flag: NE + SW corner towers, west gatehouse
    // tower and the keep.
    const float lift = 22.0f;
    struct Spot { glm::vec3 apex; int design; };
    const Spot spots[] = {
        { { 11.0f, 19.2f + lift,  11.0f}, 0 },     // NE corner tower
        { {-11.0f, 19.2f + lift, -11.0f}, 1 },     // SW corner tower
        { {-4.4f,  16.4f + lift,  11.9f}, 1 },     // west gatehouse tower
        { { 0.0f,  15.6f + lift,  -4.0f}, 0 },     // keep
    };
    for (const auto& s : spots) {
        Flag f{};
        f.pole_base = s.apex - glm::vec3(0.0f, 0.4f, 0.0f);
        f.design = s.design;
        f.p.resize(size_t(f.nx) * f.ny);
        glm::vec3 top = f.pole_base + glm::vec3(0.0f, f.pole_h - 0.05f, 0.0f);
        for (int j = 0; j < f.ny; ++j)
            for (int i = 0; i < f.nx; ++i)
                f.p[size_t(j) * f.nx + i] =
                    top + glm::vec3(f.w * float(i) / float(f.nx - 1), -f.h * float(j) / float(f.ny - 1), 0.0f);
        f.pp = f.p;
        flags_.push_back(std::move(f));
    }
}

void VulkanEngine::tick_flags(float dt) {
    if (flags_.empty()) return;
    // Wind: base direction veers slowly; speed gusts on two time scales.
    wind_t_ += dt;
    float ang = 0.6f + 0.5f * std::sin(wind_t_ * 0.031f) + 0.2f * std::sin(wind_t_ * 0.11f);
    float spd = 7.5f + 3.0f * std::sin(wind_t_ * 0.37f) + 2.0f * std::sin(wind_t_ * 1.13f + 1.7f)
              + 1.5f * std::sin(wind_t_ * 2.9f);
    wind_vel_ = glm::vec3(std::cos(ang), 0.0f, std::sin(ang)) * std::max(1.0f, spd);

    const float h = 1.0f / 120.0f;
    const glm::vec3 g(0.0f, -9.81f, 0.0f);
    for (auto& f : flags_) {
        f.acc = std::min(f.acc + dt, 0.1f);
        const int nx = f.nx, ny = f.ny;
        const float rx = f.w / float(nx - 1), ry = f.h / float(ny - 1);
        const float rd = std::sqrt(rx * rx + ry * ry);
        auto idx = [nx](int i, int j) { return size_t(j) * nx + i; };
        glm::vec3 top = f.pole_base + glm::vec3(0.0f, f.pole_h - 0.05f, 0.0f);
        std::vector<glm::vec3> force(f.p.size());
        while (f.acc >= h) {
            f.acc -= h;
            std::fill(force.begin(), force.end(), g);
            // Aerodynamic force per triangle, split onto its vertices.
            const float kAir = 0.55f;          // rho * Cd / (2 * mass/area)
            for (int j = 0; j + 1 < ny; ++j) {
                for (int i = 0; i + 1 < nx; ++i) {
                    size_t a = idx(i, j), b = idx(i + 1, j), c = idx(i, j + 1), d = idx(i + 1, j + 1);
                    const size_t tri[2][3] = { { a, b, c }, { b, d, c } };
                    for (const auto& t : tri) {
                        glm::vec3 e1 = f.p[t[1]] - f.p[t[0]], e2 = f.p[t[2]] - f.p[t[0]];
                        glm::vec3 nA = glm::cross(e1, e2);          // |nA| = 2 * area
                        float a2 = glm::length(nA);
                        if (a2 < 1e-6f) continue;
                        glm::vec3 n = nA / a2;
                        glm::vec3 v = ((f.p[t[0]] - f.pp[t[0]]) + (f.p[t[1]] - f.pp[t[1]]) +
                                       (f.p[t[2]] - f.pp[t[2]])) / (3.0f * h);
                        glm::vec3 vr = wind_vel_ - v;
                        float vn = glm::dot(vr, n);
                        glm::vec3 F = n * (vn * std::abs(vn)) * (kAir * a2 * 0.5f / (rx * ry));
                        // light tangential drag so the cloth streams
                        F += (vr - n * vn) * (0.04f * a2 * 0.5f / (rx * ry));
                        for (size_t k : t) force[k] += F * (1.0f / 3.0f);
                    }
                }
            }
            // Verlet integrate with slight damping.
            for (size_t k = 0; k < f.p.size(); ++k) {
                glm::vec3 cur = f.p[k];
                f.p[k] += (cur - f.pp[k]) * 0.995f + force[k] * (h * h);
                f.pp[k] = cur;
            }
            // Constraints: structural + shear + bending (skip-one), pins.
            auto satisfy = [&](size_t a, size_t b, float rest, float stiff) {
                glm::vec3 dlt = f.p[b] - f.p[a];
                float l = glm::length(dlt);
                if (l < 1e-6f) return;
                glm::vec3 corr = dlt * ((l - rest) / l * 0.5f * stiff);
                f.p[a] += corr;
                f.p[b] -= corr;
            };
            for (int it = 0; it < 6; ++it) {
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        if (i + 1 < nx) satisfy(idx(i, j), idx(i + 1, j), rx, 1.0f);
                        if (j + 1 < ny) satisfy(idx(i, j), idx(i, j + 1), ry, 1.0f);
                        if (i + 1 < nx && j + 1 < ny) {
                            satisfy(idx(i, j), idx(i + 1, j + 1), rd, 0.5f);
                            satisfy(idx(i + 1, j), idx(i, j + 1), rd, 0.5f);
                        }
                        if (i + 2 < nx) satisfy(idx(i, j), idx(i + 2, j), 2.0f * rx, 0.15f);
                        if (j + 2 < ny) satisfy(idx(i, j), idx(i, j + 2), 2.0f * ry, 0.15f);
                    }
                // Hoist edge clamped to the pole (rings can't move).
                for (int j = 0; j < ny; ++j) {
                    glm::vec3 pin = top + glm::vec3(0.0f, -f.h * float(j) / float(ny - 1), 0.0f);
                    f.p[idx(0, j)] = pin;
                    f.pp[idx(0, j)] = pin;
                }
            }
        }
    }
}

void VulkanEngine::tick_fx(float dt) {
    if (dt <= 0.0f) return;
    dt = std::min(dt, 0.05f);
    tick_flags(dt);
    for (auto& e : explosions_) {
        e.t += dt;
        // Smoke drift is integrated (not wind * age): gusts then only
        // nudge the cloud instead of teleporting every puff at once.
        if (e.t > 0.6f) e.drift += wind_vel_ * 0.35f * dt;
    }
    // Delayed secondary detonations (crash fuel / ordnance cook-off).
    for (size_t i = 0; i < pending_booms_.size();) {
        pending_booms_[i].delay -= dt;
        if (pending_booms_[i].delay <= 0.0f) {
            const PendingBoom b = pending_booms_[i];
            pending_booms_.erase(pending_booms_.begin() + i);
            spawn_explosion(b.pos, b.size, b.ground);
        } else ++i;
    }
    explosions_.erase(std::remove_if(explosions_.begin(), explosions_.end(),
                                     [](const Explosion& e) { return e.t > 20.0f; }),
                      explosions_.end());
    cam_shake_ *= std::exp(-dt * 3.2f);

    // Bombs: ballistic with light quadratic drag; detonate on terrain,
    // water or anything the physics raycast hits (castle, crates).
    for (auto it = bombs_.begin(); it != bombs_.end();) {
        Bomb& b = *it;
        b.prev_pos = b.pos;
        b.t += dt;
        float v = glm::length(b.vel);
        glm::vec3 drag = (v > 1e-3f) ? -b.vel * (v * 0.00025f) : glm::vec3(0.0f);
        b.vel += (glm::vec3(0.0f, -kFxG, 0.0f) + drag) * dt;
        b.pos += b.vel * dt;
        bool boom = false;
        glm::vec3 at = b.pos;
        glm::vec3 seg = b.pos - b.prev_pos;
        float sl = glm::length(seg);
        if (physics_ && sl > 1e-4f) {
            auto rh = physics_->raycast(b.prev_pos, seg / sl, sl);
            if (rh.hit) { boom = true; at = rh.position; }
        }
        float gh = std::max(sample_terrain_height(b.pos.x, b.pos.z), rt_.water_level);
        if (!boom && b.pos.y <= gh) { boom = true; at = glm::vec3(b.pos.x, gh, b.pos.z); }
        if (!boom && b.t > 30.0f) { boom = true; }
        if (boom) {
            spawn_explosion(at, 1.0f, true);
            it = bombs_.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------
// GPU resources
// ---------------------------------------------------------------------
void VulkanEngine::init_fx() {
    const std::string sd = QLIKE_SHADER_DIR;
    fx_vert_module_        = vkpipe::load_shader_module(device_, sd + "/fx.vert.spv");
    fx_ribbon_frag_module_ = vkpipe::load_shader_module(device_, sd + "/fx_ribbon.frag.spv");
    fx_volume_frag_module_ = vkpipe::load_shader_module(device_, sd + "/fx_volume.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = fx_vert_module_;
    cfg.frag = fx_ribbon_frag_module_;
    cfg.layout = pipeline_layout_;
    cfg.color_format = scene_color_format_;
    cfg.color_attachment_count = 1;
    cfg.depth_format = depth_format_;
    VkVertexInputBindingDescription vb{};
    vb.binding = 0; vb.stride = sizeof(Vertex); vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    cfg.vbindings = { vb };
    VkVertexInputAttributeDescription a0{}, a1{}, a2{};
    a0.location = 0; a0.format = VK_FORMAT_R32G32B32_SFLOAT; a0.offset = offsetof(Vertex, position);
    a1.location = 1; a1.format = VK_FORMAT_R32G32B32_SFLOAT; a1.offset = offsetof(Vertex, normal);
    a2.location = 2; a2.format = VK_FORMAT_R32G32_SFLOAT;    a2.offset = offsetof(Vertex, uv);
    cfg.vattrs = { a0, a1, a2 };
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = false;
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.premultiplied_blend = true;
    fx_ribbon_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

    // Volumes: camera outside the box -> front faces, depth-tested (the
    // opaque scene occludes the box front); inside -> back faces, no test.
    cfg.frag = fx_volume_frag_module_;
    cfg.cull = VK_CULL_MODE_BACK_BIT;
    fx_volume_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    cfg.cull = VK_CULL_MODE_FRONT_BIT;
    cfg.depth_test = false;
    fx_volume_inside_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);

    for (int i = 0; i < 2; ++i) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(Vertex) * kFxMaxVerts;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &fx_vbuf_[i], &fx_valloc_[i], &ai),
                 "fx vertex buffer");
        fx_vmapped_[i] = ai.pMappedData;
    }
    log::info("[fx] pipelines + ribbon buffers ready");
}

void VulkanEngine::destroy_fx() {
    for (VkPipeline* p : { &fx_ribbon_pipeline_, &fx_volume_pipeline_, &fx_volume_inside_pipeline_ }) {
        if (*p) vkDestroyPipeline(device_, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
    for (VkShaderModule* m : { &fx_vert_module_, &fx_ribbon_frag_module_, &fx_volume_frag_module_ }) {
        if (*m) vkDestroyShaderModule(device_, *m, nullptr);
        *m = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; ++i) {
        if (fx_vbuf_[i]) vmaDestroyBuffer(allocator_, fx_vbuf_[i], fx_valloc_[i]);
        fx_vbuf_[i] = VK_NULL_HANDLE;
        fx_vmapped_[i] = nullptr;
    }
}

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------
namespace {
struct RibbonBuilder {
    std::vector<Vertex> v;
    glm::vec3 eye{0.0f};
    // Camera-facing strip through `p` with per-point half-width, alpha
    // and tint. uv = (alpha, -1..+1 across).
    void strip(const std::vector<glm::vec3>& p, const std::vector<float>& w,
               const std::vector<float>& a, const glm::vec3& tint) {
        const size_t n = p.size();
        if (n < 2) return;
        std::vector<glm::vec3> side(n);
        for (size_t i = 0; i < n; ++i) {
            glm::vec3 t = (i + 1 < n) ? p[i + 1] - p[i] : p[i] - p[i - 1];
            if (i > 0 && i + 1 < n) t = p[i + 1] - p[i - 1];
            glm::vec3 s = glm::cross(t, eye - p[i]);
            float l = glm::length(s);
            side[i] = l > 1e-6f ? s / l * w[i] : glm::vec3(0.0f, w[i], 0.0f);
        }
        for (size_t i = 0; i + 1 < n; ++i) {
            Vertex q0{p[i] - side[i],         tint, {a[i], -1.0f}};
            Vertex q1{p[i] + side[i],         tint, {a[i],  1.0f}};
            Vertex q2{p[i + 1] + side[i + 1], tint, {a[i + 1],  1.0f}};
            Vertex q3{p[i + 1] - side[i + 1], tint, {a[i + 1], -1.0f}};
            v.insert(v.end(), { q0, q1, q2, q0, q2, q3 });
        }
    }
    void segment(glm::vec3 a, glm::vec3 b, float w, float alpha, glm::vec3 tint) {
        strip({ a, b }, { w, w }, { alpha, alpha }, tint);
    }
    void billboard(glm::vec3 c, float r, glm::vec3 right, glm::vec3 up, glm::vec3 tint) {
        glm::vec3 R = right * r, U = up * r;
        Vertex q0{c - R - U, tint, {-1.0f, -1.0f}};
        Vertex q1{c + R - U, tint, { 1.0f, -1.0f}};
        Vertex q2{c + R + U, tint, { 1.0f,  1.0f}};
        Vertex q3{c - R + U, tint, {-1.0f,  1.0f}};
        v.insert(v.end(), { q0, q1, q2, q0, q2, q3 });
    }
};
}  // namespace

void VulkanEngine::render_fx(VkCommandBuffer cmd) {
    if (!fx_ribbon_pipeline_) return;
    const FrameView& fv = current_frame_view_;
    const glm::vec3 eye = fv.eye_pos;
    const glm::vec3 cam_fwd = -glm::vec3(fv.inv_view[2]);
    const glm::vec3 cam_right = glm::vec3(fv.inv_view[0]);
    const glm::vec3 cam_up = glm::vec3(fv.inv_view[1]);
    const float now = static_cast<float>(glfwGetTime());
    const float px_ang = 1.678f / std::max(1.0f, static_cast<float>(render_extent_.height));
    auto min_w = [&](glm::vec3 p, float w, float px) {
        return std::max(w, glm::distance(p, eye) * px_ang * px);
    };
    (void)cam_fwd;

    // ---- Build ribbons, grouped by kind ----
    RibbonBuilder smoke, glow, flash;
    smoke.eye = glow.eye = flash.eye = eye;

    // Contrails + afterburner glow.
    for (const auto& f : spacejet_flights_) {
        if (f.trail.size() >= 2) {
            // Contrail (white vapour) up to the moment of the hit, then a
            // thick, dark, oily smoke column from the burning wreck. The two
            // strips share the split point so the trail stays continuous.
            std::vector<glm::vec3> p, sp; std::vector<float> w, a, sw, sa;
            p.reserve(f.trail.size() + 1);
            for (const auto& tp : f.trail) {
                float age = f.t - tp.t;
                if (tp.smoke > 0.5f) {
                    if (sp.empty() && !p.empty()) {   // bridge from the contrail
                        sp.push_back(p.back()); sw.push_back(w.back()); sa.push_back(a.back());
                    }
                    // Real smoke plumes don't fade uniformly: turbulence tears
                    // the column into clumps with clear gaps, and each clump
                    // wanders off on its own (divergence) while it spreads and
                    // thins. Everything keys off the sample's birth time, so a
                    // clump keeps its identity frame to frame (no flicker).
                    auto vn1 = [](float x) {   // 1D value noise, 0..1
                        auto h = [](float n) { return glm::fract(std::sin(n * 127.1f) * 43758.5453f); };
                        float i = std::floor(x), fr = x - i;
                        fr = fr * fr * (3.0f - 2.0f * fr);
                        return glm::mix(h(i), h(i + 1.0f), fr);
                    };
                    const float key = tp.t * 2.6f + float(f.id) * 17.3f;
                    float clump = vn1(key);
                    float clump2 = vn1(key * 2.3f + 5.1f);
                    // Gaps open up as the smoke ages (young smoke is solid).
                    float gap_thr = glm::mix(0.0f, 0.55f, glm::clamp(age / 1.2f, 0.0f, 1.0f));
                    float gap = glm::smoothstep(gap_thr, gap_thr + 0.18f, clump * 0.7f + clump2 * 0.3f);
                    // Divergence: each clump drifts sideways / up in its own
                    // direction, growing with age.
                    float ang = vn1(key * 0.7f + 31.0f) * 12.566f;
                    glm::vec3 div(std::cos(ang), 0.35f + 0.65f * vn1(key + 77.0f), std::sin(ang));
                    glm::vec3 pos = tp.p + glm::vec3(0.0f, age * 1.2f, 0.0f) + wind_vel_ * (0.3f * age) +
                                    div * (age * age * 1.6f * f.scale);
                    if (sp.empty() && !p.empty()) pos = p.back();
                    sp.push_back(pos);
                    float wd = (1.1f + age * 2.6f) * f.scale * (0.75f + 0.5f * clump2);
                    sw.push_back(min_w(tp.p, wd, 1.2f));
                    // ~2.5 s life; alpha also thins as the plume spreads.
                    float life = glm::clamp(1.0f - age / 2.6f, 0.0f, 1.0f);
                    sa.push_back(life * life * gap * std::min(1.0f, age * 4.0f + 0.3f) * 0.9f);
                    continue;
                }
                p.push_back(tp.p + glm::vec3(0.0f, age * 0.6f, 0.0f));   // slow rise
                float wd = (0.45f + age * 0.75f + std::min(tp.g, 8.0f) * 0.05f) * f.scale;
                w.push_back(min_w(tp.p, wd, 0.8f));
                float life = 1.0f - age / 6.0f;
                a.push_back(std::max(0.0f, life * life) * 0.55f);
            }
            if (!f.dead && !f.destroying) {
                glm::vec3 ex = f.pos - f.fwd * (3.0f * f.scale * 0.9f);
                p.push_back(ex); w.push_back(min_w(ex, 0.35f * f.scale, 0.8f)); a.push_back(0.0f);
            }
            if (!f.dead && f.destroying) {
                sp.push_back(f.pos); sw.push_back(min_w(f.pos, 1.0f * f.scale, 1.2f)); sa.push_back(0.6f);
            }
            smoke.strip(p, w, a, glm::vec3(0.92f, 0.93f, 0.95f));
            smoke.strip(sp, sw, sa, glm::vec3(0.07f, 0.065f, 0.06f));
        }
        if (!f.dead) {
            glm::vec3 ex = f.pos - f.fwd * (3.0f * f.scale * 0.95f);
            float flick = 0.85f + 0.15f * std::sin(now * 53.0f + float(f.id));
            glow.segment(ex, ex - f.fwd * (5.5f * f.scale * flick),
                         min_w(ex, 0.55f * f.scale, 1.2f), 1.6f * flick,
                         glm::vec3(1.0f, 0.45f, 0.15f));
            if (f.destroying) {   // burning wreck
                glow.segment(f.pos, f.pos - f.fwd * 9.0f, min_w(f.pos, 2.0f, 2.0f), 2.5f,
                             glm::vec3(1.0f, 0.35f, 0.05f));
            }
        }
    }
    // Falling bombs: short smoke wisp.
    for (const auto& b : bombs_) {
        glm::vec3 d = glm::normalize(b.vel + glm::vec3(1e-4f));
        smoke.segment(b.pos - d * 8.0f, b.pos, min_w(b.pos, 0.35f, 0.8f), 0.35f,
                      glm::vec3(0.8f));
    }
    // Laser bolts: from the barrel (converging onto the flight line) to
    // the bolt's nose. Thin white-hot core + magenta halo, additive.
    if (physics_) {
        for (const auto& pr : projectiles_) {
            glm::mat4 world;
            if (!physics_->get_body_world_matrix_h(pr.jolt_handle, world)) continue;
            glm::vec3 pos(world[3]);
            glm::vec3 vel = physics_->get_linear_velocity_h(pr.jolt_handle);
            float vl = glm::length(vel);
            glm::vec3 dir = vl > 1e-3f ? vel / vl : pr.initial_dir;
            float flown = glm::distance(pos, pr.spawn_origin);
            float conv  = std::max(0.0f, 1.0f - flown / 12.0f);
            glm::vec3 head = pos + dir * pr.half_length + pr.trail_offset * conv;
            glm::vec3 barrel = pr.spawn_origin + pr.trail_offset;
            float len = std::min(3.0f, glm::dot(head - barrel, dir));
            if (len < 0.02f) continue;
            glm::vec3 tail = head - dir * len;
            // Fade in with distance from the eye: the first metres of the
            // bolt sit right in front of the camera, where an additive glow
            // covers a big screen area and bloomed the gun to white.
            auto near_k = [&](glm::vec3 p) {
                return glm::smoothstep(1.5f, 7.0f, glm::distance(p, eye));
            };
            glow.strip({ tail, head }, { min_w(tail, 0.03f, 1.4f), min_w(head, 0.04f, 1.4f) },
                       { 0.0f, near_k(head) }, glm::vec3(1.0f, 0.25f, 0.75f));
        }
    }
    // Anti-aircraft tracers: red glowing streaks (~15 m) + muzzle flashes.
    for (const auto& r : aa_rounds_) {
        glm::vec3 d = glm::normalize(r.vel);
        float len = std::min(16.0f, glm::length(r.vel) * 0.022f);
        glm::vec3 tail = r.pos - d * len;
        glow.strip({ tail, r.pos }, { min_w(tail, 0.10f, 1.3f), min_w(r.pos, 0.16f, 1.8f) },
                   { 0.15f, 1.4f }, glm::vec3(1.0f, 0.06f, 0.03f));
    }
    for (const auto& v : aa_) {
        if (v.flash_t <= 0.0f) continue;
        glm::mat4 m[4];
        aa_part_models(v, false, m);
        glm::vec3 mz = glm::vec3(m[2] * glm::vec4(aa_muzzle_[v.flash_barrel], 1.0f));
        flash.billboard(mz, 1.1f, cam_right, cam_up, glm::vec3(1.0f, 0.55f, 0.25f) * 3.0f);
    }
    for (const auto& t : tracer_segs_) {
        float fade = 1.0f - t.age / kTracerFade;
        if (fade <= 0.0f) continue;
        float tk = glm::smoothstep(1.5f, 7.0f, glm::distance(0.5f * (t.a + t.b), eye));
        glow.segment(t.a, t.b, min_w(t.b, 0.03f * fade + 0.01f, 1.0f), 0.8f * fade * fade * tk,
                     glm::vec3(1.0f, 0.25f, 0.75f));
    }
    // Flags: cloth triangles (kind 3, lit two-sided in fx_ribbon.frag).
    // Normal slot carries the design index, uv = cloth uv.
    RibbonBuilder cloth; cloth.eye = eye;
    for (const auto& f : flags_) {
        for (int j = 0; j + 1 < f.ny; ++j)
            for (int i = 0; i + 1 < f.nx; ++i) {
                auto V = [&](int a, int b) {
                    return Vertex{ f.p[size_t(b) * f.nx + a], glm::vec3(float(f.design), 0.0f, 0.0f),
                                   { float(a) / float(f.nx - 1), float(b) / float(f.ny - 1) } };
                };
                Vertex q0 = V(i, j), q1 = V(i + 1, j), q2 = V(i + 1, j + 1), q3 = V(i, j + 1);
                cloth.v.insert(cloth.v.end(), { q0, q1, q2, q0, q2, q3 });
            }
    }
    // Explosion flashes.
    for (const auto& e : explosions_) {
        if (e.t < 0.35f) {
            float k = 1.0f - e.t / 0.35f;
            flash.billboard(e.pos + glm::vec3(0.0f, 6.0f * e.size, 0.0f),
                            e.size * 24.0f * (0.6f + e.t * 2.0f), cam_right, cam_up,
                            glm::vec3(1.0f, 0.75f, 0.45f) * (k * k * 12.0f));
        }
    }

    // ---- Volumes (fireball / smoke / dust), sorted far to near ----
    struct Vol { glm::vec3 c; float r, squash, age, kind, seed, dens, heat, fade, scroll, dist; };
    std::vector<Vol> vols;
    for (const auto& e : explosions_) {
        const float s = e.size, t = e.t;
        if (t < 3.4f) {   // fireball
            Vol v{};
            v.r = s * (2.5f + 11.5f * (1.0f - std::exp(-t * 4.2f)));
            v.c = e.pos + glm::vec3(0.0f, e.ground ? v.r * 0.55f + t * 5.0f * s : t * 2.0f, 0.0f);
            v.squash = 0.92f; v.age = t; v.kind = 0.0f; v.seed = e.seed;
            v.dens = 1.7f;
            v.heat = std::exp(-t * 1.1f) * std::min(1.0f, t / 0.04f + 0.3f);
            v.fade = 1.0f - glm::smoothstep(2.3f, 3.4f, t);
            v.scroll = t * 1.6f;
            vols.push_back(v);
        }
        // Smoke: a cluster of billowing puffs instead of one giant column.
        // Each puff is shot out of the fireball with its own velocity,
        // slowed by drag, lifted by buoyancy that decays as it cools,
        // carried by the wind, grows ~sqrt(age) and thins / frays out.
        // Positions are closed-form in age (no per-frame state).
        if (t > 0.15f && t < 16.0f) {
            const int kPuffs = e.ground ? 7 : 5;
            for (int k = 0; k < kPuffs; ++k) {
                float hk  = std::fmod(e.seed * 7.31f + float(k) * 3.77f, 6.2831853f);
                float hk2 = std::fmod(e.seed * 3.17f + float(k) * 1.93f, 1.0f);
                float start = 0.15f + 0.12f * float(k);              // staggered release
                float tt = t - start;
                if (tt <= 0.0f) continue;
                // Initial outward + upward velocity (m/s), drag 1/s.
                float out_v = (k == 0 ? 2.0f : 9.0f + 6.0f * hk2) * s;
                float up_v  = (k == 0 ? 16.0f : 7.0f + 8.0f * hk2) * s;
                float drag  = 1.2f;
                float e1 = (1.0f - std::exp(-drag * tt)) / drag;     // integrated decay
                glm::vec3 dir(std::cos(hk), 0.0f, std::sin(hk));
                glm::vec3 p = e.pos + dir * (out_v * e1);
                // Buoyant rise: initial up velocity + slow, decaying lift.
                p.y += up_v * e1 + 2.2f * s * std::min(tt, 8.0f) * (1.0f - std::exp(-tt * 0.4f));
                if (e.ground) p.y += 3.0f * s;
                // Wind drift ramps in as the puff loses its own momentum.
                p += e.drift * glm::smoothstep(0.0f, 1.2f, tt);
                Vol v{};
                v.r = s * (2.2f + 1.8f * std::sqrt(tt)) * (k == 0 ? 1.1f : 0.8f + 0.3f * hk2);
                v.r = std::min(v.r, 8.5f * s);
                v.c = p;
                v.squash = 0.85f + 0.2f * hk2;
                v.age = tt; v.kind = 1.0f; v.seed = e.seed + 31.0f + float(k) * 17.0f;
                v.dens = 1.0f * (1.0f - glm::smoothstep(1.5f, 12.0f, tt) * 0.7f);
                v.heat = 0.55f * std::exp(-tt * 1.6f);               // embers glow early on
                v.fade = glm::smoothstep(0.0f, 0.5f, tt) * (1.0f - glm::smoothstep(8.0f, 15.5f - start, tt));
                v.scroll = tt * 0.45f;                                // shader: slow boiling time
                vols.push_back(v);
            }
        }
        if (e.ground && t < 6.0f) {    // ground dust ring
            Vol v{};
            v.r = s * (4.0f + 22.0f * (1.0f - std::exp(-t * 2.0f)));
            v.c = e.pos + glm::vec3(0.0f, 0.6f * s, 0.0f);           // hugs the ground
            v.squash = 0.12f; v.age = t; v.kind = 2.0f; v.seed = e.seed + 77.0f;
            v.dens = 0.6f; v.heat = 0.0f;
            v.fade = std::min(1.0f, t / 0.15f) * (1.0f - glm::smoothstep(2.0f, 6.0f, t));
            v.scroll = t * 0.6f;
            vols.push_back(v);
        }
    }
    for (auto& v : vols) v.dist = glm::distance(v.c, eye);
    std::sort(vols.begin(), vols.end(), [](const Vol& a, const Vol& b) { return a.dist > b.dist; });

    const size_t n_smoke = smoke.v.size(), n_glow = glow.v.size(), n_flash = flash.v.size();
    if (vols.empty() && n_smoke + n_glow + n_flash + cloth.v.size() == 0) return;

    // Upload ribbon vertices.
    const uint32_t slot = static_cast<uint32_t>(frame_number_ & 1);
    Vertex* dst = static_cast<Vertex*>(fx_vmapped_[slot]);
    size_t total = 0;
    auto put = [&](const std::vector<Vertex>& src) -> std::pair<uint32_t, uint32_t> {
        size_t n = std::min(src.size(), size_t(kFxMaxVerts) - total);
        if (n) std::memcpy(dst + total, src.data(), n * sizeof(Vertex));
        auto r = std::make_pair(static_cast<uint32_t>(total), static_cast<uint32_t>(n));
        total += n;
        return r;
    };
    auto r_smoke = put(smoke.v);
    auto r_glow  = put(glow.v);
    auto r_flash = put(flash.v);
    auto r_cloth = put(cloth.v);
    vmaFlushAllocation(allocator_, fx_valloc_[slot], 0, total * sizeof(Vertex));

    // ---- Pass ----
    vkinit::transition_image(cmd, scene_color_image_,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkinit::transition_image_aspect(cmd, depth_image_,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
    VkRenderingAttachmentInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ca.imageView = scene_color_view_;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo da{};
    da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    da.imageView = depth_view_;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    da.storeOp = VK_ATTACHMENT_STORE_OP_NONE;
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { {0, 0}, render_extent_ };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vpt{ 0.0f, 0.0f, float(render_extent_.width), float(render_extent_.height), 0.0f, 1.0f };
    VkRect2D sc{ {0, 0}, render_extent_ };
    vkCmdSetViewport(cmd, 0, 1, &vpt);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    // Opaque cloth FIRST: drawn after the volumes it overwrote smoke that
    // is in front of a flag (no depth write in this pass).
    if (total > 0 && r_cloth.second > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fx_ribbon_pipeline_);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &fx_vbuf_[slot], &off);
        PushConstants pc{};
        pc.mvp = fv.vp;
        pc.model = glm::mat4(1.0f);
        pc.color = glm::vec4(3.0f, 1.0f, now, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0,
                           sizeof(PushConstants), &pc);
        vkCmdDraw(cmd, r_cloth.second, 1, r_cloth.first, 0);
    }
    // Volumes.
    if (!vols.empty()) {
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        VkPipeline bound = VK_NULL_HANDLE;
        for (const auto& v : vols) {
            glm::vec3 half(v.r, v.r * v.squash, v.r);
            glm::vec3 rel = glm::abs(eye - v.c);
            bool inside = rel.x < half.x + 0.3f && rel.y < half.y + 0.3f && rel.z < half.z + 0.3f;
            VkPipeline want = inside ? fx_volume_inside_pipeline_ : fx_volume_pipeline_;
            if (want != bound) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, want); bound = want; }
            glm::mat4 model = glm::translate(glm::mat4(1.0f), v.c) *
                              glm::scale(glm::mat4(1.0f), half * 2.0f);
            PushConstants pc{};
            pc.mvp = fv.vp * model;
            pc.model = model;
            pc.prev_mvp = fv.inv_vp;
            pc.color = glm::vec4(v.c, v.r);
            pc.emissive = glm::vec4(v.age, v.kind, v.seed, v.dens);
            pc.tex_params = glm::vec4(v.squash, v.heat, v.fade, v.scroll);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0,
                               sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
        }
    }
    // Ribbons.
    if (total > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fx_ribbon_pipeline_);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &fx_vbuf_[slot], &off);
        auto draw = [&](std::pair<uint32_t, uint32_t> r, float kind, float inten) {
            if (r.second == 0) return;
            PushConstants pc{};
            pc.mvp = fv.vp;
            pc.model = glm::mat4(1.0f);
            pc.color = glm::vec4(kind, inten, now, 0.0f);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0,
                               sizeof(PushConstants), &pc);
            vkCmdDraw(cmd, r.second, 1, r.first, 0);
        };
        draw(r_smoke, 0.0f, 1.0f);
        draw(r_glow,  1.0f, 1.0f);
        draw(r_flash, 2.0f, 1.0f);
    }
    vkCmdEndRendering(cmd);
    vkinit::transition_image_aspect(cmd, depth_image_,
                                    VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
    vkinit::transition_image(cmd, scene_color_image_,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace qlike
