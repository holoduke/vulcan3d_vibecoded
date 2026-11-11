// Terrain CPU/GPU resources: heightmap sculpting (FBM noise/erosion brush),
// grass-eligibility mask, height textures, quadtree CLOD instancing, the
// async sun-shadow bake worker, and the curvature/sky-vis aux + grass-mask
// + fog-wisp procedural textures. Split out of world.cpp (maintainability).

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

// Classic GLSL-style hash + value-noise + 3-octave FBM, used by every
// CPU sculpt / plateau-noise / erosion path. Was previously inlined as
// nested lambdas in three sites with three different signatures
// (cleanness review #8). Centralising it removes the duplicate
// `12.9898/78.233/43758.5453` magic triple and lets the compiler pick
// one inlining strategy.
static float fbm_hash(float x, float y) {
    float n = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return n - std::floor(n);
}
static float fbm_noise2(float x, float y) {
    float ix = std::floor(x), iy = std::floor(y);
    float fx = x - ix,        fy = y - iy;
    float a = fbm_hash(ix,        iy);
    float b = fbm_hash(ix + 1.0f, iy);
    float c = fbm_hash(ix,        iy + 1.0f);
    float d = fbm_hash(ix + 1.0f, iy + 1.0f);
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    return (a * (1 - ux) + b * ux) * (1 - uy) +
           (c * (1 - ux) + d * ux) * uy;
}

// Hash + value-noise with analytical derivatives, matching the
// `noised()` GLSL helper in terrain_raymarch.frag. The brush's
// "FBM erosion" mode uses this so a stroke's high-frequency detail
// matches the procedural raymarched terrain in character.
namespace {
inline float fract_f(float v) { return v - std::floor(v); }
inline float terrain_hash21(float px, float pz) {
    // Mirrors GLSL hash21(vec2 p) — dot/fract over the (x,y,x) triple.
    float p3x = fract_f(px * 0.1031f);
    float p3y = fract_f(pz * 0.1031f);
    float p3z = fract_f(px * 0.1031f);
    float dot_p = p3x * (p3y + 19.19f) +
                  p3y * (p3z + 19.19f) +
                  p3z * (p3x + 19.19f);
    p3x += dot_p; p3y += dot_p; p3z += dot_p;
    return fract_f((p3x + p3y) * p3z);
}
struct NoiseDeriv { float v, dx, dz; };
inline NoiseDeriv terrain_noised(float px, float pz) {
    float ix = std::floor(px), iz = std::floor(pz);
    float fx = px - ix,        fz = pz - iz;
    float ux  = fx * fx * (3.0f - 2.0f * fx);
    float uz  = fz * fz * (3.0f - 2.0f * fz);
    float dux = 6.0f * fx * (1.0f - fx);
    float duz = 6.0f * fz * (1.0f - fz);
    float a = terrain_hash21(ix,        iz);
    float b = terrain_hash21(ix + 1.0f, iz);
    float c = terrain_hash21(ix,        iz + 1.0f);
    float d = terrain_hash21(ix + 1.0f, iz + 1.0f);
    float value = a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
    float dvx = dux * ((b - a) + (a - b - c + d) * uz);
    float dvz = duz * ((c - a) + (a - b - c + d) * ux);
    return { value, dvx, dvz };
}
// Same FBM as terrain_raymarch.frag::terrainM — `noised` accumulates
// the analytical derivative `d`, and `n.x / (1 + dot(d,d))` damps
// high-frequency octaves on steep slopes, producing the ridge/valley
// structure the user sees in the raymarched terrain.
// Per-octave rotation matrix matches the shader's `m2`.
inline float terrain_fbm_eroded(float wx, float wz, float scale,
                                  int octaves) {
    constexpr float kM00 =  0.8f, kM01 = -0.6f;
    constexpr float kM10 =  0.6f, kM11 =  0.8f;
    float px = wx * scale, pz = wz * scale;
    float a = 0.0f, b = 1.0f;
    float dx_acc = 0.0f, dz_acc = 0.0f;
    octaves = std::clamp(octaves, 1, 12);
    for (int i = 0; i < octaves; ++i) {
        NoiseDeriv n = terrain_noised(px, pz);
        dx_acc += n.dx;
        dz_acc += n.dz;
        a += b * n.v / (1.0f + dx_acc * dx_acc + dz_acc * dz_acc);
        b *= 0.5f;
        float npx = kM00 * px + kM01 * pz;
        float npz = kM10 * px + kM11 * pz;
        px = npx * 2.0f;
        pz = npz * 2.0f;
    }
    // Re-centre roughly around 0 — fbm with `n.v` in [0,1] sums to в‰€
    // half the harmonic series. The bias is approximate and harmless;
    // the brush noise scale only needs symmetric variation around 0.
    return a - 0.5f;
}
} // namespace

// Distance-based LOD selector for terrain chunks. Used by all three
// terrain draw paths (depth pre-pass, color pass, shadow pass) so LOD
// stays consistent — if the pre-pass picked half-LOD and the color
// pass picked full, the depth values would mismatch and the color
// pass's LESS_OR_EQUAL test would discard fragments. Thresholds are
// pulled from rt_ so the user can push higher-resolution terrain
// further out via the sliders in the Graphics Settings menu.
// pick_terrain_lod moved to terrain.h so sun_shadow.cpp can share it.
// (The old pick_terrain_morph helper was dead code — removed.)

// Paint the R channel of the grass eligibility mask. Add: push toward
// 1, Remove: push toward 0. Uses the same world hit + radius the height
// brush uses, but indexes the 1024² mask over a 2048 m world (matches
// the bake in init_grass_mask_texture). Marks `grass_mask_dirty_` so
// the next safe frame re-uploads the mask to the GPU. Does NOT touch
// heights / chunks / BLAS / Jolt — pure mask edit.
static void paint_grass_mask(std::vector<uint8_t>& mask, int dim,
                              const glm::vec3& hit, float radius_m,
                              float strength_per_sec, float dt,
                              bool add) {
    if (mask.empty()) return;
    const float kWorldSide = 2048.0f;   // matches init_grass_mask_texture
    const float cells_per_metre = static_cast<float>(dim) / kWorldSide;
    // World → mask cell coords. Mask covers [-1024..+1024] m centred at
    // origin; cell (0,0) = world (-1024, -1024).
    float cx = (hit.x + kWorldSide * 0.5f) * cells_per_metre;
    float cz = (hit.z + kWorldSide * 0.5f) * cells_per_metre;
    float rcells_f = radius_m * cells_per_metre;
    int rcells = static_cast<int>(std::ceil(rcells_f)) + 1;
    int ix0 = std::max(0, static_cast<int>(std::floor(cx)) - rcells);
    int ix1 = std::min(dim - 1, static_cast<int>(std::ceil(cx))  + rcells);
    int iz0 = std::max(0, static_cast<int>(std::floor(cz)) - rcells);
    int iz1 = std::min(dim - 1, static_cast<int>(std::ceil(cz))  + rcells);
    if (ix0 > ix1 || iz0 > iz1) return;
    const float dstrength = strength_per_sec * dt;
    for (int iz = iz0; iz <= iz1; ++iz) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            float dx = static_cast<float>(ix) - cx;
            float dz = static_cast<float>(iz) - cz;
            float d  = std::sqrt(dx * dx + dz * dz);
            if (d > rcells_f) continue;
            float t = 1.0f - (d / rcells_f);
            float falloff = t * t * (3.0f - 2.0f * t);
            size_t idx = (static_cast<size_t>(iz) * static_cast<size_t>(dim) +
                          static_cast<size_t>(ix)) * 2;
            float v = static_cast<float>(mask[idx]) / 255.0f;
            float delta = dstrength * falloff;
            v += add ? delta : -delta;
            v = std::clamp(v, 0.0f, 1.0f);
            mask[idx] = static_cast<uint8_t>(v * 255.0f);
        }
    }
}

void VulkanEngine::reupload_grass_mask() {
    if (grass_mask_data_.empty() || grass_mask_.image == VK_NULL_HANDLE) return;
    const VkDeviceSize bytes = grass_mask_data_.size();
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, grass_mask_data_.data(), bytes);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(kGrassMaskDim),
                                static_cast<uint32_t>(kGrassMaskDim), 1 };
        vkCmdCopyBufferToImage(cb, stage, grass_mask_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    grass_mask_dirty_ = false;
}

bool VulkanEngine::save_grass_mask() {
    if (grass_mask_data_.empty()) return false;
    std::ofstream f("assets/level1_grass_mask.bin", std::ios::binary);
    if (!f.is_open()) {
        log::error("save_grass_mask: failed to open assets/level1_grass_mask.bin");
        return false;
    }
    int32_t dim = kGrassMaskDim;
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    f.write(reinterpret_cast<const char*>(grass_mask_data_.data()),
            grass_mask_data_.size());
    log::infof("[grass] saved mask to assets/level1_grass_mask.bin (%zu bytes)",
               grass_mask_data_.size());
    return true;
}

bool VulkanEngine::load_grass_mask() {
    std::ifstream f("assets/level1_grass_mask.bin", std::ios::binary);
    if (!f.is_open()) return false;
    int32_t fdim = 0;
    f.read(reinterpret_cast<char*>(&fdim), sizeof(fdim));
    if (fdim != kGrassMaskDim) {
        log::warnf("[grass] saved mask dim mismatch (saved %d, want %d) — ignoring",
                   fdim, kGrassMaskDim);
        return false;
    }
    const size_t want = static_cast<size_t>(kGrassMaskDim) * kGrassMaskDim * 2;
    grass_mask_data_.resize(want);
    f.read(reinterpret_cast<char*>(grass_mask_data_.data()), want);
    log::infof("[grass] loaded mask from assets/level1_grass_mask.bin (%zu bytes)",
               want);
    return true;
}

void VulkanEngine::apply_terrain_brush(float dt) {
    if (!terrain_brush_has_hit_) return;
    // Grass-paint modes only touch the eligibility mask — bail before
    // the height-array path so they work even when chunks / heights
    // happen to be empty (defensive — Phase 4 always populates both).
    if (brush_mode_is_grass(terrain_brush_mode_)) {
        paint_grass_mask(grass_mask_data_, kGrassMaskDim,
                          terrain_brush_world_pos_,
                          terrain_brush_radius_,
                          terrain_brush_strength_, dt,
                          terrain_brush_mode_ == TerrainBrushMode::GrassAdd);
        grass_mask_dirty_ = true;
        return;
    }
    if (terrain_data_.heights.empty()) return;
    const float r = terrain_brush_radius_;
    const float w = terrain_data_.cell;
    const int dim = terrain_data_.dim;
    const int W = dim + 1;

    // Convert world hit point to grid coords + brush radius in cells.
    float cx = (terrain_brush_world_pos_.x - terrain_data_.origin_x) / w;
    float cz = (terrain_brush_world_pos_.z - terrain_data_.origin_z) / w;
    int rcells = static_cast<int>(std::ceil(r / w)) + 1;
    int ix0 = std::max(0, static_cast<int>(std::floor(cx)) - rcells);
    int ix1 = std::min(W - 1, static_cast<int>(std::ceil(cx))  + rcells);
    int iz0 = std::max(0, static_cast<int>(std::floor(cz)) - rcells);
    int iz1 = std::min(W - 1, static_cast<int>(std::ceil(cz))  + rcells);
    if (ix0 > ix1 || iz0 > iz1) return;

    const float strength = terrain_brush_strength_ * dt;

    // 3-octave value-noise via cheap hash. Centred around 0 so it
    // multiplies the sculpt delta as `1 + noise·strength` — values
    // > 1 raise more, < 1 raise less. Result: sculpted patches
    // pick up natural-looking surface variation instead of flat
    // squares. Cheap (3 hashes per cell during sculpt only).
    const float n_str  = std::clamp(terrain_brush_noise_strength_, 0.0f, 1.0f);
    const float n_freq = std::max(0.02f, terrain_brush_noise_freq_);
    const bool  use_fbm_eroded = terrain_brush_use_fbm_erosion_;
    const int   fbm_octaves    = terrain_brush_fbm_octaves_;
    auto fbm_var = [n_freq, use_fbm_eroded, fbm_octaves](float wx, float wz) {
        if (use_fbm_eroded) {
            // Mesh has 1 m cells. The raymarcher's macro-mountain
            // scale (0.003 в‰€ 333 m base wavelength) means adjacent
            // cells sample nearly identical noise — the brush ends up
            // applying a smooth blob with no per-cell variation, and
            // the surface reads as faceted because the geometry has
            // no high-freq detail to break up the triangles. Higher
            // base scale (0.05) puts ~20 m at octave 0 and ≤ 0.5 m
            // at 6 octaves — every 1 m cell gets a distinct sample,
            // so brushed areas inherit cell-level ridge variation
            // similar to the procedural FBM heights.
            const float kBrushScale = 0.05f;
            return terrain_fbm_eroded(wx, wz, kBrushScale * n_freq * 4.0f,
                                       fbm_octaves);
        }
        float v = 0.5f  * fbm_noise2(wx * n_freq,         wz * n_freq) +
                  0.25f * fbm_noise2(wx * n_freq * 2.07f, wz * n_freq * 2.07f) +
                  0.125f* fbm_noise2(wx * n_freq * 4.13f, wz * n_freq * 4.13f);
        return v - 0.4375f;   // re-centre on 0
    };

    for (int iz = iz0; iz <= iz1; ++iz) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            float dx = (static_cast<float>(ix) - cx) * w;
            float dz = (static_cast<float>(iz) - cz) * w;
            float d = std::sqrt(dx * dx + dz * dz);
            if (d > r) continue;
            // Smoothstep falloff so the brush has a soft edge.
            float t = 1.0f - (d / r);
            float falloff = t * t * (3.0f - 2.0f * t);
            size_t idx = static_cast<size_t>(iz) * static_cast<size_t>(W) +
                         static_cast<size_t>(ix);
            float& h = terrain_data_.heights[idx];
            // Per-cell noise multiplier — only kicks in for additive
            // modes (raise / lower), not smooth / flatten.
            float wx = terrain_data_.origin_x + ix * terrain_data_.cell;
            float wz = terrain_data_.origin_z + iz * terrain_data_.cell;
            float n_mult = 1.0f + 2.0f * fbm_var(wx, wz) * n_str;
            switch (terrain_brush_mode_) {
                case TerrainBrushMode::Raise:
                    h += strength * falloff * n_mult;
                    break;
                case TerrainBrushMode::Lower:
                    h -= strength * falloff * n_mult;
                    break;
                case TerrainBrushMode::Smooth: {
                    // 4-tap blur toward neighbour mean.
                    int xm = std::max(0, ix - 1);
                    int xp = std::min(W - 1, ix + 1);
                    int zm = std::max(0, iz - 1);
                    int zp = std::min(W - 1, iz + 1);
                    auto at = [&](int x, int z) {
                        return terrain_data_.heights[
                            static_cast<size_t>(z) * static_cast<size_t>(W) +
                            static_cast<size_t>(x)];
                    };
                    float mean = 0.25f * (at(xm, iz) + at(xp, iz) +
                                          at(ix, zm) + at(ix, zp));
                    h += (mean - h) * std::min(1.0f, strength * 4.0f * falloff);
                    break;
                }
                case TerrainBrushMode::Flatten: {
                    float target = terrain_brush_flatten_target_;
                    h += (target - h) * std::min(1.0f, strength * 0.5f * falloff);
                    break;
                }
                case TerrainBrushMode::Erode: {
                    // Add FBM-eroded detail (same n.v/(1+|в€‡d|²) recipe
                    // as the raymarched terrain), scaled by brush
                    // falloff. The base FBM scale here is much higher
                    // than the raymarcher's 0.003 — at typical brush
                    // radii (≤ 10 m) the raymarcher's macro-mountain
                    // wavelength means the WHOLE brush samples nearly
                    // the same noise value, which feels like a smooth
                    // uniform bump (the "smoothing" symptom). Pumping
                    // the base scale to 0.05 gives ~20 m at octave 0
                    // and ~30 cm at 6 octaves — actual ridges and
                    // valleys appear inside the brush footprint.
                    const float kErodeScale = 0.05f;
                    float n = terrain_fbm_eroded(
                        wx, wz, kErodeScale * n_freq * 4.0f,
                        terrain_brush_fbm_octaves_);
                    // Erosion amplitude scales with brush radius so a
                    // larger brush makes proportionally taller ridges
                    // (a 1 m brush shouldn't punch a 4 m peak). Per-
                    // frame accumulation = noise·radius·dt · falloff;
                    // a 1-sec hold on a 6 m brush gives ~±1.5 m peak
                    // displacement — visible without obliterating the
                    // surrounding terrain.
                    float amp = terrain_brush_radius_ * 0.6f;
                    h += n * amp * dt * falloff;
                    break;
                }
                case TerrainBrushMode::ErodeSmooth: {
                    // Inverse: pull detail back toward the 4-neighbour
                    // mean. Same kernel as Smooth but applied with the
                    // erosion-strength scaling so the brush feels like
                    // the inverse of Erode rather than a duplicate of
                    // the existing Smooth mode.
                    int xm = std::max(0, ix - 1);
                    int xp = std::min(W - 1, ix + 1);
                    int zm = std::max(0, iz - 1);
                    int zp = std::min(W - 1, iz + 1);
                    auto at = [&](int x, int z) {
                        return terrain_data_.heights[
                            static_cast<size_t>(z) * static_cast<size_t>(W) +
                            static_cast<size_t>(x)];
                    };
                    float mean = 0.25f * (at(xm, iz) + at(xp, iz) +
                                          at(ix, zm) + at(ix, zp));
                    h += (mean - h) * std::min(1.0f, strength * 2.0f * falloff);
                    break;
                }
                case TerrainBrushMode::GrassAdd:
                case TerrainBrushMode::GrassRemove:
                    // Grass-paint modes branched out at the top of
                    // apply_terrain_brush — the height loop never sees them.
                    break;
            }
        }
    }
    // Castle-courtyard sink (same clamp init_world applies to the source
    // heights): a brush stroke over the castle must not raise the ground
    // through the courtyard floor slab.
    {
        const float kSinkR = 10.5f, kSinkH = 22.0f - 0.6f;
        for (int iz = iz0; iz <= iz1; ++iz) {
            const float wz = terrain_data_.origin_z + float(iz) * w;
            if (std::abs(wz) >= kSinkR) continue;
            for (int ix = ix0; ix <= ix1; ++ix) {
                const float wx = terrain_data_.origin_x + float(ix) * w;
                if (std::abs(wx) >= kSinkR) continue;
                float& hv = terrain_data_.heights[size_t(iz) * W + ix];
                hv = (std::min)(hv, kSinkH);
            }
        }
    }

    // CLOD raster picks the edit up from the heightmap TEXTURE — no mesh
    // rebake. Refresh the min/max pyramid for the touched region (culling
    // AABBs) and flag the textures + BLAS + collision for re-upload.
    update_clod_pyramid_region(terrain_clod_pyramid_,
                               terrain_data_.heights.data(),
                               terrain_data_.dim, terrain_data_.cell,
                               16.0f, ix0, ix1, iz0, iz1);
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
    terrain_height_dirty_ = true;   // re-upload delta + full height textures
}

void VulkanEngine::refresh_terrain_collision() {
    if (!terrain_jolt_dirty_) return;
    if (terrain_data_.heights.empty()) return;
    // Jolt has no incremental update for HeightFieldShape — we rebuild
    // the whole shape. Mirrors the sub-grid trim done at level load:
    // when sample_count would be odd Jolt's block-size requirement
    // forces us onto a 2048×2048 sub-grid taken from the 2049² heightmap.
    //
    // For each Jolt sample we store max(self, neighbours). Reason:
    // Jolt's HeightFieldShape is a per-cell triangulation — the
    // collision surface is a piecewise-linear plane between corners,
    // which sits BELOW the visual surface (whether the rasterized
    // mesh's smooth-shaded reading or the raymarched FBM) at sub-cell
    // peaks. Dynamic boxes settle on the linear plane and visually
    // appear "a bit inside" the terrain. Lifting the corner samples
    // to the local max raises the triangle plane so boxes sit AT or
    // slightly ABOVE the visual surface instead of below it. Player
    // ground-clamp still uses the un-lifted terrain_data_.heights
    // (separate sample_terrain_height path), so the player's feet
    // stay anchored to the visual mesh exactly as before.
    if (physics_) {
        const int W = terrain_data_.dim + 1;
        const int physics_samples = (W % 2 == 0) ? W : (W - 1);
        if (physics_samples == W) {
            physics_->add_static_heightfield(terrain_data_.heights.data(),
                                             terrain_data_.dim,
                                             glm::vec2(terrain_data_.origin_x,
                                                        terrain_data_.origin_z),
                                             terrain_data_.cell);
        } else {
            std::vector<float> jolt_heights(static_cast<size_t>(physics_samples) *
                                             static_cast<size_t>(physics_samples));
            for (int z = 0; z < physics_samples; ++z) {
                std::memcpy(jolt_heights.data() +
                                static_cast<size_t>(z) *
                                static_cast<size_t>(physics_samples),
                            terrain_data_.heights.data() +
                                static_cast<size_t>(z) * static_cast<size_t>(W),
                            static_cast<size_t>(physics_samples) * sizeof(float));
            }
            physics_->add_static_heightfield(jolt_heights.data(),
                                             physics_samples - 1,
                                             glm::vec2(terrain_data_.origin_x,
                                                        terrain_data_.origin_z),
                                             terrain_data_.cell);
        }
    }
    terrain_jolt_dirty_ = false;
}

void VulkanEngine::ensure_terrain_raster_built() {
    if (terrain_clod_patch_.vertex_buffer) return;       // already built
    if (terrain_data_.heights.empty()) return;          // no heightmap to build from
    log::info("[terrain] lazy-building rasterised mesh + CLOD patch");
    vkDeviceWaitIdle(device_);
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;
    terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                       graphics_queue_, graphics_queue_family_, hm);
    init_terrain_clod();
    // Note: the terrain BLAS is built once in init_rt and not rebuilt
    // here. Toggling raymarch off mid-session draws raster terrain
    // correctly but RT shadow/AO/GI rays from cube.frag will not see
    // it until restart. tlas_includes_terrain_blas() handles the
    // address==0 case already, so this is a quality fallback, not a
    // crash risk.
}

void VulkanEngine::ensure_grass_raster_built() {
    // Attempted once: a build that places 0 blades (empty altitude band)
    // used to repeat every frame - device wait + 800k-candidate placement
    // + a leaked blade mesh per frame.
    if (grass_raster_attempted_ || grass_.instance_count > 0) return;
    grass_raster_attempted_ = true;
    if (terrain_data_.heights.empty()) return;
    log::info("[grass] lazy-building rasterised blade placement");
    vkDeviceWaitIdle(device_);
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;
    GrassParams gp{};
    gp.height_min  = -20.0f;
    gp.height_max  = 200.0f;
    gp.half_extent = 200.0f;
    gp.keep_out_xz = glm::vec2(12.0f, 12.0f);  // full castle footprint
    gp.max_blades  = 800000;
    destroy_grass(allocator_, grass_);   // a previous empty build still owns its blade mesh
    grass_ = build_grass(device_, allocator_,
                         graphics_queue_, graphics_queue_family_,
                         hm, gp);
}

void VulkanEngine::refresh_terrain_blas() {
    if (!terrain_blas_dirty_) return;
    // BLAS rebuild is the heavy part. Defer to mouse-up so per-frame
    // cost stays low during a stroke. The merged terrain mesh's vertex
    // buffer is rebuilt from the current heightmap, then the BLAS is
    // re-built in place. RT shadows/GI will be slightly stale until
    // this fires, which reads as "lighting catches up after you stop
    // sculpting" — acceptable.
    if (terrain_mesh_.vertex_buffer && !terrain_data_.heights.empty()) {
        Heightmap hm{};
        hm.dim = terrain_data_.dim;
        hm.cell = terrain_data_.cell;
        hm.origin_x = terrain_data_.origin_x;
        hm.origin_z = terrain_data_.origin_z;
        hm.heights = terrain_data_.heights;
        // Rebuild the full-mesh vertex buffer in place by abusing the
        // chunk rebuild helper on a virtual "chunk" covering the whole
        // mesh.
        TerrainChunk fake{};
        fake.mesh = terrain_mesh_;
        fake.origin_ix = 0;
        fake.origin_iz = 0;
        fake.sample_dim = terrain_data_.dim + 1;
        // Frames in flight: the previous frame may still read the vertex
        // buffer / BLAS. Mouse-up only, so a full drain is fine.
        vkDeviceWaitIdle(device_);
        rebuild_chunk_vertices(device_, allocator_, graphics_queue_,
                               graphics_queue_family_, hm, fake);
        // Refit the terrain BLAS to the new vertices (it was never updated:
        // RT shadows / AO / GI kept tracing the pre-sculpt surface).
        VkBuffer scratch = VK_NULL_HANDLE; VmaAllocation scratch_alloc = nullptr;
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cb, &di);
            record_terrain_blas_refit(cb, &scratch, &scratch_alloc);
            mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier2(cb, &di);
        });
        if (scratch) vmaDestroyBuffer(allocator_, scratch, scratch_alloc);
        // The TLAS caches BLAS bounds at build time: force a full rebuild.
        prev_dyn_signature_ = 0;
        prev_tlas_n_ = UINT32_MAX;
    }
    terrain_blas_dirty_ = false;
}

void VulkanEngine::destroy_terrain_shadow_texture() {
    deferred_desc_sig_ = 0;   // handles may be recreated (even at the same address)
    if (terrain_shadow_view_) {
        vkDestroyImageView(device_, terrain_shadow_view_, nullptr);
        terrain_shadow_view_ = VK_NULL_HANDLE;
    }
    if (terrain_shadow_image_) {
        vmaDestroyImage(allocator_, terrain_shadow_image_, terrain_shadow_alloc_);
        terrain_shadow_image_ = VK_NULL_HANDLE;
        terrain_shadow_alloc_ = nullptr;
    }
    if (terrain_shadow_sampler_) {
        vkDestroySampler(device_, terrain_shadow_sampler_, nullptr);
        terrain_shadow_sampler_ = VK_NULL_HANDLE;
    }
    terrain_shadow_dim_ = 0;
}

namespace {
// First-time creation of the heightmap shadow image + view + sampler.
// Pulled out of rebuild_terrain_shadow_texture so the progressive tile
// path can rely on the resources existing without duplicating setup.
// Caller writes binding 6 separately; transitions the image into
// SHADER_READ_ONLY_OPTIMAL on first creation (so partial uploads can
// flip TRANSFER_DST → SHADER_READ_ONLY symmetrically).
void ensure_shadow_resources(VkDevice device, VmaAllocator alloc,
                             VkQueue queue, uint32_t qfam,
                             int W, int H,
                             VkImage& img, VmaAllocation& img_alloc,
                             VkImageView& view, VkSampler& sampler,
                             int& dim_out) {
    if (img) return;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    // RG8: R = shadow (0/1), G = blocker distance for shader-side PCSS.
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = { static_cast<uint32_t>(W),
                   static_cast<uint32_t>(H), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(alloc, &ici, &aci, &img, &img_alloc, nullptr),
             "terrain shadow image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device, &vci, nullptr, &view),
             "terrain shadow view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device, &si, nullptr, &sampler),
             "terrain shadow sampler");

    // Initialise to SHADER_READ_ONLY so the very first tick's
    // tile transition (SHADER_READ_ONLY → TRANSFER_DST → SHADER_READ_ONLY)
    // is symmetric. Without this the layout starts UNDEFINED and the
    // first frame would have to special-case the source layout.
    vkinit::one_time_submit(device, queue, qfam, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, img,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    dim_out = W;
}
} // namespace

void VulkanEngine::rebuild_terrain_shadow_texture(bool defer_bake) {
    if (terrain_data_.heights.empty()) return;
    // Reconstruct a Heightmap view over the cached data.
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;

    // Sun direction from the user settings (matches what cube.frag uses).
    float p = glm::radians(rt_.sun_pitch_deg);
    float y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun_dir(std::sin(y) * std::cos(p), std::sin(p),
                      std::cos(y) * std::cos(p));
    terrain_shadow_sun_dir_        = sun_dir;
    terrain_shadow_target_sun_dir_ = sun_dir;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_jobs_.clear();
        terrain_shadow_results_.clear();
    }

    // Sync bake stays at SS=1 for fast level load (~150ms). When the
    // user has supersample > 1 the texture is created at SS×heightmap-dim,
    // and we replicate each SS=1 texel into an SS×SS block so the
    // texture is fully populated from frame 1 — the worker then
    // re-bakes properly at the user's SS in the background.
    int ss = std::max(1, rt_.terrain_bake_supersample);
    {   // The texture is (dim+1)*ss per side: stay within the GPU limit
        // (4097 * 4 = 16388 already exceeds 16384 on many GPUs).
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        const uint32_t lim = props.limits.maxImageDimension2D;
        while (ss > 1 && uint32_t(std::max(hm.width(), hm.height()) * ss) > lim) ss /= 2;
        if (ss != rt_.terrain_bake_supersample) {
            log::warnf("[terrain] shadow bake supersample %d -> %d (max image dimension %u)",
                       rt_.terrain_bake_supersample, ss, lim);
            rt_.terrain_bake_supersample = ss;   // else the per-frame change check fires forever
        }
    }
    terrain_shadow_active_ss_ = ss;
    std::vector<uint8_t> base;
    if (defer_bake) {
        // R = 0 (lit), G = 255 (no occluder) - the worker streams in the
        // real bake right after startup, nearest tiles first.
        base.resize(static_cast<size_t>(hm.width()) * static_cast<size_t>(hm.height()) * 2);
        for (size_t i = 0; i < base.size(); i += 2) { base[i] = 0; base[i + 1] = 255; }
    } else {
        base = bake_heightmap_shadow(hm, sun_dir);
    }
    const int W = hm.width()  * ss;
    const int H = hm.height() * ss;
    std::vector<uint8_t> shadow;
    if (ss == 1) {
        shadow = std::move(base);
    } else {
        shadow.resize(static_cast<size_t>(W) * static_cast<size_t>(H) * 2);
        for (int z = 0; z < H; ++z) {
            for (int x = 0; x < W; ++x) {
                int sx = x / ss;
                int sz = z / ss;
                size_t dst = (static_cast<size_t>(z) * static_cast<size_t>(W) +
                              static_cast<size_t>(x)) * 2;
                size_t src = (static_cast<size_t>(sz) * static_cast<size_t>(hm.width()) +
                              static_cast<size_t>(sx)) * 2;
                shadow[dst]     = base[src];
                shadow[dst + 1] = base[src + 1];
            }
        }
    }

    const bool first_create = (terrain_shadow_image_ == VK_NULL_HANDLE);
    ensure_shadow_resources(device_, allocator_, graphics_queue_,
                             graphics_queue_family_, W, H,
                             terrain_shadow_image_, terrain_shadow_alloc_,
                             terrain_shadow_view_, terrain_shadow_sampler_,
                             terrain_shadow_dim_);

    // Stage upload via a scratch host-visible buffer (2 bytes/texel RG8).
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(W) * static_cast<VkDeviceSize>(H) * 2;
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
        std::memcpy(ai.pMappedData, shadow.data(), static_cast<size_t>(bytes));
    }

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = { static_cast<uint32_t>(W),
                                static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(allocator_, stage, stage_alloc);

    // Write the texture into descriptor binding 6 only on first
    // creation; the image handle doesn't change after that, so partial
    // tile uploads don't need to re-write the descriptor.
    if (first_create) {
        VkDescriptorImageInfo dii{};
        dii.sampler = terrain_shadow_sampler_;
        dii.imageView = terrain_shadow_view_;
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 6;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &dii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
}

// ---------------- Heightmap texture (binding 8) ----------------
//
// Uploads `terrain_data_.heights` as an R32_SFLOAT 2D image so
// terrain_raymarch.frag can sample the gameplay heightmap and produce
// a procedural-look surface that matches its shape exactly. One-shot
// upload at init; sculpt edits don't propagate (call this again to
// refresh if/when needed).
void VulkanEngine::init_terrain_height_texture() {
    if (terrain_data_.heights.empty()) {
        log::warn("init_terrain_height_texture: heightmap empty, skipping");
        return;
    }
    // Heightmap is stored as (dim+1) × (dim+1) — vertices, not cells —
    // so the texture must match that or the row-stride doesn't line up
    // and the upload scrambles the data (cells get shifted by 1 column
    // per row). Texture is dim+1 in each axis to match.
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;

    // Cache the max for the raymarch's upper-bound clip. Compute once
    // here — sculpt edits don't update the texture anyway.
    float max_h = 0.0f;
    for (float h : terrain_data_.heights) if (h > max_h) max_h = h;
    terrain_height_max_ = max_h;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32_SFLOAT;
    ici.extent = { static_cast<uint32_t>(W),
                   static_cast<uint32_t>(H), 1 };
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
                             &terrain_height_image_,
                             &terrain_height_alloc_, nullptr),
             "terrain height image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = terrain_height_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr, &terrain_height_view_),
             "terrain height view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;     // bilinear height interpolation
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &terrain_height_sampler_),
             "terrain height sampler");

    upload_terrain_height_payload(VK_IMAGE_LAYOUT_UNDEFINED);

    // Write binding 8.
    VkDescriptorImageInfo dii{};
    dii.sampler = terrain_height_sampler_;
    dii.imageView = terrain_height_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 8;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    log::infof("heightmap texture uploaded (%dx%d, %.1f MB)", W, H,
               static_cast<double>(W) * H * sizeof(float) / (1024.0 * 1024.0));
}

void VulkanEngine::refresh_terrain_height_texture() {
    if (!terrain_height_image_ || terrain_data_.heights.empty()) return;
    upload_terrain_height_payload(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    terrain_height_dirty_ = false;
}

// Shared upload helper for both init_ and refresh_terrain_height_texture.
// Allocates a host-mapped staging buffer, writes the delta-from-baseline
// (or raw heights if no baseline), runs a one-shot transition+copy+
// transition, frees the staging.
void VulkanEngine::upload_terrain_height_payload(VkImageLayout src_layout) {
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(W) *
                                static_cast<VkDeviceSize>(H) * sizeof(float);

    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    // DELTA from baseline so the raymarch shader can layer sculpt edits on
    // top of its own GLSL FBM. On first launch baseline == current →
    // delta is all zeros → no change to the procedural look.
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);
    if (terrain_baseline_heights_.size() == n) {
        std::vector<float> delta(n);
        for (size_t i = 0; i < n; ++i) {
            delta[i] = terrain_data_.heights[i] - terrain_baseline_heights_[i];
        }
        std::memcpy(ai.pMappedData, delta.data(), static_cast<size_t>(bytes));
    } else {
        std::memcpy(ai.pMappedData, terrain_data_.heights.data(),
                    static_cast<size_t>(bytes));
    }

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_height_image_,
                                  src_layout,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(W),
                                static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_height_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, terrain_height_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
}

bool VulkanEngine::save_terrain_heights() {
    if (terrain_data_.heights.empty()) return false;
    std::ofstream f("assets/level1_heights.bin", std::ios::binary);
    if (!f.is_open()) {
        log::error("save_terrain_heights: failed to open assets/level1_heights.bin");
        return false;
    }
    int32_t dim = terrain_data_.dim;
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    f.write(reinterpret_cast<const char*>(&terrain_data_.cell), sizeof(float));
    f.write(reinterpret_cast<const char*>(&terrain_data_.origin_x), sizeof(float));
    f.write(reinterpret_cast<const char*>(&terrain_data_.origin_z), sizeof(float));
    f.write(reinterpret_cast<const char*>(terrain_data_.heights.data()),
            terrain_data_.heights.size() * sizeof(float));
    log::infof("[terrain] saved %zu heights to assets/level1_heights.bin",
               terrain_data_.heights.size());
    return true;
}

void VulkanEngine::add_plateau_noise(float amplitude_m, float frequency) {
    if (terrain_data_.heights.empty()) return;
    // Gameplay plateau is centred at origin with a 28 m half-extent
    // (matches HeightmapParams in init_world). Apply noise inside,
    // taper it off across the 24 m blend ring so the rim stays smooth.
    const glm::vec2 plat_centre(0.0f, 0.0f);
    const glm::vec2 plat_ext   (28.0f, 28.0f);
    const float     plat_blend = 24.0f;

    auto fbm2 = [](glm::vec2 p) {
        float v = 0.5f  * fbm_noise2(p.x,         p.y) +
                  0.25f * fbm_noise2(p.x * 2.07f, p.y * 2.07f) +
                  0.125f* fbm_noise2(p.x * 4.13f, p.y * 4.13f);
        return v - 0.4375f;   // re-centre around zero
    };

    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    for (int iz = 0; iz < H; ++iz) {
        float wz = terrain_data_.origin_z + iz * terrain_data_.cell;
        for (int ix = 0; ix < W; ++ix) {
            float wx = terrain_data_.origin_x + ix * terrain_data_.cell;
            glm::vec2 d = glm::abs(glm::vec2(wx, wz) - plat_centre) - plat_ext;
            float dout = std::max(std::max(d.x, 0.0f), std::max(d.y, 0.0f));
            float t = std::clamp(dout / plat_blend, 0.0f, 1.0f);
            float w = 1.0f - t * t * (3.0f - 2.0f * t);    // 1 inside, 0 outside
            if (w <= 0.001f) continue;
            float n = fbm2(glm::vec2(wx, wz) * frequency);
            terrain_data_.heights[static_cast<size_t>(iz) *
                                   static_cast<size_t>(W) +
                                   static_cast<size_t>(ix)] +=
                n * amplitude_m * w;
        }
    }
    log::infof("[terrain] plateau noise applied (amp=%.2f m, freq=%.3f)",
               amplitude_m, frequency);
    terrain_height_dirty_ = true;
    // Physics, RT and culling must follow the new surface too (only the
    // height textures did: walk-through bumps, stale RT shadows).
    terrain_clod_pyramid_ = build_clod_pyramid(terrain_data_.heights.data(),
                                               terrain_data_.dim,
                                               terrain_data_.cell, 16.0f);
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
    refresh_terrain_blas();
    refresh_terrain_collision();
}

void VulkanEngine::add_fbm_erosion_to_sculpted(float amplitude_factor,
                                                 float frequency) {
    if (terrain_data_.heights.empty()) return;
    if (terrain_baseline_heights_.size() != terrain_data_.heights.size()) return;

    auto fbm5 = [](glm::vec2 p) {
        // 5-octave value-noise FBM via the shared fbm_noise2 helper.
        // Decreasing amplitude per octave produces the classic
        // ridged-mountain look.
        float v = 0.50000f * fbm_noise2(p.x,          p.y) +
                  0.25000f * fbm_noise2(p.x * 2.07f,  p.y * 2.07f) +
                  0.12500f * fbm_noise2(p.x * 4.13f,  p.y * 4.13f) +
                  0.06250f * fbm_noise2(p.x * 8.21f,  p.y * 8.21f) +
                  0.03125f * fbm_noise2(p.x * 16.47f, p.y * 16.47f);
        return v - 0.484375f;   // re-centre around zero
    };

    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    int touched = 0;
    for (int iz = 0; iz < H; ++iz) {
        float wz = terrain_data_.origin_z + iz * terrain_data_.cell;
        for (int ix = 0; ix < W; ++ix) {
            const size_t idx = static_cast<size_t>(iz) *
                               static_cast<size_t>(W) +
                               static_cast<size_t>(ix);
            float delta = terrain_data_.heights[idx] -
                          terrain_baseline_heights_[idx];
            // Only erode cells the user actually raised — preserve
            // procedural terrain + lowered areas as-is.
            if (delta <= 0.25f) continue;
            float wx = terrain_data_.origin_x + ix * terrain_data_.cell;
            float n = fbm5(glm::vec2(wx, wz) * frequency);
            terrain_data_.heights[idx] += n * delta * amplitude_factor;
            ++touched;
        }
    }
    log::infof("[terrain] FBM erosion baked into %d sculpted cells "
               "(amp=%.2f, freq=%.3f)", touched, amplitude_factor, frequency);
    terrain_height_dirty_ = true;
    // CLOD raster follows the heightmap texture; refresh the whole
    // min/max pyramid since the erosion touched sculpted cells anywhere.
    terrain_clod_pyramid_ = build_clod_pyramid(terrain_data_.heights.data(),
                                               terrain_data_.dim,
                                               terrain_data_.cell, 16.0f);
    terrain_blas_dirty_ = true;     // physics + RT follow the eroded surface
    terrain_jolt_dirty_ = true;
    refresh_terrain_blas();
    refresh_terrain_collision();
}

void VulkanEngine::destroy_terrain_height_texture() {
    if (terrain_height_view_) {
        vkDestroyImageView(device_, terrain_height_view_, nullptr);
        terrain_height_view_ = VK_NULL_HANDLE;
    }
    if (terrain_height_image_) {
        vmaDestroyImage(allocator_, terrain_height_image_,
                         terrain_height_alloc_);
        terrain_height_image_ = VK_NULL_HANDLE;
        terrain_height_alloc_ = nullptr;
    }
    if (terrain_height_sampler_) {
        vkDestroySampler(device_, terrain_height_sampler_, nullptr);
        terrain_height_sampler_ = VK_NULL_HANDLE;
    }
    if (terrain_height_full_view_) {
        vkDestroyImageView(device_, terrain_height_full_view_, nullptr);
        terrain_height_full_view_ = VK_NULL_HANDLE;
    }
    if (terrain_height_full_image_) {
        vmaDestroyImage(allocator_, terrain_height_full_image_,
                         terrain_height_full_alloc_);
        terrain_height_full_image_ = VK_NULL_HANDLE;
        terrain_height_full_alloc_ = nullptr;
    }
    if (terrain_height_full_sampler_) {
        vkDestroySampler(device_, terrain_height_full_sampler_, nullptr);
        terrain_height_full_sampler_ = VK_NULL_HANDLE;
    }
}

// Stages the FULL baked terrain heights (NOT the baseline delta) into
// terrain_height_full_image_. This is the exact terrain_data_.heights
// array the CDLOD mesh + physics are built from, so the water path can
// reference the real mesh surface without evaluating FBM noise.
void VulkanEngine::upload_terrain_height_full_payload(VkImageLayout src_layout) {
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(W) *
                                static_cast<VkDeviceSize>(H) * sizeof(float);
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, terrain_data_.heights.data(),
                static_cast<size_t>(bytes));
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_height_full_image_,
                                 src_layout,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(W),
                               static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_height_full_image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &region);
        vkinit::transition_image(cb, terrain_height_full_image_,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
}

// ---- Quadtree CDLOD terrain (engine side) -------------------------------

void VulkanEngine::init_terrain_clod() {
    if (terrain_data_.heights.empty()) {
        log::warn("init_terrain_clod: heightmap empty, skipping");
        return;
    }
    terrain_clod_patch_ = build_clod_patch_mesh(device_, allocator_,
                                                graphics_queue_,
                                                graphics_queue_family_);
    terrain_clod_pyramid_ = build_clod_pyramid(terrain_data_.heights.data(),
                                               terrain_data_.dim,
                                               terrain_data_.cell, 16.0f);

    // Host-visible per-frame instance ring: kFrameOverlap slots × two
    // lists (camera, shadow) × kClodMaxInstances.
    const VkDeviceSize bytes = VkDeviceSize(kFrameOverlap) * 2 *
                               kClodMaxInstances * sizeof(ClodInstance);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                             &terrain_clod_inst_buf_,
                             &terrain_clod_inst_alloc_, &ai),
             "terrain clod instance buffer");
    terrain_clod_inst_map_ = ai.pMappedData;
    log::infof("[terrain] CLOD ready: patch %u verts / %u idx, pyramid %d "
               "levels, instance ring %.0f KB",
               terrain_clod_patch_.vertex_count,
               terrain_clod_patch_.index_count,
               terrain_clod_pyramid_.levels,
               double(bytes) / 1024.0);
}

void VulkanEngine::destroy_terrain_clod() {
    destroy_mesh(allocator_, terrain_clod_patch_);
    if (terrain_clod_inst_buf_) {
        vmaDestroyBuffer(allocator_, terrain_clod_inst_buf_,
                         terrain_clod_inst_alloc_);
        terrain_clod_inst_buf_ = VK_NULL_HANDLE;
        terrain_clod_inst_alloc_ = nullptr;
        terrain_clod_inst_map_ = nullptr;
    }
}

glm::vec4 VulkanEngine::terrain_clod_hm_params() const {
    const float cell = (terrain_data_.cell > 0.0f) ? terrain_data_.cell : 1.0f;
    return glm::vec4(terrain_data_.origin_x, terrain_data_.origin_z,
                     1.0f / cell, 1.0f / float(terrain_data_.dim + 1));
}

void VulkanEngine::update_terrain_clod_instances() {
    terrain_clod_cam_count_ = terrain_clod_shadow_count_ = 0;
    if (!terrain_clod_inst_map_ || terrain_data_.heights.empty() ||
        rt_.terrain_raymarch_enabled) {
        return;
    }
    ClodParams p;
    p.world_origin_x = terrain_data_.origin_x;
    p.world_origin_z = terrain_data_.origin_z;
    p.world_side     = float(terrain_data_.dim) * terrain_data_.cell;
    p.leaf_size_m    = 16.0f;
    p.levels = 1;
    while (p.leaf_size_m * float(1 << (p.levels - 1)) < p.world_side - 0.5f)
        ++p.levels;
    p.range0 = std::max(10.0f, rt_.terrain_clod_range0);

    select_clod_nodes(p, terrain_clod_pyramid_, current_frame_view_.eye_pos,
                      terrain_clod_nodes_);

    const uint32_t slot = uint32_t(frame_number_ % kFrameOverlap);
    const uint32_t list_bytes = kClodMaxInstances * uint32_t(sizeof(ClodInstance));
    terrain_clod_cam_offset_    = slot * 2 * list_bytes;
    terrain_clod_shadow_offset_ = terrain_clod_cam_offset_ + list_bytes;
    auto* cam_dst = reinterpret_cast<ClodInstance*>(
        static_cast<char*>(terrain_clod_inst_map_) + terrain_clod_cam_offset_);
    auto* sh_dst = reinterpret_cast<ClodInstance*>(
        static_cast<char*>(terrain_clod_inst_map_) + terrain_clod_shadow_offset_);

    const Frustum& camf = current_frame_view_.frustum;
    const Frustum lightf = extract_frustum(sun_shadow_light_vp_);
    for (const ClodNode& n : terrain_clod_nodes_) {
        if (terrain_clod_cam_count_ < kClodMaxInstances &&
            aabb_visible(camf, n.aabb_min, n.aabb_max)) {
            cam_dst[terrain_clod_cam_count_++] = n.inst;
        }
        if (terrain_clod_shadow_count_ < kClodMaxInstances &&
            aabb_visible(lightf, n.aabb_min, n.aabb_max)) {
            sh_dst[terrain_clod_shadow_count_++] = n.inst;
        }
    }
}

void VulkanEngine::init_terrain_height_full_texture() {
    if (terrain_data_.heights.empty()) {
        log::warn("init_terrain_height_full_texture: heightmap empty, skipping");
        return;
    }
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32_SFLOAT;
    ici.extent = { static_cast<uint32_t>(W), static_cast<uint32_t>(H), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &terrain_height_full_image_,
                             &terrain_height_full_alloc_, nullptr),
             "terrain height full image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = terrain_height_full_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr,
                               &terrain_height_full_view_),
             "terrain height full view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr,
                             &terrain_height_full_sampler_),
             "terrain height full sampler");

    upload_terrain_height_full_payload(VK_IMAGE_LAYOUT_UNDEFINED);

    // Write binding 17.
    VkDescriptorImageInfo dii{};
    dii.sampler = terrain_height_full_sampler_;
    dii.imageView = terrain_height_full_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 17;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    log::infof("full heightmap texture uploaded (%dx%d, %.1f MB)", W, H,
               static_cast<double>(W) * H * sizeof(float) / (1024.0 * 1024.0));
}

void VulkanEngine::refresh_terrain_height_full_texture() {
    if (!terrain_height_full_image_ || terrain_data_.heights.empty()) return;
    upload_terrain_height_full_payload(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ---------------- Sun shadow map (single-cascade ortho) ----------------
//
// Standard depth-only shadow pass: one 2048² D32 depth target written
// from the sun's POV each frame, sampled by grass.vert as a
// sampler2DShadow at descriptor binding 7. Cube.frag continues to use
// RT shadow rays — we don't move the whole engine to CSM yet, this is
// the cheap path that lets dynamic occluders (castle, dyn-props) cast
// shadow on grass without firing a ray per blade.
//
// init_sun_shadow_resources / destroy_sun_shadow_resources /
// update_sun_shadow_light_vp / render_sun_shadow_pass moved to
// vk_engine/sun_shadow.cpp.

void VulkanEngine::start_terrain_shadow_worker() {
    if (terrain_data_.heights.empty()) return;
    if (terrain_shadow_worker_.joinable()) return;     // already running
    // Snapshot the heightmap into a Heightmap struct the worker can read
    // without locks. Future sculpt edits won't propagate; if/when we
    // care, call stop_terrain_shadow_worker / start_terrain_shadow_worker
    // again to rebuild the snapshot.
    terrain_shadow_baker_hm_.dim       = terrain_data_.dim;
    terrain_shadow_baker_hm_.cell      = terrain_data_.cell;
    terrain_shadow_baker_hm_.origin_x  = terrain_data_.origin_x;
    terrain_shadow_baker_hm_.origin_z  = terrain_data_.origin_z;
    terrain_shadow_baker_hm_.heights   = terrain_data_.heights;
    terrain_shadow_stop_.store(false);
    terrain_shadow_worker_ = std::thread([this]() { terrain_shadow_worker_loop(); });
    const unsigned hc = (std::max)(2u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i + 2 < hc / 2; ++i)
        terrain_shadow_helpers_.emplace_back([this]() { terrain_shadow_worker_loop(); });
}

void VulkanEngine::stop_terrain_shadow_worker() {
    if (!terrain_shadow_worker_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_stop_.store(true);
        terrain_shadow_jobs_.clear();
    }
    terrain_shadow_cv_.notify_all();
    terrain_shadow_worker_.join();
    for (auto& t : terrain_shadow_helpers_) if (t.joinable()) t.join();
    terrain_shadow_helpers_.clear();
}

void VulkanEngine::terrain_shadow_worker_loop() {
    // Background bake loop. Pops one job at a time, runs the tile bake
    // on the worker thread (zero impact on main FPS), and pushes the
    // result back. Main thread drains results in tick_terrain_shadow_progressive.
    while (true) {
        ShadowBakeJob job;
        {
            std::unique_lock<std::mutex> lk(terrain_shadow_mutex_);
            terrain_shadow_cv_.wait(lk, [&]{
                return !terrain_shadow_jobs_.empty() ||
                       terrain_shadow_stop_.load();
            });
            if (terrain_shadow_stop_.load()) return;
            job = terrain_shadow_jobs_.front();
            terrain_shadow_jobs_.pop_front();
        }
        // Bake outside the lock so other threads keep running.
        ShadowBakeResult r;
        r.ix = job.ix; r.iz = job.iz; r.w = job.w; r.h = job.h; r.ss = job.ss;
        r.sun_dir = job.sun_dir;
        r.data.resize(static_cast<size_t>(job.w) * static_cast<size_t>(job.h) * 2);
        if (job.ss <= 1) {
            bake_heightmap_shadow_tile(terrain_shadow_baker_hm_, job.sun_dir,
                                        job.ix, job.iz, job.w, job.h, r.data.data());
        } else {
            bake_heightmap_shadow_tile_ss(terrain_shadow_baker_hm_, job.sun_dir,
                                           job.ix, job.iz, job.w, job.h, job.ss,
                                           r.data.data());
        }
        {
            std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
            terrain_shadow_results_.push_back(std::move(r));
        }
    }
}

void VulkanEngine::enqueue_terrain_shadow_rebake(glm::vec3 target_sun) {
    if (terrain_data_.heights.empty()) return;
    terrain_shadow_target_sun_dir_ = target_sun;

    // Tile coordinates are in TEXTURE space (heightmap_dim+1 × ss).
    const int ss = std::max(1, terrain_shadow_active_ss_);
    const int W = (terrain_data_.dim + 1) * ss;
    const int H = (terrain_data_.dim + 1) * ss;
    const int ts = kShadowTileSize;
    const glm::vec3 cam = player_.eye_position();

    // Build the full tile list, sorted nearest-first so the worker
    // bakes near-camera tiles before far ones.
    std::vector<ShadowBakeTile> tiles;
    int n_tiles = ((W + ts - 1) / ts) * ((H + ts - 1) / ts);
    tiles.reserve(static_cast<size_t>(n_tiles));
    const float sub_cell = terrain_data_.cell / static_cast<float>(ss);
    for (int iz = 0; iz < H; iz += ts) {
        int th = std::min(ts, H - iz);
        for (int ix = 0; ix < W; ix += ts) {
            int tw = std::min(ts, W - ix);
            float wx = terrain_data_.origin_x +
                       (static_cast<float>(ix) + 0.5f * static_cast<float>(tw)) *
                       sub_cell;
            float wz = terrain_data_.origin_z +
                       (static_cast<float>(iz) + 0.5f * static_cast<float>(th)) *
                       sub_cell;
            float dx = wx - cam.x, dz = wz - cam.z;
            tiles.push_back({ix, iz, tw, th, dx * dx + dz * dz});
        }
    }
    std::sort(tiles.begin(), tiles.end(),
              [](const ShadowBakeTile& a, const ShadowBakeTile& b) {
                  return a.dist_sq < b.dist_sq;
              });

    // Replace the worker's job queue. Drop any older results too — they
    // were baked against a stale sun direction or a different ss.
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_jobs_.clear();
        terrain_shadow_results_.clear();
        for (const auto& t : tiles) {
            terrain_shadow_jobs_.push_back({t.ix, t.iz, t.w, t.h, ss, target_sun});
        }
    }
    terrain_shadow_cv_.notify_all();
}

void VulkanEngine::tick_terrain_shadow_progressive() {
    if (terrain_shadow_image_ == VK_NULL_HANDLE) return;
    // Async path: drain a few finished bake results from the worker
    // thread and upload them to the GPU image. Bake CPU work happens
    // off-thread so there's no impact on FPS even when many tiles are
    // baking. We early-return if no results are ready.
    std::vector<ShadowBakeResult> ready;
    bool jobs_remaining = false;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        const int max_take = kShadowUploadsPerFrame;
        while (ready.size() < static_cast<size_t>(max_take) &&
               !terrain_shadow_results_.empty()) {
            auto& r = terrain_shadow_results_.front();
            // Drop stale results (sun has changed since this tile was
            // queued, or supersample doesn't match the live texture).
            if (glm::distance(r.sun_dir, terrain_shadow_target_sun_dir_) < 0.001f &&
                r.ss == terrain_shadow_active_ss_) {
                ready.push_back(std::move(r));
            }
            terrain_shadow_results_.pop_front();
        }
        jobs_remaining = !terrain_shadow_jobs_.empty() ||
                         !terrain_shadow_results_.empty();
    }
    if (ready.empty()) {
        if (!jobs_remaining) {
            terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
        }
        return;
    }
    // Pack tiles into one staging buffer, upload via a single submit.
    size_t total_bytes = 0;
    for (const auto& r : ready) {
        total_bytes += static_cast<size_t>(r.w) * static_cast<size_t>(r.h) * 2;
    }
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = total_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    uint8_t* mapped = static_cast<uint8_t*>(ai.pMappedData);

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(ready.size());
    size_t off = 0;
    for (const auto& r : ready) {
        size_t bytes = static_cast<size_t>(r.w) * static_cast<size_t>(r.h) * 2;
        std::memcpy(mapped + off, r.data.data(), bytes);
        VkBufferImageCopy reg{};
        reg.bufferOffset = off;
        reg.bufferRowLength = 0;
        reg.bufferImageHeight = 0;
        reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.imageSubresource.mipLevel = 0;
        reg.imageSubresource.baseArrayLayer = 0;
        reg.imageSubresource.layerCount = 1;
        reg.imageOffset = { r.ix, r.iz, 0 };
        reg.imageExtent = { static_cast<uint32_t>(r.w),
                            static_cast<uint32_t>(r.h), 1 };
        regions.push_back(reg);
        off += bytes;
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                static_cast<uint32_t>(regions.size()),
                                regions.data());
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    if (!jobs_remaining) {
        terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
    }
}


float VulkanEngine::sample_terrain_height(float x, float z) const {
    if (terrain_data_.heights.empty() || terrain_data_.dim <= 0) return 0.0f;
    const int W = terrain_data_.dim + 1;
    float fx = (x - terrain_data_.origin_x) / terrain_data_.cell;
    float fz = (z - terrain_data_.origin_z) / terrain_data_.cell;
    if (fx < 0.0f || fz < 0.0f) return 0.0f;
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    if (ix >= terrain_data_.dim || iz >= terrain_data_.dim) return 0.0f;
    float tx = fx - static_cast<float>(ix);
    float tz = fz - static_cast<float>(iz);
    auto h = [&](int x_, int z_) {
        return terrain_data_.heights[static_cast<size_t>(z_) * static_cast<size_t>(W) +
                                     static_cast<size_t>(x_)];
    };
    float h00 = h(ix,     iz);
    float h10 = h(ix + 1, iz);
    float h01 = h(ix,     iz + 1);
    float h11 = h(ix + 1, iz + 1);
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    return h0 * (1.0f - tz) + h1 * tz;
}


// ---- Procedural terrain textures (curvature/sky-vis aux, grass mask, fog) ----

// Terrain aux bake (AAA Phases 4/5). RG8 1024² covering the 2048 m
// world: R = curvature (0.5 flat, <0.5 concave gullies — sediment +
// cavity darkening, >0.5 convex ridges — exposed rock), G = sky
// visibility from an 8-direction horizon scan (macro AO for valleys).
// CPU bake, multi-threaded; ~100 ms at load.
void VulkanEngine::init_terrain_aux_texture() {
    if (terrain_data_.heights.empty()) return;
    const int dim = 1024;
    const float kWorldSide = 2048.0f;
    std::vector<uint8_t> data(size_t(dim) * dim * 2, 0);

    auto h_at = [&](float wx, float wz) {
        return sample_terrain_height(wx, wz);
    };
    const float curv_r = 5.0f;     // curvature stencil radius (m)
    const glm::vec2 dirs[8] = {
        { 1, 0 }, { 0.7071f, 0.7071f }, { 0, 1 }, { -0.7071f, 0.7071f },
        { -1, 0 }, { -0.7071f, -0.7071f }, { 0, -1 }, { 0.7071f, -0.7071f },
    };
    const float steps[6] = { 3.0f, 6.0f, 11.0f, 20.0f, 38.0f, 70.0f };

    unsigned n_threads = (std::max)(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    int rows_per = (dim + int(n_threads) - 1) / int(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        int z0 = int(t) * rows_per;
        int z1 = (std::min)(dim, z0 + rows_per);
        if (z0 >= z1) break;
        workers.emplace_back([&, z0, z1]() {
            for (int tz = z0; tz < z1; ++tz) {
                float wz = (float(tz) / dim - 0.5f) * kWorldSide;
                for (int tx = 0; tx < dim; ++tx) {
                    float wx = (float(tx) / dim - 0.5f) * kWorldSide;
                    float h0 = h_at(wx, wz);
                    // Curvature: 4-tap Laplacian at curv_r.
                    float lap = h_at(wx + curv_r, wz) + h_at(wx - curv_r, wz)
                              + h_at(wx, wz + curv_r) + h_at(wx, wz - curv_r)
                              - 4.0f * h0;
                    // lap > 0 = concave (valley floor). Map ±~6 m of
                    // Laplacian to the full byte range around 0.5.
                    float curv = std::clamp(0.5f - lap * 0.08f, 0.0f, 1.0f);
                    // Sky visibility: per direction take the max horizon
                    // slope over the step ladder; vis = 1 at open sky.
                    float vis_sum = 0.0f;
                    for (const glm::vec2& d : dirs) {
                        float max_slope = 0.0f;
                        for (float s : steps) {
                            float dh = h_at(wx + d.x * s, wz + d.y * s) - h0;
                            if (dh > 0.0f) {
                                max_slope = (std::max)(max_slope, dh / s);
                            }
                        }
                        float ang = std::atan(max_slope);   // 0..~pi/2
                        vis_sum += 1.0f - ang * (2.0f / 3.14159265f);
                    }
                    float vis = vis_sum / 8.0f;
                    size_t o = (size_t(tz) * dim + tx) * 2;
                    data[o]     = uint8_t(curv * 255.0f + 0.5f);
                    data[o + 1] = uint8_t(std::clamp(vis, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        });
    }
    for (auto& th : workers) th.join();

    // Upload as RG8 + write binding 26.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = { uint32_t(dim), uint32_t(dim), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &terrain_aux_image_,
                            &terrain_aux_alloc_, nullptr), "terrain aux image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = terrain_aux_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &terrain_aux_view_),
             "terrain aux view");
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &terrain_aux_sampler_),
             "terrain aux sampler");

    // Staging upload.
    VkBuffer stage = VK_NULL_HANDLE; VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = data.size();
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, &ai);
    std::memcpy(ai.pMappedData, data.data(), data.size());
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_aux_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { uint32_t(dim), uint32_t(dim), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_aux_image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        vkinit::transition_image(cb, terrain_aux_image_,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);

    // Write binding 26 (runs after descriptors are created).
    VkDescriptorImageInfo dii{};
    dii.sampler = terrain_aux_sampler_;
    dii.imageView = terrain_aux_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 26;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    log::infof("[terrain] aux bake (curvature + sky-vis) %dx%d uploaded", dim, dim);
}

void VulkanEngine::destroy_terrain_aux_texture() {
    deferred_desc_sig_ = 0;   // handles may be recreated (even at the same address)
    if (terrain_aux_view_)    vkDestroyImageView(device_, terrain_aux_view_, nullptr);
    if (terrain_aux_image_)   vmaDestroyImage(allocator_, terrain_aux_image_, terrain_aux_alloc_);
    if (terrain_aux_sampler_) vkDestroySampler(device_, terrain_aux_sampler_, nullptr);
    terrain_aux_view_ = VK_NULL_HANDLE;
    terrain_aux_image_ = VK_NULL_HANDLE;
    terrain_aux_alloc_ = nullptr;
    terrain_aux_sampler_ = VK_NULL_HANDLE;
}

void VulkanEngine::init_grass_mask_texture() {
    const int dim = kGrassMaskDim;
    const float kWorldSide = 2048.0f;     // matches kHeightmapSide
    // RG8 — 2 bytes per texel. The vector lives on the engine so the
    // GrassAdd/GrassRemove brush can paint into it without rebaking the
    // noise. We populate it here, upload from it once, and reupload via
    // reupload_grass_mask() after every paint stroke.
    grass_mask_data_.assign(static_cast<size_t>(dim) * dim * 2, 0);
    std::vector<uint8_t>& data = grass_mask_data_;

    auto smoothstep_f = [](float a, float b, float v) {
        float t = std::clamp((v - a) / (b - a), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    for (int z = 0; z < dim; ++z) {
        for (int x = 0; x < dim; ++x) {
            float wx = (static_cast<float>(x) + 0.5f) / dim * kWorldSide
                       - kWorldSide * 0.5f;
            float wz = (static_cast<float>(z) + 0.5f) / dim * kWorldSide
                       - kWorldSide * 0.5f;
            // R: presence — low-freq FBM, smoothstep gate.
            float n = fbm_noise2(wx * 0.02f, wz * 0.02f);
            float presence = smoothstep_f(0.40f, 0.55f, n);
            // Castle keep-out. The raymarched grass shader plants
            // blades at terrain Y (= plateau height inside castle), so
            // without this guard blade tips poke through the castle
            // floor brushes. The mask is 1024² over a 2048 m world
            // (в‰€2 m / texel) and the shader samples it with LINEAR
            // filtering — bilinear can leak presence ~2-4 m inward
            // from a hard boundary, so we use a wider keep-out band
            // (18 m vs the 11 m outer wall) to keep ALL bilinear
            // samples inside the floor's extent at exactly zero.
            const float kCastleHalf = 18.0f;
            if (std::abs(wx) < kCastleHalf && std::abs(wz) < kCastleHalf) {
                presence = 0.0f;
            }
            // G: slope mag — finite-diff on real terrain heights, so
            // cliffs/peaks naturally read as steep. Stored normalized
            // [0..1] = raw_slope / 2.0; shader un-normalizes.
            const float kEps = 2.0f;       // metres
            float h_c = sample_terrain_height(wx, wz);
            float h_r = sample_terrain_height(wx + kEps, wz);
            float h_u = sample_terrain_height(wx, wz + kEps);
            float dx = (h_r - h_c) / kEps;
            float dz = (h_u - h_c) / kEps;
            float slope_mag = std::sqrt(dx * dx + dz * dz);
            float slope_norm = std::clamp(slope_mag * 0.5f, 0.0f, 1.0f);
            const size_t idx = (static_cast<size_t>(z) * dim + x) * 2;
            data[idx + 0] =
                static_cast<uint8_t>(std::clamp(presence, 0.0f, 1.0f) * 255.0f);
            data[idx + 1] =
                static_cast<uint8_t>(slope_norm * 255.0f);
        }
    }

    // Overlay any persisted user paint on top of the freshly baked
    // presence/slope field. Mirrors the height-load path: bake the
    // procedural baseline first so a missing save file is a no-op,
    // then replace bytes if the file is present.
    load_grass_mask();
    // Keep the road paving + verge clear of blades.
    clear_grass_mask(castle_road_, grass_mask_data_, dim, kWorldSide);

    // Upload as a sampled RG8 image.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = { static_cast<uint32_t>(dim),
                   static_cast<uint32_t>(dim), 1 };
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
                             &grass_mask_.image, &grass_mask_.alloc, nullptr),
             "grass mask image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = grass_mask_.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &grass_mask_.view),
             "grass mask view");

    // Stage + copy + transition.
    const VkDeviceSize bytes = data.size();
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, data.data(), bytes);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(dim),
                                static_cast<uint32_t>(dim), 1 };
        vkCmdCopyBufferToImage(cb, stage, grass_mask_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    log::infof("grass mask baked: %dx%d RG8 (%.1f KB)", dim, dim,
               static_cast<double>(bytes) / 1024.0);
}

void VulkanEngine::destroy_grass_mask_texture() {
    if (grass_mask_.view)  vkDestroyImageView(device_, grass_mask_.view, nullptr);
    if (grass_mask_.image) vmaDestroyImage(allocator_, grass_mask_.image, grass_mask_.alloc);
    grass_mask_ = {};
}

// Bake the 3-octave fog wisp pattern into a 256² R8 texture so the
// terrain raymarch's volumetric fog density + godray probe can sample
// it once instead of running 3× noise2 per tap. World-space q is
// rescaled to UV by 1/16 so the texture tiles every 16 q-units (в‰€800 m
// of world); the texture sampler uses REPEAT addressing so time
// scrolling and large camera moves stay seamless. Wave-pattern doesn't
// need to bit-match the GLSL noise2 — it just needs to look "wispy".
void VulkanEngine::init_fog_wisp_texture() {
    const int dim = kFogWispDim;
    std::vector<uint8_t> data(static_cast<size_t>(dim) * dim);
    // 16 q-units of pattern fit in the texture; CLAMP/REPEAT works
    // because the FBM is statistically self-similar across periods.
    const float kPeriod = 16.0f;
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            float qx = (static_cast<float>(x) / dim) * kPeriod;
            float qy = (static_cast<float>(y) / dim) * kPeriod;
            // Same weights + scales as the shader's runtime expression:
            //   w = 0.55 noise2(q) + 0.30 noise2(q*2.13) + 0.15 noise2(q*4.27)
            float w = 0.55f * fbm_noise2(qx,         qy)
                    + 0.30f * fbm_noise2(qx * 2.13f, qy * 2.13f)
                    + 0.15f * fbm_noise2(qx * 4.27f, qy * 4.27f);
            w = std::clamp(w, 0.0f, 1.0f);
            data[static_cast<size_t>(y) * dim + x] =
                static_cast<uint8_t>(w * 255.0f);
        }
    }

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = { static_cast<uint32_t>(dim), static_cast<uint32_t>(dim), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &fog_wisp_.image, &fog_wisp_.alloc, nullptr),
             "fog wisp image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = fog_wisp_.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &fog_wisp_.view),
             "fog wisp view");

    // Stage + copy + transition.
    const VkDeviceSize bytes = data.size();
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, data.data(), bytes);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, fog_wisp_.image,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(dim),
                                static_cast<uint32_t>(dim), 1 };
        vkCmdCopyBufferToImage(cb, stage, fog_wisp_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, fog_wisp_.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    log::infof("fog wisp baked: %dx%d R8 (%.1f KB)", dim, dim,
               static_cast<double>(bytes) / 1024.0);
}

void VulkanEngine::destroy_fog_wisp_texture() {
    if (fog_wisp_.view)  vkDestroyImageView(device_, fog_wisp_.view, nullptr);
    if (fog_wisp_.image) vmaDestroyImage(allocator_, fog_wisp_.image, fog_wisp_.alloc);
    fog_wisp_ = {};
}

} // namespace qlike
