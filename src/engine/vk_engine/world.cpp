// World-side state: level + meshes + skybox + texture array + viewmodel +
// dyn-prop spawning + the per-frame `render_world` raster pass that drives
// every cube/cylinder draw the engine emits. Lives here because all of these
// touch `world_`, `dyn_props_`, `physics_`, the TLAS instance cache (via
// rebuild_dyn_render_cache), and the same pipeline_layout_/scene_desc_set_.

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
#include <limits>
#include <vector>

// stb_image forward declarations. The implementation lives in skybox.cpp's
// TU (the project's single STB_IMAGE_IMPLEMENTATION owner). We need the raw
// loader here to AO-multiply the stylized-grass albedo before uploading
// (the texture upload path doesn't expose a pre-process hook).
extern "C" {
    unsigned char* stbi_load(const char*, int*, int*, int*, int);
    void           stbi_image_free(void*);
}

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
    // Re-centre roughly around 0 — fbm with `n.v` in [0,1] sums to ≈
    // half the harmonic series. The bias is approximate and harmless;
    // the brush noise scale only needs symmetric variation around 0.
    return a - 0.5f;
}
} // namespace

// Distance-based LOD selector for terrain chunks. Used by all three
// terrain draw paths (depth pre-pass, color pass, shadow pass) so LOD
// stays consistent РІР‚вЂќ if the pre-pass picked half-LOD and the color
// pass picked full, the depth values would mismatch and the color
// pass's LESS_OR_EQUAL test would discard fragments. Thresholds are
// pulled from rt_ so the user can push higher-resolution terrain
// further out via the sliders in the Graphics Settings menu.
// pick_terrain_lod moved to terrain.h so sun_shadow.cpp can share it.

// CD-LOD morph factor for a chunk currently rendered at LOD 0. Ramps
// from 0 (full LOD-0 surface) to 1 (LOD-1 stride-2 interp) over the
// last 25% of the LOD-0 distance band. Threshold scales with the
// user-selected lod1 distance so morphing stays linked to where the
// LOD switch actually happens.
static float pick_terrain_morph(const TerrainChunk& c, glm::vec3 cam_xz,
                                int lod, float lod1) {
    if (lod != 0) return 0.0f;
    // Closest-point-on-AABB distance, identical metric to
    // pick_terrain_lod above. Was chunk-centre distance which could
    // misalign the morph fade window with the picker's LOD boundary
    // after the picker switched to AABB-clamp.
    float cx = std::min(std::max(cam_xz.x, c.aabb_min.x), c.aabb_max.x);
    float cz = std::min(std::max(cam_xz.z, c.aabb_min.z), c.aabb_max.z);
    float dx = cx - cam_xz.x;
    float dz = cz - cam_xz.z;
    float d = std::sqrt(dx * dx + dz * dz);
    float fade_start = lod1 * 0.75f;
    float fade_end   = lod1;
    if (d <= fade_start) return 0.0f;
    if (d >= fade_end)   return 1.0f;
    float t = (d - fade_start) / (fade_end - fade_start);
    return t * t * (3.0f - 2.0f * t);
}

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
    if (terrain_chunks_.chunks.empty()) return;
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
    // multiplies the sculpt delta as `1 + noiseВ·strength` вЂ” values
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
            // scale (0.003 ≈ 333 m base wavelength) means adjacent
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
            // Per-cell noise multiplier вЂ” only kicks in for additive
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
                    // Add FBM-eroded detail (same n.v/(1+|∇d|²) recipe
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

    // Mark affected chunks dirty for next-frame mesh rebuild. A chunk's
    // sample range is [origin_ix, origin_ix + chunk_cells] in X and Z,
    // so we mark any chunk whose range overlaps the brush footprint.
    // Use a vector<bool> indexed by chunk id (was: linear scan of
    // terrain_dirty_chunks_ per cell вЂ” O(NВІ) on big sculpts).
    const int chunk_cells = terrain_chunks_.chunk_cells;
    std::vector<bool> already_dirty(terrain_chunks_.chunks.size(), false);
    for (int ci : terrain_dirty_chunks_) {
        if (ci >= 0 && ci < static_cast<int>(already_dirty.size())) {
            already_dirty[ci] = true;
        }
    }
    for (size_t ci = 0; ci < terrain_chunks_.chunks.size(); ++ci) {
        const auto& c = terrain_chunks_.chunks[ci];
        int cix0 = c.origin_ix;
        int cix1 = c.origin_ix + chunk_cells;
        int ciz0 = c.origin_iz;
        int ciz1 = c.origin_iz + chunk_cells;
        if (ix1 < cix0 || ix0 > cix1 || iz1 < ciz0 || iz0 > ciz1) continue;
        if (already_dirty[ci]) continue;
        already_dirty[ci] = true;
        terrain_dirty_chunks_.push_back(static_cast<int>(ci));
    }
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
    terrain_height_dirty_ = true;   // re-upload delta texture for raymarch
}

void VulkanEngine::rebuild_dirty_terrain_chunks() {
    if (terrain_dirty_chunks_.empty()) return;
    if (terrain_data_.heights.empty()) return;
    Heightmap hm{};
    hm.dim = terrain_data_.dim;
    hm.cell = terrain_data_.cell;
    hm.origin_x = terrain_data_.origin_x;
    hm.origin_z = terrain_data_.origin_z;
    hm.heights = terrain_data_.heights;
    for (int ci : terrain_dirty_chunks_) {
        if (ci < 0 || static_cast<size_t>(ci) >= terrain_chunks_.chunks.size()) continue;
        rebuild_chunk_vertices(device_, allocator_, graphics_queue_,
                               graphics_queue_family_, hm,
                               terrain_chunks_.chunks[ci]);
    }
    terrain_dirty_chunks_.clear();
}

void VulkanEngine::refresh_terrain_collision() {
    if (!terrain_jolt_dirty_) return;
    if (terrain_data_.heights.empty()) return;
    // Jolt has no incremental update for HeightFieldShape РІР‚вЂќ we rebuild
    // the whole shape. Mirrors the sub-grid trim done at level load:
    // when sample_count would be odd Jolt's block-size requirement
    // forces us onto a 2048Р“вЂ”2048 sub-grid taken from the 2049Р’Р† heightmap.
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
    if (terrain_chunks_.chunks.size() > 0) return;     // already built
    if (terrain_data_.heights.empty()) return;          // no heightmap to build from
    log::info("[terrain] lazy-building rasterised mesh + chunks");
    vkDeviceWaitIdle(device_);
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;
    terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                       graphics_queue_, graphics_queue_family_, hm);
    const int chunks_per_side = 32;
    terrain_chunks_ = build_terrain_chunks(device_, allocator_,
                                           graphics_queue_, graphics_queue_family_,
                                           hm, chunks_per_side);
    // Note: the terrain BLAS is built once in init_rt and not rebuilt
    // here. Toggling raymarch off mid-session draws raster terrain
    // correctly but RT shadow/AO/GI rays from cube.frag will not see
    // it until restart. tlas_includes_terrain_blas() handles the
    // address==0 case already, so this is a quality fallback, not a
    // crash risk.
}

void VulkanEngine::ensure_grass_raster_built() {
    if (grass_.instance_count > 0) return;              // already built
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
    gp.keep_out_xz = glm::vec2(4.0f, 4.0f);
    gp.max_blades  = 800000;
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
    // sculpting" РІР‚вЂќ acceptable.
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
        rebuild_chunk_vertices(device_, allocator_, graphics_queue_,
                               graphics_queue_family_, hm, fake);
    }
    terrain_blas_dirty_ = false;
}

void VulkanEngine::destroy_terrain_shadow_texture() {
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
// flip TRANSFER_DST РІвЂ вЂ™ SHADER_READ_ONLY symmetrically).
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
    ici.format = VK_FORMAT_R8_UNORM;
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
    vci.format = VK_FORMAT_R8_UNORM;
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
    // tile transition (SHADER_READ_ONLY РІвЂ вЂ™ TRANSFER_DST РІвЂ вЂ™ SHADER_READ_ONLY)
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

void VulkanEngine::rebuild_terrain_shadow_texture() {
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
    // user has supersample > 1 the texture is created at SSГ—heightmap-dim,
    // and we replicate each SS=1 texel into an SSГ—SS block so the
    // texture is fully populated from frame 1 вЂ” the worker then
    // re-bakes properly at the user's SS in the background.
    const int ss = std::max(1, rt_.terrain_bake_supersample);
    terrain_shadow_active_ss_ = ss;
    std::vector<uint8_t> base = bake_heightmap_shadow(hm, sun_dir);
    const int W = hm.width()  * ss;
    const int H = hm.height() * ss;
    std::vector<uint8_t> shadow;
    if (ss == 1) {
        shadow = std::move(base);
    } else {
        shadow.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
        for (int z = 0; z < H; ++z) {
            for (int x = 0; x < W; ++x) {
                int sx = x / ss;
                int sz = z / ss;
                shadow[static_cast<size_t>(z) * static_cast<size_t>(W) +
                       static_cast<size_t>(x)] =
                    base[static_cast<size_t>(sz) * static_cast<size_t>(hm.width()) +
                         static_cast<size_t>(sx)];
            }
        }
    }

    const bool first_create = (terrain_shadow_image_ == VK_NULL_HANDLE);
    ensure_shadow_resources(device_, allocator_, graphics_queue_,
                             graphics_queue_family_, W, H,
                             terrain_shadow_image_, terrain_shadow_alloc_,
                             terrain_shadow_view_, terrain_shadow_sampler_,
                             terrain_shadow_dim_);

    // Stage upload via a scratch host-visible buffer.
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(W) * static_cast<VkDeviceSize>(H);
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
    // Heightmap is stored as (dim+1) Г— (dim+1) вЂ” vertices, not cells вЂ”
    // so the texture must match that or the row-stride doesn't line up
    // and the upload scrambles the data (cells get shifted by 1 column
    // per row). Texture is dim+1 in each axis to match.
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;

    // Cache the max for the raymarch's upper-bound clip. Compute once
    // here вЂ” sculpt edits don't update the texture anyway.
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
            // Only erode cells the user actually raised вЂ” preserve
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
    // Mark every chunk dirty so the rasterised mesh rebuilds.
    terrain_dirty_chunks_.clear();
    terrain_dirty_chunks_.reserve(terrain_chunks_.chunks.size());
    for (size_t ci = 0; ci < terrain_chunks_.chunks.size(); ++ci) {
        terrain_dirty_chunks_.push_back(static_cast<int>(ci));
    }
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
// Standard depth-only shadow pass: one 2048Р’Р† D32 depth target written
// from the sun's POV each frame, sampled by grass.vert as a
// sampler2DShadow at descriptor binding 7. Cube.frag continues to use
// RT shadow rays РІР‚вЂќ we don't move the whole engine to CSM yet, this is
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
        r.data.resize(static_cast<size_t>(job.w) * static_cast<size_t>(job.h));
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

    // Tile coordinates are in TEXTURE space (heightmap_dim+1 Г— ss).
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

    // Replace the worker's job queue. Drop any older results too вЂ” they
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
        total_bytes += static_cast<size_t>(r.w) * static_cast<size_t>(r.h);
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
        size_t bytes = static_cast<size_t>(r.w) * static_cast<size_t>(r.h);
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

void VulkanEngine::init_world() {
    cube_mesh_ = create_cube_mesh(device_, allocator_, graphics_queue_, graphics_queue_family_);
    cylinder_mesh_ = create_cylinder_mesh(device_, allocator_,
                                          graphics_queue_, graphics_queue_family_);
    world_ = game::make_arena();
    log::infof("world: %u brushes", static_cast<unsigned>(world_.brushes.size()));
    // Brush model matrices are baked later, AFTER the plateau y-lift
    // below — building them here would cache un-lifted centers and the
    // castle would render 22 m under the terrain (invisible).

    // Procedural terrain: generated first so the castle can be lifted to
    // sit on the plateau height. Phase 1 is non-streaming РІР‚вЂќ one big mesh
    // (~1kmР’Р†). See docs/terrain_plan.md for streaming/LOD/sculpt phases.
    {
        HeightmapParams hp{};
        // Two constraints that fight each other:
        //   1. chunk_cells must be divisible by every LOD stride (1, 2, 4, 8)
        //      so the stride-N index loop covers every cell РІР‚вЂќ otherwise the
        //      lower-LOD index buffers leave a strip of cells uncovered at
        //      the chunk's right/bottom edge РІвЂ вЂ™ visible square seams.
        //   2. Jolt's HeightFieldShape needs sample_count divisible by
        //      block_size in [2, 8].
        //
        // chunk_cells / 8 means dim is divisible by 8 РІвЂ вЂ™ sample_count = dim+1
        // is odd РІвЂ вЂ™ Jolt fails. Resolution: pick dim = 2048 cells (=32Р“вЂ”64),
        // chunk_cells = 64 (divisible by all four strides), and feed Jolt a
        // 2048Р“вЂ”2048 sub-grid by truncating the last row/col of the 2049Р’Р†
        // heightmap. The visual mesh uses the full 2049Р’Р† so the world edge
        // stays continuous. Jolt loses one cell-width at the world edge РІР‚вЂќ
        // invisible since the player can't reach the world boundary.
        // Heightmap resolution scale вЂ” the user can pick higher density
        // via the Terrain settings menu. Defaults to 1Г— = the baseline
        // 2048-cell grid. Higher values keep the world extent the same
        // (cell shrinks proportionally) so castle/grass/spawn positions
        // are unaffected.
        const int hres = std::max(1, std::min(4, rt_.terrain_heightmap_scale));
        hp.dim = 2048 * hres;
        hp.cell_size = 1.0f / static_cast<float>(hres);
        hp.height_scale = 140.0f;
        // Plateau: tight extent (just past castle outer wall) + a
        // generous blend zone so the FBM ramps in smoothly. Without
        // the long blend the procedural mountains (±60 m peaks)
        // popped up immediately past the wall and looked super-spiky
        // — the blend gives the player a "settled-into-the-hills"
        // approach instead.
        hp.plateau_extent = glm::vec2(11.5f, 11.5f);
        hp.plateau_height = 22.0f;
        hp.plateau_blend  = 20.0f;
        hp.frequency      = 0.0014f;
        // ALWAYS use the value-noise FBM (generate_heightmap_raymarch),
        // regardless of whether the game starts in mesh or raymarch
        // mode. Rationale:
        //   * The raymarch shader evaluates this same FBM in-shader and
        //     adds sampleHeightDelta = (current − baseline). The
        //     baseline MUST be this FBM or, after a runtime mesh→
        //     raymarch toggle, the delta becomes a difference of two
        //     unrelated noise fields layered on the shader FBM →
        //     "way too spikey" terrain (the exact reported bug).
        //   * mesh / physics / grass / BLAS now all share ONE surface,
        //     so toggling modes is seamless and the world is
        //     consistent — and the planned near-distance displacement-
        //     LOD work needs the mesh to be this same FBM anyway.
        // generate_heightmap() (FastNoiseLite) is now legacy/unused.
        Heightmap hm = generate_heightmap_raymarch(hp);
        // Capture procedural baseline NOW (before any disk-overlay
        // load). The raymarch shader's heightmap-delta texture is
        // computed as `current - baseline` so sculpt edits + disk
        // overlays show up in raymarch as well as in the rasterised
        // mesh. If we captured baseline after the load, delta would
        // always be 0 for the loaded slice and raymarch would still
        // show the procedural surface.
        terrain_baseline_heights_ = hm.heights;
        // If the user has saved a sculpt-edited heightmap (via the
        // Terrain в†’ Save heightmap button), load it now and overwrite
        // the procedurally-generated one. Validates dim+cell match
        // so a heightmap saved at 2048Г—1m doesn't mis-load on a
        // 4096Г—0.5m level.
        {
            std::ifstream hf("assets/level1_heights.bin", std::ios::binary);
            if (hf.is_open()) {
                int32_t fdim = 0;
                float fcell = 0.0f, fox = 0.0f, foz = 0.0f;
                hf.read(reinterpret_cast<char*>(&fdim), sizeof(fdim));
                hf.read(reinterpret_cast<char*>(&fcell), sizeof(fcell));
                hf.read(reinterpret_cast<char*>(&fox), sizeof(fox));
                hf.read(reinterpret_cast<char*>(&foz), sizeof(foz));
                if (fdim == hm.dim &&
                    std::abs(fcell - hm.cell) < 1e-4f &&
                    std::abs(fox - hm.origin_x) < 1e-2f &&
                    std::abs(foz - hm.origin_z) < 1e-2f) {
                    size_t n = static_cast<size_t>(hm.width()) *
                               static_cast<size_t>(hm.height());
                    hf.read(reinterpret_cast<char*>(hm.heights.data()),
                            n * sizeof(float));
                    log::infof("[terrain] loaded sculpted heightmap "
                               "(%zu cells)", n);
                } else {
                    log::warnf("[terrain] saved heightmap dim/cell mismatch "
                               "вЂ” ignoring (saved %dx %.3fm, want %dx %.3fm)",
                               fdim, fcell, hm.dim, hm.cell);
                }
            }
        }
        // Cache for collision + Phase 4 sculpt access.
        terrain_data_.dim = hm.dim;
        terrain_data_.cell = hm.cell;
        terrain_data_.origin_x = hm.origin_x;
        terrain_data_.origin_z = hm.origin_z;
        terrain_data_.heights = hm.heights;
        // Bake the 32×32 hi-Z max-cell grid from the (post sculpt-
        // overlay) heightmap. Each grid cell holds the MAX terrain
        // height over its 64 m × 64 m footprint + a 15 m safety
        // margin. The margin covers (a) the stride-free exact max so
        // no real peak is under-reported, and (b) moderate runtime
        // sculpt brush raises before a rebake. The marcher only uses
        // this to SKIP empty cells (never to declare sky), so a
        // too-LOW value would clip terrain — hence the generous
        // margin. A too-HIGH value just costs a few extra FBM steps.
        {
            const int W = hm.dim + 1;                  // 2049 (row pitch)
            const int G = 32;                          // grid side
            // Cell boundaries MUST use `dim` (2048), NOT W (2049), so
            // they line up exactly with the shader's cellMaxHeight():
            //   shader c = floor((wp/2048 + 0.5) * 32) = floor(hm_x/64)
            // Using W here gave (gx*2049)/32 ≈ gx*64.03 — an off-by-
            // fraction that made the shader sample a NEIGHBOURING grid
            // cell at distance. If that neighbour's max was lower, the
            // marcher skipped a cell that actually held terrain the
            // ray would hit → distant low terrain showed as SKY.
            const int D = hm.dim;                      // 2048
            // 40 m safety (was 15). The coarse heightmap-derived max
            // under-represents the shader FBM's sub-texel spikes; on
            // rugged distant terrain a 15 m margin let the skip jump
            // over real peaks. 40 m is still far below any open-sky
            // gap so the perf benefit (skipping genuine air) is kept.
            for (int gz = 0; gz < G; ++gz) {
                int z0 = (gz * D) / G;
                int z1 = ((gz + 1) * D) / G;
                for (int gx = 0; gx < G; ++gx) {
                    int x0 = (gx * D) / G;
                    int x1 = ((gx + 1) * D) / G;
                    float mx = -1e9f;
                    for (int z = z0; z < z1; ++z) {
                        const float* row = hm.heights.data() +
                            static_cast<size_t>(z) * static_cast<size_t>(W);
                        for (int x = x0; x < x1; ++x) {
                            if (row[x] > mx) mx = row[x];
                        }
                    }
                    terrain_max_grid_[static_cast<size_t>(gz) * G + gx] =
                        mx + 40.0f;
                }
            }
            terrain_max_grid_ready_ = true;
        }
        // Skip the rasterised terrain mesh + chunks + BLAS source when the
        // procedural raymarch is the active terrain. Saves several hundred
        // MB of VRAM (chunked LOD buffers, parent_y morph buffers, merged
        // BLAS source mesh). Caveat: toggling raymarch OFF at runtime will
        // show no rasterised terrain until the user restarts.
        if (!rt_.terrain_raymarch_enabled) {
            terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                               graphics_queue_, graphics_queue_family_, hm);
            // 32×32 chunks of 64 cells each — 64 is divisible by every LOD
            // stride (1/2/4/8) so all four LODs cover every cell. ≈64 m per
            // chunk side at 1 m cells. Skirt strips on each LOD's index
            // buffer hide LOD-mismatch cracks at chunk seams.
            const int chunks_per_side = 32;
            terrain_chunks_ = build_terrain_chunks(device_, allocator_,
                                                   graphics_queue_, graphics_queue_family_,
                                                   hm, chunks_per_side);
        } else {
            log::info("[terrain] raymarch enabled — deferring mesh/chunks "
                      "build (will lazy-build if raymarch is toggled off)");
        }
        // Lift every level brush by plateau_height so the castle's y=0
        // baseline lands on the plateau surface. Cheaper than rewriting
        // level.cpp РІР‚вЂќ the brush AABBs / world matrices update through the
        // same vector.
        for (auto& b : world_.brushes) b.center.y += hp.plateau_height;
        for (auto& a : world_.aabbs)   { a.min.y += hp.plateau_height;
                                          a.max.y += hp.plateau_height; }
        // render_aabbs are the same shape as aabbs (some SPOM walls are
        // inflated by kShellInflation -- see level.cpp::to_render_aabb)
        // so they need the SAME plateau lift. Skipping this would leave
        // SPOM wall frustum culls at y=0 while the geometry sits at
        // y=plateau_height -> all walls culled out at most camera angles.
        for (auto& a : world_.render_aabbs) { a.min.y += hp.plateau_height;
                                              a.max.y += hp.plateau_height; }
        // Lift player's default spawn point above the new ground.
        player_.position.y += hp.plateau_height;

        // Now that brushes are at their final positions, pre-bake the
        // static model matrices used every frame by the depth pre-pass,
        // color pass and sun-shadow pass. Static brushes don't move
        // after this; rebuilding (translate*scale) ~190 × 3 passes per
        // frame was ~25 µs/frame and a perfect cache target. (Perf P3.)
        static_brush_models_.resize(world_.brushes.size());
        for (size_t i = 0; i < world_.brushes.size(); ++i) {
            const auto& b = world_.brushes[i];
            static_brush_models_[i] =
                glm::translate(glm::mat4(1.0f), b.center) *
                glm::scale    (glm::mat4(1.0f), b.size);
        }

        // Grass: scatter blades on the heightmap. Acceptance band +
        // slope test mean only the grass-coloured layer of the
        // terrain (low altitude, gentle slope) gets covered. Cap at
        // 200k blades for a healthy density without bloating VRAM.
        GrassParams gp{};
        // Cover the plateau too РІР‚вЂќ slopes, valleys, and the plateau
        // courtyard around the castle. Castle stones are excluded by
        // the keep-out rectangle below so blades never poke through
        // the brushes themselves.
        // Placement bounds are GENEROUS РІР‚вЂќ the runtime altitude
        // sliders (rt_.grass_alt_min/max) shape the visible band
        // without re-baking. Cover most reasonable terrain heights.
        gp.height_min = -20.0f;
        gp.height_max = 200.0f;
        // Concentrate blades in a 200m square around the castle so the
        // density is high (~12 blades/mР’Р†) where the player actually
        // sees grass. Spreading them across the full 1km map gave
        // ~0.5/mР’Р† (about a metre between blades РІР‚вЂќ looks bald).
        gp.half_extent = 200.0f;
        // Inner keep is ~6m square at origin. Keep blades out of the
        // keep interior only; the outer courtyard between keep and
        // perimeter walls gets grass too. Blades whose footprint
        // overlaps a brush stay invisible РІР‚вЂќ the wall geometry occludes
        // them from outside views.
        gp.keep_out_xz = glm::vec2(4.0f, 4.0f);
        // 800k placed blades: density slider 0..4 maps to render
        // fraction (density / 4) Р“вЂ” placed, so density=1.0 РІвЂ вЂ™ 200k,
        // density=4.0 РІвЂ вЂ™ 800k. Earlier 2M cap meant density=4 fired
        // ~10M sun shadow rays/frame (5 verts each) and pushed the
        // GPU into TDR. 800k keeps the worst case at ~4M rays which
        // the 4080 handles comfortably even with the rest of the
        // RT load (terrain shadows, AO, GI).
        gp.max_blades  = 800000;
        // Skip 800k-blade placement when raymarch grass will be drawn —
        // the rasterised grass buffer would never get sampled. Toggling
        // raymarch grass OFF at runtime requires a restart.
        if (!rt_.grass_raymarch_enabled) {
            grass_ = build_grass(device_, allocator_,
                                 graphics_queue_, graphics_queue_family_,
                                 hm, gp);
        } else {
            log::info("[grass] raymarch grass enabled — deferring placement "
                      "build (will lazy-build if raymarch grass is toggled off)");
        }
    }

    physics_ = std::make_unique<PhysicsWorld>();
    // Batch-add the level brushes РІР‚вЂќ Jolt's AddBodiesPrepare/Finalize takes
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
    // against the bumpy ground. Jolt requires the sample-count side to be
    // divisible by some block_size in [2, 8]; with dim=2048 (sample_count
    // = 2049, odd) we'd violate that, so we copy the first 2048Р“вЂ”2048 sub-
    // grid into a packed buffer and pass that with dim=2047 (sample_count
    // = 2048, block_size 8 РІСљвЂњ). Player can't reach the world edge so the
    // single-cell shrinkage is invisible.
    if (!terrain_data_.heights.empty()) {
        const int W = terrain_data_.dim + 1;        // 2049
        const int physics_samples = (W % 2 == 0) ? W : (W - 1);  // 2048
        // Build the physics-only height buffer (a copy — visual raymarch
        // keeps terrain_data_.heights as-is). Inside the castle plateau
        // we depress the floor by kCastleSink so it sits well below the
        // 5 cm castle-floor brush at Y=22.05. Without this, any FBM
        // residue or loaded sculpt deltas inside the plateau become
        // physics-only "invisible hills" (visible terrain is discarded
        // inside the castle, but Jolt still collides against them) and
        // bounce the player up and down on what looks like flat tile.
        // 0.6 m drop > kStepDown (0.45) so the heightmap-clamp branches
        // never fire when the player is on the brush floor.
        // Hard-coded to match the hp.* values used at heightmap-gen
        // time (line ~1363) — hp itself is out of scope here.
        const float kCastleSink   = 0.6f;
        const float kPlateauH     = 22.0f;
        const float kSinkRadius   = 11.5f;
        const float kSinkLimit    = kPlateauH - kCastleSink;
        const glm::vec2 plateau_c(0.0f, 0.0f);
        std::vector<float> jolt_heights(static_cast<size_t>(physics_samples) *
                                         static_cast<size_t>(physics_samples));
        for (int z = 0; z < physics_samples; ++z) {
            float wz = terrain_data_.origin_z +
                       static_cast<float>(z) * terrain_data_.cell;
            for (int x = 0; x < physics_samples; ++x) {
                float wx = terrain_data_.origin_x +
                           static_cast<float>(x) * terrain_data_.cell;
                float h = terrain_data_.heights[
                    static_cast<size_t>(z) * static_cast<size_t>(W) +
                    static_cast<size_t>(x)];
                if (std::abs(wx - plateau_c.x) < kSinkRadius &&
                    std::abs(wz - plateau_c.y) < kSinkRadius) {
                    h = std::min(h, kSinkLimit);
                }
                jolt_heights[static_cast<size_t>(z) *
                              static_cast<size_t>(physics_samples) +
                              static_cast<size_t>(x)] = h;
            }
        }
        physics_->add_static_heightfield(
            jolt_heights.data(), physics_samples - 1,
            glm::vec2(terrain_data_.origin_x, terrain_data_.origin_z),
            terrain_data_.cell);
    }

    log::infof("Jolt: %d static bodies; dynamic boxes will spawn over time "
               "(max %d, every %.2fs)",
               physics_->body_count(),
               kMaxDynProps, kSpawnInterval);
}

void VulkanEngine::init_skybox() {
    // EXR first РІР‚вЂќ keeps real sun radiance so the bloom pass naturally produces
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
        log::info("[skybox] no asset loaded РІР‚вЂќ creating 1x1 placeholder so the "
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
        // Real asset loaded РІР‚вЂќ sync the directional-light angles to the
        // brightest pixel in the panorama so cast shadows match the visible
        // sun.
        glm::vec3 d = glm::normalize(skybox_.sun_direction);
        float pitch_deg = glm::degrees(std::asin(glm::clamp(d.y, -1.0f, 1.0f)));
        float yaw_deg = glm::degrees(std::atan2(d.x, d.z));
        rt_.sun_pitch_deg = pitch_deg;
        rt_.sun_yaw_deg = yaw_deg;
        log::infof("[skybox] sun synced to UI: pitch=%.1fР’В° yaw=%.1fР’В°",
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
        // texture.cpp generates РІР‚вЂќ eliminates far-distance moirР“В© and halves
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
    // Index РІвЂ вЂ™ category. Indices ALSO match what level.cpp / spawn_random_box
    // pass into Brush::tex_albedo, so reordering here means reordering there.
    const Spec specs[] = {
        { "Ground054",         "Ground054/Ground054_2K-JPG_Color.jpg",
                               "Ground054/Ground054_2K-JPG_NormalGL.jpg" },
        { "Bricks078",         "Bricks078/Bricks078_8K-JPG_Color.jpg",
                               "Bricks078/Bricks078_8K-JPG_NormalGL.jpg" },
        { "Wood048",           "Wood048/Wood048_2K-JPG_Color.jpg",
                               "Wood048/Wood048_2K-JPG_NormalGL.jpg" },
        { "Metal042A",         "Metal042A/Metal042A_2K-JPG_Color.jpg",
                               "Metal042A/Metal042A_2K-JPG_NormalGL.jpg" },
        { "PaintedBricks001",  "PaintedBricks001/PaintedBricks001_8K-JPG_Color.jpg",
                               "PaintedBricks001/PaintedBricks001_8K-JPG_NormalGL.jpg" },
        { "Tiles130",          "Tiles130/Tiles130_8K-JPG_Color.jpg",
                               "Tiles130/Tiles130_8K-JPG_NormalGL.jpg" },
        { "Tiles074",          "Tiles074/Tiles074_8K-JPG_Color.jpg",
                               "Tiles074/Tiles074_8K-JPG_NormalGL.jpg" },
    };

    static_assert(sizeof(specs) / sizeof(specs[0]) == kFileTextureCount,
                  "specs[] must list exactly the file-loaded textures");
    for (int i = 0; i < kFileTextureCount; ++i) {
        // Albedos go through an sRGB sampler so the GPU does the gamma decode
        // automatically; normals stay UNORM (linear-space, already in [0,1]).
        albedo_textures_[i] = probe(specs[i].color_jpg,  VK_FORMAT_R8G8B8A8_SRGB);
        normal_textures_[i] = probe(specs[i].normal_jpg, VK_FORMAT_R8G8B8A8_UNORM);
    }
    // Slots 7-11: procedurally-baked seamless terrain materials.
    bake_terrain_materials();

    // SPOM displacement maps — bound to cube.frag as sampler2D u_height[5].
    // Order MUST match cube.frag's height_idx_for_albedo() switch:
    //   [0] Bricks078         (albedo idx 1) — castle outer walls
    //   [1] PaintedBricks001  (albedo idx 4) — keep walls
    //   [2] Tiles130          (albedo idx 5) — courtyard floor
    //   [3] Tiles074          (albedo idx 6) — keep interior floor
    //   [4] stylized-grass1 (albedo idx kTexGrass / 8) -- terrain grass
    //                       band material (ground under blades); also
    //                       sampled by .tese for vertex displacement
    //                       on the grass band.
    spom_height_textures_[0] = probe(
        "Bricks078/Bricks078_8K-JPG_Displacement.jpg",                 VK_FORMAT_R8G8B8A8_UNORM);
    spom_height_textures_[1] = probe(
        "PaintedBricks001/PaintedBricks001_8K-JPG_Displacement.jpg",   VK_FORMAT_R8G8B8A8_UNORM);
    spom_height_textures_[2] = probe(
        "Tiles130/Tiles130_8K-JPG_Displacement.jpg",                   VK_FORMAT_R8G8B8A8_UNORM);
    spom_height_textures_[3] = probe(
        "Tiles074/Tiles074_8K-JPG_Displacement.jpg",                   VK_FORMAT_R8G8B8A8_UNORM);

    // Stylized-grass texture set (assets/stylized_grass/...). Used as the
    // ground beneath where grass blades grow -- replaces the procedurally
    // baked grass at slot kTexGrass (8). Feeds:
    //   - cube.frag's is_terrain splat path (per-texel grass detail under
    //     the blades) via albedo_textures_[8] + normal_textures_[8].
    //   - cube.frag's per-pixel SPOM displacement in the is_terrain block
    //     and terrain_tess.tese's vertex displacement, both via
    //     spom_height_textures_[4] (height_idx_for_albedo returns 4 for
    //     albedo idx 8).
    // The albedo is multiplied by the AO map at load time to bake in the
    // crevice darkening before sampling (no extra texture binding, no
    // per-frame ALU). Paths probed under multiple roots so the engine
    // runs from build/ (Windows) or the project root (CI) alike.
    auto resolve_grass_path = [&](const char* tail) -> std::string {
        const std::string roots[] = {
            std::string("assets/stylized_grass/stylized-grass1-bl/") + tail,
            std::string("../assets/stylized_grass/stylized-grass1-bl/") + tail,
            std::string("../../assets/stylized_grass/stylized-grass1-bl/") + tail,
        };
        for (const std::string& p : roots) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    auto probe_grass = [&](const char* tail, VkFormat fmt) -> TextureSlot {
        std::string p = resolve_grass_path(tail);
        if (!p.empty()) {
            Texture2D r = upload_texture_from_file(
                device_, allocator_, graphics_queue_,
                graphics_queue_family_, p, fmt);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                return s;
            }
        }
        log::warnf("[texture] stylized-grass %s missing on every probe path",
                   tail);
        return {};
    };
    // Build the AO-multiplied albedo CPU-side. stbi_load is brought in by
    // texture.cpp's TU; the prototypes are re-declared at file scope above
    // to avoid pulling stb_image.h into world.cpp.
    TextureSlot sg_alb{};
    {
        std::string alb_path = resolve_grass_path("stylized-grass1_albedo.png");
        std::string ao_path  = resolve_grass_path("stylized-grass1_ao.png");
        if (!alb_path.empty()) {
            int aw = 0, ah = 0, ac = 0;
            unsigned char* alb_px = stbi_load(alb_path.c_str(),
                                              &aw, &ah, &ac, 4);
            if (alb_px) {
                int ow = 0, oh = 0, oc = 0;
                unsigned char* ao_px = ao_path.empty()
                    ? nullptr
                    : stbi_load(ao_path.c_str(), &ow, &oh, &oc, 4);
                // Multiply RGB by AO.r. If the AO map size differs from
                // the albedo (it shouldn't with this asset, but be safe)
                // use nearest sampling rather than skipping the AO bake.
                if (ao_px) {
                    for (int y = 0; y < ah; ++y) {
                        int ay = (oh > 0) ? (y * oh / ah) : 0;
                        for (int x = 0; x < aw; ++x) {
                            int ax = (ow > 0) ? (x * ow / aw) : 0;
                            int ai = (y  * aw + x ) * 4;
                            int oi = (ay * ow + ax) * 4;
                            unsigned int ao = ao_px[oi]; // R channel
                            // gamma-aware-ish: AO maps are usually linear,
                            // so multiply straight on the sRGB-encoded
                            // albedo. Slightly stronger than linear-space
                            // would be, which matches the artistic intent
                            // (deepen the crevices).
                            for (int c = 0; c < 3; ++c) {
                                unsigned int v = alb_px[ai + c];
                                alb_px[ai + c] =
                                    static_cast<unsigned char>((v * ao) / 255U);
                            }
                        }
                    }
                    stbi_image_free(ao_px);
                }
                Texture2D r = upload_texture_from_pixels(
                    device_, allocator_, graphics_queue_,
                    graphics_queue_family_, alb_px, aw, ah,
                    VK_FORMAT_R8G8B8A8_SRGB, "stylized-grass1_albedo*AO");
                stbi_image_free(alb_px);
                if (r.ok) {
                    sg_alb.image = r.image;
                    sg_alb.alloc = r.alloc;
                    sg_alb.view  = r.view;
                }
            } else {
                log::warnf("[texture] stylized-grass albedo decode failed");
            }
        } else {
            log::warnf("[texture] stylized-grass albedo missing on every probe path");
        }
    }
    TextureSlot sg_nrm = probe_grass(
        "stylized-grass1_normal-ogl.png", VK_FORMAT_R8G8B8A8_UNORM);
    TextureSlot sg_h   = probe_grass(
        "stylized-grass1_height.png",     VK_FORMAT_R8G8B8A8_UNORM);
    // Swap the procedural-grass slot for the stylized-grass albedo/normal.
    // The procedural bake just ran above, so destroy its slot-8 image to
    // avoid leaking before overwriting. If the disk asset failed to load
    // (probe miss) we keep the procedural fallback intact.
    auto kill_slot = [&](TextureSlot& s) {
        if (s.view)  vkDestroyImageView(device_, s.view, nullptr);
        if (s.image) vmaDestroyImage(allocator_, s.image, s.alloc);
        s = {};
    };
    if (sg_alb.image) {
        kill_slot(albedo_textures_[kTexGrass]);
        albedo_textures_[kTexGrass] = sg_alb;
    }
    if (sg_nrm.image) {
        kill_slot(normal_textures_[kTexGrass]);
        normal_textures_[kTexGrass] = sg_nrm;
    }
    // SPOM slot 4 is the stylized-grass height. Always populated; cube.frag's
    // height_idx_for_albedo() returns 4 for albedo idx kTexGrass, and the
    // .tese reads it through u_height[4] for vertex displacement.
    spom_height_textures_[4] = sg_h;

    // Restore the rocky-rugged-terrain PNG on the ROCK slot (kTexRock).
    // The user originally asked for this asset; the earlier integration
    // put it on the grass slot and the stylized-grass swap replaced it.
    // Rocky terrain (cliffs / steep slopes) is the natural home for a
    // realistic rock texture, so we overwrite the procedural rock slot
    // with the PNGs. Probes follow the same multi-root pattern as
    // probe_grass so the engine still finds them from build/ or root.
    auto resolve_rock_path = [&](const char* tail) -> std::string {
        const std::string roots[] = {
            std::string("assets/rocky_terrain/rocky-rugged-terrain-bl/") + tail,
            std::string("../assets/rocky_terrain/rocky-rugged-terrain-bl/") + tail,
            std::string("../../assets/rocky_terrain/rocky-rugged-terrain-bl/") + tail,
        };
        for (const std::string& p : roots) {
            std::ifstream f(p);
            if (f.good()) return p;
        }
        return std::string();
    };
    auto probe_rock = [&](const char* tail, VkFormat fmt) -> TextureSlot {
        std::string p = resolve_rock_path(tail);
        if (!p.empty()) {
            Texture2D r = upload_texture_from_file(
                device_, allocator_, graphics_queue_,
                graphics_queue_family_, p, fmt);
            if (r.ok) {
                TextureSlot s{};
                s.image = r.image; s.alloc = r.alloc; s.view = r.view;
                return s;
            }
        }
        log::warnf("[texture] rocky-rugged %s missing on every probe path",
                   tail);
        return {};
    };
    TextureSlot rr_alb = probe_rock(
        "rocky-rugged-terrain_1_albedo.png",    VK_FORMAT_R8G8B8A8_SRGB);
    TextureSlot rr_nrm = probe_rock(
        "rocky-rugged-terrain_1_normal-ogl.png", VK_FORMAT_R8G8B8A8_UNORM);
    TextureSlot rr_h   = probe_rock(
        "rocky-rugged-terrain_1_height.png",    VK_FORMAT_R8G8B8A8_UNORM);
    // SPOM slot 5 = rocky-rugged height. Lets the tess displacement +
    // per-pixel SPOM bump where the visible STONES are (not where the
    // grass-height noise wanted them).
    if (rr_h.image) {
        kill_slot(spom_height_textures_[5]);
        spom_height_textures_[5] = rr_h;
    }
    // Rocky-rugged goes into the GRASS-BAND NON-MASK FALLBACK slot
    // (kTexProcGrass / slot 12). cube.frag's grass splat lerps:
    //   grass_mask = 1  -> kTexGrass slot 8  (stylized green grass)
    //   grass_mask = 0  -> kTexProcGrass slot 12 (now rocky-rugged)
    // So on the green terrain band, areas without grass blades read
    // as rocky ground -- which is what the user wants visible.
    // Cliffs / steep slopes keep the procedural rock at kTexRock.
    if (rr_alb.image) {
        kill_slot(albedo_textures_[kTexProcGrass]);
        albedo_textures_[kTexProcGrass] = rr_alb;
    }
    if (rr_nrm.image) {
        kill_slot(normal_textures_[kTexProcGrass]);
        normal_textures_[kTexProcGrass] = rr_nrm;
    }
}

// ----------------------------------------------------------------------
// Procedural terrain materials.
//
// No ground textures ship on disk (only Ground054), so the 5 splat
// materials are generated at load: seamless tiling albedo (sRGB) + a
// matching tangent-space normal map (UNORM, GL/+Y convention) for each
// of rock / grass / dirt / sand / snow. cube.frag's is_terrain block
// blends them by the existing height/slope weights.
//
// Seamlessness is by construction: every noise octave uses a lattice
// whose period equals the number of cells across the [0,1) tile, so the
// value at u=1 hashes to the same lattice point as u=0. The normal map
// is the central-difference gradient of the same periodic height field,
// so it tiles too — no visible seams at any tiling rate.
// ----------------------------------------------------------------------
namespace {

inline uint32_t tm_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
// Hash of a lattice node, wrapped to `period` so the tile is seamless.
inline float tm_lat(int xi, int yi, int period, uint32_t seed) {
    xi = ((xi % period) + period) % period;
    yi = ((yi % period) + period) % period;
    uint32_t h = tm_hash(static_cast<uint32_t>(xi) * 73856093U ^
                          static_cast<uint32_t>(yi) * 19349663U ^ seed);
    return static_cast<float>(h & 0xFFFFFFu) / float(0x1000000);
}
// Periodic value noise. `cells` lattice cells span the whole tile, and
// the lattice period equals `cells`, so f(u=0)==f(u=1) → seamless.
inline float tm_vnoise(float u, float v, int cells, uint32_t seed) {
    float x = u * cells, y = v * cells;
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    float fx = x - x0, fy = y - y0;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    float a = tm_lat(x0,     y0,     cells, seed);
    float b = tm_lat(x0 + 1, y0,     cells, seed);
    float c = tm_lat(x0,     y0 + 1, cells, seed);
    float d = tm_lat(x0 + 1, y0 + 1, cells, seed);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}
// Tileable fBm: each octave's period == its cell count, so the sum
// stays seamless.
inline float tm_fbm(float u, float v, int base_cells, int octaves,
                     uint32_t seed) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    int cells = base_cells;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * tm_vnoise(u, v, cells, seed + o * 1013u);
        norm += amp;
        amp  *= 0.5f;
        cells *= 2;
    }
    return sum / norm;                       // ~0..1
}
inline float tm_clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
// Ridged multifractal — the standard "eroded mountain" basis. Each
// octave is a sharp ridge (1-|2n-1|), and its amplitude is WEIGHTED by
// the previous octave's value, so detail accumulates on exposed ridges
// and valleys stay smooth (mirrors how real erosion distributes relief
// — crisp crests, clean gullies). `gain` sharpens, `offset` ~1.
inline float tm_ridged_mf(float u, float v, int base_cells, int octaves,
                          uint32_t seed) {
    float sum = 0.0f, freqAmp = 0.5f, norm = 0.0f, prev = 1.0f;
    int   cells = base_cells;
    for (int o = 0; o < octaves; ++o) {
        float n = tm_vnoise(u, v, cells, seed + o * 2237u);
        float r = 1.0f - std::fabs(2.0f * n - 1.0f);   // ridge
        r = r * r;                                      // sharpen crest
        float contrib = r * freqAmp * tm_clamp01(prev); // weight by prev
        sum  += contrib;
        norm += freqAmp;
        prev  = r;
        freqAmp *= 0.55f;
        cells   *= 2;
    }
    return sum / norm;                       // ~0..1, ridged
}
inline float tm_smooth(float a, float b, float x) {
    float t = tm_clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}
// Linear → sRGB 8-bit. Authored colours are linear (matching the old
// in-shader palette); the array sampler is sRGB and decodes on read, so
// we must store the sRGB-encoded byte here.
inline unsigned char tm_lin2srgb8(float c) {
    c = tm_clamp01(c);
    float s = (c <= 0.0031308f) ? (c * 12.92f)
                                : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
    int b = static_cast<int>(s * 255.0f + 0.5f);
    return static_cast<unsigned char>(b < 0 ? 0 : (b > 255 ? 255 : b));
}
inline unsigned char tm_unorm8(float c) {
    int b = static_cast<int>(tm_clamp01(c) * 255.0f + 0.5f);
    return static_cast<unsigned char>(b);
}

enum TmKind { TM_ROCK, TM_GRASS, TM_DIRT, TM_SAND, TM_SNOW };

// Fill albedo (sRGB bytes) + normal (UNORM bytes, GL +Y) for one
// material. Builds a periodic height field first, then derives both the
// colour ramp and the normal from it so relief and shading agree.
void tm_generate(TmKind kind, int W, int H,
                 std::vector<unsigned char>& albedo,
                 std::vector<unsigned char>& normal) {
    albedo.resize(static_cast<size_t>(W) * H * 4);
    normal.resize(static_cast<size_t>(W) * H * 4);
    std::vector<float> hf(static_cast<size_t>(W) * H);  // low-freq: drives albedo
    std::vector<float> rf(static_cast<size_t>(W) * H);  // fine relief: drives normal

    // Per-material recipe. n_strength is the bump depth of the detail
    // normal map. The normal is the gradient of a HIGH-frequency relief
    // field (rf) so adjacent texels actually differ — the old code took
    // the gradient of the low-freq albedo field, which barely changed
    // texel-to-texel, so every normal map came out flat (≈(0,0,1)) and
    // the strength slider did nothing.
    int   base_cells, octaves;
    float n_strength;            // 0..1-ish bump depth
    uint32_t seed;
    switch (kind) {
        case TM_ROCK:  base_cells = 6;  octaves = 9; n_strength = 1.45f; seed = 11u; break;
        case TM_GRASS: base_cells = 4;  octaves = 6; n_strength = 0.45f; seed = 23u; break;
        case TM_DIRT:  base_cells = 5;  octaves = 6; n_strength = 0.65f; seed = 37u; break;
        case TM_SAND:  base_cells = 3;  octaves = 5; n_strength = 0.55f; seed = 53u; break;
        default:       base_cells = 3;  octaves = 5; n_strength = 0.30f; seed = 71u; break; // SNOW
    }

    // Height field (periodic).
    for (int y = 0; y < H; ++y) {
        float v = static_cast<float>(y) / H;
        for (int x = 0; x < W; ++x) {
            float u = static_cast<float>(x) / W;
            float f = tm_fbm(u, v, base_cells, octaves, seed);
            if (kind == TM_ROCK) {
                // Eroded-mountain rock: ridged multifractal (crisp
                // crests, smooth valleys) + a sharper high-frequency
                // drainage/gully layer carved in for fine cracks.
                float mf    = tm_ridged_mf(u, v, base_cells, octaves, seed);
                float gully = tm_ridged_mf(u, v, base_cells * 4, 4,
                                           seed + 901u);
                gully = gully * gully;                  // tighten ravines
                f = tm_clamp01(mf - 0.35f * gully);
            } else if (kind == TM_SAND) {
                // Isotropic — no baked directional content (that caused
                // either fixed world stripes or, when rotated per-pixel
                // in cube.frag, per-triangle UV seams). The shore-
                // following ripple is added smoothly in cube.frag from
                // the height field instead. Soft dunes + fine grain.
                float dune  = tm_fbm(u, v, 6,  4, seed + 7u);
                float grain = tm_fbm(u, v, 48, 2, seed + 13u);
                f = 0.70f * dune + 0.30f * grain;
            } else if (kind == TM_SNOW) {
                // Smooth dunes + sparse sparkle specks.
                float spark = tm_vnoise(u, v, 256, seed + 9u);
                f = 0.85f * f + (spark > 0.93f ? 0.15f : 0.0f);
            }
            hf[static_cast<size_t>(y) * W + x] = f;

            // Fine relief for the NORMAL map: a high-frequency field
            // whose neighbouring texels differ enough to give a real
            // gradient. Mix in the low-freq f so big shapes still cast
            // the right macro tilt.
            float fine_r = tm_fbm(u, v, base_cells * 6, octaves, seed + 101u);
            float micro  = tm_vnoise(u, v, 256, seed + 211u);
            float relief = 0.45f * f + 0.40f * fine_r + 0.15f * micro;
            if (kind == TM_ROCK) {
                // Crisp chiselled relief: the eroded height (f, already
                // ridged-MF) plus a fine high-frequency ridged-MF layer
                // → sharp ridge/crack normal at close range.
                float fine_mf = tm_ridged_mf(u, v, base_cells * 5, 5,
                                             seed + 701u);
                relief = 0.40f * f + 0.60f * fine_mf;
            } else if (kind == TM_SAND) {
                // Isotropic relief (matches the isotropic albedo). The
                // directional ripple is a smooth cube.frag overlay.
                float dn = tm_fbm(u, v, 5,  4, seed + 211u);
                float gr = tm_fbm(u, v, 40, 3, seed + 311u);
                relief = 0.6f * dn + 0.4f * gr;
            }
            rf[static_cast<size_t>(y) * W + x] = relief;
        }
    }

    auto Hsamp = [&](int x, int y) -> float {
        x = (x % W + W) % W; y = (y % H + H) % H;
        return rf[static_cast<size_t>(y) * W + x];
    };

    for (int y = 0; y < H; ++y) {
        float v = static_cast<float>(y) / H;
        for (int x = 0; x < W; ++x) {
            float u = static_cast<float>(x) / W;
            float h = hf[static_cast<size_t>(y) * W + x];
            // Fine break-up so flat patches don't read as one colour.
            float fine = tm_vnoise(u, v, 256, seed + 3u) - 0.5f;

            float rl, gl, bl;       // linear albedo
            switch (kind) {
                case TM_ROCK: {
                    float t = tm_smooth(0.25f, 0.85f, h);
                    rl = 0.085f + (0.215f - 0.085f) * t;
                    gl = 0.080f + (0.205f - 0.080f) * t;
                    bl = 0.072f + (0.190f - 0.072f) * t;
                    float s = fine * 0.045f; rl += s; gl += s; bl += s;
                } break;
                case TM_GRASS: {
                    float t = tm_smooth(0.30f, 0.72f, h);
                    // Brighter, lower-contrast green so the cross-fade
                    // with neighbouring layers reads as a smooth tonal
                    // gradient, not dark speckle over green.
                    rl = 0.085f + (0.150f - 0.085f) * t;
                    gl = 0.150f + (0.240f - 0.150f) * t;
                    bl = 0.045f + (0.075f - 0.045f) * t;
                    float clump = (tm_vnoise(u, v, 96, seed + 5u) - 0.5f) * 0.020f;
                    rl += clump * 0.6f; gl += clump; bl += clump * 0.3f;
                } break;
                case TM_DIRT: {
                    float t = tm_smooth(0.25f, 0.80f, h);
                    rl = 0.075f + (0.185f - 0.075f) * t;
                    gl = 0.048f + (0.115f - 0.048f) * t;
                    bl = 0.025f + (0.060f - 0.025f) * t;
                    // gentle embedded pebbles (no harsh dark stamps)
                    float peb = tm_vnoise(u, v, 64, seed + 6u);
                    if (peb > 0.82f) { rl *= 0.85f; gl *= 0.85f; bl *= 0.85f; }
                    float s = fine * 0.025f; rl += s; gl += s; bl += s;
                } break;
                case TM_SAND: {
                    float t = tm_smooth(0.30f, 0.75f, h);
                    rl = 0.300f + 0.070f * t;
                    gl = 0.235f + 0.060f * t;
                    bl = 0.120f + 0.040f * t;
                    float s = fine * 0.030f; rl += s; gl += s; bl += s;
                } break;
                default: { // SNOW
                    float t = tm_smooth(0.35f, 0.80f, h);
                    rl = 0.760f + 0.180f * t;
                    gl = 0.790f + 0.170f * t;
                    bl = 0.880f + 0.110f * t;     // faint cool tint
                    float s = fine * 0.020f; rl += s; gl += s; bl += s;
                } break;
            }

            size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            albedo[idx + 0] = tm_lin2srgb8(rl);
            albedo[idx + 1] = tm_lin2srgb8(gl);
            albedo[idx + 2] = tm_lin2srgb8(bl);
            albedo[idx + 3] = 255;

            // Tangent-space normal from the FINE relief field's central
            // difference. GL convention: G = +Y; a bump (higher centre)
            // tilts the normal toward -gradient. The RELIEF_GAIN turns
            // the small texel-to-texel relief delta into a visible
            // surface slope so the detail-normal strength slider has a
            // real, wide effect.
            const float RELIEF_GAIN = 14.0f;
            float g = n_strength * RELIEF_GAIN;
            float dhx = (Hsamp(x + 1, y) - Hsamp(x - 1, y)) * g;
            float dhy = (Hsamp(x, y + 1) - Hsamp(x, y - 1)) * g;
            dhx = dhx < -4.0f ? -4.0f : (dhx > 4.0f ? 4.0f : dhx);
            dhy = dhy < -4.0f ? -4.0f : (dhy > 4.0f ? 4.0f : dhy);
            float nx = -dhx, ny = -dhy, nz = 1.0f;
            float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            normal[idx + 0] = tm_unorm8(nx * inv * 0.5f + 0.5f);
            normal[idx + 1] = tm_unorm8(ny * inv * 0.5f + 0.5f);
            normal[idx + 2] = tm_unorm8(nz * inv * 0.5f + 0.5f);
            // Alpha = the relief HEIGHT field itself (0..1). cube.frag's
            // terrain parallax-occlusion march samples this — free, no
            // extra texture/binding. Rock gets an extra eroded-gully
            // term so the POM carves cracked, weathered rock.
            float reliefH = rf[static_cast<size_t>(y) * W + x];
            if (kind == TM_ROCK) {
                // POM height = eroded ridged-MF + deep drainage gullies
                // so parallax carves crisp ravines into the rock.
                float ru = static_cast<float>(x) / W;
                float rv = static_cast<float>(y) / H;
                float mf  = tm_ridged_mf(ru, rv, base_cells * 2, 6,
                                         seed + 401u);
                float gul = tm_ridged_mf(ru, rv, base_cells * 5, 4,
                                         seed + 411u);
                reliefH = tm_clamp01(0.60f * mf + 0.40f * gul * gul);
            }
            normal[idx + 3] = tm_unorm8(reliefH);
        }
    }
}

} // namespace

void VulkanEngine::bake_terrain_materials() {
    const int W = 512, H = 512;
    struct Mat { TmKind kind; int slot; const char* name; };
    const Mat mats[kTerrainMatCount] = {
        { TM_ROCK,  kTexRock,      "terrain/rock"  },
        { TM_GRASS, kTexGrass,     "terrain/grass" },
        { TM_DIRT,  kTexDirt,      "terrain/dirt"  },
        { TM_SAND,  kTexSand,      "terrain/sand"  },
        { TM_SNOW,  kTexSnow,      "terrain/snow"  },
        // Second copy of the procedural grass into slot 12. The
        // probe_grass step will overwrite slot 8 with the stylized
        // PNG; slot 12 keeps the procedural grass so cube.frag can
        // lerp grass-mask=0 -> procedural, grass-mask=1 -> stylized.
        { TM_GRASS, kTexProcGrass, "terrain/proc_grass" },
    };
    std::vector<unsigned char> alb, nrm;
    for (const Mat& m : mats) {
        tm_generate(m.kind, W, H, alb, nrm);
        Texture2D a = upload_texture_from_pixels(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            alb.data(), W, H, VK_FORMAT_R8G8B8A8_SRGB, m.name);
        Texture2D n = upload_texture_from_pixels(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            nrm.data(), W, H, VK_FORMAT_R8G8B8A8_UNORM, m.name);
        albedo_textures_[m.slot] = { a.image, a.alloc, a.view };
        normal_textures_[m.slot] = { n.image, n.alloc, n.view };
    }
    log::infof("[texture] baked %d procedural terrain materials (%dx%d)",
               kTerrainMatCount, W, H);
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
    for (int i = 0; i < kSpomMaterialCount; ++i) {
        kill(spom_height_textures_[i]);
    }
    if (texture_sampler_) {
        vkDestroySampler(device_, texture_sampler_, nullptr);
        texture_sampler_ = VK_NULL_HANDLE;
    }
}

// FSR2 mip-bias (AMD recommendation): when input is rendered below the
// display resolution, the sampler should bias towards sharper mips so
// the upscaler's frequency content matches the display. Formula from
// FSR2 docs:  log2(render_w / display_w) - 1.
//
// Called from init() between apply_render_scale() and the scene
// descriptor write so the new sampler is captured by the descriptor.
// Live toggling rt_.fsr2_enabled mid-game does NOT update the bias
// (would require waiting on the device + rewriting every descriptor
// set that captures the sampler — a follow-up).
void VulkanEngine::update_texture_sampler_for_fsr2() {
    if (!device_) return;
    float bias = 0.0f;
    if (rt_.fsr2_enabled && swapchain_extent_.width > 0 &&
        render_extent_.width > 0 &&
        swapchain_extent_.width != render_extent_.width) {
        float ratio = static_cast<float>(render_extent_.width) /
                      static_cast<float>(swapchain_extent_.width);
        bias = std::log2(ratio) - 1.0f;
    }
    if (texture_sampler_) {
        vkDestroySampler(device_, texture_sampler_, nullptr);
        texture_sampler_ = VK_NULL_HANDLE;
    }
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // Anisotropic filtering — keeps the terrain ground crisp at grazing
    // angles (the dominant "blurry everywhere" cause for a big flat
    // surface). Clamp to the device's supported max.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    // 4x, not 16x: terrain is triplanar (3 albedo + 3 normal taps per
    // fragment), so anisotropy cost is multiplied 6x over a screen-
    // filling ground. 4x removes essentially all the grazing-angle blur
    // while keeping the per-frame cost sane (16x cratered it to ~20fps).
    float aniso = std::min(4.0f, props.limits.maxSamplerAnisotropy);
    si.anisotropyEnable = (aniso > 1.0f) ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = aniso;
    si.maxLod = VK_LOD_CLAMP_NONE;
    si.mipLodBias = bias;
    vk_check(vkCreateSampler(device_, &si, nullptr, &texture_sampler_),
             "texture sampler (fsr2 update)");
    if (bias != 0.0f) {
        log::infof("[fsr2] texture sampler bias = %.2f", bias);
    }
}

// Bake grass eligibility into an RG8 texture once at level load.
//   R = presence  (low-freq FBM, smoothed 0..1)
//   G = slope mag (raw |∇h| clamped to [0..2] / 2 → [0..1])
// Storing them separately lets the shaders apply the slope-cutoff
// slider (`grass_slope_n_min`) at runtime without re-baking. Both
// raymarched grass + the terrain green-tint sample this same texture
// for consistent eligibility.
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
            // (≈2 m / texel) and the shader samples it with LINEAR
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
// rescaled to UV by 1/16 so the texture tiles every 16 q-units (≈800 m
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

void VulkanEngine::init_viewmodel() {
    // Procedural placeholder gun: a barrel (cylinder) + body (box) + grip
    // (box). Camera-space coords: +X right, +Y up, -Z forward.
    auto rotX = [](float deg) {
        return glm::rotate(glm::mat4(1.0f), glm::radians(deg),
                           glm::vec3(1.0f, 0.0f, 0.0f));
    };
    auto T = [](glm::vec3 t) { return glm::translate(glm::mat4(1.0f), t); };
    auto S = [](glm::vec3 s) { return glm::scale(glm::mat4(1.0f), s); };

    // Dark-grey palette — gunmetal body, near-black sights/barrel,
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
        // earlier 0.30m. The 180° Y-rotation flips the gun's nose to
        // point forward (camera -Z); without it the assets/gun model
        // points back at the player.
        // m254 asset orientation walk-through: bare → pointed UP, pitch
        // -90°X → pointed LEFT on its side. The asset's barrel axis
        // turned out to be -X (not +X as previously assumed) — yaw
        // +90° around Y mapped that to +Z (toward the player) and the
        // gun pointed backwards. Yaw -90° flips it the other way so
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
        log::info("spacejet: no asset at assets/spacejet/scene.gltf вЂ” skipping");
        return;
    }
    // Normalise to a sensible visual size вЂ” scale so the asset's
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
    log::infof("spacejet: loaded %zu primitives, base_scale=%.3f вЂ” flyover system armed",
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

    // Pick a flight heading. 0..2ПЂ, 0 = +Z. Spawn point is 1.4 km
    // behind the castle (i.e. opposite the heading direction) so the
    // jet flies inbound, over, and out the other side.
    float heading = frand(spacejet_rng_state_) * 6.28318530718f;
    glm::vec3 fwd{ std::sin(heading), 0.0f, std::cos(heading) };
    glm::vec3 right{ fwd.z, 0.0f, -fwd.x };

    constexpr float kTrackLen = 2800.0f;
    // Lateral jitter dropped 80 m → 25 m so every wave actually
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
        // Viewmodel is camera-attached РІР‚вЂќ prev_mvp = mvp gives zero motion,
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
    // barrel tip РІР‚вЂќ without this, the small fade-in/out cube floats
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
            // Sleeping РІвЂ вЂ™ no motion this frame. Drag prev_world forward.
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
    // РІР‚вЂќ the player's collision is still solid against tilted falling crates.
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
            // Sleeping body in a slot we already populated РІвЂ вЂ™ AABB hasn't
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
                       terrain_raymarch_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeline_layout_, 0, 1, &scene_desc_set_,
                             0, nullptr);

    PushConstants pc{};
    pc.mvp      = vp;
    pc.model    = fv.inv_vp;
    pc.prev_mvp = prev_view_proj_valid_ ? prev_view_proj_ : vp;
    // Plateau half-extent must mirror the full-res raymarch path
    // (line ~2800) — otherwise the LR pass paints a much wider
    // plateau than the full-res one, and switching scale below 100%
    // makes the area around the castle visibly jump up to plateau
    // height. Was 28.0 (pre-shrink default), full-res was already 11.5.
    pc.color    = glm::vec4(0.0f, 0.0f, 11.5f, 22.0f);
    // emissive layout for the raymarch shader:
    //   x = step factor (0.15..0.8) — march step relative to SDF gap
    //   y = lod_near_m  — ray distance at which the FBM LOD ramp starts
    //   z = lod_far_m   — ray distance at which the FBM LOD ramp ends
    //   w = lod_min_oct — minimum octave count at far end of ramp
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
    // Fullscreen-tri compose pass вЂ” sits at the START of the main
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
    // .x stays 0 — the grass shader doesn't read a step factor.
    pc.emissive = glm::vec4(
        0.0f,
        std::max(1.0f, rt_.terrain_raymarch_lod_near_m),
        std::max(rt_.terrain_raymarch_lod_near_m + 1.0f,
                 rt_.terrain_raymarch_lod_far_m),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
    pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
    pc.grass_params = glm::vec4(rt_.grass_distance, rt_.grass_wind,
                                 static_cast<float>(frame_number_) * 0.016f,
                                 0.0f);
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
    // → terrain pixel discarded → scene_color stays at sky clear
    // colour → user sees sky through the cavity gap (the symptom that
    // looks like a "squared edge" with sky leak). Letting terrain
    // compose win the depth race for those pixels fixes it; we lose
    // early-Z on castle stone walls only.
    // Only the SPOM WALL brushes (albedo 1, 4) need to skip the depth
    // pre-pass — that's how their silhouette cavity shows what's
    // behind. Floors (albedo 5, 6) are SPOM-textured but face UP, so
    // their silhouette never points at terrain. Skipping them here
    // caused TAA flicker: cube color pass and terrain compose race
    // at the 5 cm floor-vs-plateau separation, and sub-pixel jitter
    // flips the winner every frame → "flashing through the floor".
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
    for (size_t i = 0; i < dyn_props_.size(); ++i) {
        const DynRender& dr = i < dyn_render_cache_.size() ? dyn_render_cache_[i]
                                                            : DynRender{};
        if (!dr.valid) continue;
        if (!aabb_visible(frustum, dr.aabb_min, dr.aabb_max)) continue;
        // dr.model = dr.world * scale(full_size), baked in
        // rebuild_dyn_render_cache (was per-draw glm::scale here).
        push_depth(dr.model);
    }
    // Terrain: distance-LOD per chunk. Pre-pass and color pass MUST
    // pick the same LOD AND morph factor per chunk so the rasterised
    // depth values match РІР‚вЂќ otherwise the LESS_OR_EQUAL test in the
    // color pass would discard fragments. We bind terrain_depth_pipeline_
    // (same terrain.vert as the color pass, just routed to depth-only).
    if (!terrain_chunks_.chunks.empty() && !rt_.terrain_raymarch_enabled) {
        // Skip the chunked depth prime when the procedural raymarch is
        // active вЂ” its frag writes its own depth via gl_FragDepth in
        // the color pass, and rasterised chunk depths from the prime
        // would wrongly occlude that with stale heightfield depths.
        glm::vec3 cam = player_.eye_position();
        const bool tess_on = rt_.terrain_tessellation_enabled &&
                             terrain_tess_depth_pipeline_ != VK_NULL_HANDLE;
        const float tess_r2 = rt_.terrain_tess_range * rt_.terrain_tess_range;
        auto chunk_near = [&](const TerrainChunk& c) {
            glm::vec3 ctr = 0.5f * (c.aabb_min + c.aabb_max);
            float dx = ctr.x - cam.x, dz = ctr.z - cam.z;
            return (dx * dx + dz * dz) < tess_r2;
        };

        // Pass A: near chunks primed with the SAME tessellation
        // pipeline (vert/tesc/tese) and SAME push constants as the
        // color pass, so the primed depth is bit-identical → no
        // 1-ULP LESS_OR_EQUAL rejects (sky-colour patches).
        if (tess_on) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              terrain_tess_depth_pipeline_);
            for (const auto& c : terrain_chunks_.chunks) {
                if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
                if (!chunk_near(c)) continue;
                VkDeviceSize voff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &c.mesh.vertex_buffer, &voff);
                vkCmdBindIndexBuffer(cmd, c.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
                PushConstants pc{};
                pc.mvp = vp;
                pc.model = glm::mat4(1.0f);
                float sx = std::max(1e-3f, c.aabb_max.x - c.aabb_min.x);
                float sz = std::max(1e-3f, c.aabb_max.z - c.aabb_min.z);
                pc.color = glm::vec4(c.aabb_min.x, c.aabb_min.z,
                                     1.0f / sx, 1.0f / sz);
                // Tess knobs — MUST match the color pass exactly or the
                // primed depth won't equal the color depth (sky patches).
                pc.emissive = glm::vec4(rt_.terrain_tess_max_level,
                                        rt_.terrain_tess_near_m,
                                        rt_.terrain_tess_far_m,
                                        rt_.terrain_tess_falloff);
                // .y = rock vertex relief, .z = tess smoothing → .tese.
                // BOTH MUST equal the colour pass's values or primed
                // depth ≠ colour depth (sky-patch z-fight).
                pc.tex_params = glm::vec4(0.0f, rt_.terrain_rock_relief,
                                          rt_.terrain_tess_smooth, 2.0f);
                // grass_params .z/.w drive the .tese's rocky-grass HEIGHT-
                // map displacement (kRockyAmp / kDispMip). MUST match the
                // colour pass's values or the primed depth sits below the
                // colour geometry → LESS_OR_EQUAL rejects → missing tris.
                pc.grass_params = glm::vec4(rt_.terrain_sand_ripple_scale,
                                            rt_.terrain_grass_line_scale,
                                            rt_.terrain_disp_amp,
                                            rt_.terrain_disp_smooth_mip);
                vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, c.index_count_lod[0], 1, 0, 0, 0);
            }
        }

        // Pass B: the rest (and everything if tess off) with the plain
        // CD-LOD depth pipeline.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_depth_pipeline_);
        // Same scaled thresholds shared with the colour + sun-shadow
        // passes -- mismatch would cause LESS_OR_EQUAL depth rejects.
        float thresh[kTerrainLodCount - 1];
        for (int i = 0; i < kTerrainLodCount - 1; ++i)
            thresh[i] = rt_.terrain_lod_distance[i] * rt_.terrain_lod_scale;
        for (const auto& c : terrain_chunks_.chunks) {
            if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
            if (tess_on && chunk_near(c)) continue;   // primed in pass A
            int lod = pick_terrain_lod(c, cam, thresh, kTerrainLodCount - 1);
            float morph = pick_terrain_morph(c, cam, lod, thresh[0]);
            VkBuffer vbufs[2] = { c.mesh.vertex_buffer, c.parent_y_buffer };
            VkDeviceSize voffs[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
            VkBuffer ibo = (lod == 0) ? c.mesh.index_buffer : c.ibo_lod[lod - 1];
            vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
            PushConstants pc{};
            pc.mvp = vp;
            pc.model = glm::mat4(1.0f);
            pc.color = glm::vec4(1.0f, 1.0f, 1.0f, morph);
            // grass_params.z/.w drive the terrain.vert rocky vertex
            // displacement. MUST match the colour pass values exactly
            // or the primed depth sits BELOW the displaced colour
            // geometry -> LESS_OR_EQUAL rejects -> sky/black patches.
            // (Same bug class as the tess depth-prime issue earlier.)
            pc.grass_params = glm::vec4(rt_.terrain_sand_ripple_scale,
                                        rt_.terrain_grass_line_scale,
                                        rt_.terrain_disp_amp,
                                        rt_.terrain_disp_smooth_mip);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, c.index_count_lod[lod], 1, 0, 0, 0);
        }
        // Caller binds depth_pipeline_ before invoking us; it will
        // continue using that for any non-terrain post-terrain draws.
        // Re-bind so the rest of the depth pre-pass goes through it.
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

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                            0, 1, &scene_desc_set_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const bool tex_on = rt_.textures_enabled;
    // prev_view_proj_ is captured at the END of each frame from the previous
    // frame's last_view_proj_; on frame 0 it's identity (which produces a
    // single-frame motion-vector glitch РІР‚вЂќ TAA's "history_valid" gate handles
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
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
    };

    int culled_static = 0, culled_dyn = 0, drawn_static = 0, drawn_dyn = 0;
    // Terrain pass РІР‚вЂќ chunked. Each visible chunk is drawn at full LOD
    // when within `near_lod` metres of the camera, otherwise half LOD.
    // The merged-static-BLAS-style "single big draw" is reserved for the
    // RT path; for raster we want LOD + frustum cull per chunk so the
    // 2km terrain stays cheap from any viewpoint.
    // Mesh (CDLOD) terrain draws first when the raymarcher is off, so its
    // depth is in the buffer before the (optional) water-only raymarch
    // pass below — hills then correctly occlude the water plane.
    if (!rt_.terrain_raymarch_enabled && !terrain_chunks_.chunks.empty()) {
        // Switch to the terrain-specific pipeline (terrain.vert + cube.frag,
        // 2 vertex bindings РІР‚вЂќ pos/norm/uv + parent_y) so we can morph
        // between LOD 0 and LOD 1 in the vertex shader. Skirts still hide
        // LOD-mismatch cracks at higher LOD seams.
        glm::vec3 cam = player_.eye_position();
        const bool tess_on = rt_.terrain_tessellation_enabled &&
                             terrain_tess_pipeline_ != VK_NULL_HANDLE;
        const float tess_r2 = rt_.terrain_tess_range * rt_.terrain_tess_range;
        auto chunk_near = [&](const TerrainChunk& c) {
            glm::vec3 ctr = 0.5f * (c.aabb_min + c.aabb_max);
            glm::vec3 d(ctr.x - cam.x, 0.0f, ctr.z - cam.z);
            return (d.x * d.x + d.z * d.z) < tess_r2;
        };
        // The chunk parameter is unused by this lambda -- the morph factor
        // is the only per-chunk value pushed via PC here (XZ origin/size are
        // handled by the tessellated near-chunk path, not this CD-LOD one).
        auto fill_pc = [&](const TerrainChunk& /*c*/, float morph) {
            PushConstants pc{};
            pc.mvp = vp;
            pc.model = glm::mat4(1.0f);
            pc.prev_mvp = prev_vp;
            // .w doubles as the morph factor РІР‚вЂќ terrain.vert lerps Y by it.
            pc.color = glm::vec4(1.0f, 1.0f, 1.0f, morph);
            pc.emissive = glm::vec4(0.0f);
            // .x = terrain parallax-occlusion strength (the terrain
            // fragment path ignores tex_params.x/.y otherwise).
            pc.tex_params = glm::vec4(rt_.terrain_pom_strength, 0.0f,
                                      1.0f, 2.0f);
            // .x = sand ripple scale, .y = grass contour-line scale
            // (cube.frag terrain reads these; 0 = grass lines off).
            // .z = rocky vertex disp amplitude, .w = disp mip smooth
            // -- consumed by terrain_tess.tese for the grass-band
            // displacement. Wired to UI sliders so the user can dial
            // both at runtime.
            pc.grass_params = glm::vec4(rt_.terrain_sand_ripple_scale,
                                        rt_.terrain_grass_line_scale,
                                        rt_.terrain_disp_amp,
                                        rt_.terrain_disp_smooth_mip);
            vkCmdPushConstants(cmd, pipeline_layout_,
                               kPushConstantStages,
                               0, sizeof(PushConstants), &pc);
        };

        // Pass A: camera-near chunks via the GPU tessellation pipeline
        // (LOD0 triangle IBO consumed as 3-CP patches; only vertex
        // binding 0 — no parent_y/morph for near chunks).
        if (tess_on) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                (rt_.terrain_wireframe && terrain_tess_wire_pipeline_)
                    ? terrain_tess_wire_pipeline_ : terrain_tess_pipeline_);
            for (const auto& c : terrain_chunks_.chunks) {
                if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
                if (!chunk_near(c)) continue;
                VkDeviceSize voff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &c.mesh.vertex_buffer, &voff);
                vkCmdBindIndexBuffer(cmd, c.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
                PushConstants pc{};
                pc.mvp = vp;
                pc.model = glm::mat4(1.0f);
                pc.prev_mvp = prev_vp;
                // pc.color carries the chunk XZ footprint for the .tese's
                // border taper (no morph in the tess path, vColor is a
                // constant in cube.frag, so .color is free here):
                //   .xy = chunk min (x,z),  .zw = 1 / chunk size (x,z)
                float sx = std::max(1e-3f, c.aabb_max.x - c.aabb_min.x);
                float sz = std::max(1e-3f, c.aabb_max.z - c.aabb_min.z);
                pc.color = glm::vec4(c.aabb_min.x, c.aabb_min.z,
                                     1.0f / sx, 1.0f / sz);
                // Tess knobs (UI sliders) → .tesc. Must equal the
                // depth-prime pass's value so depths match.
                pc.emissive = glm::vec4(rt_.terrain_tess_max_level,
                                        rt_.terrain_tess_near_m,
                                        rt_.terrain_tess_far_m,
                                        rt_.terrain_tess_falloff);
                // .x = parallax (pixel displacement, fragment only),
                // .y = rock vertex relief, .z = tess smoothing.
                // .y/.z drive .tese geometry → MUST equal the
                // depth-prime pass's values or primed depth differs.
                pc.tex_params = glm::vec4(rt_.terrain_pom_strength,
                                          rt_.terrain_rock_relief,
                                          rt_.terrain_tess_smooth, 2.0f);
                // .x = sand ripple scale, .y = grass contour-line scale.
                pc.grass_params = glm::vec4(rt_.terrain_sand_ripple_scale,
                                            rt_.terrain_grass_line_scale,
                                            rt_.terrain_disp_amp,
                                            rt_.terrain_disp_smooth_mip);
                vkCmdPushConstants(cmd, pipeline_layout_,
                                   kPushConstantStages,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, c.index_count_lod[0], 1, 0, 0, 0);
            }
        }

        // Pass B: the rest (and everything, if tess is off) via the
        // plain CD-LOD morph pipeline.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            (rt_.terrain_wireframe && terrain_wire_pipeline_)
                ? terrain_wire_pipeline_ : terrain_pipeline_);
        // Hoist the scaled LOD thresholds out of the per-chunk loop --
        // identical to the depth pre-pass thresholds above.
        float thresh[kTerrainLodCount - 1];
        for (int i = 0; i < kTerrainLodCount - 1; ++i)
            thresh[i] = rt_.terrain_lod_distance[i] * rt_.terrain_lod_scale;
        for (const auto& c : terrain_chunks_.chunks) {
            if (!aabb_visible(frustum, c.aabb_min, c.aabb_max)) continue;
            if (tess_on && chunk_near(c)) continue;   // drawn in pass A
            int lod = pick_terrain_lod(c, cam, thresh, kTerrainLodCount - 1);
            float morph = pick_terrain_morph(c, cam, lod, thresh[0]);
            VkBuffer vbufs[2] = { c.mesh.vertex_buffer, c.parent_y_buffer };
            VkDeviceSize voffs[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
            VkBuffer ibo = (lod == 0) ? c.mesh.index_buffer : c.ibo_lod[lod - 1];
            vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
            fill_pc(c, morph);
            vkCmdDrawIndexed(cmd, c.index_count_lod[lod], 1, 0, 0, 0);
        }
        // Switch back to the cube pipeline + cube mesh for the brush loop.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

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
        // Procedural FBM ray-marched terrain вЂ” fullscreen draw, frag
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
        // 22 m height, centred at origin — tight pad just under the
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
        // emissive layout — mirrors render_terrain_raymarch_lr:
        //   x = step factor (0.15..0.8)
        //   y = lod_near_m  — ray-t at which FBM LOD ramp starts
        //   z = lod_far_m   — ray-t at which FBM LOD ramp ends
        //   w = lod_min_oct — minimum octave count at far end
        pc.emissive = glm::vec4(
            std::clamp(rt_.terrain_raymarch_step_factor, 0.15f, 0.8f),
            std::max(1.0f, rt_.terrain_raymarch_lod_near_m),
            std::max(rt_.terrain_raymarch_lod_near_m + 1.0f,
                     rt_.terrain_raymarch_lod_far_m),
            static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
        // grass_params slot вЂ” fog strength + flags (mirrors LR pass).
        pc.grass_params = glm::vec4(
            std::max(0.0f, rt_.terrain_raymarch_fog_strength),
            rt_.terrain_raymarch_relaxation ? 1.0f : 0.0f,
            rt_.terrain_raymarch_fog_godrays ? 1.0f : 0.0f,
            rt_.water_rt_reflections ? 1.0f : 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_,
                           kPushConstantStages,
                           0, sizeof(PushConstants), &pc);
        // 3 verts, 1 instance вЂ” gl_VertexIndex 0..2 covers the screen.
        vkCmdDraw(cmd, 3, 1, 0, 0);
        // Restore cube pipeline + cube mesh for the grass / dust /
        // viewmodel passes that follow.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    // Mesh-terrain mode + water: draw the REAL rasterised water plane
    // (verbatim ocean shading in water.frag) instead of the fullscreen
    // analytic water-only pass. Depth is owned by the plane geometry, so
    // the terrain↔water silhouette is a pixel-exact hardware depth test
    // — the previous analytic-vs-rasterised mismatch (edge gaps / seam
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
        // .x = water level → water.vert lifts the plane to it; the frag
        // still reads scene.water_params.y for the shaded surface.
        // .y = clarity depth (m) → Beer's-law absorption in water.frag.
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
        // few cm at 1 km — invisible to the eye, decisive for the
        // depth test.
        vkCmdSetDepthBias(cmd, /*const*/ 16.0f, /*clamp*/ 0.0f,
                          /*slope*/ 1.5f);
        VkDeviceSize woff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &water_plane_mesh_.vertex_buffer, &woff);
        vkCmdBindIndexBuffer(cmd, water_plane_mesh_.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, water_plane_mesh_.index_count, 1, 0, 0, 0);
        // Restore cube pipeline + mesh for the passes that follow.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    // Spacejet вЂ” hovering decoration on the plateau. One draw per
    // glTF primitive. Animated via sin(time) for the bob and a
    // slow yaw rotation. Raster-only; no physics, no BLAS.
    if (!spacejet_meshes_.empty() && !spacejet_flights_.empty()) {
        for (const auto& f : spacejet_flights_) {
            // Asset's nose points -Z in glTF space; we need it pointing
            // along the heading direction +Z(rotated). Yaw value
            // (heading + pi) flips so rotY puts -Z in line with fwd.
            const float yaw_now  = f.heading      + kPi;
            const float yaw_prev = f.prev_heading + kPi;
            // Bank roll proportional to turn rate вЂ” banked-into-the-turn
            // look. Cap at В±35В° so even tight turns stay readable.
            const float bank_now  = glm::clamp(-f.turn_rate * 4.0f,
                                                -0.61f, 0.61f);
            const float bank_prev = bank_now;     // bank changes very slowly
            auto build_model = [&](glm::vec3 pos, float yaw, float bank) {
                return glm::translate(glm::mat4(1.0f), pos) *
                       glm::rotate   (glm::mat4(1.0f), yaw,  glm::vec3(0,1,0)) *
                       glm::rotate   (glm::mat4(1.0f), bank, glm::vec3(0,0,1)) *
                       glm::scale    (glm::mat4(1.0f),
                                       glm::vec3(spacejet_base_scale_ * f.scale)) *
                       glm::translate(glm::mat4(1.0f), spacejet_centre_offset_);
            };
            glm::mat4 model      = build_model(f.pos,      yaw_now,  bank_now);
            glm::mat4 prev_model = build_model(f.prev_pos, yaw_prev, bank_prev);

            // Red hit-flash: emissive bumps to bright red for the
            // hit_flash_t window (~0.2 s) so the player sees damage
            // register. Destroying flights also flash continuously
            // until they're removed at the end of the fade.
            const float flash = std::max(f.destroying ? 0.6f : 0.0f,
                                          glm::clamp(f.hit_flash_t / 0.20f,
                                                     0.0f, 1.0f));
            const glm::vec4 hit_emissive(2.8f * flash, 0.05f * flash,
                                          0.05f * flash, 0.0f);
            // Tint the base colour reddish while flashing too — emissive
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

    // Grass РІР‚вЂќ instanced billboard blades. Single draw call covers
    // every blade; the grass.vert collapses out-of-range blades to
    // a degenerate triangle (NaN clip space) so no fragment work runs
    // for them. Skipped entirely when disabled in Settings or when
    // build_grass produced an empty set.
    // Raymarched grass takes over the entire grass slot when enabled —
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
        // Grass push constants РІР‚вЂќ only grass_params is read by grass.vert,
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
        uint32_t inst_n = static_cast<uint32_t>(
            static_cast<float>(grass_.instance_count) * frac);
        if (inst_n > 0) {
            vkCmdDrawIndexed(cmd, grass_.blade_mesh.index_count,
                             inst_n, 0, 0, 0);
        }
        // Restore the world pipeline + cube mesh bindings for the
        // brush loop below.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    for (size_t i = 0; i < world_.brushes.size(); ++i) {
        const auto& b = world_.brushes[i];
        const auto& a = world_.render_aabbs[i];
        if (!aabb_visible(frustum, a.min, a.max)) { ++culled_static; continue; }
        const glm::mat4& model = static_brush_models_[i];   // pre-baked, see init_world
        glm::vec4 brush_base = tex_on ? b.color : b.fallback_color;
        // Static brushes don't move РІР‚вЂќ prev_model = current.
        draw_brush(model, model, brush_base, b.emissive, b.full_emissive,
                   b.tex_albedo, b.tex_normal, b.uv_scale, false);
        ++drawn_static;
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

    // SPOM silhouette shell pass DISABLED. Iterations explored:
    //   1. cube.vert per-face extrusion (offset bug on walls)
    //   2. Paper SSDM screen-space remap (paper assumes outward bumps;
    //      our inward-recessed bricks give barely-visible extension)
    //   3. Shell mesh, normal-push only (corner gaps show sky)
    //   4. Shell mesh + lateral push (corners fill but shell overdraws
    //      walls with simpler N·L lighting → "weird shadows")
    // Real dramatic silhouette extension requires either stencil-based
    // masking (touches every depth-aware pipeline) or re-authored
    // outward-bump height maps. Leaving render_shell_pass / shell
    // pipeline in the codebase so re-enabling is one-line.
    // render_shell_pass(cmd);

    // Bullet-impact decals РІР‚вЂќ drawn AFTER static + dyn brushes so they sit
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

    // Projectiles: switch to cylinder mesh, draw each in-flight bullet.
    if (!projectiles_.empty() && physics_) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cylinder_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        for (const auto& p : projectiles_) {
            glm::mat4 world;
            // _h handle path skips the unordered_map.find that the
            // id-keyed overload does (rebuild_tlas already uses _h here).
            if (!physics_->get_body_world_matrix_h(p.jolt_handle, world)) continue;
            glm::vec3 pos(world[3]);
            glm::vec3 vel = physics_->get_linear_velocity_h(p.jolt_handle);
            // Single sqrt for both length-test + normalize (was length()
            // then normalize() which is another sqrt internally).
            float v_len2 = glm::dot(vel, vel);
            glm::vec3 dir = v_len2 > 1e-6f
                ? vel * (1.0f / std::sqrt(v_len2))
                : p.initial_dir;
            glm::mat4 model = align_local_y_to(pos, dir) *
                              glm::scale(glm::mat4(1.0f),
                                         glm::vec3(p.radius, p.half_length, p.radius));
            PushConstants pc{};
            pc.mvp = vp * model;
            pc.model = model;
            // Projectiles don't track a prev_pose yet РІР‚вЂќ zero-motion approx.
            pc.prev_mvp = pc.mvp;
            pc.color = p.color;
            pc.emissive = glm::vec4(p.emissive, 1.0f);  // full_emissive
            vkCmdPushConstants(cmd, pipeline_layout_,
                               kPushConstantStages,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, cylinder_mesh_.index_count, 1, 0, 0, 0);
        }
    }

    // Raymarched grass — deferred to here so the alpha blend reads
    // the actual castle/cube/dyn-prop colours instead of the still-
    // cleared sky background. Was running before the brush loops,
    // which made translucent blades in front of the castle show sky
    // colour through them.
    if (rt_.grass_enabled && rt_.grass_raymarch_enabled &&
        grass_rm_pipeline_ != VK_NULL_HANDLE) {
        render_grass_raymarch(cmd);
        // Restore the world cube pipeline + cube mesh for the
        // viewmodel draw immediately below.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0,
                              VK_INDEX_TYPE_UINT32);
    }

    // Viewmodel: rendered last so it overdraws world geometry only when its
    // depth wins. Its depth values are tiny (close to camera), so it sits in
    // front of everything in practice.
    // fv.inv_view is cached in compute_frame_view (was a per-frame
    // glm::inverse(fv.view) here -- a full 4x4 mat inverse is ~80 fmuls).
    draw_viewmodel(cmd, vp, fv.inv_view);

    // Voxel buildings (Session A) — drawn last of all because the voxel
    // pipeline has its own layout / descriptor set; nothing else in
    // render_world should run after it without rebinding pipeline_.
    // The viewmodel sits in front of everything so it always wins the
    // depth test against the 100 m-distant voxel tower regardless of
    // draw order. Brick-DDA ray-march into scene_color + motion_vec +
    // gl_FragDepth.
    render_voxels(cmd);

    last_draw_static_ = drawn_static;
    last_draw_dyn_    = drawn_dyn;
    last_culled_      = culled_static + culled_dyn;
}

} // namespace qlike


