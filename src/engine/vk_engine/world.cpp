// World bring-up: init_world() builds the level brushes + cube/cylinder
// meshes, loads the heightmap, places grass, and spawns the Jolt static
// bodies. The subsystems that used to share this file were split out:
//   - terrain_resources.cpp  (sculpt/height/CLOD/shadow-bake/aux textures)
//   - world_textures.cpp     (texture array, skybox, material baking)
//   - props.cpp              (viewmodel, spacejet, doors, box spawning)
//   - world_render.cpp       (compute_frame_view + the raster render passes)

#include "engine/vk_engine/internal.h"
#include "engine/world_props.h"
#include "engine/grass.h"
#include "engine/terrain.h"
#include "engine/texture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace qlike {

void VulkanEngine::init_world() {
    // load_settings() runs AFTER init_world, but world generation (road
    // levelling, vegetation / rock / scatter placement) depends on the
    // water level and terrain bands. Peek those keys from the cfg now -
    // with the defaults (water 2 m vs the saved 20.9 m) trees were placed
    // in the lake.
    {
        std::ifstream sf(kSettingsPath);
        std::string line;
        while (std::getline(sf, line)) {
            auto eq = line.find(" = ");
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq);
            float v = 0.0f;
            try { v = std::stof(line.substr(eq + 3)); } catch (...) { continue; }
            if (k == "water_level") rt_.water_level = v;
            else if (k == "terrain_h_dirt_rock_start") rt_.terrain_h_dirt_rock_start = v;
            else if (k == "terrain_h_rock_snow_start") rt_.terrain_h_rock_snow_start = v;
            else if (k == "terrain_heightmap_scale") rt_.terrain_heightmap_scale = int(v);
        }
    }
    cube_mesh_ = create_cube_mesh(device_, allocator_, graphics_queue_, graphics_queue_family_);
    cylinder_mesh_ = create_cylinder_mesh(device_, allocator_,
                                          graphics_queue_, graphics_queue_family_);
    world_ = game::make_arena();
    log::infof("world: %u brushes", static_cast<unsigned>(world_.brushes.size()));
    // Brush model matrices are baked later, AFTER the plateau y-lift
    // below — building them here would cache un-lifted centers and the
    // castle would render 22 m under the terrain (invisible).

    // Procedural terrain: generated first so the castle can be lifted to
    // sit on the plateau height. Phase 1 is non-streaming — one big mesh
    // (~1km²). See docs/terrain_plan.md for streaming/LOD/sculpt phases.
    {
        HeightmapParams hp{};
        // Two constraints that fight each other:
        //   1. chunk_cells must be divisible by every LOD stride (1, 2, 4, 8)
        //      so the stride-N index loop covers every cell — otherwise the
        //      lower-LOD index buffers leave a strip of cells uncovered at
        //      the chunk's right/bottom edge → visible square seams.
        //   2. Jolt's HeightFieldShape needs sample_count divisible by
        //      block_size in [2, 8].
        //
        // chunk_cells / 8 means dim is divisible by 8 → sample_count = dim+1
        // is odd → Jolt fails. Resolution: pick dim = 2048 cells (=32×64),
        // chunk_cells = 64 (divisible by all four strides), and feed Jolt a
        // 2048×2048 sub-grid by truncating the last row/col of the 2049²
        // heightmap. The visual mesh uses the full 2049² so the world edge
        // stays continuous. Jolt loses one cell-width at the world edge —
        // invisible since the player can't reach the world boundary.
        // Heightmap resolution scale — the user can pick higher density
        // via the Terrain settings menu. Defaults to 1× = the baseline
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
        //     adds sampleHeightDelta = (current в€’ baseline). The
        //     baseline MUST be this FBM or, after a runtime mesh→
        //     raymarch toggle, the delta becomes a difference of two
        //     unrelated noise fields layered on the shader FBM →
        //     "way too spikey" terrain (the exact reported bug).
        //   * mesh / physics / grass / BLAS now all share ONE surface,
        //     so toggling modes is seamless and the world is
        //     consistent — and the planned near-distance displacement-
        //     LOD work needs the mesh to be this same FBM anyway.
        // (The FastNoiseLite generator was removed.)
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
        // Terrain → Save heightmap button), load it now and overwrite
        // the procedurally-generated one. Validates dim+cell match
        // so a heightmap saved at 2048×1m doesn't mis-load on a
        // 4096×0.5m level.
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
                } else if (fdim > 0 && fcell > 0.0f &&
                           std::abs(float(fdim) * fcell - float(hm.dim) * hm.cell) < 1e-2f &&
                           std::abs(fox - hm.origin_x) < 1e-2f &&
                           std::abs(foz - hm.origin_z) < 1e-2f &&
                           (fdim % hm.dim == 0 || hm.dim % fdim == 0)) {
                    // Same world square, different heightmap scale. Keep
                    // the sculpt as an EDIT: delta = saved - procedural
                    // baseline at the saved grid, resampled onto the
                    // current grid and added to this grid's baseline. The
                    // FBM is evaluated at world positions, so the coarse
                    // grid's samples coincide with every r-th fine sample
                    // and the fine procedural detail survives an upscale.
                    const int SW = fdim + 1, DW = hm.dim + 1;
                    std::vector<float> saved(size_t(SW) * SW);
                    hf.read(reinterpret_cast<char*>(saved.data()), saved.size() * sizeof(float));
                    if (hf) {
                        if (fdim < hm.dim) {
                            const int r = hm.dim / fdim;
                            std::vector<float> delta(saved.size());
                            for (int z = 0; z < SW; ++z)
                                for (int x = 0; x < SW; ++x)
                                    delta[size_t(z) * SW + x] = saved[size_t(z) * SW + x] -
                                        hm.heights[size_t(z * r) * DW + size_t(x * r)];
                            // Catmull-Rom (bicubic, interpolating, C1): a bilinear
                            // upsample kept a crease along every old 1 m cell edge -
                            // visible as hard lines in the shading and RT shadows.
                            const float inv = 1.0f / float(r);
                            auto cr = [](float t, float w[4]) {
                                const float t2 = t * t, t3 = t2 * t;
                                w[0] = -0.5f * t3 + t2 - 0.5f * t;
                                w[1] =  1.5f * t3 - 2.5f * t2 + 1.0f;
                                w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
                                w[3] =  0.5f * t3 - 0.5f * t2;
                            };
                            std::vector<float> rowbuf(size_t(SW) * DW);   // x-upsampled rows
                            for (int z = 0; z < SW; ++z) {
                                for (int x = 0; x < DW; ++x) {
                                    const int x0 = std::min(x / r, fdim - 1);
                                    float w[4]; cr(float(x - x0 * r) * inv, w);
                                    float s = 0.0f;
                                    for (int k = 0; k < 4; ++k)
                                        s += w[k] * delta[size_t(z) * SW + std::clamp(x0 - 1 + k, 0, SW - 1)];
                                    rowbuf[size_t(z) * DW + x] = s;
                                }
                            }
                            for (int z = 0; z < DW; ++z) {
                                const int z0 = std::min(z / r, fdim - 1);
                                float w[4]; cr(float(z - z0 * r) * inv, w);
                                const float* rows[4];
                                for (int k = 0; k < 4; ++k)
                                    rows[k] = &rowbuf[size_t(std::clamp(z0 - 1 + k, 0, SW - 1)) * DW];
                                for (int x = 0; x < DW; ++x)
                                    hm.heights[size_t(z) * DW + x] +=
                                        w[0] * rows[0][x] + w[1] * rows[1][x] + w[2] * rows[2][x] + w[3] * rows[3][x];
                            }
                        } else {
                            const int r = fdim / hm.dim;
                            for (int z = 0; z < DW; ++z)
                                for (int x = 0; x < DW; ++x)
                                    hm.heights[size_t(z) * DW + x] = saved[size_t(z * r) * SW + size_t(x * r)];
                        }
                        log::infof("[terrain] loaded sculpted heightmap %dx -> %dx (resampled)",
                                   fdim, hm.dim);
                    } else {
                        log::warnf("[terrain] saved heightmap truncated - ignoring");
                    }
                } else {
                    log::warnf("[terrain] saved heightmap dim/cell mismatch "
                               "— ignoring (saved %dx %.3fm, want %dx %.3fm)",
                               fdim, fcell, hm.dim, hm.cell);
                }
            }
        }
        // Castle-courtyard visual sink. The physics path already clamps
        // its own copy (see init_physics' kCastleSink), but the VISUAL
        // heights kept sculpt residue — with the CLOD terrain sampling
        // the height texture directly, raised ground inside the castle
        // rendered THROUGH the courtyard floor slab (grass-weave over
        // the tiles, buried stair bases, ground bulging at wall feet
        // that read as see-through walls at grazing angles). Clamp the
        // SOURCE heights inside the courtyard so raster, BLAS, physics
        // and every height texture agree.
        {
            const float kSinkR = 10.5f, kSinkH = 22.0f - 0.6f;
            const int Wv = hm.width();
            for (int z = 0; z < Wv; ++z) {
                float wz = hm.origin_z + float(z) * hm.cell;
                if (std::abs(wz) >= kSinkR) continue;
                for (int x = 0; x < Wv; ++x) {
                    float wx = hm.origin_x + float(x) * hm.cell;
                    if (std::abs(wx) >= kSinkR) continue;
                    float& hv = hm.heights[size_t(z) * Wv + x];
                    hv = (std::min)(hv, kSinkH);
                }
            }
        }
        // Castle road: levelled spline from the gate into the valley.
        // Carved into the SOURCE heights here so raster, BLAS, Jolt, the
        // shadow/aux bakes and grass all agree with the shader paving.
        castle_road_ = make_castle_road(hm, rt_.water_level, hp.plateau_height + 0.05f);
        carve_heights(castle_road_, hm);
        log::infof("[road] %zu centreline points carved", castle_road_.pts.size());
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
            // Using W here gave (gx*2049)/32 в‰€ gx*64.03 — an off-by-
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
        // Skip the rasterised terrain (BLAS source mesh + CLOD patch)
        // when the procedural raymarch is the active terrain. Caveat:
        // toggling raymarch OFF at runtime lazy-builds via
        // ensure_terrain_raster_built.
        if (!rt_.terrain_raymarch_enabled) {
            terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                               graphics_queue_, graphics_queue_family_, hm);
            // Quadtree CLOD raster: shared instanced patch + min/max
            // pyramid. The vertex shader fetches heights from the
            // heightmap texture, so there is no per-chunk geometry to
            // bake — startup cost is a single 4.5 k-vertex mesh upload.
            init_terrain_clod();
        } else {
            log::info("[terrain] raymarch enabled — deferring mesh/CLOD "
                      "build (will lazy-build if raymarch is toggled off)");
        }
        // Lift every level brush by plateau_height so the castle's y=0
        // baseline lands on the plateau surface. Cheaper than rewriting
        // level.cpp — the brush AABBs / world matrices update through the
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
        // Castle mesh parts (round towers, roofs, arch): same lift, then upload.
        for (auto& m : world_.meshes) {
            for (auto& v : m.verts) v.p.y += hp.plateau_height;
            m.bmin.y += hp.plateau_height;
            m.bmax.y += hp.plateau_height;
        }
        // Outpost towers on prominent spots in the foothills toward the
        // mountains (world-space meshes, appended after the castle lift).
        {
            uint32_t rng = 0x70E125u;
            struct Cand { glm::vec3 p; float score; };
            std::vector<Cand> cands;
            for (int i = 0; i < 9000; ++i) {
                float a = frand_range(rng, 0.0f, 6.2831853f), d = frand_range(rng, 210.0f, 560.0f);
                float x = std::cos(a) * d, z = std::sin(a) * d;
                float h = hm.sample_world(x, z);
                if (h < rt_.water_level + 5.0f || h > 84.0f) continue;
                if (!castle_road_.empty() && castle_road_.distance(x, z) < 16.0f) continue;
                float lo = h, hi = h, ring = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    float aa = float(k) * 0.785f;
                    float hh = hm.sample_world(x + std::cos(aa) * 5.0f, z + std::sin(aa) * 5.0f);
                    lo = std::min(lo, hh); hi = std::max(hi, hh);
                    ring += hm.sample_world(x + std::cos(aa) * 40.0f, z + std::sin(aa) * 40.0f);
                }
                if (hi - lo > 3.0f) continue;
                // Prominence over the surroundings + a preference for high ground.
                cands.push_back({ glm::vec3(x, lo, z), (h - ring / 8.0f) + (h - 30.0f) * 0.12f });
            }
            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
            const int styles[7] = { 0, 3, 1, 2, 0, 1, 2 };
            for (const auto& cd : cands) {
                if (outposts_.size() >= 7) break;
                bool clash = false;
                for (const auto& o : outposts_) if (glm::distance(glm::vec2(o.x, o.z), glm::vec2(cd.p.x, cd.p.z)) < 150.0f) clash = true;
                if (clash) continue;
                const int st = styles[outposts_.size()];
                game::append_outpost_tower(world_, cd.p, st, 0xBEEF00u + uint32_t(outposts_.size()) * 977u);
                outposts_.push_back(cd.p);
                log::infof("[outpost] tower style %d at (%.0f, %.1f, %.0f)", st, cd.p.x, cd.p.y, cd.p.z);
            }
        }
        // Procedural trees, bushes, rock formations: GPU-instanced with LOD
        // over the whole map (vegetation.cpp). Adds trunk / rock colliders.
        place_aa_vehicles(hm);           // before vegetation (keep-out)
        init_aa_meshes();
        {
            auto t0 = std::chrono::steady_clock::now();
            init_vegetation_data(hm);
            auto t1 = std::chrono::steady_clock::now();
            init_castle_meshes();
            auto t2 = std::chrono::steady_clock::now();
            log::infof("[init] vegetation %.0f ms, castle meshes %.0f ms",
                       std::chrono::duration<double, std::milli>(t1 - t0).count(),
                       std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
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
        // Cover the plateau too — slopes, valleys, and the plateau
        // courtyard around the castle. Castle stones are excluded by
        // the keep-out rectangle below so blades never poke through
        // the brushes themselves.
        // Placement bounds are GENEROUS — the runtime altitude
        // sliders (rt_.grass_alt_min/max) shape the visible band
        // without re-baking. Cover most reasonable terrain heights.
        gp.height_min = -20.0f;
        gp.height_max = 200.0f;
        // Concentrate blades in a 200m square around the castle so the
        // density is high (~12 blades/m²) where the player actually
        // sees grass. Spreading them across the full 1km map gave
        // ~0.5/m² (about a metre between blades — looks bald).
        gp.half_extent = 200.0f;
        // Inner keep is ~6m square at origin. Keep blades out of the
        // keep interior only; the outer courtyard between keep and
        // perimeter walls gets grass too. Blades whose footprint
        // overlaps a brush stay invisible — the wall geometry occludes
        // them from outside views.
        gp.keep_out_xz = glm::vec2(12.0f, 12.0f);  // full castle footprint
        // 800k placed blades: density slider 0..4 maps to render
        // fraction (density / 4) × placed, so density=1.0 → 200k,
        // density=4.0 → 800k. Earlier 2M cap meant density=4 fired
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

        // Ground clutter (AAA Phase 3): rocks + pebbles scattered by
        // splat rules. Rendered regardless of which grass path is on.
        ScatterParams sp{};
        sp.water_level = rt_.water_level;
        sp.rock_band_h = rt_.terrain_h_dirt_rock_start;
        sp.snow_band_h = rt_.terrain_h_rock_snow_start;
        sp.road = &castle_road_;
        sp.rm_params = &hp;   // offsets onto the raymarched surface
        scatter_ = build_scatter(device_, allocator_,
                                 graphics_queue_, graphics_queue_family_,
                                 hm, sp);
        scatter_set_raymarch_surface(scatter_, rt_.terrain_raymarch_enabled);
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
    // Castle mesh colliders (round towers) sit after the brush AABBs.
    const size_t veg_begin = std::min(veg_col_begin_, world_.aabbs.size());
    for (size_t i = world_.brushes.size(); i < veg_begin; ++i) {
        const auto& a = world_.aabbs[i];
        sboxes.push_back({(a.min + a.max) * 0.5f, (a.max - a.min) * 0.5f});
    }
    physics_->add_static_boxes(sboxes.data(), sboxes.size());
    // Vegetation trunks + rocks: merged per-cell compound bodies.
    {
        std::vector<PhysicsWorld::StaticBox> vb;
        vb.reserve(world_.aabbs.size() - veg_begin);
        for (size_t i = veg_begin; i < world_.aabbs.size(); ++i) {
            const auto& a = world_.aabbs[i];
            vb.push_back({(a.min + a.max) * 0.5f, (a.max - a.min) * 0.5f});
        }
        physics_->add_static_box_compounds(vb.data(), vb.size());
    }
    // Heightfield collision for terrain. Player + crates collide cleanly
    // against the bumpy ground. Jolt requires the sample-count side to be
    // divisible by some block_size in [2, 8]; with dim=2048 (sample_count
    // = 2049, odd) we'd violate that, so we copy the first 2048×2048 sub-
    // grid into a packed buffer and pass that with dim=2047 (sample_count
    // = 2048, block_size 8 ✓). Player can't reach the world edge so the
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
        // 10.5 m so the depressed zone stays strictly UNDER the courtyard
        // brush floor (which extends to cr-wt/2 в‰€ 10.70 m on each axis).
        // Previously 11.5 m — wider than the brush — left a ~50 cm
        // ring where the depressed Jolt heightfield was the only collider
        // and the brush didn't cover it. Boxes landing or rolling into
        // that ring dropped to y=21.4 while the visible mesh terrain was
        // rendered at yв‰€22, reading as "box inside terrain".
        const float kSinkRadius   = 10.5f;
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




} // namespace qlike



