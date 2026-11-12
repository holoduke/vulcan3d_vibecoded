#pragma once

// Graphics (RtSettings) and gameplay (GameSettings) tuning structs: plain
// data persisted to qlike_settings.cfg by settings.cpp and driven by the
// ImGui menu in vk_ui.cpp. Extracted from the VulkanEngine god-header so a
// settings tweak no longer recompiles the whole engine include surface.

#include <glm/glm.hpp>

namespace qlike {

struct RtSettings {
        // Sun position via spherical angles (so a UI slider feels intuitive).
        float sun_pitch_deg = 52.0f;   // elevation above horizon
        float sun_yaw_deg   = 35.0f;   // around Y, 0 = +Z
        // Auto golden-hour tint. When on, low sun (pitch ≲ 20°) drifts
        // the sun + sky colours toward warm orange / pinkish-amber for a
        // sunset/sunrise look; high sun is unaffected. Lives in
        // descriptors.cpp's update_scene_ubo so every shader that reads
        // sun_color / sky_color picks it up uniformly.
        bool  auto_golden_hour = true;
        // Sun glare / flare strength. Multiplies pc.sun_color.a going
        // into compose, scaling the sun's procedural DISC + HALO + lens-
        // flare brightness uniformly. Does NOT touch scene.sun_color so
        // direct lighting + shadows are unaffected. 0 = no procedural
        // disc/halo (sky tex alone), 1 = engine default brightness.
        float sun_glare_strength = 0.4f;
        // SVGF GI denoiser (Session 2 of docs/svgf_plan.md). Default
        // OFF — when on, cube.frag EMA-blends the GI signal with a
        // reprojected history sample, cutting noise dramatically on
        // static scenes. Mild ghosting on movement (bounded by M-cap
        // ≤ 8) until Session 3 adds the à-trous spatial filter.
        bool  svgf_enabled = false;
        float sun_intensity = 2.6f;
        glm::vec3 sun_color  = glm::vec3(1.00f, 0.96f, 0.86f);
        glm::vec3 sky_color  = glm::vec3(0.55f, 0.72f, 0.95f);
        glm::vec3 ground_ambient = glm::vec3(0.18f, 0.20f, 0.24f);
        float ambient_strength = 1.0f;
        float shadow_strength  = 1.0f;
        bool  shadow_enabled = true;
        // 40 stratified samples + slope-scale bias + the TAA/à-trous denoiser
        // gives a cinema-clean soft shadow.
        int   shadow_samples = 40;
        float shadow_softness = 0.016f;
        // PCSS curve: 0 = linear (penumbra ∝ blocker distance, our default),
        //             1 = cubic (sharp on contact, much softer at distance).
        float shadow_curve = 0.5f;
        int   ao_samples = 0;         // 0 = off; 1/4/8/16
        float ao_radius  = 1.5f;

        // Path-traced GI: each "sample" follows a path of up to gi_bounces
        // segments, picking up tinted color along the way.
        int   gi_samples = 64;        // paths per pixel; 0 = off
        int   gi_bounces = 3;         // 1 = single-bounce sky GI; 2-5 = multi-bounce
        // ReSTIR-lite (session 1 of the multi-session plan in
        // docs/restir_plan.md). When enabled, gi_samples is divided by 4
        // before the shader uses it — cube.frag traces 4x fewer rays per
        // pixel, the existing TAA + spatial filter fold the noise back
        // into a stable image. Sessions 2+ replace the / 4 hack with a
        // real reservoir SSBO, motion-vec reproject, and 3-tap spatial
        // neighbour merge. Leaving the toggle here so users can A/B the
        // perf delta now and we don't have to plumb a new field later.
        bool  gi_restir_enabled = false;
        // How many bounce levels get a sun shadow ray. 0 = no shadow
        // rays in GI (every shadowed bounce uses sky_fill, fastest);
        // 1 = first bounce only (default, balanced); higher = more
        // accurate but linearly more expensive (each level adds N_gi
        // any_hit rays per pixel).
        int   gi_shadow_max_bounce = 1;
        // AO darkening floor. Raw AO is 0..1 (1 = clear, 0 = fully
        // occluded); we remap to [ao_floor, 1.0] so corners where 2-3
        // edges all block rays don't compound into near-black. 0 = AO
        // can crush surfaces to pure black; 0.5 = AO darkens at most
        // 50%; 1.0 = AO has no effect. 0.30 keeps clear contact AO at
        // simple edges while preventing the "triple-darkness corner"
        // artifact.
        float ao_floor = 0.30f;
        // Auto-exposure: scales pre-tonemap HDR by a target/scene-avg
        // ratio so dark interiors lift while bright outdoor pixels seen
        // through a doorway over-expose into bloom. Strength 0 = off,
        // 1 = full eye-adaptation, ~0.5 = stylised (the default).
        float auto_exposure_strength = 0.5f;
        // Physical sky (Hillaire 2020): 1 = atmosphere LUT sky + aerial
        // perspective, 0 = HDRI panorama. intensity scales sky radiance,
        // aerial_scale stretches haze distances (game scale vs real km).
        int   sky_mode = 1;
        float atmosphere_intensity = 5.0f;   // calibrated vs the old HDRI sky brightness
        int   tonemap_mode = 0;   // 0 ACES fitted (tuned look), 1 AgX (Sobotka), 2 Khronos PBR Neutral
        // HDR10 output (BT.2020 + PQ swapchain). Used only when the surface
        // offers A2B10G10R10 + HDR10_ST2084; falls back to SDR otherwise.
        int   hdr_output = 1;
        int   radiance_cache = 1;   // world-space GI radiance cache (SHaRC-style)
        float hdr_paper_nits = 200.0f;   // SDR-white / UI brightness
        float hdr_peak_nits  = 1000.0f;  // display peak (tonemap shoulder)
        float aerial_scale = 6.0f;
        float aerial_max_km = 8.0f;
        // Volumetric clouds (physical sky only).
        int   clouds_enabled = 1;
        float cloud_coverage = 0.45f;
        float cloud_density  = 1.0f;
        float cloud_speed    = 8.0f;   // wind, m/s
        // Froxel volumetric fog (RT-shadowed sun shafts).
        int   fog_volumetric = 1;
        float fog_density = 0.005f;        // 1/m at the reference height
        float fog_height_falloff = 0.035f; // 1/m
        float fog_anisotropy = 0.7f;       // HG g (forward scattering)
        float fog_max_dist = 300.0f;       // m
        // Static-castle BLAS merging: true = one BLAS for all 151 brushes
        // (the merged path baked at level load), false = per-brush TLAS
        // instance pointing at the shared cube BLAS (the original path).
        // Toggle exists so we can A/B autodemo to confirm the merge isn't
        // making per-pixel ray cost worse on a given GPU. Leave on by
        // default once verified.
        bool  use_merged_static_blas = true;

        // ---- Terrain shader knobs (cube.frag, raster tex_params.w==2) ----
        // Atmospheric perspective fog: distant terrain blends toward
        // the view-direction sky tint. 0=off, 1=full.
        float terrain_fog_strength = 1.0f;
        // Half-Lambert wrap: 0 = hard Lambert (back-of-ridge can go
        // pure black with shadow), 1 = full wrap (back gets ~half
        // light from the half-Lambert formula).
        float terrain_wrap_strength = 1.0f;
        // Multiplies the triplanar Ground054 detail × layer-base in the
        // terrain shading branch. >1 punches up texture detail; ~1 is
        // neutral; <1 fades the detail and reads more uniform.
        float terrain_detail_strength = 1.6f;
        // Multiplies the global shadow_softness for terrain pixels only.
        // Lower = tighter shadow cone = less PCSS dither at the cost
        // of harder shadow edges. Useful when shadow_samples is modest
        // and the terrain looks noisy.
        float terrain_shadow_softness_scale = 0.5f;
        // Layer transition heights (smoothstep edges). Tweak per-level
        // if height_scale changes; defaults match the 140m mountains.
        float terrain_h_sand_grass_start = 4.0f;
        float terrain_h_sand_grass_end   = 12.0f;
        float terrain_h_grass_dirt_start = 28.0f;
        float terrain_h_grass_dirt_end   = 42.0f;
        float terrain_h_dirt_rock_start  = 58.0f;
        float terrain_h_dirt_rock_end    = 80.0f;
        float terrain_h_rock_snow_start  = 95.0f;
        float terrain_h_rock_snow_end    = 120.0f;

        // ---- Grass ----
        bool  grass_enabled        = true;
        // When true, the rasterised grass pipeline is skipped and the
        // fullscreen raymarched grass pass takes over. Adapted from
        // shadertoy.com/view/dd2cWh — SDF blades on a domain-repeated
        // grid. Heavier per-pixel cost but no instance-buffer rebuild
        // and looks volumetric/3D rather than flat-billboarded.
        bool  grass_raymarch_enabled = false;
        // Max ray distance for the raymarched grass (metres). Cuts the
        // per-pixel step budget; beyond this, grass fades to sky.
        float grass_raymarch_distance = 150.0f;
        // Max blade-render distance in metres. Beyond this, blades
        // collapse to NaN clip space (no fragment work).
        float grass_distance       = 80.0f;
        // Peak per-blade tip wind sway in metres. The sway is per-blade
        // sin/cos so neighbours move out of phase.
        float grass_wind           = 0.04f;
        // Density: fraction of placed blades to actually draw (0..1).
        // Drives instance count in vkCmdDrawIndexed — cheap perf knob.
        float grass_density        = 0.7f;
        float grass_edge_softness  = 0.6f;   // 0 = hard patch border, 1 = long thinning fringe
        // Placement densities (applied when the world is built: restart).
        float forest_density       = 1.0f;
        float bush_density         = 1.0f;
        float flower_density       = 1.0f;
        // Vegetation LOD bias: >1 keeps higher LODs further out.
        float veg_lod_bias         = 1.0f;
        // Per-blade Y multiplier (0.3..2.0). 1.0 = built-in 0.55 m blades.
        float grass_height_scale   = 1.0f;
        // Far-distance opacity drop for the raymarched grass.
        //   0.0 = far blades stay fully opaque (alpha = 1)
        //   1.0 = far blades go fully transparent (alpha = 0)
        // Close-up blades (within `grass_cutoff_soft_dist` metres) are
        // always alpha = 1 regardless. Also affects the rasterized
        // grass.frag's side-taper threshold (legacy use of the same
        // slider).
        float grass_alpha_cutoff   = 0.4f;
        // Distance at which the user's `grass_alpha_cutoff` value
        // applies in the raymarched grass shader. Inside this radius
        // the slider value is used as-is (soft-edge dither). Past
        // this, the cutoff ramps linearly to ~0.85 by the time the
        // ray reaches `grass_raymarch_distance`, so far blades
        // discard more aggressively and thin out into the terrain
        // tint instead of reading as a uniform green far-mat.
        float grass_cutoff_soft_dist = 30.0f;
        // Slope cutoff: blades whose stored heightmap-normal Y is
        // below this fade their height to 0 (smoothstep window).
        // 1.0 = only perfectly flat ground; 0.55 = up to ~57° slope.
        // 0.85 (default) ≈ 32° max slope.
        float grass_slope_n_min    = 0.85f;
        // Distance-based density falloff (0 = uniform density, 1 = far
        // grass thinned to ~40% of near density). Implemented per-
        // blade in grass.vert via a deterministic rank vs distance-
        // driven threshold with a wide smoothstep so blades don't
        // flicker as the player walks.
        float grass_distance_density = 0.6f;
        // Altitude band: blades fade to 0 height outside [alt_min,
        // alt_max]. Placement covers a generous superset (-10..200m)
        // so the slider can reshape the band live without rebake.
        float grass_alt_min = -2.0f;
        float grass_alt_max = 35.0f;
        // Grass colour palette (linear RGB). Exposed via UI sliders.
        glm::vec3 grass_color_top         = { 0.40f, 0.55f, 0.18f };  // blade tip
        glm::vec3 grass_color_bottom      = { 0.18f, 0.22f, 0.08f };  // blade base
        glm::vec3 grass_color_ground      = { 0.18f, 0.30f, 0.09f };  // close terrain tint
        glm::vec3 grass_color_ground_far  = { 0.13f, 0.22f, 0.06f };  // far terrain tint
        // Distance (m) at which `grass_color_ground_far` fully takes
        // over from `grass_color_ground`. Near ramp scales as 15 % of
        // this so the transition stays smooth at any far-anchor.
        // Used by both grass_raymarch.frag and terrain_raymarch.frag
        // (delivered via the otherwise-unused grass_color_ground_far.w).
        float grass_ground_tint_far_distance = 200.0f;
        // Fake grass-cast shadows on the raymarched terrain.
        // Fake mask-based grass-cast-shadow on raymarched terrain.
        // Disabled by default: it walks the grass mask along the sun
        // XZ at the terrain pixel and darkens — looks like a flat
        // patch tint, not real per-blade shadow. Slider stays so a
        // user who wants the cheap effect can re-enable it.
        float grass_shadow_strength = 0.0f;    // 0 = off
        int   grass_shadow_samples  = 6;       // 0..8 sample taps along sun XZ
        float grass_shadow_max_dist = 3.0f;    // metres of reach along sun direction
        // Grass-cast shadow on the RASTERISED terrain (cube.frag is_terrain
        // path). Walks the grass eligibility mask (binding 13) along the
        // sun XZ direction with a few short steps, accumulating occlusion
        // for ground pixels covered by overhead grass. Cheap (~6 texture
        // taps per terrain pixel, gated on strength > 0). Separate from
        // grass_shadow_strength which only affects the raymarched terrain.
        bool  grass_shadow_on_terrain        = true;   // master toggle
        float grass_shadow_on_terrain_strength = 0.45f; // 0=off, 1=full dark
        int   grass_shadow_on_terrain_samples  = 5;     // 1..8 mask taps
        float grass_shadow_on_terrain_dist     = 1.6f;  // metres along sun XZ
        // Side-lit grass shading. Adds a wrap-Lambert directional term to
        // the grass blades (sunward side brighter, away-side darker).
        //   strength: 0 = flat per-blade colour (engine default), 1 = full
        //             directional contrast. ~0.4 reads as natural without
        //             losing the slider colour gradient.
        bool  grass_side_lit_enabled  = true;
        float grass_side_lit_strength = 0.45f;
        // Strength of the terrain green tint (0..1).
        float grass_ground_tint_strength = 0.85f;
        // Raymarched-grass base AO floor at the blade base. Lower = darker
        // base / more contrast top-to-bottom; higher = flatter / fuller.
        float grass_base_ao_floor = 0.35f;

        // Sun shadow map (binding 7). Two knobs:
        //   shadow_map_resolution: square texture side. Snapped at apply
        //     time to {512, 1024, 2048, 4096}. Higher = sharper edges,
        //     more VRAM + fragment cost. Image is destroyed and recreated
        //     when changed (requires a vkDeviceWaitIdle).
        //   shadow_map_world_half: half-width of the ortho frustum in
        //     world metres. Bigger = far-off ridges still cast on grass
        //     under your feet, but per-texel coverage gets coarser.
        int   shadow_map_resolution  = 1024;
        float shadow_map_world_half  = 120.0f;
        // Debug overlay: draws a wireframe of the shadow ortho frustum
        // and the bake/shadow-map cross-fade boundary on top of the
        // world. Useful for tuning shadow_map_world_half and seeing
        // exactly which casters land inside the shadow map.
        bool  shadow_debug_overlay = false;

        // Terrain debug visualisation modes (cube.frag branch on
        // scene.grass_extra2.w):
        //   0 = full shading (normal)
        //   1 = simple Lambert only (no RT shadow, AO, GI, slope blend, fog)
        //   2 = visualise per-vertex normal as RGB
        //   3 = visualise screen-space face normal as RGB
        int   terrain_debug_mode = 0;

        // Quadtree CDLOD terrain (replaces the chunk-stride LOD +
        // tessellation + densify stack). One tunable: the leaf-node
        // subdivision range. range[level] = clod_range0 * 2^level; a
        // bigger value keeps high-density rings further from the camera
        // (more triangles, more detail).
        float terrain_clod_range0 = 50.0f;     // metres, leaf ring
        // Debug: draw terrain as wireframe so the triangle/mesh density
        // is directly visible.
        bool  terrain_wireframe = false;
        // Near-field detail displacement in terrain.vert (the old
        // tessellation's visual role):
        //   amp  = 0..1 m peak metres of displacement
        //   smooth_mip = 0 = LOD0 crisp, +N = blur by N mip levels
        float terrain_disp_amp        = 0.40f;
        float terrain_disp_smooth_mip = 0.0f;
        // Per-pixel SPOM/POM far-fade distance (cube.frag is_terrain).
        // Bumps stop being raymarched past this distance; far terrain
        // uses just the splat normal. Was hardcoded 220 m.
        float terrain_pom_far_m       = 220.0f;
        // Per-pixel parallax-occlusion on terrain (rocky/eroded depth).
        // Carried to cube.frag via the terrain colour draw's
        // pc.tex_params.x (terrain path ignores that slot otherwise).
        // Gated to rocky/snow pixels + ramped to 0 by ~45 m (built-in
        // per-pixel LOD). 0 = off.
        float terrain_pom_strength   = 0.6f;   // 0..1
        // Shore-following contour ripple scales (cube.frag terrain, via
        // the terrain draw's pc.grass_params.xy). Phase = height above
        // water so the lines parallel the coast. Independent:
        //   sand_ripple_scale : beach ripple frequency
        //   grass_line_scale  : grass contour-row frequency (0 = off)
        float terrain_sand_ripple_scale = 9.0f;   // 1..40
        float terrain_grass_line_scale  = 0.0f;   // 0 = off, else 1..40
        // Procedural ground material splatting (cube.frag is_terrain).
        // Fed to cube.frag via the free slots of scene.spom_params.yzw
        // (no UBO layout change). The 5 baked materials (rock/grass/
        // dirt/sand/snow) are blended by the existing height/slope
        // weights.
        //   strength: 0 = old flat height-band colour (A/B off),
        //             1 = full per-texel baked material. Default high
        //             so the effect is clearly visible.
        //   tile_m:   metres per material texture repeat (smaller =
        //             finer detail, more obvious tiling).
        //   normal:   detail normal-map strength (0 = flat shading,
        //             1 = full baked relief).
        float ground_mat_strength = 0.85f;  // 0..1
        float ground_mat_tile_m   = 2.5f;   // 0.5..12 m
        float ground_mat_normal   = 0.80f;  // 0..1
        // Anti-tile sampling: blend a SECOND albedo/normal sample at a
        // per-pixel rotated UV with the standard sample, weighted by a
        // low-frequency noise field. Breaks up the obvious grid repeat
        // on terrain without the cost of full hex-tile / Heitz-Neyret
        // stochastic sampling (just one extra texture fetch per channel).
        // Off by default while the user A/Bs the look; can be left on
        // permanently if the cost is acceptable. Only affects cube.frag's
        // per-pixel splat path -- vertex displacement in .tese keeps the
        // strict (unrotated) UV so SPOM and tess stay aligned.
        bool  terrain_antitile        = true;
        // Strength of the rotated/offset sample in the blend. 0 = the
        // standard single-sample look, 1 = max anti-tile (and slightly
        // softer micro-detail because the two samples partially average).
        float terrain_antitile_strength = 0.55f;
        // Supersample factor for the heightmap shadow bake. 1 = native
        // (1 texel per 1m heightmap cell). 2 = 4x more texels (sub-
        // metre shadow precision). 4 = 16x more texels (sharpest
        // medium/far shadows). Memory: 1 = ~4MB, 2 = ~17MB, 4 = ~67MB.
        // Bake time scales linearly; worker thread handles it.
        int terrain_bake_supersample = 1;
        // Power exponent on n_dot_l for terrain shading. 1 = standard
        // Lambert (gradient is the natural cosine falloff). >1 sharpens
        // the lit→shadow transition (grazing slopes go darker faster);
        // <1 softens. Visible difference between e.g. 2.0 and 4.0 on
        // mountain slopes.
        float terrain_shading_contrast = 1.0f;
        // SPOM (silhouette parallax occlusion mapping) bump-depth
        // multiplier. Affects the cube.frag SPOM path on castle walls
        // (and any future SPOM-eligible materials). 1.0 = engine
        // default (~4 cm peak-to-trough at 1 m view), 0 = disabled
        // (textures render flat). Floors are gated out of SPOM in
        // height_idx_for_albedo regardless of this value (#198).
        float spom_strength = 1.0f;
        // Wall displacement technique on SPOM-eligible materials (cube.frag,
        // via grass_side_lit_params.z; menu "Wall displacement"):
        //   0 = flat (no parallax)
        //   1 = SPOM - height march + rayQuery seam probe for silhouettes
        //   2 = SSDM - same march, recessed depth written to gl_FragDepth
        //       (no ray query; inward-only relief)
        //   3 = POM with curved silhouette
        int spom_mode = 1;
        // Heightmap resolution multiplier (per-axis). 1 = baseline
        // 2048×2048 cells at 1m. 2 = 4096×4096 at 0.5m (16× more
        // samples total, ~600MB extra mesh memory). 4 = 8192×8192
        // at 0.25m (256× more samples, NOT recommended — pushes past
        // ~2GB of GPU memory on the chunk meshes alone). Applied at
        // level load — change requires a restart.
        int terrain_heightmap_scale = 1;
        // Selectable terrain renderer. false = chunked rasterised mesh
        // (default, matches the gameplay heightmap). true = fullscreen
        // procedural FBM ray-marched terrain (terrain_raymarch.frag,
        // visual-only — physics still uses the rasterised heightmap so
        // castle / cubes won't sit on the procedural surface). The
        // shader is toggled at draw time, no rebuild needed.
        bool terrain_raymarch_enabled = false;
        // Raymarch FPS knobs. Larger = better quality, slower. Defaults
        // are the values the reference document recommends; on weaker
        // GPUs drop both to ~120 / ~32 for a meaningful FPS bump.
        int  terrain_raymarch_max_steps    = 200;   // 60–300
        int  terrain_raymarch_shadow_steps = 64;    // 16–96
        // FBM octaves used by the per-march sampling (terrainM). 9 is
        // the document default; 5–6 is noticeably faster and visually
        // close because each higher octave adds half the previous
        // amplitude (last 3 of 9 contribute ~1.5 % of total).
        int  terrain_raymarch_octaves        = 9;   // 4–24
        // FBM octaves for normal sampling (terrainH). Called 3× per
        // hit pixel for finite-difference normals — biggest single
        // perf knob. 16 is the doc default; 8 is barely-distinguishable.
        int  terrain_raymarch_normal_octaves = 18;  // 4–32
        // Step-factor: ratio of the height-field gap consumed per
        // march iteration. 0.4 is conservative (no overshoot on sharp
        // ridges). 0.6–0.8 finishes the same march in ~half the
        // iterations at the cost of a tiny chance of skipping over a
        // thin spike.
        float terrain_raymarch_step_factor   = 0.4f;
        // Distance-LOD knobs. At ray distance < lod_near_m the FBM uses
        // the full octave count (terrain_raymarch_octaves for march,
        // terrain_raymarch_normal_octaves for normals). Past lod_far_m
        // it falls to lod_min_octaves. Smoothstep ramp in between. Saves
        // big at typical horizon views — sub-pixel FBM detail past
        // ~500 m is below Nyquist anyway.
        float terrain_raymarch_lod_near_m    = 120.0f;  // 20..400
        float terrain_raymarch_lod_far_m     = 600.0f;  // 200..2000
        int   terrain_raymarch_lod_min_octaves = 3;     // 2..8
        // Render-scale for the procedural raymarch ONLY. 1.0 = native
        // (current behaviour). 0.5 = half-res raymarch + bilinear
        // upscale composite (~4× fewer FBM evaluations per frame).
        // Cubes / castle / dyn-props always rasterise at native res.
        float terrain_raymarch_scale         = 1.0f;
        // Trace the PCSS sun shadow at half resolution + bilateral
        // upsample instead of inline per-pixel (big perf win). OFF =
        // the proven inline path, byte-identical.
        bool  half_rate_shadows              = false;
        // Post-upscale sharpen strength applied during the compose
        // pass. Cheap CAS-style 4-neighbour adaptive sharpen — only
        // boosts contrast around real edges, ignores flat / noisy
        // regions. 0 = off, 0.5 = subtle, 1.0 = aggressive.
        float terrain_raymarch_sharpen       = 0.6f;
        // Volumetric ground fog scale (0 = off, 1 = doc baseline,
        // 2 = thick valley fog). Multiplies both density and the
        // sun-scatter contribution so the slider behaves linearly:
        // half = half as visible at every distance.
        float terrain_raymarch_fog_strength  = 0.6f;
        // Phase 6 — relaxation cone-stepping. When true, march step
        // grows with distance (`step *= max(t*0.02, 1.0)`) so we
        // cover the same range in ~half the iterations at the cost
        // of a tiny chance of skipping thin spikes. Doc-recommended
        // for distance-heavy scenes.
        bool  terrain_raymarch_relaxation    = false;
        // Phase 7 — volumetric fog self-shadow. When true, each fog
        // step fires N tiny rays toward the sun + accumulates optical
        // depth so terrain / castle shadow the fog itself, giving a
        // god-ray look. Cheap (4 short rays per fog step), but multi
        // ALU + branches so off by default.
        bool  terrain_raymarch_fog_godrays   = false;
        // Fog band — density is non-zero between [fog_y_start, fog_y_top]
        // and falls off above/below with a soft transition. Combined
        // with a low-frequency noise so the layer isn't a perfect
        // sheet (real ground fog is wispy).
        float terrain_raymarch_fog_y_start   = 0.0f;
        float terrain_raymarch_fog_y_top     = 18.0f;
        float terrain_raymarch_fog_noise     = 0.7f;   // 0 = uniform sheet, 1 = strong wisps

        // Ocean / water plane. Rendered inside the terrain raymarch
        // (fullscreen tri) so it composites with rasterised geometry
        // via gl_FragDepth — same as the terrain itself.
        bool      water_enabled        = false;
        float     water_level          = 2.0f;     // world Y of the calm sea level
        float     water_wave_strength  = 0.06f;    // amplitude of normal perturbation
        // Wave frequency multiplier — multiplies the base scrolling
        // sin/cos frequencies. 1.0 = baseline (~30 m wavelengths).
        // 2.0 = half the wavelength (more, smaller ripples). 0.5 =
        // double the wavelength (bigger sweeping waves).
        float     water_wave_scale     = 1.0f;
        glm::vec3 water_color          = glm::vec3(0.04f, 0.18f, 0.22f);
        // RT reflections on water: when true, the water surface
        // ray-marches the FBM terrain along the reflection vector
        // and uses the shaded hit point (lambert + sky tint) as the
        // reflected colour. When false, a cheap horizon→zenith sky
        // gradient is used instead.
        bool      water_rt_reflections = true;
        // RT TLAS reflection: when true, additionally fire a ray
        // query into the TLAS along the reflect direction so cubes /
        // castle / dyn-props show up in the water surface (not just
        // the FBM terrain). Cheap (1 closest_hit ray per pixel).
        bool      water_tlas_reflections = true;
        // Shore tinting — colour the water lighter where it's shallow.
        // Depth = water_level − terrain_height_at(xz) (in metres).
        glm::vec3 water_color_shallow  = glm::vec3(0.45f, 0.62f, 0.65f);
        // How many metres of water depth the shallow→deep blend
        // covers. 4 m = noticeable beach band; 0.5 m = razor-sharp.
        float     water_shore_blend    = 4.0f;
        // Noise on the shore band so the line isn't a perfect contour.
        // 0 = clean, 1 = strongly broken.
        float     water_shore_noise    = 0.4f;
        // Shore-foam highlight band: a thin lighter tint right at the
        // water/land boundary. Fades exponentially with the noise-FREE
        // depth so the line follows the shoreline geometry exactly.
        // strength 0 = disabled.
        glm::vec3 water_foam_color     = glm::vec3(0.88f, 0.94f, 0.96f);
        float     water_foam_strength  = 0.55f;
        float     water_foam_width     = 0.6f;     // metres of depth
        // Water shading style — picks which normal+foam generator the
        // terrain raymarch's water branch uses.
        //   0 = default (directional sum-of-sines, lake-style)
        //   1 = river (P_Malin "Where the River Goes" port — advected
        //       FBM-with-derivative normal, flow-driven foam pattern.
        //       Flow direction comes from the terrain gradient at the
        //       water plane so the surface reads as if the water is
        //       flowing downhill across the level.)
        int       water_style = 0;
        // River-style knobs (only consumed when water_style >= 1):
        //   speed       multiplies the flow vector derived from the
        //               terrain gradient. 0.3..3.0 typical.
        //   normal_str  scales the river-style normal's XZ component.
        //               Higher = more bumpy surface.
        float     water_river_speed       = 1.0f;
        float     water_river_normal_str  = 1.2f;
        // More river-style knobs (see SceneUBO::water_river_extra).
        float     water_river_flow_angle  = 0.0f;   // degrees; 0 = +X
        float     water_river_time_speed  = 1.0f;
        float     water_river_detail      = 1.0f;   // UV-scale multiplier on 20×
        float     water_river_foam_amount = 1.0f;
        // Underwater extinction colour & density (river style only).
        glm::vec3 water_river_extinct_color  = glm::vec3(0.5f, 0.4f, 0.1f);
        float     water_river_extinct_density = 1.0f;
        // Lake-style knobs (only consumed when water_style == 2).
        // The defining trick is bumps fade with distance — near
        // surface keeps ripples, far surface goes mirror-flat. That's
        // the visual that reads as "lake".
        float water_lake_bump_strength = 0.5f;     // 0..1.5
        float water_lake_time_speed    = 0.4f;     // animation rate
        float water_lake_uv_scale      = 0.6f;     // UV multiplier
        float water_lake_bump_dist     = 60.0f;    // metres where bumps fade out
        // Shoreline grass tint — blades within `grass_shore_distance`
        // metres above water level fade toward `grass_shore_color`.
        // Strength 0 disables. Cheap (~13 ALU/pixel in
        // grass_raymarch.frag, no extra texture taps).
        glm::vec3 grass_shore_color    = glm::vec3(0.40f, 0.30f, 0.16f);
        float     grass_shore_strength = 0.65f;
        float     grass_shore_distance = 3.5f;     // metres above water
        // Shoreline TERRAIN tint — same shape, applied to bare ground
        // in terrain_raymarch.frag's getMaterial(). Mixed on top of the
        // existing snow/sand/grass tint pass so the beach reads as a
        // user-tunable colour band instead of relying on slope alone.
        glm::vec3 terrain_shore_color    = glm::vec3(0.55f, 0.46f, 0.30f);
        float     terrain_shore_strength = 0.55f;
        float     terrain_shore_distance = 3.0f;
        // Distance fog — standard exp² atmospheric fog. Disabled by
        // default (strength = 0). Applied at end of cube.frag /
        // terrain_raymarch.frag / grass_raymarch.frag shading.
        // Defaults tuned for "absolutely realistic" atmospheric perspective
        // on cube.frag's mesh-terrain path. Was strength = 0 (off) which
        // meant the user saw no fog at all by default. New defaults give
        // a soft cool-blue atmospheric haze that reaches ~50% at ~700 m
        // and saturates by ~1.5 km, with the height attenuation pulling
        // the layer off the mountain peaks. Tunable per scene via the
        // sliders in the menu (Quality > Fog).
        glm::vec3 distance_fog_color    = glm::vec3(0.62f, 0.70f, 0.78f);
        float     distance_fog_strength = 0.85f;    // master on
        float     distance_fog_density  = 0.0015f;  // per metre (exp^2 falloff)
        float     distance_fog_start    = 80.0f;    // metres before fog kicks in
        float     distance_fog_height   = 220.0f;   // height-falloff top (0 = uniform fog)
        float     distance_fog_max      = 0.92f;    // clamp on mix weight
        // GENERAL terrain shore tint — applied to BARE terrain near
        // water (no grass). Companion to terrain_shore_* (grass-area).
        glm::vec3 terrain_shore_general_color    = glm::vec3(0.50f, 0.42f, 0.30f);
        float     terrain_shore_general_strength = 0.55f;
        float     terrain_shore_general_distance = 3.0f;
        // Bare-shore SAND base colour — the underlying material that
        // the slope-driven beachMask reveals near water level on the
        // raymarched terrain. Was hardcoded; now slider-tunable.
        glm::vec3 terrain_sand_color = glm::vec3(0.50f, 0.45f, 0.35f);
        // Shadow casting on water: when true, the water surface
        // dims its specular and base tint where the sun shadow map
        // says it's occluded by terrain / castle / dyn-props.
        bool      water_shadows_enabled = true;
        // 0 = fully opaque (water_color is the colour you see), 1 =
        // shallow water shows the underwater terrain through it
        // (linearly attenuated by depth so deep water still hides).
        float     water_transparency    = 0.6f;
        // Depth (m) at which water becomes ~opaque. Decoupled from
        // transparency: small = only very shallow water is see-through
        // (deep stays opaque), large = clear far down. Fed to water.frag
        // via the water-plane draw's pc.color.y (no UBO-layout change).
        float     water_clarity_depth   = 2.5f;
        // Shoreline softness: metres of water DEPTH over which the
        // water fades in from the land edge. Larger = a longer, softer
        // wet-sand → water transition (and hides the heightmap-vs-
        // rasterised waterline jitter). Fed via the water draw's
        // pc.color.z.
        float     water_shore_softness  = 0.6f;
        // Foam opacity vs the (see-through) water it sits on. 0 = foam
        // is as transparent as clear shallow water, 1 = foam fully
        // opaque. Fed via the water draw's pc.color.w.
        float     water_foam_opacity    = 0.45f;
        // Multiplier on the RT shadow ray count for fragments within
        // rt_lod.x (close to the camera). 1 = unchanged, higher = more
        // rays close-by for smoother penumbras under boxes / castle.
        // Costs extra rays only on the closest pixels; distance LOD
        // continues to drop samples toward 1 ray past rt_lod.y.
        float shadow_near_mult = 1.0f;

        float gi_strength = 1.0f;
        float gi_radius   = 60.0f;
        // Spatial smoothing factor on the raymarched-terrain sky-bounce GI.
        // 0 = raw per-pixel GI from the sample loop (noisiest, most accurate).
        // 1 = collapse the GI estimate to the smooth sky-at-normal sample
        //     (no per-pixel noise but loses occluder-aware GI variation).
        // Useful when gi_samples is too low to converge on its own.
        float gi_softener = 0.5f;
        // Debug viz: when true, the raymarched-terrain shader replaces its
        // final colour with the gi_indirect term only (scaled up so it's
        // clearly visible). Lets you confirm the path-traced GI loop is
        // firing and producing meaningful contribution at this pixel.
        bool  gi_debug_viz = false;
        // Power curve on the raymarched-terrain AO. 1.0 = linear (current
        // behaviour). > 1 punches corners darker (pow makes mid AO drop
        // faster than linear). < 1 flattens. Modulates BOTH the analytical
        // ambient term AND the GI smooth fallback so cavities go genuinely
        // dark without depending solely on the ao_floor knob.
        float terrain_ao_punch = 1.5f;
        // Per-pixel RT caps for the raymarched terrain shader. Lower
        // them when the GPU TDRs; raise for visual fidelity. Defaults
        // chosen to keep total per-frame RT work under the Windows TDR
        // threshold even with the heaviest scene (kMaxDynProps=60).
        int   terrain_pcss_samples_cap   = 4;     // max PCSS rays per pixel
        int   terrain_gi_samples_cap     = 4;     // max GI primary rays per pixel
        int   terrain_gi_bounces_cap     = 1;     // max GI bounces per ray
        float terrain_ao_final_strength  = 0.45f; // 0=no final-colour AO, 1=full
        float terrain_rm_contrast        = 1.25f; // raymarched terrain: 1 = neutral
        // Specular reflection on flagged surfaces (the pedestal).
        bool  reflections_enabled = true;
        float reflection_strength = 0.5f;

        // Temporal + spatial denoiser.
        float taa_history_blend = 0.3f;   // 0 = no temporal, 1 = full history
        // Lowered from 0.95: with TAA jitter off (taa_jitter_enabled=0) the
        // history gives no AA benefit, only RT-noise smoothing when still -
        // and a heavy blend smears every moving feature, worst on the
        // ZERO-motion-vector ones (camera-attached viewmodel; shadows/GI on
        // static floor pixels). 0.3 keeps light still-frame denoising while
        // cutting the ghost trail dramatically.
        // Temporal upsample (TAAU). When true, a second temporal pass runs
        // after TAA: samples LR history bilinearly, reprojects a native-
        // resolution prev frame, neighborhood-clamps to kill ghosts.
        // Compose then samples the native upscaled image. Off by default
        // so the standard render-scale=1 path stays bit-identical.
        bool  taau_enabled       = false;
        // ---- FSR2 (full temporal upscaler) ----
        // Phased rollout — when enabled, swaps the engine's TAA / TAAU
        // path for an FSR2-style temporal accumulator. Phase 1 (this
        // session) lays the foundation: Halton(2,3) jitter, motion-vec
        // format conversion, texture mip-bias, stub C++ module.
        // Phases 2-5 add the compute passes (Prepare / DepthClip /
        // ReconstructPrevDepth / Lock / Accumulate / RCAS).
        // Default off until the implementation is feature-complete.
        bool  fsr2_enabled       = false;
        // FSR upscaler backend select: 0 = legacy FSR2 path
        // (external/FidelityFX-FSR2 static lib), 1 = FSR3 via ffx-api
        // (external/FidelityFX-SDK prebuilt DLL). Only consumed when
        // fsr2_enabled is on. See docs/fsr3_plan.md for the multi-
        // session arc.
        int   fsr_backend        = 0;
        // DLSS Ray Reconstruction (fsr_backend == 2 only): the AI model
        // denoises AND upscales the ray-traced frame from material guide
        // buffers, replacing DLSS Super Resolution. Lets the RT effects
        // run at low sample counts (dlss_rr_low_samples).
        bool  dlss_rr            = false;
        // With RR active, trace shadows / AO / GI at ~1 sample per pixel
        // and let RR resolve the noise (the speed side of RR).
        bool  dlss_rr_low_samples = true;
        // FSR3 Frame Generation. Only meaningful when fsr_backend == 1.
        // Session 3 dispatches FG to a private output image without
        // presenting it — sessions 4-6 wire it into the swapchain
        // proxy + UI composition. Default off; user must opt in.
        bool  fg_enabled         = false;
        float taa_spatial_strength = 0.7f;  // 0 = no à-trous, 1 = fully filtered

        // Bloom (single-pass spiral-tap inside the compose shader).
        bool  bloom_enabled = true;
        float bloom_strength = 0.6f;
        float bloom_threshold = 1.05f;  // luminance above which a pixel "blooms"
        float bloom_radius = 24.0f;     // largest tap offset in pixels

        // Spark-trail emissive multiplier — applied per-trail-segment in
        // draw_spark_trails. The sparks already feed bloom because they're
        // drawn as emissive cylinders; cranking this scales their HDR
        // contribution which directly drives a bigger bloom halo. Direct
        // brightness clips to ~white under ACES, so the visible difference
        // is mostly in the bloom halo width/intensity. 1.0 = current look.
        float spark_bloom = 1.0f;       // 0..5 sane range

        // Screen-space lens flare (Chapman 2017): reflects bright sources
        // around the screen center, samples ghosts along the light axis,
        // adds a halo ring + chromatic aberration. Runs in the compose pass.
        bool  lens_flare_enabled    = true;
        float lens_flare_strength   = 0.5f;
        float lens_flare_threshold  = 1.4f;   // HDR luminance cutoff
        float lens_flare_dispersal  = 0.30f;  // ghost spacing along axis
        float lens_flare_halo_width = 0.40f;  // halo offset toward center
        int   lens_flare_ghosts     = 3;      // 1..8 (3 = cinematic default)
        float lens_flare_aberration = 0.005f; // chromatic-fringe offset

        // Master textures on/off. When disabled, all texture lookups in the
        // raster + RT paths fall back to per-instance solid colors. Useful
        // for comparing texture cost vs. baseline and for the "looks like
        // the prototype" aesthetic.
        bool  textures_enabled = true;

        // Sub-pixel Halton jitter on the projection matrix → TAA accumulates
        // sub-pixel samples → supersampled/SSAA-quality edges over ~16 frames.
        // Default OFF: without per-pixel motion vectors, jitter inherently
        // causes mild edge-flicker that variance clipping can only mostly
        // hide. Toggle on if you want the smoother-edge supersampled look
        // and accept some shimmer in exchange.
        bool  taa_jitter_enabled = false;
        float taa_jitter_strength = 0.5f;

        // Post-TAA unsharp mask in compose.frag. Counteracts the soft look
        // taa.frag's à-trous spatial filter introduces. 0 disables; 0.55 is
        // a reasonable default that recovers most lost detail without
        // ringing on edges. >1 looks deliberately punchy / over-sharpened.
        float compose_sharpen_strength = 0.55f;
        // Screen-space sun shafts (crepuscular rays) in compose.frag.
        // Radial accumulation from each pixel toward the sun's screen
        // position, depth-gated to only contribute from sky pixels —
        // produces visible god-rays whenever geometry silhouettes the
        // sun direction. 0 = off; 0.6 = subtle default; 1.5 = heavy.
        // Cheap (24 fetches per pixel, sky-gated early-out per tap).
        float sun_shaft_intensity = 0.6f;
        // Final-image contrast applied in compose.frag right after the
        // ACES tonemap, before sRGB encode. 1.0 = neutral, > 1 punches
        // up midtones, < 1 flattens them. Sane range 0.5..2.0.
        float image_contrast = 1.0f;
        // Final-image brightness multiplier (post-tonemap, pre-sRGB).
        // 1.0 = neutral; range 0.3..2.5.
        float image_brightness = 1.0f;
        // Output gamma curve (post-tonemap, pre-sRGB encode). 1.0 =
        // pass-through. > 1 lifts midtones (lighter look); < 1
        // darkens. Range 0.4..2.5.
        float image_gamma = 1.0f;
        // Colour grading (compose.frag, after the tonemap).
        float grade_temperature = 0.06f;   // -1 cool .. +1 warm
        float grade_saturation  = 1.06f;   // 0 grey .. 2
        float grade_vibrance    = 0.25f;   // boosts muted colours only
        float grade_split_tone  = 0.25f;   // teal shadows / warm highlights
        float film_grain        = 0.12f;
        float vignette_strength = 0.30f;
        // Raymarched-terrain RT (shadows / AO / GI) cutoff distance.
        // Effect fades smoothly from `* 0.6` to this value, then turns
        // off entirely past it. 250 m default keeps gameplay-range
        // pixels fully RT-shaded; lower if you need perf headroom.
        float terrain_rt_lod_distance = 250.0f;

        // Render resolution scale: 0.5 = render at half-res then upscale,
        // 1.0 = native, 2.0 = render at 2× then downscale (SSAA). The scene
        // / TAA / bloom passes all run at this resolution; only the final
        // compose targets the swapchain. Single biggest perf knob in the
        // engine — fragment-shader cost (which dominates here) scales with
        // pixel count, so 0.75 ≈ 1.78× faster, 1.5 ≈ 2.25× slower.
        // Reapplying requires re-creating render targets (recreate_swapchain
        // does that automatically; the UI's Apply button forces it).
        float render_scale = 1.0f;

        // Quality preset selector. -1 = custom (sliders won the last edit);
        // 0..3 = Low / Medium / High / Ultra. Switching to a non-custom
        // preset writes a bundle of values onto the other RtSettings fields
        // (shadow_samples, gi_samples, render_scale, ...) so users don't
        // have to tune individually. See apply_quality_preset().
        int quality_preset = 2;  // High — matches existing defaults

        // AO mode. 0 = off, 1 = GTAO (screen-space, fast), 2 = RTAO (true
        // ray-traced, slower but handles off-screen occluders). GTAO is the
        // production default; RTAO is the "high-quality" preset choice.
        int ao_mode = 1;

        // Member-wise equality, ignoring padding bytes — required because
        // memcmp on this struct caught uninitialised padding as "changes" and
        // triggered spurious auto-saves every frame.
        bool operator==(const RtSettings&) const = default;
};

// Gameplay-tuning settings. Kept separate from RtSettings so the ImGui
// menu can place them in their own section instead of under graphics.
struct GameSettings {
        // Gravity in m/s² (downward). Default = 25 (~2.5 G, game-feel).
        // Slider in the menu goes up to 50× that for absurd-fall experiments.
        float gravity = 25.0f;
        // Continuous box spawner rate. Engine converts to seconds-per-spawn.
        int   cubes_per_minute = 150;
        // Bullet mass in kg. Higher = more impulse delivered to boxes on hit.
        // 10 kg @ 220 m/s = 2200 kg·m/s, ~17 m/s push on a 125 kg box.
        float bullet_mass = 10.0f;
        // Auto-fire rate while LMB is held, in rounds per second. 0 = single
        // shot per click (legacy hitscan behaviour).
        float fire_rate_rps = 6.0f;
        // Bullet exit velocity in m/s. Higher = more impulse delivered on
        // impact (since projectile mass × velocity = momentum), but Jolt's
        // LinearCast CCD has practical limits — anything above ~600 m/s with
        // 1cm-class targets starts to behave weirdly.
        float bullet_speed = 220.0f;
        // Scales spark count + initial speed at spawn. 1.0 = default. Use
        // smaller values for subtle effects, larger for explosions.
        float spark_scale   = 1.0f;

        bool operator==(const GameSettings&) const = default;
};

} // namespace qlike
