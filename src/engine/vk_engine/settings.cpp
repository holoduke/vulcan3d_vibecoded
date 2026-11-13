// Settings persistence — debounced auto-save / load / clamp. Reads
// qlike_settings.cfg in the working directory; ignored if missing (defaults
// from RtSettings / GameSettings apply). Format is one `key = value` per
// line; vec3 fields use space-separated triples.

#include "engine/vk_engine/internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace qlike {

// Quick scan of qlike_settings.cfg for the raymarch flag — load_settings()
// runs AFTER init_world() (it restores the saved player pose, which init_world
// would otherwise overwrite), but init_world needs to know whether to use
// the FBM-matching heightmap source up-front. So we peek just this one key
// without applying anything else.
void VulkanEngine::preload_terrain_raymarch_flag() {
    std::ifstream f(kSettingsPath);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        auto trim = [](std::string& s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.pop_back();
        };
        trim(key); trim(val);
        if (key == "terrain_raymarch_enabled") {
            rt_.terrain_raymarch_enabled = (std::atoi(val.c_str()) != 0);
            log::infof("preload: terrain_raymarch_enabled = %d",
                       rt_.terrain_raymarch_enabled ? 1 : 0);
            return;
        }
    }
}

// reset_player lives here too — it's the menu's "Reset" action, conceptually
// adjacent to the settings save/load flow.
void VulkanEngine::reset_player() {
    player_ = game::Player{};
    // Lift spawn to terrain plateau height + a little headroom so the
    // player doesn't fall through the ground on startup.
    if (!terrain_data_.heights.empty()) {
        player_.position.y += 16.0f;
    }
    log::info("player reset to spawn");
}

// Quality presets bundle render-scale + sample counts + soft-shadow + AO mode
// into one toggle. Indices match the UI combo: 0=Low, 1=Medium, 2=High,
// 3=Ultra. The user can still nudge individual sliders afterward; we set
// quality_preset = -1 (custom) when that happens (handled in vk_ui.cpp).
//
// Numbers are tuned for an RTX 4080 at 1280×720 — Low ≈ 4–5× the FPS of
// Ultra in this scene's overdraw profile.
void VulkanEngine::apply_quality_preset(int level) {
    rt_.quality_preset = level;
    const float prev_tr_scale = rt_.terrain_raymarch_scale;
    switch (level) {
    case 0: // Low
        rt_.render_scale       = 0.65f;
        rt_.shadow_samples     = 8;
        rt_.shadow_softness    = 0.010f;
        rt_.ao_mode            = 1;     // GTAO
        rt_.ao_samples         = 2;     // unused by GTAO but keeps RTAO sane if user toggles
        rt_.gi_samples         = 16;
        rt_.gi_bounces         = 1;
        rt_.reflections_enabled = false;
        rt_.bloom_radius       = 16.0f;
        // Raymarch terrain — aggressive perf knobs. Half-res FBM + low
        // octaves + steep LOD ramp gets the worst-case horizon view
        // running on modest hardware.
        rt_.terrain_raymarch_scale         = 0.5f;
        rt_.terrain_raymarch_max_steps     = 120;
        rt_.terrain_raymarch_shadow_steps  = 32;
        rt_.terrain_raymarch_octaves       = 6;
        rt_.terrain_raymarch_normal_octaves = 10;
        rt_.terrain_raymarch_step_factor   = 0.7f;
        rt_.terrain_raymarch_lod_near_m    = 60.0f;
        rt_.terrain_raymarch_lod_far_m     = 300.0f;
        rt_.terrain_raymarch_lod_min_octaves = 2;
        break;
    case 1: // Medium
        rt_.render_scale       = 0.85f;
        rt_.shadow_samples     = 16;
        rt_.shadow_softness    = 0.014f;
        rt_.ao_mode            = 1;     // GTAO
        rt_.ao_samples         = 4;
        rt_.gi_samples         = 32;
        rt_.gi_bounces         = 1;
        rt_.reflections_enabled = true;
        rt_.bloom_radius       = 20.0f;
        rt_.terrain_raymarch_scale         = 0.75f;
        rt_.terrain_raymarch_max_steps     = 160;
        rt_.terrain_raymarch_shadow_steps  = 48;
        rt_.terrain_raymarch_octaves       = 8;
        rt_.terrain_raymarch_normal_octaves = 14;
        rt_.terrain_raymarch_step_factor   = 0.5f;
        rt_.terrain_raymarch_lod_near_m    = 100.0f;
        rt_.terrain_raymarch_lod_far_m     = 500.0f;
        rt_.terrain_raymarch_lod_min_octaves = 3;
        break;
    case 2: // High
        rt_.render_scale       = 1.0f;
        rt_.shadow_samples     = 24;
        rt_.shadow_softness    = 0.016f;
        rt_.ao_mode            = 1;     // GTAO (still cheaper than RTAO)
        rt_.ao_samples         = 4;
        rt_.gi_samples         = 64;
        rt_.gi_bounces         = 1;
        rt_.reflections_enabled = true;
        rt_.bloom_radius       = 24.0f;
        rt_.terrain_raymarch_scale         = 1.0f;
        rt_.terrain_raymarch_max_steps     = 200;
        rt_.terrain_raymarch_shadow_steps  = 64;
        rt_.terrain_raymarch_octaves       = 9;
        rt_.terrain_raymarch_normal_octaves = 18;
        rt_.terrain_raymarch_step_factor   = 0.4f;
        rt_.terrain_raymarch_lod_near_m    = 120.0f;
        rt_.terrain_raymarch_lod_far_m     = 600.0f;
        rt_.terrain_raymarch_lod_min_octaves = 3;
        break;
    case 3: // Ultra
        rt_.render_scale       = 1.25f;  // 1.5625× pixel count → SSAA-ish
        rt_.shadow_samples     = 40;
        rt_.shadow_softness    = 0.018f;
        rt_.ao_mode            = 2;     // RTAO — true off-screen occlusion
        rt_.ao_samples         = 6;
        rt_.gi_samples         = 96;
        rt_.gi_bounces         = 2;
        rt_.reflections_enabled = true;
        rt_.bloom_radius       = 28.0f;
        rt_.terrain_raymarch_scale         = 1.0f;
        rt_.terrain_raymarch_max_steps     = 250;
        rt_.terrain_raymarch_shadow_steps  = 80;
        rt_.terrain_raymarch_octaves       = 12;
        rt_.terrain_raymarch_normal_octaves = 24;
        rt_.terrain_raymarch_step_factor   = 0.4f;
        rt_.terrain_raymarch_lod_near_m    = 200.0f;
        rt_.terrain_raymarch_lod_far_m     = 1000.0f;
        rt_.terrain_raymarch_lod_min_octaves = 5;
        break;
    default:
        rt_.quality_preset = -1; // unknown → mark custom
        return;
    }
    // Raymarch LR targets are sized at render_extent_ × terrain_raymarch_scale.
    // The preset change above may have flipped that ratio; recreate the
    // attachments so the next frame's LR pass writes a valid extent.
    if (std::abs(rt_.terrain_raymarch_scale - prev_tr_scale) > 0.01f &&
        tr_lr_color_image_) {
        recreate_terrain_raymarch_lowres();
    }
    log::infof("quality preset %d applied (render_scale=%.2f shadows=%d ao=%d gi=%d)",
               level, rt_.render_scale, rt_.shadow_samples,
               rt_.ao_mode, rt_.gi_samples);
}

// Re-allocate render targets when render_scale changes. The hard work is
// already in recreate_swapchain — wait-idle, destroy targets, re-init at
// the new resolution. Don't touch the swapchain itself; only the targets
// that depend on render_extent_.
void VulkanEngine::apply_render_scale() {
    // DLSS accepts inputs from 50 % to 100 % of the output size (measured:
    // 0.49 and below fail every evaluate with NGX InvalidParameter, for
    // super resolution and Ray Reconstruction alike, leaving a stale image).
    if (rt_.fsr2_enabled && rt_.fsr_backend == 2 && dlss_supported_ && rt_.render_scale < 0.5f) {
        log::infof("render_scale %.2f is below the DLSS minimum - using 0.50", rt_.render_scale);
        rt_.render_scale = 0.5f;
    }
    log::infof("render_scale → %.2f, re-allocating targets", rt_.render_scale);
    recreate_swapchain();
}

namespace {
std::string trim(std::string s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
} // namespace

void VulkanEngine::maybe_autosave_settings(float frame_dt) {
    constexpr float kDebounceSec = 0.5f;
    bool changed_this_frame = !(rt_ == rt_prev_frame_) ||
                              !(game_ == game_prev_frame_);
    // The terrain brush settings live outside rt_/game_; without this a
    // brush tweak was only written at a clean exit.
    const auto brush_now = std::make_tuple(
        terrain_brush_radius_, terrain_brush_strength_, int(terrain_brush_mode_),
        terrain_brush_flatten_target_, terrain_brush_noise_strength_, terrain_brush_noise_freq_,
        terrain_brush_use_fbm_erosion_, terrain_brush_fbm_octaves_);
    static auto brush_prev = brush_now;
    static auto brush_saved = brush_now;
    changed_this_frame = changed_this_frame || brush_now != brush_prev;
    bool dirty_vs_disk      = !(rt_ == rt_last_saved_) ||
                              !(game_ == game_last_saved_) || brush_now != brush_saved;
    brush_prev = brush_now;
    rt_prev_frame_   = rt_;
    game_prev_frame_ = game_;

    if (changed_this_frame) {
        // User just moved a slider → restart the debounce window.
        settings_dirty_timer_ = 0.0f;
        return;
    }
    if (dirty_vs_disk) {
        settings_dirty_timer_ += frame_dt;
        if (settings_dirty_timer_ >= kDebounceSec) {
            save_settings();
            brush_saved = brush_now;
            rt_last_saved_   = rt_;
            game_last_saved_ = game_;
            settings_dirty_timer_ = 0.0f;
        }
    } else {
        settings_dirty_timer_ = 0.0f;
    }
}

// ---------------------------------------------------------------------
// Settings registry: the ONE list of persisted keys. Save writes them in
// this order; load parses each by the member's type (float / int / bool /
// vec3 / enum). Adding a setting = one row here (+ a clamp below if it
// needs a range). Ranges and cross-field fix-ups live in load_settings'
// clamp block, so a hand-edited cfg cannot push values out of range.
// `Self` is VulkanEngine or const VulkanEngine (save is const).
// ---------------------------------------------------------------------
template <class Self, class F>
void VulkanEngine::visit_settings(Self& s, F&& f) {
    f("sun_pitch_deg", s.rt_.sun_pitch_deg);
    f("sun_yaw_deg", s.rt_.sun_yaw_deg);
    f("sun_intensity", s.rt_.sun_intensity);
    f("sun_color", s.rt_.sun_color);
    f("sky_color", s.rt_.sky_color);
    f("ground_ambient", s.rt_.ground_ambient);
    f("ambient_strength", s.rt_.ambient_strength);
    f("shadow_strength", s.rt_.shadow_strength);
    f("shadow_enabled", s.rt_.shadow_enabled);
    f("shadow_samples", s.rt_.shadow_samples);
    f("shadow_softness", s.rt_.shadow_softness);
    f("shadow_curve", s.rt_.shadow_curve);
    f("ao_samples", s.rt_.ao_samples);
    f("ao_radius", s.rt_.ao_radius);
    f("gi_samples", s.rt_.gi_samples);
    f("gi_bounces", s.rt_.gi_bounces);
    f("gi_restir_enabled", s.rt_.gi_restir_enabled);
    f("gi_shadow_max_bounce", s.rt_.gi_shadow_max_bounce);
    f("ao_floor", s.rt_.ao_floor);
    f("auto_exposure_strength", s.rt_.auto_exposure_strength);
    f("sky_mode", s.rt_.sky_mode);
    f("atmosphere_intensity", s.rt_.atmosphere_intensity);
    f("aerial_scale", s.rt_.aerial_scale);
    f("aerial_max_km", s.rt_.aerial_max_km);
    f("clouds_enabled", s.rt_.clouds_enabled);
    f("cloud_coverage", s.rt_.cloud_coverage);
    f("cloud_density", s.rt_.cloud_density);
    f("cloud_speed", s.rt_.cloud_speed);
    f("tonemap_mode", s.rt_.tonemap_mode);
    f("hdr_output", s.rt_.hdr_output);
    f("radiance_cache", s.rt_.radiance_cache);
    f("hdr_paper_nits", s.rt_.hdr_paper_nits);
    f("hdr_peak_nits", s.rt_.hdr_peak_nits);
    f("fog_volumetric", s.rt_.fog_volumetric);
    f("fog_density", s.rt_.fog_density);
    f("fog_height_falloff", s.rt_.fog_height_falloff);
    f("fog_anisotropy", s.rt_.fog_anisotropy);
    f("fog_max_dist", s.rt_.fog_max_dist);
    f("use_merged_static_blas", s.rt_.use_merged_static_blas);
    f("terrain_brush_radius", s.terrain_brush_radius_);
    f("terrain_brush_strength", s.terrain_brush_strength_);
    f("terrain_brush_mode", s.terrain_brush_mode_);
    f("terrain_brush_flatten_target", s.terrain_brush_flatten_target_);
    f("terrain_brush_noise_strength", s.terrain_brush_noise_strength_);
    f("terrain_brush_noise_freq", s.terrain_brush_noise_freq_);
    f("terrain_brush_use_fbm_erosion", s.terrain_brush_use_fbm_erosion_);
    f("terrain_brush_fbm_octaves", s.terrain_brush_fbm_octaves_);
    f("terrain_fog_strength", s.rt_.terrain_fog_strength);
    f("terrain_wrap_strength", s.rt_.terrain_wrap_strength);
    f("terrain_detail_strength", s.rt_.terrain_detail_strength);
    f("terrain_shadow_softness_scale", s.rt_.terrain_shadow_softness_scale);
    f("terrain_h_sand_grass_start", s.rt_.terrain_h_sand_grass_start);
    f("terrain_h_sand_grass_end", s.rt_.terrain_h_sand_grass_end);
    f("terrain_h_grass_dirt_start", s.rt_.terrain_h_grass_dirt_start);
    f("terrain_h_grass_dirt_end", s.rt_.terrain_h_grass_dirt_end);
    f("terrain_h_dirt_rock_start", s.rt_.terrain_h_dirt_rock_start);
    f("terrain_h_dirt_rock_end", s.rt_.terrain_h_dirt_rock_end);
    f("terrain_h_rock_snow_start", s.rt_.terrain_h_rock_snow_start);
    f("terrain_h_rock_snow_end", s.rt_.terrain_h_rock_snow_end);
    f("grass_enabled", s.rt_.grass_enabled);
    f("grass_raymarch_enabled", s.rt_.grass_raymarch_enabled);
    f("grass_raymarch_distance", s.rt_.grass_raymarch_distance);
    f("grass_distance", s.rt_.grass_distance);
    f("grass_wind", s.rt_.grass_wind);
    f("grass_density", s.rt_.grass_density);
    f("grass_edge_softness", s.rt_.grass_edge_softness);
    f("grass_near_boost", s.rt_.grass_near_boost);
    f("forest_density", s.rt_.forest_density);
    f("bush_density", s.rt_.bush_density);
    f("flower_density", s.rt_.flower_density);
    f("ground_density", s.rt_.ground_density);
    f("veg_lod_bias", s.rt_.veg_lod_bias);
    f("flower_draw_distance", s.rt_.flower_draw_distance);
    f("ground_draw_distance", s.rt_.ground_draw_distance);
    f("reed_draw_distance", s.rt_.reed_draw_distance);
    f("grass_height_scale", s.rt_.grass_height_scale);
    f("grass_alpha_cutoff", s.rt_.grass_alpha_cutoff);
    f("grass_cutoff_soft_dist", s.rt_.grass_cutoff_soft_dist);
    f("grass_slope_n_min", s.rt_.grass_slope_n_min);
    f("grass_distance_density", s.rt_.grass_distance_density);
    f("grass_alt_min", s.rt_.grass_alt_min);
    f("grass_alt_max", s.rt_.grass_alt_max);
    f("grass_color_top", s.rt_.grass_color_top);
    f("grass_color_bottom", s.rt_.grass_color_bottom);
    f("grass_color_ground", s.rt_.grass_color_ground);
    f("grass_color_ground_far", s.rt_.grass_color_ground_far);
    f("grass_ground_tint_far_distance", s.rt_.grass_ground_tint_far_distance);
    f("grass_base_ao_floor", s.rt_.grass_base_ao_floor);
    f("grass_ground_tint_strength", s.rt_.grass_ground_tint_strength);
    f("grass_shadow_strength", s.rt_.grass_shadow_strength);
    f("grass_shadow_samples", s.rt_.grass_shadow_samples);
    f("grass_shadow_max_dist", s.rt_.grass_shadow_max_dist);
    f("shadow_map_resolution", s.rt_.shadow_map_resolution);
    f("shadow_map_world_half", s.rt_.shadow_map_world_half);
    f("shadow_debug_overlay", s.rt_.shadow_debug_overlay);
    f("terrain_clod_range0", s.rt_.terrain_clod_range0);
    f("terrain_wireframe", s.rt_.terrain_wireframe);
    f("terrain_pom_strength", s.rt_.terrain_pom_strength);
    f("terrain_disp_amp", s.rt_.terrain_disp_amp);
    f("terrain_disp_smooth_mip", s.rt_.terrain_disp_smooth_mip);
    f("terrain_pom_far_m", s.rt_.terrain_pom_far_m);
    f("terrain_sand_ripple_scale", s.rt_.terrain_sand_ripple_scale);
    f("terrain_grass_line_scale", s.rt_.terrain_grass_line_scale);
    f("ground_mat_strength", s.rt_.ground_mat_strength);
    f("ground_mat_tile_m", s.rt_.ground_mat_tile_m);
    f("ground_mat_normal", s.rt_.ground_mat_normal);
    f("terrain_antitile", s.rt_.terrain_antitile);
    f("terrain_antitile_strength", s.rt_.terrain_antitile_strength);
    f("grass_shadow_on_terrain", s.rt_.grass_shadow_on_terrain);
    f("grass_shadow_on_terrain_strength", s.rt_.grass_shadow_on_terrain_strength);
    f("grass_shadow_on_terrain_samples", s.rt_.grass_shadow_on_terrain_samples);
    f("grass_shadow_on_terrain_dist", s.rt_.grass_shadow_on_terrain_dist);
    f("grass_side_lit_enabled", s.rt_.grass_side_lit_enabled);
    f("grass_side_lit_strength", s.rt_.grass_side_lit_strength);
    f("terrain_bake_supersample", s.rt_.terrain_bake_supersample);
    f("terrain_shading_contrast", s.rt_.terrain_shading_contrast);
    f("spom_strength", s.rt_.spom_strength);
    f("spom_mode", s.rt_.spom_mode);
    f("terrain_heightmap_scale", s.rt_.terrain_heightmap_scale);
    f("terrain_raymarch_enabled", s.rt_.terrain_raymarch_enabled);
    f("terrain_raymarch_max_steps", s.rt_.terrain_raymarch_max_steps);
    f("terrain_raymarch_shadow_steps", s.rt_.terrain_raymarch_shadow_steps);
    f("terrain_raymarch_octaves", s.rt_.terrain_raymarch_octaves);
    f("terrain_raymarch_normal_octaves", s.rt_.terrain_raymarch_normal_octaves);
    f("taau_enabled", s.rt_.taau_enabled);
    f("fsr2_enabled", s.rt_.fsr2_enabled);
    f("fsr_backend", s.rt_.fsr_backend);
    f("dlss_rr", s.rt_.dlss_rr);
    f("dlss_rr_low_samples", s.rt_.dlss_rr_low_samples);
    f("fg_enabled", s.rt_.fg_enabled);
    f("dlss_fg", s.rt_.dlss_fg);
    f("upscaler_auto", s.rt_.upscaler_auto);
    f("anti_lag", s.rt_.anti_lag);
    f("terrain_raymarch_step_factor", s.rt_.terrain_raymarch_step_factor);
    f("terrain_raymarch_lod_near_m", s.rt_.terrain_raymarch_lod_near_m);
    f("terrain_raymarch_lod_far_m", s.rt_.terrain_raymarch_lod_far_m);
    f("terrain_raymarch_lod_min_octaves", s.rt_.terrain_raymarch_lod_min_octaves);
    f("terrain_raymarch_scale", s.rt_.terrain_raymarch_scale);
    f("half_rate_shadows", s.rt_.half_rate_shadows);
    f("terrain_raymarch_sharpen", s.rt_.terrain_raymarch_sharpen);
    f("terrain_raymarch_fog_strength", s.rt_.terrain_raymarch_fog_strength);
    f("terrain_raymarch_relaxation", s.rt_.terrain_raymarch_relaxation);
    f("terrain_raymarch_fog_godrays", s.rt_.terrain_raymarch_fog_godrays);
    f("terrain_raymarch_fog_y_start", s.rt_.terrain_raymarch_fog_y_start);
    f("terrain_raymarch_fog_y_top", s.rt_.terrain_raymarch_fog_y_top);
    f("terrain_raymarch_fog_noise", s.rt_.terrain_raymarch_fog_noise);
    f("water_enabled", s.rt_.water_enabled);
    f("water_level", s.rt_.water_level);
    f("water_wave_strength", s.rt_.water_wave_strength);
    f("water_wave_scale", s.rt_.water_wave_scale);
    f("water_rt_reflections", s.rt_.water_rt_reflections);
    f("water_tlas_reflections", s.rt_.water_tlas_reflections);
    f("water_shore_blend", s.rt_.water_shore_blend);
    f("water_shore_noise", s.rt_.water_shore_noise);
    f("water_shadows_enabled", s.rt_.water_shadows_enabled);
    f("water_transparency", s.rt_.water_transparency);
    f("water_clarity_depth", s.rt_.water_clarity_depth);
    f("water_shore_softness", s.rt_.water_shore_softness);
    f("water_foam_opacity", s.rt_.water_foam_opacity);
    f("water_color_shallow", s.rt_.water_color_shallow);
    f("water_color", s.rt_.water_color);
    f("water_foam_color", s.rt_.water_foam_color);
    f("water_foam_strength", s.rt_.water_foam_strength);
    f("water_foam_width", s.rt_.water_foam_width);
    f("water_style", s.rt_.water_style);
    f("water_river_speed", s.rt_.water_river_speed);
    f("water_river_normal_str", s.rt_.water_river_normal_str);
    f("water_river_flow_angle", s.rt_.water_river_flow_angle);
    f("water_river_time_speed", s.rt_.water_river_time_speed);
    f("water_river_detail", s.rt_.water_river_detail);
    f("water_river_foam_amount", s.rt_.water_river_foam_amount);
    f("water_river_extinct_color", s.rt_.water_river_extinct_color);
    f("water_river_extinct_density", s.rt_.water_river_extinct_density);
    f("water_lake_bump_strength", s.rt_.water_lake_bump_strength);
    f("water_lake_time_speed", s.rt_.water_lake_time_speed);
    f("water_lake_uv_scale", s.rt_.water_lake_uv_scale);
    f("water_lake_bump_dist", s.rt_.water_lake_bump_dist);
    f("grass_shore_color", s.rt_.grass_shore_color);
    f("grass_shore_strength", s.rt_.grass_shore_strength);
    f("grass_shore_distance", s.rt_.grass_shore_distance);
    f("terrain_shore_color", s.rt_.terrain_shore_color);
    f("terrain_shore_strength", s.rt_.terrain_shore_strength);
    f("terrain_shore_distance", s.rt_.terrain_shore_distance);
    f("distance_fog_color", s.rt_.distance_fog_color);
    f("distance_fog_strength", s.rt_.distance_fog_strength);
    f("distance_fog_density", s.rt_.distance_fog_density);
    f("distance_fog_start", s.rt_.distance_fog_start);
    f("distance_fog_height", s.rt_.distance_fog_height);
    f("distance_fog_max", s.rt_.distance_fog_max);
    f("terrain_shore_general_color", s.rt_.terrain_shore_general_color);
    f("terrain_shore_general_strength", s.rt_.terrain_shore_general_strength);
    f("terrain_shore_general_distance", s.rt_.terrain_shore_general_distance);
    f("terrain_sand_color", s.rt_.terrain_sand_color);
    f("rm_ground_dry", s.rt_.rm_ground_dry);
    f("rm_ground_green", s.rt_.rm_ground_green);
    f("rm_green_amount", s.rt_.rm_green_amount);
    f("rm_red_sand", s.rt_.rm_red_sand);
    f("rm_red_sand_amount", s.rt_.rm_red_sand_amount);
    f("rm_red_sand_scale", s.rt_.rm_red_sand_scale);
    f("rm_rock_color", s.rt_.rm_rock_color);
    f("rm_ground_saturation", s.rt_.rm_ground_saturation);
    f("shadow_near_mult", s.rt_.shadow_near_mult);
    f("gi_strength", s.rt_.gi_strength);
    f("gi_radius", s.rt_.gi_radius);
    f("gi_softener", s.rt_.gi_softener);
    f("reflections_enabled", s.rt_.reflections_enabled);
    f("reflection_strength", s.rt_.reflection_strength);
    f("taa_history_blend", s.rt_.taa_history_blend);
    f("taa_spatial_strength", s.rt_.taa_spatial_strength);
    f("bloom_enabled", s.rt_.bloom_enabled);
    f("bloom_strength", s.rt_.bloom_strength);
    f("bloom_threshold", s.rt_.bloom_threshold);
    f("bloom_radius", s.rt_.bloom_radius);
    f("spark_bloom", s.rt_.spark_bloom);
    f("taa_jitter_enabled", s.rt_.taa_jitter_enabled);
    f("textures_enabled", s.rt_.textures_enabled);
    f("lens_flare_enabled", s.rt_.lens_flare_enabled);
    f("lens_flare_strength", s.rt_.lens_flare_strength);
    f("lens_flare_threshold", s.rt_.lens_flare_threshold);
    f("lens_flare_dispersal", s.rt_.lens_flare_dispersal);
    f("lens_flare_halo_width", s.rt_.lens_flare_halo_width);
    f("lens_flare_ghosts", s.rt_.lens_flare_ghosts);
    f("lens_flare_aberration", s.rt_.lens_flare_aberration);
    f("taa_jitter_strength", s.rt_.taa_jitter_strength);
    f("compose_sharpen_strength", s.rt_.compose_sharpen_strength);
    f("sun_shaft_intensity", s.rt_.sun_shaft_intensity);
    f("auto_golden_hour", s.rt_.auto_golden_hour);
    f("sun_glare_strength", s.rt_.sun_glare_strength);
    f("svgf_enabled", s.rt_.svgf_enabled);
    f("image_contrast", s.rt_.image_contrast);
    f("image_brightness", s.rt_.image_brightness);
    f("image_gamma", s.rt_.image_gamma);
    f("voxel_tower_enabled", s.rt_.voxel_tower_enabled);
    f("grade_temperature", s.rt_.grade_temperature);
    f("grade_saturation", s.rt_.grade_saturation);
    f("grade_vibrance", s.rt_.grade_vibrance);
    f("grade_split_tone", s.rt_.grade_split_tone);
    f("film_grain", s.rt_.film_grain);
    f("vignette_strength", s.rt_.vignette_strength);
    f("terrain_rt_lod_distance", s.rt_.terrain_rt_lod_distance);
    f("terrain_ao_punch", s.rt_.terrain_ao_punch);
    f("terrain_pcss_samples_cap", s.rt_.terrain_pcss_samples_cap);
    f("terrain_gi_samples_cap", s.rt_.terrain_gi_samples_cap);
    f("terrain_gi_bounces_cap", s.rt_.terrain_gi_bounces_cap);
    f("terrain_ao_final_strength", s.rt_.terrain_ao_final_strength);
    f("terrain_rm_contrast", s.rt_.terrain_rm_contrast);
    f("render_scale", s.rt_.render_scale);
    f("quality_preset", s.rt_.quality_preset);
    f("ao_mode", s.rt_.ao_mode);
    f("gravity", s.game_.gravity);
    f("cubes_per_minute", s.game_.cubes_per_minute);
    f("bullet_mass", s.game_.bullet_mass);
    f("fire_rate_rps", s.game_.fire_rate_rps);
    f("bullet_speed", s.game_.bullet_speed);
    f("spark_scale", s.game_.spark_scale);
    f("player_pos_x", s.player_.position.x);
    f("player_pos_y", s.player_.position.y);
    f("player_pos_z", s.player_.position.z);
    f("player_yaw", s.player_.yaw);
    f("player_pitch", s.player_.pitch);
}

namespace {
template <class T> void write_value(std::ostream& o, const T& v) {
    if constexpr (std::is_same_v<T, bool>)            o << (v ? 1 : 0);
    else if constexpr (std::is_same_v<T, glm::vec3>)  o << v.x << " " << v.y << " " << v.z;
    else if constexpr (std::is_enum_v<T>)             o << static_cast<int>(v);
    else                                              o << v;
}
// Returns false on a malformed value (the field is left unchanged).
template <class T> bool parse_value(const std::string& s, T& out) {
    if constexpr (std::is_same_v<T, bool>)      { out = std::stoi(s) != 0; }
    else if constexpr (std::is_same_v<T, int>)  { out = std::stoi(s); }
    else if constexpr (std::is_same_v<T, float>) {
        const float f = std::stof(s);
        if (!std::isfinite(f)) return false;   // NaN would also break the autosave change check
        out = f;
    }
    else if constexpr (std::is_same_v<T, glm::vec3>) {
        glm::vec3 v;
        if (std::sscanf(s.c_str(), "%f %f %f", &v.x, &v.y, &v.z) != 3) return false;
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) return false;
        out = v;
    } else if constexpr (std::is_enum_v<T>) {
        const int m = std::stoi(s);   // only enum: TerrainBrushMode (0..ErodeSmooth = 7)
        if (m < 0 || m > 7) return false;
        out = static_cast<T>(m);
    } else {
        static_assert(sizeof(T) == 0, "settings: add a parse_value overload for this type");
    }
    return true;
}
} // namespace

void VulkanEngine::save_settings() const {
    // Test / capture runs (--frames, --screenshot, --autodemo, --scene,
    // QLIKE_TEST_* hooks) pin or move the camera and tweak modes - never let
    // them overwrite the player's saved cfg (pose, toggles).
    static const bool test_run = [this] {
        for (const char* e : { "QLIKE_TEST_HOLD", "QLIKE_TEST_PAN", "QLIKE_TEST_SHOOTDOWN",
                               "QLIKE_TEST_BOOM", "QLIKE_TEST_BLAST", "QLIKE_TEST_FIRE", "QLIKE_TEST_CHUTE", "QLIKE_TEST_SAM",
                               "QLIKE_TEST_CRAWL", "QLIKE_TEST_FLARE" })
            if (std::getenv(e)) return true;
        return opts_.max_frames > 0 || !opts_.screenshot_path.empty() ||
               opts_.autodemo_seconds > 0.0f || opts_.scene != 0;
    }();
    if (test_run) return;
    // Write a sibling temp file and rename it over the cfg: a crash or a
    // kill mid-write (the autosave runs while playing) can no longer leave
    // a truncated cfg that silently resets every missing setting.
    const std::string tmp_path = std::string(kSettingsPath) + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::trunc);
        if (!f) {
            log::warnf("could not open %s for write", tmp_path.c_str());
            return;
        }
        f << "# quake-like RT settings - auto-generated; safe to edit\n";
        visit_settings(*this, [&](const char* key, const auto& v) {
            f << key << " = ";
            write_value(f, v);
            f << "\n";
        });
        f.flush();
        if (!f) {
            log::warnf("write to %s failed - cfg left unchanged", tmp_path.c_str());
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, kSettingsPath, ec);
    if (ec) {
        log::warnf("could not replace %s: %s", kSettingsPath, ec.message().c_str());
        return;
    }
    log::infof("settings saved to %s", kSettingsPath);
}

void VulkanEngine::load_settings() {
    std::ifstream f(kSettingsPath);
    if (!f) return;  // first run; defaults are fine
    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    // Transient state that older cfgs may still carry: never restored.
    kv.erase("terrain_edit_mode");
    int loaded = 0;
    visit_settings(*this, [&](const char* key, auto& v) {
        auto it = kv.find(key);
        if (it == kv.end()) return;
        try {
            if (parse_value(it->second, v)) ++loaded;
            else log::warnf("settings: bad value '%s' = '%s' (kept default)", key, it->second.c_str());
        } catch (...) {
            log::warnf("settings: failed to parse '%s' = '%s'", key, it->second.c_str());
        }
        kv.erase(it);
    });
    for (const auto& [k, v] : kv) log::warnf("settings: unknown key '%s' (ignored)", k.c_str());
    // Clamp loaded values into sane ranges. A hand-edited or corrupted cfg
    // shouldn't be able to set shadow_samples = -1 and crash the renderer.
    auto clampi = [](int& v, int lo, int hi) { v = std::max(lo, std::min(hi, v)); };
    auto clampf = [](float& v, float lo, float hi) { v = std::max(lo, std::min(hi, v)); };
    // Clamp pitch above horizon — below is "night", which our shaders can't
    // currently model meaningfully (the sun would shine through the ground).
    clampf(rt_.sun_pitch_deg,    5.0f,    89.0f);
    clampf(rt_.sun_yaw_deg,    -360.0f,  360.0f);
    clampf(rt_.sun_intensity,    0.0f,   16.0f);
    clampf(rt_.ambient_strength, 0.0f,    8.0f);
    clampf(rt_.shadow_strength,  0.0f,    1.0f);
    clampi(rt_.shadow_samples,   1,      128);
    clampf(rt_.shadow_softness,  0.0f,    0.30f);
    // Player pose — guard against hand-edited cfg with NaN/over-range
    // pitch (would break view_matrix and the mouse-look clamp because
    // update_player only re-clamps if mouse_dx/dy are non-zero).
    constexpr float kPitchCap = 1.5707963f - 0.01f;
    if (!std::isfinite(player_.pitch)) player_.pitch = 0.0f;
    if (!std::isfinite(player_.yaw))   player_.yaw   = 0.0f;
    clampf(player_.pitch, -kPitchCap, kPitchCap);
    clampf(rt_.shadow_curve,     0.0f,    1.0f);
    clampi(rt_.ao_samples,       0,      128);
    clampf(rt_.ao_radius,        0.05f,  16.0f);
    clampi(rt_.gi_samples,       0,      256);
    clampi(rt_.gi_bounces,       1,        8);
    clampi(rt_.gi_shadow_max_bounce, 0,    8);
    clampf(rt_.ao_floor,         0.0f,    1.0f);
    clampf(rt_.auto_exposure_strength, 0.0f, 1.5f);
    clampf(rt_.gi_strength,      0.0f,    8.0f);
    clampf(rt_.gi_radius,        0.5f,  500.0f);
    clampf(rt_.gi_softener,      0.0f,    1.0f);
    clampf(rt_.reflection_strength, 0.0f, 1.0f);
    clampf(rt_.taa_history_blend,   0.0f, 0.98f);
    clampf(rt_.taa_spatial_strength,0.0f, 1.0f);
    clampf(rt_.bloom_strength,   0.0f,    4.0f);
    clampf(rt_.bloom_threshold,  0.0f,    8.0f);
    clampf(rt_.bloom_radius,     1.0f,  256.0f);
    clampf(rt_.spark_bloom,      0.0f,    8.0f);
    clampf(rt_.taa_jitter_strength, 0.0f, 4.0f);
    clampf(rt_.compose_sharpen_strength, 0.0f, 2.5f);
    clampf(rt_.image_contrast,           0.5f, 2.0f);
    clampf(rt_.image_brightness,         0.3f, 2.5f);
    clampf(rt_.image_gamma,              0.4f, 2.5f);
    clampf(rt_.terrain_rt_lod_distance,  50.0f, 1000.0f);
    clampf(rt_.terrain_ao_punch,         0.5f,  3.0f);
    clampi(rt_.terrain_pcss_samples_cap, 2,    32);
    clampi(rt_.terrain_gi_samples_cap,   0,    32);
    clampi(rt_.terrain_gi_bounces_cap,   1,     4);
    clampf(rt_.terrain_ao_final_strength,0.0f, 1.0f);
    clampf(rt_.terrain_rm_contrast, 0.5f, 2.5f);
    clampf(rt_.render_scale,     0.4f,    2.5f);
    clampi(rt_.quality_preset,  -1,         3);  // -1 = custom
    clampi(rt_.ao_mode,          0,         3);  // 0=off,1=fast,2=RTAO,3=HBAO
    clampi(rt_.spom_mode,        0,         3);  // 0 flat, 1 SPOM, 2 SSDM, 3 POM
    clampi(rt_.grass_shadow_samples, 0,      8);  // shader loop count
    clampi(terrain_brush_fbm_octaves_, 2,    9);
    clampf(terrain_brush_radius_,  0.5f,  50.0f);
    // Shadow-bake supersample: only 1 / 2 / 4 exist (anything else made the
    // per-frame "changed?" check fire forever, or a huge texture).
    rt_.terrain_bake_supersample = rt_.terrain_bake_supersample >= 4 ? 4 : (rt_.terrain_bake_supersample >= 2 ? 2 : 1);
    clampf(game_.gravity,        0.0f, 1500.0f);
    clampi(game_.cubes_per_minute, 0,    1000);
    clampf(game_.bullet_mass,    0.5f,  200.0f);
    clampf(game_.fire_rate_rps,  0.0f,  60.0f);
    clampf(game_.bullet_speed,   20.0f, 600.0f);
    clampf(game_.spark_scale,    0.05f, 3.0f);

    log::infof("settings loaded from %s (%d keys)  textures=%d  shadows=%d/%d  gi=%d",
               kSettingsPath, loaded,
               rt_.textures_enabled ? 1 : 0,
               rt_.shadow_enabled ? 1 : 0, rt_.shadow_samples,
               rt_.gi_samples);
    rt_last_saved_   = rt_;
    rt_prev_frame_   = rt_;
    game_last_saved_ = game_;
    game_prev_frame_ = game_;
}

} // namespace qlike
