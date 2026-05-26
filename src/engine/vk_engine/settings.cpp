// Settings persistence — debounced auto-save / load / clamp. Reads
// qlike_settings.cfg in the working directory; ignored if missing (defaults
// from RtSettings / GameSettings apply). Format is one `key = value` per
// line; vec3 fields use space-separated triples.

#include "engine/vk_engine/internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

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
    bool dirty_vs_disk      = !(rt_ == rt_last_saved_) ||
                              !(game_ == game_last_saved_);
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
            rt_last_saved_   = rt_;
            game_last_saved_ = game_;
            settings_dirty_timer_ = 0.0f;
        }
    } else {
        settings_dirty_timer_ = 0.0f;
    }
}

void VulkanEngine::save_settings() const {
    std::ofstream f(kSettingsPath);
    if (!f) {
        log::warnf("could not open %s for write", kSettingsPath);
        return;
    }
    f << "# quake-like RT settings — auto-generated; safe to edit\n";
    auto v3 = [](glm::vec3 v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.4f %.4f %.4f", v.x, v.y, v.z);
        return std::string(buf);
    };
    f << "sun_pitch_deg = "      << rt_.sun_pitch_deg      << "\n";
    f << "sun_yaw_deg = "        << rt_.sun_yaw_deg        << "\n";
    f << "sun_intensity = "      << rt_.sun_intensity      << "\n";
    f << "sun_color = "          << v3(rt_.sun_color)      << "\n";
    f << "sky_color = "          << v3(rt_.sky_color)      << "\n";
    f << "ground_ambient = "     << v3(rt_.ground_ambient) << "\n";
    f << "ambient_strength = "   << rt_.ambient_strength   << "\n";
    f << "shadow_strength = "    << rt_.shadow_strength    << "\n";
    f << "shadow_enabled = "     << (rt_.shadow_enabled ? 1 : 0) << "\n";
    f << "shadow_samples = "     << rt_.shadow_samples     << "\n";
    f << "shadow_softness = "    << rt_.shadow_softness    << "\n";
    f << "shadow_curve = "       << rt_.shadow_curve       << "\n";
    f << "ao_samples = "         << rt_.ao_samples         << "\n";
    f << "ao_radius = "          << rt_.ao_radius          << "\n";
    f << "gi_samples = "         << rt_.gi_samples         << "\n";
    f << "gi_bounces = "         << rt_.gi_bounces         << "\n";
    f << "gi_restir_enabled = "  << (rt_.gi_restir_enabled ? 1 : 0) << "\n";
    f << "gi_shadow_max_bounce = " << rt_.gi_shadow_max_bounce << "\n";
    f << "ao_floor = "             << rt_.ao_floor             << "\n";
    f << "auto_exposure_strength = " << rt_.auto_exposure_strength << "\n";
    f << "use_merged_static_blas = " << (rt_.use_merged_static_blas ? 1 : 0) << "\n";
    f << "terrain_brush_radius = "   << terrain_brush_radius_   << "\n";
    f << "terrain_brush_strength = " << terrain_brush_strength_ << "\n";
    f << "terrain_brush_mode = "     << static_cast<int>(terrain_brush_mode_) << "\n";
    f << "terrain_brush_flatten_target = " << terrain_brush_flatten_target_ << "\n";
    f << "terrain_brush_noise_strength = " << terrain_brush_noise_strength_ << "\n";
    f << "terrain_brush_noise_freq = "     << terrain_brush_noise_freq_     << "\n";
    f << "terrain_brush_use_fbm_erosion = " << (terrain_brush_use_fbm_erosion_ ? 1 : 0) << "\n";
    f << "terrain_brush_fbm_octaves = "    << terrain_brush_fbm_octaves_    << "\n";
    // terrain_edit_mode is transient interaction state, NOT a saved
    // preference — persisting it meant a stray auto-save with the brush
    // active made the edit ring reappear on every later launch even
    // though the user never entered edit mode. Always start it off.
    f << "terrain_fog_strength = "    << rt_.terrain_fog_strength << "\n";
    f << "terrain_wrap_strength = "   << rt_.terrain_wrap_strength << "\n";
    f << "terrain_detail_strength = " << rt_.terrain_detail_strength << "\n";
    f << "terrain_shadow_softness_scale = " << rt_.terrain_shadow_softness_scale << "\n";
    f << "terrain_h_sand_grass_start = " << rt_.terrain_h_sand_grass_start << "\n";
    f << "terrain_h_sand_grass_end = "   << rt_.terrain_h_sand_grass_end   << "\n";
    f << "terrain_h_grass_dirt_start = " << rt_.terrain_h_grass_dirt_start << "\n";
    f << "terrain_h_grass_dirt_end = "   << rt_.terrain_h_grass_dirt_end   << "\n";
    f << "terrain_h_dirt_rock_start = "  << rt_.terrain_h_dirt_rock_start  << "\n";
    f << "terrain_h_dirt_rock_end = "    << rt_.terrain_h_dirt_rock_end    << "\n";
    f << "terrain_h_rock_snow_start = "  << rt_.terrain_h_rock_snow_start  << "\n";
    f << "terrain_h_rock_snow_end = "    << rt_.terrain_h_rock_snow_end    << "\n";
    f << "grass_enabled = "       << (rt_.grass_enabled ? 1 : 0) << "\n";
    f << "grass_raymarch_enabled = " << (rt_.grass_raymarch_enabled ? 1 : 0) << "\n";
    f << "grass_raymarch_distance = " << rt_.grass_raymarch_distance << "\n";
    f << "grass_distance = "      << rt_.grass_distance          << "\n";
    f << "grass_wind = "          << rt_.grass_wind              << "\n";
    f << "grass_density = "       << rt_.grass_density           << "\n";
    f << "grass_height_scale = "  << rt_.grass_height_scale      << "\n";
    f << "grass_alpha_cutoff = "  << rt_.grass_alpha_cutoff      << "\n";
    f << "grass_cutoff_soft_dist = " << rt_.grass_cutoff_soft_dist << "\n";
    f << "grass_slope_n_min = "   << rt_.grass_slope_n_min       << "\n";
    f << "grass_distance_density = " << rt_.grass_distance_density << "\n";
    f << "grass_alt_min = " << rt_.grass_alt_min << "\n";
    f << "grass_alt_max = " << rt_.grass_alt_max << "\n";
    f << "grass_color_top = "    << rt_.grass_color_top.x    << " "
                                  << rt_.grass_color_top.y    << " "
                                  << rt_.grass_color_top.z    << "\n";
    f << "grass_color_bottom = " << rt_.grass_color_bottom.x << " "
                                  << rt_.grass_color_bottom.y << " "
                                  << rt_.grass_color_bottom.z << "\n";
    f << "grass_color_ground = " << rt_.grass_color_ground.x << " "
                                  << rt_.grass_color_ground.y << " "
                                  << rt_.grass_color_ground.z << "\n";
    f << "grass_color_ground_far = " << rt_.grass_color_ground_far.x << " "
                                      << rt_.grass_color_ground_far.y << " "
                                      << rt_.grass_color_ground_far.z << "\n";
    f << "grass_ground_tint_far_distance = " << rt_.grass_ground_tint_far_distance << "\n";
    f << "grass_base_ao_floor = "       << rt_.grass_base_ao_floor       << "\n";
    f << "grass_ground_tint_strength = " << rt_.grass_ground_tint_strength << "\n";
    f << "grass_shadow_strength = " << rt_.grass_shadow_strength << "\n";
    f << "grass_shadow_samples = "  << rt_.grass_shadow_samples  << "\n";
    f << "grass_shadow_max_dist = " << rt_.grass_shadow_max_dist << "\n";
    f << "shadow_map_resolution = " << rt_.shadow_map_resolution << "\n";
    f << "shadow_map_world_half = " << rt_.shadow_map_world_half << "\n";
    f << "shadow_debug_overlay = "  << (rt_.shadow_debug_overlay ? 1 : 0) << "\n";
    // Per-LOD distance thresholds (7 floats for 8 LODs). Index i is the
    // distance at which the chunk leaves LOD i and switches to LOD i+1.
    for (int i = 0; i < kTerrainLodCount - 1; ++i) {
        f << "terrain_lod_distance_" << i << " = "
          << rt_.terrain_lod_distance[i] << "\n";
    }
    // Per-LOD vertex stride (8 ints). LOD 0 stride is informational
    // only -- the renderer always draws stride-1 at LOD 0.
    for (int i = 0; i < kTerrainLodCount; ++i) {
        f << "terrain_lod_stride_" << i << " = "
          << rt_.terrain_lod_stride[i] << "\n";
    }
    f << "terrain_lod_scale = " << rt_.terrain_lod_scale << "\n";
    f << "terrain_near_density = "          << rt_.terrain_near_density          << "\n";
    f << "terrain_near_density_radius_m = " << rt_.terrain_near_density_radius_m << "\n";
    f << "terrain_tessellation_enabled = " << (rt_.terrain_tessellation_enabled ? 1 : 0) << "\n";
    f << "terrain_tess_range = " << rt_.terrain_tess_range << "\n";
    f << "terrain_wireframe = " << (rt_.terrain_wireframe ? 1 : 0) << "\n";
    f << "terrain_tess_max_level = " << rt_.terrain_tess_max_level << "\n";
    f << "terrain_tess_near_m = " << rt_.terrain_tess_near_m << "\n";
    f << "terrain_tess_far_m = " << rt_.terrain_tess_far_m << "\n";
    f << "terrain_tess_falloff = " << rt_.terrain_tess_falloff << "\n";
    f << "terrain_tess_smooth = " << rt_.terrain_tess_smooth << "\n";
    f << "terrain_pom_strength = " << rt_.terrain_pom_strength << "\n";
    f << "terrain_disp_amp = " << rt_.terrain_disp_amp << "\n";
    f << "terrain_disp_smooth_mip = " << rt_.terrain_disp_smooth_mip << "\n";
    f << "terrain_pom_far_m = " << rt_.terrain_pom_far_m << "\n";
    f << "terrain_sand_ripple_scale = " << rt_.terrain_sand_ripple_scale << "\n";
    f << "terrain_grass_line_scale = " << rt_.terrain_grass_line_scale << "\n";
    f << "terrain_rock_relief = " << rt_.terrain_rock_relief << "\n";
    f << "ground_mat_strength = " << rt_.ground_mat_strength << "\n";
    f << "ground_mat_tile_m = " << rt_.ground_mat_tile_m << "\n";
    f << "ground_mat_normal = " << rt_.ground_mat_normal << "\n";
    f << "terrain_antitile = " << (rt_.terrain_antitile ? 1 : 0) << "\n";
    f << "terrain_antitile_strength = " << rt_.terrain_antitile_strength << "\n";
    f << "grass_shadow_on_terrain = " << (rt_.grass_shadow_on_terrain ? 1 : 0) << "\n";
    f << "grass_shadow_on_terrain_strength = " << rt_.grass_shadow_on_terrain_strength << "\n";
    f << "grass_shadow_on_terrain_samples = " << rt_.grass_shadow_on_terrain_samples << "\n";
    f << "grass_shadow_on_terrain_dist = " << rt_.grass_shadow_on_terrain_dist << "\n";
    f << "grass_side_lit_enabled = " << (rt_.grass_side_lit_enabled ? 1 : 0) << "\n";
    f << "grass_side_lit_strength = " << rt_.grass_side_lit_strength << "\n";
    f << "terrain_bake_supersample = " << rt_.terrain_bake_supersample << "\n";
    f << "terrain_shading_contrast = " << rt_.terrain_shading_contrast << "\n";
    f << "spom_strength = " << rt_.spom_strength << "\n";
    f << "terrain_heightmap_scale = " << rt_.terrain_heightmap_scale << "\n";
    f << "terrain_raymarch_enabled = " << (rt_.terrain_raymarch_enabled ? 1 : 0) << "\n";
    f << "terrain_raymarch_max_steps = " << rt_.terrain_raymarch_max_steps << "\n";
    f << "terrain_raymarch_shadow_steps = " << rt_.terrain_raymarch_shadow_steps << "\n";
    f << "terrain_raymarch_octaves = " << rt_.terrain_raymarch_octaves << "\n";
    f << "terrain_raymarch_normal_octaves = " << rt_.terrain_raymarch_normal_octaves << "\n";
    f << "taau_enabled = "                << (rt_.taau_enabled ? 1 : 0) << "\n";
    f << "fsr2_enabled = "                << (rt_.fsr2_enabled ? 1 : 0) << "\n";
    f << "fsr_backend = "                 << rt_.fsr_backend << "\n";
    f << "fg_enabled = "                  << (rt_.fg_enabled ? 1 : 0) << "\n";
    f << "terrain_raymarch_step_factor = " << rt_.terrain_raymarch_step_factor << "\n";
    f << "terrain_raymarch_lod_near_m = " << rt_.terrain_raymarch_lod_near_m << "\n";
    f << "terrain_raymarch_lod_far_m = " << rt_.terrain_raymarch_lod_far_m << "\n";
    f << "terrain_raymarch_lod_min_octaves = " << rt_.terrain_raymarch_lod_min_octaves << "\n";
    f << "terrain_raymarch_scale = " << rt_.terrain_raymarch_scale << "\n";
    f << "half_rate_shadows = " << (rt_.half_rate_shadows ? 1 : 0) << "\n";
    f << "terrain_raymarch_sharpen = " << rt_.terrain_raymarch_sharpen << "\n";
    f << "terrain_raymarch_fog_strength = " << rt_.terrain_raymarch_fog_strength << "\n";
    f << "terrain_raymarch_relaxation = " << (rt_.terrain_raymarch_relaxation ? 1 : 0) << "\n";
    f << "terrain_raymarch_fog_godrays = " << (rt_.terrain_raymarch_fog_godrays ? 1 : 0) << "\n";
    f << "terrain_raymarch_fog_y_start = " << rt_.terrain_raymarch_fog_y_start << "\n";
    f << "terrain_raymarch_fog_y_top = "   << rt_.terrain_raymarch_fog_y_top   << "\n";
    f << "terrain_raymarch_fog_noise = "   << rt_.terrain_raymarch_fog_noise   << "\n";
    f << "water_enabled = " << (rt_.water_enabled ? 1 : 0) << "\n";
    f << "water_level = " << rt_.water_level << "\n";
    f << "water_wave_strength = " << rt_.water_wave_strength << "\n";
    f << "water_wave_scale = " << rt_.water_wave_scale << "\n";
    f << "water_rt_reflections = " << (rt_.water_rt_reflections ? 1 : 0) << "\n";
    f << "water_tlas_reflections = " << (rt_.water_tlas_reflections ? 1 : 0) << "\n";
    f << "water_shore_blend = " << rt_.water_shore_blend << "\n";
    f << "water_shore_noise = " << rt_.water_shore_noise << "\n";
    f << "water_shadows_enabled = " << (rt_.water_shadows_enabled ? 1 : 0) << "\n";
    f << "water_transparency = " << rt_.water_transparency << "\n";
    f << "water_clarity_depth = " << rt_.water_clarity_depth << "\n";
    f << "water_shore_softness = " << rt_.water_shore_softness << "\n";
    f << "water_foam_opacity = " << rt_.water_foam_opacity << "\n";
    f << "water_color_shallow = " << rt_.water_color_shallow.r << " "
                                   << rt_.water_color_shallow.g << " "
                                   << rt_.water_color_shallow.b << "\n";
    f << "water_color = " << rt_.water_color.r << " "
                            << rt_.water_color.g << " "
                            << rt_.water_color.b << "\n";
    f << "water_foam_color = " << rt_.water_foam_color.r << " "
                                << rt_.water_foam_color.g << " "
                                << rt_.water_foam_color.b << "\n";
    f << "water_foam_strength = " << rt_.water_foam_strength << "\n";
    f << "water_foam_width = "    << rt_.water_foam_width    << "\n";
    f << "water_style = "             << rt_.water_style             << "\n";
    f << "water_river_speed = "       << rt_.water_river_speed       << "\n";
    f << "water_river_normal_str = "  << rt_.water_river_normal_str  << "\n";
    f << "water_river_flow_angle = "  << rt_.water_river_flow_angle  << "\n";
    f << "water_river_time_speed = "  << rt_.water_river_time_speed  << "\n";
    f << "water_river_detail = "      << rt_.water_river_detail      << "\n";
    f << "water_river_foam_amount = " << rt_.water_river_foam_amount << "\n";
    f << "water_river_extinct_color = " << rt_.water_river_extinct_color.r << " "
                                         << rt_.water_river_extinct_color.g << " "
                                         << rt_.water_river_extinct_color.b << "\n";
    f << "water_river_extinct_density = " << rt_.water_river_extinct_density << "\n";
    f << "water_lake_bump_strength = " << rt_.water_lake_bump_strength << "\n";
    f << "water_lake_time_speed = "    << rt_.water_lake_time_speed    << "\n";
    f << "water_lake_uv_scale = "      << rt_.water_lake_uv_scale      << "\n";
    f << "water_lake_bump_dist = "     << rt_.water_lake_bump_dist     << "\n";
    f << "grass_shore_color = " << rt_.grass_shore_color.r << " "
                                 << rt_.grass_shore_color.g << " "
                                 << rt_.grass_shore_color.b << "\n";
    f << "grass_shore_strength = " << rt_.grass_shore_strength << "\n";
    f << "grass_shore_distance = " << rt_.grass_shore_distance << "\n";
    f << "terrain_shore_color = " << rt_.terrain_shore_color.r << " "
                                   << rt_.terrain_shore_color.g << " "
                                   << rt_.terrain_shore_color.b << "\n";
    f << "terrain_shore_strength = " << rt_.terrain_shore_strength << "\n";
    f << "terrain_shore_distance = " << rt_.terrain_shore_distance << "\n";
    f << "distance_fog_color = " << rt_.distance_fog_color.r << " "
                                  << rt_.distance_fog_color.g << " "
                                  << rt_.distance_fog_color.b << "\n";
    f << "distance_fog_strength = " << rt_.distance_fog_strength << "\n";
    f << "distance_fog_density = "  << rt_.distance_fog_density  << "\n";
    f << "distance_fog_start = "    << rt_.distance_fog_start    << "\n";
    f << "distance_fog_height = "   << rt_.distance_fog_height   << "\n";
    f << "distance_fog_max = "      << rt_.distance_fog_max      << "\n";
    f << "terrain_shore_general_color = " << rt_.terrain_shore_general_color.r << " "
                                           << rt_.terrain_shore_general_color.g << " "
                                           << rt_.terrain_shore_general_color.b << "\n";
    f << "terrain_shore_general_strength = " << rt_.terrain_shore_general_strength << "\n";
    f << "terrain_shore_general_distance = " << rt_.terrain_shore_general_distance << "\n";
    f << "terrain_sand_color = " << rt_.terrain_sand_color.r << " "
                                  << rt_.terrain_sand_color.g << " "
                                  << rt_.terrain_sand_color.b << "\n";
    f << "shadow_near_mult = " << rt_.shadow_near_mult << "\n";
    f << "gi_strength = "        << rt_.gi_strength        << "\n";
    f << "gi_radius = "          << rt_.gi_radius          << "\n";
    f << "gi_softener = "        << rt_.gi_softener        << "\n";
    f << "reflections_enabled = "<< (rt_.reflections_enabled ? 1 : 0) << "\n";
    f << "reflection_strength = "<< rt_.reflection_strength << "\n";
    f << "taa_history_blend = " << rt_.taa_history_blend    << "\n";
    f << "taa_spatial_strength = " << rt_.taa_spatial_strength << "\n";
    f << "bloom_enabled = "    << (rt_.bloom_enabled ? 1 : 0) << "\n";
    f << "bloom_strength = "   << rt_.bloom_strength    << "\n";
    f << "bloom_threshold = "  << rt_.bloom_threshold   << "\n";
    f << "bloom_radius = "     << rt_.bloom_radius      << "\n";
    f << "spark_bloom = "      << rt_.spark_bloom       << "\n";
    f << "taa_jitter_enabled = " << (rt_.taa_jitter_enabled ? 1 : 0) << "\n";
    f << "textures_enabled = "   << (rt_.textures_enabled   ? 1 : 0) << "\n";
    f << "lens_flare_enabled = " << (rt_.lens_flare_enabled ? 1 : 0) << "\n";
    f << "lens_flare_strength = "    << rt_.lens_flare_strength    << "\n";
    f << "lens_flare_threshold = "   << rt_.lens_flare_threshold   << "\n";
    f << "lens_flare_dispersal = "   << rt_.lens_flare_dispersal   << "\n";
    f << "lens_flare_halo_width = "  << rt_.lens_flare_halo_width  << "\n";
    f << "lens_flare_ghosts = "      << rt_.lens_flare_ghosts      << "\n";
    f << "lens_flare_aberration = "  << rt_.lens_flare_aberration  << "\n";
    f << "taa_jitter_strength = " << rt_.taa_jitter_strength << "\n";
    f << "compose_sharpen_strength = " << rt_.compose_sharpen_strength << "\n";
    f << "sun_shaft_intensity = " << rt_.sun_shaft_intensity << "\n";
    f << "auto_golden_hour = " << (rt_.auto_golden_hour ? 1 : 0) << "\n";
    f << "sun_glare_strength = " << rt_.sun_glare_strength << "\n";
    f << "svgf_enabled = " << (rt_.svgf_enabled ? 1 : 0) << "\n";
    f << "image_contrast = " << rt_.image_contrast << "\n";
    f << "image_brightness = " << rt_.image_brightness << "\n";
    f << "image_gamma = " << rt_.image_gamma << "\n";
    f << "terrain_rt_lod_distance = " << rt_.terrain_rt_lod_distance << "\n";
    f << "terrain_ao_punch = " << rt_.terrain_ao_punch << "\n";
    f << "terrain_pcss_samples_cap = "  << rt_.terrain_pcss_samples_cap  << "\n";
    f << "terrain_gi_samples_cap = "    << rt_.terrain_gi_samples_cap    << "\n";
    f << "terrain_gi_bounces_cap = "    << rt_.terrain_gi_bounces_cap    << "\n";
    f << "terrain_ao_final_strength = " << rt_.terrain_ao_final_strength << "\n";
    f << "render_scale = "       << rt_.render_scale       << "\n";
    f << "quality_preset = "     << rt_.quality_preset     << "\n";
    f << "ao_mode = "            << rt_.ao_mode            << "\n";
    f << "gravity = "             << game_.gravity            << "\n";
    f << "cubes_per_minute = "    << game_.cubes_per_minute   << "\n";
    f << "bullet_mass = "         << game_.bullet_mass        << "\n";
    f << "fire_rate_rps = "       << game_.fire_rate_rps      << "\n";
    f << "bullet_speed = "        << game_.bullet_speed       << "\n";
    f << "spark_scale = "         << game_.spark_scale        << "\n";
    // Persist player position + view orientation so re-launching the game
    // opens at the same spot the user was inspecting (most useful while
    // tuning visuals — relaunching shouldn't mean walking back across
    // the map every time).
    f << "player_pos_x = "        << player_.position.x       << "\n";
    f << "player_pos_y = "        << player_.position.y       << "\n";
    f << "player_pos_z = "        << player_.position.z       << "\n";
    f << "player_yaw = "          << player_.yaw              << "\n";
    f << "player_pitch = "        << player_.pitch            << "\n";
    log::infof("settings saved to %s", kSettingsPath);
}

bool VulkanEngine::apply_terrain_lod_key(const std::string& key,
                                          const std::string& val) {
    // Match `terrain_lod_distance_<i>` -- 7 floats indexed 0..6.
    static const char kDistPrefix[] = "terrain_lod_distance_";
    if (key.rfind(kDistPrefix, 0) == 0) {
        int idx = std::atoi(key.c_str() + sizeof(kDistPrefix) - 1);
        if (idx >= 0 && idx < kTerrainLodCount - 1)
            rt_.terrain_lod_distance[idx] = std::stof(val);
        return true;
    }
    // Match `terrain_lod_stride_<i>` -- 8 ints indexed 0..7.
    static const char kStridePrefix[] = "terrain_lod_stride_";
    if (key.rfind(kStridePrefix, 0) == 0) {
        int idx = std::atoi(key.c_str() + sizeof(kStridePrefix) - 1);
        if (idx >= 0 && idx < kTerrainLodCount)
            rt_.terrain_lod_stride[idx] = std::stoi(val);
        return true;
    }
    return false;
}

void VulkanEngine::load_settings() {
    std::ifstream f(kSettingsPath);
    if (!f) return;  // first run; defaults are fine
    std::string line;
    int loaded = 0;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        try {
            // Flat fast-path for keys added after the main chain hit
            // MSVC's nesting depth limit (C1061). Each is independent
            // of the chain below and short-circuits via `continue`.
            if (key == "image_contrast")          { rt_.image_contrast = std::stof(val); ++loaded; continue; }
            if (key == "image_brightness")        { rt_.image_brightness = std::stof(val); ++loaded; continue; }
            if (key == "image_gamma")             { rt_.image_gamma = std::stof(val); ++loaded; continue; }
            // Brush state — these live up here in the if-continue
            // section because the giant else-if chain below already
            // hits MSVC's nested-block-depth limit; adding more
            // entries to it triggers C1061 "blocks nested too deeply".
            if (key == "terrain_brush_noise_strength")  { terrain_brush_noise_strength_  = std::stof(val); ++loaded; continue; }
            if (key == "terrain_brush_noise_freq")      { terrain_brush_noise_freq_      = std::stof(val); ++loaded; continue; }
            if (key == "terrain_brush_use_fbm_erosion") { terrain_brush_use_fbm_erosion_ = (std::stoi(val) != 0); ++loaded; continue; }
            if (key == "terrain_brush_fbm_octaves")     { terrain_brush_fbm_octaves_     = std::stoi(val); ++loaded; continue; }
            // terrain_edit_mode intentionally NOT loaded — transient
            // state, always starts off (old cfgs may still have the key;
            // it just falls through as unknown and is ignored).
            if (key == "terrain_edit_mode")             { ++loaded; continue; }
            // Player pose — peeled out of the else-if chain because
            // the recent merge added enough new keys (raymarch LOD,
            // VRS, TAAU) to push past MSVC's nested-block depth limit
            // (C1061). Restored after init_world resets defaults.
            if (key == "player_pos_x")                  { player_.position.x = std::stof(val); ++loaded; continue; }
            if (key == "player_pos_y")                  { player_.position.y = std::stof(val); ++loaded; continue; }
            if (key == "player_pos_z")                  { player_.position.z = std::stof(val); ++loaded; continue; }
            if (key == "player_yaw")                    { player_.yaw        = std::stof(val); ++loaded; continue; }
            if (key == "player_pitch")                  { player_.pitch      = std::stof(val); ++loaded; continue; }
            // ReSTIR-lite (session 1) — fast-path so it doesn't deepen the
            // already-at-limit else-if chain below.
            if (key == "gi_restir_enabled")             { rt_.gi_restir_enabled = (std::stoi(val) != 0); ++loaded; continue; }
            // FSR backend select (0=FSR2, 1=FSR3 ffx-api). Persisted alongside
            // fsr2_enabled in the fast-path section to dodge MSVC's nested-if depth limit.
            if (key == "fsr_backend")                   { rt_.fsr_backend = std::stoi(val); ++loaded; continue; }
            if (key == "fg_enabled")                    { rt_.fg_enabled  = (std::stoi(val) != 0); ++loaded; continue; }
            // Per-LOD terrain ladder (kTerrainLodCount = 8). Routed
            // through the fast-path because the else-if chain below is
            // already at MSVC's nested-block depth limit (C1061).
            if (apply_terrain_lod_key(key, val))        { ++loaded; continue; }
            // Near-camera VBO densification -- fast-path because the
            // else-if chain below already hits MSVC's nested-block
            // depth limit (C1061).
            if (key == "terrain_near_density")          { rt_.terrain_near_density          = std::clamp(std::stoi(val), 1, 8); ++loaded; continue; }
            if (key == "terrain_near_density_radius_m") { rt_.terrain_near_density_radius_m = std::stof(val); ++loaded; continue; }
            // Back-compat: legacy float keys map the first three
            // thresholds onto the new array so old cfgs still roughly
            // preserve their tuning.
            if (key == "terrain_lod1")                  { rt_.terrain_lod_distance[0] = std::stof(val); ++loaded; continue; }
            if (key == "terrain_lod2")                  { rt_.terrain_lod_distance[1] = std::stof(val); ++loaded; continue; }
            if (key == "terrain_lod3")                  { rt_.terrain_lod_distance[2] = std::stof(val); ++loaded; continue; }
            if (key == "water_foam_color") {
                glm::vec3 v(0.88f, 0.94f, 0.96f);
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.water_foam_color = v;
                }
                ++loaded; continue;
            }
            if (key == "water_foam_strength") { rt_.water_foam_strength = std::stof(val); ++loaded; continue; }
            if (key == "water_foam_width")    { rt_.water_foam_width    = std::stof(val); ++loaded; continue; }
            if (key == "water_style")            { rt_.water_style            = std::stoi(val); ++loaded; continue; }
            if (key == "water_river_speed")      { rt_.water_river_speed      = std::stof(val); ++loaded; continue; }
            if (key == "water_river_normal_str") { rt_.water_river_normal_str = std::stof(val); ++loaded; continue; }
            if (key == "water_river_flow_angle") { rt_.water_river_flow_angle = std::stof(val); ++loaded; continue; }
            if (key == "water_river_time_speed") { rt_.water_river_time_speed = std::stof(val); ++loaded; continue; }
            if (key == "water_river_detail")     { rt_.water_river_detail     = std::stof(val); ++loaded; continue; }
            if (key == "water_river_foam_amount"){ rt_.water_river_foam_amount= std::stof(val); ++loaded; continue; }
            if (key == "water_river_extinct_color") {
                glm::vec3 v = rt_.water_river_extinct_color;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.water_river_extinct_color = v;
                }
                ++loaded; continue;
            }
            if (key == "water_river_extinct_density") { rt_.water_river_extinct_density = std::stof(val); ++loaded; continue; }
            if (key == "water_lake_bump_strength") { rt_.water_lake_bump_strength = std::stof(val); ++loaded; continue; }
            if (key == "water_lake_time_speed")    { rt_.water_lake_time_speed    = std::stof(val); ++loaded; continue; }
            if (key == "water_lake_uv_scale")      { rt_.water_lake_uv_scale      = std::stof(val); ++loaded; continue; }
            if (key == "water_lake_bump_dist")     { rt_.water_lake_bump_dist     = std::stof(val); ++loaded; continue; }
            if (key == "grass_shore_color") {
                glm::vec3 v(0.40f, 0.30f, 0.16f);
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.grass_shore_color = v;
                }
                ++loaded; continue;
            }
            if (key == "grass_shore_strength") { rt_.grass_shore_strength = std::stof(val); ++loaded; continue; }
            if (key == "grass_shore_distance") { rt_.grass_shore_distance = std::stof(val); ++loaded; continue; }
            if (key == "grass_ground_tint_far_distance") { rt_.grass_ground_tint_far_distance = std::stof(val); ++loaded; continue; }
            if (key == "terrain_shore_color") {
                glm::vec3 v(0.55f, 0.46f, 0.30f);
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.terrain_shore_color = v;
                }
                ++loaded; continue;
            }
            if (key == "terrain_shore_strength") { rt_.terrain_shore_strength = std::stof(val); ++loaded; continue; }
            if (key == "terrain_shore_distance") { rt_.terrain_shore_distance = std::stof(val); ++loaded; continue; }
            if (key == "distance_fog_color") {
                glm::vec3 v(0.62f, 0.70f, 0.78f);
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.distance_fog_color = v;
                }
                ++loaded; continue;
            }
            if (key == "distance_fog_strength") { rt_.distance_fog_strength = std::stof(val); ++loaded; continue; }
            if (key == "distance_fog_density")  { rt_.distance_fog_density  = std::stof(val); ++loaded; continue; }
            if (key == "distance_fog_start")    { rt_.distance_fog_start    = std::stof(val); ++loaded; continue; }
            if (key == "distance_fog_height")   { rt_.distance_fog_height   = std::stof(val); ++loaded; continue; }
            if (key == "distance_fog_max")      { rt_.distance_fog_max      = std::stof(val); ++loaded; continue; }
            if (key == "terrain_shore_general_color") {
                glm::vec3 v = rt_.terrain_shore_general_color;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.terrain_shore_general_color = v;
                }
                ++loaded; continue;
            }
            if (key == "terrain_shore_general_strength") { rt_.terrain_shore_general_strength = std::stof(val); ++loaded; continue; }
            if (key == "terrain_shore_general_distance") { rt_.terrain_shore_general_distance = std::stof(val); ++loaded; continue; }
            if (key == "terrain_sand_color") {
                glm::vec3 v = rt_.terrain_sand_color;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.terrain_sand_color = v;
                }
                ++loaded; continue;
            }
            if (key == "terrain_rt_lod_distance") { rt_.terrain_rt_lod_distance = std::stof(val); ++loaded; continue; }
            if (key == "terrain_ao_punch")        { rt_.terrain_ao_punch = std::stof(val); ++loaded; continue; }
            if (key == "terrain_pcss_samples_cap")  { rt_.terrain_pcss_samples_cap  = std::stoi(val); ++loaded; continue; }
            if (key == "terrain_gi_samples_cap")    { rt_.terrain_gi_samples_cap    = std::stoi(val); ++loaded; continue; }
            if (key == "terrain_gi_bounces_cap")    { rt_.terrain_gi_bounces_cap    = std::stoi(val); ++loaded; continue; }
            if (key == "terrain_ao_final_strength") { rt_.terrain_ao_final_strength = std::stof(val); ++loaded; continue; }
            if (key == "grass_color_top") {
                glm::vec3 v = rt_.grass_color_top;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.grass_color_top = v;
                }
                ++loaded; continue;
            }
            if (key == "grass_color_bottom") {
                glm::vec3 v = rt_.grass_color_bottom;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.grass_color_bottom = v;
                }
                ++loaded; continue;
            }
            if (key == "grass_color_ground") {
                glm::vec3 v = rt_.grass_color_ground;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.grass_color_ground = v;
                }
                ++loaded; continue;
            }
            if (key == "grass_color_ground_far") {
                glm::vec3 v = rt_.grass_color_ground_far;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.grass_color_ground_far = v;
                }
                ++loaded; continue;
            }
            if (key == "grass_base_ao_floor")        { rt_.grass_base_ao_floor        = std::stof(val); ++loaded; continue; }
            if (key == "grass_ground_tint_strength") { rt_.grass_ground_tint_strength = std::stof(val); ++loaded; continue; }
            if (key == "grass_shadow_strength")      { rt_.grass_shadow_strength      = std::stof(val); ++loaded; continue; }
            // SPOM strength lives in the flat fast-path block, NOT in
            // the if/else-if chain below — adding one more `else if`
            // there exceeds MSVC's nested-block limit (C1061).
            if (key == "spom_strength")              { rt_.spom_strength              = std::stof(val); ++loaded; continue; }
            // Near-terrain tessellation — flat fast-path (NOT the else-if
            // chain below; one more `else if` there hits C1061).
            if (key == "terrain_tessellation_enabled") { rt_.terrain_tessellation_enabled = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "terrain_tess_range")          { rt_.terrain_tess_range          = std::stof(val); ++loaded; continue; }
            if (key == "terrain_wireframe")           { rt_.terrain_wireframe           = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "terrain_tess_max_level")      { rt_.terrain_tess_max_level      = std::stof(val); ++loaded; continue; }
            if (key == "terrain_tess_near_m")         { rt_.terrain_tess_near_m         = std::stof(val); ++loaded; continue; }
            if (key == "terrain_tess_far_m")          { rt_.terrain_tess_far_m          = std::stof(val); ++loaded; continue; }
            if (key == "terrain_tess_falloff")        { rt_.terrain_tess_falloff        = std::stof(val); ++loaded; continue; }
            if (key == "terrain_tess_smooth")         { rt_.terrain_tess_smooth         = std::stof(val); ++loaded; continue; }
            if (key == "terrain_pom_strength")        { rt_.terrain_pom_strength        = std::stof(val); ++loaded; continue; }
            if (key == "terrain_disp_amp")            { rt_.terrain_disp_amp            = std::stof(val); ++loaded; continue; }
            if (key == "terrain_disp_smooth_mip")     { rt_.terrain_disp_smooth_mip     = std::stof(val); ++loaded; continue; }
            if (key == "terrain_pom_far_m")           { rt_.terrain_pom_far_m           = std::stof(val); ++loaded; continue; }
            if (key == "terrain_sand_ripple_scale")   { rt_.terrain_sand_ripple_scale   = std::stof(val); ++loaded; continue; }
            if (key == "terrain_grass_line_scale")    { rt_.terrain_grass_line_scale    = std::stof(val); ++loaded; continue; }
            if (key == "water_clarity_depth")         { rt_.water_clarity_depth         = std::stof(val); ++loaded; continue; }
            if (key == "water_shore_softness")        { rt_.water_shore_softness        = std::stof(val); ++loaded; continue; }
            if (key == "water_foam_opacity")          { rt_.water_foam_opacity          = std::stof(val); ++loaded; continue; }
            if (key == "terrain_rock_relief")         { rt_.terrain_rock_relief         = std::stof(val); ++loaded; continue; }
            if (key == "ground_mat_strength")         { rt_.ground_mat_strength         = std::stof(val); ++loaded; continue; }
            if (key == "ground_mat_tile_m")           { rt_.ground_mat_tile_m           = std::stof(val); ++loaded; continue; }
            if (key == "ground_mat_normal")           { rt_.ground_mat_normal           = std::stof(val); ++loaded; continue; }
            if (key == "terrain_antitile")            { rt_.terrain_antitile            = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "terrain_antitile_strength")   { rt_.terrain_antitile_strength   = std::stof(val); ++loaded; continue; }
            if (key == "grass_shadow_on_terrain")     { rt_.grass_shadow_on_terrain     = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "grass_shadow_on_terrain_strength") { rt_.grass_shadow_on_terrain_strength = std::stof(val); ++loaded; continue; }
            if (key == "grass_shadow_on_terrain_samples")  { rt_.grass_shadow_on_terrain_samples  = std::stoi(val); ++loaded; continue; }
            if (key == "grass_shadow_on_terrain_dist")     { rt_.grass_shadow_on_terrain_dist     = std::stof(val); ++loaded; continue; }
            if (key == "grass_side_lit_enabled")      { rt_.grass_side_lit_enabled      = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "grass_side_lit_strength")     { rt_.grass_side_lit_strength     = std::stof(val); ++loaded; continue; }
            if (key == "grass_shadow_samples")       { rt_.grass_shadow_samples       = std::stoi(val); ++loaded; continue; }
            if (key == "half_rate_shadows")          { rt_.half_rate_shadows          = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "sun_shaft_intensity")        { rt_.sun_shaft_intensity        = std::stof(val); ++loaded; continue; }
            if (key == "auto_golden_hour")           { rt_.auto_golden_hour           = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "sun_glare_strength")         { rt_.sun_glare_strength         = std::stof(val); ++loaded; continue; }
            if (key == "svgf_enabled")               { rt_.svgf_enabled               = std::stoi(val) != 0; ++loaded; continue; }
            if (key == "grass_shadow_max_dist")      { rt_.grass_shadow_max_dist      = std::stof(val); ++loaded; continue; }
            if      (key == "sun_pitch_deg")       rt_.sun_pitch_deg = std::stof(val);
            else if (key == "sun_yaw_deg")         rt_.sun_yaw_deg = std::stof(val);
            else if (key == "sun_intensity")       rt_.sun_intensity = std::stof(val);
            else if (key == "sun_color" ||
                     key == "sky_color" ||
                     key == "ground_ambient") {
                glm::vec3 v;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    if      (key == "sun_color")      rt_.sun_color = v;
                    else if (key == "sky_color")      rt_.sky_color = v;
                    else                              rt_.ground_ambient = v;
                }
            }
            else if (key == "ambient_strength")    rt_.ambient_strength = std::stof(val);
            else if (key == "shadow_strength")     rt_.shadow_strength = std::stof(val);
            else if (key == "shadow_enabled")      rt_.shadow_enabled = std::stoi(val) != 0;
            else if (key == "shadow_samples")      rt_.shadow_samples = std::stoi(val);
            else if (key == "shadow_softness")     rt_.shadow_softness = std::stof(val);
            else if (key == "shadow_curve")        rt_.shadow_curve = std::stof(val);
            else if (key == "ao_samples")          rt_.ao_samples = std::stoi(val);
            else if (key == "ao_radius")           rt_.ao_radius = std::stof(val);
            else if (key == "gi_samples")          rt_.gi_samples = std::stoi(val);
            else if (key == "gi_bounces")          rt_.gi_bounces = std::stoi(val);
            else if (key == "gi_shadow_max_bounce") rt_.gi_shadow_max_bounce = std::stoi(val);
            else if (key == "ao_floor")             rt_.ao_floor = std::stof(val);
            else if (key == "auto_exposure_strength") rt_.auto_exposure_strength = std::stof(val);
            else if (key == "use_merged_static_blas") rt_.use_merged_static_blas = (std::stoi(val) != 0);
            else if (key == "terrain_brush_radius")   terrain_brush_radius_   = std::stof(val);
            else if (key == "terrain_brush_strength") terrain_brush_strength_ = std::stof(val);
            else if (key == "terrain_brush_mode") {
                // Enum widened from 4 → 8 (added GrassAdd/Remove + Erode +/Smooth).
                // Keep the upper bound in sync with TerrainBrushMode::ErodeSmooth.
                int m = std::stoi(val);
                if (m >= 0 && m <= 7) terrain_brush_mode_ = static_cast<TerrainBrushMode>(m);
            }
            else if (key == "terrain_brush_flatten_target") terrain_brush_flatten_target_ = std::stof(val);
            else if (key == "terrain_fog_strength")    rt_.terrain_fog_strength    = std::stof(val);
            else if (key == "terrain_wrap_strength")   rt_.terrain_wrap_strength   = std::stof(val);
            else if (key == "terrain_detail_strength") rt_.terrain_detail_strength = std::stof(val);
            else if (key == "terrain_shadow_softness_scale") rt_.terrain_shadow_softness_scale = std::stof(val);
            else if (key == "terrain_h_sand_grass_start") rt_.terrain_h_sand_grass_start = std::stof(val);
            else if (key == "terrain_h_sand_grass_end")   rt_.terrain_h_sand_grass_end   = std::stof(val);
            else if (key == "terrain_h_grass_dirt_start") rt_.terrain_h_grass_dirt_start = std::stof(val);
            else if (key == "terrain_h_grass_dirt_end")   rt_.terrain_h_grass_dirt_end   = std::stof(val);
            else if (key == "terrain_h_dirt_rock_start")  rt_.terrain_h_dirt_rock_start  = std::stof(val);
            else if (key == "terrain_h_dirt_rock_end")    rt_.terrain_h_dirt_rock_end    = std::stof(val);
            else if (key == "terrain_h_rock_snow_start")  rt_.terrain_h_rock_snow_start  = std::stof(val);
            else if (key == "terrain_h_rock_snow_end")    rt_.terrain_h_rock_snow_end    = std::stof(val);
            else if (key == "grass_enabled")      rt_.grass_enabled       = (std::stoi(val) != 0);
            else if (key == "grass_raymarch_enabled") rt_.grass_raymarch_enabled = (std::stoi(val) != 0);
            else if (key == "grass_raymarch_distance") rt_.grass_raymarch_distance = std::stof(val);
            else if (key == "grass_distance")     rt_.grass_distance      = std::stof(val);
            else if (key == "grass_wind")         rt_.grass_wind          = std::stof(val);
            else if (key == "grass_density")      rt_.grass_density       = std::stof(val);
            else if (key == "grass_height_scale") rt_.grass_height_scale  = std::stof(val);
            else if (key == "grass_alpha_cutoff") rt_.grass_alpha_cutoff  = std::stof(val);
            else if (key == "grass_cutoff_soft_dist") rt_.grass_cutoff_soft_dist = std::stof(val);
            else if (key == "grass_slope_n_min")  rt_.grass_slope_n_min   = std::stof(val);
            else if (key == "grass_distance_density") rt_.grass_distance_density = std::stof(val);
            else if (key == "grass_alt_min")  rt_.grass_alt_min  = std::stof(val);
            else if (key == "grass_alt_max")  rt_.grass_alt_max  = std::stof(val);
            else if (key == "shadow_map_resolution") rt_.shadow_map_resolution = std::stoi(val);
            else if (key == "shadow_map_world_half") rt_.shadow_map_world_half = std::stof(val);
            else if (key == "shadow_debug_overlay")  rt_.shadow_debug_overlay  = std::stoi(val) != 0;
            // Per-LOD ladder + back-compat keys live in the fast-path
            // block above (apply_terrain_lod_key + the
            // `terrain_lod1/2/3` continue checks). Adding even one
            // else-if here triggers MSVC C1061 because the chain is
            // already at the nested-block depth limit.
            else if (key == "terrain_lod_scale") rt_.terrain_lod_scale = std::stof(val);
            else if (key == "terrain_bake_supersample") rt_.terrain_bake_supersample = std::stoi(val);
            else if (key == "terrain_shading_contrast") rt_.terrain_shading_contrast = std::stof(val);
            else if (key == "terrain_heightmap_scale") rt_.terrain_heightmap_scale = std::stoi(val);
            else if (key == "terrain_raymarch_enabled") rt_.terrain_raymarch_enabled = std::stoi(val) != 0;
            else if (key == "terrain_raymarch_max_steps") rt_.terrain_raymarch_max_steps = std::stoi(val);
            else if (key == "terrain_raymarch_shadow_steps") rt_.terrain_raymarch_shadow_steps = std::stoi(val);
            else if (key == "terrain_raymarch_octaves") rt_.terrain_raymarch_octaves = std::stoi(val);
            else if (key == "terrain_raymarch_normal_octaves") rt_.terrain_raymarch_normal_octaves = std::stoi(val);
            else if (key == "taau_enabled") rt_.taau_enabled = std::stoi(val) != 0;
            else if (key == "fsr2_enabled") rt_.fsr2_enabled = std::stoi(val) != 0;
            else if (key == "terrain_raymarch_step_factor") rt_.terrain_raymarch_step_factor = std::stof(val);
            else if (key == "terrain_raymarch_lod_near_m") rt_.terrain_raymarch_lod_near_m = std::stof(val);
            else if (key == "terrain_raymarch_lod_far_m") rt_.terrain_raymarch_lod_far_m = std::stof(val);
            else if (key == "terrain_raymarch_lod_min_octaves") rt_.terrain_raymarch_lod_min_octaves = std::stoi(val);
            else if (key == "terrain_raymarch_scale") rt_.terrain_raymarch_scale = std::stof(val);
            else if (key == "terrain_raymarch_sharpen") rt_.terrain_raymarch_sharpen = std::stof(val);
            else if (key == "terrain_raymarch_fog_strength") rt_.terrain_raymarch_fog_strength = std::stof(val);
            else if (key == "terrain_raymarch_relaxation") rt_.terrain_raymarch_relaxation = std::stoi(val) != 0;
            else if (key == "terrain_raymarch_fog_godrays") rt_.terrain_raymarch_fog_godrays = std::stoi(val) != 0;
            else if (key == "terrain_raymarch_fog_y_start") rt_.terrain_raymarch_fog_y_start = std::stof(val);
            else if (key == "terrain_raymarch_fog_y_top")   rt_.terrain_raymarch_fog_y_top   = std::stof(val);
            else if (key == "terrain_raymarch_fog_noise")   rt_.terrain_raymarch_fog_noise   = std::stof(val);
            else if (key == "water_enabled")        rt_.water_enabled = std::stoi(val) != 0;
            else if (key == "water_level")          rt_.water_level = std::stof(val);
            else if (key == "water_wave_strength")  rt_.water_wave_strength = std::stof(val);
            else if (key == "water_wave_scale")     rt_.water_wave_scale = std::stof(val);
            else if (key == "water_rt_reflections") rt_.water_rt_reflections = std::stoi(val) != 0;
            else if (key == "water_tlas_reflections") rt_.water_tlas_reflections = std::stoi(val) != 0;
            else if (key == "water_shore_blend") rt_.water_shore_blend = std::stof(val);
            else if (key == "water_shore_noise") rt_.water_shore_noise = std::stof(val);
            else if (key == "water_shadows_enabled") rt_.water_shadows_enabled = std::stoi(val) != 0;
            else if (key == "water_transparency") rt_.water_transparency = std::stof(val);
            else if (key == "water_color_shallow") {
                glm::vec3 v;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.water_color_shallow = v;
                }
            }
            else if (key == "water_color") {
                glm::vec3 v;
                if (std::sscanf(val.c_str(), "%f %f %f", &v.x, &v.y, &v.z) == 3) {
                    rt_.water_color = v;
                }
            }
            else if (key == "shadow_near_mult") rt_.shadow_near_mult = std::stof(val);
            else if (key == "gi_strength")         rt_.gi_strength = std::stof(val);
            else if (key == "gi_radius")           rt_.gi_radius = std::stof(val);
            else if (key == "gi_softener")         rt_.gi_softener = std::stof(val);
            else if (key == "reflections_enabled") rt_.reflections_enabled = std::stoi(val) != 0;
            else if (key == "reflection_strength") rt_.reflection_strength = std::stof(val);
            else if (key == "taa_history_blend")    rt_.taa_history_blend = std::stof(val);
            else if (key == "taa_spatial_strength") rt_.taa_spatial_strength = std::stof(val);
            else if (key == "bloom_enabled")        rt_.bloom_enabled = std::stoi(val) != 0;
            else if (key == "bloom_strength")       rt_.bloom_strength = std::stof(val);
            else if (key == "bloom_threshold")      rt_.bloom_threshold = std::stof(val);
            else if (key == "bloom_radius")         rt_.bloom_radius = std::stof(val);
            else if (key == "spark_bloom")          rt_.spark_bloom = std::stof(val);
            else if (key == "taa_jitter_enabled")   rt_.taa_jitter_enabled = std::stoi(val) != 0;
            else if (key == "textures_enabled")     rt_.textures_enabled   = std::stoi(val) != 0;
            else if (key == "lens_flare_enabled")   rt_.lens_flare_enabled = std::stoi(val) != 0;
            else if (key == "lens_flare_strength")  rt_.lens_flare_strength = std::stof(val);
            else if (key == "lens_flare_threshold") rt_.lens_flare_threshold = std::stof(val);
            else if (key == "lens_flare_dispersal") rt_.lens_flare_dispersal = std::stof(val);
            else if (key == "lens_flare_halo_width")rt_.lens_flare_halo_width = std::stof(val);
            else if (key == "lens_flare_ghosts")    rt_.lens_flare_ghosts = std::stoi(val);
            else if (key == "lens_flare_aberration")rt_.lens_flare_aberration = std::stof(val);
            else if (key == "taa_jitter_strength")  rt_.taa_jitter_strength = std::stof(val);
            else if (key == "compose_sharpen_strength") rt_.compose_sharpen_strength = std::stof(val);
            // (image_contrast, image_brightness, image_gamma,
            //  terrain_rt_lod_distance, terrain_ao_punch are handled
            //  in the flat fast-path above.)
            else if (key == "render_scale")        rt_.render_scale = std::stof(val);
            else if (key == "quality_preset")      rt_.quality_preset = std::stoi(val);
            else if (key == "ao_mode")             rt_.ao_mode = std::stoi(val);
            else if (key == "gravity")              game_.gravity = std::stof(val);
            else if (key == "cubes_per_minute")     game_.cubes_per_minute = std::stoi(val);
            else if (key == "bullet_mass")          game_.bullet_mass = std::stof(val);
            else if (key == "fire_rate_rps")        game_.fire_rate_rps = std::stof(val);
            else if (key == "bullet_speed")         game_.bullet_speed = std::stof(val);
            else if (key == "spark_scale")          game_.spark_scale = std::stof(val);
            else {
                log::warnf("settings: unknown key '%s' (ignored)", key.c_str());
                continue;
            }
            ++loaded;
        } catch (...) {
            log::warnf("settings: failed to parse '%s' = '%s'",
                       key.c_str(), val.c_str());
        }
    }
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
    clampf(rt_.render_scale,     0.4f,    2.5f);
    clampi(rt_.quality_preset,  -1,         3);  // -1 = custom
    clampi(rt_.ao_mode,          0,         3);  // 0=off,1=fast,2=RTAO,3=HBAO
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
