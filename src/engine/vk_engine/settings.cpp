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
        break;
    default:
        rt_.quality_preset = -1; // unknown → mark custom
        return;
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
    f << "gi_shadow_max_bounce = " << rt_.gi_shadow_max_bounce << "\n";
    f << "ao_floor = "             << rt_.ao_floor             << "\n";
    f << "auto_exposure_strength = " << rt_.auto_exposure_strength << "\n";
    f << "use_merged_static_blas = " << (rt_.use_merged_static_blas ? 1 : 0) << "\n";
    f << "terrain_brush_radius = "   << terrain_brush_radius_   << "\n";
    f << "terrain_brush_strength = " << terrain_brush_strength_ << "\n";
    f << "terrain_brush_mode = "     << static_cast<int>(terrain_brush_mode_) << "\n";
    f << "terrain_brush_flatten_target = " << terrain_brush_flatten_target_ << "\n";
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
    f << "grass_distance = "      << rt_.grass_distance          << "\n";
    f << "grass_wind = "          << rt_.grass_wind              << "\n";
    f << "grass_density = "       << rt_.grass_density           << "\n";
    f << "grass_height_scale = "  << rt_.grass_height_scale      << "\n";
    f << "grass_alpha_cutoff = "  << rt_.grass_alpha_cutoff      << "\n";
    f << "grass_slope_n_min = "   << rt_.grass_slope_n_min       << "\n";
    f << "grass_distance_density = " << rt_.grass_distance_density << "\n";
    f << "grass_alt_min = " << rt_.grass_alt_min << "\n";
    f << "grass_alt_max = " << rt_.grass_alt_max << "\n";
    f << "shadow_map_resolution = " << rt_.shadow_map_resolution << "\n";
    f << "shadow_map_world_half = " << rt_.shadow_map_world_half << "\n";
    f << "shadow_debug_overlay = "  << (rt_.shadow_debug_overlay ? 1 : 0) << "\n";
    f << "terrain_lod1 = " << rt_.terrain_lod1 << "\n";
    f << "terrain_lod2 = " << rt_.terrain_lod2 << "\n";
    f << "terrain_lod3 = " << rt_.terrain_lod3 << "\n";
    f << "terrain_lod_scale = " << rt_.terrain_lod_scale << "\n";
    f << "terrain_bake_supersample = " << rt_.terrain_bake_supersample << "\n";
    f << "terrain_shading_contrast = " << rt_.terrain_shading_contrast << "\n";
    f << "terrain_heightmap_scale = " << rt_.terrain_heightmap_scale << "\n";
    f << "terrain_raymarch_enabled = " << (rt_.terrain_raymarch_enabled ? 1 : 0) << "\n";
    f << "terrain_raymarch_max_steps = " << rt_.terrain_raymarch_max_steps << "\n";
    f << "terrain_raymarch_shadow_steps = " << rt_.terrain_raymarch_shadow_steps << "\n";
    f << "terrain_raymarch_octaves = " << rt_.terrain_raymarch_octaves << "\n";
    f << "terrain_raymarch_normal_octaves = " << rt_.terrain_raymarch_normal_octaves << "\n";
    f << "terrain_raymarch_step_factor = " << rt_.terrain_raymarch_step_factor << "\n";
    f << "terrain_raymarch_scale = " << rt_.terrain_raymarch_scale << "\n";
    f << "shadow_near_mult = " << rt_.shadow_near_mult << "\n";
    f << "gi_strength = "        << rt_.gi_strength        << "\n";
    f << "gi_radius = "          << rt_.gi_radius          << "\n";
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
                int m = std::stoi(val);
                if (m >= 0 && m <= 3) terrain_brush_mode_ = static_cast<TerrainBrushMode>(m);
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
            else if (key == "grass_distance")     rt_.grass_distance      = std::stof(val);
            else if (key == "grass_wind")         rt_.grass_wind          = std::stof(val);
            else if (key == "grass_density")      rt_.grass_density       = std::stof(val);
            else if (key == "grass_height_scale") rt_.grass_height_scale  = std::stof(val);
            else if (key == "grass_alpha_cutoff") rt_.grass_alpha_cutoff  = std::stof(val);
            else if (key == "grass_slope_n_min")  rt_.grass_slope_n_min   = std::stof(val);
            else if (key == "grass_distance_density") rt_.grass_distance_density = std::stof(val);
            else if (key == "grass_alt_min")  rt_.grass_alt_min  = std::stof(val);
            else if (key == "grass_alt_max")  rt_.grass_alt_max  = std::stof(val);
            else if (key == "shadow_map_resolution") rt_.shadow_map_resolution = std::stoi(val);
            else if (key == "shadow_map_world_half") rt_.shadow_map_world_half = std::stof(val);
            else if (key == "shadow_debug_overlay")  rt_.shadow_debug_overlay  = std::stoi(val) != 0;
            else if (key == "terrain_lod1")  rt_.terrain_lod1 = std::stof(val);
            else if (key == "terrain_lod2")  rt_.terrain_lod2 = std::stof(val);
            else if (key == "terrain_lod3")  rt_.terrain_lod3 = std::stof(val);
            else if (key == "terrain_lod_scale") rt_.terrain_lod_scale = std::stof(val);
            else if (key == "terrain_bake_supersample") rt_.terrain_bake_supersample = std::stoi(val);
            else if (key == "terrain_shading_contrast") rt_.terrain_shading_contrast = std::stof(val);
            else if (key == "terrain_heightmap_scale") rt_.terrain_heightmap_scale = std::stoi(val);
            else if (key == "terrain_raymarch_enabled") rt_.terrain_raymarch_enabled = std::stoi(val) != 0;
            else if (key == "terrain_raymarch_max_steps") rt_.terrain_raymarch_max_steps = std::stoi(val);
            else if (key == "terrain_raymarch_shadow_steps") rt_.terrain_raymarch_shadow_steps = std::stoi(val);
            else if (key == "terrain_raymarch_octaves") rt_.terrain_raymarch_octaves = std::stoi(val);
            else if (key == "terrain_raymarch_normal_octaves") rt_.terrain_raymarch_normal_octaves = std::stoi(val);
            else if (key == "terrain_raymarch_step_factor") rt_.terrain_raymarch_step_factor = std::stof(val);
            else if (key == "terrain_raymarch_scale") rt_.terrain_raymarch_scale = std::stof(val);
            else if (key == "shadow_near_mult") rt_.shadow_near_mult = std::stof(val);
            else if (key == "gi_strength")         rt_.gi_strength = std::stof(val);
            else if (key == "gi_radius")           rt_.gi_radius = std::stof(val);
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
            else if (key == "render_scale")        rt_.render_scale = std::stof(val);
            else if (key == "quality_preset")      rt_.quality_preset = std::stoi(val);
            else if (key == "ao_mode")             rt_.ao_mode = std::stoi(val);
            else if (key == "gravity")              game_.gravity = std::stof(val);
            else if (key == "cubes_per_minute")     game_.cubes_per_minute = std::stoi(val);
            else if (key == "bullet_mass")          game_.bullet_mass = std::stof(val);
            else if (key == "fire_rate_rps")        game_.fire_rate_rps = std::stof(val);
            else if (key == "bullet_speed")         game_.bullet_speed = std::stof(val);
            else if (key == "spark_scale")          game_.spark_scale = std::stof(val);
            // Player pose — restored after init_world resets defaults.
            else if (key == "player_pos_x")         player_.position.x = std::stof(val);
            else if (key == "player_pos_y")         player_.position.y = std::stof(val);
            else if (key == "player_pos_z")         player_.position.z = std::stof(val);
            else if (key == "player_yaw")           player_.yaw        = std::stof(val);
            else if (key == "player_pitch")         player_.pitch      = std::stof(val);
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
    clampf(rt_.reflection_strength, 0.0f, 1.0f);
    clampf(rt_.taa_history_blend,   0.0f, 0.98f);
    clampf(rt_.taa_spatial_strength,0.0f, 1.0f);
    clampf(rt_.bloom_strength,   0.0f,    4.0f);
    clampf(rt_.bloom_threshold,  0.0f,    8.0f);
    clampf(rt_.bloom_radius,     1.0f,  256.0f);
    clampf(rt_.spark_bloom,      0.0f,    8.0f);
    clampf(rt_.taa_jitter_strength, 0.0f, 4.0f);
    clampf(rt_.compose_sharpen_strength, 0.0f, 2.5f);
    clampf(rt_.render_scale,     0.4f,    2.5f);
    clampi(rt_.quality_preset,  -1,         3);  // -1 = custom
    clampi(rt_.ao_mode,          0,         2);
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
