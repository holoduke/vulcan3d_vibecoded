// VulkanEngine UI — ImGui setup, pause menu, HUD overlay, crosshair.
// Split out of vk_engine.cpp because UI is independent of render-state plumbing
// and grew large enough to crowd the main engine file.

#include "engine/vk_engine.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"   // qlike::vk_check
#include "engine/window.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <cstddef>

namespace qlike {

void VulkanEngine::init_imgui() {
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
    };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 1000;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = sizes;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &imgui_pool_),
             "imgui pool");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Keyboard nav DISABLED. With nav on, ImGui can auto-focus a
    // pause-menu button when the menu opens, then propagate further
    // key events to it — pressing ESC over the pause menu was
    // landing on the Quit button and exiting the game. Mouse-only
    // for the menu is fine here; no controller use.
    io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard;
    // Pin imgui.ini next to qlike_settings.cfg so menu collapse /
    // expand / position state persists across restarts regardless
    // of the launch cwd. Default behaviour writes "imgui.ini" in
    // cwd, which means launching from build/ vs the repo root used
    // two different files and the saved state didn't follow you.
    io.IniFilename = "qlike_imgui.ini";
    ImGui::StyleColorsDark();

    // ===== Synthwave style ============================================
    // Neon magenta + cyan on a deep indigo background — pause menu /
    // HUD reads as a 1980s CRT title screen. Just a colour + spacing
    // override; no custom shaders, no extra geometry.
    {
        ImGuiStyle& s = ImGui::GetStyle();
        s.WindowRounding   = 6.0f;
        s.FrameRounding    = 4.0f;
        s.GrabRounding     = 4.0f;
        s.TabRounding      = 4.0f;
        s.PopupRounding    = 4.0f;
        s.ScrollbarRounding= 6.0f;
        s.WindowPadding    = ImVec2(14.0f, 12.0f);
        s.FramePadding     = ImVec2(8.0f, 5.0f);
        s.ItemSpacing      = ImVec2(10.0f, 8.0f);
        s.WindowBorderSize = 1.5f;
        s.FrameBorderSize  = 0.0f;
        s.TabBorderSize    = 0.0f;
        s.IndentSpacing    = 16.0f;

        const ImVec4 magenta      (0.95f, 0.20f, 0.85f, 1.00f);
        const ImVec4 magentaSoft  (0.65f, 0.15f, 0.60f, 1.00f);
        const ImVec4 magentaDeep  (0.20f, 0.05f, 0.20f, 1.00f);
        const ImVec4 cyan         (0.20f, 0.95f, 0.95f, 1.00f);
        const ImVec4 cyanSoft     (0.15f, 0.55f, 0.65f, 1.00f);
        const ImVec4 indigoBg     (0.05f, 0.03f, 0.10f, 0.96f);
        const ImVec4 indigoFrame  (0.10f, 0.07f, 0.18f, 1.00f);
        const ImVec4 indigoFrameHi(0.18f, 0.12f, 0.32f, 1.00f);
        const ImVec4 textBright   (0.95f, 0.92f, 1.00f, 1.00f);
        const ImVec4 textDim      (0.70f, 0.78f, 0.95f, 1.00f);

        ImVec4* c = s.Colors;
        c[ImGuiCol_Text]                 = textBright;
        c[ImGuiCol_TextDisabled]         = textDim;
        c[ImGuiCol_WindowBg]             = indigoBg;
        c[ImGuiCol_PopupBg]              = indigoBg;
        c[ImGuiCol_Border]               = magenta;
        c[ImGuiCol_BorderShadow]         = ImVec4(0,0,0,0);
        c[ImGuiCol_FrameBg]              = indigoFrame;
        c[ImGuiCol_FrameBgHovered]       = indigoFrameHi;
        c[ImGuiCol_FrameBgActive]        = ImVec4(0.30f, 0.18f, 0.50f, 1.0f);
        c[ImGuiCol_TitleBg]              = magentaDeep;
        c[ImGuiCol_TitleBgActive]        = magentaSoft;
        c[ImGuiCol_TitleBgCollapsed]     = magentaDeep;
        c[ImGuiCol_MenuBarBg]            = magentaDeep;
        c[ImGuiCol_ScrollbarBg]          = ImVec4(0.05f, 0.03f, 0.12f, 0.6f);
        c[ImGuiCol_ScrollbarGrab]        = magentaSoft;
        c[ImGuiCol_ScrollbarGrabHovered] = magenta;
        c[ImGuiCol_ScrollbarGrabActive]  = cyan;
        c[ImGuiCol_CheckMark]            = cyan;
        c[ImGuiCol_SliderGrab]           = cyan;
        c[ImGuiCol_SliderGrabActive]     = magenta;
        c[ImGuiCol_Button]               = ImVec4(0.30f, 0.10f, 0.40f, 1.0f);
        c[ImGuiCol_ButtonHovered]        = magenta;
        c[ImGuiCol_ButtonActive]         = cyan;
        c[ImGuiCol_Header]               = magentaSoft;
        c[ImGuiCol_HeaderHovered]        = magenta;
        c[ImGuiCol_HeaderActive]         = cyan;
        c[ImGuiCol_Separator]            = magentaSoft;
        c[ImGuiCol_SeparatorHovered]     = cyan;
        c[ImGuiCol_SeparatorActive]      = cyan;
        c[ImGuiCol_ResizeGrip]           = magentaSoft;
        c[ImGuiCol_ResizeGripHovered]    = magenta;
        c[ImGuiCol_ResizeGripActive]     = cyan;
        c[ImGuiCol_Tab]                  = ImVec4(0.18f, 0.08f, 0.30f, 1.0f);
        c[ImGuiCol_TabHovered]           = magenta;
        c[ImGuiCol_TabActive]            = ImVec4(0.55f, 0.12f, 0.55f, 1.0f);
        c[ImGuiCol_TabUnfocused]         = ImVec4(0.10f, 0.05f, 0.18f, 1.0f);
        c[ImGuiCol_TabUnfocusedActive]   = magentaSoft;
        c[ImGuiCol_PlotLines]            = cyan;
        c[ImGuiCol_PlotLinesHovered]     = magenta;
        c[ImGuiCol_PlotHistogram]        = magenta;
        c[ImGuiCol_PlotHistogramHovered] = cyan;
        c[ImGuiCol_DragDropTarget]       = cyan;
        c[ImGuiCol_NavHighlight]         = cyan;
    }
    // ===================================================================

    ImGui_ImplGlfw_InitForVulkan(window_->handle(), true);

    VkFormat color_fmt = swapchain_format_;
    VkPipelineRenderingCreateInfo prci{};
    prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount = 1;
    prci.pColorAttachmentFormats = &color_fmt;
    prci.depthAttachmentFormat = depth_format_;

    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = instance_;
    ii.PhysicalDevice = physical_device_;
    ii.Device = device_;
    ii.QueueFamily = graphics_queue_family_;
    ii.Queue = graphics_queue_;
    ii.DescriptorPool = imgui_pool_;
    ii.MinImageCount = 2;
    ii.ImageCount = static_cast<uint32_t>(swapchain_images_.size());
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = true;
    ii.PipelineRenderingCreateInfo = prci;
    ImGui_ImplVulkan_Init(&ii);

    imgui_initialized_ = true;
    log::info("ImGui initialized (Vulkan, dynamic rendering)");
}

void VulkanEngine::destroy_imgui() {
    if (!imgui_initialized_) return;
    // Force-flush settings to disk before tearing down the context.
    // ImGui's auto-save runs on a timer (default 5 s) and would
    // otherwise lose state if the user quits within a few seconds of
    // collapsing/moving a window.
    ImGuiIO& io = ImGui::GetIO();
    if (io.IniFilename) {
        ImGui::SaveIniSettingsToDisk(io.IniFilename);
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (imgui_pool_) vkDestroyDescriptorPool(device_, imgui_pool_, nullptr);
    imgui_pool_ = VK_NULL_HANDLE;
    imgui_initialized_ = false;
}

void VulkanEngine::build_hud_ui() {
    if (state_ != State::Playing) return;

    // Push current frame samples into the rolling history (Three.js
    // stats-style ring buffer).
    {
        int& head = stats_history_head_;
        fps_history_[head] = ema_fps_;
        ms_history_[head]  = last_frame_dt_ * 1000.0f;
        head = (head + 1) % kStatsHistory;
    }

    // Helper: render one Three.js-style stats panel — big number on
    // top, mini histogram below, accent colour bar. ImDrawList for
    // the custom panel + ImGui for text inside the panel rect.
    auto draw_panel = [&](ImVec2 origin, const char* label,
                          float current, float min_v, float max_v,
                          const float* history, int history_count, int head,
                          ImU32 accent) {
        const ImVec2 size(120.0f, 48.0f);
        const ImU32 bg = IM_COL32(0, 0, 16, 220);
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        dl->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), bg);
        // Top accent strip — colour identifies which stat panel.
        dl->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + 2.0f), accent);
        // Big current value, top-left.
        char buf[48];
        snprintf(buf, sizeof(buf), "%.0f %s", current, label);
        dl->AddText(ImVec2(origin.x + 4.0f, origin.y + 4.0f), accent, buf);
        // Min/Max line below the big number.
        snprintf(buf, sizeof(buf), "(%.0f-%.0f)", min_v, max_v);
        dl->AddText(ImVec2(origin.x + 4.0f, origin.y + 18.0f),
                    IM_COL32(200, 200, 220, 180), buf);
        // Histogram strip at the bottom of the panel.
        const float graph_y      = origin.y + 32.0f;
        const float graph_h      = 14.0f;
        const float graph_x_left = origin.x + 4.0f;
        const float graph_w      = size.x - 8.0f;
        // Span = max-min, padded so a flat history shows a clear baseline.
        const float span = std::max(1.0f, max_v - min_v);
        const float bar_w = graph_w / float(history_count);
        for (int i = 0; i < history_count; ++i) {
            // Read chronologically: oldest at head, newest at head-1.
            int idx = (head + i) % history_count;
            float v = history[idx];
            float h = std::min(1.0f, std::max(0.0f, (v - min_v) / span)) * graph_h;
            float bx0 = graph_x_left + float(i) * bar_w;
            float bx1 = bx0 + bar_w + 0.5f;
            dl->AddRectFilled(ImVec2(bx0, graph_y + graph_h - h),
                              ImVec2(bx1, graph_y + graph_h),
                              accent);
        }
    };

    // Stats panels — stacked vertically in the top-left.
    {
        // Min/max over the live history.
        float fps_min = fps_history_[0], fps_max = fps_history_[0];
        float ms_min  = ms_history_[0],  ms_max  = ms_history_[0];
        for (int i = 1; i < kStatsHistory; ++i) {
            fps_min = std::min(fps_min, fps_history_[i]);
            fps_max = std::max(fps_max, fps_history_[i]);
            ms_min  = std::min(ms_min,  ms_history_[i]);
            ms_max  = std::max(ms_max,  ms_history_[i]);
        }
        ImVec2 p(8.0f, 8.0f);
        draw_panel(p, "FPS", ema_fps_, fps_min, fps_max,
                   fps_history_, kStatsHistory, stats_history_head_,
                   IM_COL32(180, 220, 70, 255));  // green
        p.y += 52.0f;
        draw_panel(p, "MS", last_frame_dt_ * 1000.0f, ms_min, ms_max,
                   ms_history_, kStatsHistory, stats_history_head_,
                   IM_COL32(220, 110, 180, 255));  // pink
    }

    // Compact gameplay info — small ImGui window below the stats
    // panels. Kept minimal: position, score, draw counts, stance.
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowPos(ImVec2(8.0f, 8.0f + 52.0f * 2 + 4.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::Begin("##hud_info", nullptr, flags);
    glm::vec2 hv(player_.velocity.x, player_.velocity.z);
    float speed_h = glm::length(hv);
    const char* stance = "walk";
    if (speed_h > 11.0f)      stance = "SPRINT";
    else if (speed_h < 4.0f && (player_.velocity.x != 0.0f ||
                                 player_.velocity.z != 0.0f))
        stance = "crawl";
    ImGui::Text("pos %5.1f %5.1f %5.1f",
                player_.position.x, player_.position.y, player_.position.z);
    ImGui::Text("spd %5.1f m/s  %s%s", speed_h, stance,
                player_.on_ground ? "" : "  (air)");
    ImGui::Text("score %d  dyn %zu/%d", score_, dyn_props_.size(), kMaxDynProps);
    ImGui::Text("draws %d+%d  culled %d  phys %d",
                last_draw_static_, last_draw_dyn_, last_culled_,
                last_physics_ticks_);
    const uint64_t v_warn = g_validation_warning_count.load(std::memory_order_relaxed);
    const uint64_t v_err  = g_validation_error_count.load(std::memory_order_relaxed);
    if (v_err > 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "vk %llu warn / %llu ERR",
                           static_cast<unsigned long long>(v_warn),
                           static_cast<unsigned long long>(v_err));
    } else if (v_warn > 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.4f, 1.0f),
                           "vk %llu warn",
                           static_cast<unsigned long long>(v_warn));
    }
    ImGui::End();

    // Crosshair: small dot + tick marks at screen center.
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 cen(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImU32 col = IM_COL32(255, 255, 255, 220);
    const ImU32 col_shadow = IM_COL32(0, 0, 0, 180);
    dl->AddCircleFilled(ImVec2(cen.x, cen.y), 2.5f, col_shadow);
    dl->AddCircleFilled(ImVec2(cen.x, cen.y), 1.5f, col);
    dl->AddLine(ImVec2(cen.x - 9, cen.y), ImVec2(cen.x - 4, cen.y), col, 1.5f);
    dl->AddLine(ImVec2(cen.x + 4, cen.y), ImVec2(cen.x + 9, cen.y), col, 1.5f);
    dl->AddLine(ImVec2(cen.x, cen.y - 9), ImVec2(cen.x, cen.y - 4), col, 1.5f);
    dl->AddLine(ImVec2(cen.x, cen.y + 4), ImVec2(cen.x, cen.y + 9), col, 1.5f);
}

void VulkanEngine::build_menu_ui() {
    if (state_ != State::Paused) return;
    ImGuiViewport* vp = ImGui::GetMainViewport();
    // Cond_FirstUseEver: centred on first launch, then ImGui restores
    // whatever position / size the user left in qlike_imgui.ini.
    float max_h = vp->WorkSize.y * 0.90f;
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_FirstUseEver,
                             ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(560.0f, max_h), ImGuiCond_FirstUseEver);
    // No NoMove / NoSavedSettings — we want the user to drag the
    // menu and have ImGui persist position + open/closed state of
    // every CollapsingHeader inside it across restarts. The title
    // bar is the drag handle; "Pause Menu" is the persistent ID
    // that ImGui keys saved state against in qlike_imgui.ini.
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_AlwaysVerticalScrollbar;
    ImGui::Begin("Pause Menu", nullptr, flags);

    // Big neon "PAUSED" banner — drawn via ImGui's foreground colour
    // override so it stands out against the indigo background.
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.20f, 0.85f, 1.0f));
    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextUnformatted("// PAUSED //");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();

    // Action buttons in a horizontal row across the top so the
    // user can always see + click them without scrolling.
    const ImVec2 btn((ImGui::GetContentRegionAvail().x - 20.0f) / 3.0f, 0.0f);
    if (ImGui::Button("Resume", btn)) {
        state_ = State::Playing;
        window_->set_cursor_captured(true);
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart", btn)) {
        reset_player();
        state_ = State::Playing;
        window_->set_cursor_captured(true);
    }
    ImGui::SameLine();
    if (ImGui::Button("Quit", btn)) {
        window_->request_close();
    }
    ImGui::Spacing();

    // ===== Tabs ============================================================
    if (!ImGui::BeginTabBar("##settings_tabs",
                              ImGuiTabBarFlags_FittingPolicyScroll)) {
        ImGui::End();
        return;
    }

    // ----- Game tab --------------------------------------------------------
    if (ImGui::BeginTabItem("Game")) {
        ImGui::SliderFloat("gravity (m/s^2)", &game_.gravity, 0.5f, 1250.0f, "%.1f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("cubes / minute", &game_.cubes_per_minute, 0, 1000);
        ImGui::SliderFloat("bullet mass (kg)",  &game_.bullet_mass, 0.5f, 200.0f,
                           "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("bullet speed (m/s)", &game_.bullet_speed, 20.0f, 600.0f,
                           "%.0f");
        ImGui::SliderFloat("fire rate (rps)", &game_.fire_rate_rps, 0.0f, 30.0f,
                           "%.1f");
        ImGui::SliderFloat("spark scale",     &game_.spark_scale,   0.05f, 3.0f,
                           "%.2f");
        ImGui::EndTabItem();
    }

    // ----- Graphics tab ----------------------------------------------------
    if (ImGui::BeginTabItem("Graphics")) {
        if (ImGui::CollapsingHeader("Resolution & Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Quality preset combo. Switching writes a bundle of values onto rt_;
        // tweaking any individual slider afterward marks preset = -1 (custom).
        const char* kPresetLabels[] = {
            "Low (0.65× res, 8spp shadows, GTAO, 16 GI)",
            "Medium (0.85× res, 16spp, GTAO, 32 GI)",
            "High (1.0× res, 24spp, GTAO, 64 GI)",
            "Ultra (1.25× res, 40spp, RTAO, 96 GI / 2-bounce)"
        };
        int preset = std::clamp(rt_.quality_preset, -1, 3);
        const char* current_label = (preset >= 0 && preset <= 3)
            ? kPresetLabels[preset] : "Custom (sliders edited)";
        if (ImGui::BeginCombo("preset", current_label)) {
            for (int i = 0; i < 4; ++i) {
                bool is_selected = (preset == i);
                if (ImGui::Selectable(kPresetLabels[i], is_selected)) {
                    apply_quality_preset(i);
                    apply_render_scale();
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // Render-scale slider — 0.5..2.0. Apply re-allocates targets.
        // Drag changes only the value; you have to click Apply (or change
        // window size) for it to take effect, because resizing live render
        // targets requires a wait-idle.
        ImGui::SliderFloat("render scale", &rt_.render_scale, 0.4f, 2.0f, "%.2fx");
        ImGui::SameLine();
        if (ImGui::Button("Apply")) {
            apply_render_scale();
        }
        ImGui::TextDisabled("0.5× = half-res (huge speedup), 1.5×+ = SSAA");

        // Upscale presets — combine render_scale + TAAU + sharpening
        // into a single click. Each one mirrors the way a vendor
        // upscaler exposes Quality / Balanced / Performance modes.
        ImGui::TextUnformatted("Upscale preset:");
        ImGui::SameLine();
        auto apply_upscale = [&](float scale, bool taau_on, float sharp) {
            rt_.render_scale            = scale;
            rt_.taau_enabled            = taau_on;
            rt_.compose_sharpen_strength = sharp;
            rt_.quality_preset           = -1;
            apply_render_scale();
        };
        if (ImGui::Button("Native"))  apply_upscale(1.00f, false, 0.30f);
        ImGui::SameLine();
        if (ImGui::Button("Quality")) apply_upscale(0.77f, true,  0.80f);
        ImGui::SameLine();
        if (ImGui::Button("Balanced")) apply_upscale(0.67f, true, 1.10f);
        ImGui::SameLine();
        if (ImGui::Button("Perf"))    apply_upscale(0.50f, true,  1.40f);
        ImGui::TextDisabled("Quality/Balanced/Perf use TAAU + extra sharpen.");

        // AO mode combo.
        const char* kAoLabels[] = { "off", "GTAO (screen-space, fast)",
                                    "RTAO (true ray-traced, slow)" };
        int ao = std::clamp(rt_.ao_mode, 0, 2);
        if (ImGui::BeginCombo("AO mode", kAoLabels[ao])) {
            for (int i = 0; i < 3; ++i) {
                bool is_selected = (ao == i);
                if (ImGui::Selectable(kAoLabels[i], is_selected)) {
                    rt_.ao_mode = i;
                    rt_.quality_preset = -1; // custom
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        }   // end CollapsingHeader Resolution & Quality

        if (ImGui::CollapsingHeader("Graphics Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SeparatorText("Materials");
        ImGui::Checkbox("textures enabled", &rt_.textures_enabled);

        ImGui::SeparatorText("Sun");
        ImGui::SliderFloat("pitch (deg)", &rt_.sun_pitch_deg, 5.0f, 89.0f);
        ImGui::SliderFloat("yaw (deg)", &rt_.sun_yaw_deg, -180.0f, 180.0f);
        ImGui::Checkbox("Auto golden-hour tint (warm sun + sky when low)",
                        &rt_.auto_golden_hour);
        ImGui::SliderFloat("intensity", &rt_.sun_intensity, 0.0f, 8.0f);
        ImGui::ColorEdit3("sun color", &rt_.sun_color.x);

        ImGui::SeparatorText("Sky / Ambient");
        ImGui::ColorEdit3("sky color", &rt_.sky_color.x);
        ImGui::ColorEdit3("ground ambient", &rt_.ground_ambient.x);
        ImGui::SliderFloat("ambient strength", &rt_.ambient_strength, 0.0f, 3.0f);

        ImGui::SeparatorText("Shadows (RT, contact-hardening)");
        ImGui::Checkbox("enabled", &rt_.shadow_enabled);
        ImGui::Checkbox("half-rate shadow pass (perf)",
                        &rt_.half_rate_shadows);
        ImGui::TextDisabled(
            "Trace the PCSS sun shadow at half res + bilateral\n"
            "upsample instead of per-pixel. Big perf win; OFF =\n"
            "the proven inline path.");
        ImGui::SliderInt("samples", &rt_.shadow_samples, 1, 128);
        ImGui::SliderFloat("near samples multiplier", &rt_.shadow_near_mult, 1.0f, 8.0f, "%.2fx");
        ImGui::SliderFloat("softness", &rt_.shadow_softness, 0.0f, 0.15f);
        ImGui::SliderFloat("strength", &rt_.shadow_strength, 0.0f, 1.0f);
        ImGui::SliderFloat("curve (linear→expo)", &rt_.shadow_curve, 0.0f, 1.0f);

        ImGui::SeparatorText("Castle materials (SPOM)");
        // Per-pixel parallax depth on castle walls (and any future
        // SPOM-eligible materials). 0 = flat, 1 = engine default
        // (~4 cm peak-to-trough at 1 m view), 2.5 = exaggerated
        // bricks. At 0 the SPOM march is short-circuited entirely so
        // there's no GPU cost for users who want flat textures.
        ImGui::SliderFloat("SPOM strength", &rt_.spom_strength, 0.0f, 2.5f, "%.2f");

        ImGui::SeparatorText("Sun shadow map (grass receiver)");
        // Resolution dropdown — apply triggers a vkDeviceWaitIdle + image
        // recreation in the next frame's tick. 4096 doubles VRAM, halves
        // texel size; 512 quarters fragment cost vs 1024.
        const char* res_labels[] = { "512", "1024", "2048", "4096", "8192" };
        int res_idx = 1;
        if      (rt_.shadow_map_resolution <= 512)  res_idx = 0;
        else if (rt_.shadow_map_resolution <= 1024) res_idx = 1;
        else if (rt_.shadow_map_resolution <= 2048) res_idx = 2;
        else if (rt_.shadow_map_resolution <= 4096) res_idx = 3;
        else                                         res_idx = 4;
        if (ImGui::Combo("shadow map resolution", &res_idx, res_labels, 5)) {
            const int values[] = { 512, 1024, 2048, 4096, 8192 };
            rt_.shadow_map_resolution = values[res_idx];
        }
        ImGui::SliderFloat("shadow map world size (m)",
                            &rt_.shadow_map_world_half, 30.0f, 400.0f);
        ImGui::Checkbox("debug overlay (frustum + bake bounds)",
                         &rt_.shadow_debug_overlay);

        ImGui::SeparatorText("Ambient Occlusion (RT)");
        ImGui::SliderInt("AO samples (0=off)", &rt_.ao_samples, 0, 64);
        ImGui::SliderFloat("AO radius", &rt_.ao_radius, 0.1f, 8.0f);
        ImGui::SliderFloat("AO floor (corner-pile-up cap)",
                            &rt_.ao_floor, 0.0f, 1.0f);

        ImGui::SeparatorText("Tone / Exposure / Contrast / Brightness");
        ImGui::SliderFloat("Auto-exposure (eye adaptation)",
                            &rt_.auto_exposure_strength, 0.0f, 1.5f);
        ImGui::SliderFloat("image contrast",
                            &rt_.image_contrast,   0.5f, 2.0f);
        ImGui::SliderFloat("image brightness",
                            &rt_.image_brightness, 0.3f, 2.5f);
        ImGui::SliderFloat("image gamma (curve)",
                            &rt_.image_gamma,      0.4f, 2.5f);

        ImGui::SeparatorText("Path-traced GI (multi-bounce)");
        ImGui::SliderInt("GI samples (0=off)", &rt_.gi_samples, 0, 128);
        // ReSTIR-lite (session 1 of multi-session plan — see
        // docs/restir_plan.md). Toggle quarters gi_samples at runtime;
        // the existing TAA + spatial filter smooths the noise. Sessions
        // 2+ replace the / 4 hack with proper reservoir reuse.
        ImGui::Checkbox("ReSTIR-lite (samples / 4, lean on TAA)",
                          &rt_.gi_restir_enabled);
        ImGui::SliderInt("GI bounces", &rt_.gi_bounces, 1, 5);
        ImGui::SliderInt("GI shadow bounces (sun shadow on bounce hits)",
                          &rt_.gi_shadow_max_bounce, 0, 5);
        ImGui::SliderFloat("GI strength", &rt_.gi_strength, 0.0f, 3.0f);
        ImGui::SliderFloat("GI radius", &rt_.gi_radius, 1.0f, 400.0f);
        ImGui::SliderFloat("GI softener (raymarch terrain)",
                          &rt_.gi_softener, 0.0f, 1.0f);
        ImGui::Checkbox("GI debug viz (raymarch terrain)",
                          &rt_.gi_debug_viz);
        ImGui::SliderFloat("RT LOD distance (raymarch terrain, m)",
                          &rt_.terrain_rt_lod_distance, 50.0f, 1000.0f);
        ImGui::SliderFloat("AO darkness (raymarch terrain)",
                          &rt_.terrain_ao_punch, 0.5f, 3.0f);

        ImGui::SeparatorText("Raymarched terrain RT (per-pixel caps)");
        ImGui::TextDisabled("Higher = better visuals + heavier GPU. "
                            "Lower if you hit GPU TDR / device_lost.");
        ImGui::SliderInt("PCSS shadow rays cap (terrain)",
                          &rt_.terrain_pcss_samples_cap, 2, 16);
        ImGui::SliderInt("GI rays cap (terrain)",
                          &rt_.terrain_gi_samples_cap, 0, 16);
        ImGui::SliderInt("GI bounces cap (terrain)",
                          &rt_.terrain_gi_bounces_cap, 1, 3);
        ImGui::SliderFloat("AO final strength (terrain)",
                          &rt_.terrain_ao_final_strength, 0.0f, 1.0f);

        ImGui::SeparatorText("Anti-aliasing (TAA + Halton sub-pixel jitter)");
        ImGui::Checkbox("sub-pixel jitter", &rt_.taa_jitter_enabled);
        ImGui::SliderFloat("jitter strength", &rt_.taa_jitter_strength, 0.0f, 2.0f);
        ImGui::SliderFloat("history blend", &rt_.taa_history_blend, 0.0f, 0.98f);
        ImGui::Checkbox("TAAU (temporal upsample)", &rt_.taau_enabled);
        ImGui::TextDisabled("Adds a temporal upscale pass after TAA. Best with\n"
                             "render_scale < 1.0 — renders at LR, reconstructs\n"
                             "at native via motion-vector reproject + variance\n"
                             "clamp. ~1.5x perf gain at render_scale=0.67.");
        // FSR2 — phased rollout. Phase 1 (foundation) is shipped:
        // Halton(2,3) jitter with sample-count-aware phase, FrameView
        // jitter capture. Phases 2-5 (compute passes for prepare /
        // depth-clip / reconstruct-prev-depth / lock / accumulate /
        // RCAS) are pending. Toggle currently only switches the
        // jitter sequence; the actual upscale path falls back to
        // FSR1 EASU+RCAS in compose.frag until Phase 5 lands.
        ImGui::Checkbox("FSR upscale (renders at render_scale, upscales to native)",
                        &rt_.fsr2_enabled);
        // Backend select — only meaningful when FSR upscale is on. FSR3
        // uses the new ffx-api ABI through the prebuilt vendored DLL;
        // FSR2 uses the legacy static-lib path. Frame generation (FSR3
        // sessions 3-6) is NOT yet wired — this only swaps the upscaler.
        const char* fsr_backends[] = { "FSR2 (legacy static)", "FSR3 (ffx-api DLL)" };
        ImGui::Combo("FSR backend", &rt_.fsr_backend, fsr_backends,
                     IM_ARRAYSIZE(fsr_backends));
        if (fsr3_fatal_) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                "FSR3 unavailable on this device (CreateContext AVed in SDK DLL "
                "— see qlike.log). Backend auto-reverted to FSR2.");
        }
        // FSR3 Frame Generation. Only meaningful when FSR3 backend is
        // active. Session 3 dispatches FG to a private image that
        // isn't presented yet — toggle visibly costs GPU time but
        // produces no perceptible delta. Session 4 wires the present
        // chain so this becomes the FPS-doubling switch.
        if (rt_.fsr_backend == 1 && !fsr3_fatal_) {
            ImGui::Checkbox("FSR3 Frame Generation (perceived FPS ×2)",
                            &rt_.fg_enabled);
            if (fsr3_fg_fatal_) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                    "FG unavailable (init failed) — see qlike.log");
            } else if (rt_.fg_enabled && fg_runtime_disabled_) {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                    "FG auto-disabled (render < 50 fps base — interpolation "
                    "judder dominates below this; will re-enable above 60).");
            } else if (rt_.fg_enabled && fsr3_swapchain_active_) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1.0f),
                    "FG active — SDK pacing real + generated frames");
            }
        }
        ImGui::TextDisabled("Phase 1/5 — Halton(2,3) jitter active.\n"
                             "Compute passes pending in future sessions.");
        ImGui::SliderFloat("spatial strength", &rt_.taa_spatial_strength, 0.0f, 1.0f);
        // Compose-pass unsharp mask — runs in compose.frag (5 texelFetches),
        // recovers detail the TAA spatial filter softened. 0 disables; 0.55
        // is the default; >1 looks deliberately punchy.
        ImGui::SliderFloat("post-TAA sharpening", &rt_.compose_sharpen_strength, 0.0f, 2.0f);
        ImGui::SliderFloat("sun shafts (god rays)", &rt_.sun_shaft_intensity, 0.0f, 2.0f);

        ImGui::SeparatorText("Bloom (compose-pass spiral-tap)");
        ImGui::Checkbox("bloom enabled", &rt_.bloom_enabled);
        ImGui::SliderFloat("bloom strength", &rt_.bloom_strength, 0.0f, 2.0f);
        ImGui::SliderFloat("bloom threshold", &rt_.bloom_threshold, 0.5f, 3.0f);
        ImGui::SliderFloat("bloom radius (px)", &rt_.bloom_radius, 4.0f, 80.0f);
        ImGui::SliderFloat("spark bloom", &rt_.spark_bloom, 0.0f, 5.0f);

        ImGui::SeparatorText("Lens flare (Chapman screen-space)");
        ImGui::Checkbox("flare enabled", &rt_.lens_flare_enabled);
        ImGui::SliderFloat("flare strength",   &rt_.lens_flare_strength,   0.0f, 2.0f);
        ImGui::SliderFloat("flare threshold",  &rt_.lens_flare_threshold,  0.0f, 4.0f);
        ImGui::SliderFloat("ghost dispersal",  &rt_.lens_flare_dispersal,  0.05f, 0.6f);
        ImGui::SliderFloat("halo width",       &rt_.lens_flare_halo_width, 0.0f, 0.5f);
        ImGui::SliderInt  ("ghost count",      &rt_.lens_flare_ghosts,     1, 8);
        ImGui::SliderFloat("aberration",       &rt_.lens_flare_aberration, 0.0f, 0.02f, "%.4f");

        if (ImGui::Button("Reset to defaults")) {
            rt_ = RtSettings{};
        }
        ImGui::SameLine();
        if (ImGui::Button("Low")) {
            rt_.shadow_samples = 1;
            rt_.shadow_softness = 0.0f;
            rt_.ao_samples = 0;
            rt_.gi_samples = 0;
            rt_.gi_bounces = 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("High")) {
            rt_.shadow_samples = 4;  rt_.shadow_softness = 0.025f;
            rt_.ao_samples = 8;      rt_.ao_radius = 1.5f;
            rt_.gi_samples = 4;      rt_.gi_strength = 1.0f;
            rt_.gi_radius = 60.0f;   rt_.gi_bounces = 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Ultra")) {
            rt_.shadow_samples = 8;  rt_.shadow_softness = 0.04f;
            rt_.ao_samples = 16;     rt_.ao_radius = 2.5f;
            rt_.gi_samples = 8;      rt_.gi_strength = 1.0f;
            rt_.gi_radius = 100.0f;  rt_.gi_bounces = 2;
        }
        ImGui::SameLine();
        if (ImGui::Button("Insane")) {
            // The all-in knob — pushes shadow samples beyond the conservative
            // defaults; on an RTX 4080 it should still hold ~60 fps at 720p.
            rt_.shadow_samples = 16; rt_.shadow_softness = 0.06f;
            rt_.ao_samples = 32;     rt_.ao_radius = 4.0f;
            rt_.gi_samples = 16;     rt_.gi_strength = 1.0f;
            rt_.gi_radius = 200.0f;  rt_.gi_bounces = 4;
        }
        }   // end CollapsingHeader Graphics Settings
        ImGui::EndTabItem();
    }

    // ----- Terrain tab (terrain renderer + grass) -------------------------
    if (ImGui::BeginTabItem("Terrain")) {
        if (ImGui::CollapsingHeader("Terrain", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SeparatorText("Renderer");
        // Mode switch is always visible; everything below adapts to it
        // so you only see the settings for the selected terrain path.
        if (ImGui::Checkbox("Procedural raymarched terrain (FBM)",
                            &rt_.terrain_raymarch_enabled)) {
            // Mesh(water-only) wants full-res, FBM wants the scaled LR
            // buffer — resize when the mode flips.
            recreate_terrain_raymarch_lowres();
        }
        ImGui::TextDisabled("Switches the whole terrain renderer. Toggling\n"
                             "lazy-builds raster mesh + grass on demand.\n"
                             "Terrain BLAS (RT shadows on terrain) is built once\n"
                             "at load — toggling off mid-session means no RT\n"
                             "terrain shadow contact until restart.");

        if (rt_.terrain_raymarch_enabled) {
        ImGui::SeparatorText("FBM raymarch");
        ImGui::SliderInt("Raymarch steps",
                         &rt_.terrain_raymarch_max_steps, 60, 300);
        ImGui::SliderInt("Raymarch shadow steps",
                         &rt_.terrain_raymarch_shadow_steps, 16, 96);
        ImGui::SliderInt("FBM octaves (march)",
                         &rt_.terrain_raymarch_octaves, 4, 24);
        ImGui::SliderInt("FBM octaves (normals)",
                         &rt_.terrain_raymarch_normal_octaves, 4, 32);
        ImGui::SliderFloat("Step factor",
                           &rt_.terrain_raymarch_step_factor, 0.15f, 0.8f, "%.2f");
        // Distance-LOD ramp — drops FBM octaves with ray distance so the
        // expensive 3× terrainH() calls in calcNormal collapse to the
        // floor count past `lod_far_m`. Biggest perf win at horizon
        // views where most rays travel >500m.
        ImGui::SliderFloat("LOD near (m)",
                           &rt_.terrain_raymarch_lod_near_m, 20.0f, 400.0f, "%.0f");
        ImGui::SliderFloat("LOD far (m)",
                           &rt_.terrain_raymarch_lod_far_m, 200.0f, 2000.0f, "%.0f");
        ImGui::SliderInt("LOD min octaves",
                         &rt_.terrain_raymarch_lod_min_octaves, 2, 8);
        // Render-scale for the raymarch only — biggest single perf
        // knob. 0.5 = quarter the FBM evaluations per frame; bilinear
        // upscale + depth-aware composite hides most of the softness
        // because terrain noise is low-frequency anyway.
        const char* tr_scale_labels[] = { "100% (native)", "75%", "50%", "33%", "25%" };
        const float tr_scale_values[] = { 1.0f, 0.75f, 0.5f, 0.333f, 0.25f };
        int tr_scale_idx = 0;
        for (int i = 0; i < 5; ++i)
            if (std::abs(rt_.terrain_raymarch_scale - tr_scale_values[i]) < 0.02f) {
                tr_scale_idx = i; break;
            }
        if (ImGui::Combo("Raymarch render scale", &tr_scale_idx,
                          tr_scale_labels, 5)) {
            rt_.terrain_raymarch_scale = tr_scale_values[tr_scale_idx];
            recreate_terrain_raymarch_lowres();
        }
        ImGui::SliderFloat("Raymarch sharpen", &rt_.terrain_raymarch_sharpen,
                           0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Volumetric fog strength",
                           &rt_.terrain_raymarch_fog_strength, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Fog Y start (m)",
                           &rt_.terrain_raymarch_fog_y_start, -20.0f, 60.0f, "%.1f");
        ImGui::SliderFloat("Fog Y top (m)",
                           &rt_.terrain_raymarch_fog_y_top, -20.0f, 80.0f, "%.1f");
        ImGui::SliderFloat("Fog noise (wisps)",
                           &rt_.terrain_raymarch_fog_noise, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Relaxation cone-stepping (Phase 6)",
                         &rt_.terrain_raymarch_relaxation);
        ImGui::Checkbox("Fog god-rays (Phase 7)",
                         &rt_.terrain_raymarch_fog_godrays);
        // TextDisabled is printf-style; passing "%s" + a string is the
        // safe way to display literal `%` characters (e.g. "50%").
        ImGui::TextDisabled("%s",
            "Lower steps / fewer octaves / higher step factor = faster.\n"
            "50% render scale = 4x fewer FBM evaluations.\n"
            "Sharpen recovers detail lost to bilinear upscaling.\n"
            "Fog 0 = off, 1 = baseline, 2 = thick valley fog.\n"
            "Relaxation: faster march at the risk of skipping spikes.\n"
            "God-rays: 4 short rays per fog step, ~20% slower fog.");
        } else {
        ImGui::SeparatorText("Mesh / tessellation");
        ImGui::Checkbox("Near GPU tessellation + displacement",
                         &rt_.terrain_tessellation_enabled);
        ImGui::TextDisabled("Live toggle. OFF = plain CD-LOD mesh (no\n"
                             "subdivision/displacement near the camera).");
        ImGui::SliderFloat("Tessellation range (m)",
                           &rt_.terrain_tess_range, 30.0f, 250.0f, "%.0f");
        ImGui::Checkbox("Wireframe (show triangles / mesh density)",
                         &rt_.terrain_wireframe);
        }   // end FBM / mesh renderer split

        ImGui::SeparatorText("Ocean / water");
        ImGui::Checkbox("Water enabled", &rt_.water_enabled);
        ImGui::SliderFloat("Water level (Y)",
                           &rt_.water_level, -10.0f, 50.0f, "%.1f m");
        ImGui::SliderFloat("Wave strength",
                           &rt_.water_wave_strength, 0.0f, 0.6f, "%.3f");
        ImGui::SliderFloat("Wave scale (freq mult)",
                           &rt_.water_wave_scale, 0.1f, 5.0f, "%.2f");
        // Water style dropdown — picks which normal+foam generator.
        // Each style brings its own visual character; the engine's
        // depth / extinction / reflection / shadow stack stays
        // shared so the user keeps their colour + shore controls.
        const char* k_water_styles[] = {
            "Default (sum of sines)",
            "River (P_Malin — advected FBM)",
            "Lake (distance-faded bumps)",
        };
        ImGui::Combo("Water style", &rt_.water_style,
                     k_water_styles, IM_ARRAYSIZE(k_water_styles));
        if (rt_.water_style == 1) {
            ImGui::SliderFloat("river flow speed",
                               &rt_.water_river_speed, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("river normal strength",
                               &rt_.water_river_normal_str, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("river flow angle (deg)",
                               &rt_.water_river_flow_angle, -180.0f, 180.0f, "%.0f");
            ImGui::SliderFloat("river time speed",
                               &rt_.water_river_time_speed, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("river detail scale",
                               &rt_.water_river_detail, 0.1f, 4.0f, "%.2f");
            ImGui::SliderFloat("river foam amount",
                               &rt_.water_river_foam_amount, 0.0f, 3.0f, "%.2f");
            ImGui::ColorEdit3("river underwater tint",
                              &rt_.water_river_extinct_color.x);
            ImGui::SliderFloat("river underwater density",
                               &rt_.water_river_extinct_density, 0.0f, 4.0f, "%.2f");
        } else if (rt_.water_style == 2) {
            ImGui::SliderFloat("lake bump strength",
                               &rt_.water_lake_bump_strength, 0.0f, 1.5f, "%.2f");
            ImGui::SliderFloat("lake time speed",
                               &rt_.water_lake_time_speed, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("lake uv scale",
                               &rt_.water_lake_uv_scale, 0.1f, 4.0f, "%.2f");
            ImGui::SliderFloat("lake bump fade dist (m)",
                               &rt_.water_lake_bump_dist, 5.0f, 500.0f, "%.0f");
            ImGui::TextDisabled("Far surface goes mirror-flat past fade dist.");
        }
        ImGui::ColorEdit3("Deep water color", &rt_.water_color.x);
        ImGui::ColorEdit3("Shallow water color", &rt_.water_color_shallow.x);
        ImGui::SliderFloat("Shore band (m)",
                           &rt_.water_shore_blend, 0.5f, 20.0f, "%.1f");
        ImGui::SliderFloat("Shore noise",
                           &rt_.water_shore_noise, 0.0f, 1.0f, "%.2f");
        ImGui::ColorEdit3("Foam color", &rt_.water_foam_color.x);
        ImGui::SliderFloat("Foam strength",
                           &rt_.water_foam_strength, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Foam width (m)",
                           &rt_.water_foam_width, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Foam opacity",
                           &rt_.water_foam_opacity, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "How opaque the foam is vs the see-through water under it.\n"
            "0 = foam as transparent as clear shallow water (light\n"
            "surf), 1 = foam fully opaque.");
        ImGui::Checkbox("RT terrain reflections", &rt_.water_rt_reflections);
        ImGui::Checkbox("RT cube/castle reflections (TLAS)",
                         &rt_.water_tlas_reflections);
        ImGui::Checkbox("Sun shadows on water", &rt_.water_shadows_enabled);
        ImGui::SliderFloat("Transparency",
                           &rt_.water_transparency, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "Master see-through strength. 0 = always opaque,\n"
            "1 = full clarity (up to the clarity depth below).");
        ImGui::SliderFloat("Clarity depth (m)",
                           &rt_.water_clarity_depth, 0.2f, 20.0f, "%.1f");
        ImGui::TextDisabled("%s",
            "How deep you can see, INDEPENDENT of strength.\n"
            "Small (~1 m) = only very shallow water is clear, deep\n"
            "water stays opaque (realistic shallows). Large = see\n"
            "the seabed far down (tropical lagoon).");
        ImGui::SliderFloat("Shoreline softness (m)",
                           &rt_.water_shore_softness, 0.05f, 4.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "Metres of depth the water fades in over at the land\n"
            "edge. Larger = a longer, smoother wet-sand → water\n"
            "transition (also hides the waterline jitter).");
        // Quick presets so the user has a starting point per mood.
        if (ImGui::Button("Tropical")) {
            rt_.water_color         = glm::vec3(0.04f, 0.30f, 0.40f);
            rt_.water_color_shallow = glm::vec3(0.55f, 0.85f, 0.90f);
            rt_.water_shore_blend   = 6.0f;
            rt_.water_shore_noise   = 0.45f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Lake")) {
            rt_.water_color         = glm::vec3(0.05f, 0.16f, 0.20f);
            rt_.water_color_shallow = glm::vec3(0.30f, 0.55f, 0.55f);
            rt_.water_shore_blend   = 3.0f;
            rt_.water_shore_noise   = 0.40f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Deep ocean")) {
            rt_.water_color         = glm::vec3(0.02f, 0.10f, 0.18f);
            rt_.water_color_shallow = glm::vec3(0.10f, 0.35f, 0.50f);
            rt_.water_shore_blend   = 8.0f;
            rt_.water_shore_noise   = 0.30f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Murky")) {
            rt_.water_color         = glm::vec3(0.10f, 0.13f, 0.07f);
            rt_.water_color_shallow = glm::vec3(0.30f, 0.32f, 0.20f);
            rt_.water_shore_blend   = 2.0f;
            rt_.water_shore_noise   = 0.55f;
        }
        ImGui::TextDisabled("%s",
            "Cheap analytical-wave ocean folded into the\n"
            "raymarch shader. Rendered when ray hits y=level\n"
            "before terrain. Schlick fresnel + sky reflection\n"
            "+ specular highlight + depth-fade tint.");

        ImGui::SeparatorText("Heightmap resolution");
        const char* hres_labels[] = {
            "1x (2048 cells, ~17 MB)",
            "2x (4096 cells, ~270 MB *)",
            "4x (8192 cells, NOT RECOMMENDED — multi-GB)"
        };
        const int hres_values[] = { 1, 2, 4 };
        int hres_idx = 0;
        for (int i = 0; i < 3; ++i)
            if (rt_.terrain_heightmap_scale == hres_values[i]) { hres_idx = i; break; }
        if (ImGui::Combo("Heightmap scale", &hres_idx, hres_labels, 3)) {
            rt_.terrain_heightmap_scale = hres_values[hres_idx];
        }
        ImGui::TextDisabled("* Restart required — heightmap generated at level load.");

        // ---- Mesh-path-only settings (hidden in FBM raymarch mode) ----
        if (!rt_.terrain_raymarch_enabled) {
        ImGui::SeparatorText("LOD distances");
        ImGui::SliderFloat("Terrain detail (multiplier)",
                            &rt_.terrain_lod_scale, 0.5f, 4.0f, "%.2fx");
        ImGui::SliderFloat("LOD 0->1 (m)", &rt_.terrain_lod1,  20.0f, 1000.0f);
        ImGui::SliderFloat("LOD 1->2 (m)", &rt_.terrain_lod2,  60.0f, 2000.0f);
        ImGui::SliderFloat("LOD 2->3 (m)", &rt_.terrain_lod3, 120.0f, 4000.0f);
        if (rt_.terrain_lod2 < rt_.terrain_lod1 + 20.0f)
            rt_.terrain_lod2 = rt_.terrain_lod1 + 20.0f;
        if (rt_.terrain_lod3 < rt_.terrain_lod2 + 20.0f)
            rt_.terrain_lod3 = rt_.terrain_lod2 + 20.0f;

        ImGui::SeparatorText("Near tessellation density (live)");
        ImGui::SliderFloat("Tess max subdivision",
                           &rt_.terrain_tess_max_level, 1.0f, 64.0f, "%.0f");
        ImGui::SliderFloat("Tess full-detail dist (m)",
                           &rt_.terrain_tess_near_m, 1.0f, 60.0f, "%.0f");
        ImGui::SliderFloat("Tess fade-out dist (m)",
                           &rt_.terrain_tess_far_m, 10.0f, 250.0f, "%.0f");
        ImGui::SliderFloat("Tess falloff (<1 = denser near)",
                           &rt_.terrain_tess_falloff, 0.2f, 3.0f, "%.2f");
        ImGui::SliderFloat("Tess smoothing (before texture)",
                           &rt_.terrain_tess_smooth, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "0 = raw per-type relief, 1 = smooth low-frequency swell.\n"
            "Low-passes the tessellated geometry + normal before the\n"
            "material splat is applied.");
        ImGui::SliderFloat("Rock relief (vertex geometry)",
                           &rt_.terrain_rock_relief, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "Real tessellated geometry on rock — the coarse/silhouette\n"
            "half. Kept chunky even when Tess smoothing is high.");
        ImGui::SliderFloat("Sand ripple scale",
                           &rt_.terrain_sand_ripple_scale, 1.0f, 40.0f, "%.1f");
        ImGui::TextDisabled(
            "Beach ripple spacing. Lines follow the shoreline (phase =\n"
            "height above water), so they curve with the coast.");
        ImGui::SliderFloat("Grass line scale",
                           &rt_.terrain_grass_line_scale, 0.0f, 40.0f, "%.1f");
        ImGui::TextDisabled(
            "Grass contour rows, same shore-following logic.\n"
            "0 = off. Independent of the sand ripple scale.");
        ImGui::SliderFloat("Rock parallax (per-pixel depth)",
                           &rt_.terrain_pom_strength, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "Per-pixel parallax of the eroded rock height — the FINE\n"
            "half. Pair with Rock relief above + Material strength > 0.\n"
            "Only on rocky pixels, ramps to 0 by ~90 m (LOD). 0 = off.");
        if (rt_.terrain_tess_far_m < rt_.terrain_tess_near_m + 5.0f)
            rt_.terrain_tess_far_m = rt_.terrain_tess_near_m + 5.0f;

        ImGui::SeparatorText("Ground material (procedural splat)");
        ImGui::SliderFloat("Material strength",
                           &rt_.ground_mat_strength, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "0 = old flat height-band colour (A/B off), 1 = full per-texel\n"
            "rock/grass/dirt/sand/snow blended by height + slope.");
        ImGui::SliderFloat("Material tile size (m)",
                           &rt_.ground_mat_tile_m, 0.5f, 12.0f, "%.2f");
        ImGui::SliderFloat("Detail normal strength",
                           &rt_.ground_mat_normal, 0.0f, 1.0f, "%.2f");

        ImGui::SeparatorText("Shading");
        ImGui::SliderFloat("atmospheric fog",  &rt_.terrain_fog_strength,    0.0f, 1.5f);
        ImGui::SliderFloat("light wrap",       &rt_.terrain_wrap_strength,   0.0f, 1.0f);
        ImGui::SliderFloat("detail brightness",&rt_.terrain_detail_strength, 0.0f, 3.0f);
        ImGui::SliderFloat("shadow softness x",&rt_.terrain_shadow_softness_scale, 0.05f, 1.5f);
        ImGui::SliderFloat("Terrain shading contrast",
                            &rt_.terrain_shading_contrast, 0.5f, 6.0f, "%.2f");
        {
            const char* ss_labels[] = { "1x (native)", "2x (sub-cell)", "4x (sharp far)" };
            const int ss_values[] = { 1, 2, 4 };
            int ss_idx = 0;
            for (int i = 0; i < 3; ++i) if (rt_.terrain_bake_supersample == ss_values[i]) { ss_idx = i; break; }
            if (ImGui::Combo("Shadow bake supersample", &ss_idx, ss_labels, 3)) {
                rt_.terrain_bake_supersample = ss_values[ss_idx];
            }
        }
        }   // end mesh-path-only settings

        // ---- Shared (both terrain modes) ----
        if (ImGui::TreeNode("Layer transitions (height in metres)")) {
            ImGui::SliderFloat("sand->grass start", &rt_.terrain_h_sand_grass_start,  -10.0f,  60.0f);
            ImGui::SliderFloat("sand->grass end",   &rt_.terrain_h_sand_grass_end,    -10.0f,  60.0f);
            ImGui::SliderFloat("grass->dirt start", &rt_.terrain_h_grass_dirt_start,    0.0f, 100.0f);
            ImGui::SliderFloat("grass->dirt end",   &rt_.terrain_h_grass_dirt_end,      0.0f, 100.0f);
            ImGui::SliderFloat("dirt->rock start",  &rt_.terrain_h_dirt_rock_start,    20.0f, 150.0f);
            ImGui::SliderFloat("dirt->rock end",    &rt_.terrain_h_dirt_rock_end,      20.0f, 150.0f);
            ImGui::SliderFloat("rock->snow start",  &rt_.terrain_h_rock_snow_start,    50.0f, 200.0f);
            ImGui::SliderFloat("rock->snow end",    &rt_.terrain_h_rock_snow_end,      50.0f, 200.0f);
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Shoreline tint", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Grass-area shore (under blades):");
            ImGui::ColorEdit3("grass shore colour",
                              &rt_.terrain_shore_color.x);
            ImGui::SliderFloat("grass shore strength",
                               &rt_.terrain_shore_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("grass shore distance (m)",
                               &rt_.terrain_shore_distance,
                               0.1f, 20.0f, "%.1f");
            ImGui::Separator();
            ImGui::TextDisabled("Bare-terrain shore (no grass):");
            ImGui::ColorEdit3("bare shore colour",
                              &rt_.terrain_shore_general_color.x);
            ImGui::SliderFloat("bare shore strength",
                               &rt_.terrain_shore_general_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("bare shore distance (m)",
                               &rt_.terrain_shore_general_distance,
                               0.1f, 20.0f, "%.1f");
            ImGui::Separator();
            ImGui::TextDisabled("Beach base material (slope-driven):");
            ImGui::ColorEdit3("sand colour",
                              &rt_.terrain_sand_color.x);
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Distance fog")) {
            ImGui::ColorEdit3("fog colour",
                              &rt_.distance_fog_color.x);
            ImGui::SliderFloat("fog strength",
                               &rt_.distance_fog_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("fog density",
                               &rt_.distance_fog_density,
                               0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("fog start (m)",
                               &rt_.distance_fog_start,
                               0.0f, 500.0f, "%.0f");
            ImGui::SliderFloat("fog height top (m)",
                               &rt_.distance_fog_height,
                               0.0f, 400.0f, "%.0f");
            ImGui::SliderFloat("fog max",
                               &rt_.distance_fog_max,
                               0.0f, 1.0f, "%.2f");
            ImGui::TextDisabled("strength=0 disables. height_top=0 -> uniform fog.");
            ImGui::TreePop();
        }

        ImGui::SeparatorText("Debug");
        const char* terrain_debug_labels[] = {
            "off (full shading)",
            "simple Lambert (no RT/AO/GI)",
            "show vertex normal",
            "show face normal",
            "4: Lambert + RT shadow",
            "5: + height layers",
            "6: + slope-rock",
            "7: + cavity AO",
            "8: + triplanar detail",
            "9: + RTAO",
            "10: + GI bounce",
        };
        ImGui::Combo("terrain debug", &rt_.terrain_debug_mode,
                      terrain_debug_labels, 11);
        }   // end CollapsingHeader Terrain

        // Grass folded into the Terrain tab — they're conceptually
        // the same surface family for the player, and the previous
        // top-level Grass section was rarely opened.
        if (ImGui::CollapsingHeader("Grass")) {
            ImGui::Checkbox("grass enabled", &rt_.grass_enabled);
            ImGui::Checkbox("RAYMARCH grass (SDF blades, replaces rasterised)",
                             &rt_.grass_raymarch_enabled);
            ImGui::SliderFloat("raymarch distance (m)",
                                &rt_.grass_raymarch_distance, 5.0f, 400.0f);
            ImGui::SliderFloat("density (0..4)", &rt_.grass_density, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("height scale",   &rt_.grass_height_scale, 0.30f, 2.0f);
            ImGui::SliderFloat("distance (m)",   &rt_.grass_distance,   10.0f, 200.0f);
            ImGui::SliderFloat("wind (m)",       &rt_.grass_wind,        0.0f, 0.30f);
            ImGui::SliderFloat("alpha cutoff",   &rt_.grass_alpha_cutoff, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("cutoff soft-dist (m)",
                               &rt_.grass_cutoff_soft_dist, 5.0f, 150.0f, "%.0f");
            ImGui::SliderFloat("slope cutoff",   &rt_.grass_slope_n_min, 0.55f, 1.0f, "%.2f");
            ImGui::SliderFloat("distance density falloff",
                               &rt_.grass_distance_density, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("altitude min (m)", &rt_.grass_alt_min, -20.0f, 200.0f);
            ImGui::SliderFloat("altitude max (m)", &rt_.grass_alt_max, -20.0f, 200.0f);
            ImGui::ColorEdit3("blade tip colour",
                              &rt_.grass_color_top.x);
            ImGui::ColorEdit3("blade base colour",
                              &rt_.grass_color_bottom.x);
            ImGui::SliderFloat("base AO floor",
                               &rt_.grass_base_ao_floor, 0.0f, 1.0f, "%.2f");
            ImGui::ColorEdit3("ground tint close",
                              &rt_.grass_color_ground.x);
            ImGui::ColorEdit3("ground tint far",
                              &rt_.grass_color_ground_far.x);
            ImGui::SliderFloat("ground tint strength",
                               &rt_.grass_ground_tint_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("ground tint far distance (m)",
                               &rt_.grass_ground_tint_far_distance,
                               50.0f, 1000.0f, "%.0f");
            ImGui::Separator();
            ImGui::TextUnformatted("Shoreline tint (drier near water)");
            ImGui::ColorEdit3("shore colour", &rt_.grass_shore_color.x);
            ImGui::SliderFloat("shore strength",
                               &rt_.grass_shore_strength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("shore distance (m)",
                               &rt_.grass_shore_distance, 0.1f, 20.0f, "%.1f");
            ImGui::Separator();
            ImGui::TextUnformatted("Grass shadow on terrain");
            ImGui::SliderFloat("shadow strength",
                               &rt_.grass_shadow_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderInt  ("shadow samples",
                               &rt_.grass_shadow_samples, 0, 8);
            ImGui::SliderFloat("shadow reach (m)",
                               &rt_.grass_shadow_max_dist,
                               0.2f, 5.0f, "%.1f");
        }

        ImGui::EndTabItem();
    }

    // ---------------- Terrain Edit ----------------
    // Sculpt brush + heightmap save/load + plateau noise + FBM erosion
    // bake. Lives in its own tab so the rendering knobs in the
    // "Terrain" tab stay focused on look (renderer / layers / debug)
    // and the destructive editing tools are one click away from the
    // tab bar instead of buried inside a CollapsingHeader.
    if (ImGui::BeginTabItem("Terrain Edit")) {
        ImGui::SeparatorText("Heightmap save / load");
        if (ImGui::Button("Save heightmap")) {
            save_terrain_heights();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("→ assets/level1_heights.bin (auto-loaded next launch)");
        if (ImGui::Button("Save grass mask")) {
            save_grass_mask();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("→ assets/level1_grass_mask.bin (auto-loaded next launch)");

        // Plateau noise — adds gentle relief on the castle pad so it
        // isn't a perfect flat. Two sliders (amplitude + frequency)
        // and an Apply button so accidental drag doesn't constantly
        // edit the heightmap.
        ImGui::SeparatorText("Plateau noise (one-shot)");
        static float plateau_amp_m = 0.6f;
        static float plateau_freq  = 0.20f;
        ImGui::SliderFloat("Plateau noise amp (m)", &plateau_amp_m,
                           0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Plateau noise freq",    &plateau_freq,
                           0.05f, 1.0f, "%.2f");
        if (ImGui::Button("Apply plateau noise")) {
            add_plateau_noise(plateau_amp_m, plateau_freq);
            log::info("[ui] plateau noise applied — Save heightmap to persist");
        }

        ImGui::SeparatorText("Sculpt brush");
        ImGui::Checkbox("Edit mode (left = brush mode, right = inverse)",
                        &terrain_edit_mode_);
        if (terrain_edit_mode_) {
            const char* modes[] = {"Raise", "Lower", "Smooth", "Flatten",
                                    "Grass +", "Grass −",
                                    "Erode +", "Erode smooth"};
            int mode_i = static_cast<int>(terrain_brush_mode_);
            if (ImGui::Combo("brush mode (left-click)", &mode_i, modes,
                              IM_ARRAYSIZE(modes))) {
                terrain_brush_mode_ = static_cast<TerrainBrushMode>(mode_i);
            }
            ImGui::TextDisabled("right-click inverts Raise↔Lower, Grass+↔Grass−, Erode+↔Erode smooth");
            ImGui::SliderFloat("brush radius (m)", &terrain_brush_radius_, 1.0f, 60.0f);
            ImGui::SliderFloat("brush strength (m/s)", &terrain_brush_strength_,
                               0.5f, 60.0f);
            ImGui::SliderFloat("brush noise (raise/lower)",
                                &terrain_brush_noise_strength_, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("brush noise freq",
                                &terrain_brush_noise_freq_, 0.05f, 1.0f, "%.2f");
            ImGui::Checkbox("Use FBM erosion detail (matches raymarched terrain)",
                            &terrain_brush_use_fbm_erosion_);
            if (terrain_brush_use_fbm_erosion_) {
                ImGui::SliderInt("FBM octaves",
                                  &terrain_brush_fbm_octaves_, 2, 9);
            }
            if (terrain_brush_mode_ == TerrainBrushMode::Flatten) {
                ImGui::SliderFloat("flatten target Y",
                                   &terrain_brush_flatten_target_, 0.0f, 200.0f);
            }
            if (terrain_brush_has_hit_) {
                ImGui::Text("brush at: (%.1f, %.1f, %.1f)",
                            terrain_brush_world_pos_.x,
                            terrain_brush_world_pos_.y,
                            terrain_brush_world_pos_.z);
            } else {
                ImGui::TextDisabled("aim at terrain to engage brush");
            }
        }

        ImGui::SeparatorText("FBM erosion bake (one-shot, sculpted area)");
        static float fbm_amp_factor = 0.30f;
        static float fbm_freq       = 0.04f;
        ImGui::SliderFloat("FBM erosion amplitude", &fbm_amp_factor, 0.05f, 1.0f);
        ImGui::SliderFloat("FBM erosion freq",       &fbm_freq,        0.005f, 0.5f);
        if (ImGui::Button("Apply FBM erosion to sculpted area")) {
            add_fbm_erosion_to_sculpted(fbm_amp_factor, fbm_freq);
        }

        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();

    ImGui::Spacing();
    ImGui::TextDisabled("ESC to resume");
    ImGui::End();
}

} // namespace qlike
