#pragma once

#include <cstdint>
#include <string>

struct GLFWwindow;
struct VkSurfaceKHR_T;
struct VkInstance_T;

namespace qlike {

struct WindowConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    std::string title = "quake-like";
    bool capture_cursor = false;
};

struct InputFrame {
    bool fwd = false, back = false, left = false, right = false;
    bool jump = false;
    bool sprint = false;    // Left Shift — 1.5× walk speed
    bool crawl = false;     // Left Ctrl — 0.4× walk speed (sprint wins if both)
    bool menu_key = false;  // ESC; engine edge-detects to toggle pause
    bool edit_key = false;  // E; engine edge-detects to toggle terrain edit mode
    bool brush_smaller = false; // `[` — level; engine shrinks brush while held
    bool brush_larger  = false; // `]` — level; engine grows brush while held
    bool brush_mode_prev = false; // Q; engine edge-detects to cycle brush mode
    bool brush_mode_next = false; // R; engine edge-detects to cycle brush mode
    bool fire = false;      // left mouse, edge-triggered (one click = one)
    bool fire_held = false; // left mouse, level (true while button is down)
    bool alt_fire_held = false; // right mouse, level (true while button is down)
                                // — used by sculpt brush as "lower" alt
    bool screenshot = false;// F12, edge-triggered
    double mouse_dx = 0.0;
    double mouse_dy = 0.0;
};

class Window {
public:
    explicit Window(const WindowConfig& cfg);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    GLFWwindow* handle() const { return window_; }
    bool should_close() const;
    void poll_events();

    void framebuffer_size(uint32_t& w, uint32_t& h) const;
    VkSurfaceKHR_T* create_surface(VkInstance_T* instance) const;

    // Read accumulated input since the last call, then clear edge-triggered bits.
    InputFrame consume_input();

    void set_cursor_captured(bool capture);
    bool cursor_captured() const { return cursor_captured_; }
    bool has_focus() const;

    void request_close();

private:
    GLFWwindow* window_ = nullptr;
    bool cursor_captured_ = false;
    double last_cursor_x_ = 0.0;
    double last_cursor_y_ = 0.0;
    bool first_cursor_sample_ = true;
    bool fire_edge_ = false;
    bool prev_fire_ = false;
    bool prev_screenshot_ = false;
};

} // namespace qlike
