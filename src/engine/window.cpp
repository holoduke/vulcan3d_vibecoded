#include "engine/window.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>

namespace qlike {

namespace {
int g_glfw_refcount = 0;
}

Window::Window(const WindowConfig& cfg) {
    if (g_glfw_refcount == 0) {
        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    }
    ++g_glfw_refcount;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(static_cast<int>(cfg.width),
                               static_cast<int>(cfg.height),
                               cfg.title.c_str(), nullptr, nullptr);
    if (!window_) throw std::runtime_error("glfwCreateWindow failed");

    glfwSetWindowUserPointer(window_, this);

    if (cfg.capture_cursor) set_cursor_captured(true);

    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(window_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
}

Window::~Window() {
    if (window_) glfwDestroyWindow(window_);
    if (--g_glfw_refcount == 0) glfwTerminate();
}

bool Window::should_close() const { return glfwWindowShouldClose(window_); }
void Window::poll_events() { glfwPollEvents(); }
void Window::request_close() { glfwSetWindowShouldClose(window_, GLFW_TRUE); }
bool Window::has_focus() const {
    return glfwGetWindowAttrib(window_, GLFW_FOCUSED) == GLFW_TRUE;
}

void Window::framebuffer_size(uint32_t& w, uint32_t& h) const {
    int iw = 0, ih = 0;
    glfwGetFramebufferSize(window_, &iw, &ih);
    w = static_cast<uint32_t>(iw);
    h = static_cast<uint32_t>(ih);
}

VkSurfaceKHR_T* Window::create_surface(VkInstance_T* instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(reinterpret_cast<VkInstance>(instance),
                                window_, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("glfwCreateWindowSurface failed");
    }
    return reinterpret_cast<VkSurfaceKHR_T*>(surface);
}

void Window::set_cursor_captured(bool capture) {
    cursor_captured_ = capture;
    glfwSetInputMode(window_, GLFW_CURSOR,
                     capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    first_cursor_sample_ = true;  // re-anchor delta tracking
}

InputFrame Window::consume_input() {
    InputFrame f{};
    auto down = [&](int key) { return glfwGetKey(window_, key) == GLFW_PRESS; };
    f.fwd   = down(GLFW_KEY_W);
    f.back  = down(GLFW_KEY_S);
    f.left  = down(GLFW_KEY_A);
    f.right = down(GLFW_KEY_D);
    f.jump     = down(GLFW_KEY_SPACE);
    f.sprint   = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);
    f.crawl    = down(GLFW_KEY_LEFT_CONTROL) || down(GLFW_KEY_RIGHT_CONTROL);
    f.menu_key = down(GLFW_KEY_ESCAPE);

    bool fire_now = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    f.fire = fire_now && !prev_fire_;  // rising edge
    f.fire_held = fire_now;             // level (held)
    prev_fire_ = fire_now;

    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(window_, &cx, &cy);
    if (first_cursor_sample_) {
        last_cursor_x_ = cx;
        last_cursor_y_ = cy;
        first_cursor_sample_ = false;
    }
    f.mouse_dx = cx - last_cursor_x_;
    f.mouse_dy = cy - last_cursor_y_;
    last_cursor_x_ = cx;
    last_cursor_y_ = cy;

    return f;
}

} // namespace qlike
