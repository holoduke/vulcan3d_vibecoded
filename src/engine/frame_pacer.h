// Precise frame-rate cap.
//
// std::this_thread::sleep_until() on Windows wakes on the scheduler tick
// (1 ms at best, 15.6 ms by default), so a 5.55 ms (180 Hz) target was
// overshot by ~1 ms EVERY frame: 150 fps on a 180 Hz display, with uneven
// frame times. FramePacer sleeps on a high-resolution waitable timer up to
// a short margin before the deadline and spins the rest, and schedules
// each deadline from the previous DEADLINE (not from the late wake-up), so
// a late frame is made up by the next one instead of lowering the rate.
#pragma once

#include <chrono>

namespace qlike {

class FramePacer {
public:
    using clock = std::chrono::steady_clock;

    explicit FramePacer(std::chrono::nanoseconds period);
    ~FramePacer();
    FramePacer(const FramePacer&) = delete;
    FramePacer& operator=(const FramePacer&) = delete;

    // Blocks until this frame's deadline, then arms the next one.
    void wait();
    // Change the cap (e.g. DLSS Frame Generation: refresh / multiplier).
    void set_period(std::chrono::nanoseconds p) { period_ = p; }

private:
    std::chrono::nanoseconds period_;
    clock::time_point next_;
    void* timer_ = nullptr;   // HANDLE (high-resolution waitable timer) or null
};

} // namespace qlike
