#include "engine/frame_pacer.h"

#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

namespace qlike {

namespace {
// Woken this long before the deadline; the remainder is spun. The
// high-resolution timer is accurate to ~0.1-0.5 ms.
constexpr std::chrono::microseconds kSpinMargin{ 700 };
}

FramePacer::FramePacer(std::chrono::nanoseconds period)
    : period_(period), next_(clock::now() + period) {
#ifdef _WIN32
    timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                    TIMER_ALL_ACCESS);   // Windows 10 1803+
#endif
}

FramePacer::~FramePacer() {
#ifdef _WIN32
    if (timer_) CloseHandle(static_cast<HANDLE>(timer_));
#endif
}

void FramePacer::wait() {
    auto now = clock::now();
    // More than a frame late (hitch, window drag, breakpoint): do not try
    // to catch up with a burst of uncapped frames.
    if (now > next_ + period_) next_ = now;

    if (now < next_) {
        const auto coarse = next_ - kSpinMargin;
#ifdef _WIN32
        if (timer_ && now < coarse) {
            LARGE_INTEGER due;
            due.QuadPart = -std::chrono::duration_cast<std::chrono::nanoseconds>(coarse - now).count() / 100;
            if (due.QuadPart < 0 && SetWaitableTimer(static_cast<HANDLE>(timer_), &due, 0, nullptr, nullptr, FALSE))
                WaitForSingleObject(static_cast<HANDLE>(timer_), 50);
        } else if (!timer_ && now < coarse - std::chrono::milliseconds(2)) {
            std::this_thread::sleep_until(coarse - std::chrono::milliseconds(2));
        }
        while (clock::now() < next_) _mm_pause();
#else
        if (now < coarse) std::this_thread::sleep_until(coarse);
        while (clock::now() < next_) std::this_thread::yield();
#endif
    }
    next_ += period_;
}

} // namespace qlike
