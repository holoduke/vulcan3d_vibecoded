#include "engine/audio.h"
#include "engine/log.h"

// miniaudio is a single-header library. Define the implementation in
// exactly this TU; every other audio.cpp consumer just sees the small
// AudioEngine wrapper.
//
// Vorbis (.ogg) isn't supported by the core miniaudio decoders — you
// have to bring your own. miniaudio ships extras/stb_vorbis.c for
// exactly this purpose; the documented dance is "header-only declare
// stb_vorbis BEFORE miniaudio's implementation, then include
// stb_vorbis again WITHOUT the header-only flag for its definitions".
// That way miniaudio's internal Vorbis path links cleanly.
#define STB_VORBIS_HEADER_ONLY
#include "extras/stb_vorbis.c"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#undef STB_VORBIS_HEADER_ONLY
#include "extras/stb_vorbis.c"

#include <list>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qlike {

namespace {

float frand_pm(std::default_random_engine& rng, float amt) {
    if (amt <= 0.0f) return 0.0f;
    std::uniform_real_distribution<float> d(-amt, amt);
    return d(rng);
}

} // namespace

struct AudioEngine::Impl {
    ma_engine engine{};
    bool ok = false;

    // Clips indexed by ClipID — was unordered_map<string,Clip>, but
    // every play call allocated a std::string for the lookup. Now O(1)
    // array indexed by enum + zero allocations on the hot path.
    std::string clip_paths[static_cast<size_t>(ClipID::Count)];

    // Active one-shot voices. We can't free a ma_sound until it's
    // finished playing, so we keep them in a list and reap drained ones
    // at each set_listener call.
    std::list<ma_sound> oneshots;

    // (loops removed — see comment in start_loop deletion below.)

    std::default_random_engine rng{ std::random_device{}() };
};

AudioEngine::AudioEngine() : impl_(std::make_unique<Impl>()) {
    ma_result r = ma_engine_init(nullptr, &impl_->engine);
    if (r != MA_SUCCESS) {
        log::warnf("[audio] ma_engine_init failed (%d) — sound disabled", r);
        impl_->ok = false;
        return;
    }
    impl_->ok = true;
    log::info("[audio] miniaudio engine initialized");
}

AudioEngine::~AudioEngine() {
    if (!impl_->ok) return;
    for (auto& s : impl_->oneshots) ma_sound_uninit(&s);
    ma_engine_uninit(&impl_->engine);
}

bool AudioEngine::ok() const { return impl_ && impl_->ok; }

void AudioEngine::set_listener(glm::vec3 pos, glm::vec3 forward) {
    if (!impl_->ok) return;
    ma_engine_listener_set_position(&impl_->engine, 0, pos.x, pos.y, pos.z);
    ma_engine_listener_set_direction(&impl_->engine, 0,
                                       forward.x, forward.y, forward.z);
    // Reap drained one-shots.
    for (auto it = impl_->oneshots.begin(); it != impl_->oneshots.end();) {
        if (ma_sound_is_playing(&*it) == MA_FALSE) {
            ma_sound_uninit(&*it);
            it = impl_->oneshots.erase(it);
        } else {
            ++it;
        }
    }
}

void AudioEngine::load_clip(ClipID id, std::string_view path) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    impl_->clip_paths[idx].assign(path);
}

// Reap drained one-shots so the list stays bounded even if
// set_listener() stops being called (paused loop, menu open).
// Without this, every fired shot leaks an ma_sound between
// listener updates.
static void reap_oneshots(std::list<ma_sound>& oneshots) {
    for (auto it = oneshots.begin(); it != oneshots.end();) {
        if (ma_sound_is_playing(&*it) == MA_FALSE) {
            ma_sound_uninit(&*it);
            it = oneshots.erase(it);
        } else {
            ++it;
        }
    }
}

void AudioEngine::play_at(ClipID id, glm::vec3 pos,
                          float volume, float pitch_jitter, float volume_jitter) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return;
    reap_oneshots(impl_->oneshots);
    impl_->oneshots.emplace_back();
    ma_sound& s = impl_->oneshots.back();
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(),
                                 MA_SOUND_FLAG_DECODE,
                                 nullptr, nullptr, &s) != MA_SUCCESS) {
        impl_->oneshots.pop_back();
        return;
    }
    ma_sound_set_position(&s, pos.x, pos.y, pos.z);
    ma_sound_set_volume(&s, volume * (1.0f + frand_pm(impl_->rng, volume_jitter)));
    ma_sound_set_pitch (&s, 1.0f + frand_pm(impl_->rng, pitch_jitter));
    ma_sound_start(&s);
}

void AudioEngine::play_local(ClipID id,
                              float volume, float pitch_jitter, float volume_jitter) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return;
    reap_oneshots(impl_->oneshots);
    impl_->oneshots.emplace_back();
    ma_sound& s = impl_->oneshots.back();
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(),
                                 MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION,
                                 nullptr, nullptr, &s) != MA_SUCCESS) {
        impl_->oneshots.pop_back();
        return;
    }
    ma_sound_set_volume(&s, volume * (1.0f + frand_pm(impl_->rng, volume_jitter)));
    ma_sound_set_pitch (&s, 1.0f + frand_pm(impl_->rng, pitch_jitter));
    ma_sound_start(&s);
}

// start_loop / stop_loop / is_loop_playing removed — no call sites.
// If we ever need looping audio (ambience, footstep loop), add an
// enum-keyed version that mirrors play_at / play_local rather than
// resurrecting the string-keyed map.

void AudioEngine::set_master_volume(float v) {
    if (!impl_->ok) return;
    ma_engine_set_volume(&impl_->engine, v);
}

} // namespace qlike
