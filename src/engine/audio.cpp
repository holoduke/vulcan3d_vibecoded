#include "engine/audio.h"
#include "engine/log.h"

// miniaudio is a single-header library. Define the implementation in
// exactly this TU; every other audio.cpp consumer just sees the small
// AudioEngine wrapper.
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

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

    // Each loaded clip is decoded once into a ma_sound that we use as a
    // template. miniaudio doesn't directly support "clone the decoded
    // data + spawn many independent voices" via a single API, so we
    // store the file path next to the template; play_at allocates a
    // fresh ma_sound from the file (uses miniaudio's internal resource
    // manager which caches decoded audio so this is cheap after first
    // load).
    struct Clip {
        std::string path;
    };
    std::unordered_map<std::string, Clip> clips;

    // Active one-shot voices. We can't free a ma_sound until it's
    // finished playing, so we keep them in a list and reap drained ones
    // at each set_listener call.
    std::list<ma_sound> oneshots;

    // Long-running loops keyed by name. start_loop creates one,
    // stop_loop tears it down.
    std::unordered_map<std::string, ma_sound> loops;

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
    for (auto& [name, s] : impl_->loops) ma_sound_uninit(&s);
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

void AudioEngine::load_clip(std::string_view name, std::string_view path) {
    if (!impl_->ok) return;
    impl_->clips[std::string(name)] = Impl::Clip{ std::string(path) };
}

void AudioEngine::play_at(std::string_view name, glm::vec3 pos,
                          float volume, float pitch_jitter, float volume_jitter) {
    if (!impl_->ok) return;
    auto it = impl_->clips.find(std::string(name));
    if (it == impl_->clips.end()) return;
    impl_->oneshots.emplace_back();
    ma_sound& s = impl_->oneshots.back();
    if (ma_sound_init_from_file(&impl_->engine, it->second.path.c_str(),
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

void AudioEngine::play_local(std::string_view name,
                              float volume, float pitch_jitter, float volume_jitter) {
    if (!impl_->ok) return;
    auto it = impl_->clips.find(std::string(name));
    if (it == impl_->clips.end()) return;
    impl_->oneshots.emplace_back();
    ma_sound& s = impl_->oneshots.back();
    if (ma_sound_init_from_file(&impl_->engine, it->second.path.c_str(),
                                 MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION,
                                 nullptr, nullptr, &s) != MA_SUCCESS) {
        impl_->oneshots.pop_back();
        return;
    }
    ma_sound_set_volume(&s, volume * (1.0f + frand_pm(impl_->rng, volume_jitter)));
    ma_sound_set_pitch (&s, 1.0f + frand_pm(impl_->rng, pitch_jitter));
    ma_sound_start(&s);
}

void AudioEngine::start_loop(std::string_view name, float volume) {
    if (!impl_->ok) return;
    std::string key(name);
    if (impl_->loops.count(key)) {
        ma_sound_set_volume(&impl_->loops[key], volume);
        return;
    }
    auto it = impl_->clips.find(key);
    if (it == impl_->clips.end()) return;
    ma_sound& s = impl_->loops[key];
    if (ma_sound_init_from_file(&impl_->engine, it->second.path.c_str(),
                                 MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION,
                                 nullptr, nullptr, &s) != MA_SUCCESS) {
        impl_->loops.erase(key);
        return;
    }
    ma_sound_set_looping(&s, MA_TRUE);
    ma_sound_set_volume(&s, volume);
    ma_sound_start(&s);
}

void AudioEngine::stop_loop(std::string_view name) {
    if (!impl_->ok) return;
    std::string key(name);
    auto it = impl_->loops.find(key);
    if (it == impl_->loops.end()) return;
    ma_sound_uninit(&it->second);
    impl_->loops.erase(it);
}

bool AudioEngine::is_loop_playing(std::string_view name) const {
    if (!impl_->ok) return false;
    return impl_->loops.count(std::string(name)) > 0;
}

void AudioEngine::set_master_volume(float v) {
    if (!impl_->ok) return;
    ma_engine_set_volume(&impl_->engine, v);
}

} // namespace qlike
