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

#include <algorithm>
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

    // Effect voices (explosions): sound -> optional low-pass node ->
    // endpoint. std::list keeps addresses stable for the mixer thread.
    struct Effect {
        ma_sound    sound;
        ma_lpf_node lpf;
        bool        has_lpf = false;
    };
    std::list<Effect> effects;

    // Tracked voices (jet flybys): heap-allocated so the ma_sound never
    // moves while miniaudio's mixer thread references it.
    std::unordered_map<uint32_t, ma_sound*> tracked;
    uint32_t next_handle = 1;

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
    for (auto& kv : impl_->tracked) { ma_sound_uninit(kv.second); delete kv.second; }
    for (auto& e : impl_->effects) {
        ma_sound_uninit(&e.sound);
        if (e.has_lpf) ma_lpf_node_uninit(&e.lpf, nullptr);
    }
    ma_engine_uninit(&impl_->engine);
}

bool AudioEngine::ok() const { return impl_ && impl_->ok; }

void AudioEngine::set_listener(glm::vec3 pos, glm::vec3 forward) {
    if (!impl_->ok) return;
    ma_engine_listener_set_position(&impl_->engine, 0, pos.x, pos.y, pos.z);
    ma_engine_listener_set_direction(&impl_->engine, 0,
                                       forward.x, forward.y, forward.z);
    // Reap finished effects (scheduled-but-not-started ones report
    // playing, so delayed explosions survive until they have sounded).
    for (auto it = impl_->effects.begin(); it != impl_->effects.end();) {
        if (ma_sound_is_playing(&it->sound) == MA_FALSE && ma_sound_at_end(&it->sound)) {
            ma_sound_uninit(&it->sound);
            if (it->has_lpf) ma_lpf_node_uninit(&it->lpf, nullptr);
            it = impl_->effects.erase(it);
        } else {
            ++it;
        }
    }
    // Reap finished tracked voices (handle then reads as not playing).
    for (auto it = impl_->tracked.begin(); it != impl_->tracked.end();) {
        if (ma_sound_is_playing(it->second) == MA_FALSE) {
            ma_sound_uninit(it->second);
            delete it->second;
            it = impl_->tracked.erase(it);
        } else {
            ++it;
        }
    }
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

uint32_t AudioEngine::play_tracked(ClipID id, glm::vec3 pos, float volume,
                                   float rate, float start_s) {
    if (!impl_->ok) return 0;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return 0;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return 0;
    ma_sound* s = new ma_sound;
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(), MA_SOUND_FLAG_DECODE,
                                nullptr, nullptr, s) != MA_SUCCESS) {
        delete s;
        return 0;
    }
    ma_sound_set_attenuation_model(s, ma_attenuation_model_none);
    ma_sound_set_doppler_factor(s, 0.0f);
    ma_sound_set_position(s, pos.x, pos.y, pos.z);
    ma_sound_set_volume(s, volume);
    ma_sound_set_pitch(s, rate);
    if (start_s > 0.0f) {
        ma_uint32 sr = 0;
        ma_sound_get_data_format(s, nullptr, nullptr, &sr, nullptr, 0);
        if (sr) ma_sound_seek_to_pcm_frame(s, ma_uint64(double(start_s) * sr));
        ma_sound_set_fade_in_milliseconds(s, 0.0f, 1.0f, 700);   // mid-clip start: no hard onset
    }
    ma_sound_start(s);
    const uint32_t h = impl_->next_handle++;
    if (impl_->next_handle == 0) impl_->next_handle = 1;
    impl_->tracked[h] = s;
    return h;
}

void AudioEngine::set_voice(uint32_t h, glm::vec3 pos, float volume) {
    if (!impl_->ok) return;
    auto it = impl_->tracked.find(h);
    if (it == impl_->tracked.end()) return;
    ma_sound_set_position(it->second, pos.x, pos.y, pos.z);
    ma_sound_set_volume(it->second, volume);
}

void AudioEngine::play_effect(ClipID id, glm::vec3 pos, float volume, float pitch,
                              float delay_s, float lpf_hz) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return;
    if (impl_->effects.size() > 24) return;        // runaway guard (bomb carpets)
    impl_->effects.emplace_back();
    Impl::Effect& e = impl_->effects.back();
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(), MA_SOUND_FLAG_DECODE,
                                nullptr, nullptr, &e.sound) != MA_SUCCESS) {
        impl_->effects.pop_back();
        return;
    }
    if (lpf_hz > 0.0f && lpf_hz < 18000.0f) {
        ma_lpf_node_config lc = ma_lpf_node_config_init(
            ma_engine_get_channels(&impl_->engine), ma_engine_get_sample_rate(&impl_->engine),
            lpf_hz, 2);
        if (ma_lpf_node_init(ma_engine_get_node_graph(&impl_->engine), &lc, nullptr, &e.lpf) == MA_SUCCESS) {
            e.has_lpf = true;
            ma_node_attach_output_bus(&e.lpf, 0, ma_engine_get_endpoint(&impl_->engine), 0);
            ma_node_attach_output_bus(&e.sound, 0, &e.lpf, 0);
        }
    }
    ma_sound_set_attenuation_model(&e.sound, ma_attenuation_model_none);
    ma_sound_set_doppler_factor(&e.sound, 0.0f);
    ma_sound_set_position(&e.sound, pos.x, pos.y, pos.z);
    ma_sound_set_volume(&e.sound, volume);
    ma_sound_set_pitch(&e.sound, pitch);
    if (delay_s > 0.0f)
        ma_sound_set_start_time_in_milliseconds(
            &e.sound, ma_engine_get_time_in_milliseconds(&impl_->engine) + ma_uint64(delay_s * 1000.0f));
    ma_sound_start(&e.sound);
}

void AudioEngine::set_voice_rate(uint32_t h, float rate) {
    if (!impl_->ok) return;
    auto it = impl_->tracked.find(h);
    if (it != impl_->tracked.end()) ma_sound_set_pitch(it->second, rate);
}

float AudioEngine::voice_time(uint32_t h) const {
    if (!impl_->ok) return 0.0f;
    auto it = impl_->tracked.find(h);
    if (it == impl_->tracked.end()) return 0.0f;
    float t = 0.0f;
    ma_sound_get_cursor_in_seconds(it->second, &t);
    return t;
}

void AudioEngine::stop_voice(uint32_t h, float fade_s) {
    if (!impl_->ok) return;
    auto it = impl_->tracked.find(h);
    if (it == impl_->tracked.end()) return;
    ma_uint64 ms = ma_uint64((std::max)(0.0f, fade_s) * 1000.0f);
    // Fade from the current volume, then stop (reaped in set_listener).
    ma_sound_set_stop_time_with_fade_in_milliseconds(
        it->second, ma_engine_get_time_in_milliseconds(&impl_->engine) + ms, ms);
}

bool AudioEngine::voice_playing(uint32_t h) const {
    if (!impl_->ok) return false;
    auto it = impl_->tracked.find(h);
    return it != impl_->tracked.end() && ma_sound_is_playing(it->second) == MA_TRUE;
}

void AudioEngine::set_master_volume(float v) {
    if (!impl_->ok) return;
    ma_engine_set_volume(&impl_->engine, v);
}

} // namespace qlike
