#pragma once

#include <glm/vec3.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace qlike {

// Compile-time clip handles. Maps to a small array inside the engine —
// no string allocation per play call. Add new entries as the game
// grows; the count is the array size.
enum class ClipID : uint8_t {
    Shot = 0,
    Impact,
    Jump,
    Laser,      // player laser gun (assets/sounds/laser_shot.wav)
    AaBurst,    // anti-aircraft burst, the unmodified recording (assets/sounds/aa_burst.wav)
    SamLaunch,  // missile launch: bang + rocket roar (assets/sounds/sam_launch.wav, synthesised)
    JetFlyby1,  // jet flybys (15 s recordings, one pass each)
    JetFlyby2,
    JetFlyby3,
    ExplClose1, ExplClose2, ExplClose3, ExplClose4, ExplClose5,   // war explosions, close-up
    ExplFar1, ExplFar2, ExplFar3, ExplFar4, ExplFar5,              // distant rumbling explosions
    Count
};

// Thin wrapper around miniaudio's high-level engine. Hides the C-style
// pointer dance and exposes only what the game needs: load a clip once
// from disk, then trigger one-shots at world positions, or start/stop a
// 2D loop (footsteps, ambient).
//
// Lifecycle: construct AudioEngine once at engine init (after window
// creation, before draw); call set_listener every frame from the camera;
// destruct on shutdown (the dtor stops everything cleanly).
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    // True if miniaudio came up successfully. False = no audio device or
    // init failure; trigger calls become silent no-ops.
    bool ok() const;

    // Update the 3D listener (camera position + forward). Call once per
    // render frame.
    void set_listener(glm::vec3 pos, glm::vec3 forward);

    // Pre-load a clip and bind it to an enum slot. Subsequent trigger
    // calls reference the ClipID directly (no string lookup). File
    // path is relative to CWD; falls back silently if missing.
    void load_clip(ClipID id, std::string_view path);

    // One-shot at a world position with optional pitch/volume jitter.
    // pitch_jitter and volume_jitter are ± randomised offsets (e.g.
    // 0.05 = ±5%). Returns immediately, plays for the clip's duration.
    void play_at(ClipID id, glm::vec3 pos,
                 float volume = 1.0f,
                 float pitch_jitter = 0.05f,
                 float volume_jitter = 0.05f);

    // 2D one-shot (no spatialisation). Used for player-local sounds
    // like the jump grunt where head position == listener.
    void play_local(ClipID id,
                     float volume = 1.0f,
                     float pitch_jitter = 0.05f,
                     float volume_jitter = 0.05f);

    // Tracked spatial voice: panned by position, NO distance attenuation
    // (the caller drives volume - e.g. a flyby recording already carries
    // its own approach / recede loudness) and no Doppler. `rate` scales
    // playback speed (and pitch); `start_s` seeks into the clip.
    // Returns a handle (0 = failed).
    uint32_t play_tracked(ClipID id, glm::vec3 pos, float volume,
                          float rate, float start_s);
    // Fire-and-forget spatial effect: panned, no distance attenuation
    // (caller sets volume), starts after `delay_s` (speed of sound) and
    // runs through its own low-pass at `lpf_hz` (air absorption; <= 0 or
    // >= 18 kHz = no filter).
    void     play_effect(ClipID id, glm::vec3 pos, float volume, float pitch,
                         float delay_s, float lpf_hz, int lpf_order = 2, float start_s = 0.0f);
    size_t   active_effects() const;          // scheduled + playing play_effect voices
    void     set_voice(uint32_t h, glm::vec3 pos, float volume);
    void     set_voice_rate(uint32_t h, float rate);
    float    voice_time(uint32_t h) const;   // playback position in the clip (s)
    void     stop_voice(uint32_t h, float fade_s);
    bool     voice_playing(uint32_t h) const;

    // Master volume (0..1). Persisted via settings.cpp.
    void set_master_volume(float v);

    // Freeze / resume every voice (pause menu). Scheduled starts keep their
    // remaining delay because the engine clock stops with the device.
    void set_paused(bool paused);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qlike
