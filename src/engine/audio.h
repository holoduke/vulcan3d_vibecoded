#pragma once

#include <glm/vec3.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace qlike {

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

    // Pre-load a clip and tag it with a name. Subsequent trigger calls
    // refer to that name. File path is relative to CWD; falls back
    // silently if the file is missing.
    void load_clip(std::string_view name, std::string_view path);

    // One-shot at a world position with optional pitch/volume jitter.
    // pitch_jitter and volume_jitter are ± randomised offsets (e.g.
    // 0.05 = ±5%). Returns immediately, plays for the clip's duration.
    void play_at(std::string_view name, glm::vec3 pos,
                 float volume = 1.0f,
                 float pitch_jitter = 0.05f,
                 float volume_jitter = 0.05f);

    // 2D one-shot (no spatialisation). Used for player-local sounds
    // like the jump grunt where head position == listener.
    void play_local(std::string_view name,
                     float volume = 1.0f,
                     float pitch_jitter = 0.05f,
                     float volume_jitter = 0.05f);

    // Looping 2D sound. Multiple loops can be active under different
    // names. start_loop is idempotent — calling again with the same name
    // is a no-op while the loop is already playing.
    void start_loop(std::string_view name, float volume = 1.0f);
    void stop_loop(std::string_view name);
    bool is_loop_playing(std::string_view name) const;

    // Master volume (0..1). Persisted via settings.cpp.
    void set_master_volume(float v);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qlike
