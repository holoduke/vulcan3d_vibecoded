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

    // Master volume (0..1). Persisted via settings.cpp.
    void set_master_volume(float v);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qlike
