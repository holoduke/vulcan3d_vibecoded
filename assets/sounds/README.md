## Sound assets

The engine loads these clips at startup. Missing files are silently
skipped — the game runs without sound if this dir is empty.

| File          | Trigger                     | Suggested length |
|---------------|-----------------------------|------------------|
| `shot.wav`    | Each fired bullet (local)   | 0.2-0.4 s        |
| `impact.wav`  | Bullet hits surface (3D)    | 0.3-0.6 s        |
| `jump.wav`    | Player leaves ground        | 0.2-0.4 s        |
| `step.wav`    | Looping while walking       | 0.4-0.8 s seamless loop |
| `land.wav`    | Player touches ground       | 0.2-0.4 s        |

Files must be `.wav` (PCM 16-bit / float). miniaudio also accepts MP3 +
FLAC + OGG; if you change extensions update the paths in
`src/engine/vk_engine.cpp`'s `init()` block.

### Where to get CC0 assets fast

- **kenney.nl/assets** — has `Impact Sounds`, `Footsteps`, `Sci-fi Sounds`
  packs. CC0, drop-in ready.
- **opengameart.org** — filter by license = CC0 / Public Domain.
  Search "gunshot", "footstep concrete", "thump".
- **freesound.org** — filter by License = Creative Commons 0.

Pick anything that fits the durations above; rename to the file names
in the table; you're done.
