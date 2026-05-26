# quake-like

Vulkan 1.3 + KHR ray-query first-person shooter. C++20, Windows-first.

A small-arena combat playground that pushes modern rendering techniques
on a single graphics queue: inline RT shadows / GI, denoised via SVGF,
upscaled via FSR2 / FSR3 frame-gen, on procedural CDLOD heightmap
terrain with grass and water.

## Building

```powershell
./build.ps1 -Config Release
./build/quake_like.exe
```

Requires Visual Studio 2022 + Vulkan SDK 1.3.x with ray-query support
(RTX 2060+, RX 6700+, Arc A380+). Dependencies are pulled by CMake's
`FetchContent` (GLFW, glm, ImGui, VMA, Vulkan-Bootstrap, JoltPhysics,
tinygltf, tinyexr, miniaudio, FastNoiseLite). FidelityFX is vendored
under `external/`.

## Rendering pipeline (high level)

| Pass | What |
|------|------|
| Depth pre-pass | Brushes + dyn props + terrain (depth only). |
| World colour | cube.frag draws brushes / dyn props / terrain with inline RT shadows (PCSS or half-rate consumer), AO (off / GTAO-cheap / RTAO / HBAO), GI (ReSTIR), SPOM parallax, anti-tile, FBM far detail. MRT writes scene_color + motion_vec. |
| Sun shadow LR | Half-rate occlusion image — bilateral-upsampled in cube.frag when enabled. |
| Grass raymarch | SDF blades over the heightmap, marched in screen space. |
| Water plane | 64×64 subdivided rasterised plane with depth-bias, foam, atmospheric water tint, RT reflections. |
| Tessellation (optional) | Near terrain chunks get GPU tess for vertex displacement. |
| SVGF | Variance-moments reprojection + 3-pass à-trous (toggle). |
| FSR3 / FSR2 / TAA | Upscale chain. |
| Bloom | Karis-averaged down + tent up mip chain. |
| Compose | Tonemap + lens flare + chromatic + vignette + ImGui. |

## Terrain

CDLOD heightmap chunks, 8 LOD levels (strides 1/2/4/8/16/32/64/128),
per-LOD distance + density sliders, CD-LOD morph for LOD 0→1, vertex
skirts on chunk edges for crack-free seams. Optional GPU tessellation
near the camera. Per-chunk VBO supersample (1×/2×/4×/8×) inside a
configurable near-radius for dense LOD 0 mesh past heightmap
resolution. 5-material procedural splat + 2 PNG materials (stylized
grass + rocky-rugged) with anti-tile rotated-sample blend, gated by
the grass-eligibility mask. Per-pixel SPOM + per-vertex displacement
both sample the same rocky height map — they stay aligned. Atmospheric
fog (exp² distance + height attenuation + Mie sun-aligned forward scatter).

## Combat

- Lasers fire as glowing cylinders, gravity-arced, ricochet on shallow hits.
- Particle sparks via Jolt rigid bodies.
- Spacejet flyovers (1.5-4 s cadence, 95-130 m altitude, biased toward
  multi-jet formations) — shootable, 3-hit kill, big screen-flash + sphere
  burst on destruction. Hit-test uses segment-vs-AABB (no tunneling on
  fast closures).
- Top-centre damage HUD shows last-hit jet's health, fades over 3 s.

## Movement

Quake-feel ground accel + air-accel, double jump (1 air jump, +10%
height; up to +20% bonus when strafe-jumping).

## In-game controls

- WASD + Space (jump) + Shift (sprint) + Ctrl (crawl)
- Mouse-look + LMB (fire) / RMB (alt; sculpt-lower in edit mode)
- E — toggle terrain edit mode (sculpt brush)
- 9 — toggle terrain wireframe
- [ / ] — shrink / grow sculpt brush
- Q / R — cycle brush mode (raise / lower / erosion / paint)
- F12 — screenshot
- ESC — pause menu (all sliders live here)

## Project layout

- `src/engine/` — Vulkan engine: pipelines, descriptors, terrain,
  SVGF, FSR, sun shadow, grass, water, audio.
- `src/game/` — gameplay: player movement, physics integration, level.
- `shaders/` — GLSL 4.60 (compiled to SPIR-V via glslang at build time).
- `assets/` — meshes, textures, audio.
- `docs/` — implementation plans (terrain migration, ReSTIR, SVGF, FSR3, etc.).
- `external/FidelityFX-FSR2`, `external/FidelityFX-SDK` — vendored upscalers.

## Tuning surface

Most rendering knobs are live sliders in the pause menu (Quality,
Terrain, Terrain LOD, Grass, Water, Fog, AO, Shadow, Bloom). Settings
persist in `qlike_settings.cfg`.

## Recent backlog highlights

- SVGF deep denoiser (variance moments + 3-pass à-trous)
- HBAO (ao_mode 3)
- Dense LOD 0 supersample with per-chunk hysteresis
- Anti-tile rotated 2nd sample on terrain splat
- Realistic atmospheric fog (Mie scattering + height attenuation)
- 8-LOD CDLOD with per-band UI sliders, AABB-clamp distance picker
- Rasterised water plane with depth-bias + 64×64 subdiv to kill
  shoreline wobble
- Stylized grass + rocky-rugged texture integration with grass-mask
  gating
