# FSR2 Integration — Phase 2+ Plan

AMD's `FidelityFX-FSR2` is now vendored at `external/FidelityFX-FSR2`
(MIT license, CC BY-NC-SA-style attribution required for redistribution).
This file documents what each subsequent session needs to do.

## Current state (end of Phase 1)
- Submodule added at `external/FidelityFX-FSR2` (commit pinned).
- Engine has `rt_.fsr2_enabled` flag (UI checkbox + persisted).
- Halton(2,3) jitter with sample-count-aware phase already routes
  through `compute_frame_view` when `fsr2_enabled` is true.
- `FrameView::jitter` (NDC) is captured per frame.
- Nothing in the engine actually consumes FSR2 yet — toggle is safe.

## Phase 2 (next session) — Build system

Goal: `quake_like.exe` links against FSR2's static libraries without
runtime use yet. Verify the SDK compiles inside our CMake project.

Tasks:
1. **DXC dependency.** FSR2 auto-compiles HLSL→SPIR-V via DXC at
   build time. Either:
   - Add a CMake step to download DXC (e.g. via FetchContent from
     the DirectX-Compiler GitHub releases), OR
   - Document that DXC must be on PATH, OR
   - Disable `FSR2_AUTO_COMPILE_SHADERS` and ship pre-compiled SPIR-V
     headers from a prebuilt FSR2 release.
2. **CMake subdirectory.** In our top-level `CMakeLists.txt` add
   `add_subdirectory(external/FidelityFX-FSR2/src/ffx-fsr2-api)`.
   Set FFX_FSR2_API_DX12=OFF, FFX_FSR2_API_VK=ON before the add.
3. **Link.** Add `ffx_fsr2_api_x64` and `ffx_fsr2_api_vk_x64` to
   the engine target's `target_link_libraries`.
4. **Include path.** Add `external/FidelityFX-FSR2/src/ffx-fsr2-api`
   to `target_include_directories`.
5. Build. Expect 50+ link errors initially around DXC; iterate.

## Phase 3 — C++ context lifecycle

Tasks:
1. New module `src/engine/vk_engine/fsr2.cpp` mirroring `taa.cpp`.
   - `init_fsr2()` — allocate scratch memory, create context once
     swapchain is sized.
   - `recreate_fsr2_context()` — called from `recreate_swapchain`.
   - `destroy_fsr2()` — context destroy + free.
2. Pull in `<ffx_fsr2.h>` and `<vk/ffx_fsr2_vk.h>`.
3. Use `ffxFsr2GetInterfaceVK` to wire the Vulkan backend.
4. Verify init/destroy cycles cleanly under preset switching.

## Phase 4 — Per-frame dispatch

Tasks:
1. In `draw()` after the world pass, prepare a
   `FfxFsr2DispatchDescription`:
   - `color` = scene_color_image_ (LR, post-tonemap or HDR)
   - `depth` = depth_image_ (LR)
   - `motionVectors` = motion_vec_image_ (LR, in UV space)
   - `motionVectorScale` = `vec2(render_extent_.width, render_extent_.height)`
     (tells FSR2 to multiply UV motion to get pixel motion)
   - `output` = a new HR image we allocate
   - `jitterOffset` = `current_frame_view_.jitter` (NDC)
   - `frameTimeDelta` = `last_frame_dt_ * 1000.0f` (milliseconds)
   - `enableSharpening` = true
   - `sharpness` = `rt_.compose_sharpen_strength * 0.5`
   - `reset` = `taau_history_valid_ == false`
   - `renderSize` = `render_extent_`
2. Call `ffxFsr2ContextDispatch(ctx, &desc)`.
3. Transition output to SHADER_READ_ONLY for compose.

## Phase 5 — Compose integration + texture mip-bias

Tasks:
1. In compose.frag, when `fsr2_enabled` route from FSR2 output
   image instead of `history_color`. Skip the EASU+RCAS path
   (FSR2 already did it).
2. Texture mip-bias on every material sampler:
   `mip_lod_bias = -log2(swapchain_w / render_w) - 1.0`. Apply
   in `init_textures()` when `fsr2_enabled`.
3. Reactive mask (Phase 5b) — generate from particles/glass
   alpha; FSR2 uses it to reduce ghosting on transparent.

## Known integration gotchas
- **Pre-tonemap vs post-tonemap input.** FSR2 prefers HDR linear
  pre-tonemap. Our compose currently runs after TAA, post-tonemap.
  We may need to add an FSR2 branch that runs BEFORE compose.
- **Motion-vec sign.** FSR2 expects `motion = current_uv − prev_uv`
  in pixel units. Our cube.frag writes UV-space (correct) and the
  sign is `current − prev` (correct). Just need motionVectorScale.
- **Jitter pattern.** Already correct (Halton(2,3) with phase count).
- **Reset on resize.** Set `reset=true` for one frame after
  recreate_swapchain.

## License notes
- AMD FidelityFX-FSR2 is MIT (`external/FidelityFX-FSR2/LICENSE.txt`).
- Vendored as git submodule; no source copied into our tree.
- Distribution-ready as long as `LICENSE.txt` is preserved.
