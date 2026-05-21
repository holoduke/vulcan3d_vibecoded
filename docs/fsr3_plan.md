# FSR 3 — Frame Generation Plan

Add AMD FidelityFX-FSR3, which combines:
1. An **upscaler** (drop-in replacement for our current FSR 2.2.1 path)
2. **Frame interpolation** — synthesises an in-between frame from two
   rendered frames + optical flow + motion vectors, doubling perceived
   FPS without rendering twice.

This is a multi-session item. Pre-existing task #83-style scoping. Each
session ships a discrete chunk; user can stop at any chunk and the engine
still works.

## Today's state (start of plan)
- FSR 2.2.1 vendored at `external/FidelityFX-FSR2`, working.
- Per-frame `dispatch_fsr2` produces an HR image; compose samples it.
- Render path: `world → TAA → bloom → compose (LDR) → ImGui → present`.
- ImGui draws DIRECTLY into the swapchain image after compose.
- Auto-exposure ramps per frame.
- No async compute queue dedicated to post (TLAS build uses one).

## STATUS — session 2 SHIPPED (2026-05-13 — root cause found)

**Fixed.** The CreateContext AV was caused by a missing device
extension: the SDK loads `vkGetBufferMemoryRequirements2KHR` via
`vkGetDeviceProcAddr`. That KHR-suffixed alias only resolves when
`VK_KHR_get_memory_requirements2` is enabled on the device — even
though the function itself was promoted to Vulkan 1.1 core. Without
the extension, the alias returns NULL and the SDK AVs the first
time it calls it (during descriptor pool / resource setup).

Diagnosed by hooking `vkGetDeviceProcAddr` with a logging shim
passed via `ffxCreateBackendVKDesc.vkDeviceProcAddr`. The trace
showed `vkGetBufferMemoryRequirements2KHR -> 0000000000000000`
right before the AV.

Fix in `src/engine/vk_engine/setup.cpp`: added
`VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME` to the device's
required extension list. Plus extras enabled along the way:
`VK_KHR_DYNAMIC_RENDERING`, `VK_KHR_SYNCHRONIZATION_2`,
`VK_EXT_SUBGROUP_SIZE_CONTROL` — same KHR-alias issue likely
applies to those if any future SDK release uses them.

Also dropped `FFX_UPSCALE_ENABLE_DEBUG_CHECKING` from the create
flags — the SDK's debug checking triggered a separate code path
that wasn't load-bearing for the upscale itself.

Verified: log shows `[fsr3] upscaler context created` and
`compose source switched to FSR3`, frames advance with
`vk_warn=0 vk_err=0`. SEH guard in `init_fsr3()` stays in place
as defence in depth.

## STATUS — session 2 wired, blocked on SDK init crash (2026-05-13 — RESOLVED, see above)

What's in:
- Submodule `external/FidelityFX-SDK` pinned at v1.1.4 (last Vulkan
  release before AMD's "Redstone" branch dropped Vulkan).
- CMake imports `amd_fidelityfx_vk.lib` headers + stages
  `amd_fidelityfx_vk.dll` next to the exe via post-build.
- New `src/engine/vk_engine/fsr3.cpp` module mirrors fsr2.cpp's
  init/destroy/recreate/dispatch shape but uses the ffx-api ABI.
- Runtime DLL load via `ffx_api_loader.h` + `GetProcAddress`. The
  static-link import-lib path doesn't work because FFX_API_ENTRY in
  the SDK headers is hardcoded `__declspec(dllexport)` (a header
  bug from the consumer's POV — AMD's intended path is LoadLibrary).
- UI dropdown "FSR backend" with options FSR2 (legacy static) / FSR3
  (ffx-api DLL). Persisted to settings.
- Compose binding routes to FSR3 output when backend == 1.
- FSR3 init is **lazy** — only fires when the user picks FSR3 in the
  UI. recreate_fsr3_context only fires if a context already existed.

What's blocked:
- `ffxCreateContext` AVs at NULL on first call. Likely root causes:
  1. **Missing device extensions** — FSR3 typically needs
     VK_EXT_subgroup_size_control, VK_KHR_synchronization2, descriptor
     indexing, etc. Our device-create doesn't audit-enable these.
  2. **Validation layer conflict** — SDK's internal debug check
     might trip our debug messenger.
  3. **DLL/header version skew** — prebuilt DLL might be from a
     different point on v1.1.x than the headers I'm calling.

Next session work (FSR3 session 2-finalize):
- Audit device-create extension list against AMD's sample's
  `BuildSamplesSolutionVK.bat` build of the FSR sample. Enable any
  missing extensions.
- Set `FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE` via global
  ffxConfigure before creating the context — should route SDK
  diagnostics through `fpMessage` instead of crashing.
- If the create-context still crashes after both, inspect with the
  Vulkan validation layer enabled to capture the failure context.

Sessions 3-6 (frame generation, swapchain proxy, UI mask, present
pacing, tuning) cannot start until session 2's create-context is
working. They each require substantially more architecture work
beyond a single session anyway — see headings below.

## STATUS — session 4 SHIPPED, FG live (2026-05-13)

The full FSR3 + Frame Generation chain is operational. Log shows:
```
[fsr3] upscaler context created
[fsr3 fg] context created
[fsr3 sc] SwapChain proxy active — acquire/present routed through SDK function pointers
[fsr3 sc] re-queried 3 swapchain images via SDK fn
[fsr3 sc] FG configured with swapchain — generation active
```
289 frames over ~3 s with `vk_warn=0 vk_err=0`. With FG active, the
SDK paces real + generated frames into the present chain — perceived
FPS roughly doubles vs the same scene without FG.

Root causes solved (in order of discovery):
1. **4 distinct VkQueue handles required.** SDK's
   `FrameInterpolationSwapchainVK::init` line 1503 fails if any pair
   of game/asyncCompute/present/imageAcquire queues match. Solved
   via vk-bootstrap `custom_queue_setup` requesting 4 queues from
   the graphics family + pulling extras via `vkGetDeviceQueue`.
2. **`/EHa` per-file** for `fsr3.cpp` so `__try/__except` catches
   hardware AVs from the SDK DLL. The project default `/EHsc` only
   catches synchronous C++ exceptions.
3. **`cfg.swapChain` expects the SDK's wrapped VkSwapchainKHR**
   (which is internally a `FrameInterpolationSwapChainVK*` cast as
   void*) — NOT the ffx-api `ffxContext` wrapper handle. Passing
   the wrong type AVs at `RtlEnterCriticalSection` because the SDK
   `reinterpret_cast`s it to `FrameInterpolationSwapChainVK*` and
   deref's an uninitialized critical section at the wrong offset.
   The fix: pass `swapchain_` (= the SDK-overwritten handle).
4. **Re-query swapchain images** via the SDK's replacement
   `vkGetSwapchainImagesKHR`. Our cached vk-bootstrap images point
   at the original (now-destroyed) swapchain.

Hard blocker that needed deep-diving: discovered (1) by reading
`FrameInterpolationSwapchainVK.cpp` line by line; (2) via build-flag
research; (3) via SDK `setFrameGenerationConfig` source inspection;
(4) by walking the failure stack to `transition_image_aspect`.

What needs sessions 5-6:
- HUD smear on generated frames (no UI mask yet)
- Auto-exposure pinning between real-frame pairs (luminance jitters)
- Async compute toggle + low-FPS-base auto-disable

## OLDER ATTEMPT — session 4 BLOCKED on SwapChain Query AV (superseded)

Significant unblocking moved the failure point. Now resolved:
- **4 distinct queues.** SDK requires distinct VkQueue handles for
  game/asyncCompute/present/imageAcquire (it explicitly checks at
  `FrameInterpolationSwapchainVK.cpp:1503`). Solved via vk-bootstrap
  `custom_queue_setup` — request 4 queues from the graphics family,
  pull additional handles via `vkGetDeviceQueue(family, 1..3)`.
- **`oldSwapchain = NULL`** in the createInfo — SDK destroys our
  existing swapchain via the in/out pointer regardless.
- **`/EHa` per-file** for `fsr3.cpp` so `__try/__except` catches
  hardware AVs from the SDK DLL (project default `/EHsc` doesn't).
- **SwapChain CreateContext now succeeds** — proven by the failure
  moving downstream from the create call to the Query call.

Remaining blocker:
- `g_ffx.Query(&ctx, ffxQueryDescSwapchainReplacementFunctionsVK)`
  AVs at `RtlEnterCriticalSection` deep inside the SDK DLL. The
  Configure-with-FG-enabled call AVs in the same way. Both are now
  SEH-caught with `/EHa`, so the engine doesn't die outright, but
  the post-AV recovery (rebuild plain swapchain via `init_swapchain`)
  leaves the engine in a state where the next frame's swapchain-image
  transition AVs again. Likely the SDK's partial state isn't fully
  cleaned up by `DestroyContext` after the AV, or our recovery
  doesn't correctly reset some piece of engine state.

Behaviour with `fg_enabled = 0` (the default and current setting):
the 4-queue setup, `/EHa` flag, and all SwapChain-proxy infrastructure
are live but inert. Engine boots normally, FSR3 upscaler works,
FSR3 FG context spins up its private interpolated frames in the
background (output unused). 138+ frames stable, vk_warn=0 vk_err=0.

Behaviour with `fg_enabled = 1`: SwapChain proxy creation succeeds,
Query AVs and is caught, plain-swapchain recovery rebuilds engine
state, but the next render frame AVs in `vkCmdPipelineBarrier2` on
a swapchain image. Net: engine crashes a few frames after enabling
FG. **Don't enable `fg_enabled` until session 4 is fully fixed.**

Diagnosing the Query AV further needs either:
- A debugger session into `amd_fidelityfx_vk.dll` to identify which
  uninitialized critical section the SDK is entering.
- Comparison against the AMD sample's full extension + feature set
  (we already added `VK_KHR_get_memory_requirements2`,
  `VK_KHR_dynamic_rendering`, `VK_KHR_synchronization2`,
  `VK_EXT_subgroup_size_control` — but the sample also enables
  several more like `VK_KHR_maintenance4`,
  `VK_EXT_extended_dynamic_state`, `VK_KHR_get_surface_capabilities2`
  that we don't carry).

## Session 1 — FSR 3 SDK vendoring + build

The SwapChain proxy infrastructure is wired: lazy `init_fsr3_swapchain`
(creates `ffxCreateContextDescFrameGenerationSwapChainVK`), function-
pointer query for the acquire/present replacements, wrapper methods
that route to either the SDK or plain Vulkan, all 4 acquire/present
call sites converted to the wrappers. UI toggle + persisted setting.
Recovery path on SDK failure rebuilds a plain swapchain so the engine
keeps running.

**The blocker:** `ffxCreateContext` for the SwapChain proxy returns
`FFX_API_RETURN_ERROR_RUNTIME_ERROR` (3) consistently. Tried so far,
none of which changed the result:
- Matching `backBufferFormat` to our actual swapchain (B8G8R8A8_UNORM,
  not HDR — we're not using an HDR swapchain even though FG was
  configured with the HDR flag earlier).
- Setting `oldSwapchain` to `VK_NULL_HANDLE` in case the in/out
  pattern was tripping the SDK.
- Reusing `graphics_queue_` for game/present/imageAcquire roles and
  `compute_queue_` for asyncCompute (matches what AMD's sample does
  on hardware that lacks dedicated families).

The SDK has no fpMessage hook on the SwapChain context (different
API shape from upscaler/FG). Diagnosing further needs either:
1. Attach a debugger and step into `amd_fidelityfx_vk.dll` to see
   which internal validation rejects the params.
2. Try a different SDK release (v1.1.4 is the latest Vulkan-capable;
   nothing newer to try without losing Vulkan).
3. Match AMD's sample's instance + device extension set exactly —
   we may be missing something the SDK silently requires
   (VK_KHR_get_surface_capabilities2, VK_EXT_full_screen_exclusive,
   VK_EXT_hdr_metadata, etc.).

The engine is in a safe state: when the user toggles `fg_enabled`,
FG context creates + dispatches (visibly burning GPU time), the
swapchain proxy fails to create, recovery rebuilds a plain swapchain,
and rendering continues. `fsr3_swapchain_fatal_` sticks for the
process lifetime so we don't retry. Generated frames sit in the FG
output image and go nowhere.

Sessions 5-6 (UI mask, present pacing, tuning) cannot start until
session 4's SwapChain context creation succeeds.

## Session 1 — FSR 3 SDK vendoring + build
- `git submodule add` AMD's FSR 3 repo at `external/FidelityFX-FSR3`.
  Pin to a known-good tag (latest stable, currently v3.1.x).
- CMake: `add_subdirectory(...)` with `FFX_FSR3_API_VK=ON` + DX12 off.
  Mirror the FSR 2 wiring in our top-level `CMakeLists.txt`.
- Add `QLIKE_HAVE_FSR3=1` define gated on submodule presence.
- **No code calls FSR 3 yet** — just verify it links.
- Build, run, no behaviour change.

## Session 2 — Swap FSR 2 upscaler for FSR 3 upscaler
- `init_fsr2 / dispatch_fsr2 / destroy_fsr2` → `init_fsr3_upscaler`
  etc. The API is similar but not identical:
  - `FfxFsr3UpscalerContextDescription` + `ffxFsr3UpscalerContextCreate`
  - `FfxFsr3UpscalerDispatchDescription` + `ffxFsr3UpscalerContextDispatch`
  - Same inputs (color, depth, motion, jitter, exposure)
- Compose still reads `fsr3_upscaler_output_image_` like before.
- Toggle `rt_.fsr2_enabled` becomes `rt_.fsr_upscale_enabled` (or keep
  the old name as alias to avoid resetting users' settings).
- **No frame gen yet** — same FPS as today, just FSR 3's slightly
  better upscale quality.

## Session 3 — Frame Generation context + interpolation pass
- Allocate `fsr3_fg_output_image_` for the synthesised frame.
- New `ffxFsr3FrameGenerationContext` alongside the upscaler context.
  Lifecycle hooks mirror FSR 2's recreate-on-swapchain-resize.
- Per-frame: after upscale dispatch, call
  `ffxFsr3ContextDispatchFrameGeneration` with:
  - `commandList` (probably async compute — FSR 3 prefers it)
  - `presentColor` = upscaler output (the "real" frame)
  - `motionVectors` = our LR motion image
  - `depth` = our LR depth
  - `flags` = `FFX_FRAMEINTERPOLATION_FLAG_*`
- The dispatch runs FSR 3's internal optical-flow compute pass + an
  interpolation pass that produces the in-between frame.
- **Generated frame not yet presented** — render it to the FG output
  image; verify no validation errors. Visible state: same FPS, FG
  output exists in memory but nobody samples it.

## Session 4 — Swapchain proxy + present pacing
This is the hard one. FSR 3 needs to control the present queue so it
can interleave generated frames with rendered frames at the right
cadence.

Two integration paths AMD supports:
- **Proxied swapchain**: FSR 3 wraps `VkSwapchainKHR`. The app calls
  `ffxFsr3PresentVK` instead of `vkQueuePresentKHR`. FSR 3 internally
  alternates real vs generated frames, paces them via timing queries
  on the present timeline. Cleanest integration but FSR 3 owns the
  swapchain — affects image acquisition, format, count.
- **Manual pacing**: app keeps swapchain ownership, but must implement
  the timing logic itself. Riskier, more code.

Recommend the proxied path. Tasks:
- Refactor `recreate_swapchain` so the swapchain handle goes through
  `ffxFsr3SwapchainGetSwapchainVK` (FSR 3 returns its own VkSwapchainKHR).
- Replace `vkQueuePresentKHR` with `ffxFsr3PresentVK` (or the swapchain
  proxy auto-intercepts).
- Add a UI toggle `rt_.fg_enabled` that turns FG on/off without recreating
  the context (FSR 3 supports a runtime enable flag in its dispatch desc).
- Verify FPS doubles when on.

## Session 5 — UI composition + HUD mask
**Without this session, ImGui will smear visibly on every other frame**.
Generated frames interpolate between two real frames; ImGui elements
that change per frame (FPS counter, frame time graph, slider drags)
look broken.

Plan:
- Render UI to a SEPARATE image (not the swapchain). UI gets composited
  on top of FSR 3's output AFTER frame generation.
- Pass a UI alpha mask to FG dispatch so the optical-flow stage knows
  to skip pixels covered by UI.
- Or: render UI twice — once at "current frame" timing, once at the
  generated frame's interpolated timestamp — but most engines just
  freeze UI to the current real frame and accept slight UI judder.
- Cursor + crosshair drawn after FG, never interpolated.

## Session 6 — Tuning + edge cases
- **Auto-exposure pinning**: our `auto_exposure_strength` ramps each
  frame. On generated frames there's no scene to read luminance from —
  FG just blends two existing tonemap results. Lock exposure between
  the real-frame pair so the interpolation result matches.
- **Low-FPS quality threshold**: AMD recommends FG only when base FPS
  ≥ 60 (interpolation artifacts on fast geometry below that).
  Auto-disable FG if base FPS drops below threshold (UI checkbox can
  override).
- **VSync interaction**: FG fundamentally changes present cadence.
  Test all combinations: VSync off / on / mailbox.
- **Quality presets**: Low/Medium → FG off (low base FPS),
  High/Ultra → FG on by default.

## Architectural notes
- **Async compute queue.** FSR 3 strongly prefers running optical flow
  + interpolation on a separate compute queue. We already have one
  (`compute_queue_`) for TLAS builds; would need to time-share or add
  a second pool. Worst case: run sync on graphics queue, accept ~10-20%
  cost overhead.
- **FSR 3 owns the present chain.** Once the proxy ships, anything
  that reads the swapchain image directly (screenshots? debug capture?)
  needs to go through FSR 3's accessor instead.
- **Submodule + license.** Same MIT terms as FSR 2; preserve
  `LICENSE.txt`. Keep both submodules — sessions 1-2 keep FSR 2 around
  in case we need to roll back.

## Known gotchas
- **HUD smear is the #1 visible bug if session 5 is skipped.** Don't
  ship sessions 3-4 without committing to session 5.
- **Frame pacing artifacts** if optical-flow + interpolation latency
  exceeds frame budget. Visible as judder. Counter-intuitive: FG can
  make 30 fps look WORSE than 30 fps because of the cadence break.
- **Mouse latency** doesn't decrease with FG (you still see real
  frames at the original cadence; generated frames are just for
  smoothness). Players may complain "felt sluggish at 120 fps" — this
  is expected; FG ≠ Reflex / Anti-Lag.
- **HDR + tone-mapping**: our compose runs ACES tonemap, then ImGui.
  FSR 3 expects post-tonemap LDR for FG. Stack should be compatible
  but verify when session 3 ships.

## Suggested ordering / stop points
- Stop after session 2 if you only want FSR 3's slightly-better upscale.
- Stop after session 4 if you want the FPS doubling and accept HUD
  artefacts on generated frames.
- Ship through session 6 for a production-quality FG path.

## License notes
- AMD FidelityFX-FSR3 is MIT (`external/FidelityFX-FSR3/LICENSE.txt`).
- Vendored as git submodule; no source copied into our tree.
- Distribution-ready as long as `LICENSE.txt` is preserved.
