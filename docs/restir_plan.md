# ReSTIR GI — Multi-Session Plan

Replace the current uniform Monte-Carlo GI in `cube.frag` with
reservoir-resampled importance sampling (ReSTIR, Bitterli et al. 2020).
Today's GI fires `gi_samples` (default 64, Ultra 96) cosine-weighted
hemisphere rays per pixel and averages — most contribute little.
ReSTIR's reservoirs reuse a single carefully-chosen sample across
frames (temporal) and across neighbours (spatial), converging to
~64-sample uniform quality at 1–2 rays per pixel.

This is a multi-session item. Pre-existing `task #83`. Each session
ships a discrete chunk of infrastructure or behaviour change.

## Session 1 (shipped)
- `rt_.gi_restir_enabled` flag + UI toggle + persisted setting.
- When on, `update_scene_ubo` divides `gi_samples` by 4 before handing
  it to the shader. cube.frag's loop is unchanged. The TAA + spatial
  filter folds the noise back into a stable image — visible 4× perf
  win on the GI loop without any algorithm change.
- Purpose: give the user a dial today + an honest baseline to compare
  the proper reservoir pipeline against in later sessions.

## Session 2 — Reservoir SSBO foundation
- New module `src/engine/vk_engine/restir.cpp`. Mirrors taa.cpp style.
- Two ping-pong SSBOs sized at `render_extent_.w * h * sizeof(Reservoir)`
  with `Reservoir` = `{ vec3 sample_dir, vec3 radiance, float W,
  float w_sum, uint M, uint pad }` = 32 B. ~28 MB at 1280×720.
- Bound at `scene_desc` bindings 15 (read-prev) and 16 (write-cur).
- Pool size update + `write_scene_descriptors_once` signature growth.
- cube.frag writes a 1-sample reservoir per fragment via `imageStore`-
  style indexed SSBO write. Doesn't yet read history — that's session 3.
- Net visible delta: none yet. Foundation only.

## STATUS — session 5 shipped, ReSTIR re-enabled (2026-05-19)

Sessions 3+4 were quarantined (2026-05-13) after shipping visible
wobble. Diagnosed causes:
1. **Single-buffer aliasing race.** Both scene_desc bindings 15/16
   aliased `reservoir_buf_[0]`. Same-frame writes propagated to other
   fragments' spatial-tap reads → fragment-order radiance bleed.
2. **Normal-only disocclusion.** The accept test was `dot(prev_N,
   cur_N) > restir_params.z`, and `.z` was actually the FSR jitter
   delta (~0.0005), so it accepted essentially any same-hemisphere
   normal — pulling GI from different surfaces at the same
   orientation (parallel walls, separate floor patches).

### Session 5 fix (shipped) — ring buffer, no descriptor surgery

Rather than the originally-planned FIF descriptor sets / descriptor-
indexing (sizable Vulkan-lifetime risk), session 5 keeps the single
aliased buffer and makes it a **3-region ring**:

- `restir.cpp`: reservoir SSBO sized ×3 (`kReservoirRing = 3`),
  126 MB at 1280×720. Bindings 15/16 still alias it.
- `cube.frag`: write region `frame%3`, read region `(frame+2)%3`,
  both derived from `scene.rt_flags.w`. A spatial tap never sees the
  current frame's partial writes (kills race #1). Race-free under
  `kFrameOverlap==2`: frame F's write region was last touched by
  F-3 and read by F-2, both retired by the F-2 fence the CPU waits
  on before recording F. (2 regions would NOT be safe.)
- **Depth-aware disocclusion** (kills race #2): the writing surface's
  `cam_dist` is packed into the reservoir's free `pad` slot
  (`floatBitsToUint`). Temporal accept requires BOTH
  `dot(prev_N, N) > 0.92` (~23°, a shader constant — never needed a
  UBO slot) AND `|prev_cam_dist - cam_dist| < 0.04·dist + 0.20 m`.
- **Spatial 3-tap REMOVED** (session-4 path deleted). Measured A/B
  (identical autodemo path): off 4.66 ms median / on-with-spatial
  17.1 ms (32 ms mean, 69 ms in the castle) / on-temporal-only
  4.71 ms. The inline per-fragment spatial gather (3 divergent 48-B
  SSBO loads/GI pixel) cost ~4× more than the ¼-sample ray saving it
  was meant to converge. Temporal-only ReSTIR is perf-neutral vs off
  and still recovers the quality. Any future spatial reuse MUST be a
  separate cheap compute pass, never an inline fragment gather.
- **Re-enable**: `restir_params.x` is driven by `rt_.gi_restir_enabled`
  again (was `fsr_active` — verified `.x` has no other shader
  consumer, so this is independent of FSR; fixed a latent bug where
  ReSTIR silently ran whenever FSR was on with a garbage threshold).
  `.zw` still carries the FSR jitter delta, untouched.

No UBO layout change, no new binding, no descriptor/device-feature
change. The session-1 `gi_samples / 4` reduction stays in effect when
ReSTIR is on (reservoir reuse converges it back to full quality).

## Session 3 — Temporal reuse
- Per-frame: read previous-frame reservoir at the motion-vec-reprojected
  pixel. Combine via WRS (weighted reservoir sampling):
  - `w_cur  = M_cur * target_pdf_cur(R_cur.sample) * R_cur.W`
  - `w_prev = M_prev * target_pdf_cur(R_prev.sample) * R_prev.W`
  - Pick R_cur with prob `w_cur / (w_cur + w_prev)`, else R_prev.
  - `M = min(M_cur + M_prev, M_max)` (M_max ~32–64; bounds variance).
- Disocclusion: reject prev when `|prev_depth - cur_depth| > τ` or
  `dot(prev_normal, cur_normal) < cos(20°)`. Reset M to current.
- Final shading: `gi = target_pdf(sample) * W`. Cap at `vec3(6.0)` like
  the current loop's firefly clamp.
- Net visible delta: GI noise drops dramatically as M grows; static
  scenes converge to near-perfect quality at 1 ray/pixel.

## Session 4 — Spatial reuse
- Separate compute pass (or fragment pass) that runs after temporal
  but before shading. For each pixel, picks 3 neighbour reservoirs
  (Halton-sampled within ~30 px radius) and merges them into the
  current pixel's reservoir using the same WRS formula.
- Edge-stopping: skip neighbours whose normal/depth disagrees too much.
- Net visible delta: noise drops further on disocclusion frames + edges
  where temporal alone hasn't yet converged.

## Session 5 — Tuning + edge cases
- M_cap selection per quality preset.
- Firefly clamps that don't crush bright legitimate samples.
- "Reset on big camera move" so cameras teleporting don't carry stale
  reservoirs.
- Quality preset wiring: Low/Med use 1 ray + temporal-only; High/Ultra
  use 2 rays + temporal + spatial.

## Architectural notes
- Our GI lives in `cube.frag` as inline ray tracing (RT-in-fragment).
  Most reference ReSTIR implementations are compute-pass deferred.
  Keeping fragment-pass works because the SSBO write index is the
  pixel coord (gl_FragCoord) and the depth pre-pass guarantees one
  fragment per pixel for opaque draws.
- Particles / projectiles / viewmodel skip the depth prepass and
  could double-write the reservoir. Plan: gate the writeback on a
  push-constant flag set per draw call.
- Sky_vis (the ambient-attenuating sky-miss-fraction) is currently
  derived from the GI rays. With 1 ray/pixel that signal becomes
  noisy — track it in the reservoir's `w_sum` channel or compute it
  separately. Lower priority — dark interiors will look slightly
  noisier until session 5.
