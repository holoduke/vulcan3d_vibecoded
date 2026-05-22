# SVGF GI Denoiser — Multi-Session Plan

Add a Spatiotemporal Variance-Guided Filtering pass for the GI signal so
1-sample-per-pixel GI (ReSTIR on, or `gi_samples=1`) converges to the
same visual quality as 32+ uniform samples — without the inline 3-tap
spatial gather that caused the 3.7× regression in ReSTIR session 4.

Reference: Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering:
Real-Time Reconstruction for Path-Traced Global Illumination."

This is the deferred `4a-svgf-deep` from the perf roadmap. Real
multi-session work: it touches the render-graph, adds a new MRT slot,
a new descriptor set, two new fragment passes, and a cube.frag
composite change. Done correctly it makes ReSTIR's quartered-sample
mode visually acceptable as a default; done wrong it bakes in stale GI
or ghosts at moving silhouettes.

## Why now

- Inline per-fragment spatial reuse was tried in ReSTIR session 4 and
  killed: 3 divergent 48-B SSBO loads per GI pixel cost more than the
  3/4 GI rays saved. A separate cheap compute/fragment pass that runs
  at half-res with coalesced reads is the canonical fix.
- ReSTIR temporal alone leaves visible flicker on moving silhouettes
  and reservoir-disocclusion frames; SVGF's variance-guided à-trous
  is the standard cure.
- Compose-stage TAA gives a stable final image but blurs the *whole
  composed colour*. The GI signal is already albedo-multiplied + mixed
  with direct + AO by then, so TAA can't apply GI-aware weights.

## Architectural choice

The GI signal needs to be written out separately from cube.frag's
main colour, denoised, then composited back. Existing TAA runs on the
final composite — leave it alone. SVGF runs upstream on the
unmodulated irradiance.

**Storage image, not MRT.** Original plan was a third colour
attachment, but every pipeline that draws into the main color pass
(cube, terrain, terrain_tess, terrain_raymarch, water, grass) would
then need its fragment shader to write something to that location and
its pipeline config to declare 3 attachments. Massive shotgun
change. A storage image bound at scene_desc and written via
`imageStore` from cube.frag alone gets the same data with zero
touches to the other shaders. Format `R16G16B16A16_SFLOAT` (storage
+ sampled + colour, universally supported as a storage image).

```
depth pre-pass
  └─→ half-rate shadow (optional)
      └─→ MAIN COLOR PASS (cube.frag) — unchanged MRT (2 attachments)
            PLUS imageStore to gi_irradiance_image_ at the point in
            main() right before gi_indirect is multiplied by albedo.
                ↓
          SVGF temporal: accumulate gi_irradiance + luminance moments
          (μ, μ²) into history, reproject via motion_vec.
                ↓
          SVGF variance: per-pixel variance from moments, 3×3
          Gaussian on M<4 pixels to prevent black spots.
                ↓
          SVGF à-trous: 3 passes at stride 1, 2, 4 with edge-stop
          weights (normal, depth, luminance-vs-variance).
                ↓
          COMPOSITE (cube.frag pass 2, or a fullscreen pass):
              final_gi = filtered_gi * albedo * gi_strength
              hdr     += final_gi
                ↓
          TAA → upscale → bloom → compose
```

## Storage budget

At 1280×720 internal:
- `gi_irradiance` (R11G11B10F): 1280×720 × 4 B = ~3.7 MB
- `gi_geom` (R32F depth + RGBA8 oct-encoded normal): could pack into
  R32G32 = 8 B/px = ~7.4 MB
- `gi_history` (R11G11B10F + R16G16F moments): ~7.4 MB
- `gi_moments_prev` history slot: ~3.7 MB
- Total: ~22 MB extra. Fine.

History images use the same ping-pong / FIF strategy as TAA's history.

## Phase plan

### Session 1 — MRT plumbing
Goal: cube.frag emits raw GI as a third colour attachment; existing
behaviour unchanged (the new attachment is allocated and written but
ignored downstream).

- Allocate `gi_irradiance_image_` (R11G11B10F at render_extent) +
  `gi_geom_image_` (R32G32_UINT for packed depth+normal).
- Extend the main render pass to 3 colour attachments. GraphicsPipelineConfig
  already supports `color_formats` vector — add to cube + terrain pipelines.
- cube.frag: write `out_gi = gi_indirect_pre_albedo;` (capture the value
  BEFORE the `* albedo * scene.rt_params2.x` multiply), `out_geom = pack(depth, N);`.
- Lifecycle: `init_svgf_targets()` / `destroy_svgf_targets()`, hook
  into `recreate_swapchain`.
- Verify: render-graph still passes, vk_err=0, no visible change.

### Session 2 — Temporal accumulation
Goal: per-pixel running mean + luminance second moment via motion-vec
reprojection. Without spatial filter yet.

- New fragment pass `svgf_temporal.frag` reading:
  - `gi_irradiance` (current frame, raw)
  - `gi_history` (previous frame's filtered irradiance)
  - `gi_moments_prev` (previous frame's [μ, μ², M])
  - `motion_vec`, `gi_geom` for disocclusion
- Output: new `gi_history` (clamped, accumulated) + new `gi_moments` (M-incremented).
- Disocclusion: reject prev when |prev_depth − cur_depth| > τ_d OR
  dot(prev_N, cur_N) < cos(20°) OR prev_uv off-screen → M = 1.
- Bind both history images per-FIF (mirror TAA's `history_image_[2]`).
- Cube.frag composite: read `gi_history` instead of `gi_irradiance`
  for shading. Still no spatial filter; result should look identical
  to inline GI for static scenes, ghost mildly on movement (no filter
  to hide it yet — that's session 3).
- Visible delta: dramatically reduced GI noise on static scenes;
  expected slight ghosting until session 3.

### Session 3 — Variance + 3-pass à-trous
Goal: the actual denoise — turns the noisy temporal-only result into
the SVGF look.

- Variance pass: read moments, compute σ² = max(0, μ² − μ²),
  3×3 Gaussian on σ² for pixels with M < 4 (low-confidence
  estimate → spatial reconstruction).
- 3 à-trous passes, stride 1, 2, 4. Per-tap weight:
  - `w_n = max(0, dot(N_p, N_q))^σ_n` (normal, σ_n=128)
  - `w_d = exp(-|d_p − d_q| / (σ_d * |∇d · Δp| + ε))` (depth)
  - `w_l = exp(-|L_p − L_q| / (σ_l * √variance + ε))` (luminance)
  - tap weight = `w_n * w_d * w_l * b3_spline_kernel`
- Variance also à-trous-filtered (squared weights, halved σ_l).
- Output last à-trous's color as the filtered GI.
- Cube.frag composite reads the post-filter GI.
- Visible delta: clean diffuse GI, soft penumbras, no banding.
  Some over-blur on contact shadows (needs session 4 tuning).

### Session 4 — Edge-stop tuning + reservoir integration
Goal: dial the weights for our scene's brick/terrain mix; integrate
with ReSTIR (SVGF runs AFTER ReSTIR's temporal reservoir reuse).

- Tune σ_n / σ_d / σ_l per visual A/B (autodemo + interior castle).
- Add UI sliders behind `rt_.svgf_strength` (0 disables, falls back
  to current path).
- ReSTIR + SVGF stack: ReSTIR handles sample importance; SVGF handles
  noise from the surviving samples. Removing inline GI spatial taps
  is permanent (session 4 already did that).
- Visible delta: comparable to 64-sample uniform GI at ~2 rays/pixel.

### Session 5 — Tuning + edge cases
- Bright-pixel disocclusion (fireflies through history).
- Reservoir reset on big camera moves.
- Quality preset wiring: Low = SVGF off (saves the ~1.5 ms denoise
  cost), Med/High/Ultra = SVGF on.
- Document final perf numbers (expected: −0.5 to +1.5 ms net vs
  current ReSTIR-on, with dramatically better convergence).

## Estimated perf budget

At 1280×720 internal, SVGF cost when on:
- Temporal pass: ~0.4 ms (1 fragment per pixel, 3-4 texture reads)
- Variance pass: ~0.3 ms (3×3 sum)
- 3× à-trous passes: ~1.0–1.5 ms total (B3 spline + edge stops)
- Composite: free (folded into cube.frag's existing colour pass)
- **Total**: ~1.7–2.2 ms

Offset: GI ray count can drop from 6 → 1 per pixel without visible
quality loss (the whole point). That saves 5× the BLAS traversal cost
on GI-bound pixels (castle interior). Net: SVGF should be perf-positive
in GI-heavy scenes, perf-negative on open terrain (where there's no
GI to denoise).

Gate behind preset accordingly.

## Risks

1. **Ghosting at moving silhouettes** — disocclusion thresholds too
   loose. Mitigation: per-session A/B + the autodemo orbit's smooth
   motion is a friendly stress case.
2. **MRT bandwidth** — adding 2 attachments to the main pass costs
   write bandwidth on every cube/terrain fragment regardless of
   whether SVGF is enabled. Mitigation: lazy — only output when
   `rt_.svgf_enabled`. If that requires runtime pipeline switching,
   accept the unconditional MRT cost (the writes are coherent and
   the pipeline cache rebuild on toggle is worse).
3. **Specular reflection contamination** — SVGF assumes diffuse GI;
   reflective surfaces (water, polished metal) should bypass the
   filter. Mitigation: only filter the diffuse irradiance term;
   leave the reflection pass alone (it's a separate ray bundle
   anyway).
4. **The "8/10 right" trap** — SVGF often looks fine in motion but
   reveals over-blur on stationary close-up surfaces. Mitigation:
   take stationary screenshots at each session for A/B.

## Out of scope

- ReSTIR DI (direct illumination via reservoirs). Our direct sun
  lighting is already crisp via the half-rate shadow pass + PCSS.
- Glossy reflection denoising. Different filter (BRDF-aware), not
  classic SVGF.
- Sky-vis denoising. Already done via the cheap probe in cube.frag
  for both terrain and brushes.
