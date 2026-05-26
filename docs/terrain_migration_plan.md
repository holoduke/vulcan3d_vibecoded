# Terrain migration: tessellation -> fixed CDLOD

**Status:** plan only. Existing tess path (terrain_tess.{vert,tesc,tese})
stays live until each phase below ships and is verified.

**Why:** the current tessellation path mixes a static CDLOD chunk mesh
(LOD0 + LOD1 + morph) with a runtime tess pipeline that subdivides near
chunks 1..64x for the rocky-grass vertex displacement. That gives
unpredictable per-frame cost, requires a tess-density-vs-displacement-
frequency match (low tess + high amp = spikes), and the tess stages
(VS->TCS->TES->FS) add pipeline complexity. For a deterministic
heightmap-driven world this is overkill -- a fixed-mesh CDLOD with 4-6
LOD levels gives stable cost, simpler shaders, and a mesh that is
BLAS-ready for RT without rebuilds.

## End state

- 6 LOD levels (LOD0 highest, LOD5 lowest) per chunk
- LOD0 vert spacing 0.5 m (vs current ~1-2 m). Each chunk ~32x32 m =
  64x64 quads + skirts = ~8.5k verts at LOD0
- LOD1: 1 m verts (~16x16 quads), LOD2: 2 m, LOD3: 4 m, LOD4: 8 m,
  LOD5: 16 m. Each LOD = previous / 4 verts
- Per-LOD vertex y pre-baked from current heightmap + the rocky-grass
  displacement at displacement amplitude (currently 0.40 m)
- CD-LOD morphing between adjacent LODs (existing CD-LOD shader stays
  but now applied across all 6 levels, not just 0<->1)
- Vertex skirts (~1 m drop) on every chunk edge to hide LOD-boundary
  T-junctions without per-pair edge matching
- Tess pipeline + all .tesc/.tese files deleted; terrain.vert/.frag
  is the only terrain path

## Phases

### Phase 1: bake the LOD0 dense mesh

- New function `bake_chunk_lod(chunk_xz, lod_level, ...)` in
  `src/engine/terrain.cpp` (or a new `terrain_lod.cpp`)
- For lod=0, produce a (64+2 skirt) x (64+2) grid of vertices at
  0.5 m spacing. Sample `terrain_heights_` (existing baked heightmap)
  + the rocky displacement texture (CPU-side load of the same PNG the
  GPU samples in terrain_tess.tese)
- Skirt verts at chunk edges drop their y by 1 m so neighbour LOD
  mismatches don't punch sky holes
- Index buffer is regular grid + skirt fan; ~12k triangles per LOD0
  chunk
- Verify visually by drawing ONE chunk with the new mesh, tess disabled

### Phase 2: bake LODs 1-5

- Reuse bake_chunk_lod with halved resolution per level
- Each level: ~25% of the previous level's vertices
- Total memory per chunk across all 6 LODs ~= LOD0 * 1.33 (geometric
  series), so ~17k verts and ~11k tris per chunk in all LODs combined
- World is currently ~64x64 chunks. Total terrain verts ~= 64 * 64 *
  17k = 70M verts -- this is the only concern. Mitigate by streaming
  LOD0/1 within 200 m of camera and LOD3-5 always; LOD2 within 500 m

### Phase 3: streaming + draw

- Per-frame: for each visible chunk, pick LOD = distance-bucket
  (already used by CDLOD code)
- Two adjacent LODs may be picked for a chunk if it straddles the
  morph window; pass morph factor in pc.color.w (same as today)
- Render via the existing `terrain_pipeline_` (no tess). The same
  cube.frag is_terrain branch consumes the output -- vert/frag
  interface unchanged
- This is a drop-in replacement for the tess pipeline draw at
  world.cpp ~line 3650, gated by an rt_.use_tessellated_terrain knob
  for the cross-over period

### Phase 4: delete tess code

- After 1-2 sessions of side-by-side use: remove terrain_tess_*.glsl,
  the tess shader modules, terrain_tess_pipeline_ /
  terrain_tess_depth_pipeline_ / terrain_tess_wire_pipeline_, the
  related setup.cpp + descriptors.cpp + world.cpp branches, the
  tessellationShader device feature request, and the UI sliders for
  tess max_level / near_m / far_m / falloff
- Disp amp / disp smooth / POM far sliders stay (they drive cube.frag
  SPOM, not the geometry)

### Phase 5: rebuild BLAS from LOD0/1 directly

- Today the BLAS uses the base CDLOD mesh; with denser LOD0 the BLAS
  has higher fidelity and the cube.frag GI ray-query terrain hits
  agree with the visible surface much more precisely
- One-time bake at chunk-stream-in; no per-frame BLAS rebuild

## Tradeoffs / risks

- Memory: 70M terrain verts at 32 bytes each = 2.2 GB. Mitigate via
  per-chunk LOD streaming (typical scene shows ~30 chunks, ~80 MB)
- Bake time: building all 6 LODs for a chunk is ~few-ms CPU work; do
  on background thread (the existing chunk-streaming worker already
  has the harness)
- The rocky-grass displacement is now baked, not live: changing the
  rocky height texture or the disp amp slider needs a chunk re-bake.
  Solution: re-bake on the worker thread when the rt_ slider crosses
  a threshold; or expose two modes ("baked" / "dynamic, current
  tess path") until the user is happy

## Open questions (worth answering before Phase 1)

- Do we want the displacement amplitude to remain a runtime slider, or
  bake at fixed 0.40 m? Runtime adds re-bake complexity
- Skirts vs explicit T-junction edge matching: skirts are simpler but
  show as faint vertical strips at LOD boundaries on shallow terrain
  (worst-case 1 m drop visible). Edge matching is exact but needs
  per-pair index buffers
- Should LOD0 be 0.5 m or 0.25 m verts? 0.25 m gives finer rocky
  silhouettes but 4x the vert count
