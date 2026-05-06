# Terrain renderer — phased plan

The arena's flat-floor + brick-wall perimeter is being replaced with a
procedurally-generated heightmap terrain. Target world size: 5 km × 5 km
with a flat plateau hosting the castle. AAA references: GTA's TerrainGen,
Fortnite's Volumetric Heightmap, UE's Landscape, Frostbite's terrain.

## Architecture (target end state)

```
        +---------------------+
        |  HeightmapSource    |   FastNoiseLite + edits
        |  (in-memory grid +  |   (Phase 1: single 1km² grid;
        |   patch overlays)   |    Phase 2: tile manager / disk LRU)
        +---------+-----------+
                  |
        +---------v-----------+
        |  ChunkBuilder       |   per-chunk vertex/index + Jolt
        |  (worker thread)    |   HeightFieldShape on physics thread
        +---------+-----------+
                  |
        +---------v-----------+
        |  TerrainRenderer    |   per-frame: pick visible chunks (quad-
        |  + LOD selector     |   tree), bind chunk mesh, draw. Maintains
        |                     |   one merged BLAS per visible LOD ring.
        +---------------------+
```

## Phases

### Phase 1 — heightmap terrain replaces the arena floor

Scope: a single non-streaming terrain region (~1km²), procedural
heightmap, Jolt heightfield collision, RT-visible (BLAS in TLAS so
shadows / GI / AO work). Remove the arena perimeter walls + flat floor
brush + outer lanterns.

Steps:
1. Add `FastNoiseLite` (single header, MIT) via FetchContent.
2. New `src/engine/terrain.{h,cpp}`:
   - `Heightmap generate_heightmap(seed, dim, cell_size, ...)` — fBM with
     plateau-flatten in a configurable rectangle (so the castle can sit
     on level ground).
   - `Mesh build_terrain_mesh(const Heightmap&)` — vertex grid with
     proper per-vertex normals.
3. `PhysicsWorld::add_heightfield_terrain(...)` — wrap Jolt's
   `HeightFieldShape` (cheaper + correct collision vs convex boxes).
4. `level.cpp`: drop the ground brush, the 4 perimeter walls, the 4 outer
   lanterns. Keep the castle and inner geometry.
5. `world.cpp init_world`: build heightmap, upload mesh, register
   collision, expose the chosen castle anchor point so the castle is
   placed on the flat region.
6. `rt.cpp`: build a third BLAS (terrain), add a TLAS instance for it,
   add a materials slot.
7. Render: existing pipeline + new draw call in `render_world` (binds
   terrain mesh and pushes a constant material).

Out of scope for Phase 1: streaming, LOD, slope-blended texturing,
sculpting.

### Phase 2 — streaming + clipmap LOD

- Split the world into 64m × 64m chunks (8192 chunks for 5km²).
- Quad-tree LOD: full-res near the player, coarser further out.
- Async chunk gen on the existing worker thread; cap concurrent in-flight
  builds.
- Skirts (vertical drape strips) at LOD seams to hide T-junctions.
- One BLAS per *visible* chunk; cap visible chunks so the TLAS doesn't
  blow past current 768 limit. Probably ~100 chunks visible.
- Persistent on-disk heightmap (save edits to a tile file).

### Phase 3 — terrain shading

- New `terrain.frag`: triplanar albedo + normal sampling for 4 layers
  (rock, dirt, grass, snow).
- Layer masks blended by slope + height + per-vertex tint.
- Distance detail texture overlaid up to ~50m, fades out.
- Optional: tessellation for displacement near the player.

### Phase 4 — in-game sculpt

- Raycast cursor onto terrain → world-space brush center.
- Brush ops: raise, lower, smooth, paint (layer mask).
- Stroke commit:
  - patch the in-memory heightmap region;
  - rebuild affected chunks' meshes;
  - refresh affected BLASes;
  - rebuild the affected Jolt heightfield region.
- Save edits to `world.terrain` next to the exe so re-launches preserve them.

## Library choices

- **FastNoiseLite** (single header, MIT): noise generation. Battle-tested,
  used by many indie/AAA projects.
- **Jolt HeightFieldShape**: built-in heightfield collision, already
  available in the Jolt build.
- No third-party "terrain engine" libs — most are tied to Unity/Unreal or
  scoped wider than we need (e.g. World Machine, Gaia). Custom is the
  practical path.

## Open questions for later phases

- Virtual texturing for ground albedo when world > 1km² (lots of unique
  detail is required for it to look good without obvious tiling).
- GPU-driven indirect terrain draw (compute culling + indirect dispatch).
- Foliage scatter (grass/rock instances).
- Water plane and shoreline blending.
- Time of day shifts on terrain shading.

This doc is tracked under `docs/` and should be updated as phases land.
