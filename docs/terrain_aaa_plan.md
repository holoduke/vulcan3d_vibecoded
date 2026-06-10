# Terrain AAA-quality plan

Goal: take the terrain from "procedural heightfield with a colour splat"
to the visual class of Death Stranding / Horizon / RDR2. The geometry
pipeline (quadtree CDLOD, vertex-free, 0.25 m near triangles — commit
8775057) is done and is NOT the bottleneck anymore. What makes terrain
read as AAA, in order of visual impact:

1. Material quality + blending (Phase 1)
2. Near-field micro-detail: displacement + parallax everywhere (Phase 2)
3. Ground clutter — rocks, pebbles, debris meshes (Phase 3)
4. Heightfield enrichment — erosion, curvature, flow (Phase 4)
5. Terrain-specific lighting — AO bakes, micro-shadowing (Phase 5)
6. Atmosphere & distance readability (Phase 6)

Each phase is independently shippable and visually testable with
`--scene 5` / meadow captures. Estimated effort assumes current
single-TU style (cube.frag terrain branch + world.cpp).

---

## Phase 1 — Material overhaul (biggest single win)

Today: 5 procedurally-baked 512² layers (rock/grass/dirt/sand/snow)
linearly blended by height + slope, one detail FBM on top. 512² baked
noise reads as "indie procedural" no matter what geometry is under it.

Tasks:
1. Replace the 5 baked layers with real photogrammetry PBR sets at 2K
   (albedo + normal + roughness + height + AO). ambientCG-style sets,
   2-3 per biome layer (e.g. rock: cliff granite + scree; grass: lawn +
   dry patches; dirt: cracked + loamy). Loaded into one texture array.
   The asset pipeline (texture.cpp array upload, mips) already exists.
2. Height-aware blending: replace `mix(a, b, w)` with height-lerp
   (blend by `w + (heightA - heightB)` sharpened) so grass grows BETWEEN
   stones and dirt fills rock cracks instead of cross-fading. This one
   change is the most recognisable "AAA splat" feature.
3. Per-layer UV scale + per-layer triplanar gate (cliff layers triplanar,
   flat layers planar) instead of the single global tile size.
4. Macro variation: one 4K low-frequency albedo-tint + roughness map
   over the whole 2048 m world (can be baked from the heightmap:
   altitude/moisture noise) multiplied over every layer. Kills the
   "same hue everywhere" look at middle distance.
5. Slope/curvature-driven layer rules (needs curvature from Phase 4 or
   a cheap screen-space approximation initially): exposed rock on convex
   ridges, sediment in concave gullies.

Files: cube.frag (is_terrain branch rewrite into a small layered-splat
function), world.cpp (texture loading), new assets/. Effort: 2-3 sessions.
Acceptance: meadow + mountain captures look photographic at 10-200 m.

## Phase 2 — Near-field micro detail

Today: one rocky height map displaces vertices near camera
(terrain_disp_amp) and a single-layer POM on rocky pixels.

Tasks:
1. Per-layer displacement: VS samples the ACTIVE layer's height texture
   (from Phase 1 array) instead of the single rocky map; amplitude per
   layer (sand dunes ≠ rock relief ≠ grass sod).
2. POM for every layer within ~25 m (cube.frag already has the SPOM
   marcher — generalise the terrain branch to use the layer array).
3. Micro-normal detail layer (high-frequency tiling normal) blended in
   the last 10 m so the ground never goes flat at boot-height distance.
4. Contact decals support (footsteps/impacts already have decals on
   brushes — enable on terrain).

Files: terrain.vert, cube.frag. Effort: 1-2 sessions.
Acceptance: kneel-down screenshot is crisp; silhouettes bumpy at 5 m.

## Phase 3 — Ground clutter scatter (the AAA "secret")

AAA terrain is never bare heightfield: 60-80 % of near pixels are
covered by instanced meshes. We already have the machinery pattern
(grass instancing + CLOD instance buffers).

Tasks:
1. Author/import 6-10 small meshes: 3 rock sizes, pebble clusters,
   sticks, tufts, flowers (CC0 photogrammetry, gltf loader exists).
2. Scatter system: CPU bake like build_grass but rule-driven by the
   Phase-1 splat weights (pebbles on dirt, rocks near cliffs, tufts on
   grass). ~100-300 k instances, distance-culled like grass_distance.
3. Render as one instanced draw per mesh through cube.frag's standard
   path (they get RT shadows/GI for free); include in merged-static BLAS
   for grounded contact shadows.
4. Large hero rocks: hand-placed list (10-30) with collision.

Files: new src/engine/scatter.cpp/.h, world.cpp draw + BLAS hookup.
Effort: 2-3 sessions. Acceptance: no 5 m circle of bare ground anywhere
on a walk from castle to lake.

## Phase 4 — Heightfield enrichment

Today: FBM + derivative erosion at generation, 0.5 m cells.

Tasks:
1. Offline hydraulic erosion bake (droplet sim, ~200k iterations over
   the 4097² grid, one-time tool run, cache to disk next to the sculpt
   save). Produces: eroded heights + flow accumulation + sediment maps.
2. Curvature + cavity map baked from final heights (one pass) — feeds
   Phase 1 layer rules and Phase 5 cavity darkening.
3. Flow/wetness: darken + raise roughness along flow-accumulation lines
   (creek beds), feed sediment colour into the splat.

Files: new tools/erode.cpp or terrain.cpp bake path + 2 new R8 textures.
Effort: 2 sessions. Acceptance: mountainsides show drainage channels and
talus fans instead of pure noise.

## Phase 5 — Terrain lighting upgrades

Today: RT shadows/AO/GI per pixel (good!), but no baked macro AO, no
micro-shadowing, sky visibility approximated.

Tasks:
1. Horizon/AO bake from the heightmap (16-direction horizon scan, R8) —
   cheap macro occlusion for valleys; multiplied into ambient. The
   heightmap shadow baker (bake_heightmap_shadow) is the template.
2. Micro-shadowing from normal-map slope ("micro shadow" trick:
   shadow *= saturate(1 + dot(N_detail, L) * k)) — sells material relief
   at zero ray cost.
3. Far-field: cross-fade per-pixel RT shadow to the baked horizon shadow
   past ~200 m (removes the largest remaining far-noise source).
4. Specular/roughness response on wet/rock pixels (currently pure
   Lambert on terrain) — low sun glances off rock and creek beds.

Files: cube.frag, world.cpp bake. Effort: 1-2 sessions.

## Phase 6 — Atmosphere & distance readability

Tasks:
1. Aerial perspective: physically-inspired scattering ramp (existing
   distance_fog has the knobs; replace the flat colour with a
   sun-angle-dependent rayleigh/mie mix).
2. Height fog layers in valleys at dawn (terrain_raymarch fog exists for
   the raymarch path — port the cheap version to the mesh path compose).
3. Distance colour grading: desaturate + blue-shift far terrain.

Files: cube.frag / compose.frag. Effort: 1 session.

---

## Sequencing & dependencies

- Phase 1 first; 2 and 3 in either order after it (3 reuses 1's splat
  weights for scatter rules). 4 feeds 1's rules but a screen-space
  curvature approximation lets Phase 1 ship without it.
- 5 and 6 are independent polish, any time.
- Every phase: capture scenes 1-7 before/after + re-bake the SAD refs.

## Out of scope (deliberately)

- Virtual texturing / clipmap caches: not needed at 2 km world scale.
- Nanite-style GPU subdivision: CDLOD at 0.25 m near cells is below
  pixel size at 1080p beyond ~3 m; geometry is not the limiter.
- World streaming: single-region world by design.
