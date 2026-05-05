## Roadmap — multi-session items

Items that need their own session — too big for inline implementation in a single chat.

### Terrain renderer with smart streaming

**Goal.** Replace the flat 60×60 m floor with a real heightfield terrain that the castle sits on. Streaming: load tiles around the camera, drop tiles outside view-distance, paged from disk so the world can be much larger than VRAM/RAM.

**Scope review.** A correct, robust streaming terrain is genuinely 1–2 weeks of work. The pieces:

1. **Heightfield data format.** Either a baked binary blob of 16-bit heights at fixed grid spacing, or live-procedural via FBM noise. Bake-once is faster at runtime; procedural is infinite. Procedural + per-tile cache to disk gives the best of both.
2. **CDLOD / clipmap mesh.** Continuous-distance LOD: tile size constant in world space, but tessellation density falls off with camera distance. Standard implementations: Filip Strugar's CDLOD paper (2010), or terrain clipmaps (Asirvatham & Hoppe 2005). Avoids LOD seams via vertex morph at boundaries.
3. **Streaming layer.** Async I/O thread loads paged heightfield blocks into a ring buffer. Main thread queries "tile (x,z)" and gets a GPU-ready handle if loaded, an "in-flight" sentinel otherwise. Block size ~ 256×256 samples (~64 KB at 16-bit).
4. **GPU representation.** One large vertex buffer for the LOD mesh template, indexed per-tile via per-instance push-constant transform. Per-tile heightmap as `R16_UNORM` texture, sampled in the vertex shader to displace `y`. Normal recomputed in the pixel shader from the heightmap derivative for correct shading.
5. **Triplanar texturing.** Already have triplanar in `cube.frag`; same idea for terrain — sample ground/grass/rock from the existing texture array, blend by slope (steep faces → rock; flat tops → grass; surfaces near water → sand).
6. **Castle placement.** Compute heightfield value at castle origin, set `make_arena` to use that y as floor offset.
7. **RT integration.** The terrain mesh has to enter the TLAS as one or more BLAS instances. Static, so only built once. With 256×256 tile heightmaps that's 130k triangles per tile — BLAS per tile, group nearby tiles into one BLAS via `vkCmdBuildAccelerationStructures` batched call.

**Open-source libraries / references:**
- Strugar's [CDLOD paper](https://0fps.net/2014/06/02/voxel-terrain-and-cdlod/) (foundational, no code drop-in)
- [vkTerrain](https://github.com/SaschaWillems/Vulkan/tree/master/examples/terraintessellation) — Sascha Willems' tessellation-based heightmap example, Vulkan-native; doesn't do streaming but the GPU pipeline is reusable
- [bgfx terrain](https://github.com/bkaradzic/bgfx/tree/master/examples/26-occlusion) — clipmap reference
- [OGRE Terrain Component](https://github.com/OGRECave/ogre/tree/master/Components/Terrain) — full implementation, Apache 2; would need extracting (heavy OGRE dependency, but the algorithm is the value)
- [voxelfarm / Roblox-style streaming](https://github.com/voxelfarm/voxelfarm) — overkill for heightfield-only, but the streaming infra is good
- [tinygltf](https://github.com/syoyo/tinygltf) we already link — for static foliage/rock-set placement on terrain

**Recommended path.** Start with non-streaming heightmap + tessellation (Willems example as starting point), get terrain rendering and integrated with RT. Then add CDLOD for distance LOD. Then add async I/O streaming. Each step is a session.

---

### Scene baking ("quick baker in settings")

**Goal.** Bake a static representation of the GI / shadow contribution from rt.cpp's path tracer and use it as a fast lookup at runtime. Combine with RT: baked = static contribution, RT = dynamic-only delta.

**Scope review.** Real-engine bakes are days of work — UV unwrapping, light-map atlas packing, multi-bounce path traced per texel, blue-noise dither, then runtime sampling. A "quick baker" usable at engine init is achievable but still ~1–2 sessions:

1. **Lightmap target.** One R16G16B16A16_SFLOAT texture per static brush face (or atlas-packed for all faces). Resolution 32×32 or 64×64 per face is plenty for diffuse GI.
2. **Per-texel ray budget.** Reuse cube.frag's GI loop: 64 cosine-weighted samples × 1–2 bounces, sampling the static-only TLAS. ~10 minutes per face on RTX 4080.
3. **Bake API.** A `--bake` CLI flag writes lightmaps to `bake/` next to assets. Engine startup detects them, loads them, and `cube.frag` samples them as ambient term.
4. **RT delta.** With baked indirect, we'd reduce `rt_.gi_samples` to 0 by default and only run RT for the dynamic contribution (shadows from dynamic boxes onto static surfaces, GI from dynamic emissives, sun shadows on dynamic surfaces).
5. **Settings UI.** Button "Bake static lighting" in the menu — runs the bake on the current TLAS, freezes UI for the bake duration, writes files.

**References:**
- [The Witness lightmapping postmortem](https://medium.com/@bgolus/the-witness-lightmaps-65bb12ce95b6)
- [Frostbite GI baking talk](https://www.ea.com/frostbite/news/precomputed-global-illumination-in-frostbite)
- [PMaterial light/shadow caching in UE](https://docs.unrealengine.com/4.27/en-US/RenderingAndGraphics/LightingAndShadows/StaticLighting/) — workflow reference
- We could reuse `xatlas` (https://github.com/jpcy/xatlas) for UV unwrapping if we move beyond per-face atlases.

**Recommended path.** Start with per-face atlases (no UV unwrap needed — each face gets a 32×32 grid). Bake AO term first (simpler, single bounce). Then add 1-bounce diffuse GI. Then optionally 2nd bounce.

---

### Honest scope assessment

Both items are real graphics-programming projects. Each session can pick one and ship a small but meaningful slice — the right cadence is "one feature per chat." Listed roughly in priority for an engine demo:

1. Static-bake AO (smallest, biggest visible win)
2. Heightmap terrain (no streaming, fixed-size — gets terrain on screen)
3. Static-bake diffuse GI (replaces 64-sample inline GI)
4. Terrain LOD
5. Terrain streaming
