## Vulkan best-practices audit (2024-2026)

Notes from reading recent Khronos / NVIDIA / AMD writeups, oriented to where this engine sits today and what's worth shipping next. Existing baseline:

- Vulkan 1.3, dynamic rendering (no render passes), sync2
- Async-compute queue for TLAS rebuild
- VMA, with `MAPPED_BIT` on host-visible buffers
- VkPipelineCache load/save
- Per-frame command pools, kFrameOverlap=2
- Inline RT via KHR_ray_query in cube.frag
- Sync + best-practices + GPU-assisted validation in Debug only (Release builds skip them)

### High-ROI quick wins

1. **Timeline semaphores** instead of binary across queues. The graphics submit currently waits on a binary `tlas_build_done` per frame; with a timeline semaphore we'd track a monotonic counter, eliminate a per-frame `vkCreateSemaphore` setup cost, and make CPU-side waits trivial. ~1 session refactor.
   - Khronos: https://www.khronos.org/blog/vulkan-timeline-semaphores
   - https://themaister.net/blog/2023/08/14/yet-another-blog-explaining-vulkan-synchronization/

2. **VK_EXT_descriptor_buffer**. Replaces `vkUpdateDescriptorSets` with memcpy into a buffer; ~2-5% CPU win, opens the door for proper bindless. Better target than push-descriptors as a stepping stone.
   - https://developer.nvidia.com/blog/vulkan-descriptor-buffer-extension-streamlines-resource-binding/
   - Sample: https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/descriptor_buffer_basic

3. **VK_EXT_host_image_copy** for texture upload. Eliminates the staging-buffer + transfer-queue dance — host writes directly into the device-optimal image. Big win on level-load.
   - https://www.khronos.org/blog/copying-images-on-the-host-in-vulkan

4. **Compute-only Jimenez bloom chain** — replace any blit-based downsample with a compute-shader downsample (13-tap dual filter) and upsample (9-tap tent). Saves layout transitions and a few hundred microseconds/frame; cleaner banding.
   - Jimenez SIGGRAPH 2014: https://advances.realtimerendering.com/s2014/

### Algorithmic upgrades

5. **ReSTIR DI** for the inline-RT lighting in `cube.frag`. Currently we do uniform 32-sample hemisphere GI per pixel — most of those rays miss bright sources entirely. ReSTIR DI keeps a per-pixel reservoir, importance-samples toward the bright-source TLAS instances, then traces ONE ray. Same visual result with 1-2 rays vs 32. Pairs naturally with our existing à-trous filter.
   - https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
   - https://github.com/NVIDIAGameWorks/RTXDI

6. **FSR 2.2** as a drop-in replacement for our current TAA — gives both AA AND upscaling for free, and the implementation is open source under MIT.
   - https://gpuopen.com/fidelityfx-superresolution-2/

### TLAS / BLAS

7. **Bake static brushes into one BLAS** instead of N TLAS instances pointing at the cube BLAS. With 150+ static brushes (castle ate that quota) the TLAS gets 150+ entries to traverse per ray; one combined BLAS would mean 1 TLAS entry. Major ray-traversal speed win, modest one-time bake cost. (Geometry stays the same — combine into a single pre-transformed index buffer.)

8. **BLAS compaction** for static cube/cylinder BLASes — set `ALLOW_COMPACTION_BIT`, build, then `vkCmdCopyAccelerationStructureKHR` with COMPACT mode. ~30-50% memory + faster traversal. The cube/cylinder BLASes are tiny so memory savings are small here, but #7 above makes this worthwhile.
   - https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/

9. **PREFER_FAST_TRACE for static, PREFER_FAST_BUILD for dynamic** — already correct in our code. Cube + cylinder are FAST_TRACE; TLAS is FAST_TRACE + ALLOW_UPDATE for cheap UPDATE-mode rebuilds.

### Anti-patterns to verify we're NOT doing

- `vkQueueWaitIdle` / `vkDeviceWaitIdle` in steady state — only on shutdown / resize. Confirmed: we only do these on shutdown + recreate_swapchain.
- Many tiny `VkSubmitInfo`s — should batch into one `vkQueueSubmit2` with multiple infos. Confirmed: we already submit graphics work as a single submit per frame.
- `VK_IMAGE_LAYOUT_GENERAL` everywhere — costs perf on AMD. Confirmed: we use specific layouts (COLOR_ATTACHMENT_OPTIMAL, SHADER_READ_ONLY_OPTIMAL, etc).
- Allocating descriptor sets per draw — confirmed: all desc sets allocated once at init or on resize.
- Per-frame VkPipeline creation — confirmed: pipelines built once at init.

### Rec ordering

If picking one thing per session: (1) static-brush BLAS bake → (2) ReSTIR DI → (3) timeline semaphores. The first is the largest perf win for this scene's geometry pattern.

---

## Static-brush BLAS bake — implementation plan

The single largest RT speedup available to this engine. Currently every static
brush is a separate TLAS instance pointing at the cube BLAS — at 150+ brushes
that's 150+ entries each ray traverses. Combining them into ONE BLAS makes the
TLAS see "1 static + N dynamic" instead of "150 static + N dynamic".

**Files touched:**
- `src/engine/vk_engine/rt.cpp` — bake function + TLAS rebuild change
- `src/engine/vk_engine.h` — new buffer/AS handles
- `shaders/cube.frag` — material lookup on hit
- `src/engine/vk_engine/descriptors.cpp` — bind new SSBO

**Steps:**

1. **Build combined geometry buffers** in `init_rt()` after cube/cylinder BLAS:
   - Iterate `world_.brushes`, for each brush transform the cube's 24 verts into
     world space (apply translate + scale). Append into `static_combined_vbo`
     (N × 24 vec3).
   - Append cube's 36 indices, each offset by the brush's first-vertex index,
     into `static_combined_ibo` (N × 36 uint32).
   - Build a parallel `static_prim_to_brush` SSBO: 12 entries per brush
     (one per triangle), each storing the brush's index in
     `static_brush_materials_`. Total = N × 12 × 4 bytes ≈ 7 KB at 150
     brushes — trivial.

2. **Build the combined BLAS** with `PREFER_FAST_TRACE_BIT_KHR | ALLOW_COMPACTION_BIT_KHR`.
   After build, run a `vkCmdCopyAccelerationStructureKHR(COMPACT)` to
   shrink the AS storage — this matters here because the combined BLAS
   stores ~1850 triangles vs the cube's 12.

3. **Modify `rebuild_tlas()`**:
   - Drop the static-brush memcpy block.
   - Insert a single TLAS instance: identity transform, `instanceCustomIndex
     = kStaticCombinedFlag` (any sentinel, e.g. `0xFFFFFF`), pointing at
     `static_combined_blas_device_address_`.
   - Continue with dyn props / projectiles as before.

4. **Bind the prim-to-brush SSBO** at scene_desc set, e.g. binding 6.

5. **Shader change in `cube.frag::closest_hit_material`**:
   ```glsl
   bool closest_hit_material(...) {
       ... rayQueryProceedEXT loop ...
       int idx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
       if (idx == 0xFFFFFF) {
           int prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
           idx = int(static_prim_to_brush[prim]);
       }
       out_idx = idx;
       ...
   }
   ```
   Same change for the `closest_hit` path that needs material lookup.

**Expected impact:**
- TLAS instance count for the castle scene: ~240 → ~90 (drops the 150 static
  brushes to 1).
- Per-ray traversal: O(log 240) → O(log 90) — about a 1.5x ray throughput
  win.
- `kMaxDynProps` can go back up from 100 → 200 without GPU saturation.
- TLAS rebuild on the compute queue gets faster too (fewer instance writes,
  faster AS build).

**Risk / testing:**
- The instance-vs-prim material lookup branch is hot; verify no shader
  perf regression on simple scenes.
- Static brushes can no longer be hot-edited at runtime (no big deal —
  level data is fixed). Texture toggle (`textures_enabled`) currently
  re-bakes static_brush_materials_; that path still works, just doesn't
  need to rebuild the BLAS.
- Validate that `kStaticCombinedFlag` doesn't collide with any real
  custom index (we currently use 0..N for materials; pick a sentinel
  outside that range, e.g. 0x00FFFFFF since instanceCustomIndex is 24-bit).
