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
