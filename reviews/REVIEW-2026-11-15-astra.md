# FLAWS

Reviewed commit `15b0b7a` through `git show`; all findings below are from reading. No GPU programs ran; live integration and visual quality were not checked.

1. **The lifecycle test reads outside its input.** `src/stage1_test.cpp:test5` allocates `Frame(640,360)`, then updates `Tile(1920,1080)` using stride 640. Both `cpu_phosphor_update` and `phosphor_update_buf` overrun their inputs. Index arithmetic confirms a last byte of 2,769,919 against 921,600 allocated bytes. The passing lifecycle claim in `decision.md` is unreliable. Resize the fixture before rerunning; use memory checking.
2. **Fresh surfaces read uninitialized state.** `src/pipelines.hpp:CudaSurf::create/GlslSurf::create` allocate without clearing. `make_step(fresh)` sets weights to one, but `phosphor_step` still evaluates `prev + (target-prev)`. NaNs survive; large finite garbage loses the target through cancellation. Directly adopt the target without reading old state. Stage 1 zero-initializes, so misses this.
3. **Fallback can retain mapped GL resources.** `src/crt_render.cpp:main` catches failures after `CudaPath::begin` without guaranteed `end`. Subsequent GL uploads can access mapped textures, which [NVIDIA defines as undefined](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPHICS.html). Passthrough is not proven safe after arbitrary CUDA errors.
4. **Activity stores race.** `kernels/phosphor.cu:phosphor_update_buf/tex` concurrently write `*active=1`. Equal values do not remove the [data race](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html). Reduce within each block, then issue one atomic OR.
5. **“Owes no frames” excludes animated glass.** `CudaPath::active_result` covers phosphor only; `make_optics` animates tracking/flicker on still content. Choose whether idle freezes those effects or schedules their next event.
6. **Timings need narrower labels.** `src/stage2_bench.cpp:Bench::present/run` measures an offscreen blit plus forced completion, not display presentation or input latency. `Bench::breakdown` includes synchronization, object creation/destruction, memset and presentation: 0.12 ms is not isolated mapping cost. Half-float GLSL suggests bandwidth pressure; it does not prove CUDA is memory-bound. Clock/order effects remain.

# SPEED

Ranked implementation order. Gains are unmeasured estimates for this 3080 Laptop, relative to the current path; they do not add together.

1. **Half-float storage, float arithmetic:** likely 20–45% less kernel time, supported indirectly by the GLSL result. Keep texture filtering. Quantize targets consistently and handle rounding stalls; merely widening epsilon to half an ULP can still stall at small step weights. Validate every input code at several refresh rates. Risk: earlier settling and altered dim tails.
2. **Hardware sRGB input decode:** sensible; [`CU_TRSF_SRGB`](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html) removes three decode powers per update pixel. Estimate 5–20% of full kernel time. Remove software decode and preserve a separate raw sampler for `passthrough_tex`, otherwise passthrough becomes dark. Hardware rounding and actual texture support need checking. Output encoding powers remain; test a bounded approximation separately.
3. **Remove synchronous activity readback:** `CudaPath::active_result` forces completion. Use pinned asynchronous readback, consume a completed previous flag, conservatively redraw while unknown. Batch tiles under one map/unmap. Estimate 0–0.1 ms wall savings for one tile; larger benefits with many tiles. Risk: extra redraws and synchronization bugs.
4. **Damage plus active-block lists:** update changed or still-decaying blocks; retain settled state. Dilate output damage for warp, aberration and glow. Potentially skip over 90% of update pixels during localized typing; whole-frame savings depend on optics. Risk: stale trails, scrolling and bookkeeping overhead.
5. **Keep 16×16 or 32×8:** `out/block_sweep.csv` already finds equal p95, while 32×32 loses. Expect under 5% from further shape tuning; risk is lower occupancy. Profile register/texture pressure before shared-memory rewrites.

Do not simply retain mapped textures while GL uses them. External memory/semaphores can replace that ownership mechanism, but need a new interop design. Keep update and optics separate: neighbor gathers make naive in-place fusion unsafe.

# COOL

1. **Brightness-dependent beam width plus multiscale linear-light halation.** Bright strokes swell; dim strokes stay sharp. CUDA makes downsampled separable glow and extra state practical, though GLSL can also do them.
2. **Two decay terms with channel-specific tails.** Keep the slow term faint and bounded; call it afterglow. Actual burn-in is a different, long-term phenomenon.
3. **Pixel-footprint-filtered grille/slot mask.** Anchor it to tube coordinates; suppress it below resolvable size to prevent moiré.
4. **Degauss only on command/startup.** Cut periodic wobble and exaggerated colour splitting during reading.

# HIS TWO QUESTIONS

**Any OS/NVIDIA machine? No.** Native Linux and Windows are realistic ports with supported drivers/GPUs. Current EGL/POSIX hosting needs platform adaptation; `Cuda::init` must select the GPU owning the graphics context rather than device zero. Embedding in an app is simpler than filtering arbitrary desktop windows: those need OS/compositor-specific integration. CUDA capability alone is insufficient. [Modern macOS lacks CUDA](https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/); [WSL lacks this OpenGL interop](https://docs.nvidia.com/cuda/wsl-user-guide/). Other machines were not checked.

**Snappy in a window? Likely on this GPU.** Recorded 1080p wall p50 is 0.67 ms; that fits comfortably within 144 Hz's 6.94 ms budget. Scheduling, contention, resizing and input-to-display latency were not checked. The timings justify a live prototype; they do not require restricting CUDA to offline video.

# ONE CREATIVE DIRECTION

**A vector-tube mode:** accept terminal glyph strokes or paths and deposit beam energy directly into phosphor, bypassing the raster scan aesthetic. Moving strokes leave continuous trails; ordinary window content retains the raster fallback.
