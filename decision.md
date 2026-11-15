# Decision: keep CUDA, keep GLSL, or investigate further

Written 2026-11-15, from the measurements in `results.csv` and `out/`. Machine:
NVIDIA GeForce RTX 3080 Laptop GPU (16 GB), driver 615.78.08 (CUDA driver API
13.4), NVRTC 12.9, Hyprland 0.56.2 running the desktop on the same GPU
throughout. Nothing ran inside Hyprland.

## Verdict

**Keep GLSL for the desktop. Keep the CUDA renderer as a video engine.** This is
a recommendation; the choice is Parker's.

The phosphor surface works and is worth having: the plugin has no temporal
state today (`audit.md`), and a letter that fades instead of vanishing is a
visible change (`out/typing_existing.mp4` beside `out/typing_cuda_50.mp4`). But
CUDA does not earn its place on the desktop. It ties the shader on speed and
costs 16 W more on this laptop while a trail is live.

## The evidence

### 1. Correctness: passes

`bin/stage1_test`, 24 checks over the handoff's six tests, all pass. The CUDA
state equals the CPU reference within 1.2e-07 (linear light, white = 1). The
suite was run against four deliberately broken kernels and failed on each
(`tools/mutate.sh`). The GLSL and CUDA outputs differ by 0.04 of a 255-step
value on average at 1080p; 0.05% of values differ by more than one step, all at
hard scanline thresholds (`out/glsl_vs_cuda_image.csv`: maximum 51 with
scanlines, 9 without).

### 2. The 1.0 ms gate: met at 1080p at full clock, by both

p95 GPU time in the pass, GPU at full clock (about 1500 to 1800 MHz), 600
frames each. "Added" is the path minus the shipped glass shader.

| Size | Shipped glass | GLSL phosphor | added | CUDA phosphor | added | GLSL, half-float state | added |
|---|---|---|---|---|---|---|---|
| 960 x 540 | 0.027 ms | 0.101 ms | +0.07 | 0.094 ms | +0.07 | 0.052 ms | +0.03 |
| 1920 x 1080 | 0.096 ms | 0.366 ms | **+0.27** | 0.354 ms | **+0.26** | 0.173 ms | +0.08 |
| 3840 x 2160 | 0.416 ms | 1.430 ms | +1.01 | 1.540 ms | +1.12 | 0.656 ms | +0.24 |

- The proposed target, no more than 1.0 ms p95 GPU overhead at the primary test
  resolution, is met by both paths at 1080p with room to spare.
- At 4K both sit a hair over it with 32-bit float state.
- The half-float row is a timing only. It shows the cost is memory traffic. At
  half precision the settle test can stall one step from a bright target, so
  idle detection is not proven for it. That is the next thing to fix if 4K
  matters.

### 3. CUDA against GLSL: a tie on GPU time, a loss on wall time

Kernel time and shader time are within 4% of each other at 540p and 1080p, and
GLSL is 7% faster at 4K. End to end (idle GPU to a finished, presented frame,
p50, full clock, 1080p): shipped glass 0.17 ms, GLSL 0.47 ms, CUDA 0.67 ms.
CUDA's extra 0.2 ms is the hand-off with OpenGL: 0.12 ms to map the textures at
every size, more to unmap (`out/cuda_breakdown.csv`).

What CUDA gave that the shader does not: the state is updated in place (32 MB
at 1080p against GLSL's 63 MB for two ping-pong textures), and the "does this
tile still owe a frame" flag is written from inside the update kernel where
GLSL needs an extra occlusion-query pass, read one frame late.

### 4. Power and clock: the deciding measurement

`out/clock_trace.csv` and `out/clock_samples.csv`: forty seconds of 60 Hz
redraw at 1080p per path, pass time and GPU clock sampled each second. One run
per path; power is the whole card with the desktop on it.

| Path | First 15 s | After 20 s | GPU clock after 20 s | GPU power after 20 s |
|---|---|---|---|---|
| Shipped glass | 0.10 ms | 0.71 ms | 210 MHz | 18 W |
| GLSL phosphor | 0.37 ms | 3.03 ms | 480 MHz | 22 W |
| CUDA phosphor | 0.36 ms | 0.36 ms | 1545 MHz | 38 W |

After about 16 seconds of light 60 Hz work this GPU drops its clock. The shader
paths drop with it and get seven to eight times slower per pass, still far
inside a 16.7 ms frame. With CUDA kernels running each frame the clock never
drops. That is 16 W more than the GLSL phosphor for as long as trails are live,
which on a terminal someone is typing in is most of the time.

Two honest consequences:

- **In low gear the GLSL phosphor misses the gate as written**: it adds 2.3 ms
  over the shipped glass (3.03 against 0.71). The GPU chose that clock because
  the load was light, and the frame still takes under a fifth of its budget.
  Whether the gate should be read at full clock or at the clock the desktop
  actually runs at is a question for whoever owns the gate.
- **Every `paced_60hz` row in `results.csv` is at one of two clocks.** The
  `gpu_mhz_at_end` column says which. Read the slow GLSL rows as the low clock,
  not as a slow shader. Why CUDA holds the clock up, and whether a driver
  setting changes that, was not investigated.

### 5. Idle: no persistent redraw

From white, the trail is below the visible threshold 433 ms after the content
vanishes (theory 440 ms for a 50 ms decay), then the surface reports that it
owes no frames and stays quiet; a still screen owes none from the first frame
(`out/hold_cuda.mp4`: 0 of 120 frames). The lab reports the flag. It does not
schedule compositor frames; that wiring belongs to a Hyprland stage.

### 6. Fallback: exists

Any CUDA error in `bin/crt_render` passes frames through untouched and says so
once. The original shader is not modified and remains the default everywhere.

## Against the handoff's promotion rule

| Promote only if | CUDA | GLSL |
|---|---|---|
| Correctness tests pass | yes | matches CUDA's output |
| The visual difference is worth the complexity | yes, the fade is visible | same picture |
| p95 overhead at 1080p is 1.0 ms or less | yes at full clock (+0.26) | yes at full clock (+0.27); no at the low clock (+2.3) |
| No persistent idle penalty | idle is clean; **while active it pins the GPU at full clock, +16 W** | idle is clean |
| No session instability | not tested inside Hyprland | not tested inside Hyprland |
| A supported fallback exists | yes | yes |

## What would come next, if approved

1. The GLSL phosphor inside the native pass, behind its own switch, on a
   disposable session. Needs: a state texture per window (the pass already
   holds the `PHLWINDOW`), and the `active` flag driving frame scheduling.
2. Half-float state with a settle test that is safe at that precision.
3. The CUDA renderer as a video tool: `bin/crt_render` already takes any
   footage through `ffmpeg`. The physically richer model (beam sweep, phosphor
   dot pattern, colour bleed) belongs there, where there is no frame budget.

## Departures from the handoff, and why

- **One agent did both halves.** The handoff split the work between Fable and
  Astra; splitting work across agents is Parker's to start.
- **CUDA state is one buffer updated in place**, not two ping-pong buffers. Each
  thread reads and writes only its own pixel, so the second buffer buys nothing.
- **The glass was added to the CUDA path in this round.** The handoff said to
  hold spatial effects until correctness and a baseline existed; they do
  (`cuda_update` and `glsl_update` rows are the fade alone), and Parker's
  request named the glass effects.
- **Glow is taken in linear light**, then the remaining dials are applied
  display-referred exactly as `crt-glass.frag` applies them, so the look tracks
  the shipped dials.
- **Resize resets** the surface to the current frame. No reprojection.
- **The GLSL idle answer is read one frame late.** Reading it in the same frame
  stalls the pipeline, and no compositor would do that.
- **No CUDA toolkit was installed.** NVRTC from PyPI wheels in `.toolchain/`
  compiles the kernel at run time; the driver's `libcuda.so` runs it.
