# CUDA virtual phosphor surface (lab)

A standalone research prototype. It gives one tile of a desktop a phosphor that
remembers: when a letter changes, the old one fades instead of vanishing, and
the glass in front of it (curve, rim, scanlines, tracking band, glare) is drawn
from that fading state.

It is built three ways so the comparison is fair:

| Path | What it is |
|---|---|
| `existing` | the shipped `crt/crt-glass.frag`, unmodified, read at run time: one pass, no memory |
| `glsl` | phosphor state + glass as fragment shaders (`shaders/`) |
| `cuda` | phosphor state + glass as CUDA kernels (`kernels/phosphor.cu`) on textures shared with OpenGL, no CPU copies |

**Nothing here touches the live desktop.** It renders offscreen through a
headless EGL context, installs nothing outside this folder, and reads
`crt/crt-glass.frag` without writing it. Read `audit.md` first, then
`decision.md`.

## Build

Needs an NVIDIA GPU and driver, `g++`, `libepoxy`, `uv` and `ffmpeg`. No CUDA
toolkit: the kernel is compiled at run time by NVRTC and run through the
driver's own `libcuda.so`.

```bash
make toolchain
```
```bash
make
```

`make toolchain` downloads NVRTC and the CUDA headers from PyPI into
`.toolchain/` (about 320 MB, ignored by git). Delete the folder to undo it.

## Run

Can this box do it? One fact per line:

```bash
bin/probe .
```

Stage 1, correctness against the CPU reference (24 checks, exit 0 when all pass):

```bash
bin/stage1_test .
```

Stage 2, the benchmark. Writes `results.csv` and three smaller tables in `out/`.
About three minutes; close anything heavy on the GPU first:

```bash
bin/stage2_bench .
```

The tests' own teeth: four deliberately broken kernels, each of which the suite must fail:

```bash
tools/mutate.sh
```

How the GPU's clock moves under a 60 Hz load, one line per second (this is the
measurement `decision.md` turns on; sample `nvidia-smi` beside it):

```bash
bin/stage2_bench . power cuda_full 2400
```

The live demo: terminal footage through two renderers at once, side by side in
one window, in real time (close it with `q`; `f` is fullscreen). `--dry` runs
every act with no window:

```bash
tools/demo.sh
```

Footage. Any video in, CRT video out:

```bash
ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgba - | bin/crt_render . --size 1280x720 --fps 60 --path cuda --decay 50 | ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - -pix_fmt yuv420p out.mp4
```

The built-in terminal fixtures (`typing`, `flash`, `tiles`, `hold`):

```bash
uv run --with pillow tools/fixture.py typing --size 1280x720 --fps 60 | ffmpeg -y -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - -c:v libx264rgb -crf 0 out/src_typing.mkv
```
```bash
tools/render.sh typing cuda typing_cuda_50
```

## Dials

`--decay r,g,b` and `--rise` are the phosphor's time constants in milliseconds
(one value sets all three channels). They are example values for looking at,
not calibrated CRT physics. `--set name=value,...` reaches every glass dial:
`curv scan scan_step glow glow_px bloom vign specular glare tracking track_rest
track_sweep band_h flicker aberr k1 k2`. `--tiles "x,y,w,h;x,y,w,h"` gives each
rectangle its own surface; the rest of the frame passes through.

## Layout

```
phosphor_params.h     the two parameter structs every path shares
kernels/phosphor.cu   the CUDA kernels: phosphor update, glass, passthrough
shaders/              the matched GLSL: update.frag, optics.frag, probe.frag
src/cpu_ref.hpp       the CPU reference the kernel is judged against
src/cudart.hpp        NVRTC + driver API, no toolkit
src/glrt.hpp          headless EGL + OpenGL helpers
src/pipelines.hpp     the three paths behind one shape
src/probe.cpp         Stage 0: what the box can do
src/stage1_test.cpp   Stage 1: the six deterministic tests
src/stage2_bench.cpp  Stage 2: the benchmark
src/crt_render.cpp    frames in, CRT frames out
tools/fixture.py      deterministic terminal footage
tools/summarize.py    results.csv as the tables decision.md quotes
tools/build_brief.py  the local review page (machine-specific paths; not for a public push)
audit.md  decision.md  results.csv
```
