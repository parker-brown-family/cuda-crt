# cuda-crt

A CRT you can run on a GPU: a phosphor that remembers, behind curved glass.

When a letter on screen changes, the old one fades instead of vanishing, glows a
little, and leaves a faint ghost. The glass in front (curve, rounded rim,
scanlines, a tracking band, glare) is drawn from that fading light. It runs as
CUDA kernels on textures shared with OpenGL, so a frame never leaves the GPU.

This is a research lab, not a product. It began as an experiment to find out
whether CUDA earns a place doing what a shader already does well, and it keeps
the evidence: every path is measured against the same glass shader the
[omarchy-crt](https://github.com/parker-brown-family/omarchy-crt) plugin ships.
Read `decision.md` for what was found.

**Nothing here touches a live desktop.** It renders offscreen through a headless
EGL context and installs nothing outside this folder.

## What is in it

| Path | What it is |
|---|---|
| `existing` | the shipped omarchy-crt glass shader, unmodified (`baseline/`): one pass, no memory |
| `glsl` | phosphor state + glass as fragment shaders (`shaders/`), the matched comparison |
| `cuda` | first pass: the same phosphor and glass as CUDA kernels |
| `cuda2` | second pass: half-precision state, halation, a phosphor burn ghost, swelling strokes, a power-on degauss, an optional aperture mask |
| vector mode | no input picture: a beam's strokes deposit light straight into the phosphor |

## Build

Needs an NVIDIA GPU and driver, `g++`, `libepoxy`, `uv` and `ffmpeg`. No CUDA
toolkit: the kernels are compiled at run time by NVRTC and run through the
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

Can this machine do it? One fact per line:

```bash
bin/probe .
```

The correctness suite: each kernel against a CPU reference (exit 0 when all pass):

```bash
bin/stage1_test .
```

Do the tests have teeth? Seven deliberately broken kernels, each of which the
suite must fail:

```bash
tools/mutate.sh
```

The benchmark. Writes `results.csv` and smaller tables in `out/`. About five
minutes; close anything heavy on the GPU first:

```bash
tools/final_bench.sh
```

The live demo: two renderers side by side in one window, in real time (`q`
closes it, `f` is fullscreen; `--dry` runs every act with no window):

```bash
tools/demo.sh
```

One live tube with a panel of sliders beside it. Every move shows on the next
frame: the ghost's strength and linger, each colour's fade, barrel warp,
scanlines, the tracking bar and how hard it disturbs the picture, jiggle,
flicker, and phosphor presets (`q` closes it; `--dry 1` proves the pipeline
with no window):

```bash
tools/tube.sh
```

Any video in, CRT video out:

```bash
ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgba - | bin/crt_render . --size 1280x720 --fps 60 --path cuda2 | ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - -pix_fmt yuv420p out.mp4
```

Vector mode, no input:

```bash
bin/crt_render . --size 1280x720 --fps 60 --vector lissajous --frames 480 --decay 40,120,70 --set scan=0,tracking=0,flicker=0,halation=0.55 | ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - -pix_fmt yuv420p lissajous.mp4
```

## Dials

`--decay r,g,b` and `--rise` are the phosphor's time constants in milliseconds
(one value sets all three). `--burn rise,decay` sets the ghost's. They are
values for the eye, not measured from a real tube.

`bin/crt_render . --dials` prints every dial with its range as JSON
(`tools/dials.json` is that output; `tools/presets.json` holds the presets).
`--knobs FILE` re-reads a file of values whenever it changes, which is how the
slider panel drives a running window.

`--set name=value,...` reaches the rest: `curv scan scan_step glow glow_px bloom
vign specular glare tracking track_rest track_sweep band_h flicker aberr k1 k2`
for the glass, and for the second pass `ghost ghost_decay ghost_rise halation
beam mask track_bend track_noise track_chroma jiggle degauss_at`. `--tiles "x,y,w,h;x,y,w,h"` gives each rectangle its own phosphor;
the rest of the frame passes through.

## Where it runs

Linux with an NVIDIA driver: built and measured here. Windows with an NVIDIA
driver: the kernels are portable, the EGL and POSIX code around them is not yet.
Not WSL (no CUDA/OpenGL texture sharing there), not macOS, not AMD or Intel
graphics. The `glsl` path is the one that runs anywhere.

## Layout

```
phosphor_params.h      the parameter structs every path shares
phosphor_half.h        half-precision conversion, shared by kernels and CPU reference
kernels/phosphor.cu    the STATE: phosphor update kernels, held bit-exact to the CPU
kernels/glass.cu       drawn FROM the state: glass, halation, vector mode (fast math)
shaders/               the matched GLSL: update.frag, optics.frag, probe.frag
baseline/              the shipped omarchy-crt glass shader, pinned
src/cpu_ref.hpp        the CPU reference the kernels are judged against
src/cudart.hpp         NVRTC + driver API, no toolkit
src/glrt.hpp           headless EGL + OpenGL helpers
src/pipelines.hpp      every path behind one shape
src/probe.cpp  src/stage1_test.cpp  src/stage2_bench.cpp  src/crt_render.cpp
tools/                 fixtures, demo, benchmark runner, mutation check, summaries
reviews/               an independent review of the first pass
audit.md  decision.md  results.csv
```

## Licence

MIT. The baseline shader is from omarchy-crt, same author, same licence.
