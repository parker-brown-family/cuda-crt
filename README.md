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

The renderer runs offscreen through a headless EGL context. The optional
`make doom` launcher loads the CUDA window pass into your running Hyprland for
the game, then restores the preceding CUDA setup when the game closes.

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

## Inside Hyprland

`hypr/cuda-pass.cpp` is a Hyprland 0.56 plugin that runs the same kernels on one
window (or a whole monitor), after the windows are drawn and before the bar and
notifications. Clicks are mapped through the glass: the cursor stays where your
hand puts it, and while Hyprland works out what the pointer is over it is given
the point of the window drawn under the cursor, from the same `tube_map.h` the
picture is bent with (the tracking bar, jiggle and switch-on wobble included).
By default the curve runs across the whole edge and the picture is drawn 6%
smaller, so the rim never sits on a window's own edges.

Everything it reads and writes is one folder, `~/.local/state/cuda-crt`:
`root` (this checkout), `target` (`0x...` address, `class:...`, `title:...`, or
absent for the whole monitor), `look` (`tube` or `phosphor`), `set` (dial
overrides read every frame), `ttl` (stand down after N seconds), `off`. It
writes `pass.log`, `perf.csv` (a row every 5 s), and `pointer.why` (why the
pointer was or was not mapped).

```bash
hypr/build ~/.local/state/cuda-crt/cuda-pass.so && hyprctl plugin load ~/.local/state/cuda-crt/cuda-pass.so
```

The proofs run in omarchy-crt's sealed seat, a sandboxed headless Hyprland that
cannot reach a live desktop (point `OMARCHY_CRT` at a checkout):

```bash
hypr/prove /tmp/cuda-seat out/hypr          # 17 checks: textures, picture, cost, idle, a failure mid-frame, unload, one window on two monitors
```
```bash
hypr/prove-aim /tmp/cuda-seat out/hypr      # real clicks on 13 targets, found in the picture; ONLY=F fullscreen, ONLY=G with the bar bending
```
```bash
hypr/prove-browser /tmp/cuda-seat out/hypr  # a real Chromium watched for a frozen picture and lost clicks; CYCLES=n reloads first
```

Any video in, CRT video out:

```bash
ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgba - | bin/crt_render . --size 1280x720 --fps 60 --path cuda2 | ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - -pix_fmt yuv420p out.mp4
```

Vector mode, no input:

```bash
bin/crt_render . --size 1280x720 --fps 60 --vector lissajous --frames 480 --decay 40,120,70 --set scan=0,tracking=0,flicker=0,halation=0.55 | ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 60 -i - -pix_fmt yuv420p lissajous.mp4
```

## DOOM

Build the pinned Crispy Doom engine, fetch and checksum-check the original
shareware episode, and build the CUDA window pass:

DOOM setup also needs CMake and SDL2 development headers.

```bash
make doom-setup
```

Play fullscreen with the CRT effect:

```bash
make doom
```

For a window instead:

```bash
tools/doom.sh --crt --windowed
```

Or install it as an application: `doom-crt` on your PATH and **DOOM CRT** in the app
menu (`make doom-uninstall` takes both off). With no terminal to print to, a failure
becomes a desktop notification and `~/.local/state/cuda-crt/doom-launch.log`:

```bash
make doom-install
```

**The CRT's settings are in the game:** Esc, Options, CRT Glass. Three pages hold every
dial the pass reads, plus power, presets and a reset; left and right move a dial, Page
Up and Page Down turn the page, and the glass changes as you press. The look is saved
for the next session in the same file the settings panel uses, and either can be open
while the other is changed. DOOM started without `--crt` has no such page.

WASD or arrows move forward/back and strafe. The mouse turns and aims up/down;
left-click fires, E or right-click uses doors, Shift runs, and 1–7 or the wheel
selects weapons. Alt+Enter toggles the game's fullscreen mode; Esc opens its
menu. Sound effects and OPL music are enabled by default; pass `-nosound` to
`tools/doom.sh` when silence is wanted. Setup builds a pinned SDL2_mixer audio
backend locally, so no system-wide audio package installation is needed.

`tools/doom-audio-check` checks effects and OPL music separately against a
temporary, inaudible PulseAudio/PipeWire sink. It records neither the microphone
nor other applications and removes the test sink afterward.

The launcher loads its own CUDA pass, targets the game by class so fullscreen
changes cannot strand it on an old window address, and restores the preceding
CUDA state and loaded plugin when the game exits. The pass keeps a fullscreen
target on Hyprland's composited path: its lone-window shortcut skips the render
stage where CUDA draws the glass. Other fullscreen applications keep that
shortcut. No Hyprland config file or persistent window rule is changed.

`tools/doom-controls.cfg` and `tools/doom-fps.cfg` seed the game's profile once
under `$XDG_STATE_HOME/cuda-crt/doom` (normally
`~/.local/state/cuda-crt/doom`). Later in-game sensitivity adjustments persist.
`DOOM_ENGINE` and `DOOM_WAD` can select an existing engine and your own game data.
`tools/doom.sh` without `--crt` opens the game alone.

The engine source, game data and compiled plugin are rebuildable files under
ignored `.toolchain/` and `bin/` directories. The pinned engine is Crispy Doom
7.2.0 at `4556710c0d119aa7a79d1c0cdaa2b265fe4b15e2`, GPL-2.0, running as a
separate process. The shareware WAD has SHA-256
`1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771`.
No engine binary or game data is committed to this MIT repository.

Two proofs run inside omarchy-crt's sealed compositor, with no access to the
live desktop. The fullscreen proof forces the shortcut eligible, checks that
CUDA blocks it only for its target, and captures the effect across fullscreen,
windowed and fullscreen-again transitions:

```bash
SOLITARY=1 FULLSCREEN_ONLY=1 hypr/prove-doom /tmp/cuda-doom-seat reports/doom-fullscreen
```

The session proof launches the shipped command, verifies WASD strafing and
firing from DOOM's recorded input, checks the saved mouse-look profile, and
checks cleanup. `PREVIOUS=1` also loads an existing CUDA pass first and checks
that it is restored:

```bash
PREVIOUS=1 hypr/prove-doom-session /tmp/cuda-doom-session-seat reports/doom-session
```

`hypr/solitary-proof.cpp` is a test fixture for that sealed seat, never loaded
by the live launcher. The original `hypr/prove-doom` movie/input proof remains
available with a local Chocolate Doom build; its sampled 5 fps capture is
visual evidence, not a gameplay performance measurement.

The menu proof walks every page with real key presses and checks the files, the
picture, the saved look and a second launch:

```bash
hypr/prove-doom-menu /tmp/cuda-menu-seat reports/doom-menu
```

The menu is `doom/crispy-crt-menu.patch` and `doom/m_crt.c`, applied to the pinned
engine by `make doom-setup`. Those files are GPL-2.0-or-later like the engine they
modify; the rest of this repository stays MIT. Their dial table is generated from
`tools/dials.json` and `tools/presets.json`, the panel's own source.

## Dials

Open the live CRT settings panel from the repository:

```bash
make crt-config
```

The sliders change the active Hyprland CUDA pass and save a profile for later
`make doom` sessions. Opening the panel reads the current settings without
changing the picture. It exposes the compositor's supported controls, including
curve, safe edge, phosphor fade, glow, scanlines and tracking. The panel service
runs under systemd on loopback port 8788; the browser opens as a small app window.
Settings live in `~/.local/state/cuda-crt/tube-profile.set` (or the equivalent
under `XDG_STATE_HOME`). The offline `tools/tube.sh` preview keeps its own panel.

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
tube_map.h             the curve and the transient motion: the ONLY definition, used by picture and pointer
kernels/phosphor.cu    the STATE: phosphor update kernels, held bit-exact to the CPU
kernels/glass.cu       drawn FROM the state: glass, halation, vector mode (fast math)
shaders/               the matched GLSL: update.frag, optics.frag, probe.frag
baseline/              the shipped omarchy-crt glass shader, pinned
src/cpu_ref.hpp        the CPU reference the kernels are judged against
src/cudart.hpp         NVRTC + driver API, no toolkit
src/glrt.hpp           headless EGL + OpenGL helpers
src/look.hpp           the dials and the per-frame scalars the clock drives
src/cuda_path.hpp      the CUDA side of a frame (no OpenGL header, so a plugin can use it)
src/pipelines.hpp      every path behind one shape
src/probe.cpp  src/stage1_test.cpp  src/stage2_bench.cpp  src/crt_render.cpp
tools/                 fixtures, demo, live tube, slider panel (knobs.py), benchmark runner, mutation check
hypr/                  the Hyprland plugin, its build, and its three proofs
docs/research-notes.md what outside sources say, and what was done with it
reviews/               an independent review of the first pass, and that reviewer's later build note and Hyprland plan
audit.md  decision.md  results.csv
```

## Licence

MIT. The baseline shader is from omarchy-crt, same author, same licence.
