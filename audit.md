# Stage 0 audit: what exists, what is stateful, and what the box can do

Written 2026-11-15 for the CUDA virtual phosphor surface experiment. Everything
here was read from the repository at `bcfd334` (main, in step with origin) or
measured on this machine by `bin/probe`. Nothing under `crt/`, `native/`, the
installed plugin or the Hyprland session was changed.

## The finding that shapes the rest

**The Omarchy CRT glass has no memory.** `crt/crt-glass.frag` is a pure function
of the current frame and, when motion is on, a clock. Nothing in it reads a
previous frame. The same is true of the native pass, which runs the same shader
source. So a phosphor that lingers after a letter changes is new behaviour for
this plugin. The plugin has no afterglow to improve; the surface adds one.

| Effect in `crt-glass.frag` | Depends on | Stateful? |
|---|---|---|
| Per-tile barrel curve (`tube_map`, `TUBE_K1`/`TUBE_K2`) | pixel position, tile rect | no |
| Chromatic split at the rim (`ABERR`) | position | no |
| Scanlines (`SCAN`, `SCAN_STEP`) | row within the tile | no |
| 4-tap glow (`GLOW`) | this frame's neighbours | no |
| Centre phosphor wash (`BLOOM`), vignette (`VIGN`), specular, glare | position | no |
| Tracking band (`TRACKING`) | position and `time` | no: a function of the clock |
| Flicker bursts (`FLICKER`), jiggle | `time` | no: a function of the clock |
| Colour grade | this pixel | no |

Terminal Delight's glass pass (`gpui_wgpu/src/crt_pass.wgsl`) is the same kind of
thing: per-rect curve, rounded glass rim with a lip shadow, glare, a bright-pass
halo. Its header says so directly: "no history or additional blur target is
held". Its scanlines and tracking band are painted as quads in `app/src/crt.rs`.
I did not find the code for Terminal Delight's own letter-change persistence in
the branch that is checked out (`usage-pace-burndown`), so this audit says
nothing about how it works there.

## Compatibility hazards for any later integration

1. **Motion already costs damage tracking.** Hyprland treats a screen shader that
   declares `uniform float time` as animated and redraws the whole screen every
   frame; the plugin moves `debug:damage_tracking` between three tiers to cope
   (README, "Motion costs damage tracking"). A phosphor that is still fading
   needs frames nobody else asked for. That is the same problem with a
   different trigger, and it is why the surface reports an `active` flag: the
   host asks for another frame only while a tile's trail is still above the
   visible threshold.
2. **A monitor with no damage gets no frames** (README, "The screen you are not
   looking at"). A trail on an idle monitor would freeze mid-fade unless the
   surface's `active` flag is wired to frame scheduling. The lab proves the
   flag; it does not schedule compositor frames.
3. **Tile identity is positional today.** `crt/tubes-geometry` builds the tube
   list from `hyprctl clients -j` as rectangles in focus order and never reads a
   window's `address`. A tube's index changes when focus changes, so the index
   cannot key a history buffer. The native pass does hold a `PHLWINDOW` per row
   (`native/crt-pass.cpp`, `tubes()`), which is a stable identity. A phosphor
   surface per window therefore fits the native pass and does not fit the
   screen-shader path, which cannot keep textures between frames at all.
4. **The native pass is tied to one exact Hyprland build** and refuses to load
   on any other (`PLUGIN_INIT`, API hash check). It hooks three functions by
   name. A CUDA stage would live inside that process: a CUDA error path that
   throws or hangs there takes the session with it. The handoff's rule stands:
   not the first target.
5. **The native pass already copies the framebuffer once per frame** into a
   scratch texture (`glBlitFramebuffer`, one scratch per framebuffer size). That
   scratch is the natural input for a phosphor update, and a second texture of
   the same kind is the natural place for state.

## Safety switches that exist, and were left alone

`crt toggle` (the whole effect), `crt native on|off` (the native pass, loaded
into the running session only, never into config), `crt passthrough` (per-surface
opt-out), and the `crt-tubes.service` guard that refuses to start if the plugin
has been removed. The lab uses none of them and changes none of them.

## What this box can do (measured by `bin/probe`)

| Question | Answer |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB, compute capability 8.6 |
| Driver | 615.78.08, exposing CUDA driver API 13.4 |
| CUDA toolkit | not installed. The lab uses NVRTC 12.9 from PyPI wheels in `.toolchain/` (320 MB, this folder only) to compile the kernel at run time, and the driver's own `libcuda.so` to run it. No `nvcc`, no system package, no sudo. |
| Compositor | Hyprland 0.56.2 |
| Monitors | DP-2 1920x1080 at 60 Hz; eDP-1 2560x1600 at 60 Hz (165 Hz available) |
| Headless OpenGL on the NVIDIA GPU | yes: EGL device platform, OpenGL 4.5 core |
| Can CUDA share an OpenGL texture, desktop GL 4.5? | yes: RGBA8 read, RGBA8 surface write, RGBA16F, RGBA32F, SRGB8_ALPHA8 all register and map |
| Same question for OpenGL ES 3.2, which Hyprland runs? | yes, the same five formats register and map in a GLES 3.2 context |

The last row is the one that mattered for a later Hyprland proof: a compositor's
GLES textures are shareable with CUDA on this driver. It was measured in a
standalone context, not inside Hyprland.

## Fixtures

All deterministic, no files, no randomness that is not seeded:

- `src/stage1_test.cpp` draws its own frames: a square on for 200 ms, seeded
  noise that changes every other frame, a constant image, two tiles on a desk.
- `src/scene.hpp` draws dot-matrix terminal rows with a scroll offset and a
  cursor, for the benchmark's four rotating input frames.
- `tools/fixture.py` draws real-font terminal footage (typing, a scroll burst, a
  clear; the 200 ms flash; two tiles) as raw frames for `bin/crt_render`.

## The integration interface this lab proves

One surface per stable tile id. The caller hands it the frame, the tile's
origin in the frame, and the real elapsed time:

```
surface.create(w, h)                 new, resized or shown again: no history
update(surface, frame, ox, oy, dt)   in place; origin can change freely (a move)
optics(surface, out, ox, oy, look)   the glass, reading the state
active = active_result()             does any tile still owe a frame?
surface.destroy()                    hidden or closed
```

- **Move**: only `ox, oy` change. The state is tile-local, so the trail travels
  with the tile (test 4, bit for bit).
- **Resize**: reset. The old surface is destroyed and the new one adopts the
  current frame, so there is no ghost and no flash. Reprojection was not built:
  a resize reflows a terminal's text, so the old trail would be in the wrong
  place anyway.
- **Failure**: any CUDA error is caught by the caller and the frame goes through
  untouched (`bin/crt_render` does this and says so once on stderr).
