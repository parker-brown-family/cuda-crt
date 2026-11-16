# Research notes

What outside sources say that bears on this lab, and what was done with it.
These came from a research notebook of about 40 collected sources, queried on
2026-11-15. The notebook's answers are a machine's reading of those sources:
**every number below is a lead to check against the original, not a measurement
of ours**, unless it says "measured here".

## Phosphor decay is not one exponential

- Real phosphors need a sum of terms: fast exponentials plus, for some, a
  power-law tail `I(t) ~ (t + a)^-b` with `b` a little over 1.
- Fitted impulse responses for a colour monitor's P22 phosphors, from Kuhn,
  *Optical Time-Domain Eavesdropping Risks of CRT Displays* (2002), as the
  notebook quotes them:
  - red: a pure sum of exponentials, effectively gone in 1 to 2 ms
  - green: `210e-6 * (t + 5.5 us)^-1.1` plus exponentials at 150 kHz, 700 kHz, 5 MHz
  - blue: `190e-6 * (t + 5 us)^-1.11` plus exponentials at 100 kHz, 1.1 MHz, 4 MHz
- Persistence classes, qualitative only in the sources: P1 medium, P4 medium to
  short, P31 medium-short, P39 medium-long. No decay-to-10% times were found.

**What it means here.** A colour TV tube's light is gone well inside one 60 Hz
frame, so visible frame-to-frame trails belong to the green and amber terminal
phosphors, not to television. And the faint long tail is real: it is the
power-law term. The lab's "phosphor burn ghost" (a slow second term in the
state's fourth channel) is a one-exponential stand-in for that tail. A closer
model would sum three or four exponentials at log-spaced time constants. Not
built. The presets in `tools/presets.json` are named "-like" for this reason:
they are shaped by these classes and tuned by eye.

## The beam

- Spot profile: a 2D Gaussian; CRT-Royale uses a generalised Gaussian with a
  shape parameter between 2 and 4.
- Blooming: the spot widens as beam current rises (space-charge repulsion).
  CRT-Royale models it with `beam_spot_power = 1/3`. No analytic law was found.
- Used here as the `beam` dial: a neighbour's spill is weighted by its own
  brightness, so bright strokes swell. Artistic, not fitted.

## Masks

Aperture-grille pitch 0.31 mm (Sony PVM-20M4U / BVM-20F1U); slot pitch 0.60,
0.75, 0.90 mm for 13, 19 and 25 inch tubes; colour shadow-mask pitch up to
0.25 mm (all via the CRT-Royale README and a CRT displays text). The lab's
`mask` dial is one screen pixel per stripe, which is the finest a panel can
show and says nothing about a real pitch.

## Running an effect on one window in Hyprland

- **HyprWindowShade** hooks the compositor's texture draw (`hkGLDrawTex`,
  `hkUseShader`), runs its pass offscreen, hands the texture back. It keeps no
  history between frames and redraws continuously only when a shader declares
  `uniform float time`.
- A **Liquid Glass** plugin runs as a decoration pass under the client, keeps
  persistent framebuffers per surface across frames, and propagates damage only
  where regions intersect.
- These are the two known patterns for the Hyprland proof: hook the window's
  texture draw, or draw as a decoration with per-surface state. omarchy-crt's
  own native pass is a third: one pass over the whole monitor before overlays.

## Sharing textures between CUDA and a graphics API

- Mapping and unmapping through CUDA's graphics interop is reported at 0.15 to
  0.50 ms a frame, against 0.01 to 0.03 ms for Vulkan with timeline semaphores.
  **Measured here:** 0.12 ms to begin a frame and 0.19 to 0.33 ms to end one
  with OpenGL (`out/cuda_breakdown.csv`), which sits inside that range.
- VUDA (a research system) reports up to 85% higher throughput by breaking
  CUDA and Vulkan context isolation. Not something a plugin can use.
- Zero-copy window capture on Linux goes through DMA-BUF; one thread reports
  stale reads on NVIDIA under a specific PCIe setting. Relevant only if this
  ever captures windows from outside the compositor.

## What the sources do not have

A model for a rolling hum bar or tracking band, for horizontal sync jitter, or
for ringing: "not in sources" for CRT-Royale, Guest Advanced and Lola Retro.
The lab's tracking bar (bend, tearing, static, colour split) is drawn from
memory of how worn tape and ground-loop hum look, with no source behind it.
