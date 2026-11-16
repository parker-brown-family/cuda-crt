# One-window CUDA proof inside Hyprland

Paper design against omarchy-crt `bcfd334`; no compositor, window or GPU test was run. The reported 0.60 ms cost and GLES interop support are handoff evidence, not remeasured here.

## Enter through the existing pass

Keep `native/crt-pass.cpp:PLUGIN_INIT`'s `RENDER_POST_WINDOWS` insertion and branch inside `CCrtPass::draw`, immediately after its scratch blit. This already sits below top/overlay layers. Avoid another private texture-draw hook.

Start with one explicitly selected, opaque, unobscured, unscaled window, wholly on one output. Reject overlap, transparency, spanning, lock and unsupported transforms to GLSL. This proves an in-compositor CUDA pass, not isolated surface capture. `tubes()` holds `PHLWINDOW` only in temporary `SRow`; `STube` currently discards identity. Add a weak owner reference and a monotonic lifetime token to the selected tube.

Own history in a plugin registry keyed by that lifetime token, with weak window ownership, pixel extent, output identity/scale/transform, GL-context generation, CUDA state, textures, registrations and last monotonic update time. Movement preserves tile-local history; resizing resets directly to current input. Destroy on unmap/close; never key by rectangle or reusable pointer alone.

## Keep a clean fallback picture

`SScratch/g_scratch` currently cache by framebuffer size. Preserve that unregistered GLSL source. Add a CUDA-only scratch copy plus tile-sized output/FBO; register their GL_RGBA8 textures once, input read-only and output surface-write. This extra source isolates the existing shader from a failed mapping.

Select CUDA with `cuGLGetDevices` for the current GLES context; restore the previous CUDA context afterward. Map both CUDA textures, update phosphor, render optics, unmap, then copy output onto the selected destination rectangle only after success. Keep unrelated regions untouched; restore read/draw FBOs, viewport, scissor, texture units, program and blend state.

Unregister before deleting/reallocating either texture. Window resize, output migration/scale/transform change, or context-generation change invalidates registrations/history; output change requires rechecking the device even at identical dimensions. After actual context loss, abandon old handles and recreate only with a verified new context.

For initialization, registration, mapping, allocation, launch or copy failure: catch at the pass boundary, stop CUDA scheduling, log once, and draw the existing GLSL shader from untouched scratch. Use scoped unmapping; quarantine resources whose ownership cannot be recovered. Do not sample their output or repeatedly retry. Native crashes and GPU/device loss cannot be made harmless by catching exceptions; survival remains an explicit test gate, not a guarantee.

## Request only the needed frames

Replace this proof's whole-monitor loop in `PLUGIN_INIT` (`RENDER_POST` calls `damageMonitor`) with a pending selected-window flag. In `RENDER_POST`, use `damageBox` for its logical box plus sampling/glow margins; the installed `IHyprRenderer::damageBox(CBox,bool)` defaults to scheduling a frame. Implementation semantics are not checked.

Damage the complete source footprint so the scratch contains freshly composited pixels, never yesterday's warped output. Constrain both CUDA commit and GLSL fallback to the damaged region; today's `CCrtPass::draw` disables scissoring and needs this change. Unknown/pending activity requests another frame; settled phosphor stops. Freeze tracking/jiggle for the idle proof, then test their explicit animation scheduling separately. Cancel pending work on unload, alongside existing `PLUGIN_EXIT` cleanup.

## Three risks and cheapest experiments

1. **Actual compositor textures/context reject interop or stall ownership.** In the sealed seat, first register/map/unmap the dedicated textures and copy a colour canary. Inject failures at every boundary; require GLSL output and continued IPC responsiveness.
2. **Partial damage feeds history back or redraws everything.** Flash a patch, stop client commits, and count frames until settling. Keep normal damage tracking, no `keep-drawing`; compare a static neighboring window, resize, move, obscure and close the target.
3. **GPU fault containment fails.** Test recoverable API errors and unload/reload before any faulting kernel. A lost shared GPU may also lose GL; failure to preserve the seat blocks promotion.

Build with `native/build`'s `CRT_PASS_SOURCE` override and a separate output. Have the owner extend `native/rig/scene`, `native/rig/up`, `native/sandbox` and `CDump`/`native/rig/shot`, then reuse `native/rig/prove`'s overlay oracle and `native/rig/prove-owner`'s fallback checks. The sandbox removes live seat/input/Wayland access; `headless-gpu.cpp` supplies NVIDIA layouts and `nodump.c` suppresses crash notifications. Third-party window plugins were not checked.
