# CRT sliders and the one-window proof plan

The panel is built on `astra/knobs`, in the separate `cuda-crt-astra` clone. It reads dial names, groups, ranges, labels, units and help from JSON. The accompanying [Hyprland plan](PLAN-2026-11-15-hyprland-proof.md) is a paper design; no compositor or GPU program was started.

Run against the samples:

```bash
python3 tools/knobs.py --dials tools/samples/dials.json --presets tools/samples/presets.json --file /tmp/k.json
```

The panel serves at `http://127.0.0.1:8787/`. `--port` changes the port; zero chooses a free one. With no path arguments, it prefers `tools/dials.json` and `tools/presets.json`, otherwise uses the samples. Custom/real dials without a presets file get no preset buttons. Metadata loads at startup; restart after changing the dial schema.

Sliders queue a save 35 ms after the first input in a batch. Sustained dragging cannot keep postponing it. Requests run one at a time, with the newest pending values sent next. Browser scheduling and HTTP latency can exceed 50 ms; browser interaction and the 420 px layout were reasoned through, not visually checked. The script uses plain DOM methods and same-origin requests only.

Preset buttons merge their values into current settings; Reset restores every default. Ghost None sets `ghost` to zero and `ghost_decay` to its minimum; Short uses their defaults; Long uses their maxima. Missing ghost dials are skipped. The buttons disable when neither exists.

`GET /api/dials`, `/api/presets` and `/api/knobs` expose metadata and current state. `POST /api/knobs` accepts `{"values":{"curv":1.5},"preset":null}`. Partial sets are allowed; the file always contains every dial. Updates clear the preset unless a known preset name is supplied. Unknown names and non-finite/non-numeric values are refused; values clamp to limits. Host, Origin and JSON content-type checks protect writes. No CORS access is granted.

Each publication increments `rev`, including startup. Existing values and revision survive restart; retired names are dropped and new dials receive defaults. A same-directory temporary file is flushed, synced and replaced atomically. Use one panel process per output file; the renderer only reads it.

Validation:

```text
python3 tools/test_knobs.py
Ran 10 tests in 0.454s
OK
```

The tests include 200 HTTP writes with a concurrent file reader, revision sequencing, concurrent clients, metadata, restoration, clamping and rejected requests. Python parsing, JavaScript syntax, and a CLI launch with the sample arguments also passed; the CLI served HTTP 200 and a complete initial state, then stopped cleanly. No browser was opened.

No required JSON-contract changes. Optional future metadata could specify the preferred Short/Long ghost values; current shortcuts use the published defaults and limits. Difficulty remained 3/10.
