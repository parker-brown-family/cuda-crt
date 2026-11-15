#!/usr/bin/env python3
"""Draw the review page's measured figures from the lab's own output files, so a
bar's length is computed from results.csv and never typed.

    python3 tools/brief_figures.py cost       the GPU-time chart (inline SVG)
    python3 tools/brief_figures.py handoff    where a CUDA frame's wall time goes (inline SVG)
    python3 tools/brief_figures.py clock      pass time and GPU clock over 40 s at 60 Hz (inline SVG)
    python3 tools/brief_figures.py low_clock  the settled low-clock numbers as JSON
    python3 tools/brief_figures.py img PATH   a JPEG as a data URI, so the page carries its stills

Colours are the house classes: struct (blue) what ships today, ready (green) the
shader phosphor, mine (amber) the CUDA phosphor, crit (red) the gate.
"""
import base64
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summary():
    out = subprocess.run([sys.executable, os.path.join(HERE, "tools", "summarize.py"), "--json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def cost(pacing="back_to_back"):
    s = summary()
    paths = [("existing", "struct", "shipped glass"), ("glsl_full", "ready", "shader phosphor"), ("cuda_full", "mine", "CUDA phosphor")]
    W, left, top, rowh, gap = 760, 150, 34, 22, 20
    peak = max(max(e["pacing"][pacing][p]["gpu_p95"] for p, _, _ in paths) for e in s["sizes"])
    peak = max(peak, max(e["pacing"][pacing]["existing"]["gpu_p95"] + s["gate_ms"] for e in s["sizes"]))
    scale = (W - left - 190) / peak
    H = top + len(s["sizes"]) * (3 * rowh + gap) + 30
    o = ['<svg viewBox="0 0 %d %d" role="img" aria-label="GPU time per frame at three sizes for the shipped glass, the shader phosphor and the CUDA phosphor, with the one millisecond gate drawn on each">' % (W, H)]
    y = top
    for e in s["sizes"]:
        block = e["pacing"][pacing]
        o.append('<text class="ink" x="8" y="%d" font-size="15" font-weight="700">%s</text>' % (y + rowh + 6, e["size"].replace("x", " × ")))
        o.append('<text class="ink2" x="8" y="%d" font-size="11">%.1f M pixels</text>' % (y + rowh + 22, eval(e["size"].replace("x", "*")) / 1e6))
        gate_x = left + (block["existing"]["gpu_p95"] + s["gate_ms"]) * scale
        for i, (p, cls, label) in enumerate(paths):
            v = block[p]["gpu_p95"]
            yy = y + i * rowh
            o.append('<rect class="%s" x="%d" y="%d" width="%.1f" height="%d" rx="3"/>' % (cls, left, yy + 2, max(v * scale, 2), rowh - 5))
            o.append('<text class="ink" x="%.1f" y="%d" font-size="12">%.2f ms  %s</text>' % (left + v * scale + 8, yy + rowh - 7, v, label))
        o.append('<line class="crit" x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke-width="2" stroke-dasharray="5 4"/>' % (gate_x, y - 4, gate_x, y + 3 * rowh + 2))
        o.append('<text class="crit" x="%.1f" y="%d" font-size="11" text-anchor="middle">gate</text>' % (gate_x, y - 8))
        y += 3 * rowh + gap
    o.append('<text class="ink2" x="%d" y="%d" font-size="11">GPU time in the pass, 95th percentile, GPU at full clock. Gate = shipped glass + 1.0 ms.</text>' % (left, H - 8))
    o.append("</svg>")
    return "\n".join(o)


def handoff():
    rows = list(csv.DictReader(open(os.path.join(HERE, "out", "cuda_breakdown.csv"))))
    W, left, rowh = 760, 150, 34
    peak = max(float(r["map_ms_p50"]) + float(r["kernels_synced_ms_p50"]) + float(r["unmap_present_ms_p50"]) for r in rows)
    scale = (W - left - 70) / peak
    H = 30 + len(rows) * (rowh + 14) + 44
    o = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Where a CUDA frame&#39;s wall time goes at three sizes: taking the textures from OpenGL, running the kernels, and handing them back">' % (W, H)]
    y = 26
    for r in rows:
        parts = [(float(r["map_ms_p50"]), "unk"), (float(r["kernels_synced_ms_p50"]), "mine"), (float(r["unmap_present_ms_p50"]), "unk")]
        o.append('<text class="ink" x="8" y="%d" font-size="15" font-weight="700">%s × %s</text>' % (y + 22, r["width"], r["height"]))
        x = left
        for v, cls in parts:
            w = v * scale
            o.append('<rect class="%s" x="%.1f" y="%d" width="%.1f" height="%d" rx="3"/>' % (cls, x, y, max(w - 2, 2), rowh))
            if w > 44:
                o.append('<text class="ink" x="%.1f" y="%d" font-size="12" text-anchor="middle">%.2f</text>' % (x + w / 2, y + 22, v))
            x += w
        o.append('<text class="ink2" x="%.1f" y="%d" font-size="12">%.2f ms</text>' % (x + 6, y + 22, sum(v for v, _ in parts)))
        y += rowh + 14
    o.append('<rect class="unk" x="%d" y="%d" width="14" height="12" rx="2"/><text class="ink2" x="%d" y="%d" font-size="12">take the textures from OpenGL · hand them back and show the frame</text>' % (left, y + 4, left + 20, y + 15))
    o.append('<rect class="mine" x="%d" y="%d" width="14" height="12" rx="2"/><text class="ink2" x="%d" y="%d" font-size="12">the two kernels (phosphor, then glass)</text>' % (left, y + 22, left + 20, y + 33))
    o.append("</svg>")
    return "\n".join(o)


def clock():
    """Pass time and GPU clock, second by second, over 40 s of 60 Hz redraw at 1080p."""
    tr, ck = {}, {}
    for r in csv.DictReader(open(os.path.join(HERE, "out", "clock_trace.csv"))):
        tr.setdefault(r["path"], []).append(float(r["gpu_ms_p50"]))
    for r in csv.DictReader(open(os.path.join(HERE, "out", "clock_samples.csv"))):
        ck.setdefault(r["path"], []).append((float(r["clock_mhz"]), float(r["power_w"])))
    paths = [("existing", "struct", "shipped glass"), ("glsl_full", "ready", "shader phosphor"), ("cuda_full", "mine", "CUDA phosphor")]
    W, left, right = 760, 56, 190
    pw = W - left - right
    n = 40
    X = lambda i: left + i * pw / (n - 1)

    def panel(y0, h, peak, series, unit, title, fmt):
        o = ['<text class="ink" x="%d" y="%d" font-size="13" font-weight="700">%s</text>' % (left, y0 - 8, title)]
        o.append('<line class="wire" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (left, y0 + h, left + pw, y0 + h))
        o.append('<text class="ink2" x="%d" y="%d" font-size="11" text-anchor="end">0</text>' % (left - 6, y0 + h + 4))
        o.append('<text class="ink2" x="%d" y="%d" font-size="11" text-anchor="end">%s</text>' % (left - 6, y0 + 4, fmt % peak))
        ends = []
        for (p, cls, label), vals in series:
            pts = " ".join("%.1f,%.1f" % (X(i), y0 + h - min(v, peak) / peak * h) for i, v in enumerate(vals[:n]))
            o.append('<polyline class="%s" points="%s" fill="none" stroke-width="2.5" style="fill:none"/>' % (cls, pts))
            last = sorted(vals[20:n])[len(vals[20:n]) // 2]
            ends.append((y0 + h - min(last, peak) / peak * h, cls, "%s  %s" % (fmt % last + unit, label)))
        ends.sort()
        yy = -99
        for y, cls, text in ends:  # keep the end labels from landing on each other
            yy = max(y + 4, yy + 15)
            o.append('<text class="%s" x="%d" y="%.1f" font-size="12">%s</text>' % (cls, left + pw + 8, yy, text))
        return o

    o = ['<svg viewBox="0 0 %d 400" role="img" aria-label="Second by second for forty seconds: the time each path takes per frame, and the GPU clock underneath. After sixteen seconds the two shader paths slow down as the GPU clock drops; the CUDA path stays flat because it holds the clock up">' % W]
    o += panel(34, 140, 4.0, [((p, c, l), tr[p]) for p, c, l in paths], " ms", "time in the pass, each second's median (spikes above 4 ms clipped)", "%.2f")
    o += panel(226, 110, 1600, [((p, c, l), [v[0] for v in ck[p]][1:]) for p, c, l in paths], " MHz", "GPU clock", "%.0f")
    for sec in (0, 10, 16, 20, 30, 39):
        o.append('<text class="ink2" x="%.1f" y="356" font-size="11" text-anchor="middle">%d s</text>' % (X(sec), sec))
    o.append('<line class="unk" x1="%.1f" y1="26" x2="%.1f" y2="338" stroke-width="1.5"/>' % (X(16), X(16)))
    pw_txt = "  ·  ".join("%s %.0f W" % (l, sorted(v[1] for v in ck[p][21:40])[9]) for p, c, l in paths)
    o.append('<text class="ink2" x="%d" y="384" font-size="12">GPU power after second 20, whole card, desktop included:  %s</text>' % (left, pw_txt))
    o.append("</svg>")
    return "\n".join(o)


def low_clock():
    """The settled low-gear numbers (median of seconds 20 to 39), as JSON."""
    tr, ck = {}, {}
    for r in csv.DictReader(open(os.path.join(HERE, "out", "clock_trace.csv"))):
        tr.setdefault(r["path"], []).append(float(r["gpu_ms_p50"]))
    for r in csv.DictReader(open(os.path.join(HERE, "out", "clock_samples.csv"))):
        ck.setdefault(r["path"], []).append((float(r["clock_mhz"]), float(r["power_w"])))
    med = lambda v: sorted(v)[len(v) // 2]
    return json.dumps({p: {"first_15s_ms": med(tr[p][:15]), "after_20s_ms": med(tr[p][20:40]), "after_20s_mhz": med([c for c, _ in ck[p][21:41]]),
                           "after_20s_watts": med([w for _, w in ck[p][21:41]]), "first_15s_watts": med([w for _, w in ck[p][3:16]])} for p in tr}, indent=1)


def img(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


if __name__ == "__main__":
    what = sys.argv[1]
    print({"cost": cost, "handoff": handoff, "clock": clock, "low_clock": low_clock}[what]() if what != "img" else img(sys.argv[2]))
