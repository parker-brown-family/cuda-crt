#!/usr/bin/env python3
"""Turn results.csv into the tables decision.md quotes, so no number is retyped.

    python3 tools/summarize.py            markdown to stdout
    python3 tools/summarize.py --json     the same numbers as JSON (for the brief's charts)

Overhead = a path's time minus the shipped glass shader's time at the same size
and pacing: what adding a phosphor surface costs on top of what runs today.
The gate is the handoff's proposal: p95 GPU overhead of 1.0 ms or less.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_MS = 1.0


def load():
    rows = {}
    with open(os.path.join(HERE, "results.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["path"], int(r["width"]), r["pacing"])] = r
    return rows


def main():
    rows = load()
    sizes = [(960, 540), (1920, 1080), (3840, 2160)]
    out = {"gate_ms": GATE_MS, "sizes": []}
    for w, h in sizes:
        entry = {"size": "%dx%d" % (w, h), "pacing": {}}
        for pacing in ("back_to_back", "paced_60hz", "paced_144hz"):
            base = rows.get(("existing", w, pacing))
            if not base:
                continue
            block = {}
            for path in ("existing", "glsl_update", "glsl_full_noprobe", "glsl_full", "glsl_full_16f", "cuda_update", "cuda_full"):
                r = rows.get((path, w, pacing))
                if not r:
                    continue
                g95, t95 = float(r["gpu_ms_p95"]), float(r["total_ms_p95"])
                block[path] = {
                    "gpu_p50": float(r["gpu_ms_p50"]), "gpu_p95": g95, "gpu_p99": float(r["gpu_ms_p99"]),
                    "total_p50": float(r["total_ms_p50"]), "total_p95": t95, "total_p99": float(r["total_ms_p99"]),
                    "cpu_p50": float(r["cpu_ms_p50"]), "vram_mb": float(r["state_vram_mb"]),
                    "gpu_overhead_p95": round(g95 - float(base["gpu_ms_p95"]), 4),
                    "total_overhead_p50": round(float(r["total_ms_p50"]) - float(base["total_ms_p50"]), 4),
                }
            entry["pacing"][pacing] = block
        out["sizes"].append(entry)
    if "--json" in sys.argv:
        json.dump(out, sys.stdout, indent=1)
        return
    for entry in out["sizes"]:
        for pacing, block in entry["pacing"].items():
            print("\n### %s, %s\n" % (entry["size"], pacing.replace("_", " ")))
            print("| path | GPU p50 | GPU p95 | GPU p99 | total p50 | total p95 | CPU p50 | state VRAM | GPU p95 over the shipped glass | gate (1.0 ms) |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for path, v in block.items():
                gate = "baseline" if path == "existing" else ("pass" if v["gpu_overhead_p95"] <= GATE_MS else "FAIL")
                print("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f MB | %s | %s |" % (
                    path, v["gpu_p50"], v["gpu_p95"], v["gpu_p99"], v["total_p50"], v["total_p95"], v["cpu_p50"], v["vram_mb"],
                    "" if path == "existing" else "%+.3f" % v["gpu_overhead_p95"], gate))


if __name__ == "__main__":
    main()
