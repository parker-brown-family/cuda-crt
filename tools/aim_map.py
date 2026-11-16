#!/usr/bin/env python3
"""How far does the glass move things, and how much of the window does the rim cover?

    python3 tools/aim_map.py [W H]        a table for a window of W x H logical pixels (default 1181 950)
    python3 tools/aim_map.py --json W H   the same as JSON, plus a coarse grid for drawing

It evaluates the SAME formula as tube_map.h (kept in step by tools/test_aim_map.py,
which compiles the header and compares). "Moved" is the distance between where a
thing is drawn and where it really is: the reach a hand has to make up when nothing
maps the pointer, and the size of the miss when the picture is bent and clicks are not.
"""
import json, math, sys


def gather(k1, k2, core, inset, lx, ly):
    cx, cy = lx - 0.5, ly - 0.5
    r2 = cx * cx + cy * cy
    c2 = core * 0.45
    e = (r2 - c2) / (0.5 - c2) if r2 > c2 else 0.0
    q = 0.5 * e ** (1.0 + 2.0 * core)
    f = (1.0 + k1 * q + k2 * q * q) * (1.0 + inset)
    return 0.5 + cx * f, 0.5 + cy * f


def moved_px(shape, W, H, lx, ly):
    """Pixels between a screen point and the picture point shown there, with the
    uniform shrink of an inset taken out (a shrink moves things but does not bend them)."""
    k1, k2, core, inset = shape
    sx, sy = gather(k1, k2, core, inset, lx, ly)
    fx, fy = 0.5 + (lx - 0.5) * (1 + inset), 0.5 + (ly - 0.5) * (1 + inset)
    return math.hypot((sx - fx) * W, (sy - fy) * H)


def covered(shape, W, H, n=240):
    """Share of the WINDOW's own area that is under the rim's shadow or cut by a rounded corner."""
    k1, k2, core, inset = shape
    short = min(W, H)
    corner = min(max(short * 0.055, 10.0), 72.0)
    lip = min(max(short * 0.045, 8.0), 48.0)
    pad = inset * 0.5
    hit = tot = 0
    for j in range(n):
        for i in range(n):
            wx, wy = (i + 0.5) / n, (j + 0.5) / n               # a point of the window's picture
            gx, gy = (wx + pad) / (1 + 2 * pad), (wy + pad) / (1 + 2 * pad)   # where that is on the glass
            qx = abs(gx * W - W / 2) - W / 2 + corner
            qy = abs(gy * H - H / 2) - H / 2 + corner
            sd = min(max(qx, qy), 0.0) + math.hypot(max(qx, 0.0), max(qy, 0.0)) - corner
            tot += 1
            hit += 1 if sd > -lip * 0.5 else 0                   # outside the glass, or in the darker half of the shadow
    return hit / tot


SHAPES = {"classic barrel": (0.14, 0.06, 0.0, 0.0), "flat middle (core 0.6)": (0.14, 0.06, 0.6, 0.0),
          "flat middle + safe edge (core 0.6, inset 0.05)": (0.14, 0.06, 0.6, 0.05)}


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    W, H = (float(args[0]), float(args[1])) if len(args) >= 2 else (1181.0, 950.0)
    out = {"window": [W, H], "shapes": {}}
    n = 200
    for name, shape in SHAPES.items():
        vals = [moved_px(shape, W, H, (i + 0.5) / n, (j + 0.5) / n) for j in range(n) for i in range(n)]
        out["shapes"][name] = {
            "centre_px": moved_px(shape, W, H, 0.5, 0.5), "mid_side_px": moved_px(shape, W, H, 0.98, 0.5), "mid_top_px": moved_px(shape, W, H, 0.5, 0.02),
            "corner_px": moved_px(shape, W, H, 0.98, 0.02), "worst_px": max(vals),
            "share_within_4px": sum(1 for v in vals if v <= 4.0) / len(vals),
            "share_within_12px": sum(1 for v in vals if v <= 12.0) / len(vals),
            "window_covered_by_rim": covered(shape, W, H),
            "grid": [[round(moved_px(shape, W, H, (i + 0.5) / 24, (j + 0.5) / 16), 1) for i in range(24)] for j in range(16)],
        }
    if "--json" in sys.argv:
        json.dump(out, sys.stdout)
        return
    print("window %d x %d logical px; 'moved' = how far a thing is drawn from where it really is" % (W, H))
    print("| curve | centre | mid side | corner | worst | window within 4 px | within 12 px | window under the rim |")
    print("|---|---|---|---|---|---|---|---|")
    for name, v in out["shapes"].items():
        print("| %s | %.0f px | %.0f px | %.0f px | %.0f px | %.0f%% | %.0f%% | %.1f%% |" % (name, v["centre_px"], v["mid_side_px"], v["corner_px"], v["worst_px"],
              100 * v["share_within_4px"], 100 * v["share_within_12px"], 100 * v["window_covered_by_rim"]))


if __name__ == "__main__":
    main()
