#!/usr/bin/env python3
"""tools/aim_map.py re-types the tube's formula in Python. This compiles tube_map.h
itself and checks the two agree, so the table cannot drift from the code that draws."""
import os, random, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import aim_map
src = '#include <stdio.h>\n#include "tube_map.h"\nint main(void){TubeMap m; float x,y,sx,sy; while(scanf("%f %f %f %f %f %f",&m.k1,&m.k2,&m.core,&m.inset,&x,&y)==6){tube_gather(m,x,y,&sx,&sy); printf("%.9g %.9g\\n",sx,sy);} return 0;}\n'
with tempfile.TemporaryDirectory() as d:
    open(d + "/t.c", "w").write(src)
    subprocess.run(["gcc", "-O1", "-I" + HERE, d + "/t.c", "-o", d + "/t", "-lm"], check=True)
    random.seed(7)
    cases = [(random.uniform(0, 0.3), random.uniform(0, 0.15), random.uniform(0, 0.95), random.uniform(0, 0.12), random.random(), random.random()) for _ in range(4000)]
    cases += [(0.14, 0.06, 0.0, 0.0, x, y) for x in (0, 0.5, 1) for y in (0, 0.5, 1)]
    out = subprocess.run([d + "/t"], input="\n".join(" ".join("%.9g" % v for v in c) for c in cases), capture_output=True, text=True, check=True).stdout.split("\n")
worst = 0.0
for c, line in zip(cases, out):
    sx, sy = (float(v) for v in line.split())
    px, py = aim_map.gather(*c)
    worst = max(worst, abs(sx - px), abs(sy - py))
ok = worst < 2e-5
print("%s  tube_map.h and aim_map.py agree on %d points: largest difference %.1e of the tile" % ("ok  " if ok else "FAIL", len(cases), worst))
sys.exit(0 if ok else 1)
