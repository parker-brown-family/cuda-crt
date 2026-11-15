#!/usr/bin/env bash
# Do the Stage 1 tests have teeth? Break the kernel four ways, one at a time, in
# a scratch copy, and require the suite to fail each time. The real kernel is
# never edited. Exit 0 only if every mutant is caught.
set -uo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/kernels" "$tmp/out"
cp phosphor_params.h "$tmp/"
missed=0
mutant() {  # name, sed expression
    sed -E "$2" kernels/phosphor.cu > "$tmp/kernels/phosphor.cu"
    if cmp -s kernels/phosphor.cu "$tmp/kernels/phosphor.cu"; then echo "  BROKEN MUTANT  $1 (the edit matched nothing)"; missed=1; return; fi
    if bin/stage1_test "$tmp" > "$tmp/log" 2>&1; then echo "  MISSED  $1"; missed=1
    else echo "  caught  $1  ($(grep -c FAIL "$tmp/log") checks failed: $(grep FAIL "$tmp/log" | head -1 | sed -E 's/ +!.*//; s/^ +FAIL +//'))"; fi
}
mutant "rise and decay swapped"            's/float a = target > prev \? a_rise : a_decay;/float a = target > prev ? a_decay : a_rise;/'
mutant "tile origin ignored"               's/\(size_t\)\(oy \+ y\) \* frame_w \+ \(ox \+ x\)/(size_t)(y) * frame_w + (x)/'
mutant "never settles (no snap)"           's/if \(fabsf\(target - next\) < eps\) return target;/if (false) return target;/'
mutant "decay applied to gamma-encoded values" 's/srgb_to_linear\(px\[c\] \* \(1\.0f \/ 255\.0f\)\)/(px[c] * (1.0f \/ 255.0f))/'
exit $missed
