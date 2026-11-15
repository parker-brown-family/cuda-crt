#!/usr/bin/env bash
# Do the Stage 1 tests have teeth? Break the kernel seven ways, one at a time, in
# a scratch copy, and require the suite to fail each time. The real kernel is
# never edited. Exit 0 only if every mutant is caught.
set -uo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/kernels" "$tmp/out"
cp phosphor_params.h phosphor_half.h "$tmp/"
missed=0
mutant() {  # name, sed expression, [file the edit applies to: the kernel, unless named]
    local file=${3:-kernels/phosphor.cu}
    cp kernels/phosphor.cu "$tmp/kernels/phosphor.cu"; cp phosphor_half.h "$tmp/phosphor_half.h"
    sed -E "$2" "$file" > "$tmp/$file"
    if cmp -s "$file" "$tmp/$file"; then echo "  BROKEN MUTANT  $1 (the edit matched nothing)"; missed=1; return; fi
    bin/stage1_test "$tmp" > "$tmp/log" 2>&1
    local rc=$? n
    n=$(grep -c "^  FAIL" "$tmp/log")
    # Only exit 1 with named failing checks counts. A mutant that does not compile, or a
    # suite that crashes, is the harness failing, and reading that as "caught" would be a lie.
    if [ $rc -eq 0 ]; then echo "  MISSED  $1"; missed=1
    elif [ $rc -ne 1 ] || [ "$n" -eq 0 ]; then echo "  BROKEN HARNESS  $1 (exit $rc, $n failing checks): $(tail -1 "$tmp/log" | cut -c1-90)"; missed=1
    else echo "  caught  $1  ($n checks failed, first: $(grep "^  FAIL" "$tmp/log" | head -1 | sed -E 's/ +!.*//; s/^ +FAIL +//'))"; fi
}
mutant "rise and decay swapped"            's/float a = target > prev \? a_rise : a_decay;/float a = target > prev ? a_decay : a_rise;/'
mutant "tile origin ignored"               's/\(size_t\)\(oy \+ y\) \* frame_w \+ \(ox \+ x\)/(size_t)(y) * frame_w + (x)/'
mutant "never settles (no snap)"           's/if \(!\(fabsf\(target - next\) >= eps\)\) return target;  \/\/ settled/if (false) return target;  \/\/ settled/'
mutant "half state: stall rule removed"    's/ \|\| hn == hprev\) return ht;/) return ht;/'
mutant "half state: rounds down, not to nearest" 's/if \(rem > half \|\| \(rem == half && \(r & 1u\)\)\) r\+\+;/if (false) r++;/' phosphor_half.h
mutant "burn never drains"                 's/s\.w = phosphor_step_h\(s\.w, lum, p\.a_burn_rise, p\.a_burn_decay,/s.w = phosphor_step_h(s.w, lum, p.a_burn_rise, 0.0f,/'
mutant "decay applied to gamma-encoded values" 's/srgb_to_linear\(px\[c\] \* \(1\.0f \/ 255\.0f\)\)/(px[c] * (1.0f \/ 255.0f))/'
exit $missed
