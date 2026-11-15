#!/usr/bin/env bash
# The benchmark as it should be run: record GPU conditions either side, then summarise.
cd "$(dirname "$0")/.."
cond() { nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader; cat /proc/loadavg; }
cond > out/gpu_conditions_before.txt 2>&1
bin/stage2_bench . > out/bench.log 2>&1
echo "bench exit $?" >> out/bench.log
cond > out/gpu_conditions_after.txt 2>&1
tail -1 out/bench.log
python3 tools/summarize.py | grep -E "###|existing|glsl_full |cuda_full|cuda_v2"
cat out/block_sweep.csv out/cuda_breakdown.csv out/glsl_vs_cuda_image.csv out/gpu_conditions_before.txt out/gpu_conditions_after.txt
