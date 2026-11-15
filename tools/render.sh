#!/usr/bin/env bash
# render.sh <fixture> <path> <out-name> [crt_render options...]
# One lossless source clip through one path, out as an MP4 in out/.
set -euo pipefail
cd "$(dirname "$0")/.."
src=$1 path=$2 out=$3; shift 3
size=${SIZE:-1280x720} fps=${FPS:-60}
ffmpeg -loglevel error -i "out/src_$src.mkv" -f rawvideo -pix_fmt rgba - |
  bin/crt_render . --size "$size" --fps "$fps" --path "$path" "$@" |
  ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s "$size" -r "$fps" -i - \
    -c:v libx264 -crf 15 -pix_fmt yuv420p -movflags +faststart "out/$out.mp4"
