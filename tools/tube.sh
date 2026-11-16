#!/usr/bin/env bash
# tube.sh: one live tube, and a panel of sliders that changes it while it runs.
# Terminal footage loops through the CUDA renderer in real time into a window;
# tools/knobs.py serves the sliders at http://127.0.0.1:8787 and every move
# shows on the very next frame. q in the window ends everything.
#
#   tools/tube.sh                 the window + the panel (about ten minutes, then it closes)
#   tools/tube.sh --dry 2         two loops into nothing, no window, no panel: proves the pipeline
#   tools/tube.sh -- --tiles ...  anything after -- goes to bin/crt_render
set -uo pipefail
cd "$(dirname "$0")/.."
S=${SIZE:-1280x720} PORT=${PORT:-8787} K=out/knobs.json
mkdir -p out
[ -s out/src_typing.mkv ] || uv run -q --with pillow tools/fixture.py typing --size "$S" --fps 60 2>/dev/null |
    ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s "$S" -r 60 -i - -c:v libx264rgb -crf 0 -preset veryfast out/src_typing.mkv
dry=0 loops=${LOOPS:-50}
if [ "${1:-}" = "--dry" ]; then dry=1; loops=$(( ${2:-1} - 1 )); shift; shift || true; fi
[ "${1:-}" = "--" ] && shift
feed() { ffmpeg -loglevel error -re -stream_loop "$loops" -i out/src_typing.mkv -f rawvideo -pix_fmt rgba - 2>/dev/null |
             bin/crt_render . --size "$S" --fps 60 --path cuda2 --knobs "$K" "$@" 2>/dev/null; }
if [ $dry -eq 1 ]; then
    feed "$@" | ffmpeg -hide_banner -loglevel error -stats -f rawvideo -pix_fmt rgba -s "$S" -r 60 -i - -f null - 2>&1 | tr '\r' '\n' | tail -1
    exit
fi
panel=""
if [ -f tools/knobs.py ]; then
    python3 tools/knobs.py --dials tools/dials.json --presets tools/presets.json --file "$K" --port "$PORT" >/dev/null 2>&1 & panel=$!
    trap '[ -n "$panel" ] && kill "$panel" 2>/dev/null' EXIT
    echo "sliders: http://127.0.0.1:$PORT"
fi
feed "$@" | mpv --no-terminal --title="CUDA CRT, live tube" --force-window=immediate --untimed --no-cache --keep-open=no \
    --demuxer=rawvideo --demuxer-rawvideo-w="${S%x*}" --demuxer-rawvideo-h="${S#*x}" --demuxer-rawvideo-mp-format=rgba --demuxer-rawvideo-fps=60 -
