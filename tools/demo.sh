#!/usr/bin/env bash
# demo.sh: the renderer running LIVE, side by side, in one window.
# Terminal footage is fed in real time through two renderers at once and shown
# next to each other. Nothing is pre-rendered: close the window (q) and the
# pipeline stops. `demo.sh --check FILE.png` renders one frame of act 1 to a
# file and `demo.sh --dry` runs every act into nothing, both without a window.
#
#   tools/demo.sh            three passes of the four acts, about two minutes
#   LOOPS=1 tools/demo.sh    one pass
set -uo pipefail
cd "$(dirname "$0")/.."
S=960x540
FONT=/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf
mkdir -p out
for scene in typing flash tiles; do  # the footage, once, at the demo's size
    [ -s "out/demo_$scene.mkv" ] || uv run -q --with pillow tools/fixture.py "$scene" --size $S --fps 60 2>"out/demo_$scene.txt" |
        ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s $S -r 60 -i - -c:v libx264rgb -crf 0 -preset veryfast "out/demo_$scene.mkv"
done
TILES=$(sed -n 's/^tiles: //p' out/demo_tiles.txt)

# act <footage> <left label> <left renderer args> <right label> <right renderer args> [repeats]
act() {
    local src=$1 ll=$2 la=$3 rl=$4 ra=$5 reps=${6:-0} d
    d=$(mktemp -d); mkfifo "$d/i1" "$d/i2" "$d/a" "$d/b"
    # One decode, paced at real time, written to both renderers. Each renderer's
    # output fifo is opened read-write so neither waits on the other's reader.
    ffmpeg -loglevel error -re -stream_loop "$reps" -i "out/demo_$src.mkv" \
        -f rawvideo -pix_fmt rgba -y "$d/i1" -f rawvideo -pix_fmt rgba -y "$d/i2" 2>/dev/null & local p1=$!
    bin/crt_render . --size $S --fps 60 $la <"$d/i1" 1<>"$d/a" 2>/dev/null & local p2=$!
    bin/crt_render . --size $S --fps 60 $ra <"$d/i2" 1<>"$d/b" 2>/dev/null & local p3=$!
    local T="fontfile=$FONT:fontsize=20:fontcolor=0x8fe8b0:y=12" rc
    ffmpeg -loglevel error -probesize 32 -analyzeduration 0 -f rawvideo -pix_fmt rgba -s $S -r 60 -i "$d/a" \
        -probesize 32 -analyzeduration 0 -f rawvideo -pix_fmt rgba -s $S -r 60 -i "$d/b" \
        -filter_complex "[0][1]hstack=inputs=2,pad=iw:ih+44:0:44:color=0x0b0f0c,drawtext=$T:x=20:text='$ll',drawtext=$T:x=w/2+20:text='$rl'" \
        -f rawvideo -pix_fmt rgba - 2>/dev/null & local ph=$!
    # If a renderer dies before writing (or both finish and the join still sits
    # there), stop the join two seconds later instead of waiting on it forever.
    ( while kill -0 "$p2" 2>/dev/null || kill -0 "$p3" 2>/dev/null; do sleep 0.5; done; sleep 2; kill -9 "$ph" 2>/dev/null ) & local pw=$!
    wait "$ph"; rc=$?
    kill "$pw" 2>/dev/null
    # The window closing ends the ffmpeg above; the three feeders are then stopped
    # by pid, because a renderer blocked on a full fifo never sees the pipe close.
    kill "$p1" "$p2" "$p3" 2>/dev/null
    wait "$p1" "$p2" "$p3" 2>/dev/null
    rm -rf "$d"
    return $rc
}
show() {  # stops at the first act whose window has gone
    act typing "SHIPPED GLASS  ·  no memory" "--path existing" "CUDA PHOSPHOR  ·  fades in 50 ms" "--path cuda --decay 50" &&
    act typing "CUDA  ·  fades in 50 ms" "--path cuda --decay 50" "CUDA  ·  green lingers 180 ms" "--path cuda --decay 60,180,90" &&
    act flash "SHIPPED GLASS  ·  a 200 ms flash" "--path existing --set tracking=0,flicker=0" "CUDA PHOSPHOR  ·  the same flash" "--path cuda --set tracking=0,flicker=0" 2 &&
    act tiles "WHAT THE APPS DREW" "--path pass" "CUDA  ·  one phosphor per tile" "--path cuda --tiles $TILES" 1
}
if [ "${1:-}" = "--check" ]; then
    act typing "SHIPPED GLASS  ·  no memory" "--path existing" "CUDA PHOSPHOR  ·  fades in 50 ms" "--path cuda --decay 50" |
        ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s 1920x584 -r 60 -i - -vf "select='eq(n,560)'" -frames:v 1 "$2"
    exit
fi
if [ "${1:-}" = "--dry" ]; then  # every act, no window: count the frames that come out
    show | ffmpeg -hide_banner -loglevel error -stats -f rawvideo -pix_fmt rgba -s 1920x584 -r 60 -i - -f null - 2>&1 | tr '\r' '\n' | tail -1
    exit
fi
for _ in $(seq "${LOOPS:-3}"); do show || break; done |
    mpv --no-terminal --title="CUDA CRT, live" --force-window=immediate --untimed --no-cache --keep-open=no \
        --demuxer=rawvideo --demuxer-rawvideo-w=1920 --demuxer-rawvideo-h=584 --demuxer-rawvideo-mp-format=rgba --demuxer-rawvideo-fps=60 -
