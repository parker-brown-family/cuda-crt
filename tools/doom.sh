#!/usr/bin/env bash
# Launch the real game. The isolated CUDA proof is hypr/prove-doom.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" == --crt ]]; then
  shift
  exec python3 "$root/tools/doom-crt.py" "$@"
fi
engine="${DOOM_ENGINE:-$root/.toolchain/crispy-doom/build/src/crispy-doom}"
if [[ -z "${DOOM_ENGINE:-}" && ! -x "$engine" ]]; then
  engine="$root/.toolchain/chocolate-doom/build/src/chocolate-doom"
fi
wad="${DOOM_WAD:-$root/.toolchain/doom1.wad}"
if [[ ! -x "$engine" || ! -f "$wad" ]]; then
  echo 'Set DOOM_ENGINE to Crispy Doom or Chocolate Doom and DOOM_WAD to your IWAD. See README.md.' >&2
  exit 2
fi
profile=()
if [[ "$(basename "$engine")" == crispy-doom ]]; then
  state="${XDG_STATE_HOME:-$HOME/.local/state}/cuda-crt/doom"
  mkdir -p "$state"
  # Seed once: the game can save subsequent sensitivity and menu adjustments.
  [[ -f "$state/default.cfg" ]] || cp "$root/tools/doom-controls.cfg" "$state/default.cfg"
  [[ -f "$state/fps.cfg" ]] || cp "$root/tools/doom-fps.cfg" "$state/fps.cfg"
  profile=(-config "$state/default.cfg" -extraconfig "$state/fps.cfg" -savedir "$state")
fi
exec "$engine" -iwad "$wad" -window -width 960 -height 720 -nosound "${profile[@]}" "$@"
