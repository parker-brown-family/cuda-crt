#!/usr/bin/env python3
"""Own one playable DOOM session and restore the preceding CUDA state on exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windowed", action="store_true")
    options, game_args = parser.parse_known_args()
    env = dict(os.environ)
    if not env.get("HYPRLAND_INSTANCE_SIGNATURE"):
        instances = json.loads(subprocess.check_output(["hyprctl", "instances", "-j"], env=env))
        if len(instances) != 1:
            raise RuntimeError("Select your compositor with HYPRLAND_INSTANCE_SIGNATURE before launching DOOM.")
        env["HYPRLAND_INSTANCE_SIGNATURE"] = instances[0]["instance"]
        env["WAYLAND_DISPLAY"] = instances[0]["wl_socket"]
    env.setdefault("SDL_VIDEODRIVER", "wayland")
    state = Path(env.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "cuda-crt"
    state.mkdir(parents=True, exist_ok=True)
    lock = open(state / "doom-session.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A DOOM CRT session is already running.") from None

    def ctl(*args):
        p = subprocess.run(["hyprctl", *args], env=env, capture_output=True, text=True, timeout=10)
        if p.returncode:
            raise RuntimeError("hyprctl: " + p.stdout + p.stderr)
        return p.stdout

    def put(name, value):
        p = state / (".doom-" + name)
        p.write_bytes(value)
        p.replace(state / name)

    built = ROOT / "bin/cuda-pass-doom.so"
    if not built.is_file():
        raise RuntimeError("Run make doom-setup first to build the game and CUDA plugin.")
    subprocess.run(["make", "bin/cuda-pass-doom.so"], cwd=ROOT, check=True)
    previous = {n: (state / n).read_bytes() if (state / n).exists() else None
                for n in ("root", "target", "look", "set", "ttl", "off", "loaded")}
    loaded = any(p["name"] == "cuda-pass" for p in json.loads(ctl("plugin", "list", "-j")))
    old_plugin = previous["loaded"].decode().strip() if previous["loaded"] else None
    if loaded and (not old_plugin or not Path(old_plugin).is_file()):
        raise RuntimeError("The existing CUDA plugin has no recoverable loaded path; record its path in cuda-crt/loaded first.")
    plugin = state / ("cuda-pass-doom-" + str(os.getpid()) + ".so")
    shutil.copyfile(built, plugin)
    game = None
    installed = old_removed = False
    try:
        if loaded:
            ctl("plugin", "unload", old_plugin)
            old_removed = True
        put("root", (str(ROOT) + "\n").encode())
        put("look", b"tube\n")
        put("set", b"halation=0.08,glow=0.15,beam=0.2,glow_knee=0.2\n")
        for name in ("off", "ttl"):
            (state / name).unlink(missing_ok=True)
        args = [str(ROOT / "tools/doom.sh"), "-nogui"]
        if not options.windowed:
            args.append("-fullscreen")
        args += game_args or ["-warp", "1", "1"]
        existing = {w["address"] for w in json.loads(ctl("clients", "-j"))}
        game = subprocess.Popen(args, cwd=ROOT, env=env)
        win = None
        for _ in range(100):
            if game.poll() is not None:
                raise RuntimeError("DOOM exited before opening a window.")
            # A client in the sealed seat's sibling PID namespace reports pid 0.
            # In that case identify its newly opened DOOM surface, not another app.
            win = next((w for w in json.loads(ctl("clients", "-j"))
                        if w["address"] not in existing and
                        (w["pid"] == game.pid or (w["pid"] == 0 and w["class"] in ("crispy-doom", "chocolate-doom")))), None)
            if win:
                break
            time.sleep(0.1)
        if not win:
            raise RuntimeError("DOOM did not open a window within ten seconds.")
        # A class selector survives fullscreen remapping and SDL window recreation.
        put("target", ("class:" + win["class"] + "\n").encode())
        ctl("plugin", "load", str(plugin))
        installed = True
        put("loaded", (str(plugin) + "\n").encode())
        ctl("dispatch", 'hl.dsp.focus({window="address:' + win["address"] + '"})')
        if not options.windowed:
            ctl("dispatch", 'hl.dsp.window.fullscreen({window="address:' + win["address"] + '",mode="fullscreen",action="set"})')
        print("DOOM + CUDA CRT: WASD/arrows move, mouse aims, left-click fires, E uses, 1-7 selects weapons.", flush=True)
        return game.wait()
    finally:
        if game is not None and game.poll() is None:
            game.terminate()
            try:
                game.wait(timeout=5)
            except subprocess.TimeoutExpired:
                game.kill(); game.wait()
        if installed:
            ctl("plugin", "unload", str(plugin))
        for name, value in previous.items():
            if value is None:
                (state / name).unlink(missing_ok=True)
            else:
                put(name, value)
        if old_removed:
            ctl("plugin", "load", old_plugin)
        plugin.unlink(missing_ok=True)


if __name__ == "__main__":
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print("doom-crt:", error, file=sys.stderr)
        sys.exit(1)
