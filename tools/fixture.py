#!/usr/bin/env python3
"""Deterministic terminal footage for the phosphor surface, as raw RGBA frames on stdout.

    uv run --with pillow tools/fixture.py typing --size 1280x720 --fps 60 | bin/crt_render . --size 1280x720 ...

Scenes
  typing   a prompt types a command with a blinking cursor, output scrolls, the screen clears
  flash    a white square, then a green character, each lit for 200 ms on black (test 1)
  tiles    two terminals side by side on a desk; the left one flashes, the right one must not care
  hold     one still screen for 2 s (idle: nothing should move after the first frames)

Nothing here is random: the same arguments give the same bytes.
"""
import argparse
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"
BG = (8, 12, 9)
GREEN = (70, 255, 130)
DIM = (30, 150, 70)
AMBER = (255, 176, 40)
WHITE = (235, 245, 235)

LOG = [
    "$ cargo build --release",
    "   Compiling phosphor-surface v0.1.0",
    "   Compiling crt-glass v0.2.0",
    "    Finished release [optimized] target(s) in 4.21s",
    "$ ./bin/stage1_test",
    "1. a square appears for 200 ms, then disappears      ok",
    "2. content that changes every frame                  ok",
    "3. constant image; agreement across refresh rates    ok",
    "4. two tiles: no shared history                      ok",
    "5. resize, hide/show, backend failure                ok",
    "6. idle: no redraw once the trail has faded          ok",
    "all checks passed",
    "$ nvidia-smi --query-gpu=name --format=csv,noheader",
    "NVIDIA GeForce RTX 3080 Laptop GPU",
    "$ git log --oneline -3",
    "bcfd334 Drops land where the screen shows them",
    "fc54857 Version 0.2.0 survives a reload",
    "89df602 crt native status now says unknown",
]


class Term:
    """A character grid drawn with a real monospace font."""

    def __init__(self, w, h, size=22, fg=GREEN):
        self.w, self.h, self.fg = w, h, fg
        self.font = ImageFont.truetype(FONT, size)
        box = self.font.getbbox("M")
        self.cw = int(round(self.font.getlength("M")))
        self.ch = int(size * 1.38)
        self.pad = 18
        self.rows = (h - 2 * self.pad) // self.ch
        self.lines = []  # (text, colour)

    def put(self, text, colour=None):
        self.lines.append((text, colour or self.fg))
        self.lines = self.lines[-self.rows:]

    def draw(self, cursor=None, partial=None):
        """partial: text of the line being typed. cursor: True to draw the block."""
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        shown = list(self.lines)
        if partial is not None:
            shown = (shown + [(partial, self.fg)])[-self.rows:]
        for i, (text, colour) in enumerate(shown):
            d.text((self.pad, self.pad + i * self.ch), text, font=self.font, fill=colour)
        if cursor and shown:
            text = shown[-1][0] if partial is not None else ""
            row = len(shown) - 1 if partial is not None else min(len(shown), self.rows - 1)
            x = self.pad + int(self.font.getlength(text))
            y = self.pad + row * self.ch
            d.rectangle([x, y + 3, x + self.cw, y + self.ch - 4], fill=self.fg)
        return img


def emit(img):
    sys.stdout.buffer.write(img.convert("RGBA").tobytes())


def typing(w, h, fps):
    t = Term(w, h)
    n = 0

    def frame(partial=None, blink=True):
        nonlocal n
        cur = (n // (fps // 2)) % 2 == 0 if blink else True
        emit(t.draw(cursor=cur, partial=partial))
        n += 1

    for _ in range(fps // 2):
        frame(partial="$ ")
    for line in LOG:
        if line.startswith("$"):
            for k in range(2, len(line) + 1):  # typed, 3 frames a key
                for _ in range(3):
                    frame(partial=line[:k], blink=False)
            for _ in range(fps // 4):
                frame(partial=line)
            t.put(line)
        else:
            t.put(line, DIM if line.startswith("   ") else t.fg)
            for _ in range(4):  # output lands fast
                frame(partial="")
    for _ in range(fps // 2):
        frame(partial="$ ")
    # a burst of scrolling: 40 lines, one every 2 frames
    for k in range(40):
        t.put("%04x  %s" % (k * 4099 % 65536, LOG[k % len(LOG)][:58]), DIM if k % 3 else t.fg)
        frame(partial="")
        frame(partial="")
    for _ in range(fps // 3):
        frame(partial="$ ")
    for k in range(2, 8):
        for _ in range(3):
            frame(partial="$ clear"[:k], blink=False)
    t.lines = []
    for _ in range(fps):  # the clear: everything that was lit now decays
        frame(partial="$ ")


def flash(w, h, fps):
    font = ImageFont.truetype(FONT, int(h * 0.42))
    on = int(round(0.2 * fps))

    def blank(k):
        for _ in range(k):
            emit(Image.new("RGB", (w, h), (0, 0, 0)))

    blank(fps // 4)
    for _ in range(on):
        img = Image.new("RGB", (w, h), (0, 0, 0))
        s = h // 4
        ImageDraw.Draw(img).rectangle([w // 2 - s, h // 2 - s, w // 2 + s, h // 2 + s], fill=(255, 255, 255))
        emit(img)
    blank(fps)
    for _ in range(on):
        img = Image.new("RGB", (w, h), (0, 0, 0))
        ImageDraw.Draw(img).text((w // 2, h // 2), "A", font=font, fill=(0, 255, 0), anchor="mm")
        emit(img)
    blank(fps)


def tiles(w, h, fps):
    gap = 16
    tw, th = (w - 3 * gap) // 2, h - 2 * gap
    left, right = Term(tw, th, 20, GREEN), Term(tw, th, 20, AMBER)
    for line in LOG[:9]:
        left.put(line)
        right.put(line[::-1][:40])
    desk = Image.new("RGB", (w, h), (26, 28, 44))
    for n in range(fps * 3):
        img = desk.copy()
        li = left.draw(cursor=(n // (fps // 2)) % 2 == 0, partial="$ ")
        if fps <= n < fps + int(0.2 * fps):  # the left tile flashes at its right edge
            ImageDraw.Draw(li).rectangle([tw - 90, th // 3, tw - 1, 2 * th // 3], fill=(255, 255, 255))
        img.paste(li, (gap, gap))
        img.paste(right.draw(cursor=True, partial="> "), (2 * gap + tw, gap))
        emit(img)
    sys.stderr.write("tiles: %d,%d,%d,%d;%d,%d,%d,%d\n" % (gap, gap, tw, th, 2 * gap + tw, gap, tw, th))


def hold(w, h, fps):
    t = Term(w, h)
    for line in LOG[:12]:
        t.put(line)
    img = t.draw(cursor=True, partial="$ ")
    for _ in range(fps * 2):
        emit(img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["typing", "flash", "tiles", "hold"])
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    {"typing": typing, "flash": flash, "tiles": tiles, "hold": hold}[a.scene](W, H, a.fps)
