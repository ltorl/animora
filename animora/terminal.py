"""Terminal helpers: progress bars, colour, and inline image previews (ANSI truecolor half-blocks).

No window is ever opened; previews are drawn with Unicode ``▀`` characters whose foreground /
background colours encode two pixel rows each. Works in any truecolor terminal (iTerm2, Terminal,
kitty, WezTerm, VS Code, tmux with Tc). Falls back to 256-colour when ``COLORTERM`` is unset.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np


def supports_color(stream=None):
    stream = stream or sys.stderr
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def truecolor():
    return os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit") or os.environ.get("TERM_PROGRAM") in (
        "iTerm.app", "vscode", "WezTerm", "ghostty")


def style(text, *codes, stream=None):
    if not supports_color(stream):
        return text
    return "".join(f"\033[{c}m" for c in codes) + text + "\033[0m"


def bold(t):
    return style(t, 1)


def dim(t):
    return style(t, 2)


def green(t):
    return style(t, 32)


def red(t):
    return style(t, 31)


def yellow(t):
    return style(t, 33)


def cyan(t):
    return style(t, 36)


def fmt_time(sec):
    if sec is None or not np.isfinite(sec):
        return "--:--"
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    """Single-line progress bar on stderr with ETA. ``update(done_fraction, label)``."""

    def __init__(self, total_label="", stream=None, enabled=True):
        self.stream = stream or sys.stderr
        self.enabled = enabled and hasattr(self.stream, "isatty") and self.stream.isatty()
        self.t0 = time.time()
        self.label = total_label
        self._last = 0.0

    def update(self, frac, label=None, force=False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last < 0.1 and frac < 1.0:
            return
        self._last = now
        frac = min(max(frac, 0.0), 1.0)
        el = now - self.t0
        eta = el / frac - el if frac > 1e-4 else None
        cols = shutil.get_terminal_size((80, 20)).columns
        text = f" {label or self.label}  {frac * 100:5.1f}%  {fmt_time(el)} elapsed, ETA {fmt_time(eta)} "
        width = max(10, cols - len(text) - 3)
        n = int(frac * width)
        bar = "█" * n + "░" * (width - n)
        self.stream.write("\r" + style(bar, 36, stream=self.stream) + text)
        self.stream.flush()

    def done(self, msg=None):
        if self.enabled:
            self.stream.write("\r\033[K")
            self.stream.flush()
        if msg:
            print(msg, file=self.stream)


def log(msg, stream=None):
    print(msg, file=stream or sys.stderr, flush=True)


def _rgb256(r, g, b):
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + int((r - 8) / 247 * 24)
    return 16 + 36 * int(r / 255 * 5 + 0.5) + 6 * int(g / 255 * 5 + 0.5) + int(b / 255 * 5 + 0.5)


def image_to_ansi(img, width=None, height=None):
    """Convert a display-space float image (H,W,3|4, 0-1) to an ANSI string using half blocks."""
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, 2)
    if img.shape[2] == 4:  # composite over checker
        a = img[..., 3:4]
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        chk = (((yy // 8) + (xx // 8)) % 2)[..., None] * 0.15 + 0.35
        img = img[..., :3] * a + chk * (1 - a)
    cols, rows = shutil.get_terminal_size((100, 40))
    H, W = img.shape[:2]
    if width is None:
        width = min(cols - 1, W)
        if height is not None:
            width = min(width, int(height * 2 * W / H))
        else:
            width = min(width, int((rows - 3) * 2 * W / H))
    width = max(4, int(width))
    h2 = max(2, int(round(width * H / W / 2)) * 2)
    from PIL import Image
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((width, h2), Image.LANCZOS)
    a = np.asarray(im).astype(int)
    tc = truecolor()
    out = []
    for y in range(0, h2, 2):
        line = []
        for x in range(width):
            t, b = a[y, x], a[y + 1, x]
            if tc:
                line.append(f"\033[38;2;{t[0]};{t[1]};{t[2]}m\033[48;2;{b[0]};{b[1]};{b[2]}m▀")
            else:
                line.append(f"\033[38;5;{_rgb256(*t)}m\033[48;5;{_rgb256(*b)}m▀")
        out.append("".join(line) + "\033[0m")
    return "\n".join(out)


def show_image(img, width=None, height=None, stream=None):
    stream = stream or sys.stdout
    stream.write(image_to_ansi(img, width, height) + "\n")
    stream.flush()


def play(frames_iter, fps=12, width=None, loops=1):
    """Play display images in place in the terminal (an ASCII 'flipbook')."""
    frames = list(frames_iter)
    if not frames:
        return
    dt = 1.0 / max(1e-3, fps)
    first = True
    try:
        sys.stdout.write("\033[?25l")
        for _ in range(max(1, loops)):
            for img in frames:
                t = time.time()
                s = image_to_ansi(img, width)
                if not first:
                    sys.stdout.write(f"\033[{s.count(chr(10)) + 1}A")
                sys.stdout.write(s + "\n")
                sys.stdout.flush()
                first = False
                time.sleep(max(0.0, dt - (time.time() - t)))
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
