"""Graph editor without a window: plot animation curves in the terminal or to a PNG, and
edit keyframes of a scene file from the command line.

* :func:`plot_terminal` draws curves with Unicode braille dots (2x4 sub-cells per character)
  in colour, with key markers, value axis labels and a frame ruler.
* :func:`plot_png` draws the same graph at full resolution (grid, curves, keys, tangent
  handles, current-frame line) with Pillow.
* :func:`edit_keys` changes keys in the *raw* scene file (JSON) so the rest of the file is
  kept exactly as written: add / move / delete keys, set interpolation and tangents.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .params import Param

COMP_NAMES = ("x", "y", "z", "w")
PALETTE = [(230, 80, 70), (90, 200, 90), (80, 140, 240), (240, 190, 60), (200, 100, 220),
           (60, 200, 200), (240, 130, 40), (180, 180, 180)]
_COMP_IDX = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2}


# ----------------------------------------------------------------------------- sampling
def find_param(scene, path):
    """Return ``(param, component_index|None)`` for ``node.attr[.comp]`` or ``(None, None)``."""
    parts = str(path).split(".")
    node = scene.nodes.get(parts[0])
    if node is None or len(parts) < 2:
        return None, None
    attr = parts[1]
    comp = None
    if len(parts) > 2:
        comp = _COMP_IDX.get(parts[2])
        if comp is None:
            try:
                comp = int(parts[2])
            except ValueError:
                comp = None
    prm = None
    if attr in ("translate", "rotate", "scale", "pivot", "visible"):
        prm = getattr(node, attr, None)
    elif attr in node.attributes:
        prm = node.attributes[attr]
    elif attr in node.params:
        prm = node.params[attr]
    return (prm if isinstance(prm, Param) else None), comp


def param_keys(prm, comp=None):
    """Keys of a Param as ``[(frame, value, interp)]`` (component-aware; [] if not keyed)."""
    if prm is None:
        return []
    if prm.kind == "parts":
        if comp is not None and comp < len(prm.parts):
            return param_keys(prm.parts[comp], None)
        return []
    if prm.kind != "curve":
        return []
    out = []
    for k in prm.curve.keys:
        v = np.asarray(k.value, float).ravel()
        val = float(v[comp]) if comp is not None and comp < v.size else (float(v[0]) if v.size == 1 else v)
        out.append((k.frame, val, k.interp))
    return out


def sample(scene, channels, frames):
    """Evaluate channels over frames. Returns ``[(label, values[N], keys)]`` one per scalar curve.

    A vector channel without a component (``ball.translate``) expands to one curve per
    component (``ball.translate.x`` ...).
    """
    series = []
    for ch in channels:
        vals = [np.atleast_1d(np.asarray(scene.get_value(ch, f), float)).ravel() for f in frames]
        arr = np.array(vals)
        prm, comp = find_param(scene, ch)
        if arr.shape[1] == 1:
            ks = [(f, v if np.ndim(v) == 0 else float(np.ravel(v)[0]), i) for f, v, i in param_keys(prm, comp)]
            series.append((ch, arr[:, 0], ks))
        else:
            for c in range(arr.shape[1]):
                ks = []
                if prm is not None:
                    if prm.kind == "parts":
                        ks = param_keys(prm, c)
                    else:
                        ks = [(f, float(np.ravel(v)[c]), i) for f, v, i in param_keys(prm, None)
                              if np.size(v) > c]
                name = COMP_NAMES[c] if c < 4 else str(c)
                series.append((f"{ch}.{name}", arr[:, c], ks))
    return series


def _range(series, lo=None, hi=None):
    allv = np.concatenate([s[1] for s in series] + [np.array([k[1] for k in s[2]], float) for s in series if s[2]])
    allv = allv[np.isfinite(allv)]
    vlo = float(allv.min()) if lo is None and allv.size else (lo if lo is not None else 0.0)
    vhi = float(allv.max()) if hi is None and allv.size else (hi if hi is not None else 1.0)
    if vhi - vlo < 1e-9:
        vlo, vhi = vlo - 1.0, vhi + 1.0
    pad = (vhi - vlo) * 0.06
    return vlo - (pad if lo is None else 0), vhi + (pad if hi is None else 0)


def _fmt(v):
    a = abs(v)
    if a < 1e-9:
        return "0"
    if a >= 1000 or (0 < a < 0.01):
        return f"{v:.2e}"
    return f"{v:.3g}" if a < 10 else f"{v:.1f}"


# ----------------------------------------------------------------------------- terminal
_BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def plot_terminal(series, frames, width=None, height=None, color=True, vmin=None, vmax=None, current=None):
    """Return a multi-line string drawing ``series`` (from :func:`sample`) with braille dots."""
    cols, rows = shutil.get_terminal_size((100, 32))
    width = int(width or max(30, cols - 12))
    height = int(height or max(8, min(24, rows - 8)))
    W, H = width * 2, height * 4
    lo, hi = _range(series, vmin, vmax)
    f0, f1 = float(frames[0]), float(frames[-1])
    span = max(1e-9, f1 - f0)
    cells = np.zeros((height, width), np.int32)
    owner = -np.ones((height, width), np.int32)
    keyc = {}

    def to_px(f, v):
        x = (f - f0) / span * (W - 1)
        y = (1 - (v - lo) / (hi - lo)) * (H - 1)
        return x, y

    for si, (_, vals, keys) in enumerate(series):
        fr = np.asarray(frames, float)
        # resample densely for smooth lines
        xs = np.linspace(f0, f1, W * 2)
        ys = np.interp(xs, fr, vals)
        prev = None
        for f, v in zip(xs, ys):
            if not np.isfinite(v):
                prev = None
                continue
            x, y = to_px(f, v)
            pts = [(x, y)]
            if prev is not None:  # fill vertical gaps so steep segments stay connected
                n = int(abs(y - prev[1]) // 1)
                pts = [(prev[0] + (x - prev[0]) * t, prev[1] + (y - prev[1]) * t)
                       for t in np.linspace(0, 1, n + 2)]
            for px, py in pts:
                ix, iy = int(round(px)), int(round(py))
                if 0 <= ix < W and 0 <= iy < H:
                    cy, cx = iy // 4, ix // 2
                    cells[cy, cx] |= _BRAILLE[iy % 4][ix % 2]
                    owner[cy, cx] = si
            prev = (x, y)
        for f, v, _ in keys:
            if f0 <= f <= f1 and np.isfinite(v):
                x, y = to_px(f, v)
                cx, cy = int(round(x)) // 2, int(round(y)) // 4
                if 0 <= cx < width and 0 <= cy < height:
                    keyc[(cy, cx)] = si
    cur_col = None
    if current is not None and f0 <= current <= f1:
        cur_col = int(round((current - f0) / span * (W - 1))) // 2

    def paint(text, si):
        if not color or si < 0:
            return text
        r, g, b = PALETTE[si % len(PALETTE)]
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

    lines = []
    label_w = 9
    for cy in range(height):
        v = hi - (cy / max(1, height - 1)) * (hi - lo)
        lab = _fmt(v).rjust(label_w) if cy % 3 == 0 or cy == height - 1 else " " * label_w
        row = []
        for cx in range(width):
            if (cy, cx) in keyc:
                row.append(paint("◆", keyc[(cy, cx)]))
            elif cells[cy, cx]:
                row.append(paint(chr(0x2800 + cells[cy, cx]), owner[cy, cx]))
            elif cx == cur_col:
                row.append("\033[2m│\033[0m" if color else "|")
            elif lo < 0 < hi and cy == int(round((hi / (hi - lo)) * (height - 1))):
                row.append("\033[2m┈\033[0m" if color else "-")
            else:
                row.append(" ")
        lines.append(f"{lab} ┤" + "".join(row))
    # frame ruler
    ruler = [" "] * width
    labels = [" "] * (width + 8)
    nt = max(2, min(10, width // 10))
    for i in range(nt + 1):
        cx = int(round(i / nt * (width - 1)))
        ruler[cx] = "┬"
        fl = f0 + i / nt * span
        s = str(int(round(fl))) if span >= nt * 2 or abs(fl - round(fl)) < 1e-6 else f"{fl:.2f}".rstrip("0").rstrip(".")
        for j, ch in enumerate(s):
            if cx + j < len(labels):
                labels[cx + j] = ch
    lines.append(" " * label_w + " └" + "".join(ruler).replace(" ", "─"))
    lines.append(" " * (label_w + 2) + "".join(labels).rstrip() + "  (frame)")
    cur, curlen = [], 0
    for i, (name, _, k) in enumerate(series):
        item = f" {name}" + (f" ({len(k)} keys)" if k else "")
        if cur and curlen + len(item) + 4 > width:
            lines.append(" " * (label_w + 2) + "  ".join(cur))
            cur, curlen = [], 0
        cur.append(paint("━━", i) + item)
        curlen += len(item) + 4
    if cur:
        lines.append(" " * (label_w + 2) + "  ".join(cur))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- PNG
def plot_png(series, frames, path, size=(1400, 700), vmin=None, vmax=None, current=None, title=None,
             dark=True, scene=None, channels=None):
    """Render a graph-editor style image (curves, keys, tangent handles) to ``path``."""
    from PIL import Image, ImageDraw
    Wd, Hd = int(size[0]), int(size[1])
    ss = 2
    W, H = Wd * ss, Hd * ss
    bg = (38, 38, 42) if dark else (250, 250, 250)
    gridc = (58, 58, 64) if dark else (225, 225, 225)
    axisc = (110, 110, 120) if dark else (140, 140, 140)
    textc = (200, 200, 205) if dark else (40, 40, 40)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    ml, mr, mt, mb = 90 * ss, 20 * ss, (40 if title else 20) * ss, 60 * ss
    lo, hi = _range(series, vmin, vmax)
    f0, f1 = float(frames[0]), float(frames[-1])
    span = max(1e-9, f1 - f0)

    def P(f, v):
        return (ml + (f - f0) / span * (W - ml - mr), mt + (1 - (v - lo) / (hi - lo)) * (H - mt - mb))

    def nice_step(r, n):
        raw = r / n
        mag = 10 ** np.floor(np.log10(raw))
        for m in (1, 2, 5, 10):
            if raw <= m * mag:
                return m * mag
        return 10 * mag

    vs = nice_step(hi - lo, 8)
    v = np.ceil(lo / vs) * vs
    while v <= hi:
        _, y = P(f0, v)
        d.line([(ml, y), (W - mr, y)], fill=axisc if abs(v) < vs * 1e-6 else gridc, width=ss)
        d.text((8 * ss, y - 6 * ss), _fmt(v), fill=textc, font_size=12 * ss)
        v += vs
    fs = max(1.0, nice_step(span, 12))
    f = np.ceil(f0 / fs) * fs
    while f <= f1 + 1e-9:
        x, _ = P(f, lo)
        d.line([(x, mt), (x, H - mb)], fill=gridc, width=ss)
        d.text((x - 8 * ss, H - mb + 8 * ss), f"{f:g}", fill=textc, font_size=12 * ss)
        f += fs
    d.text((W // 2 - 20 * ss, H - 26 * ss), "frame", fill=textc, font_size=12 * ss)
    if current is not None and f0 <= current <= f1:
        x, _ = P(current, lo)
        d.line([(x, mt), (x, H - mb)], fill=(220, 60, 60), width=2 * ss)
    if title:
        d.text((ml, 10 * ss), title, fill=textc, font_size=16 * ss)
    fr = np.asarray(frames, float)
    xs = np.linspace(f0, f1, max(200, (W - ml - mr) // 2))
    for si, (name, vals, keys) in enumerate(series):
        col = PALETTE[si % len(PALETTE)]
        ys = np.interp(xs, fr, vals)
        pts = [P(a, b) for a, b in zip(xs, ys) if np.isfinite(b)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2 * ss, joint="curve")
        # tangent handles: slope from the sampled curve around each key
        hl = span / 30.0
        for kf, kv, _ in keys:
            if not (f0 <= kf <= f1):
                continue
            a = np.interp(kf - 0.5, fr, vals)
            b = np.interp(kf + 0.5, fr, vals)
            slope = b - a
            for sgn in (-1, 1):
                hx, hy = P(kf + sgn * hl, kv + sgn * slope * hl)
                kx, ky = P(kf, kv)
                d.line([(kx, ky), (hx, hy)], fill=(150, 150, 160), width=ss)
                d.ellipse([hx - 3 * ss, hy - 3 * ss, hx + 3 * ss, hy + 3 * ss], fill=(150, 150, 160))
            kx, ky = P(kf, kv)
            r = 5 * ss
            d.polygon([(kx, ky - r), (kx + r, ky), (kx, ky + r), (kx - r, ky)], fill=(255, 255, 255), outline=col)
        ly = mt + 6 * ss + si * 18 * ss
        d.rectangle([W - mr - 230 * ss, ly + 4 * ss, W - mr - 210 * ss, ly + 8 * ss], fill=col)
        d.text((W - mr - 204 * ss, ly - 2 * ss), name + (f"  ({len(keys)} keys)" if keys else ""),
               fill=textc, font_size=12 * ss)
    im = im.resize((Wd, Hd), Image.LANCZOS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


# ----------------------------------------------------------------------------- key editing
def _find_node_spec(data, name):
    """Find the dict for node ``name`` in a raw scene dict (nodes list/dict and children)."""
    nodes = data.get("nodes", [])
    stack = list(nodes.items()) if isinstance(nodes, dict) else [(None, n) for n in nodes]
    while stack:
        key, n = stack.pop()
        if not isinstance(n, dict):
            continue
        if n.get("name", key) == name:
            return n
        ch = n.get("children", [])
        stack.extend(list(ch.items()) if isinstance(ch, dict) else [(None, c) for c in ch])
    return None


_ALIASES = {"position": "translate", "rotation": "rotate"}


def edit_keys(path, channel, op, frame=None, value=None, interp=None, new_frame=None,
              in_slope=None, out_slope=None, pre=None, post=None, defaults=None):
    """Edit keys of ``channel`` (``node.attr[.comp]``) in the JSON scene file at ``path``.

    ``op``: ``set`` (insert/replace a key), ``delete``, ``move`` (frame -> new_frame),
    ``interp`` (change interpolation of the key at ``frame``; all keys if frame is None),
    ``tangent`` (set in/out slopes), ``infinity`` (pre/post modes), ``clear`` (make constant).
    Returns the new spec of the attribute.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("key editing works on .json scene files")
    data = json.loads(path.read_text())
    parts = channel.split(".")
    if len(parts) < 2:
        raise ValueError("channel must be node.attr[.component]")
    node = _find_node_spec(data, parts[0])
    if node is None:
        raise ValueError(f"no node {parts[0]!r} in {path}")
    attr = parts[1]
    for a, b in _ALIASES.items():
        if attr == b and a in node and b not in node:
            attr = a
    comp = None
    if len(parts) > 2:
        comp = _COMP_IDX.get(parts[2])
        if comp is None:
            comp = int(parts[2])
    default = (defaults or {}).get(parts[1], 0.0)
    spec = node.get(attr, default)
    prm = Param(spec)

    def edit_curve(p, val):
        if op == "set":
            if p.kind != "curve":
                cur = p.evaluate(frame) if p.kind == "const" else None
                p.kind, p.curve = "curve", None
                from .curves import AnimCurve
                p.curve = AnimCurve()
                if cur is not None and val is None:
                    val = cur
            if val is None:
                val = p.curve.evaluate(frame) if p.curve.keys else 0.0
            p.curve.add_key(frame, val, interp, in_slope, out_slope)
        elif p.kind != "curve":
            raise ValueError(f"{channel} is not keyframed")
        elif op == "delete":
            p.curve.remove_key(frame)
            if not p.curve.keys:
                p.kind, p.const, p.curve = "const", 0.0, None
        elif op == "move":
            k = _key_at(p, frame)
            p.curve.remove_key(frame)
            p.curve.add_key(new_frame, k.value, k.interp, k.in_slope, k.out_slope)
        elif op == "interp":
            for k in p.curve.keys:
                if frame is None or abs(k.frame - frame) < 1e-6:
                    k.interp = interp
            p.curve._slopes = None
        elif op == "tangent":
            k = _key_at(p, frame)
            if in_slope is not None:
                k.in_slope = np.asarray(in_slope, float)
            if out_slope is not None:
                k.out_slope = np.asarray(out_slope, float)
            p.curve._slopes = None
        elif op == "infinity":
            if pre:
                p.curve.pre = pre
            if post:
                p.curve.post = post
        elif op == "clear":
            v = p.curve.evaluate(frame if frame is not None else p.curve.keys[0].frame)
            p.kind, p.const, p.curve = "const", v, None
        return p

    if comp is None and prm.kind == "parts" and op in ("interp", "infinity", "clear"):
        for part in prm.parts:
            if part.kind == "curve":
                edit_curve(part, None)
    elif comp is None:
        edit_curve(prm, value)
    else:
        # per-component editing: split the vector into parts
        if prm.kind != "parts":
            if prm.kind == "curve":
                n = np.size(prm.curve.keys[0].value) if prm.curve.keys else 3
                comps = []
                for c in range(n):
                    from .curves import AnimCurve
                    cc = AnimCurve(pre=prm.curve.pre, post=prm.curve.post)
                    for k in prm.curve.keys:
                        cc.add_key(k.frame, float(np.ravel(k.value)[c]), k.interp)
                    comps.append(Param(cc))
            else:
                base = np.atleast_1d(np.asarray(prm.evaluate(0.0), float))
                if base.size == 1:
                    base = np.repeat(base, 3)
                comps = [Param(float(b)) for b in base]
            prm = Param(0.0)
            prm.kind, prm.parts = "parts", comps
        while len(prm.parts) <= comp:
            prm.parts.append(Param(0.0))
        edit_curve(prm.parts[comp], value)
    new_spec = prm.to_spec()
    node[attr] = new_spec
    path.write_text(json.dumps(data, indent=2))
    return new_spec


def _key_at(p, frame):
    for k in p.curve.keys:
        if abs(k.frame - frame) < 1e-6:
            return k
    raise ValueError(f"no key at frame {frame:g}")
