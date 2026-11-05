"""Keyframe animation curves (Maya-style tangents, easing presets, infinity/cycle modes).

A :class:`AnimCurve` maps a frame number (float) to a value. Values may be scalars or
vectors (any fixed-length list/array, e.g. positions or colours); vectors are interpolated
component-wise.

Interpolation is set **per key** and controls the segment that *leaves* that key:

=================  =========================================================================
``auto``           (default) smooth Hermite with monotone clamping: no overshoot, flat at
                   extremes - like Maya's *auto* tangents
``spline``         Catmull-Rom style smooth tangents (may overshoot) - Maya *spline*
``clamped``        spline, but flat where neighbouring keys have equal values - Maya *clamped*
``flat``           zero-slope tangents (ease in & out of every key)
``linear``         straight lines
``step``           hold value until the next key (``constant`` is an alias)
``bezier``         Hermite using explicit ``in``/``out`` slopes (value units per frame);
                   missing slopes fall back to ``auto``
any easing name    e.g. ``ease_in_out``, ``bounce_out``, ``elastic_out``, ``back_in``,
                   ``cubic-bezier(0.25,0.1,0.25,1)`` - see :data:`EASINGS`
=================  =========================================================================

Infinity modes (``pre`` / ``post``): ``constant``, ``linear``, ``cycle``, ``cycle_offset``
(cycle with value offset - Maya *cycle with offset*), ``oscillate`` (ping-pong).
"""
from __future__ import annotations

import math
import re
from bisect import bisect_right

import numpy as np


# ------------------------------------------------------------------------------------------
# easing functions  u in [0,1] -> [0,1]-ish
# ------------------------------------------------------------------------------------------
def _bounce_out(u):
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375


def _elastic_out(u):
    if u <= 0 or u >= 1:
        return float(u >= 1)
    c4 = 2 * math.pi / 3
    return 2 ** (-10 * u) * math.sin((u * 10 - 0.75) * c4) + 1


def _back_in(u, s=1.70158):
    return (s + 1) * u ** 3 - s * u * u


def _make_family(fin):
    def f_out(u):
        return 1 - fin(1 - u)

    def f_inout(u):
        return fin(2 * u) / 2 if u < 0.5 else 1 - fin(2 - 2 * u) / 2

    return fin, f_out, f_inout


_FAMILIES = {
    "quad": lambda u: u * u,
    "cubic": lambda u: u ** 3,
    "quart": lambda u: u ** 4,
    "quint": lambda u: u ** 5,
    "sine": lambda u: 1 - math.cos(u * math.pi / 2),
    "expo": lambda u: 0.0 if u <= 0 else 2 ** (10 * u - 10),
    "circ": lambda u: 1 - math.sqrt(max(0.0, 1 - u * u)),
    "back": _back_in,
    "elastic": lambda u: 1 - _elastic_out(1 - u),
    "bounce": lambda u: 1 - _bounce_out(1 - u),
}

EASINGS = {
    "linear": lambda u: u,
    "smoothstep": lambda u: u * u * (3 - 2 * u),
    "smootherstep": lambda u: u ** 3 * (u * (u * 6 - 15) + 10),
}
for _name, _fin in _FAMILIES.items():
    _i, _o, _io = _make_family(_fin)
    EASINGS[f"{_name}_in"] = _i
    EASINGS[f"{_name}_out"] = _o
    EASINGS[f"{_name}_in_out"] = _io
EASINGS["ease_in"] = EASINGS["cubic_in"]
EASINGS["ease_out"] = EASINGS["cubic_out"]
EASINGS["ease_in_out"] = EASINGS["cubic_in_out"]
EASINGS["ease"] = EASINGS["sine_in_out"]

TANGENT_TYPES = ("auto", "spline", "clamped", "flat", "linear", "step", "constant", "bezier")
INFINITY_MODES = ("constant", "linear", "cycle", "cycle_offset", "oscillate")

_CB_RE = re.compile(r"cubic[-_]bezier\(\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*\)")


def cubic_bezier(x1, y1, x2, y2):
    """CSS-style ``cubic-bezier`` timing function (P0=(0,0), P3=(1,1))."""

    def bez(t, a, b):
        return 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t * t * b + t ** 3

    def dbez(t, a, b):
        return 3 * (1 - t) ** 2 * a + 6 * (1 - t) * t * (b - a) + 3 * t * t * (1 - b)

    def f(u):
        if u <= 0:
            return 0.0
        if u >= 1:
            return 1.0
        t = u
        for _ in range(8):
            x = bez(t, x1, x2) - u
            d = dbez(t, x1, x2)
            if abs(x) < 1e-7:
                break
            if abs(d) < 1e-6:
                break
            t -= x / d
        if not (0 <= t <= 1) or abs(bez(t, x1, x2) - u) > 1e-4:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                t = (lo + hi) / 2
                if bez(t, x1, x2) < u:
                    lo = t
                else:
                    hi = t
        return bez(t, y1, y2)

    return f


def get_easing(name):
    """Return an easing callable for ``name`` (see :data:`EASINGS`) or ``None``."""
    if callable(name):
        return name
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in EASINGS:
        return EASINGS[key]
    m = _CB_RE.match(name.strip().lower())
    if m:
        return cubic_bezier(*(float(g) for g in m.groups()))
    return None


def is_interp_name(name):
    if not isinstance(name, str):
        return False
    k = name.strip().lower().replace("-", "_")
    return k in TANGENT_TYPES or get_easing(name) is not None


# ------------------------------------------------------------------------------------------
# keys & curves
# ------------------------------------------------------------------------------------------
class Key:
    """A single keyframe.

    ``interp`` controls the outgoing segment; ``in_slope``/``out_slope`` (value units per
    frame) override the automatically computed tangents.
    """

    __slots__ = ("frame", "value", "interp", "in_slope", "out_slope")

    def __init__(self, frame, value, interp="auto", in_slope=None, out_slope=None):
        self.frame = float(frame)
        self.value = np.asarray(value, dtype=np.float64)
        self.interp = (interp or "auto").strip().lower().replace("-", "_") if isinstance(interp, str) else "auto"
        if self.interp == "constant":
            self.interp = "step"
        self.in_slope = None if in_slope is None else np.asarray(in_slope, dtype=np.float64)
        self.out_slope = None if out_slope is None else np.asarray(out_slope, dtype=np.float64)

    def to_spec(self):
        v = self.value.tolist()
        fr = int(self.frame) if float(self.frame).is_integer() else self.frame
        if self.in_slope is None and self.out_slope is None:  # compact list form
            return [fr, v] if self.interp == "auto" else [fr, v, self.interp]
        d = {"frame": fr, "value": v}
        if self.interp != "auto":
            d["interp"] = self.interp
        if self.in_slope is not None:
            d["in"] = self.in_slope.tolist()
        if self.out_slope is not None:
            d["out"] = self.out_slope.tolist()
        return d

    def __repr__(self):
        return f"Key({self.frame:g}, {self.value.tolist()}, {self.interp!r})"


class AnimCurve:
    """Keyframed curve: ``curve(frame) -> value``."""

    def __init__(self, keys=None, pre="constant", post="constant", default_interp="auto"):
        self.keys: list[Key] = []
        self.pre = pre
        self.post = post
        self.default_interp = default_interp
        self._slopes = None
        for k in keys or []:
            if isinstance(k, Key):
                self.keys.append(k)
            else:
                self.keys.append(parse_key(k, default_interp))
        self.keys.sort(key=lambda k: k.frame)
        for m in (pre, post):
            if m not in INFINITY_MODES:
                raise ValueError(f"bad infinity mode {m!r}; use one of {INFINITY_MODES}")

    # ---- editing ----
    def add_key(self, frame, value, interp=None, in_slope=None, out_slope=None):
        """Insert or replace a key at ``frame``. Returns the :class:`Key`."""
        k = Key(frame, value, interp or self.default_interp, in_slope, out_slope)
        for i, e in enumerate(self.keys):
            if abs(e.frame - k.frame) < 1e-9:
                self.keys[i] = k
                break
        else:
            self.keys.append(k)
            self.keys.sort(key=lambda kk: kk.frame)
        self._slopes = None
        return k

    key = add_key

    def remove_key(self, frame):
        self.keys = [k for k in self.keys if abs(k.frame - frame) > 1e-9]
        self._slopes = None

    @property
    def frame_range(self):
        if not self.keys:
            return (0.0, 0.0)
        return (self.keys[0].frame, self.keys[-1].frame)

    @property
    def times(self):
        return [k.frame for k in self.keys]

    def is_animated(self):
        if len(self.keys) < 2:
            return False
        v0 = self.keys[0].value
        return any(not np.allclose(k.value, v0) for k in self.keys[1:]) or any(
            k.out_slope is not None and np.any(k.out_slope != 0) for k in self.keys)

    # ---- tangents ----
    def _compute_slopes(self):
        n = len(self.keys)
        ins, outs = [None] * n, [None] * n
        for i, k in enumerate(self.keys):
            prev = self.keys[i - 1] if i > 0 else None
            nxt = self.keys[i + 1] if i < n - 1 else None
            v = k.value
            # interp governing the incoming segment is the previous key's interp
            in_mode = prev.interp if prev is not None else k.interp
            out_mode = k.interp
            ins[i] = self._slope(k, prev, nxt, in_mode, side="in")
            outs[i] = self._slope(k, prev, nxt, out_mode, side="out")
            if k.in_slope is not None:
                ins[i] = k.in_slope * np.ones_like(v)
            if k.out_slope is not None:
                outs[i] = k.out_slope * np.ones_like(v)
        self._slopes = (ins, outs)

    @staticmethod
    def _slope(k, prev, nxt, mode, side):
        v = k.value
        zero = np.zeros_like(v)
        dl = (v - prev.value) / max(1e-9, k.frame - prev.frame) if prev is not None else None
        dr = (nxt.value - v) / max(1e-9, nxt.frame - k.frame) if nxt is not None else None
        if mode in ("flat", "step"):
            return zero
        if mode == "linear":
            if side == "in":
                return dl if dl is not None else (dr if dr is not None else zero)
            return dr if dr is not None else (dl if dl is not None else zero)
        if prev is None or nxt is None:
            if mode == "spline":
                d = dr if dr is not None else dl
                return d if d is not None else zero
            return zero
        spline = (nxt.value - prev.value) / max(1e-9, nxt.frame - prev.frame)
        if mode == "spline":
            return spline
        if mode == "clamped":
            flat = (np.abs(v - prev.value) < 1e-6) | (np.abs(nxt.value - v) < 1e-6)
            return np.where(flat, 0.0, spline)
        # auto (also bezier fallback): monotone-preserving (Fritsch-Carlson)
        ext = (dl * dr) <= 0
        lim = 3.0 * np.minimum(np.abs(dl), np.abs(dr))
        s = np.clip(spline, -lim, lim)
        return np.where(ext, 0.0, s)

    # ---- evaluation ----
    def __call__(self, frame):
        return self.evaluate(frame)

    def evaluate(self, frame):
        keys = self.keys
        if not keys:
            return 0.0
        if len(keys) == 1:
            return _out(keys[0].value)
        if self._slopes is None:
            self._compute_slopes()
        f0, f1 = keys[0].frame, keys[-1].frame
        span = f1 - f0
        offset = None
        frame = float(frame)
        if frame < f0:
            mode = self.pre
            if mode == "constant":
                return _out(keys[0].value)
            if mode == "linear":
                if keys[0].interp == "linear":
                    s = (keys[1].value - keys[0].value) / max(1e-9, keys[1].frame - f0)
                else:
                    s = self._slopes[1][0]
                return _out(keys[0].value + s * (frame - f0))
            frame, cycles = _wrap(frame, f0, span, mode)
            if mode == "cycle_offset":
                offset = (keys[-1].value - keys[0].value) * cycles
        elif frame > f1:
            mode = self.post
            if mode == "constant":
                return _out(keys[-1].value)
            if mode == "linear":
                p = keys[-2]
                s = (keys[-1].value - p.value) / max(1e-9, f1 - p.frame) if p.interp == "linear" else self._slopes[0][-1]
                return _out(keys[-1].value + s * (frame - f1))
            frame, cycles = _wrap(frame, f0, span, mode)
            if mode == "cycle_offset":
                offset = (keys[-1].value - keys[0].value) * cycles
        v = self._eval_inside(frame)
        if offset is not None:
            v = v + offset
        return _out(v)

    def _eval_inside(self, frame):
        keys = self.keys
        times = [k.frame for k in keys]
        i = bisect_right(times, frame) - 1
        i = max(0, min(i, len(keys) - 2))
        a, b = keys[i], keys[i + 1]
        dt = b.frame - a.frame
        if dt <= 0:
            return b.value
        u = min(1.0, max(0.0, (frame - a.frame) / dt))
        mode = a.interp
        if mode == "step":
            return a.value if frame < b.frame else b.value
        if mode == "linear":
            return a.value + (b.value - a.value) * u
        ease = get_easing(mode) if mode not in TANGENT_TYPES else None
        if ease is not None:
            return a.value + (b.value - a.value) * ease(u)
        m0 = self._slopes[1][i] * dt
        m1 = self._slopes[0][i + 1] * dt
        u2, u3 = u * u, u * u * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        return h00 * a.value + h10 * m0 + h01 * b.value + h11 * m1

    def sample(self, frames):
        return np.array([self.evaluate(f) for f in frames])

    def derivative(self, frame, h=1e-3):
        return (np.asarray(self.evaluate(frame + h)) - np.asarray(self.evaluate(frame - h))) / (2 * h)

    def to_spec(self):
        d = {"keys": [k.to_spec() for k in self.keys]}
        if self.pre != "constant":
            d["pre"] = self.pre
        if self.post != "constant":
            d["post"] = self.post
        return d

    def __repr__(self):
        return f"AnimCurve({self.keys!r}, pre={self.pre!r}, post={self.post!r})"


def _wrap(frame, f0, span, mode):
    if span <= 0:
        return f0, 0
    rel = frame - f0
    cycles = math.floor(rel / span)
    local = rel - cycles * span
    if mode == "oscillate" and (cycles % 2 != 0):
        local = span - local
    return f0 + local, cycles


def _out(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 0:
        return float(v)
    return v.copy()


def parse_key(k, default_interp="auto"):
    """Parse a key spec.

    Accepted forms::

        [frame, value]
        [frame, value, "interp"]
        [frame, value, {"interp": "bezier", "in": 0.0, "out": 2.5}]
        {"frame": 12, "value": [0, 1, 0], "interp": "ease_out", "in": ..., "out": ...}
    """
    if isinstance(k, Key):
        return k
    if isinstance(k, dict):
        frame = k.get("frame", k.get("f", k.get("t", k.get("time"))))
        if frame is None:
            raise ValueError(f"key {k!r} has no 'frame'")
        value = k.get("value", k.get("v"))
        interp = k.get("interp", k.get("tangent", k.get("ease", default_interp)))
        ins = k.get("in", k.get("in_slope", k.get("in_tangent")))
        outs = k.get("out", k.get("out_slope", k.get("out_tangent")))
        return Key(frame, value, interp, ins, outs)
    if isinstance(k, (list, tuple)) and len(k) >= 2:
        frame, value = k[0], k[1]
        interp, ins, outs = default_interp, None, None
        if len(k) >= 3:
            extra = k[2]
            if isinstance(extra, str):
                interp = extra
            elif isinstance(extra, dict):
                interp = extra.get("interp", extra.get("ease", default_interp))
                ins = extra.get("in", extra.get("in_slope"))
                outs = extra.get("out", extra.get("out_slope"))
        return Key(frame, value, interp, ins, outs)
    raise ValueError(f"cannot parse key {k!r}")
