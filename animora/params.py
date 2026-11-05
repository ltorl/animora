"""Animatable parameters.

Every animatable attribute in Animora (transforms, material values, light intensity, camera
focal length, ...) is stored as a :class:`Param`. A Param can be created from:

* a constant:                        ``0.5`` or ``[1, 2, 3]``
* keyframes:                         ``{"keys": [[1, 0], [24, 5, "ease_out"]], "post": "cycle"}``
* a bare key list (shorthand):       ``[[1, 0], [24, 5]]``  (list of ``[frame, value...]``)
* an expression:                     ``"=sin(t*tau)*2"`` or ``{"expr": "..."}``
* per-component mix:                 ``[0, "=sin(t)", {"keys": [[1,0],[10,3]]}]``
* keys + additive expression:        ``{"keys": [...], "add": "=noise(t*3)*0.1"}``
* keys + multiplying expression:     ``{"keys": [...], "mul": "=1+0.1*sin(t)"}``
* set driven key (Maya SDK):         ``{"driver": "ctrl.translate.y", "keys": [[0, 0], [1, 90]]}``
  -- the curve's x axis is the *driver value* instead of the frame.
"""
from __future__ import annotations

import numpy as np

from .curves import AnimCurve, parse_key
from .expressions import Expression, compile_expr


def _is_key_list(v):
    """True for ``[[frame, value], ...]`` shorthand (every item a list starting with a number)."""
    if not isinstance(v, (list, tuple)) or not v:
        return False
    for item in v:
        if isinstance(item, dict) and "frame" in item:
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return False
        if not isinstance(item[0], (int, float)) or isinstance(item[0], bool):
            return False
    return True


class Param:
    """An animatable value: ``param(frame, ctx) -> value``."""

    __slots__ = ("kind", "const", "curve", "expr", "parts", "add", "mul", "size", "driver")

    def __init__(self, spec=0.0):
        self.kind = "const"
        self.const = None
        self.curve = None
        self.expr = None
        self.parts = None
        self.add = None
        self.mul = None
        self.size = None
        self.driver = None
        self.set(spec)

    # ------------------------------------------------------------------
    def set(self, spec):
        self.curve = self.expr = self.parts = self.add = self.mul = self.driver = None
        if isinstance(spec, Param):
            for s in self.__slots__:
                setattr(self, s, getattr(spec, s))
            return self
        if isinstance(spec, AnimCurve):
            self.kind, self.curve = "curve", spec
        elif isinstance(spec, Expression):
            self.kind, self.expr = "expr", spec
        elif isinstance(spec, str):
            if spec.strip().startswith("="):
                self.kind, self.expr = "expr", compile_expr(spec)
            else:
                self.kind, self.const = "const", spec  # string constant (e.g. a name)
        elif isinstance(spec, dict):
            if "keys" in spec:
                self.kind = "curve"
                self.curve = AnimCurve(spec["keys"], pre=spec.get("pre", "constant"),
                                       post=spec.get("post", "constant"),
                                       default_interp=spec.get("interp", "auto"))
                if spec.get("driver"):
                    self.kind, self.driver = "driven", str(spec["driver"])
            elif "expr" in spec:
                self.kind, self.expr = "expr", compile_expr(spec["expr"])
            elif "value" in spec:
                self.kind, self.const = "const", _arr(spec["value"])
            else:
                raise ValueError(f"cannot interpret animatable value {spec!r}")
            if "add" in spec:
                self.add = Param(spec["add"])
            if "mul" in spec:
                self.mul = Param(spec["mul"])
        elif _is_key_list(spec):
            self.kind, self.curve = "curve", AnimCurve([parse_key(k) for k in spec])
        elif isinstance(spec, (list, tuple)) and any(
                isinstance(c, (str, dict, list, tuple, Param)) for c in spec):
            self.kind = "parts"
            self.parts = [Param(c) for c in spec]
        else:
            self.kind, self.const = "const", _arr(spec)
        return self

    # ------------------------------------------------------------------
    def is_animated(self):
        if self.kind == "const":
            return (self.add is not None and self.add.is_animated()) or (
                self.mul is not None and self.mul.is_animated())
        if self.kind == "curve":
            return self.curve.is_animated() or (self.add is not None and self.add.is_animated())
        if self.kind == "parts":
            return any(p.is_animated() for p in self.parts)
        return True  # expressions / driven keys are assumed time-varying

    def evaluate(self, frame=0.0, ctx=None):
        if self.kind == "const":
            v = self.const
        elif self.kind == "curve":
            v = self.curve.evaluate(frame)
        elif self.kind == "driven":
            dv = 0.0
            if ctx is not None and "get" in ctx:
                try:
                    dv = _scalar(ctx["get"](self.driver))
                except (KeyError, ValueError, TypeError, RecursionError):
                    dv = 0.0
            v = self.curve.evaluate(dv)
        elif self.kind == "expr":
            if ctx is None:
                from .expressions import make_context
                ctx = make_context(frame)
            v = self.expr.evaluate(ctx)
        else:
            v = np.array([_scalar(p.evaluate(frame, ctx)) for p in self.parts], np.float64)
        if self.mul is not None:
            v = np.asarray(v, np.float64) * self.mul.evaluate(frame, ctx)
        if self.add is not None:
            v = np.asarray(v, np.float64) + self.add.evaluate(frame, ctx)
        if isinstance(v, np.ndarray):
            return v.copy() if v.ndim else float(v)
        return v

    __call__ = evaluate

    # ---- convenience ----
    def key(self, frame, value, interp=None, **kw):
        """Add a keyframe (converts a constant Param into a curve)."""
        if self.kind != "curve":
            self.kind = "curve"
            self.curve = AnimCurve()
        self.curve.add_key(frame, value, interp, kw.get("in_slope"), kw.get("out_slope"))
        return self

    def keys(self):
        return self.curve.keys if self.kind == "curve" else []

    def key_frames(self):
        out = set()
        if self.kind == "curve":
            out.update(k.frame for k in self.curve.keys)
        if self.kind == "parts":
            for p in self.parts:
                out.update(p.key_frames())
        return out

    def to_spec(self):
        if self.kind == "const":
            c = self.const
            base = c.tolist() if isinstance(c, np.ndarray) else c
            if self.add is None and self.mul is None:
                return base
            d = {"value": base}
        elif self.kind in ("curve", "driven"):
            d = self.curve.to_spec()
            if self.driver:
                d["driver"] = self.driver
        elif self.kind == "expr":
            if self.add is None and self.mul is None:
                return "=" + self.expr.source
            d = {"expr": self.expr.source}
        else:
            return [p.to_spec() for p in self.parts]
        if self.add is not None:
            d["add"] = self.add.to_spec()
        if self.mul is not None:
            d["mul"] = self.mul.to_spec()
        return d

    def __repr__(self):
        return f"Param({self.to_spec()!r})"


def _arr(v):
    if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    try:
        a = np.asarray(v, np.float64)
        return float(a) if a.ndim == 0 else a
    except (TypeError, ValueError):
        return v


def _scalar(v):
    a = np.asarray(v, np.float64)
    return float(a) if a.ndim == 0 else float(a.ravel()[0])


def as_param(v):
    return v if isinstance(v, Param) else Param(v)


def _anim_leaf(v):
    if isinstance(v, str):
        return v.strip().startswith("=")
    if isinstance(v, dict):
        return "keys" in v or "expr" in v
    return False


class TreeParam:
    """A nested structure (lists / dicts, e.g. an SDF ``shapes`` tree or loft ``sections``) whose
    leaves may be animated (``"=expr"``, ``{"keys": ...}``, ``{"expr": ...}``). Plain key-list
    shorthand is *not* recognised inside trees (it is ambiguous with point lists)."""

    __slots__ = ("tree", "animated")

    def __init__(self, spec):
        self.animated = False
        self.tree = self._build(spec)

    def _build(self, o):
        if _anim_leaf(o):
            self.animated = True
            return Param(o)
        if isinstance(o, dict):
            return {k: self._build(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            if o and any(_anim_leaf(x) for x in o) and all(
                    _anim_leaf(x) or isinstance(x, (int, float)) for x in o):
                self.animated = True
                return Param(list(o))
            return [self._build(x) for x in o]
        return o

    def is_animated(self):
        return self.animated

    def evaluate(self, frame=0.0, ctx=None):
        def ev(o):
            if isinstance(o, Param):
                v = o.evaluate(frame, ctx)
                return v.tolist() if isinstance(v, np.ndarray) else v
            if isinstance(o, dict):
                return {k: ev(v) for k, v in o.items()}
            if isinstance(o, list):
                return [ev(x) for x in o]
            return o
        return ev(self.tree)
