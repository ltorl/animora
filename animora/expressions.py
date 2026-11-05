"""Safe expression language for animating parameters (Maya-expression / AE-expression style).

Any animatable value in a scene file may be a string starting with ``=`` (or a dict
``{"expr": "..."}``), e.g. ``"=sin(t*2*pi)*3"``. Expressions are parsed with Python's ``ast``
module and evaluated against a whitelist - no attribute access to dunder names, no imports,
no calls to anything outside the documented namespace.

Variables available
-------------------
``t``           time in seconds (``frame / fps``)
``frame``/``f`` current frame (float, sub-frame during motion blur)
``fps``         frames per second
``start``,``end`` scene frame range
``u``           normalised time ``(frame-start)/(end-start)`` in [0,1]
``pi``, ``tau``, ``e``
user vars      anything in the scene's ``"variables"`` block

Functions
---------
math: ``sin cos tan asin acos atan atan2 sinh cosh tanh sqrt exp log log10 pow abs floor
ceil round min max sign fmod hypot radians degrees``

animation: ``clamp(x,lo,hi)  lerp(a,b,u)  smoothstep(e0,e1,x)  linstep(e0,e1,x)
remap(x,a0,a1,b0,b1)  step(edge,x)  pulse(a,b,x)  fract(x)  mod(a,b)
ease(name,u)`` (any easing from :mod:`animora.curves`)
``noise(x[,y,z])  fbm(x[,y,z,octaves])  turbulence(...)  wiggle(freq,amp[,octaves,seed])``
``rand(seed)`` (deterministic, [0,1))  ``choose(i, a, b, c, ...)``
``bounce(t, height, period)`` (absolute-sine hop) ``spring(t, freq, damping)``
``saw(x)  tri(x)  square(x)``  (period 1 waves in [0,1])
vectors: ``vec(x,y,z)  length(v)  normalize(v)  dot(a,b)  cross(a,b)  mix(a,b,u)``
scene: ``get("node.attr")`` value of another node's channel at the current frame
(e.g. ``get("ball.translate")[1]``) and ``at(frame, "node.attr")``.

Lists ``[a, b, c]`` produce vectors; conditional ``a if cond else b`` and comparison /
boolean operators are supported.
"""
from __future__ import annotations

import ast
import math
import operator

import numpy as np

from . import noise as _noise
from .curves import get_easing


class ExpressionError(ValueError):
    pass


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_CMPOPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi) if isinstance(x, np.ndarray) else min(max(x, lo), hi)


def _smoothstep(e0, e1, x):
    u = _clamp((x - e0) / (e1 - e0) if e1 != e0 else float(x >= e1))
    return u * u * (3 - 2 * u)


def _linstep(e0, e1, x):
    return _clamp((x - e0) / (e1 - e0) if e1 != e0 else float(x >= e1))


def _remap(x, a0, a1, b0, b1):
    if a1 == a0:
        return b0
    return b0 + (x - a0) * (b1 - b0) / (a1 - a0)


def _ease(name, u):
    f = get_easing(name)
    if f is None:
        raise ExpressionError(f"unknown easing {name!r}")
    return f(_clamp(float(u)))


def _choose(i, *opts):
    if not opts:
        raise ExpressionError("choose() needs options")
    return opts[int(i) % len(opts)]


def _bounce(t, height=1.0, period=1.0):
    return height * abs(math.sin(math.pi * t / period))


def _spring(t, freq=2.0, damping=4.0):
    """Damped spring settling from 0 to 1."""
    if t <= 0:
        return 0.0
    return 1.0 - math.exp(-damping * t) * math.cos(2 * math.pi * freq * t)


def _vec(*a):
    if len(a) == 1:
        return np.asarray(a[0], np.float64)
    return np.array(a, np.float64)


def _normalize(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _fract(x):
    return x - math.floor(x)


BASE_NAMESPACE = {
    "pi": math.pi, "tau": math.tau, "e": math.e, "inf": math.inf,
    "True": True, "False": False, "None": None,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
    "pow": np.power, "abs": abs, "floor": math.floor, "ceil": math.ceil, "round": round,
    "min": min, "max": max, "sign": np.sign, "fmod": math.fmod, "hypot": math.hypot,
    "radians": math.radians, "degrees": math.degrees, "int": int, "float": float,
    "clamp": _clamp, "lerp": lambda a, b, u: a + (b - a) * u, "mix": lambda a, b, u: a + (b - a) * u,
    "smoothstep": _smoothstep, "linstep": _linstep, "remap": _remap,
    "step": lambda edge, x: float(x >= edge), "pulse": lambda a, b, x: float(a <= x < b),
    "fract": _fract, "mod": lambda a, b: a % b, "ease": _ease, "choose": _choose,
    "noise": lambda x, y=0.0, z=0.0, seed=0: _noise.perlin3(x, y, z, int(seed)),
    "fbm": lambda x, y=0.0, z=0.0, octaves=4, seed=0: _noise.fbm(x, y, z, int(octaves), seed=int(seed)),
    "turbulence": lambda x, y=0.0, z=0.0, octaves=4, seed=0: _noise.turbulence(x, y, z, int(octaves), seed=int(seed)),
    "rand": lambda *a: _noise.hash01(*a),
    "bounce": _bounce, "spring": _spring,
    "saw": lambda x: _fract(x), "tri": lambda x: 1 - abs(2 * _fract(x) - 1),
    "square": lambda x: float(_fract(x) < 0.5),
    "vec": _vec, "length": lambda v: float(np.linalg.norm(v)), "normalize": _normalize,
    "dot": lambda a, b: float(np.dot(a, b)), "cross": lambda a, b: np.cross(a, b),
}

_ALLOWED_ATTRS = {"x": 0, "y": 1, "z": 2, "r": 0, "g": 1, "b": 2, "w": 3}


class Expression:
    """A compiled expression. Call :meth:`evaluate` with a context dict."""

    def __init__(self, source: str):
        src = source.strip()
        if src.startswith("="):
            src = src[1:]
        self.source = src
        try:
            self.tree = ast.parse(src, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"syntax error in expression {source!r}: {exc.msg}") from None
        self._check(self.tree)

    def _check(self, node):
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
                   ast.Call, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.Subscript,
                   ast.Attribute, ast.keyword, ast.Slice, ast.And, ast.Or)
        for n in ast.walk(node):
            if isinstance(n, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.expr_context)):
                continue
            if not isinstance(n, allowed):
                raise ExpressionError(f"construct {type(n).__name__} not allowed in expression {self.source!r}")
            if isinstance(n, ast.Attribute) and n.attr not in _ALLOWED_ATTRS:
                raise ExpressionError(f"attribute .{n.attr} not allowed (use .x .y .z / .r .g .b)")
            if isinstance(n, ast.Name) and n.id.startswith("_"):
                raise ExpressionError("names starting with '_' are not allowed")

    def evaluate(self, ctx: dict):
        return self._eval(self.tree.body, ctx)

    __call__ = evaluate

    def _eval(self, n, ctx):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float, str, bool)) or n.value is None:
                return n.value
            raise ExpressionError(f"constant {n.value!r} not allowed")
        if isinstance(n, ast.Name):
            if n.id in ctx:
                return ctx[n.id]
            if n.id in BASE_NAMESPACE:
                return BASE_NAMESPACE[n.id]
            raise ExpressionError(f"unknown name {n.id!r} in expression {self.source!r}")
        if isinstance(n, ast.BinOp):
            a, b = self._eval(n.left, ctx), self._eval(n.right, ctx)
            if isinstance(a, list):
                a = np.asarray(a, np.float64)
            if isinstance(b, list):
                b = np.asarray(b, np.float64)
            return _BINOPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp):
            v = self._eval(n.operand, ctx)
            if isinstance(v, list):
                v = np.asarray(v, np.float64)
            return _UNOPS[type(n.op)](v)
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                v = True
                for x in n.values:
                    v = self._eval(x, ctx)
                    if not v:
                        return v
                return v
            v = False
            for x in n.values:
                v = self._eval(x, ctx)
                if v:
                    return v
            return v
        if isinstance(n, ast.Compare):
            left = self._eval(n.left, ctx)
            for op, comp in zip(n.ops, n.comparators):
                right = self._eval(comp, ctx)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self._eval(n.body, ctx) if self._eval(n.test, ctx) else self._eval(n.orelse, ctx)
        if isinstance(n, (ast.List, ast.Tuple)):
            vals = [self._eval(e, ctx) for e in n.elts]
            try:
                return np.asarray(vals, np.float64)
            except (TypeError, ValueError):
                return vals
        if isinstance(n, ast.Subscript):
            base = self._eval(n.value, ctx)
            idx = self._eval(n.slice, ctx) if not isinstance(n.slice, ast.Slice) else slice(
                *(self._eval(s, ctx) if s is not None else None for s in (n.slice.lower, n.slice.upper, n.slice.step)))
            if isinstance(idx, float):
                idx = int(idx)
            return base[idx]
        if isinstance(n, ast.Attribute):
            base = self._eval(n.value, ctx)
            return base[_ALLOWED_ATTRS[n.attr]]
        if isinstance(n, ast.Call):
            fn = self._eval(n.func, ctx)
            if not callable(fn):
                raise ExpressionError(f"{ast.unparse(n.func)} is not callable")
            args = [self._eval(a, ctx) for a in n.args]
            kwargs = {k.arg: self._eval(k.value, ctx) for k in n.keywords}
            return fn(*args, **kwargs)
        raise ExpressionError(f"unsupported expression node {type(n).__name__}")

    def __repr__(self):
        return f"Expression({self.source!r})"


_CACHE: dict[str, Expression] = {}


def compile_expr(src: str) -> Expression:
    e = _CACHE.get(src)
    if e is None:
        e = Expression(src)
        _CACHE[src] = e
    return e


def evaluate(src: str, **ctx):
    """Evaluate expression ``src`` with variables ``ctx`` (convenience helper)."""
    return compile_expr(src).evaluate(ctx)


def make_context(frame, fps=24.0, start=1.0, end=100.0, vars=None, getter=None):
    """Build the standard evaluation context for a frame."""
    span = max(1e-9, end - start)
    ctx = {
        "frame": float(frame), "f": float(frame), "t": float(frame) / fps, "time": float(frame) / fps,
        "fps": float(fps), "start": float(start), "end": float(end), "u": (float(frame) - start) / span,
    }
    if vars:
        ctx.update(vars)
    if getter is not None:
        ctx["get"] = lambda path: getter(path, frame)
        ctx["at"] = lambda fr, path: getter(path, fr)
    t = ctx["t"]
    ctx["wiggle"] = lambda freq=1.0, amp=1.0, octaves=2, seed=0: _noise.wiggle(t, freq, amp, int(octaves), int(seed))
    return ctx
