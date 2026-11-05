"""Procedural modelling: signed-distance (SDF) CSG, metaballs, 3-D text, sweep, loft, and
remesh / boolean / decimate tools.

Everything here returns an :class:`animora.geometry.Mesh`, so the results work with materials,
modifiers, skinning, cloth, export and the renderer like any other mesh.

SDF modelling (``primitive: "sdf"``)
------------------------------------
``shapes`` is a list evaluated in order. Each shape has a ``type``, its own parameters, an
optional transform (``translate`` / ``rotate`` (degrees) / ``scale``) and how it combines with
everything before it: ``op`` = ``union`` (default) / ``subtract`` / ``intersect``, ``blend`` =
smooth-blend radius (0 = sharp CSG). Per-shape shaping: ``round`` (inflate / round edges),
``shell`` (hollow with that wall thickness), ``displace`` ({amplitude, frequency, octaves, seed}
noise), ``twist`` (degrees per unit along Y), ``bend`` (degrees per unit along X, around Z),
``mirror`` ("x", "xz", ...), ``repeat`` (cell spacing [x,y,z], 0 = no repeat on that axis) with
optional ``count`` limit. ``group`` shapes nest a sub-``shapes`` list (a CSG tree).

Shape types: sphere(radius), box(size [x,y,z]), torus(major_radius, minor_radius),
capsule(radius, height), cylinder(radius, height), cone(radius, height, radius_top),
ellipsoid(radii), octahedron(radius), plane(normal, offset), text(text, size, depth, font,
bevel, align), mesh(node: another mesh node, or verts/tris), metaballs(balls, threshold), group.

The surface is extracted with *surface nets* on a ``resolution``-cell grid (longest axis)
and each vertex is then projected exactly onto the zero level set (Newton steps on the SDF),
so curved surfaces are smooth and accurate even at moderate resolutions.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np

from . import mathutil as mu
from .geometry import Mesh

# --------------------------------------------------------------------------------- SDF helpers


def _len(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_sphere(p, radius=0.5):
    return _len(p) - radius


def sd_box(p, size=(1, 1, 1)):
    b = np.broadcast_to(np.asarray(size, float).ravel(), (3,)) / 2
    q = np.abs(p) - b
    return _len(np.maximum(q, 0)) + np.minimum(np.max(q, axis=-1), 0)


def sd_torus(p, major_radius=0.5, minor_radius=0.15):
    q = np.stack([np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - major_radius, p[:, 1]], -1)
    return _len(q) - minor_radius


def sd_capsule(p, radius=0.25, height=1.0):
    h = max(height - 2 * radius, 0) / 2
    q = p.copy()
    q[:, 1] = p[:, 1] - np.clip(p[:, 1], -h, h)
    return _len(q) - radius


def sd_cylinder(p, radius=0.5, height=1.0):
    d = np.stack([np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - radius, np.abs(p[:, 1]) - height / 2], -1)
    return np.minimum(np.max(d, -1), 0) + _len(np.maximum(d, 0))


def sd_cone(p, radius=0.5, height=1.0, radius_top=0.0):
    """Capped cone centred on the origin, base (``radius``) at -height/2 (IQ)."""
    h = height / 2
    r1, r2 = radius, radius_top
    q = np.stack([np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2), p[:, 1]], -1)
    k1 = np.array([r2, h])
    k2 = np.array([r2 - r1, 2 * h])
    ca = np.stack([q[:, 0] - np.minimum(q[:, 0], np.where(q[:, 1] < 0, r1, r2)), np.abs(q[:, 1]) - h], -1)
    t = np.clip(((k1 - q) @ k2) / max(float(k2 @ k2), 1e-12), 0, 1)
    cb = q - k1 + k2 * t[:, None]
    s = np.where((cb[:, 0] < 0) & (ca[:, 1] < 0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(np.sum(ca * ca, -1), np.sum(cb * cb, -1)))


def sd_ellipsoid(p, radii=(0.5, 0.3, 0.4)):
    r = np.broadcast_to(np.asarray(radii, float).ravel(), (3,))
    k0 = _len(p / r)
    k1 = _len(p / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def sd_octahedron(p, radius=0.5):
    return (np.sum(np.abs(p), -1) - radius) * 0.57735027


def sd_plane(p, normal=(0, 1, 0), offset=0.0):
    n = mu.normalize(np.asarray(normal, float))
    return p @ n - offset


SHAPES = {"sphere": sd_sphere, "box": sd_box, "cube": sd_box, "torus": sd_torus, "capsule": sd_capsule,
          "cylinder": sd_cylinder, "cone": sd_cone, "ellipsoid": sd_ellipsoid, "octahedron": sd_octahedron,
          "plane": sd_plane}

_SHAPE_BOUND = {  # local bounding half-extent of each shape (for automatic grid bounds)
    "sphere": lambda s: [s.get("radius", 0.5)] * 3,
    "box": lambda s: list(np.broadcast_to(np.asarray(s.get("size", 1.0), float), (3,)) / 2),
    "torus": lambda s: [s.get("major_radius", 0.5) + s.get("minor_radius", 0.15), s.get("minor_radius", 0.15),
                        s.get("major_radius", 0.5) + s.get("minor_radius", 0.15)],
    "capsule": lambda s: [s.get("radius", 0.25), max(s.get("height", 1.0) / 2, s.get("radius", 0.25)),
                          s.get("radius", 0.25)],
    "cylinder": lambda s: [s.get("radius", 0.5), s.get("height", 1.0) / 2, s.get("radius", 0.5)],
    "cone": lambda s: [max(s.get("radius", 0.5), s.get("radius_top", 0.0)), s.get("height", 1.0) / 2,
                       max(s.get("radius", 0.5), s.get("radius_top", 0.0))],
    "ellipsoid": lambda s: list(np.broadcast_to(np.asarray(s.get("radii", [0.5, 0.3, 0.4]), float), (3,))),
    "octahedron": lambda s: [s.get("radius", 0.5)] * 3,
}
_SHAPE_BOUND["cube"] = _SHAPE_BOUND["box"]

_NON_PARAM = ("type", "op", "blend", "translate", "rotate", "scale", "round", "shell", "displace", "twist", "bend",
              "mirror", "repeat", "count", "shapes", "rotate_order", "name", "enabled", "bounds", "resolution")


# --------------------------------------------------------------------------------- text field
_FONT_DIRS = ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
              os.path.expanduser("~/Library/Fonts"), "/usr/share/fonts", "/usr/local/share/fonts",
              "C:/Windows/Fonts"]
_FONT_DEFAULTS = ["Helvetica.ttc", "HelveticaNeue.ttc", "Arial.ttf", "Arial Bold.ttf", "DejaVuSans.ttf",
                  "DejaVuSans-Bold.ttf", "LiberationSans-Regular.ttf", "arial.ttf"]


def find_font(name=None):
    """Resolve a font name or path to a file (searches the system font folders)."""
    if name and os.path.isfile(name):
        return name
    cands = []
    if name:
        n = str(name)
        cands += [n, n + ".ttf", n + ".ttc", n + ".otf"]
    else:
        cands += _FONT_DEFAULTS
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for c in cands:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                return p
        if name:  # recursive, case-insensitive match
            low = [c.lower() for c in cands]
            for root, _dirs, files in os.walk(d):
                for f in files:
                    if f.lower() in low:
                        return os.path.join(root, f)
    return None


def _edt_1d_sq(g):
    """Squared distance transform along the last axis: min_x' (x - x')^2 + g(x')^2, brute force."""
    H, W = g.shape
    x = np.arange(W, dtype=np.float32)
    out = np.empty_like(g)
    step = max(1, int(4e7 // max(W * W, 1)))
    for s in range(0, H, step):
        gg = g[s:s + step]
        out[s:s + step] = np.min((x[None, :, None] - x[None, None, :]) ** 2 + gg[:, None, :] ** 2, axis=2)
    return out


def _edt(mask):
    """Exact Euclidean distance (pixels) from every pixel to the nearest True pixel."""
    H, W = mask.shape
    INF = np.float32(H + W)
    g = np.where(mask, 0, INF).astype(np.float32)
    for i in range(1, H):  # column pass: distance to nearest set pixel in the column
        g[i] = np.minimum(g[i], g[i - 1] + 1)
    for i in range(H - 2, -1, -1):
        g[i] = np.minimum(g[i], g[i + 1] + 1)
    return np.sqrt(_edt_1d_sq(g))


_TEXT_CACHE: "OrderedDict[str, tuple]" = OrderedDict()


def text_field(text, size=1.0, font=None, align="center", line_spacing=1.2, px=None):
    """2-D signed distance field of ``text`` (cap height ~ ``size``). Returns (sdf2d, x0, y0, cell)
    where sdf2d[row, col] is the distance at world (x0 + col*cell, y0 - row*cell)."""
    key = json.dumps([text, size, font, align, line_spacing, px])
    if key in _TEXT_CACHE:
        _TEXT_CACHE.move_to_end(key)
        return _TEXT_CACHE[key]
    from PIL import Image, ImageDraw, ImageFont
    px = int(px or 160)
    path = find_font(font)
    try:
        f = ImageFont.truetype(path, px) if path else ImageFont.load_default(size=px)
    except Exception:  # noqa: BLE001
        f = ImageFont.load_default(size=px)
    lines = str(text).split("\n")
    asc, desc = f.getmetrics()
    lh = int((asc + desc) * line_spacing)
    widths = [max(1, int(f.getlength(ln))) for ln in lines]
    pad = px // 4
    W = max(widths) + 2 * pad
    H = lh * (len(lines) - 1) + asc + desc + 2 * pad
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        x = pad + {"left": 0, "right": max(widths) - widths[i]}.get(align, (max(widths) - widths[i]) // 2)
        dr.text((x, pad + i * lh), ln, fill=255, font=f)
    a = np.asarray(img, np.float32) / 255.0
    inside = a >= 0.5
    d_out = _edt(inside)
    d_in = _edt(~inside)
    sd = np.where(inside, -(d_in - 0.5), d_out - 0.5).astype(np.float32)
    # cap height (approx. height of 'H') -> size
    try:
        bb = f.getbbox("H")
        cap = max(1, bb[3] - bb[1])
    except Exception:  # noqa: BLE001
        cap = asc * 0.72
    cell = float(size) / cap
    # origin: horizontally centred / left / right aligned, baseline of the first line at y=0
    x0 = -{"left": pad, "right": pad + max(widths)}.get(align, pad + max(widths) / 2.0) * cell
    y0 = (pad + asc) * cell
    out = (sd * cell, x0, y0, cell)
    _TEXT_CACHE[key] = out
    if len(_TEXT_CACHE) > 16:
        _TEXT_CACHE.popitem(last=False)
    return out


def _bilinear(img, r, c):
    H, W = img.shape
    r = np.clip(r, 0, H - 1.001)
    c = np.clip(c, 0, W - 1.001)
    r0 = r.astype(np.int64)
    c0 = c.astype(np.int64)
    fr, fc = r - r0, c - c0
    return ((img[r0, c0] * (1 - fc) + img[r0, c0 + 1] * fc) * (1 - fr)
            + (img[r0 + 1, c0] * (1 - fc) + img[r0 + 1, c0 + 1] * fc) * fr)


def sd_text(p, text="Animora", size=1.0, depth=0.2, font=None, bevel=0.0, align="center", line_spacing=1.2):
    sd2, x0, y0, cell = text_field(text, size, font, align, line_spacing)
    H, W = sd2.shape
    c = (p[:, 0] - x0) / cell
    r = (y0 - p[:, 1]) / cell
    d2 = _bilinear(sd2, r, c)
    # outside the raster: add the distance to the raster box
    ox = np.maximum(np.maximum(-c, c - (W - 1)), 0) * cell
    oy = np.maximum(np.maximum(-r, r - (H - 1)), 0) * cell
    d2 = d2 + np.sqrt(ox * ox + oy * oy)
    bevel = float(min(max(bevel, 0.0), depth / 2 * 0.999))
    w = np.stack([d2 + bevel, np.abs(p[:, 2]) - (depth / 2 - bevel)], -1)
    return np.minimum(np.max(w, -1), 0) + _len(np.maximum(w, 0)) - bevel


def _text_bound(s):
    sd2, x0, y0, cell = text_field(s.get("text", "Animora"), s.get("size", 1.0), s.get("font"),
                                   s.get("align", "center"), s.get("line_spacing", 1.2))
    H, W = sd2.shape
    lo = np.array([x0, y0 - (H - 1) * cell, -s.get("depth", 0.2) / 2])
    hi = np.array([x0 + (W - 1) * cell, y0, s.get("depth", 0.2) / 2])
    # shrink to the actual glyph extents
    rows = np.where((sd2 < cell).any(1))[0]
    cols = np.where((sd2 < cell).any(0))[0]
    if rows.size and cols.size:
        lo[0], hi[0] = x0 + cols[0] * cell, x0 + cols[-1] * cell
        hi[1], lo[1] = y0 - rows[0] * cell, y0 - rows[-1] * cell
    return lo, hi


# --------------------------------------------------------------------------------- metaballs
def metaball_field(p, balls, threshold=0.5):
    """Wyvill metaball field mapped to a distance-like function (negative inside)."""
    F = np.zeros(len(p))
    for b in balls:
        c = np.asarray(b.get("center", b.get("translate", [0, 0, 0])), float)
        R = float(b.get("radius", 0.5))
        s = float(b.get("strength", 1.0))
        r2 = np.sum((p - c) ** 2, -1) / (R * R)
        F += s * np.where(r2 < 1, (1 - r2) ** 3, 0.0)
    # scale so the gradient is roughly 1 near the surface
    rmean = np.mean([float(b.get("radius", 0.5)) for b in balls]) if balls else 1.0
    return (threshold - F) * rmean * 0.6 / max(threshold, 1e-6) ** 0.5


# --------------------------------------------------------------------------------- shape tree
class _Shape:
    def __init__(self, spec):
        self.spec = spec
        self.type = str(spec.get("type", "sphere")).lower()
        self.op = str(spec.get("op", "union")).lower()
        self.blend = float(spec.get("blend", 0.0))
        t = spec.get("translate", [0, 0, 0])
        r = spec.get("rotate", [0, 0, 0])
        s = spec.get("scale", 1.0)
        s = np.broadcast_to(np.asarray(s, float).ravel() if np.ndim(s) else np.full(3, float(s)), (3,))
        self.uscale = float(np.min(np.abs(s)))
        R = mu.euler_matrix(np.asarray(r, float), spec.get("rotate_order", "xyz"))
        self.M = mu.translate(np.asarray(t, float)) @ R @ mu.scale(s)
        self.Minv = np.linalg.inv(self.M)
        self.children = [_Shape(c) for c in spec.get("shapes", []) if c.get("enabled", True)] \
            if self.type == "group" else []
        self.params = {k: v for k, v in spec.items() if k not in _NON_PARAM}
        if self.type == "mesh":
            self._mesh_init()
        if self.type not in SHAPES and self.type not in ("group", "text", "mesh", "metaballs"):
            raise ValueError(f"sdf: unknown shape type {self.type!r}; available: "
                             f"{sorted(list(SHAPES) + ['group', 'text', 'mesh', 'metaballs'])}")

    def _mesh_init(self):
        V = np.asarray(self.spec.get("verts"), float).reshape(-1, 3)
        T = np.asarray(self.spec.get("tris"), np.int64).reshape(-1, 3)
        self.mesh_vt = (V, T)

    def local_eval(self, q):
        t = self.type
        if t == "group":
            return evaluate_shapes(self.children, q)
        if t == "text":
            return sd_text(q, **{k: v for k, v in self.params.items()
                                 if k in ("text", "size", "depth", "font", "bevel", "align", "line_spacing")})
        if t == "mesh":
            from .render import native as nat
            V, T = self.mesh_vt
            return nat.mesh_sdf(V, T, q).astype(np.float64)
        if t == "metaballs":
            return metaball_field(q, self.params.get("balls", []), float(self.params.get("threshold", 0.5)))
        fn = SHAPES[t]
        import inspect
        sig = inspect.signature(fn).parameters
        return fn(q, **{k: v for k, v in self.params.items() if k in sig})

    def eval(self, p):
        sp = self.spec
        q = mu.transform_points(self.Minv, p)
        if sp.get("mirror"):
            for ax in str(sp["mirror"]).lower():
                if ax in "xyz":
                    i = "xyz".index(ax)
                    q[:, i] = np.abs(q[:, i])
        if sp.get("repeat") is not None:
            c = np.broadcast_to(np.asarray(sp["repeat"], float).ravel(), (3,))
            cnt = sp.get("count")
            for i in range(3):
                if c[i] > 0:
                    k = np.round(q[:, i] / c[i])
                    if cnt is not None:
                        n = np.broadcast_to(np.asarray(cnt, float).ravel(), (3,))[i]
                        k = np.clip(k, -np.floor((n - 1) / 2), np.ceil((n - 1) / 2))
                    q[:, i] = q[:, i] - c[i] * k
        if sp.get("twist"):
            a = np.radians(float(sp["twist"])) * q[:, 1]
            ca, sa = np.cos(a), np.sin(a)
            x, z = q[:, 0].copy(), q[:, 2].copy()
            q[:, 0], q[:, 2] = ca * x - sa * z, sa * x + ca * z
        if sp.get("bend"):
            a = np.radians(float(sp["bend"])) * q[:, 0]
            ca, sa = np.cos(a), np.sin(a)
            x, y = q[:, 0].copy(), q[:, 1].copy()
            q[:, 0], q[:, 1] = ca * x - sa * y, sa * x + ca * y
        d = self.local_eval(q) * self.uscale
        if sp.get("twist") or sp.get("bend"):
            d = d * 0.7  # domain warps stretch distances: stay conservative
        if sp.get("round"):
            d = d - float(sp["round"])
        if sp.get("shell"):
            d = np.abs(d) - float(sp["shell"]) / 2
        if sp.get("displace"):
            dp = sp["displace"]
            if not isinstance(dp, dict):
                dp = {"amplitude": float(dp)}
            from .noise import fbm
            fr = float(dp.get("frequency", 3.0))
            amp = float(dp.get("amplitude", 0.05))
            n = fbm(p[:, 0] * fr, p[:, 1] * fr, p[:, 2] * fr, octaves=int(dp.get("octaves", 3)),
                    seed=int(dp.get("seed", 0)))
            d = d + amp * np.asarray(n, float)
        return d

    def bounds(self):
        """World AABB (lo, hi) of the additive extent, or None for unbounded shapes."""
        t = self.type
        if t == "plane":
            return None
        if t == "group":
            bs = [c.bounds() for c in self.children if c.op == "union"]
            bs = [b for b in bs if b is not None]
            if not bs:
                return None
            lo = np.min([b[0] for b in bs], 0)
            hi = np.max([b[1] for b in bs], 0)
        elif t == "text":
            lo, hi = _text_bound(self.params)
        elif t == "mesh":
            V = self.mesh_vt[0]
            lo, hi = V.min(0), V.max(0)
        elif t == "metaballs":
            bl = self.params.get("balls", [])
            if not bl:
                return None
            cs = np.array([b.get("center", b.get("translate", [0, 0, 0])) for b in bl], float)
            rs = np.array([float(b.get("radius", 0.5)) for b in bl])
            lo, hi = (cs - rs[:, None]).min(0), (cs + rs[:, None]).max(0)
        else:
            e = np.asarray(_SHAPE_BOUND[t](self.params), float)
            lo, hi = -e, e
        sp = self.spec
        if sp.get("repeat") is not None and sp.get("count") is not None:
            c = np.broadcast_to(np.asarray(sp["repeat"], float).ravel(), (3,))
            n = np.broadcast_to(np.asarray(sp["count"], float).ravel(), (3,))
            ext = np.where(c > 0, c * (n - 1) / 2 + c * 0.5, 0)
            lo, hi = lo - ext, hi + ext
        elif sp.get("repeat") is not None:
            return None
        if sp.get("mirror"):
            for ax in str(sp["mirror"]).lower():
                if ax in "xyz":
                    i = "xyz".index(ax)
                    m = max(abs(lo[i]), abs(hi[i]))
                    lo[i], hi[i] = -m, m
        pad = float(sp.get("round", 0) or 0) + float(sp.get("shell", 0) or 0)
        dp = sp.get("displace")
        if dp:
            pad += abs(float(dp.get("amplitude", 0.05) if isinstance(dp, dict) else dp)) * 1.5
        if sp.get("twist") or sp.get("bend"):
            r = float(np.max(np.abs(np.concatenate([lo, hi]))))
            lo, hi = np.full(3, -r * 1.5), np.full(3, r * 1.5)
        lo, hi = lo - pad, hi + pad
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        cw = mu.transform_points(self.M, corners)
        return cw.min(0) - self.blend, cw.max(0) + self.blend


def evaluate_shapes(shapes, p):
    d = None
    for s in shapes:
        ds = s.eval(p)
        if d is None:
            d = ds if s.op != "subtract" else -ds
        elif s.op == "subtract":
            d = smax(d, -ds, s.blend)
        elif s.op == "intersect":
            d = smax(d, ds, s.blend)
        else:
            d = smin(d, ds, s.blend)
    return np.full(len(p), 1e3) if d is None else d


def shapes_bounds(shapes):
    bs = [s.bounds() for s in shapes if s.op in ("union",)]
    if shapes and shapes[0].op == "intersect":
        bs = [shapes[0].bounds()]
    bs = [b for b in bs if b is not None]
    inter = [s.bounds() for s in shapes if s.op == "intersect"]
    inter = [b for b in inter if b is not None]
    if not bs and not inter:
        return None
    if bs:
        lo = np.min([b[0] for b in bs], 0)
        hi = np.max([b[1] for b in bs], 0)
    else:
        lo, hi = inter[0]
    for b in inter:
        lo, hi = np.maximum(lo, b[0]), np.minimum(hi, b[1])
    return lo, hi


# --------------------------------------------------------------------------------- surface nets
_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def surface_nets(F, lo, cell, func=None, project_iters=3, smooth_angle=60.0, name="sdf"):
    """Polygonise the zero level set of the grid ``F[x, y, z]`` (negative inside).

    ``lo`` = world position of F[0,0,0], ``cell`` = spacing (scalar or 3-vector). With ``func``
    (points -> distance) vertices are projected onto the exact surface."""
    F = np.asarray(F, np.float64)
    nx, ny, nz = F.shape
    cell = np.broadcast_to(np.asarray(cell, float), (3,))
    lo = np.asarray(lo, float)
    inside = F < 0
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    # cell corner values
    cv = [F[i:i + cx, j:j + cy, k:k + cz] for i, j, k in _CORNERS]
    ci = [inside[i:i + cx, j:j + cy, k:k + cz] for i, j, k in _CORNERS]
    anyin = np.zeros((cx, cy, cz), bool)
    allin = np.ones((cx, cy, cz), bool)
    for c in ci:
        anyin |= c
        allin &= c
    active = anyin & ~allin
    if not active.any():
        return None
    acc = np.zeros((cx, cy, cz, 3))
    cnt = np.zeros((cx, cy, cz))
    for a, b in _EDGES:
        fa, fb = cv[a], cv[b]
        cross = (ci[a] != ci[b])
        t = np.where(cross, fa / np.where(cross, fa - fb, 1.0), 0.0)
        pa, pb = _CORNERS[a], _CORNERS[b]
        for k in range(3):
            acc[..., k] += np.where(cross, pa[k] + (pb[k] - pa[k]) * t, 0.0)
        cnt += cross
    idx = -np.ones((cx, cy, cz), np.int64)
    act = np.nonzero(active)
    nv = len(act[0])
    idx[act] = np.arange(nv)
    rel = acc[act] / cnt[act][:, None]
    V = lo + (np.stack(act, -1) + rel) * cell
    quads = []
    # x-edges: (i,j,k)-(i+1,j,k) for 1<=j<=ny-2, 1<=k<=nz-2, cells (i, j-1..j, k-1..k)
    e = inside[:-1, 1:-1, 1:-1] != inside[1:, 1:-1, 1:-1]
    i, j, k = np.nonzero(e)
    j, k = j + 1, k + 1
    flip = inside[i, j, k]
    q = np.stack([idx[i, j - 1, k - 1], idx[i, j, k - 1], idx[i, j, k], idx[i, j - 1, k]], -1)
    quads.append(np.where(flip[:, None], q, q[:, ::-1]))
    # y-edges
    e = inside[1:-1, :-1, 1:-1] != inside[1:-1, 1:, 1:-1]
    i, j, k = np.nonzero(e)
    i, k = i + 1, k + 1
    flip = inside[i, j, k]
    q = np.stack([idx[i - 1, j, k - 1], idx[i - 1, j, k], idx[i, j, k], idx[i, j, k - 1]], -1)
    quads.append(np.where(flip[:, None], q, q[:, ::-1]))
    # z-edges
    e = inside[1:-1, 1:-1, :-1] != inside[1:-1, 1:-1, 1:]
    i, j, k = np.nonzero(e)
    i, j = i + 1, j + 1
    flip = inside[i, j, k]
    q = np.stack([idx[i - 1, j - 1, k], idx[i, j - 1, k], idx[i, j, k], idx[i - 1, j, k]], -1)
    quads.append(np.where(flip[:, None], q, q[:, ::-1]))
    Q = np.concatenate(quads, 0)
    Q = Q[(Q >= 0).all(1)]
    if not len(Q):
        return None
    if func is not None and project_iters > 0:
        h = float(cell.min())
        eps = h * 0.25
        for _ in range(project_iters):
            d = func(V)
            g = np.stack([(func(V + eps * np.eye(3)[a]) - func(V - eps * np.eye(3)[a])) / (2 * eps)
                          for a in range(3)], -1)
            gg = np.maximum(np.sum(g * g, -1), 1e-12)
            step = (d / gg)[:, None] * g
            sl = _len(step)
            step *= np.minimum(1.0, h / np.maximum(sl, 1e-12))[:, None]
            V = V - step
    # compact unused verts
    used = np.zeros(len(V), bool)
    used[Q.ravel()] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(int(used.sum()))
    V = V[used]
    Q = remap[Q]
    # split each quad along its shorter diagonal (better shading on curved surfaces)
    d02 = np.sum((V[Q[:, 0]] - V[Q[:, 2]]) ** 2, -1)
    d13 = np.sum((V[Q[:, 1]] - V[Q[:, 3]]) ** 2, -1)
    a = np.where((d02 <= d13)[:, None], Q[:, [0, 1, 2]], Q[:, [1, 2, 3]])
    b = np.where((d02 <= d13)[:, None], Q[:, [0, 2, 3]], Q[:, [1, 3, 0]])
    T = np.concatenate([a, b], 0)
    return Mesh(V, np.full(len(T), 3), T.ravel(), smooth_angle=smooth_angle, name=name)


def mesh_function(func, lo, hi, resolution=64, smooth_angle=60.0, project=3, name="sdf"):
    """Polygonise ``func`` (points -> signed distance) inside the box [lo, hi]."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ext = np.maximum(hi - lo, 1e-6)
    res = int(np.clip(resolution, 4, 400))
    cell = float(ext.max()) / res
    lo = lo - 2 * cell
    n = np.ceil((ext + 4 * cell) / cell).astype(int) + 1
    xs, ys, zs = [lo[a] + np.arange(n[a]) * cell for a in range(3)]
    F = np.empty((n[0], n[1], n[2]))
    # evaluate in slabs to bound memory
    for i0 in range(0, n[0], max(1, int(2e6 // (n[1] * n[2])))):
        i1 = min(n[0], i0 + max(1, int(2e6 // (n[1] * n[2]))))
        X, Y, Z = np.meshgrid(xs[i0:i1], ys, zs, indexing="ij")
        F[i0:i1] = func(np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)).reshape(X.shape)
    m = surface_nets(F, lo, cell, func=func if project else None, project_iters=project,
                     smooth_angle=smooth_angle, name=name)
    if m is None:
        return Mesh(np.zeros((0, 3)), [], [], name=name)
    return m


_MESH_CACHE: "OrderedDict[str, Mesh]" = OrderedDict()


def _cached(key, build):
    if key in _MESH_CACHE:
        _MESH_CACHE.move_to_end(key)
        return _MESH_CACHE[key]
    m = build()
    _MESH_CACHE[key] = m
    if len(_MESH_CACHE) > 32:
        _MESH_CACHE.popitem(last=False)
    return m


def _jkey(*a):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.round(7).tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    return json.dumps(a, default=conv, sort_keys=True)


# --------------------------------------------------------------------------------- primitives
def sdf(shapes=(), resolution=64, bounds=None, smooth_angle=60.0, project=3):
    """Mesh a CSG tree of signed-distance shapes (see module docs)."""
    def build():
        tree = [_Shape(s) for s in shapes if s.get("enabled", True)]
        if not tree:
            return Mesh(np.zeros((0, 3)), [], [], name="sdf")
        if bounds is not None:
            lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        else:
            b = shapes_bounds(tree)
            if b is None:
                raise ValueError("sdf: cannot infer bounds (only unbounded shapes) - give 'bounds'")
            lo, hi = b
        return mesh_function(lambda p: evaluate_shapes(tree, p), lo, hi, resolution, smooth_angle, project)
    return _cached(_jkey("sdf", list(shapes), resolution, bounds, smooth_angle, project), build)


def metaballs(balls=(), threshold=0.5, resolution=64, smooth_angle=180.0):
    """Metaball (blobby) surface. ``balls``: [{center, radius, strength}] (strength < 0 subtracts)."""
    return sdf([{"type": "metaballs", "balls": list(balls), "threshold": threshold}], resolution,
               smooth_angle=smooth_angle)


def text(text="Animora", size=1.0, depth=0.2, bevel=0.02, font=None, align="center", line_spacing=1.2,
         resolution=None, smooth_angle=40.0):
    """Extruded, bevelled 3-D text. ``size`` = cap height; baseline of the first line at y=0,
    text faces +Z and is centred on z=0. ``font`` = font file or system font name."""
    if resolution is None:
        nchar = max(len(ln) for ln in str(text).split("\n"))
        resolution = int(np.clip(48 * nchar, 96, 400))
    return sdf([{"type": "text", "text": text, "size": size, "depth": depth, "bevel": bevel, "font": font,
                 "align": align, "line_spacing": line_spacing}], resolution, smooth_angle=smooth_angle)


# --------------------------------------------------------------------------------- sweep / loft
def profile_points(profile="circle", radius=0.1, segments=16):
    """2-D profile from a name ("circle", "square", "star", "rect"/[w,h] dict) or a point list."""
    if isinstance(profile, str):
        n = int(segments)
        if profile == "circle":
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return np.stack([np.cos(a), np.sin(a)], -1) * radius
        if profile in ("square", "box"):
            return np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float) * radius
        if profile == "star":
            k = max(3, n // 2)
            a = np.linspace(0, 2 * np.pi, 2 * k, endpoint=False) + np.pi / 2
            r = np.where(np.arange(2 * k) % 2 == 0, 1.0, 0.45) * radius
            return np.stack([np.cos(a) * r, np.sin(a) * r], -1)
        if profile in ("triangle", "tri"):
            a = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 2
            return np.stack([np.cos(a), np.sin(a)], -1) * radius
        raise ValueError(f"unknown profile {profile!r} (circle, square, star, triangle or a point list)")
    if isinstance(profile, dict):
        w, h = (profile.get("size") or [radius * 2, radius * 2])[:2]
        rr = float(profile.get("round", 0))
        if rr <= 0:
            return np.array([[w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2], [w / 2, -h / 2]], float)
        pts = []
        for cxs, cys, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
            c = np.array([cxs * (w / 2 - rr), cys * (h / 2 - rr)])
            for a in np.radians(np.linspace(a0, a0 + 90, 5)):
                pts.append(c + rr * np.array([np.cos(a), np.sin(a)]))
        return np.array(pts)
    return np.asarray(profile, float).reshape(-1, 2)


def _resample_closed(P, n, closed=True):
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]]) if closed else P
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=not closed)
    return np.stack([np.interp(t, s, Q[:, k]) for k in range(Q.shape[1])], -1)


def _frames(path, closed=False, up=None):
    """Parallel-transport frames along a polyline: tangents T, normals N, binormals B."""
    P = np.asarray(path, float)
    n = len(P)
    T = np.zeros_like(P)
    if closed:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    else:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    T = mu.normalize(T)
    if up is not None:
        u = mu.normalize(np.asarray(up, float))
        N0 = u - T[0] * (u @ T[0])
        if np.linalg.norm(N0) < 1e-6:
            up = None
    if up is None:
        a = np.array([0, 1, 0]) if abs(T[0][1]) < 0.9 else np.array([1, 0, 0])
        N0 = a - T[0] * (a @ T[0])
    N = np.zeros_like(P)
    N[0] = mu.normalize(N0)
    for i in range(1, n):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        if s < 1e-9:
            N[i] = N[i - 1]
            continue
        ang = np.arctan2(s, T[i - 1] @ T[i])
        k = v / s
        x = N[i - 1]
        N[i] = x * np.cos(ang) + np.cross(k, x) * np.sin(ang) + k * (k @ x) * (1 - np.cos(ang))
    if closed and n > 2:  # distribute the closing twist
        v = N[-1] - T[0] * (N[-1] @ T[0])
        a = np.arctan2(np.cross(mu.normalize(v), N[0]) @ T[0], mu.normalize(v) @ N[0])
        for i in range(n):
            th = a * i / n
            N[i] = N[i] * np.cos(th) + np.cross(T[i], N[i]) * np.sin(th)
    B = np.cross(T, N)
    return T, N, B


def _grid_mesh(rings, closed_u, closed_v, caps, name, smooth_angle):
    """rings: (nv, nu, 3) ring vertices. Builds quads + optional end caps + per-corner UVs."""
    nv, nu, _ = rings.shape
    V = rings.reshape(-1, 3)
    faces, uvs = [], []
    us = nu if closed_u else nu - 1
    vs = nv if closed_v else nv - 1
    for j in range(vs):
        j1 = (j + 1) % nv
        for i in range(us):
            i1 = (i + 1) % nu
            faces.append([j * nu + i, j * nu + i1, j1 * nu + i1, j1 * nu + i])
            uvs += [[i / us, j / vs], [(i + 1) / us, j / vs], [(i + 1) / us, (j + 1) / vs], [i / us, (j + 1) / vs]]
    V = list(V)
    if caps and not closed_v and closed_u:
        for j, rev in ((0, True), (nv - 1, False)):
            c = rings[j].mean(0)
            ci = len(V)
            V.append(c)
            for i in range(nu):
                a, b = j * nu + i, j * nu + (i + 1) % nu
                faces.append([ci, b, a] if rev else [ci, a, b])
                uvs += [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    V = np.asarray(V)
    # orient outward: positive signed volume for closed-ish surfaces
    m = Mesh(V, [len(f) for f in faces], [i for f in faces for i in f], uvs=np.asarray(uvs),
             smooth_angle=smooth_angle, name=name)
    return m


def sweep(path=((0, 0, 0), (0, 1, 0)), profile="circle", radius=0.1, segments=16, samples=8, smooth=True,
          closed=False, caps=True, twist=0.0, scale_start=1.0, scale_end=1.0, up=None, smooth_angle=60.0):
    """Sweep a 2-D ``profile`` along a 3-D ``path`` (parallel-transport frames, no flipping).

    ``twist`` (degrees over the whole path), ``scale_start`` / ``scale_end`` taper, ``closed``
    path (rings), ``caps`` on open ends, ``smooth`` = Catmull-Rom path smoothing with ``samples``
    points per segment."""
    from .scene import catmull_rom
    P = np.asarray(path, float).reshape(-1, 3)
    if smooth and len(P) > 2:
        P = catmull_rom(P, int(samples), closed)
        if closed and np.allclose(P[0], P[-1]):
            P = P[:-1]
    prof = profile_points(profile, radius, segments)
    closed_u = True
    T, N, B = _frames(P, closed, up)
    n = len(P)
    rings = np.zeros((n, len(prof), 3))
    for i in range(n):
        t = i / max(n - 1, 1)
        s = scale_start + (scale_end - scale_start) * t
        a = np.radians(twist) * t
        ca, sa = np.cos(a), np.sin(a)
        x = prof[:, 0] * ca - prof[:, 1] * sa
        y = prof[:, 0] * sa + prof[:, 1] * ca
        rings[i] = P[i] + s * (np.outer(x, N[i]) + np.outer(y, B[i]))
    return _grid_mesh(rings, closed_u, closed, caps, "sweep", smooth_angle)


def loft(sections=(), samples=8, points=None, smooth=True, closed=False, caps=True, open_sections=False,
         smooth_angle=60.0):
    """Skin a surface through ``sections``.

    Each section is a list of 3-D points, or a dict ``{"profile": <2-D profile or name>,
    "radius", "translate", "rotate", "scale"}`` (profile lies in the local XZ plane). Sections
    are resampled to the same point count (``points``, default the largest) by arc length, then
    interpolated with Catmull-Rom (``smooth``) using ``samples`` rings between sections."""
    from .scene import catmull_rom
    secs = []
    for s in sections:
        if isinstance(s, dict):
            pr = profile_points(s.get("profile", "circle"), float(s.get("radius", 0.5)), int(s.get("segments", 24)))
            P = np.stack([pr[:, 0], np.zeros(len(pr)), -pr[:, 1]], -1)
            sc = s.get("scale", 1.0)
            M = mu.compose(s.get("translate", [0, 0, 0]), s.get("rotate", [0, 0, 0]),
                           np.broadcast_to(np.asarray(sc, float), (3,)))
            P = mu.transform_points(M, P)
        else:
            P = np.asarray(s, float).reshape(-1, 3)
        secs.append(P)
    if len(secs) < 2:
        raise ValueError("loft needs at least two sections")
    npts = int(points or max(len(s) for s in secs))
    secs = np.stack([_resample_closed(s, npts, not open_sections) for s in secs])  # (ns, npts, 3)
    if smooth and len(secs) > 2:
        flat = secs.transpose(1, 0, 2)  # per point: path across sections
        rows = [catmull_rom(flat[i], int(samples), closed) for i in range(npts)]
        rings = np.stack(rows, 1)
        if closed and np.allclose(rings[0], rings[-1]):
            rings = rings[:-1]
    else:
        ns = len(secs)
        tt = np.linspace(0, ns - 1, (ns - 1) * int(samples) + 1)
        i0 = np.minimum(np.floor(tt).astype(int), ns - 2)
        f = (tt - i0)[:, None, None]
        rings = secs[i0] * (1 - f) + secs[i0 + 1] * f
    return _grid_mesh(rings, not open_sections, closed, caps and not open_sections, "loft", smooth_angle)


# --------------------------------------------------------------------------------- mesh tools
def mesh_sdf_function(mesh):
    from .render import native as nat
    T = mesh.triangles()[0]
    V = mesh.verts

    def f(p):
        return nat.mesh_sdf(V, T, p).astype(np.float64)
    return f


def remesh(mesh, resolution=64, smooth_angle=None, offset=0.0, project=2, **_):
    """Voxel remesh: rebuild the mesh as an even, watertight surface (fills holes, fuses
    intersecting parts). ``offset`` grows (+) or shrinks (-) the surface."""
    if mesh.num_verts == 0:
        return mesh
    key = _jkey("remesh", _mesh_hash(mesh), resolution, smooth_angle, offset, project)
    return _cached(key, lambda: _remesh(mesh, resolution, smooth_angle, offset, project))


def _remesh(mesh, resolution, smooth_angle, offset, project):
    f0 = mesh_sdf_function(mesh)
    f = (lambda p: f0(p) - offset) if offset else f0
    lo, hi = mesh.verts.min(0) - abs(offset), mesh.verts.max(0) + abs(offset)
    m = mesh_function(f, lo, hi, int(resolution), mesh.smooth_angle if smooth_angle is None else smooth_angle,
                      project, name=mesh.name)
    return m


def boolean(mesh, op="difference", _other=None, resolution=96, blend=0.0, smooth_angle=None, project=3, **_):
    """Volumetric boolean with another mesh (``object`` node): union / difference / intersect.

    Both meshes are converted to exact signed-distance fields and recombined (optionally with a
    smooth ``blend`` radius), then re-polygonised - robust for any input, even open or
    self-intersecting meshes."""
    if _other is None or _other.num_verts == 0:
        return mesh
    key = _jkey("bool", _mesh_hash(mesh), _mesh_hash(_other), op, resolution, blend, smooth_angle, project)
    return _cached(key, lambda: _boolean(mesh, op, _other, resolution, blend, smooth_angle, project))


def _mesh_hash(m):
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    h.update(np.ascontiguousarray(m.verts).tobytes())
    h.update(np.ascontiguousarray(m.indices).tobytes())
    return h.hexdigest()


def _boolean(mesh, op, _other, resolution, blend, smooth_angle, project):
    fa = mesh_sdf_function(mesh)
    fb = mesh_sdf_function(_other)
    op = str(op).lower()
    if op in ("union", "add"):
        f = lambda p: smin(fa(p), fb(p), blend)  # noqa: E731
        lo = np.minimum(mesh.verts.min(0), _other.verts.min(0))
        hi = np.maximum(mesh.verts.max(0), _other.verts.max(0))
    elif op in ("intersect", "intersection"):
        f = lambda p: smax(fa(p), fb(p), blend)  # noqa: E731
        lo = np.maximum(mesh.verts.min(0), _other.verts.min(0))
        hi = np.minimum(mesh.verts.max(0), _other.verts.max(0))
    else:
        f = lambda p: smax(fa(p), -fb(p), blend)  # noqa: E731
        lo, hi = mesh.verts.min(0), mesh.verts.max(0)
    if np.any(hi <= lo):
        return Mesh(np.zeros((0, 3)), [], [], name=mesh.name)
    pad = blend + 1e-3
    return mesh_function(f, lo - pad, hi + pad, int(resolution),
                         mesh.smooth_angle if smooth_angle is None else smooth_angle, project, name=mesh.name)


def decimate(mesh, ratio=0.5, cell=None, **_):
    """Reduce polygon count by vertex clustering. ``ratio`` ~ fraction of vertices to keep
    (or give an explicit ``cell`` size)."""
    if mesh.num_verts < 8:
        return mesh
    V = mesh.verts
    lo, hi = V.min(0), V.max(0)
    if cell is None:
        area = float(np.sum(_triangle_areas(mesh)))
        target = max(8, int(mesh.num_verts * float(ratio)))
        cell = np.sqrt(max(area, 1e-12) / target) * 1.1
    key = np.floor((V - lo) / cell).astype(np.int64)
    dims = key.max(0) + 1
    kid = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inv = np.unique(kid, return_inverse=True)
    newV = np.zeros((len(uniq), 3))
    np.add.at(newV, inv, V)
    newV /= np.bincount(inv, minlength=len(uniq))[:, None]
    T = inv[mesh.triangles()[0]]
    ok = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T = T[ok]
    # drop duplicate triangles
    s = np.sort(T, 1)
    _, first = np.unique(s, axis=0, return_index=True)
    T = T[np.sort(first)]
    return Mesh(newV, np.full(len(T), 3), T.ravel(), smooth_angle=mesh.smooth_angle, name=mesh.name)


def _triangle_areas(mesh):
    T = mesh.triangles()[0]
    V = mesh.verts
    return 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)


def extrude(mesh, distance=0.1, faces=None, direction=None, inset=0.0, **_):
    """Extrude faces along their normals (or a fixed ``direction``), with side walls.

    ``faces``: list of face indices, or {"normal": [x,y,z], "angle": deg} to select faces facing a
    direction (default: all faces). ``inset`` shrinks the extruded caps toward their centres."""
    V = mesh.verts.copy()
    counts, idx, off = mesh.counts, mesh.indices, mesh.offsets
    nf = len(counts)
    fn = mesh.face_normals() if hasattr(mesh, "face_normals") else None
    if fn is None:
        fn = np.zeros((nf, 3))
        for f in range(nf):
            p = V[idx[off[f]:off[f + 1]]]
            fn[f] = mu.normalize(np.cross(p[1] - p[0], p[2] - p[0]))
    if faces is None:
        sel = np.ones(nf, bool)
    elif isinstance(faces, dict):
        n = mu.normalize(np.asarray(faces.get("normal", [0, 1, 0]), float))
        sel = fn @ n >= np.cos(np.radians(float(faces.get("angle", 30))))
    else:
        sel = np.zeros(nf, bool)
        sel[np.asarray(faces, int)] = True
    newV = list(V)
    new_counts, new_idx = [], []
    for f in range(nf):
        loop = list(idx[off[f]:off[f + 1]])
        if not sel[f]:
            new_counts.append(len(loop))
            new_idx += loop
            continue
        d = (np.asarray(direction, float) if direction is not None else fn[f]) * distance
        c = V[loop].mean(0)
        top = []
        for vi in loop:
            p = V[vi] + d
            if inset:
                p = p + (c + d - p) * float(inset)
            top.append(len(newV))
            newV.append(p)
        n = len(loop)
        for k in range(n):
            a, b = loop[k], loop[(k + 1) % n]
            new_counts.append(4)
            new_idx += [a, b, top[(k + 1) % n], top[k]]
        new_counts.append(n)
        new_idx += top
    return Mesh(np.asarray(newV), new_counts, new_idx, smooth_angle=mesh.smooth_angle, face_mat=None,
                name=mesh.name)


def mesh_stats(mesh):
    """Basic modelling diagnostics: counts, area, volume, bounds, open edges."""
    T = mesh.triangles()[0]
    V = mesh.verts
    vol = float(np.sum(np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]]))) / 6.0)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    _, c = np.unique(e, axis=0, return_counts=True)
    lo, hi = (V.min(0), V.max(0)) if len(V) else (np.zeros(3), np.zeros(3))
    return {"verts": int(mesh.num_verts), "faces": int(mesh.num_faces), "triangles": int(len(T)),
            "area": float(np.sum(_triangle_areas(mesh))), "volume": vol, "bounds": [lo.tolist(), hi.tolist()],
            "open_edges": int(np.sum(c == 1)), "non_manifold_edges": int(np.sum(c > 2))}
