"""Mesh modifiers / deformers (a non-destructive stack, evaluated per frame).

In a scene file a mesh node takes ``"modifiers": [ {...}, {...} ]``; they run top to bottom.
Every numeric parameter is animatable (keys or expressions). Deformer axis parameters take
``"x"``, ``"y"`` or ``"z"``.

=================  ==========================================================================
type               parameters (defaults)
=================  ==========================================================================
``subdivide``      ``levels`` (1). Catmull-Clark. Use ``"render_levels"`` to only subdivide
                   for final renders (``--preview`` uses ``levels``).
``transform``      ``translate``, ``rotate`` (deg), ``scale`` - bakes a transform into points
``bend``           ``angle`` (0 deg), ``axis`` ("y" - the length axis), ``direction`` ("x"),
                   ``center`` ([0,0,0]), ``limits`` ([lo, hi] along axis; default mesh bounds)
``twist``          ``angle`` (0 deg total), ``axis`` ("y"), ``center``, ``limits``
``taper``          ``factor`` (0 -> top scaled by 1+factor), ``axis`` ("y"), ``curve`` ("linear"|
                   "smooth"), ``limits``
``squash``         ``amount`` (1 = none, 0.5 = squashed to half height, 2 = stretched),
                   ``axis`` ("y"), ``pivot`` ([0,0,0]); volume preserving. Alias ``squash_stretch``
``wave``           ``amplitude`` (0.1), ``wavelength`` (1), ``speed`` (1 wavelengths/s),
                   ``direction`` ("x" or "radial"), ``axis`` ("y" displacement axis),
                   ``falloff`` (0 = none), ``phase`` (0)
``noise``          ``amplitude`` (0.1), ``scale`` (1 = feature size), ``octaves`` (3), ``seed``,
                   ``speed`` (0 = static; >0 evolves over time), ``mode`` ("normal"|"vector")
``inflate``        ``amount`` (0.05) - push along vertex normals
``spherify``       ``amount`` (1), ``radius`` (auto = mean distance), ``center``
``smooth``         ``iterations`` (5), ``factor`` (0.5) - Laplacian smoothing
``array``          ``count`` (3), ``offset`` ([1,0,0] per copy), ``rotate`` (deg per copy),
                   ``scale`` (per copy, multiplicative)
``mirror``         ``axis`` ("x"), ``merge`` (true), ``tolerance`` (1e-4)
``solidify``       ``thickness`` (0.05), ``offset`` (-1 inwards ... 1 outwards)
``ripple``         ``amplitude`` (0.1), ``wavelength`` (0.5), ``speed`` (1), ``center``
                   ([0,0,0]), ``decay`` (0.5) - radial ripples in Y from a point
``flip``           flips face winding / normals
``weld``           ``tolerance`` (1e-5)
``smooth_shading`` ``angle`` (180) - sets the auto-smooth angle
``displace``       ``amplitude``, ``texture`` (``"noise"``/``"voronoi"``/``"ridged"``),
                   ``scale``, ``seed`` - procedural displacement along normals
``lattice``        ``resolution`` ([2,2,2]), ``offsets``: {"i,j,k": [dx,dy,dz], ...},
                   ``bounds`` (auto) - free-form deformation lattice
=================  ==========================================================================
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry
from . import mathutil as mu
from . import noise as _noise

_AX = {"x": 0, "y": 1, "z": 2}


def _axis(v, default="y"):
    if v is None:
        v = default
    if isinstance(v, (int, float)):
        return int(v)
    return _AX[str(v).lower().lstrip("+-")]


def _limits(P, ax, limits):
    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    else:
        lo, hi = float(P[:, ax].min()) if len(P) else 0.0, float(P[:, ax].max()) if len(P) else 1.0
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return lo, hi


def bend(mesh, angle=0.0, axis="y", direction="x", center=(0, 0, 0), limits=None, **_):
    ang = math.radians(float(angle))
    if abs(ang) < 1e-7:
        return mesh
    a, d = _axis(axis), _axis(direction, "x")
    c = np.asarray(center, float)
    P = mesh.verts - c
    lo, hi = _limits(P, a, None if limits is None else np.asarray(limits) - c[a])
    length = hi - lo if limits is not None else max(abs(hi), abs(lo), 1e-9)
    s = P[:, a]
    sc = np.clip(s, lo, hi) if limits is not None else s
    theta = sc / length * ang
    R = length / ang
    x = P[:, d]
    out = P.copy()
    out[:, d] = R - (R - x) * np.cos(theta)
    out[:, a] = (R - x) * np.sin(theta)
    if limits is not None:  # continue straight beyond the limits
        over = s - sc
        out[:, d] += over * np.sin(theta)
        out[:, a] += over * np.cos(theta)
    return mesh.with_verts(out + c)


def twist(mesh, angle=0.0, axis="y", center=(0, 0, 0), limits=None, **_):
    ang = math.radians(float(angle))
    if abs(ang) < 1e-7:
        return mesh
    a = _axis(axis)
    c = np.asarray(center, float)
    P = mesh.verts - c
    lo, hi = _limits(P, a, None if limits is None else np.asarray(limits) - c[a])
    u = np.clip((P[:, a] - lo) / (hi - lo), 0, 1)
    th = u * ang
    i, j = [k for k in range(3) if k != a]
    if a == 1:
        i, j = 2, 0  # keep right-handed rotation about +Y
    ct, st = np.cos(th), np.sin(th)
    out = P.copy()
    out[:, i] = P[:, i] * ct - P[:, j] * st
    out[:, j] = P[:, i] * st + P[:, j] * ct
    return mesh.with_verts(out + c)


def taper(mesh, factor=0.0, axis="y", curve="linear", center=(0, 0, 0), limits=None, **_):
    if abs(factor) < 1e-9:
        return mesh
    a = _axis(axis)
    c = np.asarray(center, float)
    P = mesh.verts - c
    lo, hi = _limits(P, a, None if limits is None else np.asarray(limits) - c[a])
    u = np.clip((P[:, a] - lo) / (hi - lo), 0, 1)
    if curve == "smooth":
        u = u * u * (3 - 2 * u)
    s = 1 + float(factor) * u
    out = P.copy()
    for k in range(3):
        if k != a:
            out[:, k] *= s
    return mesh.with_verts(out + c)


def squash(mesh, amount=1.0, axis="y", pivot=(0, 0, 0), **_):
    amount = float(amount)
    if abs(amount - 1) < 1e-9:
        return mesh
    amount = max(amount, 1e-4)
    a = _axis(axis)
    pv = np.asarray(pivot, float)
    P = mesh.verts - pv
    side = 1 / math.sqrt(amount)
    sc = np.full(3, side)
    sc[a] = amount
    return mesh.with_verts(P * sc + pv)


def wave(mesh, amplitude=0.1, wavelength=1.0, speed=1.0, direction="x", axis="y", falloff=0.0, phase=0.0,
         center=(0, 0, 0), _time=0.0, **_):
    P = mesh.verts
    a = _axis(axis)
    c = np.asarray(center, float)
    if direction == "radial":
        q = P - c
        q[:, a] = 0
        dist = np.linalg.norm(q, axis=1)
    else:
        dist = P[:, _axis(direction, "x")] - c[_axis(direction, "x")]
    disp = float(amplitude) * np.sin(2 * math.pi * (dist / max(float(wavelength), 1e-9) - float(speed) * _time) + phase)
    if falloff:
        disp *= np.exp(-float(falloff) * np.abs(dist))
    out = P.copy()
    out[:, a] += disp
    return mesh.with_verts(out)


def ripple(mesh, amplitude=0.1, wavelength=0.5, speed=1.0, center=(0, 0, 0), decay=0.5, _time=0.0, **_):
    c = np.asarray(center, float)
    q = mesh.verts - c
    r = np.hypot(q[:, 0], q[:, 2])
    front = float(speed) * _time * wavelength * 2 + 1e-9
    env = np.exp(-float(decay) * r) * np.clip(1 - (r - front) / max(wavelength, 1e-9), 0, 1)
    out = mesh.verts.copy()
    out[:, 1] += float(amplitude) * env * np.sin(2 * math.pi * (r / max(wavelength, 1e-9) - speed * _time))
    return mesh.with_verts(out)


def noise_deform(mesh, amplitude=0.1, scale=1.0, octaves=3, seed=0, speed=0.0, mode="normal", _time=0.0, **_):
    P = mesh.verts
    f = 1.0 / max(float(scale), 1e-9)
    z = _time * float(speed)
    if mode == "vector":
        d = _noise.noise_vec3(P * f + z, int(seed), int(octaves)) * float(amplitude)
        return mesh.with_verts(P + d)
    n = mesh.vertex_normals()
    h = _noise.fbm(P[:, 0] * f + z, P[:, 1] * f, P[:, 2] * f - z, int(octaves), seed=int(seed))
    return mesh.with_verts(P + n * (h * float(amplitude))[:, None])


def displace(mesh, amplitude=0.1, texture="noise", scale=1.0, seed=0, octaves=4, **kw):
    P = mesh.verts
    f = 1.0 / max(float(scale), 1e-9)
    n = mesh.vertex_normals()
    if texture == "ridged":
        h = 1 - np.abs(_noise.fbm(P[:, 0] * f, P[:, 1] * f, P[:, 2] * f, int(octaves), seed=int(seed)))
        h = h * h - 0.5
    elif texture == "voronoi":
        h = _voronoi_f1(P * f, int(seed)) - 0.5
    else:
        h = _noise.fbm(P[:, 0] * f, P[:, 1] * f, P[:, 2] * f, int(octaves), seed=int(seed))
    return mesh.with_verts(P + n * (h * float(amplitude))[:, None])


def _voronoi_f1(p, seed):
    cell = np.floor(p)
    best = np.full(len(p), 9.0)
    rng_base = np.array([127.1, 311.7, 74.7])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = cell + np.array([dx, dy, dz])
                h = np.sin(c @ rng_base + seed * 13.13) * 43758.5453
                jit = np.stack([h - np.floor(h), (h * 1.37) - np.floor(h * 1.37), (h * 1.71) - np.floor(h * 1.71)], 1)
                d = np.linalg.norm(c + jit - p, axis=1)
                best = np.minimum(best, d)
    return np.clip(best, 0, 1)


def inflate(mesh, amount=0.05, **_):
    if abs(amount) < 1e-12:
        return mesh
    return mesh.with_verts(mesh.verts + mesh.vertex_normals() * float(amount))


def spherify(mesh, amount=1.0, radius=None, center=None, **_):
    c = mesh.verts.mean(axis=0) if center is None else np.asarray(center, float)
    q = mesh.verts - c
    d = np.linalg.norm(q, axis=1, keepdims=True)
    r = float(np.mean(d)) if radius is None else float(radius)
    target = q / np.maximum(d, 1e-12) * r
    return mesh.with_verts(c + q + (target - q) * float(amount))


def smooth(mesh, iterations=5, factor=0.5, **_):
    E, _ = mesh.edges()
    P = mesh.verts.copy()
    nv = len(P)
    deg = np.bincount(E.ravel(), minlength=nv).astype(float)
    for _ in range(int(iterations)):
        acc = np.zeros_like(P)
        for k in range(3):
            acc[:, k] = np.bincount(E[:, 0], weights=P[E[:, 1], k], minlength=nv) + \
                np.bincount(E[:, 1], weights=P[E[:, 0], k], minlength=nv)
        avg = acc / np.maximum(deg, 1)[:, None]
        P = P + (avg - P) * float(factor) * (deg > 0)[:, None]
    return mesh.with_verts(P)


def array(mesh, count=3, offset=(1, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1), **_):
    count = max(1, int(round(float(count))))
    off = np.asarray(offset, float) * np.ones(3)
    rot = np.asarray(rotate, float) * np.ones(3)
    sc = np.asarray(scale, float) * np.ones(3)
    step = mu.compose(off, rot, sc)
    M = np.eye(4)
    parts = []
    for _ in range(count):
        parts.append(mesh.transformed(M) if not np.allclose(M, np.eye(4)) else mesh.copy())
        M = M @ step
    out = geometry.merge(parts, mesh.name)
    out.smooth_angle = mesh.smooth_angle
    return out


def mirror(mesh, axis="x", merge=True, tolerance=1e-4, **_):
    a = _axis(axis, "x")
    S = np.eye(4)
    S[a, a] = -1
    m2 = mesh.transformed(S).flipped()
    out = geometry.merge([mesh, m2], mesh.name)
    out.smooth_angle = mesh.smooth_angle
    if merge:
        # only weld verts lying on the mirror plane
        P = out.verts
        on = np.abs(P[:, a]) < tolerance
        P[on, a] = 0.0
        key = np.round(P / max(tolerance, 1e-9)).astype(np.int64)
        key[~on] = np.stack([np.arange((~on).sum()) + 10 ** 12] * 3, 1)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        out = geometry.Mesh(P[first], out.counts, inv.ravel()[out.indices], out.uvs, None, out.smooth_angle,
                            out.face_mat, out.name)
    return out


def solidify(mesh, thickness=0.05, offset=-1.0, **_):
    n = mesh.vertex_normals()
    t = float(thickness)
    outer = mesh.verts + n * t * (offset + 1) / 2
    inner = mesh.verts + n * t * (offset - 1) / 2
    a = mesh.with_verts(outer).copy()
    b = mesh.with_verts(inner).flipped()
    nv = len(mesh.verts)
    merged = geometry.merge([a, b], mesh.name)
    # side walls along boundary edges
    E, ce = mesh.edges()
    cnt = np.bincount(ce, minlength=len(E))
    nxt = mesh.corner_next()
    bc = np.flatnonzero(cnt[ce] == 1)
    if len(bc):
        v0 = mesh.indices[bc]
        v1 = mesh.indices[nxt[bc]]
        quads = np.stack([v1, v0, v0 + nv, v1 + nv], 1)
        counts = np.concatenate([merged.counts, np.full(len(bc), 4)])
        idx = np.concatenate([merged.indices, quads.ravel()])
        uvs = None
        if merged.uvs is not None:
            uvs = np.concatenate([merged.uvs, np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (len(bc), 1))])
        fm = np.concatenate([merged.face_mat, mesh.face_mat[mesh.corner_face()[bc]]])
        merged = geometry.Mesh(merged.verts, counts, idx, uvs, None, mesh.smooth_angle, fm, mesh.name)
    merged.smooth_angle = min(mesh.smooth_angle, 60.0)
    return merged


def transform(mesh, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1), rotate_order="xyz", **_):
    sc = np.asarray(scale, float) * np.ones(3)
    return mesh.transformed(mu.compose(np.asarray(translate, float), np.asarray(rotate, float), sc, rotate_order))


def flip(mesh, **_):
    return mesh.flipped()


def weld(mesh, tolerance=1e-5, **_):
    return mesh.welded(float(tolerance))


def smooth_shading(mesh, angle=180.0, **_):
    m = mesh.copy()
    m.smooth_angle = float(angle)
    return m


def subdivide(mesh, levels=1, render_levels=None, _preview=False, **_):
    lv = levels if (_preview or render_levels is None) else render_levels
    return mesh.subdivided(int(lv))


def lattice(mesh, resolution=(2, 2, 2), offsets=None, bounds=None, **_):
    """Trilinear-Bernstein free-form deformation."""
    res = [max(2, int(r)) for r in (np.asarray(resolution) * np.ones(3))]
    P = mesh.verts
    if bounds is None:
        lo, hi = P.min(0), P.max(0)
    else:
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    span = np.maximum(hi - lo, 1e-9)
    stu = (P - lo) / span
    disp = np.zeros(tuple(res) + (3,))
    for key, d in (offsets or {}).items():
        i, j, k = (int(x) for x in str(key).replace(" ", "").split(","))
        disp[i, j, k] = np.asarray(d, float)
    from math import comb
    out = P.copy()
    l, m, n = (r - 1 for r in res)
    Bs = [np.stack([comb(l, i) * stu[:, 0] ** i * (1 - stu[:, 0]) ** (l - i) for i in range(l + 1)], 1),
          np.stack([comb(m, j) * stu[:, 1] ** j * (1 - stu[:, 1]) ** (m - j) for j in range(m + 1)], 1),
          np.stack([comb(n, k) * stu[:, 2] ** k * (1 - stu[:, 2]) ** (n - k) for k in range(n + 1)], 1)]
    out += np.einsum("pi,pj,pk,ijkc->pc", Bs[0], Bs[1], Bs[2], disp)
    return mesh.with_verts(out)


MODIFIERS = {
    "subdivide": subdivide, "subdiv": subdivide, "smooth_mesh": subdivide,
    "bend": bend, "twist": twist, "taper": taper, "squash": squash, "squash_stretch": squash,
    "wave": wave, "ripple": ripple, "noise": noise_deform, "displace": displace, "inflate": inflate,
    "spherify": spherify, "smooth": smooth, "array": array, "mirror": mirror, "solidify": solidify,
    "transform": transform, "flip": flip, "weld": weld, "smooth_shading": smooth_shading, "lattice": lattice,
}

#: Modifiers that change topology (cannot be matched vertex-for-vertex across time unless
#: deterministic - all of ours are deterministic, so motion blur still works).
TOPOLOGY_CHANGING = {"subdivide", "subdiv", "smooth_mesh", "array", "mirror", "solidify", "weld", "remesh",
                     "boolean", "decimate", "extrude"}


def _register_modeling():
    from . import modeling as _md
    MODIFIERS.update({"remesh": _md.remesh, "boolean": _md.boolean, "decimate": _md.decimate,
                      "extrude": _md.extrude})


_register_modeling()


def apply_stack(mesh, stack, frame=0.0, ctx=None, fps=24.0, preview=False, stage=None):
    """Apply a modifier stack. ``stack`` is a list of dicts whose values may be :class:`Param`.

    ``stage``: ``None`` = all, ``"pre"`` = everything before the first ``subdivide`` that is
    flagged ``"after_deform": true`` (used so subdivision runs after skinning), ``"post"`` = rest.
    """
    from .params import Param
    if not stack:
        return mesh
    for mod in stack:
        if not mod.get("enabled", True):
            continue
        kind = mod["type"]
        post = bool(mod.get("after_deform", kind in ("subdivide", "subdiv", "smooth_mesh")))
        if stage == "pre" and post:
            continue
        if stage == "post" and not post:
            continue
        fn = MODIFIERS.get(kind)
        if fn is None:
            raise ValueError(f"unknown modifier {kind!r}; available: {sorted(MODIFIERS)}")
        kw = {}
        for k, v in mod.items():
            if k in ("type", "enabled", "after_deform", "name"):
                continue
            kw[k] = v.evaluate(frame, ctx) if isinstance(v, Param) else v
        kw["_time"] = frame / fps
        if kind in ("subdivide", "subdiv", "smooth_mesh"):
            kw["_preview"] = preview
        mesh = fn(mesh, **kw)
    return mesh
