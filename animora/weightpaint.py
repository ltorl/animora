"""Skin weight painting, editing, import/export and heat-map preview.

Maya's Paint Skin Weights tool works interactively; in Animora every brush stroke is a line of
JSON listed under the skin's ``"weight_paint"`` key. That makes painting reproducible and lets you
diff and version it. Strokes run in order, after the automatic or explicit weights are computed.

    "skin": {
      "root": "hips",
      "weight_paint": [
        {"joint": "spine2", "op": "add", "value": 0.4, "at": "spine2", "radius": 0.18, "mirror": false},
        {"joint": "l_elbow", "op": "replace", "value": 1.0, "points": [[0.55,1.4,0],[0.62,1.4,0]],
         "radius": 0.05, "falloff": "smooth", "mirror": true},
        {"op": "smooth", "bone": "l_elbow", "radius": 0.08, "iterations": 3},
        {"op": "prune", "threshold": 0.02},
        {"op": "max_influences", "count": 4}
      ],
      "locked_joints": ["hips"]
    }

Brush operations take a ``joint``: ``add``, ``subtract``, ``replace`` and ``scale``.
``smooth`` blends all influences with their mesh neighbours, and ``hammer`` is a full-strength smooth.

Where the brush acts:
* ``center`` - a world-space point in the bind pose.
* ``at`` - the position of a joint.
* ``bone`` - the joint-to-child segment.
* ``points`` - a polyline stroke.
* ``vertices`` - explicit vertex indices.
* ``box`` - an axis-aligned box ``[[min], [max]]``.
* None of these - a flood fill over the whole mesh.

Brush settings: ``radius`` and ``falloff``, which is one of ``smooth``, ``linear``, ``constant``,
``gaussian`` or ``needle``. ``strength`` is the opacity. ``surface: true`` measures distance along
the mesh, so a brush on one leg never bleeds onto the other. ``mirror: true`` (or ``"x"``/``"y"``/``"z"``)
repeats the stroke on the opposite side with the mirrored joint name, e.g. ``l_elbow`` → ``r_elbow``.

Global operations, which need no joint:
* ``prune`` - drop weights below a threshold.
* ``normalize``
* ``max_influences``
* ``mirror`` - copy one side's weights to the other with joint names swapped. Same as Maya's Mirror Skin Weights.
* ``move`` - move a joint's weight onto another joint.
* ``copy`` - copy a joint's weight column onto another joint.

Normalization works like Maya's interactive mode. When a brush raises or lowers a joint, the
other unlocked influences of that vertex take up the difference in proportion. Locked joints
(``lock`` on the stroke, or ``locked_joints`` on the skin) never change.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

FALLOFFS = ("smooth", "linear", "constant", "gaussian", "needle")
BRUSH_OPS = ("add", "subtract", "replace", "scale", "smooth", "hammer")
GLOBAL_OPS = ("prune", "normalize", "max_influences", "mirror", "move", "copy")
DEFAULT_VALUE = {"add": 0.1, "subtract": 0.1, "replace": 1.0, "scale": 0.9}


# ----------------------------------------------------------------------------- names
_PAIRS = [
    (r"(?i)left", r"(?i)right"),
]


def mirror_name(name, candidates=None):
    """Return the opposite-side joint name (``Left``↔``Right``, ``l_``↔``r_``, ``_l``↔``_r``,
    ``.L``↔``.R``, ``LHip``↔``RHip`` ...). When ``candidates`` is given only a name in it is
    returned (else ``name`` itself, i.e. a centre joint)."""
    def swap_case(m, a, b):
        s = m.group(0)
        rep = b if s.lower() == a else a
        if s.isupper():
            return rep.upper()
        if s[0].isupper():
            return rep.capitalize()
        return rep

    rules = [
        lambda n: re.sub(r"(?i)left|right", lambda m: swap_case(m, "left", "right"), n, count=1),
        lambda n: re.sub(r"^([lLrR])([_.\- ])", lambda m: {"l": "r", "r": "l", "L": "R", "R": "L"}[m.group(1)] + m.group(2), n),
        lambda n: re.sub(r"([_.\- ])([lLrR])$", lambda m: m.group(1) + {"l": "r", "r": "l", "L": "R", "R": "L"}[m.group(2)], n),
        lambda n: re.sub(r"([_.\- ])([lLrR])([_.\- ])", lambda m: m.group(1) + {"l": "r", "r": "l", "L": "R", "R": "L"}[m.group(2)] + m.group(3), n, count=1),
        lambda n: re.sub(r"^([LR])(?=[A-Z])", lambda m: "R" if m.group(1) == "L" else "L", n),
        lambda n: re.sub(r"^([lr])(?=[a-z]{3,})", lambda m: "r" if m.group(1) == "l" else "l", n),
    ]
    for r in rules:
        out = r(name)
        if out != name and (candidates is None or out in candidates):
            return out
    return name


# ----------------------------------------------------------------------------- geometry helpers
def nearest_indices(src, dst, chunk=2048):
    """Index of the nearest ``src`` point for every ``dst`` point (chunked brute force)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    s2 = np.einsum("ij,ij->i", src, src)
    out = np.empty(len(dst), np.int64)
    for i in range(0, len(dst), chunk):
        d = dst[i:i + chunk]
        D = s2[None, :] - 2.0 * d @ src.T
        out[i:i + chunk] = np.argmin(D, axis=1)
    return out


def _falloff(x, kind):
    x = np.clip(x, 0.0, None)
    if kind == "constant":
        f = (x <= 1.0).astype(float)
    elif kind == "linear":
        f = 1.0 - x
    elif kind == "gaussian":
        f = np.exp(-4.0 * x * x) * (x <= 1.0)
    elif kind == "needle":
        f = (1.0 - x) ** 4
    else:  # smooth
        f = 1.0 - x * x * (3.0 - 2.0 * x)
    return np.clip(np.where(x <= 1.0, f, 0.0), 0.0, 1.0)


def _seg_dist(P, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _surface_distance(P, E, seed_d, radius):
    """Approximate geodesic distance: seed distances relaxed along mesh edges (bounded)."""
    d = seed_d.copy()
    elen = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    for _ in range(4096):
        nd = d.copy()
        np.minimum.at(nd, E[:, 0], d[E[:, 1]] + elen)
        np.minimum.at(nd, E[:, 1], d[E[:, 0]] + elen)
        nd = np.minimum(nd, d)
        changed = np.any(nd < d - 1e-12)
        d = nd
        if not changed:
            break
    d[d > radius * 4] = np.inf
    return d


class PaintContext:
    """Everything a stroke needs: bind-pose vertex positions, mesh edges, joint positions."""

    def __init__(self, P, joints, edges=None, joint_pos=None, joint_children=None, joint_parent=None):
        self.P = np.asarray(P, float)
        self.joints = list(joints)
        self.edges = edges
        self.joint_pos = joint_pos or {}
        self.joint_children = joint_children or {}
        self.joint_parent = joint_parent or {}

    @classmethod
    def from_scene(cls, scene, node, mesh, P, joints, bind_frame):
        from .rig import bind_world, _vertex_neighbors
        st = scene.state(bind_frame)
        pos, kids, par = {}, {}, {}
        for n, nd in scene.nodes.items():
            if nd.type == "joint":
                pos[n] = bind_world(scene, st, n)[:3, 3]
        for n in pos:
            p = scene.nodes[n].parent
            par[n] = p
            if p in pos:
                kids.setdefault(p, []).append(n)
        try:
            E = _vertex_neighbors(mesh)
        except Exception:
            E = None
        return cls(P, joints, E, pos, kids, par)

    def jidx(self, name):
        if name not in self.joints:
            raise ValueError(f"weight_paint: {name!r} is not one of the skin joints {self.joints}")
        return self.joints.index(name)


# ----------------------------------------------------------------------------- brush mask
def _reflect(v, axis, origin=0.0):
    v = np.array(v, float)
    v[..., axis] = 2 * origin - v[..., axis]
    return v


def _stroke_mirror(stroke, axis, origin, joints):
    s = dict(stroke)
    s.pop("mirror", None)
    for k in ("center",):
        if k in s:
            s[k] = _reflect(s[k], axis, origin).tolist()
    if "points" in s:
        s["points"] = _reflect(s["points"], axis, origin).tolist()
    if "box" in s:
        b = _reflect(s["box"], axis, origin)
        s["box"] = [np.minimum(b[0], b[1]).tolist(), np.maximum(b[0], b[1]).tolist()]
    for k in ("at", "bone", "joint", "to", "from"):
        if isinstance(s.get(k), str):
            s[k] = mirror_name(s[k], None)
    if "lock" in s:
        s["lock"] = [mirror_name(j) for j in s["lock"]]
    if "vertices" in s:
        s["_mirror_vertices"] = (axis, origin)
    return s


def brush_mask(stroke, ctx):
    """Per-vertex brush influence in [0, 1] for one stroke."""
    P = ctx.P
    radius = float(stroke.get("radius", 0.1))
    kind = stroke.get("falloff", "smooth")
    if kind not in FALLOFFS:
        raise ValueError(f"weight_paint: falloff must be one of {FALLOFFS}")
    d = None
    if "vertices" in stroke:
        f = np.zeros(len(P))
        idx = np.asarray(stroke["vertices"], int)
        if "_mirror_vertices" in stroke:
            ax, o = stroke["_mirror_vertices"]
            idx = nearest_indices(P, _reflect(P[idx], ax, o))
        f[idx] = 1.0
        return f
    if "box" in stroke:
        lo, hi = (np.asarray(v, float) for v in stroke["box"])
        return np.all((P >= lo) & (P <= hi), axis=1).astype(float)
    if "center" in stroke:
        d = np.linalg.norm(P - np.asarray(stroke["center"], float), axis=1)
    elif "at" in stroke:
        if stroke["at"] not in ctx.joint_pos:
            raise ValueError(f"weight_paint: joint {stroke['at']!r} not found")
        d = np.linalg.norm(P - ctx.joint_pos[stroke["at"]], axis=1)
    elif "bone" in stroke:
        j = stroke["bone"]
        if j not in ctx.joint_pos:
            raise ValueError(f"weight_paint: joint {j!r} not found")
        a = ctx.joint_pos[j]
        kids = ctx.joint_children.get(j, [])
        if kids:
            d = np.min([_seg_dist(P, a, ctx.joint_pos[c]) for c in kids], axis=0)
        else:
            p = ctx.joint_parent.get(j)
            b = a + (a - ctx.joint_pos[p]) * 0.5 if p in ctx.joint_pos else a
            d = _seg_dist(P, a, b)
    elif "points" in stroke:
        pts = np.asarray(stroke["points"], float).reshape(-1, 3)
        if len(pts) == 1:
            d = np.linalg.norm(P - pts[0], axis=1)
        else:
            d = np.min([_seg_dist(P, pts[i], pts[i + 1]) for i in range(len(pts) - 1)], axis=0)
    if d is None:
        return np.ones(len(P))           # flood
    if stroke.get("surface") and ctx.edges is not None and len(ctx.edges):
        # seed with the vertices the brush actually touches (closest vertex at least)
        seed = np.full(len(P), np.inf)
        near = d <= max(radius * 0.25, float(d.min()) + 1e-9)
        seed[near] = d[near]
        d = _surface_distance(P, ctx.edges, seed, radius)
    return _falloff(d / max(radius, 1e-9), kind)


# ----------------------------------------------------------------------------- weight edits
def _set_column(W, k, new, rows, locked, parent_col=None):
    """Set W[rows, k] = new and renormalize the other unlocked influences proportionally."""
    if not len(rows):
        return
    nj = W.shape[1]
    lock = np.zeros(nj, bool)
    lock[list(locked)] = True
    lock[k] = False
    others = ~lock
    others[k] = False
    Wr = W[rows]
    lsum = Wr[:, lock].sum(axis=1)
    cap = np.clip(1.0 - lsum, 0.0, 1.0)
    new = np.clip(new, 0.0, cap)
    osum = Wr[:, others].sum(axis=1)
    rem = cap - new
    has = osum > 1e-12
    scale = np.where(has, rem / np.maximum(osum, 1e-12), 0.0)
    Wr[:, others] *= scale[:, None]
    # nobody to give the remainder to: the parent joint (or, failing that, the joint keeps it)
    need = (~has) & (rem > 1e-9)
    if np.any(need):
        if parent_col is not None and parent_col != k and not lock[parent_col]:
            Wr[need, parent_col] = rem[need]
        else:
            new = np.where(need, cap, new)
    Wr[:, k] = new
    W[rows] = Wr


def _smooth(W, E, f, iterations, locked):
    if E is None or not len(E):
        return W
    nv, nj = W.shape
    lock = np.zeros(nj, bool)
    lock[list(locked)] = True
    rows = f > 0
    for _ in range(max(1, int(iterations))):
        acc = np.zeros_like(W)
        cnt = np.zeros(nv)
        np.add.at(acc, E[:, 0], W[E[:, 1]])
        np.add.at(acc, E[:, 1], W[E[:, 0]])
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        avg = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], W)
        Wn = W + (avg - W) * f[:, None]
        if lock.any():
            Wn[:, lock] = W[:, lock]
            free = ~lock
            target = 1.0 - W[:, lock].sum(axis=1)
            s = Wn[:, free].sum(axis=1)
            Wn[:, free] *= np.where(s > 1e-12, target / np.maximum(s, 1e-12), 0.0)[:, None]
        else:
            Wn /= np.maximum(Wn.sum(axis=1, keepdims=True), 1e-12)
        W = np.where(rows[:, None], Wn, W)
    return W


def normalize(W):
    s = W.sum(axis=1, keepdims=True)
    W = np.where(s > 1e-12, W / np.maximum(s, 1e-12), W)
    zero = s[:, 0] <= 1e-12
    if np.any(zero):
        W[zero, 0] = 1.0
    return W


def prune(W, threshold=0.01, locked=()):
    W = W.copy()
    keep = np.zeros(W.shape[1], bool)
    keep[list(locked)] = True
    small = (W < threshold) & ~keep[None, :]
    # never prune every influence of a vertex
    allgone = small.all(axis=1)
    if np.any(allgone):
        small[allgone, np.argmax(W[allgone], axis=1)] = False
    W[small] = 0.0
    return normalize(W)


def limit_influences(W, count=4):
    count = int(count)
    if count <= 0 or count >= W.shape[1]:
        return W
    W = W.copy()
    idx = np.argsort(-W, axis=1)[:, count:]
    np.put_along_axis(W, idx, 0.0, axis=1)
    return normalize(W)


def mirror_weights(W, ctx, axis=0, direction="+", origin=0.0, tolerance=None):
    """Copy weights from one side of the mesh to the other, swapping left/right joint names."""
    P = ctx.P
    src_side = P[:, axis] - origin >= 0 if direction.startswith("+") else P[:, axis] - origin <= 0
    dst = np.where(~src_side)[0]
    if not len(dst):
        return W
    srci = np.where(src_side)[0]
    nn = srci[nearest_indices(P[srci], _reflect(P[dst], axis, origin))]
    if tolerance is not None:
        ok = np.linalg.norm(_reflect(P[dst], axis, origin) - P[nn], axis=1) <= float(tolerance)
        dst, nn = dst[ok], nn[ok]
    perm = [ctx.joints.index(mirror_name(j, ctx.joints)) for j in ctx.joints]
    W = W.copy()
    Wd = np.zeros((len(dst), W.shape[1]))
    for a, b in enumerate(perm):
        Wd[:, b] += W[nn, a]
    W[dst] = Wd
    return W


def apply_ops(W, ops, ctx, locked=()):
    """Apply a ``weight_paint`` list (strokes and global operations) to weights ``W`` (V, J)."""
    W = np.array(W, float, copy=True)
    base_locked = {ctx.jidx(j) for j in locked}
    ops = [ops] if isinstance(ops, dict) else list(ops or [])
    expanded = []
    for s in ops:
        s = dict(s)
        expanded.append(s)
        m = s.get("mirror")
        if m and s.get("op", "add") in BRUSH_OPS:
            axis = "xyz".index(m) if isinstance(m, str) and m in "xyz" else 0
            expanded.append(_stroke_mirror(s, axis, float(s.get("mirror_origin", 0.0)), ctx.joints))
    for s in expanded:
        op = s.get("op", "add" if "joint" in s else None)
        lk = base_locked | {ctx.jidx(j) for j in s.get("lock", [])}
        if op in ("smooth", "hammer"):
            f = brush_mask(s, ctx) * (1.0 if op == "hammer" else float(s.get("strength", 0.5)))
            W = _smooth(W, ctx.edges, f, s.get("iterations", 1), lk)
        elif op in ("add", "subtract", "replace", "scale"):
            if "joint" not in s:
                raise ValueError(f"weight_paint: '{op}' stroke needs a 'joint'")
            k = ctx.jidx(s["joint"])
            if k in lk:
                continue
            f = brush_mask(s, ctx) * float(s.get("strength", 1.0))
            rows = np.where(f > 1e-9)[0]
            if not len(rows):
                continue
            v = float(s.get("value", DEFAULT_VALUE[op]))
            old = W[rows, k]
            fr = f[rows]
            if op == "add":
                new = old + v * fr
            elif op == "subtract":
                new = old - v * fr
            elif op == "replace":
                new = old + (v - old) * fr
            else:
                new = old * (1.0 - (1.0 - v) * fr)
            p = ctx.joint_parent.get(s["joint"])
            _set_column(W, k, new, rows, lk, ctx.joints.index(p) if p in ctx.joints else None)
        elif op == "prune":
            W = prune(W, float(s.get("threshold", 0.01)), {ctx.jidx(j) for j in s.get("lock", [])})
        elif op == "normalize":
            W = normalize(W)
        elif op == "max_influences":
            W = limit_influences(W, s.get("count", s.get("value", 4)))
        elif op == "mirror":
            ax = s.get("axis", "x")
            W = mirror_weights(W, ctx, "xyz".index(ax) if isinstance(ax, str) else int(ax),
                               str(s.get("direction", "+")), float(s.get("origin", 0.0)), s.get("tolerance"))
        elif op in ("move", "copy"):
            a, b = ctx.jidx(s["from"]), ctx.jidx(s["to"])
            f = brush_mask(s, ctx) * float(s.get("strength", 1.0))
            amount = W[:, a] * f
            if op == "move":
                W[:, b] += amount
                W[:, a] -= amount
            else:
                rows = np.where(f > 1e-9)[0]
                _set_column(W, b, W[rows, b] + (W[rows, a] - W[rows, b]) * f[rows], rows, lk | {a})
        else:
            raise ValueError(f"weight_paint: unknown op {op!r} (brush: {BRUSH_OPS}, global: {GLOBAL_OPS})")
    return W


# ----------------------------------------------------------------------------- import / export
def export_weights(path, W, joints, P=None, node=None, threshold=1e-4):
    """Write weights as sparse JSON ``{joint: [[vertex, weight], ...]}`` (+ bind positions for
    position-based import onto a different mesh)."""
    W = np.asarray(W, float)
    data = {"format": "animora-skin-weights", "version": 1, "node": node, "num_verts": int(len(W)),
            "joints": list(joints), "weights": {}}
    for k, j in enumerate(joints):
        idx = np.where(W[:, k] > threshold)[0]
        if len(idx):
            data["weights"][j] = [[int(i), round(float(W[i, k]), 6)] for i in idx]
    if P is not None:
        data["positions"] = np.round(np.asarray(P, float), 6).tolist()
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return data


def load_weights(path, joints, P, base_dir=".", method="auto"):
    """Read a weights JSON (see :func:`export_weights`) for a mesh with bind positions ``P``.

    ``method`` ``"index"`` matches vertex numbers; ``"position"`` transfers by closest bind-pose
    position (for a re-topologised or differently-subdivided mesh); ``"auto"`` uses index when
    the vertex count matches and position otherwise."""
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    with open(path) as fh:
        data = json.load(fh)
    wspec = data.get("weights", data)
    fj = data.get("joints") or list(wspec)
    n_file = int(data.get("num_verts", len(P)))
    Wf = np.zeros((n_file, len(fj)))
    for k, j in enumerate(fj):
        vals = np.asarray(wspec.get(j, []), float)
        if vals.ndim == 2 and len(vals):
            Wf[vals[:, 0].astype(int), k] = vals[:, 1]
        elif vals.ndim == 1 and len(vals) == n_file:
            Wf[:, k] = vals
    use_pos = method == "position" or (method == "auto" and n_file != len(P))
    if use_pos:
        if "positions" not in data:
            raise ValueError(f"{path}: vertex count {n_file} != mesh {len(P)} and the file has no positions")
        nn = nearest_indices(np.asarray(data["positions"], float), P)
        Wf = Wf[nn]
    W = np.zeros((len(P), len(joints)))
    missing = []
    for k, j in enumerate(fj):
        if j in joints:
            W[:, joints.index(j)] += Wf[:, k]
        elif Wf[:, k].any():
            missing.append(j)
    if missing:
        raise ValueError(f"{path}: joints {missing} are not skin joints of this mesh")
    return normalize(W)


def copy_weights(W_src, P_src, joints_src, P_dst, joints_dst):
    """Closest-point skin weight transfer between meshes (Maya *Copy Skin Weights*)."""
    nn = nearest_indices(P_src, P_dst)
    W = np.zeros((len(P_dst), len(joints_dst)))
    for k, j in enumerate(joints_src):
        if j in joints_dst:
            W[:, joints_dst.index(j)] += W_src[nn, k]
    return normalize(W)


# ----------------------------------------------------------------------------- heat map preview
def heat_color(w):
    """Maya-like weight ramp: black (0) → blue → cyan → green → yellow → red → white-hot (1)."""
    w = np.clip(np.asarray(w, float), 0, 1)
    stops = np.array([0.0, 0.08, 0.3, 0.5, 0.7, 0.9, 1.0])
    cols = np.array([[0.05, 0.05, 0.07], [0.1, 0.15, 0.8], [0.0, 0.75, 0.9], [0.1, 0.85, 0.2],
                     [1.0, 0.9, 0.1], [1.0, 0.25, 0.05], [1.0, 0.95, 0.9]])
    return np.stack([np.interp(w, stops, cols[:, c]) for c in range(3)], -1)


def influence_colors(n):
    """Distinct colours per joint (golden-angle hues) for the 'all influences' view."""
    import colorsys
    return np.array([colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.75, 0.95) for i in range(n)])


def rasterize(verts, tris, vcol, cam_matrix, fov_y, width, height, bg=(0.16, 0.17, 0.19), light=None):
    """Tiny z-buffered Gouraud rasterizer (numpy) for previews: returns float RGB (H, W, 3)."""
    Vinv = np.linalg.inv(cam_matrix)
    Pc = verts @ Vinv[:3, :3].T + Vinv[:3, 3]
    z = -Pc[:, 2]
    f = 0.5 * height / math.tan(fov_y / 2)
    zs = np.maximum(z, 1e-6)
    sx = width * 0.5 + Pc[:, 0] / zs * f
    sy = height * 0.5 - Pc[:, 1] / zs * f
    # per-vertex normals for shading (camera headlight + soft key)
    n = np.zeros_like(verts)
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    for c in range(3):
        np.add.at(n, tris[:, c], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    view = cam_matrix[:3, 3] - verts
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    key = np.asarray(light if light is not None else cam_matrix[:3, :3] @ np.array([0.4, 0.6, 0.7]), float)
    key /= np.linalg.norm(key)
    shade = 0.28 + 0.52 * np.abs(np.einsum("ij,ij->i", n, view)) + 0.25 * np.clip(n @ key, 0, 1)
    col = vcol * shade[:, None]
    img = np.empty((height, width, 3))
    yy = np.linspace(0, 1, height)[:, None, None]
    img[:] = np.asarray(bg) * (1.1 - 0.25 * yy)
    zbuf = np.full((height, width), np.inf)
    ok = np.all(z[tris] > 1e-4, axis=1)
    for t in np.where(ok)[0]:
        i0, i1, i2 = tris[t]
        x0, x1, x2 = sx[i0], sx[i1], sx[i2]
        y0, y1, y2 = sy[i0], sy[i1], sy[i2]
        xa, xb = int(max(0, math.floor(min(x0, x1, x2)))), int(min(width - 1, math.ceil(max(x0, x1, x2))))
        ya, yb = int(max(0, math.floor(min(y0, y1, y2)))), int(min(height - 1, math.ceil(max(y0, y1, y2))))
        if xa > xb or ya > yb:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        a = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
        b = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
        c = 1.0 - a - b
        inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        if not inside.any():
            continue
        # perspective-correct depth
        iz = a / zs[i0] + b / zs[i1] + c / zs[i2]
        depth = 1.0 / np.maximum(iz, 1e-12)
        zb = zbuf[ya:yb + 1, xa:xb + 1]
        win = inside & (depth < zb)
        if not win.any():
            continue
        zb[win] = depth[win]
        wa, wb, wc = (a / zs[i0] * depth)[win], (b / zs[i1] * depth)[win], (c / zs[i2] * depth)[win]
        img[ya:yb + 1, xa:xb + 1][win] = wa[:, None] * col[i0] + wb[:, None] * col[i1] + wc[:, None] * col[i2]
    return np.clip(img, 0, 1), (sx, sy, z)


def _auto_camera(P, view="front", aspect=4 / 3, margin=1.12):
    """Camera that frames the points ``P`` from ``view`` (fits the projected extent)."""
    from . import mathutil as mu
    lo, hi = P.min(axis=0), P.max(axis=0)
    c = (lo + hi) / 2
    fov = math.radians(35.0)
    dirs = {"front": [0.25, 0.12, 1.0], "back": [-0.25, 0.12, -1.0], "left": [1.0, 0.12, 0.2],
            "right": [-1.0, 0.12, 0.2], "top": [0.0, 1.0, 0.05]}
    d = np.asarray(dirs.get(view, dirs["front"]), float)
    d /= np.linalg.norm(d)
    R = mu.look_at_matrix(c + d, c)[:3, :3]
    L = (P - c) @ R                      # camera-aligned coordinates (x right, y up, z toward eye)
    ty, tx = math.tan(fov / 2), math.tan(fov / 2) * aspect
    # distance so every point fits: |x| <= (dist - z) * tan
    dist = float(np.max(np.maximum(np.abs(L[:, 0]) * margin / tx, np.abs(L[:, 1]) * margin / ty) + L[:, 2]))
    dist = max(dist, float(np.max(L[:, 2])) + 1e-3, 1e-3)
    return mu.look_at_matrix(c + d * dist, c), fov


def render_heatmap(scene, node_name, joint=None, frame=None, width=640, height=480, camera="auto",
                   view="front", skeleton=True, posed=True):
    """Render the skin weights of ``joint`` on ``node_name`` as a heat map (or every influence in
    its own colour when ``joint`` is None). Returns float RGB (H, W, 3) in display space."""
    from .rig import skin_data, skin_mesh, bind_world
    from . import mathutil as mu
    node = scene.nodes[node_name]
    if "skin" not in node.spec:
        raise ValueError(f"{node_name!r} has no skin")
    from .rig import _skin_spec
    sk = _skin_spec(node)
    bf = float(sk.get("bind_frame", scene.frame_start))
    frame = bf if frame is None else float(frame)
    rest = scene.object_mesh(node_name, bf)
    data = skin_data(scene, node, rest)
    W, joints = data["weights"], data["joints"]
    if posed:
        wm = skin_mesh(scene, node, rest, frame)
        P = np.asarray(wm.verts, float)
    else:
        P = mu.transform_points(data["mesh_bind"], rest.verts)
    tris = np.asarray(rest.triangles()[0], np.int64)
    if joint is None:
        vcol = W @ influence_colors(len(joints))
    else:
        if joint not in joints:
            raise ValueError(f"{joint!r} is not a skin joint of {node_name!r}: {joints}")
        vcol = heat_color(W[:, joints.index(joint)])
    if camera == "auto" or camera is None:
        M, fov = _auto_camera(P, view, width / height)
    else:
        from .scene import camera_params
        cp = camera_params(scene, camera, frame, width / height)
        M, fov = cp["matrix"], cp["fov_y"]
    img, _ = rasterize(P, tris, vcol, M, fov, width, height)
    if skeleton:
        img = _draw_skeleton(img, scene, joints, frame if posed else bf, M, fov, joint, posed)
    return img


def _draw_skeleton(img, scene, joints, frame, M, fov, highlight, posed):
    from PIL import Image, ImageDraw
    from .rig import bind_world
    H, Wd = img.shape[:2]
    st = scene.state(frame)
    pos = {j: (st.world(j) if posed else bind_world(scene, st, j))[:3, 3] for j in joints}
    Vinv = np.linalg.inv(M)
    f = 0.5 * H / math.tan(fov / 2)

    def proj(p):
        c = Vinv[:3, :3] @ p + Vinv[:3, 3]
        z = max(-c[2], 1e-6)
        return (Wd * 0.5 + c[0] / z * f, H * 0.5 - c[1] / z * f), -c[2] > 1e-4

    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im, "RGBA")
    rad = max(2, int(H / 160))
    for j in joints:
        p = scene.nodes[j].parent
        a, oka = proj(pos[j])
        if p in pos:
            b, okb = proj(pos[p])
            if oka and okb:
                dr.line([b, a], fill=(235, 235, 245, 150), width=max(1, rad // 2 + 1))
    for j in joints:
        a, ok = proj(pos[j])
        if not ok:
            continue
        hot = j == highlight
        r = rad * (1.8 if hot else 1)
        dr.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r],
                   fill=(255, 80, 200, 230) if hot else (250, 250, 255, 190), outline=(20, 20, 30, 200))
    return np.asarray(im).astype(float) / 255.0
