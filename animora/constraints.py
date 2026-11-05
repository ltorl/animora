"""Transform constraints (aim, point, orient, parent, scale, path, limit).

A constraint is a dict in a node's ``constraints`` list::

    {"type": "aim", "target": "ball", "weight": 1.0}

Constraints are evaluated in order *after* the node's own local transform has been combined
with its parent's world matrix. Each one produces a new world matrix that is blended with the
incoming one by ``weight`` (animatable, 0..1).
"""
from __future__ import annotations

import math

import numpy as np

from . import mathutil as mu
from .params import Param

CONSTRAINT_TYPES = ("aim", "look_at", "point", "orient", "parent", "scale", "path", "motion_path",
                    "limit", "copy_transforms")


def _ax(v, default):
    from .scene import axis_vector
    return axis_vector(v, default)


def _targets(spec):
    """Normalise target specification -> list of (target, weight Param)."""
    t = spec.get("targets", spec.get("target"))
    if t is None:
        raise ValueError(f"constraint {spec.get('type')!r} needs 'target' or 'targets'")
    if isinstance(t, (str, dict)):
        t = [t]
    elif isinstance(t, (list, tuple)) and len(t) and all(isinstance(x, (int, float)) for x in t):
        t = [list(t)]  # a single point
    out = []
    for item in t:
        if isinstance(item, dict):
            name = item.get("node", item.get("target", item.get("point")))
            out.append((name, Param(item.get("weight", 1.0))))
        else:
            out.append((item, Param(1.0)))
    return out


def _target_world(st, target, frame):
    if isinstance(target, str):
        if target not in st.scene.nodes:
            raise ValueError(f"constraint target {target!r} does not exist")
        return st.world(target)
    p = np.asarray(Param(target).evaluate(frame, st.ctx), float).ravel()
    m = np.eye(4)
    m[:3, 3] = p[:3]
    return m


def _weighted_targets(st, spec):
    mats, ws = [], []
    for tgt, w in _targets(spec):
        mats.append(_target_world(st, tgt, st.frame))
        ws.append(max(0.0, float(w.evaluate(st.frame, st.ctx))))
    return mats, np.asarray(ws)


def _with_rot(W, R3):
    """Replace rotation of W (keeping translation and scale)."""
    s = np.linalg.norm(W[:3, :3], axis=0)
    out = W.copy()
    out[:3, :3] = R3 * s[None, :]
    return out


def _rot(W):
    _, R, _ = mu.decompose(W)
    return R


def _default_aim_axis(node):
    return "-z" if node.type in ("camera", "light") else "x"


# --------------------------------------------------------------------------------------------
def c_aim(st, node, W, pw, spec, idx):
    mats, ws = _weighted_targets(st, spec)
    if ws.sum() <= 0:
        return W
    tp = sum(w * m[:3, 3] for m, w in zip(mats, ws)) / ws.sum()
    pos = W[:3, 3]
    d = tp - pos
    if np.linalg.norm(d) < 1e-9:
        return W
    aim_axis = _ax(spec.get("aim_axis"), _default_aim_axis(node))
    up_axis = _ax(spec.get("up_axis"), "y")
    up_type = spec.get("world_up_type", "object" if spec.get("world_up_object") else "vector")
    if up_type == "object" and spec.get("world_up_object"):
        up = st.world(spec["world_up_object"])[:3, 3] - pos
    elif up_type == "object_rotation" and spec.get("world_up_object"):
        up = st.world(spec["world_up_object"])[:3, :3] @ _ax(spec.get("world_up", "y"), "y")
    elif up_type == "none":
        up = W[:3, :3] @ up_axis
    else:
        up = np.asarray(Param(spec.get("world_up", [0, 1, 0])).evaluate(st.frame, st.ctx), float)
    R = mu.aim_rotation(d, aim_axis, up, up_axis)
    off = spec.get("offset")
    if off is not None:
        R = R @ mu.euler_matrix(Param(off).evaluate(st.frame, st.ctx), "xyz")[:3, :3]
    return _with_rot(W, R)


def c_point(st, node, W, pw, spec, idx):
    mats, ws = _weighted_targets(st, spec)
    if ws.sum() <= 0:
        return W
    p = sum(w * m[:3, 3] for m, w in zip(mats, ws)) / ws.sum()
    off = np.asarray(Param(spec.get("offset", [0, 0, 0])).evaluate(st.frame, st.ctx), float)
    if spec.get("maintain_offset"):
        off = off + _bind_offset(st, node, spec, idx, "point")
    out = W.copy()
    p = p + off
    for i, ax in enumerate("xyz"):
        if ax in str(spec.get("skip", "")):
            p[i] = W[i, 3]
    out[:3, 3] = p
    return out


def c_orient(st, node, W, pw, spec, idx):
    mats, ws = _weighted_targets(st, spec)
    if ws.sum() <= 0:
        return W
    qs = []
    for m in mats:
        q = mu.matrix_to_quat(_rot(m))
        if qs and np.dot(qs[0], q) < 0:
            q = -q
        qs.append(q)
    q = mu.quat_normalize(sum(w * q for q, w in zip(qs, ws)))
    R = mu.quat_to_matrix3(q)
    off = spec.get("offset")
    if off is not None:
        R = R @ mu.euler_matrix(Param(off).evaluate(st.frame, st.ctx), "xyz")[:3, :3]
    if spec.get("maintain_offset"):
        R = R @ _bind_offset(st, node, spec, idx, "orient")
    return _with_rot(W, R)


def c_scale(st, node, W, pw, spec, idx):
    mats, ws = _weighted_targets(st, spec)
    if ws.sum() <= 0:
        return W
    s = sum(w * np.linalg.norm(m[:3, :3], axis=0) for m, w in zip(mats, ws)) / ws.sum()
    out = W.copy()
    out[:3, :3] = _rot(W) * s[None, :]
    return out


def c_parent(st, node, W, pw, spec, idx):
    mats, ws = _weighted_targets(st, spec)
    if ws.sum() <= 0:
        return W
    maintain = spec.get("maintain_offset", True)
    outs = []
    for k, m in enumerate(mats):
        if maintain:
            off = _bind_offset(st, node, spec, idx, "parent", k)
            outs.append(m @ off)
        else:
            outs.append(m.copy())
    return outs[0] if len(outs) == 1 else mu.blend_matrices(outs, ws)


def _bind_offset(st, node, spec, idx, kind, k=0):
    """Offset captured at ``bind_frame`` (default: first frame) between the node and its target."""
    cache = node.__dict__.setdefault("_constraint_offsets", {})
    key = (idx, kind, k)
    if key in cache:
        return cache[key]
    scene = st.scene
    bf = float(spec.get("bind_frame", scene.frame_start))
    bst = scene.state(bf)
    own = bst.parent_world(node.name) @ bst.local(node.name)
    tgts = _targets(spec)
    tm = _target_world(bst, tgts[k][0], bf)
    if kind == "parent":
        off = np.linalg.inv(tm) @ own
    elif kind == "point":
        if len(tgts) > 1:
            mats = [_target_world(bst, t, bf) for t, _ in tgts]
            ws = np.array([float(w.evaluate(bf, bst.ctx)) for _, w in tgts])
            tp = sum(w * m[:3, 3] for m, w in zip(mats, ws)) / max(ws.sum(), 1e-9)
        else:
            tp = tm[:3, 3]
        off = own[:3, 3] - tp
    else:  # orient
        off = _rot(tm).T @ _rot(own)
    cache[key] = off
    return off


def path_frame(pts, u, arc=True):
    """Position and tangent at parameter ``u`` (0..1) along a polyline (arc-length if ``arc``)."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if arc:
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total <= 0:
            return pts[0], np.array([0, 0, -1.0])
        s = np.clip(u, 0.0, 1.0) * total
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        t = (s - cum[i]) / max(seg[i], 1e-12)
    else:
        x = np.clip(u, 0.0, 1.0) * (len(pts) - 1)
        i = int(min(int(x), len(pts) - 2))
        t = x - i
    p = pts[i] * (1 - t) + pts[i + 1] * t
    # smooth tangent: blend neighbouring segment directions
    d0 = pts[i + 1] - pts[i]
    tan = mu.normalize(d0)
    if t > 0.5 and i + 2 < len(pts):
        tan = mu.normalize(mu.lerp(tan, mu.normalize(pts[i + 2] - pts[i + 1]), t - 0.5))
    elif t < 0.5 and i > 0:
        tan = mu.normalize(mu.lerp(mu.normalize(pts[i] - pts[i - 1]), tan, t + 0.5))
    return p, tan


def c_path(st, node, W, pw, spec, idx):
    scene = st.scene
    if "curve" in spec:
        pts = scene.curve_points(spec["curve"], st.frame, world=True)
    elif "points" in spec:
        from .scene import catmull_rom
        raw = np.asarray(Param(spec["points"]).evaluate(st.frame, st.ctx), float).reshape(-1, 3)
        pts = catmull_rom(raw, int(spec.get("samples", 16)), bool(spec.get("closed", False))) \
            if spec.get("smooth", True) and len(raw) > 2 else raw
    else:
        raise ValueError("path constraint needs 'curve' (a curve node) or 'points'")
    if "u" in spec:
        u = float(Param(spec["u"]).evaluate(st.frame, st.ctx))
    else:  # default: traverse the whole path over the frame range
        f0 = float(spec.get("start_frame", scene.frame_start))
        f1 = float(spec.get("end_frame", scene.frame_end))
        u = (st.frame - f0) / max(f1 - f0, 1e-9)
    if spec.get("loop") or spec.get("cycle"):
        u = u % 1.0
    p, tan = path_frame(pts, u, spec.get("arc_length", True))
    out = W.copy()
    out[:3, 3] = p + np.asarray(Param(spec.get("offset", [0, 0, 0])).evaluate(st.frame, st.ctx), float)
    if spec.get("follow", True):
        aim_axis = _ax(spec.get("aim_axis"), _default_aim_axis(node))
        up_axis = _ax(spec.get("up_axis"), "y")
        up = np.asarray(Param(spec.get("world_up", [0, 1, 0])).evaluate(st.frame, st.ctx), float)
        bank = float(Param(spec.get("bank", 0.0)).evaluate(st.frame, st.ctx))
        if bank:
            du = 0.01
            _, t2 = path_frame(pts, min(1.0, u + du), True)
            _, t1 = path_frame(pts, max(0.0, u - du), True)
            turn = float(np.dot(np.cross(t1, t2), mu.normalize(up)))
            ang = float(np.clip(turn * bank * 50.0, -float(spec.get("bank_limit", 60.0)),
                                float(spec.get("bank_limit", 60.0))))
            q = mu.quat_axis_angle(tan, -ang)
            up = mu.quat_to_matrix3(q) @ up
        R = mu.aim_rotation(tan, aim_axis, up, up_axis)
        out = _with_rot(out, R)
    return out


def c_limit(st, node, W, pw, spec, idx):
    """Clamp the node's local (parent-space) translate / rotate / scale."""
    L = np.linalg.inv(pw) @ W
    t, R, s = mu.decompose(L)
    order = node.rotate_order
    r = mu.matrix_to_euler(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]), order)
    g = lambda k: (None if spec.get(k) is None else
                   np.asarray(Param(spec[k]).evaluate(st.frame, st.ctx), float) * np.ones(3))

    def clamp(v, lo, hi):
        v = np.asarray(v, float).copy()
        if lo is not None:
            v = np.where(np.isnan(lo), v, np.maximum(v, np.nan_to_num(lo, nan=-np.inf)))
        if hi is not None:
            v = np.where(np.isnan(hi), v, np.minimum(v, np.nan_to_num(hi, nan=np.inf)))
        return v

    def gn(k):
        v = spec.get(k)
        if v is None:
            return None
        v = [np.nan if x is None else x for x in (v if isinstance(v, (list, tuple)) else [v] * 3)]
        return np.asarray(Param(v).evaluate(st.frame, st.ctx), float)

    t = clamp(t, gn("translate_min"), gn("translate_max"))
    r = clamp(r, gn("rotate_min"), gn("rotate_max"))
    s = clamp(s, gn("scale_min"), gn("scale_max"))
    return pw @ mu.compose(t, r, s, order)


HANDLERS = {"aim": c_aim, "look_at": c_aim, "point": c_point, "orient": c_orient, "parent": c_parent,
            "copy_transforms": c_parent, "scale": c_scale, "path": c_path, "motion_path": c_path,
            "limit": c_limit}


def apply_constraints(st, node, W, pw):
    for i, spec in enumerate(node.constraints):
        kind = spec.get("type")
        fn = HANDLERS.get(kind)
        if fn is None:
            raise ValueError(f"node {node.name!r}: unknown constraint {kind!r}; use one of {sorted(HANDLERS)}")
        if spec.get("enabled", True) is False:
            continue
        if kind == "copy_transforms":
            spec = dict(spec, maintain_offset=spec.get("maintain_offset", False))
        w = float(Param(spec.get("weight", 1.0)).evaluate(st.frame, st.ctx))
        if w <= 0:
            continue
        Wc = fn(st, node, W, pw, spec, i)
        if w >= 1:
            W = Wc
        else:
            W = mu.blend_matrices([W, Wc], [1 - w, w])
    return W
