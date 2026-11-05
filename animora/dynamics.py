"""Dynamics: rigid bodies, cloth (PBD), particles and secondary-motion jiggle.

All simulations are deterministic, step forward from the scene start frame, cache every integer
frame and interpolate for sub-frames (motion blur). Caches can be baked to ``.npz`` files.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import mathutil as mu
from . import noise as nz
from .params import Param

DEFAULT_DYNAMICS = {
    "gravity": [0.0, -9.81, 0.0],
    "ground": True,
    "ground_height": 0.0,
    "ground_friction": 0.6,
    "ground_restitution": 0.4,
    "substeps": 8,
    "cache": None,
}

SPHERE_PRIMS = ("uv_sphere", "ico_sphere", "sphere")



def _cross3(a, b):
    """Cross product of two 3-vectors (much faster than np.cross for single vectors)."""
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])

def _pv(v, frame=0.0, ctx=None):
    """Evaluate a raw spec value that may be an animated Param spec."""
    if isinstance(v, (int, float)):
        return float(v)
    return Param(v).evaluate(frame, ctx)


# ============================================================================================
# collision shapes
# ============================================================================================
SHAPE_KINDS = ("auto", "sphere", "box", "plane", "capsule", "cylinder", "mesh")


def _shape_of(scene, name, spec=None, frame=None):
    """Return a collision shape description for node ``name`` in object space.

    Kinds: ``sphere``, ``box`` (``plane``), ``capsule`` and ``cylinder`` (Y axis), and ``mesh``
    (exact signed distance to the node's triangle mesh, any shape, concave allowed; ``convex``
    and ``hull`` are aliases). ``auto`` picks from the primitive."""
    node = scene.nodes[name]
    spec = dict(spec or {})
    kind = spec.get("shape", "auto")
    kind = {"convex": "mesh", "hull": "mesh", "sdf": "mesh"}.get(kind, kind)
    prim = node.spec.get("primitive")
    fr = frame if frame is not None else scene.frame_start
    if kind == "auto":
        if prim in SPHERE_PRIMS:
            kind = "sphere"
        elif prim == "plane" or node.type == "cloth":
            kind = "plane"
        elif prim in ("cylinder", "capsule"):
            kind = prim
        else:
            kind = "box"

    def pp(key, default=None):
        if key in spec:
            return float(spec[key])
        if key in node.prim_params:
            return float(node.prim_params[key].evaluate(0))
        return default

    if kind == "sphere":
        r = spec.get("radius")
        if r is None:
            r = float(node.prim_params["radius"].evaluate(0)) if "radius" in node.prim_params else None
        if r is None:
            m = scene.object_mesh(name, fr)
            r = float(np.max(np.linalg.norm(m.verts, axis=1))) if m is not None else 0.5
        return {"kind": "sphere", "radius": float(r), "center": np.zeros(3)}
    if kind in ("capsule", "cylinder"):
        r, h = pp("radius"), pp("height")
        c = np.zeros(3)
        if r is None or h is None or prim not in ("capsule", "cylinder"):
            m = scene.object_mesh(name, fr)
            if m is not None and m.num_verts:
                lo, hi = (np.asarray(v) for v in m.bounds())
                c = (lo + hi) * 0.5
                half = (hi - lo) * 0.5
                r = r if r is not None else float(max(half[0], half[2]))
                h = h if h is not None else float(2 * half[1])
            r = 0.5 if r is None else r
            h = 1.0 if h is None else h
        hh = max(h / 2 - r, 0.0) if kind == "capsule" else h / 2
        return {"kind": kind, "radius": float(r), "half_height": float(hh), "center": c}
    if kind == "mesh":
        m = scene.object_mesh(name, fr)
        if m is None or not m.num_verts:
            raise ValueError(f"{name}: shape 'mesh' needs mesh geometry")
        T, _ = m.triangles()
        lo, hi = (np.asarray(v) for v in m.bounds())
        c = (lo + hi) * 0.5
        return {"kind": "mesh", "verts": np.asarray(m.verts, float) - c, "tris": np.asarray(T, np.int32),
                "center": c}
    if kind in ("box", "plane"):
        if "size" in spec:
            s = spec["size"]
            half = np.asarray([s] * 3 if np.ndim(s) == 0 else s, float) * 0.5
            c = np.zeros(3)
        else:
            m = scene.object_mesh(name, fr)
            if m is None:
                half, c = np.full(3, 0.5), np.zeros(3)
            else:
                lo, hi = m.bounds()
                half = (np.asarray(hi) - np.asarray(lo)) * 0.5
                c = (np.asarray(hi) + np.asarray(lo)) * 0.5
        half = np.maximum(half, 1e-3)
        return {"kind": "box", "half": half, "center": c}
    raise ValueError(f"{name}: unknown collision shape {kind!r} ({', '.join(SHAPE_KINDS)})")


def _collider_shape(scene, name, spec, frame):
    """Collider shape for cloth / particles: primitives use analytic shapes, any other mesh
    (characters, props, deforming geometry) uses its exact mesh signed distance at ``frame``."""
    spec = dict(spec or {})
    node = scene.nodes[name]
    if spec.get("shape", "auto") == "auto" and node.spec.get("primitive") not in \
            SPHERE_PRIMS + ("cube", "box", "plane", "cylinder", "capsule") and node.type != "cloth":
        spec["shape"] = "mesh"
    kind = spec.get("shape", "auto")
    if kind in ("mesh", "convex", "hull", "sdf"):
        return _shape_of(scene, name, spec, frame)
    return _shape_of(scene, name, spec)


def _world_shape(shape, W):
    """Instantiate a shape in world space given the node world matrix (scale baked in)."""
    t, R, s = mu.decompose(W)
    s = np.abs(s)
    c = mu.transform_points(W, shape["center"][None])[0]
    k = shape["kind"]
    if k == "sphere":
        return {"kind": "sphere", "c": c, "R": R, "r": shape["radius"] * float(np.max(s))}
    if k in ("capsule", "cylinder"):
        return {"kind": k, "c": c, "R": R, "r": shape["radius"] * float(max(s[0], s[2])),
                "hh": shape["half_height"] * float(s[1])}
    if k == "mesh":
        return {"kind": "mesh", "c": c, "R": R, "V": shape["verts"] * s, "T": shape["tris"]}
    return {"kind": "box", "c": c, "R": R, "h": shape["half"] * s}


def _sdf_world(shp, P):
    """Signed distance (negative inside) and outward unit normal of world points ``P`` (N,3)."""
    P = np.asarray(P, float).reshape(-1, 3)
    k = shp["kind"]
    if k == "ground":
        n = np.zeros_like(P)
        n[:, 1] = 1.0
        return P[:, 1] - shp["y"], n
    if k == "sphere":
        d = P - shp["c"]
        L = np.linalg.norm(d, axis=1)
        n = d / np.maximum(L, 1e-9)[:, None]
        n[L < 1e-9] = [0, 1, 0]
        return L - shp["r"], n
    l = (P - shp["c"]) @ shp["R"]
    if k == "box":
        h = shp["h"]
        q = np.abs(l) - h
        qo = np.maximum(q, 0)
        out = np.linalg.norm(qo, axis=1)
        mx = q.max(1)
        d = out + np.minimum(mx, 0)
        sg = np.where(l >= 0, 1.0, -1.0)
        nl = sg * qo / np.maximum(out, 1e-12)[:, None]
        inside = out <= 1e-12
        if np.any(inside):
            ax = np.argmax(q[inside], axis=1)
            ni = np.zeros((int(inside.sum()), 3))
            ni[np.arange(len(ax)), ax] = sg[inside][np.arange(len(ax)), ax]
            nl[inside] = ni
    elif k == "capsule":
        v = l.copy()
        v[:, 1] -= np.clip(l[:, 1], -shp["hh"], shp["hh"])
        L = np.linalg.norm(v, axis=1)
        nl = v / np.maximum(L, 1e-9)[:, None]
        nl[L < 1e-9] = [1, 0, 0]
        d = L - shp["r"]
    elif k == "cylinder":
        rad = np.hypot(l[:, 0], l[:, 2])
        radial = np.stack([l[:, 0], np.zeros(len(l)), l[:, 2]], 1) / np.maximum(rad, 1e-9)[:, None]
        radial[rad < 1e-9] = [1, 0, 0]
        dx = rad - shp["r"]
        dy = np.abs(l[:, 1]) - shp["hh"]
        ydir = np.zeros_like(l)
        ydir[:, 1] = np.where(l[:, 1] >= 0, 1.0, -1.0)
        ox, oy = np.maximum(dx, 0), np.maximum(dy, 0)
        out = np.hypot(ox, oy)
        d = out + np.minimum(np.maximum(dx, dy), 0)
        nl = (radial * ox[:, None] + ydir * oy[:, None]) / np.maximum(out, 1e-12)[:, None]
        inside = out <= 1e-12
        nl[inside] = np.where((dx > dy)[inside, None], radial[inside], ydir[inside])
    elif k == "mesh":
        from .render import native as nat
        d, cp = nat.mesh_sdf(shp["V"], shp["T"], l, closest=True)
        d = d.astype(float)
        v = l - cp
        L = np.linalg.norm(v, axis=1)
        nl = v / np.maximum(L, 1e-12)[:, None] * np.where(d < 0, -1.0, 1.0)[:, None]
        nl[L < 1e-12] = [0, 1, 0]
    else:
        raise ValueError(f"unknown shape kind {k!r}")
    return d, nl @ shp["R"].T


def _collide_points(P, shp, thickness=0.0):
    """Push points out of a world shape. Returns (newP, normals, hitmask)."""
    n = np.zeros_like(P)
    hit = np.zeros(len(P), bool)
    if len(P) == 0:
        return P, n, hit
    d, nn = _sdf_world(shp, P)
    hit = d < thickness
    if np.any(hit):
        P = P.copy()
        P[hit] = P[hit] + nn[hit] * (thickness - d[hit])[:, None]
        n[hit] = nn[hit]
    return P, n, hit


def _shape_samples(local, scale):
    """Surface sample points of a shape (body frame, centred, scale applied) for generic contacts."""
    from . import geometry as geo
    k = local["kind"]
    s = np.abs(scale)
    if k == "box":
        h = local["half"] * s
        g = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                      if (x, y, z) != (0, 0, 0)], float)
        return g * h
    if k == "capsule":
        r = local["radius"] * max(s[0], s[2])
        hh = local["half_height"] * s[1]
        return geo.capsule(r, 2 * hh + 2 * r, segments=12, rings=3).verts
    if k == "cylinder":
        r = local["radius"] * max(s[0], s[2])
        hh = local["half_height"] * s[1]
        m = geo.cylinder(r, 2 * hh, segments=16, height_segments=2)
        caps = np.array([[0, -hh, 0], [0, hh, 0]] + [[0.5 * r * math.cos(a), sy * hh, 0.5 * r * math.sin(a)]
                                                      for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)
                                                      for sy in (-1, 1)])
        return np.vstack([m.verts, caps])
    if k == "mesh":
        V = local["verts"] * s
        T = local["tris"]
        pts = np.vstack([V, V[T].mean(1)]) if len(T) < 4000 else V
        if len(pts) > 600:
            idx = np.random.default_rng(0).choice(len(pts), 600, replace=False)
            pts = pts[idx]
        return pts
    return np.zeros((0, 3))


# ============================================================================================
# rigid bodies
# ============================================================================================
class _Body:
    __slots__ = ("name", "active", "x", "q", "v", "w", "inv_m", "inv_I", "shape", "scale", "e", "mu",
                 "start", "lin_damp", "ang_damp", "W_prev")


def _spread_idx(P, depth, k=8):
    """Indices of up to ``k`` contact points: deepest first, spatially spread out."""
    order = np.argsort(-depth)
    if len(order) <= k:
        return list(order)
    ext = float(np.max(P.max(0) - P.min(0))) if len(P) else 0.0
    minsep = ext / (k ** 0.5) * 0.5
    keep = []
    for idx in order:
        if all(np.linalg.norm(P[idx] - P[q]) > minsep for q in keep):
            keep.append(idx)
            if len(keep) >= k:
                break
    return keep


def _spread(P, depth, k=8):
    return [P[i] for i in _spread_idx(np.asarray(P).reshape(-1, 3), np.asarray(depth), k)]


def _box_corners(b):
    h = b.shape["h"]
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    R = mu.quat_to_matrix3(b.q)
    return b.x + (s * h) @ R.T


class RigidWorld:
    def __init__(self, dyn):
        self.dyn = dyn
        self.scene = dyn.scene
        self.names = [n for n in self.scene.order if "rigid_body" in self.scene.nodes[n].spec]
        self.cache = {}   # frame(int) -> {name: (x, q)}
        self.bodies = None
        self.frame = None

    # ---------------------------------------------------------------- setup
    def _init(self):
        sc = self.scene
        f0 = int(math.floor(sc.frame_start))
        bodies = []
        for name in self.names:
            node = sc.nodes[name]
            rb = node.spec["rigid_body"]
            rb = {"type": rb} if isinstance(rb, str) else (dict(rb) if isinstance(rb, dict) else {})
            W = sc.state(f0).world(name, dynamics=False)
            t, R, s = mu.decompose(W)
            b = _Body()
            b.name = name
            b.active = rb.get("type", "active") in ("active", "dynamic")
            b.start = float(rb.get("start_frame", f0))
            local_shape = _shape_of(sc, name, rb)
            ws = _world_shape(local_shape, W)
            b.shape = ws
            b.shape["local"] = local_shape
            b.shape["samples"] = _shape_samples(local_shape, s)
            b.scale = s
            b.x = ws["c"].copy()
            b.q = mu.matrix_to_quat(R)
            fps = sc.fps
            b.v = np.asarray(rb.get("velocity", [0, 0, 0]), float)
            b.w = np.radians(np.asarray(rb.get("angular_velocity", [0, 0, 0]), float))
            mass = float(rb.get("mass", 1.0))
            b.e = float(rb.get("restitution", rb.get("bounce", 0.4)))
            b.mu = float(rb.get("friction", 0.5))
            b.lin_damp = float(rb.get("linear_damping", 0.02))
            b.ang_damp = float(rb.get("angular_damping", 0.05))
            if b.active and mass > 0:
                b.inv_m = 1.0 / mass
                k = ws["kind"]
                if k == "sphere":
                    I = np.full(3, 0.4 * mass * ws["r"] ** 2)
                elif k in ("capsule", "cylinder"):
                    r = ws["r"]
                    h = 2 * ws["hh"] + (1.2 * r if k == "capsule" else 0.0)
                    ixx = mass * (3 * r * r + h * h) / 12.0
                    I = np.array([ixx, 0.5 * mass * r * r, ixx])
                else:
                    d = 2 * ws["h"] if k == "box" else (ws["V"].max(0) - ws["V"].min(0))
                    I = mass / 12.0 * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
                    if k == "mesh":
                        I *= 0.7   # mass concentrated inward compared with the bounding box
                b.inv_I = 1.0 / np.maximum(I, 1e-9)
            else:
                b.inv_m = 0.0
                b.inv_I = np.zeros(3)
            # world-space offset of shape centre from node origin (in body rotation frame)
            b.W_prev = W
            bodies.append(b)
        self.bodies = bodies
        self.frame = f0
        self._store(f0)

    def _store(self, f):
        self.cache[f] = {b.name: (b.x.copy(), b.q.copy()) for b in self.bodies}

    def _inv_I_world(self, b):
        R = mu.quat_to_matrix3(b.q)
        return R @ np.diag(b.inv_I) @ R.T

    # ---------------------------------------------------------------- contacts
    def _contacts(self, ground_y):
        C = []  # (i, j, point, normal (from j to i), depth)
        B = self.bodies
        for i, b in enumerate(B):
            if b.inv_m == 0:
                continue
            if self.dyn.ground:
                if b.shape["kind"] == "sphere":
                    d = ground_y - (b.x[1] - b.shape["r"])
                    if d > 0:
                        C.append((i, -1, b.x - np.array([0, b.shape["r"], 0]), np.array([0, 1.0, 0]), d))
                else:
                    pts = _box_corners(b) if b.shape["kind"] == "box" else self._samples(b)
                    for p in _spread(pts[pts[:, 1] < ground_y], ground_y - pts[pts[:, 1] < ground_y][:, 1]):
                        C.append((i, -1, p, np.array([0, 1.0, 0]), ground_y - p[1]))
            for j, o in enumerate(B):
                if j == i or (o.inv_m > 0 and j < i):
                    continue
                C += self._pair(i, b, j, o)
        return C

    @staticmethod
    def _samples(b):
        return b.x + b.shape["samples"] @ mu.quat_to_matrix3(b.q).T

    def _generic_pair(self, i, a, j, b):
        """Contacts between arbitrary shapes: surface samples of each body against the exact
        signed distance of the other (capsule, cylinder, mesh and mixed pairs)."""
        out = []
        ra = a.shape.get("r") if a.shape["kind"] == "sphere" else None
        rb = b.shape.get("r") if b.shape["kind"] == "sphere" else None
        # cheap bounding-sphere reject
        if np.linalg.norm(a.x - b.x) > self._bound_r(a) + self._bound_r(b):
            return out
        if ra is not None:
            d, n = _sdf_world(b.shape, a.x[None])
            if d[0] < ra:
                out.append((i, j, a.x - n[0] * ra, n[0], float(ra - d[0])))
            return out
        if rb is not None:
            d, n = _sdf_world(a.shape, b.x[None])
            if d[0] < rb:
                out.append((i, j, b.x - n[0] * rb, -n[0], float(rb - d[0])))
            return out
        Sa = self._samples(a)
        d, n = _sdf_world(b.shape, Sa)
        m = d < 0
        cand = [(Sa[k], n[k], -d[k]) for k in np.nonzero(m)[0]]
        Sb = self._samples(b)
        d, n = _sdf_world(a.shape, Sb)
        m = d < 0
        cand += [(Sb[k], -n[k], -d[k]) for k in np.nonzero(m)[0]]
        if not cand:
            return out
        P = np.array([c[0] for c in cand])
        D = np.array([c[2] for c in cand])
        for k in _spread_idx(P, D):
            out.append((i, j, cand[k][0], cand[k][1], float(cand[k][2])))
        return out

    @staticmethod
    def _bound_r(b):
        if "bound_r" not in b.shape:
            k = b.shape["kind"]
            if k == "sphere":
                r = b.shape["r"]
            elif k == "box":
                r = float(np.linalg.norm(b.shape["h"]))
            else:
                r = float(np.max(np.linalg.norm(b.shape["samples"], axis=1))) * 1.05 + 1e-6
            b.shape["bound_r"] = r
        return b.shape["bound_r"]

    def _pair(self, i, a, j, b):
        ka, kb = a.shape["kind"], b.shape["kind"]
        out = []
        if ka not in ("sphere", "box") or kb not in ("sphere", "box"):
            return self._generic_pair(i, a, j, b)
        if ka == "sphere" and kb == "sphere":
            d = a.x - b.x
            L = np.linalg.norm(d)
            pen = a.shape["r"] + b.shape["r"] - L
            if pen > 0 and L > 1e-9:
                n = d / L
                out.append((i, j, a.x - n * a.shape["r"], n, pen))
        elif ka == "sphere" and kb == "box":
            hit = self._sphere_box(a, b)
            if hit:
                p, n, pen = hit
                out.append((i, j, p, n, pen))
        elif ka == "box" and kb == "sphere":
            hit = self._sphere_box(b, a)
            if hit:
                p, n, pen = hit
                out.append((i, j, p, -n, pen))
        else:  # box-box: SAT pair normal, then penetrating corners of each box
            out += self._box_box(i, a, j, b)
        return out

    @staticmethod
    def _box_box(i, a, j, b):
        """Separating-axis test over the 15 box axes; contacts use one consistent pair normal
        (pointing from ``b`` to ``a``) so resting stacks get stable face-face support."""
        Ra, Rb = mu.quat_to_matrix3(a.q), mu.quat_to_matrix3(b.q)
        ha, hb = a.shape["h"], b.shape["h"]
        d = a.x - b.x
        axes = [Ra[:, k] for k in range(3)] + [Rb[:, k] for k in range(3)]
        for u in range(3):
            for v in range(3):
                c = np.cross(Ra[:, u], Rb[:, v])
                L = np.linalg.norm(c)
                if L > 1e-6:
                    axes.append(c / L)
        best, bn = np.inf, None
        for k, ax in enumerate(axes):
            ra = float(np.sum(ha * np.abs(Ra.T @ ax)))
            rb = float(np.sum(hb * np.abs(Rb.T @ ax)))
            dist = float(np.dot(d, ax))
            pen = ra + rb - abs(dist)
            if pen <= 0:
                return []
            if k >= 6:
                pen = pen * 1.05 + 1e-4   # prefer face axes (more stable contact manifolds)
            if pen < best:
                best, bn = pen, (ax if dist >= 0 else -ax)
        n = bn
        out = []
        ext_b = float(np.dot(b.x, n)) + float(np.sum(hb * np.abs(Rb.T @ n)))   # b's surface facing a
        ext_a = float(np.dot(a.x, n)) - float(np.sum(ha * np.abs(Ra.T @ n)))   # a's surface facing b
        margin = 0.02
        for p in _box_corners(a):
            lp = (p - b.x) @ Rb
            if np.all(np.abs(lp) <= hb + margin * np.maximum(hb, 0.05) * 4):
                dep = ext_b - float(np.dot(p, n))
                if 0 < dep <= best * 1.5 + 1e-3:
                    out.append((i, j, p, n, dep))
        for q in _box_corners(b):
            lq = (q - a.x) @ Ra
            if np.all(np.abs(lq) <= ha + margin * np.maximum(ha, 0.05) * 4):
                dep = float(np.dot(q, n)) - ext_a
                if 0 < dep <= best * 1.5 + 1e-3:
                    out.append((i, j, q, n, dep))
        if not out:  # edge-edge: one contact between the deepest support points
            sa = a.x - Ra @ (ha * np.sign(Ra.T @ n))
            sb = b.x + Rb @ (hb * np.sign(Rb.T @ n))
            out.append((i, j, 0.5 * (sa + sb), n, best))
        return out

    @staticmethod
    def _sphere_box(s, b):
        R = mu.quat_to_matrix3(b.q)
        lp = (s.x - b.x) @ R
        h = b.shape["h"]
        cp = np.clip(lp, -h, h)
        d = lp - cp
        L = np.linalg.norm(d)
        r = s.shape["r"]
        if L > 1e-9:
            if L >= r:
                return None
            n = R @ (d / L)
            return b.x + R @ cp, n, r - L
        # centre inside the box
        pen = h - np.abs(lp)
        ax = int(np.argmin(pen))
        nl = np.zeros(3)
        nl[ax] = 1.0 if lp[ax] >= 0 else -1.0
        return s.x - (R @ nl) * r, R @ nl, float(pen[ax] + r)

    # ---------------------------------------------------------------- stepping
    def _step(self, dt, t_frame):
        g = self.dyn.gravity
        B = self.bodies
        sc = self.scene
        # kinematic (passive) bodies follow their animation
        for b in B:
            if b.inv_m == 0:
                W = sc.state(t_frame).world(b.name, dynamics=False)
                ws = _world_shape(b.shape["local"], W)
                newx = ws["c"]
                b.v = (newx - b.x) / dt
                b.x = newx
                b.q = mu.matrix_to_quat(mu.decompose(W)[1])
                b.shape.update({k: v for k, v in ws.items() if k != "kind"})
        for b in B:
            if b.inv_m == 0 or t_frame < b.start:
                continue
            b.v = b.v + g * dt
            b.v *= max(0.0, 1.0 - b.lin_damp * dt)
            b.w *= max(0.0, 1.0 - b.ang_damp * dt)
        frozen = []
        for b in B:   # bodies whose start_frame hasn't arrived act as static colliders
            if b.inv_m > 0 and t_frame < b.start:
                frozen.append((b, b.inv_m, b.inv_I, b.v, b.w))
                b.inv_m, b.inv_I = 0.0, np.zeros(3)
                b.v, b.w = np.zeros(3), np.zeros(3)
        C = self._contacts(self.dyn.ground_y)
        if C:
            Iw = {i: self._inv_I_world(B[i]) for i in range(len(B))}
            prep = []
            for (i, j, p, n, depth) in C:
                a = B[i]
                o = B[j] if j >= 0 else None
                ra = p - a.x
                rb = p - o.x if o is not None else np.zeros(3)
                t1 = _cross3(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
                t1 /= np.linalg.norm(t1)
                t2 = _cross3(n, t1)

                def eff(dirv, i=i, j=j, ra=ra, rb=rb, o=o):
                    k = a.inv_m + float(np.dot(dirv, _cross3(Iw[i] @ _cross3(ra, dirv), ra)))
                    if o is not None and o.inv_m > 0:
                        k += o.inv_m + float(np.dot(dirv, _cross3(Iw[j] @ _cross3(rb, dirv), rb)))
                    return max(k, 1e-9)
                vr = a.v + _cross3(a.w, ra) - ((o.v + _cross3(o.w, rb)) if o is not None else 0.0)
                vn0 = float(np.dot(vr, n))
                e = max(a.e, o.e if o is not None else self.dyn.ground_e)
                target = -e * vn0 if vn0 < -1.0 else 0.0
                mu_ = math.sqrt(a.mu * (o.mu if o is not None else self.dyn.ground_mu))
                prep.append([i, j, ra, rb, n, t1, t2, eff(n), eff(t1), eff(t2), target, mu_, 0.0, 0.0, 0.0])

            def rel_v(c):
                a = B[c[0]]
                o = B[c[1]] if c[1] >= 0 else None
                v = a.v + _cross3(a.w, c[2])
                if o is not None:
                    v = v - (o.v + _cross3(o.w, c[3]))
                return v

            def apply(c, J):
                a = B[c[0]]
                a.v = a.v + J * a.inv_m
                a.w = a.w + Iw[c[0]] @ _cross3(c[2], J)
                if c[1] >= 0:
                    o = B[c[1]]
                    if o.inv_m > 0:
                        o.v = o.v - J * o.inv_m
                        o.w = o.w - Iw[c[1]] @ _cross3(c[3], J)
            for it in range(12):
                for c in prep:
                    n, t1, t2 = c[4], c[5], c[6]
                    vr = rel_v(c)
                    # normal (accumulated, clamped >= 0)
                    dj = (c[10] - float(np.dot(vr, n))) / c[7]
                    old = c[12]
                    c[12] = max(old + dj, 0.0)
                    apply(c, (c[12] - old) * n)
                    # friction (accumulated, clamped to the Coulomb cone)
                    lim = c[11] * c[12]
                    vr = rel_v(c)
                    for (tv, kidx, aidx) in ((t1, 8, 13), (t2, 9, 14)):
                        dj = -float(np.dot(vr, tv)) / c[kidx]
                        old = c[aidx]
                        c[aidx] = min(max(old + dj, -lim), lim)
                        apply(c, (c[aidx] - old) * tv)
        # integrate
        for b in B:
            if b.inv_m == 0 or t_frame < b.start:
                continue
            b.x = b.x + b.v * dt
            wl = np.linalg.norm(b.w)
            if wl > 1e-12:
                dq = mu.quat_axis_angle(b.w / wl, math.degrees(wl * dt))
                b.q = mu.quat_normalize(mu.quat_mul(dq, b.q))
        # positional correction (deepest contact per pair, so multi-corner contacts don't over-push)
        deepest = {}
        for (i, j, p, n, depth) in self._contacts(self.dyn.ground_y):
            k = (i, j)
            if k not in deepest or depth > deepest[k][1]:
                deepest[k] = (n, depth)
        for (i, j), (n, depth) in deepest.items():
            a = B[i]
            o = B[j] if j >= 0 else None
            wsum = a.inv_m + (o.inv_m if o is not None else 0.0)
            if wsum <= 0:
                continue
            corr = max(depth - 0.001, 0.0) * 0.5 / wsum * n
            a.x = a.x + corr * a.inv_m
            if o is not None:
                o.x = o.x - corr * o.inv_m
        for b, m, I, v, w in frozen:
            b.inv_m, b.inv_I, b.v, b.w = m, I, v, w

    def advance_to(self, f):
        if self.bodies is None:
            self._init()
        sub = self.dyn.substeps
        dt = 1.0 / self.scene.fps / sub
        while self.frame < f:
            for s in range(sub):
                self._step(dt, self.frame + (s + 1) / sub)
            self.frame += 1
            self._store(self.frame)

    def transform(self, name, frame, W):
        fi = int(math.floor(frame))
        f0 = int(math.floor(self.scene.frame_start))
        if fi < f0:
            return None
        if fi + 1 not in self.cache and fi not in self.cache or (frame > fi and fi + 1 not in self.cache):
            if self.dyn.loaded:
                fi = min(fi, max(self.cache))
            else:
                self.advance_to(fi + (1 if frame > fi else 0))
        x0, q0 = self.cache[fi][name]
        if frame > fi and fi + 1 in self.cache:
            x1, q1 = self.cache[fi + 1][name]
            t = frame - fi
            x = x0 * (1 - t) + x1 * t
            q = mu.quat_slerp(q0, q1, t)
        else:
            x, q = x0, q0
        b = next(bb for bb in self.bodies if bb.name == name) if self.bodies else None
        _, _, s = mu.decompose(W)
        local = b.shape["local"] if b is not None else _shape_of(self.scene, name,
                                                                self.scene.nodes[name].spec.get("rigid_body"))
        R = mu.quat_to_matrix3(q)
        M = np.eye(4)
        M[:3, :3] = R * s[None, :]
        # x is the world centre of the shape; recover the node origin
        M[:3, 3] = x - M[:3, :3] @ local["center"]
        return M


# ============================================================================================
# jiggle
# ============================================================================================
class JiggleSim:
    def __init__(self, dyn, name):
        self.dyn = dyn
        self.scene = dyn.scene
        self.name = name
        js = self.scene.nodes[name].spec["jiggle"]
        js = {} if js is True else dict(js)
        self.k = float(js.get("stiffness", 120.0))
        self.c = float(js.get("damping", 8.0))
        self.amount = float(js.get("amount", 1.0))
        self.grav = float(js.get("gravity", 0.0))
        self.cache = {}
        self.frame = None

    def _target(self, t):
        return self.scene.state(t).world(self.name, dynamics=False)[:3, 3]

    def offset(self, frame):
        sc = self.scene
        f0 = int(math.floor(sc.frame_start))
        fi = int(math.floor(frame))
        if fi < f0:
            return np.zeros(3)
        if self.frame is None:
            self.x = self._target(f0).copy()
            self.v = np.zeros(3)
            self.frame = f0
            self.cache[f0] = np.zeros(3)
        need = fi + (1 if frame > fi else 0)
        sub = max(4, self.dyn.substeps)
        dt = 1.0 / sc.fps / sub
        while self.frame < need:
            for s in range(sub):
                t = self.frame + (s + 1) / sub
                tg = self._target(t)
                a = self.k * (tg - self.x) - self.c * self.v + self.dyn.gravity * self.grav
                self.v = self.v + a * dt
                self.x = self.x + self.v * dt
            self.frame += 1
            self.cache[self.frame] = self.x - self._target(self.frame)
        o0 = self.cache.get(fi, np.zeros(3))
        if frame > fi:
            o1 = self.cache.get(fi + 1, o0)
            t = frame - fi
            return (o0 * (1 - t) + o1 * t) * self.amount
        return o0 * self.amount


# ============================================================================================
# cloth
# ============================================================================================
def _select_pins(spec, rest, W0=None):
    """Pin specification -> vertex indices (object-space rest mesh)."""
    V = rest.verts
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = np.maximum(hi - lo, 1e-9)
    out = set()
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    for it in items:
        if isinstance(it, (int, np.integer)):
            out.add(int(it))
        elif isinstance(it, str):
            if it == "corners":
                for sx in (-1, 1):
                    for sz in (-1, 1):
                        out.add(int(np.argmax(sx * V[:, 0] + sz * V[:, 2])))
            elif it in ("top", "bottom"):
                ax = 1 if ext[1] > 1e-6 else 2
                val = hi[ax] if it == "top" else lo[ax]
                if ext[1] <= 1e-6 and it == "top":
                    val = lo[2]  # flat plane: "top" = far (-z) edge
                out |= set(np.nonzero(np.abs(V[:, ax] - val) < ext[ax] * 1e-3 + 1e-6)[0].tolist())
            elif it.startswith("edge_"):
                _, mm, axn = it.split("_")
                ax = "xyz".index(axn)
                val = hi[ax] if mm == "max" else lo[ax]
                out |= set(np.nonzero(np.abs(V[:, ax] - val) < ext[ax] * 1e-3 + 1e-6)[0].tolist())
            elif it == "center":
                out.add(int(np.argmin(np.linalg.norm(V - (lo + hi) / 2, axis=1))))
            else:
                raise ValueError(f"unknown pin {it!r} (corners, top, bottom, center, edge_min_x, edge_max_z, ...)")
        elif isinstance(it, dict):
            c = np.asarray(it.get("near", it.get("center", [0, 0, 0])), float)
            r = float(it.get("radius", 0.05))
            out |= set(np.nonzero(np.linalg.norm(V - c, axis=1) <= r)[0].tolist())
        else:
            raise ValueError(f"bad pin entry {it!r}")
    return np.array(sorted(i for i in out if 0 <= i < len(V)), int)


def _bend_pairs(mesh):
    """Vertex pairs opposite each interior edge (for bending resistance)."""
    tri = mesh.to_triangles()["vidx"]
    T = len(tri)
    if T == 0:
        return np.zeros((0, 2), int)
    e = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        lo = np.minimum(tri[:, a], tri[:, b])
        hi = np.maximum(tri[:, a], tri[:, b])
        e.append(np.stack([lo, hi, tri[:, c]], axis=1))
    e = np.concatenate(e)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e = e[order]
    same = (e[1:, 0] == e[:-1, 0]) & (e[1:, 1] == e[:-1, 1])
    idx = np.nonzero(same)[0]
    pairs = np.stack([e[idx, 2], e[idx + 1, 2]], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pairs


class ClothSim:
    def __init__(self, dyn, name):
        self.dyn = dyn
        self.scene = sc = dyn.scene
        self.name = name
        node = sc.nodes[name]
        self.node = node
        sp = node.spec
        self.rest = sc.cloth_rest_mesh(name)
        self.f0 = int(math.floor(float(sp.get("start_frame", sc.frame_start))))
        W0 = sc.state(self.f0).world(name, dynamics=False)
        self.x = mu.transform_points(W0, self.rest.verts)
        self.v = np.zeros_like(self.x)
        E, _ = self.rest.edges()
        self.E = E
        self.L0 = np.linalg.norm(self.x[E[:, 0]] - self.x[E[:, 1]], axis=1)
        self.Bp = _bend_pairs(self.rest)
        self.B0 = np.linalg.norm(self.x[self.Bp[:, 0]] - self.x[self.Bp[:, 1]], axis=1) if len(self.Bp) else None
        self.pins = _select_pins(sp["pin"], self.rest) if sp.get("pin") is not None else np.zeros(0, int)
        self.stretch = float(sp.get("stiffness", 1.0))
        self.bend = float(sp.get("bend", sp.get("bend_stiffness", 0.2)))
        self.iters = int(sp.get("iterations", 25))
        self.damping = float(sp.get("damping", 0.01))
        self.drag = float(sp.get("drag", 0.0))
        self.thick = float(sp.get("thickness", 0.01))
        self.friction = float(sp.get("friction", 0.4))
        self.mass_scale = float(sp.get("gravity_scale", 1.0))
        self.substeps = int(sp.get("substeps", max(4, dyn.substeps)))
        self.wind = sp.get("wind")
        self.turb = sp.get("turbulence", 0.0)
        self.seed = int(sp.get("seed", 0))
        self.colliders = list(sp.get("colliders", []))
        self.self_ground = bool(sp.get("ground", dyn.ground))
        # self-collision (vertex-vertex repulsion with rest-distance filtering, native hash grid)
        self.self_collision = bool(sp.get("self_collision", False))
        avg_edge = float(np.mean(self.L0)) if len(self.L0) else 0.05
        self.self_thick = float(sp.get("self_thickness", 0.9 * avg_edge))
        self.self_ignore = float(sp.get("self_rest_ignore", max(2.0 * avg_edge, 1.5 * self.self_thick)))
        self.self_every = max(1, int(sp.get("self_collision_every", 5)))
        self.self_contacts = 0
        self._rest_world = self.x.astype(np.float32).copy()
        self.cache = {self.f0: self.x.copy()}
        self.frame = self.f0
        # mass weighting: pinned verts are infinite mass
        self.inv_w = np.ones(len(self.x))
        self.inv_w[self.pins] = 0.0
        # ---- tearing: per-corner vertex indices may change as the cloth rips
        tear = sp.get("tear", False)
        self.tear = bool(tear) if not isinstance(tear, (int, float)) or isinstance(tear, bool) else True
        self.tear_strain = float(sp.get("tear_strain", tear if (isinstance(tear, (int, float))
                                                                and not isinstance(tear, bool)) else 1.5))
        self.tear_max = max(1, int(sp.get("tear_max", 6)))           # vertex splits per substep
        self.tear_var = float(sp.get("tear_variation", 0.15))         # per-edge threshold jitter
        self.tear_start = float(sp.get("tear_start", self.f0))        # no tearing before this frame
        self.idx = self.rest.indices.copy()                           # current per-corner vertex ids
        self.vsrc = np.arange(len(self.x))                            # rest vertex each sim vertex came from
        self.parent = np.arange(len(self.x))                          # vertex each split vertex came from
        self.topos = [self.idx.copy()]                                # topology versions
        self.topo_at = {self.f0: 0}
        self.torn = 0
        if self.tear:
            self._corner_face = np.repeat(np.arange(len(self.rest.counts)), self.rest.counts)
            self._update_tear_thresh()

    def _collider_shapes(self, t):
        out = []
        sc = self.scene
        for c in self.colliders:
            if c == "ground":
                continue
            spec = c if isinstance(c, dict) else {"node": c}
            nm = spec["node"]
            W = sc.state(t).world(nm, dynamics=True)
            out.append(_world_shape(_collider_shape(sc, nm, spec, t), W))
        if self.self_ground or "ground" in self.colliders:
            out.append({"kind": "ground", "y": self.dyn.ground_y})
        return out

    def _step(self, dt, t):
        sc = self.scene
        ctx = sc.context(t)
        g = self.dyn.gravity * self.mass_scale
        x, v = self.x, self.v
        acc = np.broadcast_to(g, x.shape).copy()
        if self.wind is not None:
            w = np.asarray(Param(self.wind).evaluate(t, ctx), float)
            acc += w
        turb = float(_pv(self.turb, t, ctx)) if self.turb else 0.0
        if turb:
            acc += turb * nz.noise_vec3(x * 1.5 + np.array([0, 0, t / sc.fps * 0.7]), self.seed)
        if self.drag:
            acc -= self.drag * v
        v = v + acc * dt
        v *= (1.0 - self.damping)
        p = x + v * dt
        # pins follow the node transform
        if len(self.pins):
            W = sc.state(t).world(self.name, dynamics=False)
            p[self.pins] = mu.transform_points(W, self.rest.verts[self.vsrc[self.pins]])
        iw = self.inv_w
        E, L0 = self.E, self.L0
        shapes = self._collider_shapes(t)
        for it in range(self.iters):
            # stretch
            d = p[E[:, 1]] - p[E[:, 0]]
            L = np.linalg.norm(d, axis=1)
            wsum = iw[E[:, 0]] + iw[E[:, 1]]
            ok = (L > 1e-12) & (wsum > 0)
            corr = np.zeros_like(d)
            corr[ok] = ((L[ok] - L0[ok]) / (L[ok] * wsum[ok]))[:, None] * d[ok] * self.stretch
            dp = np.zeros_like(p)
            cnt = np.zeros(len(p))
            np.add.at(dp, E[:, 0], corr * iw[E[:, 0], None])
            np.add.at(dp, E[:, 1], -corr * iw[E[:, 1], None])
            np.add.at(cnt, E[:, 0], 1)
            np.add.at(cnt, E[:, 1], 1)
            # bending
            if self.B0 is not None and self.bend > 0:
                Bp = self.Bp
                d = p[Bp[:, 1]] - p[Bp[:, 0]]
                L = np.linalg.norm(d, axis=1)
                wsum = iw[Bp[:, 0]] + iw[Bp[:, 1]]
                ok = (L > 1e-12) & (wsum > 0)
                corr = np.zeros_like(d)
                corr[ok] = ((L[ok] - self.B0[ok]) / (L[ok] * wsum[ok]))[:, None] * d[ok] * self.bend
                np.add.at(dp, Bp[:, 0], corr * iw[Bp[:, 0], None])
                np.add.at(dp, Bp[:, 1], -corr * iw[Bp[:, 1], None])
                np.add.at(cnt, Bp[:, 0], 1)
                np.add.at(cnt, Bp[:, 1], 1)
            p = p + 1.5 * dp / np.maximum(cnt, 1)[:, None]
            if self.self_collision and (it % self.self_every == self.self_every - 1 or it == self.iters - 1):
                p = self._self_collide(p, x)
            for shp in shapes:
                p, n, hit = _collide_points(p, shp, self.thick)
        # velocity update + friction on contacts
        v = (p - x) / dt
        for shp in shapes:
            _, n, hit = _collide_points(p, shp, self.thick + 1e-4)
            if np.any(hit):
                vn = np.sum(v[hit] * n[hit], axis=1, keepdims=True)
                vt = v[hit] - vn * n[hit]
                v[hit] = vt * (1 - self.friction) + np.maximum(vn, 0) * n[hit]
        if len(self.pins):
            v[self.pins] = 0.0
        self.x, self.v = p, v
        if self.tear and t >= self.tear_start:
            self._tear()

    # ------------------------------------------------------------------ tearing
    def _update_tear_thresh(self):
        # stable per-edge jitter (hash of the rest vertex ids) so cracks wander naturally
        a, b = self.vsrc[self.E[:, 0]], self.vsrc[self.E[:, 1]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        h = ((lo * 73856093) ^ (hi * 19349663) ^ (self.seed * 83492791)) & 0xFFFF
        self._tear_thresh = self.tear_strain * (1.0 + self.tear_var * (h / 65535.0 - 0.5) * 2.0)

    def _split_vertex(self, a, d):
        """Split vertex ``a`` along the plane through it with normal ``d`` (the stress direction):
        faces on the +d side get a new copy of the vertex, opening a crack perpendicular to the
        stress. Returns False when every incident face lies on one side (nothing to split)."""
        if not hasattr(self, "_offs"):
            self._offs = np.concatenate([[0], np.cumsum(self.rest.counts)[:-1]])
        corners = np.nonzero(self.idx == a)[0]
        if len(corners) < 2:
            return False
        faces = self._corner_face[corners]
        cnt = self.rest.counts
        cen = np.empty((len(faces), 3))
        for i, f in enumerate(faces):
            o = self._offs[f]
            cen[i] = self.x[self.idx[o:o + cnt[f]]].mean(axis=0)
        side = (cen - self.x[a]) @ d > 0
        if side.all() or not side.any():
            return False
        n = len(self.x)
        self.idx[corners[side]] = n
        self.x = np.vstack([self.x, self.x[a]])
        self.v = np.vstack([self.v, self.v[a]])
        self.vsrc = np.append(self.vsrc, self.vsrc[a])
        self.parent = np.append(self.parent, a)
        self.inv_w = np.append(self.inv_w, self.inv_w[a])
        self._rest_world = np.vstack([self._rest_world, self._rest_world[a]])
        if self.inv_w[a] == 0.0:
            self.pins = np.append(self.pins, n)
        return True

    def _tear(self):
        if not len(self.E):
            return
        x, E = self.x, self.E
        d = x[E[:, 1]] - x[E[:, 0]]
        L = np.linalg.norm(d, axis=1)
        strain = L / np.maximum(self.L0, 1e-12)
        cand = np.nonzero(strain > self._tear_thresh)[0]
        if not len(cand):
            return
        cand = cand[np.argsort(-strain[cand])]
        used = set()
        n_split = 0
        for e in cand:
            if n_split >= self.tear_max:
                break
            a, b = int(E[e, 0]), int(E[e, 1])
            if a in used or b in used:
                continue
            dirn = d[e] / max(L[e], 1e-12)
            # split the endpoint that has the most incident faces (interior first), else the other
            order = sorted((a, b), key=lambda q: -int(np.count_nonzero(self.idx == q)))
            for q in order:
                if self.inv_w[q] == 0.0 and self.inv_w[a + b - q] != 0.0:
                    continue  # prefer to rip free cloth rather than duplicate pins
                if self._split_vertex(q, dirn if q == a else -dirn):
                    used.update((a, b))
                    n_split += 1
                    break
        if n_split:
            self.torn += n_split
            self._rebuild_constraints()

    def _rebuild_constraints(self):
        from .geometry import Mesh
        m = Mesh(self.x, self.rest.counts, self.idx)
        self.E, _ = m.edges()
        R = self._rest_world.astype(np.float64)
        self.L0 = np.linalg.norm(R[self.E[:, 0]] - R[self.E[:, 1]], axis=1)
        self.Bp = _bend_pairs(m)
        self.B0 = np.linalg.norm(R[self.Bp[:, 0]] - R[self.Bp[:, 1]], axis=1) if len(self.Bp) else None
        self._update_tear_thresh()
        self.topos.append(self.idx.copy())

    def topology(self, frame):
        """Per-corner vertex indices at ``frame`` (None when the cloth never tore)."""
        if len(self.topos) == 1:
            return None
        fi = int(math.floor(frame))
        if fi < self.f0:
            return None
        if frame > fi and (fi + 1) in self.topo_at:
            fi += 1  # sub-frame samples use the later topology (verts() extends to match)
        keys = [k for k in self.topo_at if k <= fi]
        return self.topos[self.topo_at[max(keys)]] if keys else None

    def _self_collide(self, p, prev):
        from .render import native as nat
        q = np.ascontiguousarray(p, np.float32)
        pinned = q[self.pins].copy() if len(self.pins) else None
        self.self_contacts = nat.self_collide(q, prev.astype(np.float32), self._rest_world, self.self_thick,
                                              self.self_ignore, iterations=2)
        if pinned is not None:
            q[self.pins] = pinned
        return q.astype(np.float64)

    def advance_to(self, f):
        dt = 1.0 / self.scene.fps / self.substeps
        while self.frame < f:
            for s in range(self.substeps):
                self._step(dt, self.frame + (s + 1) / self.substeps)
            self.frame += 1
            self.cache[self.frame] = self.x.copy()
            self.topo_at[self.frame] = len(self.topos) - 1

    def _extend(self, a, n):
        """Grow a vertex array from an earlier frame to ``n`` vertices (split copies start where
        their parent vertex was)."""
        if len(a) >= n:
            return a[:n]
        out = np.empty((n, 3))
        out[:len(a)] = a
        for k in range(len(a), n):
            out[k] = out[self.parent[k]]
        return out

    def verts(self, frame):
        fi = int(math.floor(frame))
        if fi < self.f0:
            W = self.scene.state(frame).world(self.name, dynamics=False)
            return mu.transform_points(W, self.rest.verts)
        need = fi + (1 if frame > fi else 0)
        if need not in self.cache:
            if self.dyn.loaded and self.cache:
                need = min(need, max(self.cache))
                fi = min(fi, need)
            else:
                self.advance_to(need)
        a = self.cache[fi]
        if frame > fi and fi + 1 in self.cache:
            t = frame - fi
            b = self.cache[fi + 1]
            return self._extend(a, len(b)) * (1 - t) + b * t
        return a


# ============================================================================================
# particles
# ============================================================================================
def _curve_eval(curve, x):
    """Piecewise-linear curve [[x0,y0],[x1,y1],...] evaluated at array x; scalars are constants."""
    if curve is None:
        return np.ones_like(x)
    if isinstance(curve, (int, float)):
        return np.full_like(x, float(curve))
    c = np.asarray(curve, float)
    return np.interp(x, c[:, 0], c[:, 1])


class ParticleSim:
    def __init__(self, dyn, name):
        self.dyn = dyn
        self.scene = sc = dyn.scene
        self.name = name
        self.node = sc.nodes[name]
        sp = self.node.spec
        self.sp = sp
        em = sp.get("emitter", {})
        self.em = {"shape": em} if isinstance(em, str) else dict(em)
        self.start = float(sp.get("start_frame", sc.frame_start))
        self.end = float(sp.get("end_frame", sc.frame_end))
        self.max_n = int(sp.get("max_particles", 5000))
        self.seed = int(sp.get("seed", 1))
        self.f0 = int(math.floor(min(self.start, sc.frame_start)))
        self.P = np.zeros((0, 3))
        self.V = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.life = np.zeros(0)
        self.size = np.zeros(0)
        self.ids = np.zeros(0, np.int64)
        self.next_id = 0
        self.accum = 0.0
        self.frame = self.f0
        self.cache = {}
        self._store()
        self._mesh_src = None

    def _store(self):
        self.cache[self.frame] = (self.P.copy(), self.V.copy(), self.age.copy(), self.life.copy(),
                                  self.size.copy(), self.ids.copy())

    # ---------------------------------------------------------------- emission
    def _emit_points(self, n, rng, W):
        em = self.em
        shape = em.get("shape", "point")
        if shape == "point":
            P = np.zeros((n, 3))
            N = np.tile([0, 1.0, 0], (n, 1))
        elif shape == "sphere":
            r = float(em.get("radius", 0.5))
            d = rng.normal(size=(n, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            surf = em.get("surface", False)
            rr = r if surf else r * rng.random(n)[:, None] ** (1 / 3)
            P = d * rr
            N = d
        elif shape == "box":
            s = em.get("size", 1.0)
            s = np.asarray([s] * 3 if np.ndim(s) == 0 else s, float)
            P = (rng.random((n, 3)) - 0.5) * s
            N = np.tile([0, 1.0, 0], (n, 1))
        elif shape in ("disc", "circle"):
            r = float(em.get("radius", 0.5))
            a = rng.random(n) * 2 * math.pi
            rr = r * np.sqrt(rng.random(n))
            P = np.stack([np.cos(a) * rr, np.zeros(n), np.sin(a) * rr], axis=1)
            N = np.tile([0, 1.0, 0], (n, 1))
        elif shape == "mesh":
            src = em.get("node")
            m = self.scene.object_mesh(src, self.frame) if src else None
            if m is None:
                raise ValueError(f"particles {self.name!r}: mesh emitter needs 'node'")
            tri = m.to_triangles()
            Pt = tri["P"]
            area = 0.5 * np.linalg.norm(np.cross(Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0]), axis=1)
            cdf = np.cumsum(area)
            k = np.searchsorted(cdf, rng.random(n) * cdf[-1])
            k = np.clip(k, 0, len(Pt) - 1)
            u, v = rng.random(n), rng.random(n)
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            P = Pt[k, 0] + u[:, None] * (Pt[k, 1] - Pt[k, 0]) + v[:, None] * (Pt[k, 2] - Pt[k, 0])
            N = mu.normalize(np.cross(Pt[k, 1] - Pt[k, 0], Pt[k, 2] - Pt[k, 0]))
            Wm = self.scene.state(self.frame).world(src)
            P = mu.transform_points(Wm, P)
            N = mu.normalize(N @ np.linalg.inv(Wm[:3, :3]))
            return P, N
        else:
            raise ValueError(f"particles {self.name!r}: unknown emitter shape {shape!r} "
                             "(point, sphere, box, disc, mesh)")
        P = mu.transform_points(W, P)
        N = mu.normalize(N @ W[:3, :3].T)
        return P, N

    def _emit(self, n, t, rng, W, ctx):
        if n <= 0:
            return
        n = min(n, self.max_n - len(self.P))
        if n <= 0:
            return
        sp = self.sp
        P, N = self._emit_points(n, rng, W)
        d = sp.get("direction", "normal" if self.em.get("shape") in ("sphere", "mesh") else [0, 1, 0])
        if isinstance(d, str) and d == "normal":
            base = N
        else:
            base = np.tile(mu.normalize(W[:3, :3] @ np.asarray(_pv(d, t, ctx), float)), (n, 1))
        spread = math.radians(float(_pv(sp.get("spread", 15.0), t, ctx)))
        # random direction in a cone around base
        cos_max = math.cos(min(spread, math.pi))
        z = cos_max + (1 - cos_max) * rng.random(n)
        phi = rng.random(n) * 2 * math.pi
        sxy = np.sqrt(np.maximum(0, 1 - z * z))
        local = np.stack([sxy * np.cos(phi), sxy * np.sin(phi), z], axis=1)
        dirs = np.empty_like(local)
        for i in range(n):  # build frame around base[i]
            b = base[i]
            a = np.array([1.0, 0, 0]) if abs(b[0]) < 0.9 else np.array([0, 1.0, 0])
            tx = mu.normalize(np.cross(a, b))
            ty = np.cross(b, tx)
            dirs[i] = local[i, 0] * tx + local[i, 1] * ty + local[i, 2] * b
        speed = float(_pv(sp.get("speed", 2.0), t, ctx))
        sr = float(sp.get("speed_random", 0.2))
        V = dirs * (speed * (1 + sr * (rng.random(n) * 2 - 1)))[:, None]
        life = float(_pv(sp.get("lifetime", 2.0), t, ctx))
        lr = float(sp.get("lifetime_random", 0.3))
        L = np.maximum(1e-3, life * (1 + lr * (rng.random(n) * 2 - 1)))
        sz = float(_pv(sp.get("size", 0.05), t, ctx))
        szr = float(sp.get("size_random", 0.3))
        S = np.maximum(1e-5, sz * (1 + szr * (rng.random(n) * 2 - 1)))
        # spread births inside the step so emission is smooth
        self.P = np.concatenate([self.P, P])
        self.V = np.concatenate([self.V, V])
        self.age = np.concatenate([self.age, np.zeros(n)])
        self.life = np.concatenate([self.life, L])
        self.size = np.concatenate([self.size, S])
        self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + n)])
        self.next_id += n

    # ---------------------------------------------------------------- forces
    def _forces(self, t, ctx):
        sp = self.sp
        A = np.broadcast_to(self.dyn.gravity * float(sp.get("gravity_scale", 1.0)), self.P.shape).copy()
        drag = float(sp.get("drag", 0.0))
        if drag:
            A -= drag * self.V
        tsec = t / self.scene.fps
        for f in sp.get("fields", []) or []:
            ft = f.get("type")
            s = float(_pv(f.get("strength", 1.0), t, ctx))
            if ft in ("turbulence", "curl", "noise"):
                fr = float(f.get("frequency", 1.0))
                q = self.P * fr + np.array([0, 0, tsec * float(f.get("speed", 0.5))])
                A += s * (nz.curl_noise(q, int(f.get("seed", 0))) if ft == "curl"
                          else nz.noise_vec3(q, int(f.get("seed", 0))))
            elif ft == "wind":
                d = np.asarray(_pv(f.get("direction", [1, 0, 0]), t, ctx), float)
                A += s * mu.normalize(d) - float(f.get("drag", 0.0)) * self.V
            elif ft == "vortex":
                c = np.asarray(f.get("center", [0, 0, 0]), float)
                ax = mu.normalize(np.asarray(f.get("axis", [0, 1, 0]), float))
                r = self.P - c
                A += s * np.cross(ax, r) / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 0.1)
                A -= float(f.get("pull", 0.0)) * (r - (r @ ax)[:, None] * ax)
            elif ft in ("attract", "repel"):
                tg = f.get("target", f.get("point", [0, 0, 0]))
                c = self.scene.state(t).world(tg)[:3, 3] if isinstance(tg, str) else np.asarray(tg, float)
                r = c - self.P
                L = np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-3)
                fall = float(f.get("falloff", 1.0))
                mag = s / np.maximum(L, 0.2) ** fall
                A += (mag * r / L) * (1 if ft == "attract" else -1)
            elif ft == "drag":
                A -= s * self.V
            elif ft == "gravity":
                A += s * np.asarray(f.get("direction", [0, -1, 0]), float)
            else:
                raise ValueError(f"unknown particle field {ft!r} (turbulence, curl, wind, vortex, attract, "
                                 "repel, drag, gravity)")
        return A

    # ---------------------------------------------------------------- step
    def _step(self, dt, t):
        sc = self.scene
        ctx = sc.context(t)
        sp = self.sp
        W = sc.state(t).world(self.name)
        if self.start <= t <= self.end:
            rate = float(_pv(sp.get("rate", 100.0), t, ctx))
            self.accum += rate * dt
            burst = sp.get("burst")
            n = int(self.accum)
            self.accum -= n
            if burst and abs(t - float(burst.get("frame", self.start))) < 0.5 * dt * sc.fps + 1e-9:
                n += int(burst.get("count", 100))
            rng = np.random.default_rng((self.seed * 1000003 + int(round(t * 1000))) & 0xFFFFFFFF)
            self._emit(n, t, rng, W, ctx)
        if len(self.P):
            A = self._forces(t, ctx)
            self.V = self.V + A * dt
            self.P = self.P + self.V * dt
            self.age = self.age + dt
            if sp.get("collide_ground", self.dyn.ground):
                gy = self.dyn.ground_y
                below = self.P[:, 1] - self.size * 0.5 < gy
                if np.any(below):
                    bounce = float(sp.get("bounce", 0.3))
                    fr = float(sp.get("friction", 0.2))
                    self.P[below, 1] = gy + self.size[below] * 0.5
                    vy = self.V[below, 1]
                    self.V[below, 1] = np.where(vy < 0, -vy * bounce, vy)
                    self.V[below, 0] *= (1 - fr)
                    self.V[below, 2] *= (1 - fr)
            for c in sp.get("colliders", []) or []:
                spec = c if isinstance(c, dict) else {"node": c}
                shp = _world_shape(_collider_shape(sc, spec["node"], spec, t), sc.state(t).world(spec["node"]))
                newP, n, hit = _collide_points(self.P, shp, 0.0)
                if np.any(hit):
                    vn = np.sum(self.V[hit] * n[hit], axis=1, keepdims=True)
                    self.V[hit] -= (1 + float(sp.get("bounce", 0.3))) * np.minimum(vn, 0) * n[hit]
                    self.P = newP
            alive = self.age < self.life
            if not np.all(alive):
                self.P, self.V, self.age = self.P[alive], self.V[alive], self.age[alive]
                self.life, self.size, self.ids = self.life[alive], self.size[alive], self.ids[alive]

    def advance_to(self, f):
        sub = max(1, int(self.sp.get("substeps", 2)))
        dt = 1.0 / self.scene.fps / sub
        while self.frame < f:
            for s in range(sub):
                self._step(dt, self.frame + (s + 1) / sub)
            self.frame += 1
            self._store()

    def state(self, frame):
        fi = int(math.floor(frame))
        if fi < self.f0:
            return None
        if fi not in self.cache:
            if self.dyn.loaded and self.cache:
                fi = min(fi, max(self.cache))
            else:
                self.advance_to(fi)
        P, V, age, life, size, ids = self.cache[fi]
        return {"P": P, "V": V, "age": age, "life": life, "size": size, "id": ids, "frame": fi}


# ============================================================================================
# manager
# ============================================================================================
class Dynamics:
    """Owns every simulation in a scene. Created automatically by :class:`animora.scene.Scene`."""

    def __init__(self, scene):
        self.scene = scene
        cfg = dict(DEFAULT_DYNAMICS)
        cfg.update(scene.data.get("dynamics", {}) or {})
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], float)
        self.ground = bool(cfg["ground"])
        self.ground_y = float(cfg["ground_height"])
        self.ground_mu = float(cfg["ground_friction"])
        self.ground_e = float(cfg["ground_restitution"])
        self.substeps = int(cfg["substeps"])
        self.rigid = RigidWorld(self) if any("rigid_body" in n.spec for n in scene.nodes.values()) else None
        self.jiggle = {n: JiggleSim(self, n) for n, nd in scene.nodes.items() if nd.spec.get("jiggle")}
        self.cloth = {}
        self.particle_sims = {}
        self.loaded = False
        self._busy = set()
        cache = cfg.get("cache")
        if cache:
            path = scene.resolve_path(cache)
            if os.path.exists(path):
                self.load(path)

    # ---------------------------------------------------------------- hooks used by Scene
    def world_override(self, name, frame, W):
        node = self.scene.nodes[name]
        if name in self._busy:
            return None
        if self.rigid is not None and "rigid_body" in node.spec:
            rb = node.spec["rigid_body"]
            typ = rb if isinstance(rb, str) else (rb.get("type", "active") if isinstance(rb, dict) else "active")
            if typ in ("active", "dynamic"):
                self._busy.add(name)
                try:
                    return self.rigid.transform(name, frame, W)
                finally:
                    self._busy.discard(name)
        if name in self.jiggle:
            self._busy.add(name)
            try:
                off = self.jiggle[name].offset(frame)
            finally:
                self._busy.discard(name)
            M = W.copy()
            M[:3, 3] = M[:3, 3] + off
            return M
        return None

    def _cloth(self, name):
        if name not in self.cloth:
            self.cloth[name] = ClothSim(self, name)
        return self.cloth[name]

    def cloth_verts(self, name, frame):
        return self._cloth(name).verts(frame)

    def cloth_topology(self, name, frame):
        """(indices, vsrc) of a torn cloth at ``frame``; None while the topology is the rest one."""
        c = self._cloth(name)
        idx = c.topology(frame)
        if idx is None:
            return None
        return idx, c.vsrc

    def _psim(self, name):
        if name not in self.particle_sims:
            self.particle_sims[name] = ParticleSim(self, name)
        return self.particle_sims[name]

    def particles(self, name, frame):
        """Particle state dict at the integer frame <= ``frame`` (P, V, age, life, size, id)."""
        return self._psim(name).state(frame)

    # ---------------------------------------------------------------- baking
    def simulate(self, end=None, progress=None):
        """Run every simulation up to ``end`` (default: scene end frame)."""
        sc = self.scene
        end = int(math.ceil(end if end is not None else sc.frame_end))
        names_cloth = [n for n, nd in sc.nodes.items() if nd.type == "cloth"]
        names_p = [n for n, nd in sc.nodes.items() if nd.type == "particles"]
        f = int(math.floor(sc.frame_start))
        while f <= end:
            if self.rigid is not None:
                self.rigid.advance_to(f)
            for n in self.jiggle:
                self.jiggle[n].offset(f)
            for n in names_cloth:
                self._cloth(n).verts(f)
            for n in names_p:
                self._psim(n).state(f)
            if progress:
                progress(f, end)
            f += 1

    def bake(self, path, end=None, progress=None):
        """Simulate and write all caches to ``path`` (.npz)."""
        self.simulate(end, progress)
        data = {}
        if self.rigid is not None:
            for f, d in self.rigid.cache.items():
                for n, (x, q) in d.items():
                    data[f"rigid|{n}|{f}"] = np.concatenate([x, q])
        for n, j in self.jiggle.items():
            for f, o in j.cache.items():
                data[f"jiggle|{n}|{f}"] = o
        for n, c in self.cloth.items():
            for f, v in c.cache.items():
                data[f"cloth|{n}|{f}"] = v.astype(np.float32)
            if len(c.topos) > 1:
                for i, t in enumerate(c.topos):
                    data[f"clothtopo|{n}|{i}"] = t.astype(np.int32)
                data[f"clothtopoat|{n}|0"] = np.array(sorted(c.topo_at.items()), np.int64)
                data[f"clothvsrc|{n}|0"] = c.vsrc.astype(np.int64)
                data[f"clothparent|{n}|0"] = c.parent.astype(np.int64)
        for n, p in self.particle_sims.items():
            for f, st in p.cache.items():
                for k, arr in zip(("P", "V", "age", "life", "size", "id"), st):
                    data[f"part|{n}|{f}|{k}"] = arr
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(path, **data)
        return path

    def load(self, path):
        """Load a baked cache; frames beyond the bake hold the last cached frame."""
        z = np.load(path)
        part = {}
        for key in z.files:
            kind, rest = key.split("|", 1)
            if kind == "rigid" and self.rigid is not None:
                n, f = rest.rsplit("|", 1)
                arr = z[key]
                if self.rigid.bodies is None:
                    self.rigid._init()
                self.rigid.cache.setdefault(int(f), {})[n] = (arr[:3], arr[3:])
            elif kind == "jiggle" and rest.rsplit("|", 1)[0] in self.jiggle:
                n, f = rest.rsplit("|", 1)
                j = self.jiggle[n]
                j.cache[int(f)] = z[key]
                j.frame = max(j.frame or -10 ** 9, int(f))
            elif kind == "cloth":
                n, f = rest.rsplit("|", 1)
                if n in self.scene.nodes:
                    c = self._cloth(n)
                    c.cache[int(f)] = z[key].astype(np.float64)
                    c.frame = max(c.frame, int(f))
            elif kind in ("clothtopo", "clothtopoat", "clothvsrc", "clothparent"):
                n, i = rest.rsplit("|", 1)
                if n in self.scene.nodes:
                    c = self._cloth(n)
                    arr = z[key]
                    if kind == "clothtopo":
                        i = int(i)
                        while len(c.topos) <= i:
                            c.topos.append(None)
                        c.topos[i] = arr.astype(np.int64)
                    elif kind == "clothtopoat":
                        c.topo_at = {int(a): int(b) for a, b in arr}
                    elif kind == "clothvsrc":
                        c.vsrc = arr.astype(np.int64)
                    else:
                        c.parent = arr.astype(np.int64)
            elif kind == "part":
                n, f, k = rest.split("|")
                part.setdefault(n, {}).setdefault(int(f), {})[k] = z[key]
        for n, frames in part.items():
            if n not in self.scene.nodes:
                continue
            p = self._psim(n)
            for f, d in frames.items():
                p.cache[f] = tuple(d[k] for k in ("P", "V", "age", "life", "size", "id"))
            p.frame = max(p.cache)
        if self.rigid is not None and self.rigid.cache:
            self.rigid.frame = max(self.rigid.cache)
        self.loaded = True


# ============================================================================================
# particle rendering geometry
# ============================================================================================
def _particle_template(scene, node, preview):
    rs = node.spec.get("render", {})
    rs = {"shape": rs} if isinstance(rs, str) else dict(rs)
    key = ("ptemplate", node.name, preview)
    if key in scene._rest_cache:
        return scene._rest_cache[key], rs
    from .geometry import make_primitive
    shape = rs.get("shape", "sphere")
    if "instance" in rs:
        m = scene.object_mesh(rs["instance"], scene.frame_start, preview)
    elif shape == "sphere":
        m = make_primitive("ico_sphere", radius=0.5, subdivisions=1 if preview else int(rs.get("subdivisions", 2)))
    elif shape == "cube":
        m = make_primitive("cube", size=1.0)
    elif shape in ("quad", "plane"):
        m = make_primitive("plane", size=1.0)
    elif shape == "tetra":
        m = make_primitive("ico_sphere", radius=0.5, subdivisions=0)
    else:
        raise ValueError(f"particles {node.name!r}: render shape {shape!r} unknown (sphere, cube, quad, tetra, "
                         "or {'instance': node})")
    tri = m.to_triangles()
    scene._rest_cache[key] = tri
    return tri, rs


def particle_geometry(scene, node, frame, f0, f1, preview=False):
    """Triangles for every live particle at shutter open ``f0`` and close ``f1`` (motion blur)."""
    dyn = scene.dynamics
    st = dyn.particles(node.name, frame)
    if st is None or len(st["P"]) == 0:
        return None
    tri, rs = _particle_template(scene, node, preview)
    fps = scene.fps
    base = st["frame"]
    P, V = st["P"], st["V"]
    life_t = np.clip((st["age"] + (frame - base) / fps) / st["life"], 0, 1)
    size = st["size"] * _curve_eval(node.spec.get("size_over_life"), life_t)
    keep = size > 1e-6
    if not np.all(keep):
        P, V, size = P[keep], V[keep], size[keep]
        ids = st["id"][keep]
    else:
        ids = st["id"]
    n = len(P)
    if n == 0:
        return None
    align = rs.get("align", "random" if "instance" in rs else "none")
    R = None
    if align == "velocity":
        R = np.stack([mu.aim_rotation(v if np.linalg.norm(v) > 1e-9 else [0, 1, 0], (0, 1, 0)) for v in V])
    elif align == "random":
        rng_q = []
        for i in ids:
            r = np.random.default_rng(int(i) * 2654435761 % (2 ** 32))
            q = r.normal(size=4)
            rng_q.append(q / np.linalg.norm(q))
        R = np.stack([mu.quat_to_matrix3(q) for q in rng_q])
    tP, tN = tri["P"], tri["N"]  # (T,3,3)
    T = len(tP)

    def at(f):
        pos = P + V * ((f - base) / fps)
        if R is not None:
            loc = np.einsum("nij,tcj->ntci", R, tP)
        else:
            loc = np.broadcast_to(tP, (n,) + tP.shape)
        Pp = loc * size[:, None, None, None] + pos[:, None, None, :]
        return Pp.reshape(n * T, 3, 3)

    if R is not None:
        Nn = np.einsum("nij,tcj->ntci", R, tN).reshape(n * T, 3, 3)
    else:
        Nn = np.broadcast_to(tN, (n,) + tN.shape).reshape(n * T, 3, 3)
    UV = np.broadcast_to(tri["UV"], (n,) + tri["UV"].shape).reshape(n * T, 3, 2)
    mat = np.broadcast_to(tri["mat"], (n, T)).reshape(-1)
    P0 = at(f0)
    t0 = {"P": P0, "N": np.ascontiguousarray(Nn), "UV": np.ascontiguousarray(UV), "mat": mat,
          "PREF": np.broadcast_to(tP, (n,) + tP.shape).reshape(n * T, 3, 3) * 1.0}
    t1 = None
    if f1 != f0:
        t1 = {"P": at(f1), "N": t0["N"]}
    return t0, t1
