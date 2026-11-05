"""Hair and fur: procedural strands grown on a mesh surface, groomed and optionally simulated.

A ``hair`` node grows ``count`` strands from the surface of a ``source`` mesh node. The roots are
area-weighted samples stored as (triangle, barycentric) pairs, so they stick to the surface
through any deformation of the source (skinning, blend shapes, cloth, modifiers). Each frame the
strands are styled (length, droop under gravity, comb direction, curl, frizz noise, clumping,
groom regions) and optionally run through a strand dynamics solver (Verlet with inextensible
segments, root-to-tip stiffness toward the styled shape, collisions). The result is emitted as
thin tapered tube (or ribbon) geometry with per-strand colour variation via ``uv``: ``u`` = strand
random value, ``v`` = 0 at the root, 1 at the tip (use a ``gradient`` texture on ``v`` for root to
tip colouring).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from . import mathutil as mu
from . import noise as nz

HAIR_KEYS = ("source", "count", "length", "length_variation", "segments", "width", "tip_width", "shape",
             "sides", "gravity", "comb", "comb_strength", "curl", "curl_frequency", "frizz", "frizz_scale",
             "clump", "clump_count", "clump_shape", "groom", "seed", "dynamics", "stiffness", "damping",
             "substeps", "colliders", "ground", "start_frame", "preview_fraction", "mask")


def _f(spec, key, default, frame=0.0):
    v = spec.get(key, default)
    if isinstance(v, (int, float)):
        return float(v)
    from .params import Param
    return float(Param(v).evaluate(frame))


class HairSystem:
    def __init__(self, scene, name):
        self.scene = sc = scene
        self.name = name
        node = sc.nodes[name]
        self.spec = sp = node.spec
        src = sp.get("source")
        if src not in sc.nodes:
            raise ValueError(f"hair {name!r}: 'source' must name a mesh node (got {src!r})")
        self.source = src
        self.seed = int(sp.get("seed", 0))
        self.f0 = int(math.floor(float(sp.get("start_frame", sc.frame_start))))
        rest = sc.world_mesh(src, self.f0)
        T, _ = rest.triangles()
        self.T = np.asarray(T, np.int64)
        V = rest.verts
        a = np.linalg.norm(np.cross(V[self.T[:, 1]] - V[self.T[:, 0]], V[self.T[:, 2]] - V[self.T[:, 0]]), axis=1)
        w = a * self._mask_weights(V)
        rng = np.random.default_rng(self.seed)
        n = int(sp.get("count", 2000))
        if w.sum() <= 0:
            raise ValueError(f"hair {name!r}: source has no area in the masked region")
        self.tri = rng.choice(len(self.T), n, p=w / w.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        self.bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
        self.rand = rng.random(n)                 # per-strand random (length, colour)
        self.phase = rng.random(n) * 2 * math.pi  # curl phase
        self.S = max(2, int(sp.get("segments", 8)))
        cc = int(sp.get("clump_count", max(1, n // 40)))
        self.clump_of = None
        if float(sp.get("clump", 0.0)) > 0 and cc > 0:
            roots0, _ = self._roots(V)
            centers = rng.choice(n, min(cc, n), replace=False)
            d = ((roots0[:, None, :] - roots0[None, centers, :]) ** 2).sum(-1) if n * len(centers) < 4e7 else None
            if d is None:
                self.clump_of = centers[np.zeros(n, int)]
            else:
                self.clump_of = centers[np.argmin(d, axis=1)]
        self.dyn = bool(sp.get("dynamics", False))
        self.cache = OrderedDict()
        self.sim_cache = {}
        self.sim_frame = None

    # ------------------------------------------------------------------ roots
    def _mask_weights(self, V):
        """Optional density mask: {"axis": "y", "min": .., "max": ..} or a list of regions with
        ``center``/``radius`` (strands only grow inside)."""
        m = self.spec.get("mask")
        T = self.T
        C = V[T].mean(1)
        if not m:
            return np.ones(len(T))
        regions = m if isinstance(m, list) else [m]
        w = np.zeros(len(T))
        for r in regions:
            if "center" in r:
                d = np.linalg.norm(C - np.asarray(r["center"], float), axis=1)
                w = np.maximum(w, (d <= float(r.get("radius", 1.0))).astype(float))
            else:
                ax = "xyz".index(r.get("axis", "y"))
                lo, hi = float(r.get("min", -1e30)), float(r.get("max", 1e30))
                w = np.maximum(w, ((C[:, ax] >= lo) & (C[:, ax] <= hi)).astype(float))
        if any(r.get("invert") for r in regions):
            w = 1 - w
        return w

    def _roots(self, V):
        T = self.T[self.tri]
        b = self.bary
        P = V[T[:, 0]] * b[:, :1] + V[T[:, 1]] * b[:, 1:2] + V[T[:, 2]] * b[:, 2:]
        N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        return P, N

    # ------------------------------------------------------------------ styling
    def styled(self, frame):
        """World-space styled strand points (N, S+1, 3) at ``frame`` (no dynamics)."""
        sc, sp = self.scene, self.spec
        m = sc.world_mesh(self.source, frame)
        R, N = self._roots(m.verts)
        n, S = len(R), self.S
        L = _f(sp, "length", 0.3, frame) * (1 - _f(sp, "length_variation", 0.3, frame) * self.rand)
        gravity = _f(sp, "gravity", 0.3, frame)
        comb = np.asarray(sp.get("comb", [0, 0, 0]), float)
        comb_s = _f(sp, "comb_strength", 0.0 if not np.any(comb) else 0.5, frame)
        curl = _f(sp, "curl", 0.0, frame)
        curl_f = _f(sp, "curl_frequency", 3.0, frame)
        frizz = _f(sp, "frizz", 0.0, frame)
        frizz_sc = _f(sp, "frizz_scale", 8.0, frame)
        # groom regions: local length scale / comb / gravity overrides
        Lscale = np.ones(n)
        comb_v = np.broadcast_to(comb, (n, 3)).copy()
        comb_w = np.full(n, comb_s)
        grav_w = np.full(n, gravity)
        for g in sp.get("groom", []) or []:
            c = np.asarray(g.get("center", [0, 0, 0]), float)
            rad = float(g.get("radius", 1.0))
            d = np.linalg.norm(R - c, axis=1)
            wt = np.clip(1 - d / max(rad, 1e-9), 0, 1)
            wt = wt * wt * (3 - 2 * wt) * float(g.get("weight", 1.0))
            if "length" in g:
                Lscale *= 1 + wt * (float(g["length"]) - 1)
            if "comb" in g:
                comb_v = comb_v * (1 - wt[:, None]) + np.asarray(g["comb"], float) * wt[:, None]
                comb_w = comb_w * (1 - wt) + float(g.get("comb_strength", 0.8)) * wt
            if "gravity" in g:
                grav_w = grav_w * (1 - wt) + float(g["gravity"]) * wt
        L = L * Lscale
        seg = (L / S)[:, None]
        gdir = np.array([0.0, -1.0, 0.0])
        # orthonormal frame around the root normal for curl
        ref = np.where(np.abs(N[:, 1:2]) < 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
        B1 = np.cross(N, ref)
        B1 /= np.maximum(np.linalg.norm(B1, axis=1, keepdims=True), 1e-12)
        B2 = np.cross(N, B1)
        P = np.empty((n, S + 1, 3))
        P[:, 0] = R
        cn = np.linalg.norm(comb_v, axis=1, keepdims=True)
        comb_dir = np.where(cn > 1e-9, comb_v / np.maximum(cn, 1e-9), 0.0)
        # comb direction projected onto the surface tangent plane
        comb_dir = comb_dir - N * np.sum(comb_dir * N, 1, keepdims=True)
        comb_dir /= np.maximum(np.linalg.norm(comb_dir, axis=1, keepdims=True), 1e-9)
        d0 = N * (1 - comb_w[:, None]) + comb_dir * comb_w[:, None]
        d0 /= np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-9)
        for k in range(1, S + 1):
            t = k / S
            d = d0 + gdir * grav_w[:, None] * t * 2.0
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            P[:, k] = P[:, k - 1] + d * seg
        if curl > 0:
            t = np.arange(S + 1) / S
            ang = self.phase[:, None] + t[None] * curl_f * 2 * math.pi
            amp = curl * L[:, None] * 0.15 * np.minimum(t[None] * 4, 1)
            P += (np.cos(ang) * amp)[..., None] * B1[:, None] + (np.sin(ang) * amp)[..., None] * B2[:, None]
        if frizz > 0:
            t = (np.arange(S + 1) / S)[None, :, None]
            q = P * frizz_sc + self.rand[:, None, None] * 17.0
            P += nz.noise_vec3(q, self.seed) * frizz * L[:, None, None] * 0.3 * t
        clump = _f(sp, "clump", 0.0, frame)
        if clump > 0 and self.clump_of is not None:
            shape = _f(sp, "clump_shape", 1.0, frame)
            t = (np.arange(S + 1) / S) ** max(shape, 1e-3)
            Ck = P[self.clump_of]
            # move the clump guide strand to each follower's root so the clump keeps its spread
            w = (clump * t)[None, :, None]
            P = P * (1 - w) + Ck * w
        return P

    # ------------------------------------------------------------------ dynamics
    def _colliders(self, frame):
        from .dynamics import _collider_shape, _world_shape
        sc = self.scene
        out = []
        for c in self.spec.get("colliders", []) or []:
            spec = c if isinstance(c, dict) else {"node": c}
            out.append(_world_shape(_collider_shape(sc, spec["node"], spec, frame), sc.state(frame).world(spec["node"])))
        from .dynamics import DEFAULT_DYNAMICS
        cfg = dict(DEFAULT_DYNAMICS)
        cfg.update(sc.data.get("dynamics", {}) or {})
        if bool(self.spec.get("ground", cfg["ground"])):
            out.append({"kind": "ground", "y": float(cfg["ground_height"])})
        return out

    def simulated(self, frame):
        fi = int(math.floor(frame))
        if fi <= self.f0:
            return self.styled(frame)
        if fi in self.sim_cache and (frame == fi or fi + 1 in self.sim_cache):
            a = self.sim_cache[fi]
            if frame > fi:
                t = frame - fi
                return a * (1 - t) + self.sim_cache[fi + 1] * t
            return a
        need = fi + (1 if frame > fi else 0)
        sp = self.spec
        sub = max(1, int(sp.get("substeps", 4)))
        stiff = float(sp.get("stiffness", 0.15))
        damp = float(sp.get("damping", 0.05))
        fps = self.scene.fps
        dt = 1.0 / fps / sub
        g = np.array([0, -9.81, 0])
        if self.sim_frame is None:
            X = self.styled(self.f0)
            self.state = (X.copy(), X.copy())
            self.sim_frame = self.f0
            self.sim_cache[self.f0] = X.copy()
        X, Xp = self.state
        S = self.S
        # stiffness profile: strong at the root, soft at the tip
        prof = (stiff * (1 - np.arange(S + 1) / S) ** 2 + stiff * 0.1)[None, :, None]
        while self.sim_frame < need:
            f_a = self.sim_frame
            Ta, Tb = self.styled(f_a), self.styled(f_a + 1)
            rest = np.linalg.norm(np.diff(Tb, axis=1), axis=2)
            shapes = self._colliders(f_a + 1)
            for s in range(sub):
                u = (s + 1) / sub
                tgt = Ta * (1 - u) + Tb * u
                V = (X - Xp) * (1 - damp)
                Xp = X
                X = X + V + g * dt * dt
                X[:, 0] = tgt[:, 0]
                for _ in range(3):
                    X += (tgt - X) * prof
                    X[:, 0] = tgt[:, 0]
                    # inextensible segments, root to tip
                    for k in range(1, S + 1):
                        d = X[:, k] - X[:, k - 1]
                        L = np.linalg.norm(d, axis=1, keepdims=True)
                        X[:, k] = X[:, k - 1] + d / np.maximum(L, 1e-12) * rest[:, k - 1:k]
                    for shp in shapes:
                        flat = X[:, 1:].reshape(-1, 3)
                        from .dynamics import _collide_points
                        flat, _, _ = _collide_points(flat, shp, 0.002)
                        X[:, 1:] = flat.reshape(len(X), S, 3)
            self.sim_frame += 1
            self.sim_cache[self.sim_frame] = X.copy()
        self.state = (X, Xp)
        return self.simulated(frame)

    # ------------------------------------------------------------------ geometry
    def mesh(self, frame, preview=False):
        from .geometry import Mesh
        key = (round(float(frame), 4), preview)
        if key in self.cache:
            return self.cache[key]
        P = self.simulated(frame) if self.dyn else self.styled(frame)
        sp = self.spec
        if preview:
            frac = float(sp.get("preview_fraction", 0.25))
            keep = self.rand < frac
            P, rnd = P[keep], self.rand[keep]
        else:
            rnd = self.rand
        m = strands_to_mesh(P, rnd, width=_f(sp, "width", 0.004, frame), tip_width=_f(sp, "tip_width", 0.2, frame),
                            shape=sp.get("shape", "tube"), sides=int(sp.get("sides", 3)), name=self.name)
        W = self.scene.state(frame).world(self.name)
        m = m.transformed(np.linalg.inv(W))
        self.cache[key] = m
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return m


def strands_to_mesh(P, rnd, width=0.004, tip_width=0.2, shape="tube", sides=3, name="hair"):
    """Turn strands ``P`` (N, K, 3) into tapered tubes (``sides`` >= 3) or flat ribbons.

    ``width`` is the root diameter (world units); ``tip_width`` the tip width as a fraction of it.
    UVs: u = per-strand random value, v = 0 root .. 1 tip."""
    from .geometry import Mesh
    n, K, _ = P.shape
    if n == 0:
        return Mesh(np.zeros((0, 3)), [], [], name=name)
    T = np.gradient(P, axis=1)
    T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-12)
    ref = np.where(np.abs(T[..., 1:2]) < 0.9, np.array([0, 1.0, 0]), np.array([1.0, 0, 0]))
    B1 = np.cross(T, ref)
    B1 /= np.maximum(np.linalg.norm(B1, axis=2, keepdims=True), 1e-12)
    B2 = np.cross(T, B1)
    t = np.linspace(0, 1, K)
    rad = 0.5 * width * (1 - (1 - tip_width) * t)[None, :, None]
    if shape == "ribbon":
        ring = [B1 * rad, -B1 * rad]
        ns = 2
    else:
        ns = max(3, int(sides))
        ring = [(math.cos(2 * math.pi * j / ns) * B1 + math.sin(2 * math.pi * j / ns) * B2) * rad for j in range(ns)]
    V = np.stack([P + r for r in ring], axis=2)  # (n, K, ns, 3)
    verts = V.reshape(-1, 3)
    base = (np.arange(n) * K * ns)[:, None, None]
    k = np.arange(K - 1)[None, :, None]
    jn = ns if shape != "ribbon" else 1
    j = np.arange(jn)[None, None, :]
    j2 = (j + 1) % ns
    a = base + k * ns + j
    b = base + k * ns + j2
    c = base + (k + 1) * ns + j2
    d = base + (k + 1) * ns + j
    quads = np.stack([a, b, c, d], -1).reshape(-1, 4)
    counts = np.full(len(quads), 4)
    # per-corner uvs
    kk = np.broadcast_to(k, a.shape)
    vv = np.stack([kk, kk, kk + 1, kk + 1], -1).reshape(-1, 4) / (K - 1)
    uu = np.broadcast_to(rnd[:, None, None, None], a.shape + (4,)).reshape(-1, 4)
    uvs = np.stack([uu, vv], -1).reshape(-1, 2)
    return Mesh(verts, counts, quads.ravel(), uvs=uvs, smooth_angle=180.0, name=name)


def get_hair(scene, name):
    d = scene.__dict__.setdefault("_hair_systems", {})
    if name not in d:
        d[name] = HairSystem(scene, name)
    return d[name]


def hair_mesh(scene, name, frame, preview=False):
    """Object-space strand geometry of hair node ``name`` at ``frame``."""
    return get_hair(scene, name).mesh(frame, preview)


def hair_curves(scene, name, frame):
    """World-space strand control points (N, segments+1, 3) at ``frame`` (styled + simulated)."""
    h = get_hair(scene, name)
    return h.simulated(frame) if h.dyn else h.styled(frame)
