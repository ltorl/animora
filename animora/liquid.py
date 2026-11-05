"""FLIP-quality liquids with Position-Based Fluids (``type: "liquid"`` nodes).

A liquid node is a simulation domain box (``size``, centred on the node origin, oriented by its
world matrix at the start frame). Particles are emitted by ``emitters``, advanced by the native
PBF solver (Macklin & Mueller 2013: incompressibility via density constraints, tensile-instability
correction, XSPH viscosity, vorticity confinement), collide with the domain walls and with any
mesh nodes listed in ``colliders`` (exact signed-distance grids, recomputed when they move), and
are meshed every frame into a smooth watertight surface (kernel splatting + surface nets with
sub-voxel projection) which renders like any mesh with the node's ``material``.

    {"name": "water", "type": "liquid", "size": [4, 2.5, 1.6], "translate": [0, 1.25, 0],
     "spacing": 0.04, "material": "water",
     "emitters": [{"type": "fill", "shape": "box", "center": [-1.4, -0.5, 0], "size": [1.2, 1.4, 1.6]},
                  {"type": "stream", "center": [1.5, 0.8, 0], "radius": 0.12,
                   "velocity": [-2.5, 0, 0], "start": 10, "end": 60}],
     "colliders": ["rock"], "cache": "water.npz"}

Positions in emitters are in the domain's local space (origin = node origin).
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

import numpy as np

from . import mathutil as mu
from .params import Param

LIQUID_KEYS = ("size", "spacing", "substeps", "iterations", "viscosity", "vorticity", "gravity", "max_particles",
               "emitters", "colliders", "cache", "start_frame", "open_top", "friction", "damping", "seed",
               "cfl", "max_speed", "max_correction", "surface_radius", "surface_resolution", "iso", "smooth_iterations", "tension",
               "smooth_angle")


def _kernel_lattice_sum(spacing, h, kernel):
    r = int(np.ceil(h / spacing)) + 1
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1].reshape(3, -1).T * spacing
    return float(kernel((g ** 2).sum(1)).sum())


class LiquidSim:
    def __init__(self, scene, name):
        self.scene = scene
        self.name = name
        node = scene.nodes[name]
        self.node = node
        sp = node.spec
        self.spec = sp
        f0 = scene.frame_start
        size = np.asarray(node.p("size", [2.0, 2.0, 2.0], f0, None), float)
        if size.ndim == 0:
            size = np.full(3, float(size))
        self.size = size
        self.lo, self.hi = -size / 2, size / 2
        self.s = float(sp.get("spacing", 0.05))
        self.h = 2.0 * self.s
        h2 = self.h ** 2
        poly6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.rho0 = _kernel_lattice_sum(self.s, self.h, lambda r2: np.where(r2 < h2, poly6 * (h2 - r2) ** 3, 0.0))
        self.iterations = int(sp.get("iterations", 4))
        self.max_particles = int(sp.get("max_particles", 300000))
        self.start = float(sp.get("start_frame", f0))
        self.rng = np.random.RandomState(int(sp.get("seed", 7)))
        self.params = {k: Param(sp[k]) for k in ("viscosity", "vorticity", "gravity", "friction", "damping")
                       if k in sp}
        self.emitters = []
        for e in sp.get("emitters", []) or []:
            e = dict(e)
            self.emitters.append({k: (Param(v) if k in ("center", "velocity", "radius", "size", "rate") else v)
                                  for k, v in e.items()})
            self.emitters[-1]["_done"] = False
            self.emitters[-1]["_acc"] = 0.0
        self.colliders = list(sp.get("colliders", []) or [])
        self.cache = sp.get("cache")
        if self.cache and not os.path.isabs(self.cache):
            self.cache = os.path.join(scene.base_dir or ".", self.cache)
        W = scene.world_matrix(name, f0)
        self.W0 = W
        self.inv0 = np.linalg.inv(W)
        self.pos = np.zeros((0, 3), np.float32)
        self.vel = np.zeros((0, 3), np.float32)
        self.frame = int(np.floor(self.start)) - 1
        self.frames = {}
        self._sdf = None
        self._sdf_key = None
        self._mesh_cache = OrderedDict()
        if self.cache and os.path.exists(self.cache):
            self._load_cache()

    # ------------------------------------------------------------------ helpers
    def _pv(self, key, default, frame):
        p = self.params.get(key)
        return default if p is None else p.evaluate(frame, None)

    def _ev(self, e, key, default, frame):
        v = e.get(key)
        if v is None:
            return default
        return v.evaluate(frame, None) if isinstance(v, Param) else v

    def _lattice(self, lo, hi, jitter=0.08):
        n = np.maximum(1, np.floor((np.asarray(hi) - np.asarray(lo)) / self.s).astype(int))
        g = np.mgrid[0:n[0], 0:n[1], 0:n[2]].reshape(3, -1).T.astype(np.float64)
        p = lo + (g + 0.5) * self.s
        p += self.rng.uniform(-jitter, jitter, p.shape) * self.s
        return p

    def _add(self, p, v):
        p = np.asarray(p, np.float32).reshape(-1, 3)
        if not len(p):
            return
        r = self.s * 0.5
        keep = np.all((p > self.lo + r) & (p < self.hi - r), axis=1)
        p = p[keep]
        room = self.max_particles - len(self.pos)
        if room <= 0:
            return
        p = p[:room]
        v = np.broadcast_to(np.asarray(v, np.float32), p.shape)
        self.pos = np.ascontiguousarray(np.concatenate([self.pos, p]), np.float32)
        self.vel = np.ascontiguousarray(np.concatenate([self.vel, v]), np.float32)

    def _emit(self, frame, dt):
        for e in self.emitters:
            kind = str(e.get("type", "fill")).lower()
            start = float(e.get("start", e.get("frame", self.start)))
            end = float(e.get("end", start if kind == "fill" else 1e9))
            if frame < start or frame > end + 1e-6:
                continue
            vel = np.asarray(self._ev(e, "velocity", [0.0, 0.0, 0.0], frame), float)
            c = np.asarray(self._ev(e, "center", [0.0, 0.0, 0.0], frame), float)
            if kind == "fill":
                if e["_done"]:
                    continue
                e["_done"] = True
                shape = str(e.get("shape", "box")).lower()
                if shape == "mesh" or "object" in e:
                    pts = self._fill_mesh(e.get("object"), frame)
                else:
                    sz = np.asarray(self._ev(e, "size", 1.0, frame), float)
                    sz = np.broadcast_to(sz, (3,)) if sz.ndim == 0 else sz
                    if shape in ("sphere", "ball"):
                        rad = float(self._ev(e, "radius", float(sz.max()) / 2, frame))
                        pts = self._lattice(c - rad, c + rad)
                        pts = pts[np.linalg.norm(pts - c, axis=1) <= rad]
                    else:
                        pts = self._lattice(c - sz / 2, c + sz / 2)
                self._add(pts, vel)
            elif kind in ("stream", "nozzle", "inflow"):
                rad = float(self._ev(e, "radius", 0.1, frame))
                speed = float(np.linalg.norm(vel))
                if speed < 1e-6:
                    continue
                d = vel / speed
                e["_acc"] += speed * dt
                # disc lattice perpendicular to the stream direction
                a = mu.normalize(np.cross(d, [0, 1, 0] if abs(d[1]) < 0.9 else [1, 0, 0]))
                b = np.cross(d, a)
                k = int(np.ceil(rad / self.s))
                u, w = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
                uv = np.stack([u.ravel(), w.ravel()], -1) * self.s
                uv = uv[np.linalg.norm(uv, axis=1) <= rad]
                while e["_acc"] >= self.s:
                    e["_acc"] -= self.s
                    jit = self.rng.uniform(-0.1, 0.1, (len(uv), 2)) * self.s
                    q = uv + jit
                    pts = c + q[:, :1] * a + q[:, 1:] * b + d * e["_acc"]
                    self._add(pts, vel)

    def _fill_mesh(self, obj, frame):
        from .render import native as nat
        if obj not in self.scene.nodes:
            raise ValueError(f"liquid {self.name!r}: fill emitter object {obj!r} not found")
        m = self.scene.world_mesh(obj, frame)
        V = mu.transform_points(self.inv0, m.verts)
        T, _ = m.triangles()
        lo, hi = V.min(0), V.max(0)
        pts = self._lattice(lo, hi)
        d = nat.mesh_sdf(V, T, pts)
        return pts[d < -self.s * 0.25]

    def _collider_sdf(self, frame):
        if not self.colliders:
            return None
        from .render import native as nat
        Vs, Ts, off = [], [], 0
        for cn in self.colliders:
            if cn not in self.scene.nodes:
                continue
            m = self.scene.world_mesh(cn, frame)
            if m is None or not m.num_verts:
                continue
            V = mu.transform_points(self.inv0, m.verts)
            T, _ = m.triangles()
            Vs.append(V)
            Ts.append(T + off)
            off += len(V)
        if not Vs:
            return None
        V = np.concatenate(Vs).astype(np.float32)
        T = np.concatenate(Ts).astype(np.int32)
        key = hashlib.blake2b(V.tobytes() + T.tobytes(), digest_size=12).hexdigest()
        if key == self._sdf_key:
            return self._sdf
        cell = max(self.s, float(self.size.max()) / 160.0)
        lo = self.lo - cell
        n = np.ceil((self.size + 2 * cell) / cell).astype(int) + 1
        xs, ys, zs = [lo[a] + np.arange(n[a]) * cell for a in range(3)]
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
        d = nat.mesh_sdf(V, T, pts).reshape(Z.shape)
        self._sdf = (np.ascontiguousarray(d, np.float32), lo, cell)
        self._sdf_key = key
        return self._sdf

    def _params(self, dt, frame):
        g = np.asarray(self._pv("gravity", [0.0, -9.81, 0.0], frame), float)
        if g.ndim == 0:
            g = np.array([0.0, float(g), 0.0])
        g = self.inv0[:3, :3] @ g  # world -> domain local (ignoring scale skew)
        visc = float(self._pv("viscosity", 0.01, frame))
        vort = float(self._pv("vorticity", 0.0, frame))
        fric = float(self._pv("friction", 0.1, frame))
        damp = float(self._pv("damping", 0.0, frame))
        return np.array([dt, self.h, self.rho0, self.iterations, g[0], g[1], g[2], *self.lo, *self.hi,
                         visc, vort, float(self.spec.get("tension", 0.001)) * 1.0, self.s * 0.5, 100.0, fric,
                         1.0 if self.spec.get("open_top", False) else 0.0, damp,
                         float(self.spec.get("max_speed", self._auto_vmax(g))),
                         float(self.spec.get("max_correction", 0.25))], np.float32)

    def _auto_vmax(self, g):
        fall = np.sqrt(2.0 * max(float(np.linalg.norm(g)), 1e-6) * float(self.size.max()))
        em = 0.0
        for e in self.emitters:
            v = e.get("velocity")
            if v is not None:
                try:
                    em = max(em, float(np.linalg.norm(v.evaluate(self.frame, None))))
                except Exception:  # noqa: BLE001
                    pass
        return 1.5 * (fall + em)

    def _step(self, dt, frame):
        from .render import native as nat
        self._emit(frame, dt)
        if not len(self.pos):
            return
        sdf = self._collider_sdf(frame)
        prm = self._params(dt, frame)
        if sdf is None:
            nat.pbf_step(self.pos, self.vel, prm)
        else:
            nat.pbf_step(self.pos, self.vel, prm, sdf[0], sdf[1], sdf[2])

    # ------------------------------------------------------------------ public
    def advance_to(self, f):
        """Simulate up to integer frame ``f``.

        Sub-steps are adaptive: at least ``substeps`` per frame (default: a 1/240 s step) and more
        whenever the fastest particle would travel over ``cfl`` (0.3) kernel radii in one step."""
        f = int(np.floor(f))
        fps = float(self.scene.fps)
        cfl = float(self.spec.get("cfl", 0.3))
        min_sub = int(self.spec.get("substeps", max(1, int(np.ceil(240.0 / fps)))))
        while self.frame < f:
            t, end = 0.0, 1.0 / fps
            while t < end - 1e-9:
                vmax = float(np.sqrt((self.vel ** 2).sum(1).max())) if len(self.vel) else 0.0
                dt = min(end / min_sub, cfl * self.h / max(vmax, 1e-6), end - t)
                dt = max(dt, end / 64.0)
                if end - t - dt < 1e-6 * end:
                    dt = end - t
                t += dt
                self._step(dt, self.frame + t * fps)
            self.frame += 1
            self.frames[self.frame] = self.pos.copy()

    def at(self, frame):
        """Particle positions (N,3) in domain-local space at integer ``frame``."""
        fi = int(np.floor(frame))
        if fi < self.start:
            return np.zeros((0, 3), np.float32)
        if fi not in self.frames:
            if fi < self.frame:
                self.__init__(self.scene, self.name)
            self.advance_to(fi)
        return self.frames[fi]

    def save(self, path):
        ks = sorted(self.frames)
        counts = np.array([len(self.frames[k]) for k in ks], np.int64)
        allp = np.concatenate([self.frames[k] for k in ks]) if ks else np.zeros((0, 3), np.float32)
        np.savez_compressed(path, frames=np.asarray(ks, np.int32), counts=counts, pos=allp.astype(np.float32),
                            vel=self.vel.astype(np.float32), last=np.int32(self.frame),
                            emit=np.array([[e["_done"], e["_acc"]] for e in self.emitters], np.float64)
                            .reshape(-1, 2), spacing=np.float32(self.s))

    def _load_cache(self):
        z = np.load(self.cache)
        if "spacing" in z and abs(float(z["spacing"]) - self.s) > 1e-9:
            return  # cache made with different settings: ignore
        off = 0
        for k, c in zip(z["frames"], z["counts"]):
            self.frames[int(k)] = z["pos"][off:off + int(c)]
            off += int(c)
        if len(z["frames"]):
            self.frame = int(z["frames"][-1])
            self.pos = np.ascontiguousarray(self.frames[self.frame], np.float32)
            if int(z["last"]) == self.frame and len(z["vel"]) == len(self.pos):
                self.vel = np.ascontiguousarray(z["vel"], np.float32)
            else:
                self.vel = np.zeros_like(self.pos)
            for e, (done, acc) in zip(self.emitters, z["emit"]):
                e["_done"], e["_acc"] = bool(done), float(acc)

    # ------------------------------------------------------------------ surface
    def surface(self, frame, preview=False):
        """Smooth watertight surface Mesh (domain-local space) at ``frame``."""
        from .geometry import Mesh
        from .modeling import surface_nets
        fi = int(np.floor(frame))
        key = (fi, preview)
        if key in self._mesh_cache:
            self._mesh_cache.move_to_end(key)
            return self._mesh_cache[key]
        P = self.at(fi)
        sp = self.spec
        if len(P) == 0:
            m = Mesh(np.zeros((0, 3)), [], [], name=self.name)
        else:
            m = surface_from_particles(P, self.s, radius=float(sp.get("surface_radius", 2.5)),
                                       resolution=float(sp.get("surface_resolution", 1.0 if preview else 2.0)),
                                       iso=float(sp.get("iso", 0.5)),
                                       smooth=int(sp.get("smooth_iterations", 3)),
                                       smooth_angle=float(sp.get("smooth_angle", 180.0)), name=self.name,
                                       lo=self.lo, hi=self.hi)
        self._mesh_cache[key] = m
        while len(self._mesh_cache) > 4:
            self._mesh_cache.popitem(last=False)
        return m


def _blur(F, iterations):
    for _ in range(iterations):
        for ax in range(3):
            a = np.pad(F, [(1, 1) if i == ax else (0, 0) for i in range(3)], mode="edge")
            sl = lambda s, e: tuple(slice(s, e if e else None) if i == ax else slice(None) for i in range(3))  # noqa
            F = 0.25 * a[sl(0, -2)] + 0.5 * a[sl(1, -1)] + 0.25 * a[sl(2, None)]
    return F


def surface_from_particles(P, spacing, radius=2.0, resolution=2.0, iso=0.5, smooth=1, smooth_angle=180.0,
                           name="liquid", lo=None, hi=None):
    """Mesh the liquid surface of particles ``P`` (N,3) sampled at ``spacing``.

    ``radius``: kernel radius in particle spacings; ``resolution``: voxels per spacing;
    ``iso``: fraction of the bulk (fully-filled) density used as the surface level."""
    from .geometry import Mesh
    from .modeling import surface_nets
    from .render import native as nat
    R = radius * spacing
    cell = spacing / max(0.25, resolution)
    plo = P.min(0) - R - 2 * cell
    phi = P.max(0) + R + 2 * cell
    if lo is not None:
        plo = np.maximum(plo, np.asarray(lo) - 2 * cell)
        phi = np.minimum(phi, np.asarray(hi) + 2 * cell)
    dims = np.ceil((phi - plo) / cell).astype(int) + 1
    while np.prod(dims) > 40e6:  # keep memory bounded
        cell *= 1.25
        dims = np.ceil((phi - plo) / cell).astype(int) + 1
    G = nat.splat(P, dims, plo, cell, R)
    bulk = _kernel_lattice_sum(spacing, R, lambda r2: np.where(r2 < R * R, (1 - r2 / (R * R)) ** 3, 0.0))
    G = _blur(G, smooth) if smooth > 0 else G
    F = (iso * bulk - G).transpose(2, 1, 0)  # [x, y, z], negative inside

    def sampler(pts):
        q = (pts - plo) / cell
        q = np.clip(q, 0, np.array(F.shape) - 1.001)
        i = np.floor(q).astype(int)
        t = q - i
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wgt = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * \
                          (t[:, 2] if dz else 1 - t[:, 2])
                    out = out + wgt * F[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return out / (bulk / R)  # roughly distance-scaled for the Newton projection

    Fs = F / (bulk / R)
    m = surface_nets(Fs, plo, cell, func=sampler, project_iters=2, smooth_angle=smooth_angle, name=name)
    if m is None:
        return Mesh(np.zeros((0, 3)), [], [], name=name)
    return m


def get_sim(scene, name):
    cache = scene.__dict__.setdefault("_liquid_sims", {})
    if name not in cache:
        cache[name] = LiquidSim(scene, name)
    return cache[name]


def liquid_mesh(scene, name, frame, preview=False):
    """Object-space surface mesh of liquid node ``name`` at ``frame``."""
    return get_sim(scene, name).surface(frame, preview)


def liquid_particles(scene, name, frame):
    """World-space particle positions of liquid node ``name``."""
    sim = get_sim(scene, name)
    return mu.transform_points(sim.W0, sim.at(frame))
