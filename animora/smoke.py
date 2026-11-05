"""Smoke and fire solver (stable-fluids / Bridson style), feeding ``volume`` nodes.

A ``volume`` node with a ``smoke`` block simulates on a voxel grid filling the node's box::

    {"name": "fire", "type": "volume", "size": [2, 4, 2], "translate": [0, 2, 0],
     "density": 8, "albedo": 0.3, "blackbody": 8, "temperature": [700, 2400],
     "smoke": {"resolution": 48, "buoyancy": 4, "cooling": 1.5, "dissipation": 0.2,
               "vorticity": 2, "turbulence": 1,
               "sources": [{"shape": "sphere", "center": [0, -1.7, 0], "radius": 0.35,
                            "density": 2, "heat": 1, "velocity": [0, 1, 0]}]}}

Per substep: emit → buoyancy (heat lifts, soot weighs down) + wind + turbulence + vorticity
confinement → semi-Lagrangian velocity advection → pressure projection (Jacobi, open
boundaries) → MacCormack (BFECC-limited) advection of density and heat → cooling/dissipation.

Sources: ``shape`` sphere / box, ``center`` / ``radius`` / ``size`` in the node's local units
(relative to the box centre), animatable ``density`` / ``heat`` / ``velocity`` / ``center``,
optional ``follow``: node name (centre tracks that node's world position), ``start`` / ``end``
frame window, ``noise`` (0..1 breaks up the emission mask for flickering flames).
Obstacles: ``colliders``: list of node names or dicts ``{"node": name, "shape": "mesh" |
"sphere" | "box", "thickness": t}``. ``mesh`` (default) voxelises the node's actual animated /
deforming geometry through a signed-distance field (winding-number sign, so open meshes work),
and the obstacle's per-vertex velocity is interpolated onto the solid cells so moving or
skinned colliders push smoke (moving-wall boundary condition in the pressure solve). Open
surfaces get a shell ``thickness`` (default half a voxel). ``sphere`` / ``box`` use the bounding
sphere / box. ``floor`` (default true) makes the bottom face of the domain a wall.

Performance: with the native library (default) advection, vorticity confinement and the
pressure solve (red-black SOR, ``pressure_iterations`` default 60, ``sor`` relaxation 1.7) run
multithreaded in C++, ~20-60x faster than the numpy fallback (``"native": false``).
``cache``: path to an ``.npz`` written after simulation and re-used when it exists
(``animora bake`` also writes it).
"""
from __future__ import annotations

import os

import numpy as np

from . import mathutil as mu
from .params import Param

try:
    from .render import native as _nat
except Exception:  # pragma: no cover
    _nat = None


def _native_ok():
    try:
        return _nat is not None and _nat.available()
    except Exception:
        return False


def _sample(f, x, y, z):
    """Trilinear sample of f[z,y,x] at cell-centre coordinates (clamped)."""
    nz, ny, nx = f.shape
    x = np.clip(x, 0, nx - 1.001)
    y = np.clip(y, 0, ny - 1.001)
    z = np.clip(z, 0, nz - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    z0 = z.astype(np.int32)
    fx, fy, fz = x - x0, y - y0, z - z0
    flat = f.ravel()
    base = (z0 * ny + y0) * nx + x0
    sy, sz = nx, nx * ny

    def g(o):
        return flat[base + o]
    c00 = g(0) * (1 - fx) + g(1) * fx
    c10 = g(sy) * (1 - fx) + g(sy + 1) * fx
    c01 = g(sz) * (1 - fx) + g(sz + 1) * fx
    c11 = g(sz + sy) * (1 - fx) + g(sz + sy + 1) * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


def _sample_minmax(f, x, y, z):
    nz, ny, nx = f.shape
    x0 = np.clip(np.floor(x).astype(np.int32), 0, nx - 2)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, ny - 2)
    z0 = np.clip(np.floor(z).astype(np.int32), 0, nz - 2)
    flat = f.ravel()
    base = (z0 * ny + y0) * nx + x0
    vals = [flat[base + o] for o in (0, 1, nx, nx + 1, nx * ny, nx * ny + 1, nx * ny + nx, nx * ny + nx + 1)]
    return np.minimum.reduce(vals), np.maximum.reduce(vals)


def _grad(f):
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gz = np.zeros_like(f)
    gx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) * 0.5
    gy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) * 0.5
    gz[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) * 0.5
    return gx, gy, gz


class SmokeSim:
    def __init__(self, scene, name):
        self.scene = scene
        self.name = name
        node = scene.nodes[name]
        self.node = node
        spec = dict(node.spec.get("smoke") or {})
        self.spec = spec
        size = np.asarray(node.p("size", [2.0, 2.0, 2.0], scene.frame_start, None), float)
        if size.ndim == 0:
            size = np.full(3, float(size))
        self.size = size
        res = int(spec.get("resolution", 40))
        self.h = float(size.max()) / res
        self.nx, self.ny, self.nz = [max(4, int(round(s / self.h))) for s in size]
        shp = (self.nz, self.ny, self.nx)
        self.d = np.zeros(shp, np.float32)
        self.T = np.zeros(shp, np.float32)
        self.u = np.zeros(shp, np.float32)
        self.v = np.zeros(shp, np.float32)
        self.w = np.zeros(shp, np.float32)
        self.p = np.zeros(shp, np.float32)
        zz, yy, xx = np.meshgrid(np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij")
        self.cx = xx.astype(np.float32)
        self.cy = yy.astype(np.float32)
        self.cz = zz.astype(np.float32)
        # local position (box centred at origin) of every cell centre
        self.lx = (self.cx + 0.5) * self.h - size[0] / 2
        self.ly = (self.cy + 0.5) * self.h - size[1] / 2
        self.lz = (self.cz + 0.5) * self.h - size[2] / 2
        self.substeps = int(spec.get("substeps", 1))
        self.start = float(spec.get("start_frame", scene.frame_start))
        self.frame = self.start - 1
        self.frames = {}
        self.rng = np.random.RandomState(int(spec.get("seed", 1)))
        self.sources = []
        for s in spec.get("sources", [{}]):
            self.sources.append({k: (Param(v) if k not in ("shape", "follow") else v) for k, v in s.items()})
        self.params = {k: Param(v) for k, v in spec.items()
                       if k in ("buoyancy", "weight", "cooling", "dissipation", "vorticity", "turbulence", "wind",
                                "turbulence_scale", "max_heat")}
        self.colliders = [c if isinstance(c, dict) else {"node": c} for c in spec.get("colliders", [])]
        self.native = bool(spec.get("native", True)) and _native_ok()
        self.floor = bool(spec.get("floor", True))
        self.solid = None
        self.svel = None
        self._col_key = None
        self.svel_g = None
        self.cache = spec.get("cache")
        if self.cache and not os.path.isabs(self.cache):
            self.cache = os.path.join(scene.base_dir or ".", self.cache)
        self._turb = None
        if self.cache and os.path.exists(self.cache):
            self._load_cache()

    # ------------------------------------------------------------------ helpers
    def _pv(self, key, default, frame):
        p = self.params.get(key)
        return default if p is None else p.evaluate(frame, None)

    def _world_to_local(self, frame):
        W = self.scene.world_matrix(self.name, frame)
        return np.linalg.inv(W)

    def _emit(self, dt, frame):
        inv = None
        for s in self.sources:
            ev = lambda k, d: (s[k].evaluate(frame, None) if k in s else d)  # noqa: E731
            if frame < float(ev("start", -1e9)) or frame > float(ev("end", 1e9)):
                continue
            c = np.asarray(ev("center", [0.0, -self.size[1] / 2 + 0.25 * self.size[1] * 0.4, 0.0]), float)
            if s.get("follow"):
                if inv is None:
                    inv = self._world_to_local(frame)
                pw = self.scene.world_matrix(s["follow"], frame)[:3, 3]
                c = mu.transform_points(inv, pw)
            shape = s.get("shape", "sphere")
            if shape == "box":
                hs = np.asarray(ev("size", [0.5, 0.2, 0.5]), float) / 2
                q = np.maximum.reduce([np.abs(self.lx - c[0]) / hs[0], np.abs(self.ly - c[1]) / hs[1],
                                       np.abs(self.lz - c[2]) / hs[2]])
            else:
                r = float(ev("radius", 0.15 * self.size.max()))
                q = np.sqrt((self.lx - c[0]) ** 2 + (self.ly - c[1]) ** 2 + (self.lz - c[2]) ** 2) / max(r, 1e-6)
            m = np.clip(1.0 - q, 0, 1) * 3.0
            m = np.clip(m, 0, 1).astype(np.float32)
            nz = float(ev("noise", 0.0))
            if nz > 0:
                m *= np.clip(1 - nz + nz * 2 * self.rng.rand(*m.shape), 0, 1).astype(np.float32)
            if not m.any():
                continue
            dens = float(ev("density", 1.0))
            heat = float(ev("heat", 1.0))
            vel = np.asarray(ev("velocity", [0.0, 1.0, 0.0]), float)
            self.d = np.maximum(self.d, m * dens)
            self.T = np.maximum(self.T, m * heat)
            sel = m > 0
            for comp, val in zip((self.u, self.v, self.w), vel):
                comp[sel] = comp[sel] * (1 - m[sel]) + val * m[sel]

    def _turbulence(self, frame, amount):
        # smooth random vector field, refreshed every frame, interpolated between frames
        scale = float(self._pv("turbulence_scale", 0.5, frame))
        n = max(3, int(round(self.size.max() / max(scale, 1e-3))) + 2)
        key = int(np.floor(frame))
        if self._turb is None or self._turb[0] != key:
            rs = np.random.RandomState((int(self.spec.get("seed", 1)) * 7919 + key) & 0x7FFFFFFF)
            self._turb = (key, rs.randn(3, n, n, n).astype(np.float32))
        fld = self._turb[1]
        if self.native:
            return list(_nat.fluid_resample(fld, self.d.shape, amount))
        sx = self.cx / max(self.nx - 1, 1) * (n - 1)
        sy = self.cy / max(self.ny - 1, 1) * (n - 1)
        sz = self.cz / max(self.nz - 1, 1) * (n - 1)
        return [amount * _sample(fld[i], sx, sy, sz) for i in range(3)]

    def _vorticity(self, eps, dt):
        if self.native:
            for a in ("u", "v", "w"):
                setattr(self, a, np.ascontiguousarray(getattr(self, a), np.float32))
            _nat.fluid_vorticity(self.u, self.v, self.w, eps * dt)
            return
        u, v, w = self.u, self.v, self.w
        uy, uz = _grad(u)[1:]
        vx, _, vz = _grad(v)
        wx, wy, _ = _grad(w)
        ox, oy, oz = wy - vz, uz - wx, vx - uy
        mag = np.sqrt(ox * ox + oy * oy + oz * oz)
        gx, gy, gz = _grad(mag)
        gl = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-6
        nx_, ny_, nz_ = gx / gl, gy / gl, gz / gl
        s = eps * dt  # grid units
        self.u += s * (ny_ * oz - nz_ * oy)
        self.v += s * (nz_ * ox - nx_ * oz)
        self.w += s * (nx_ * oy - ny_ * ox)

    def _project(self, iters=40):
        if self.native:
            for a in ("u", "v", "w", "p"):
                setattr(self, a, np.ascontiguousarray(getattr(self, a), np.float32))
            _nat.fluid_project(self.u, self.v, self.w, self.p, self.solid, self.svel_g, iters,
                               float(self.spec.get("sor", 1.7)), self.floor)
            return
        u, v, w = self.u, self.v, self.w
        sol = self.solid.astype(bool) if self.solid is not None else None
        if sol is not None:
            sv = self.svel_g
            for k, comp in enumerate((u, v, w)):
                comp[sol] = 0 if sv is None else sv[k][sol]
        div = np.zeros_like(u)
        div[1:-1, 1:-1, 1:-1] = 0.5 * (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2] + v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]
                                       + w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1])
        I = (slice(1, -1),) * 3
        shifts = [(slice(1, -1), slice(1, -1), slice(2, None)), (slice(1, -1), slice(1, -1), slice(None, -2)),
                  (slice(1, -1), slice(2, None), slice(1, -1)), (slice(1, -1), slice(None, -2), slice(1, -1)),
                  (slice(2, None), slice(1, -1), slice(1, -1)), (slice(None, -2), slice(1, -1), slice(1, -1))]
        p = self.p * 0.9  # warm start
        if sol is None:
            for _ in range(iters):
                pn = np.zeros_like(p)
                pn[I] = (sum(p[sh] for sh in shifts) - div[I]) / 6.0
                p = pn
        else:
            F = (~sol).astype(np.float32)
            k = sum(F[sh] for sh in shifts)
            solve = (~sol[I]) & (k > 0)
            kk = np.maximum(k, 1)
            div[sol] = 0
            p[sol] = 0
            for _ in range(iters):
                pn = np.zeros_like(p)
                pn[I] = np.where(solve, (sum(p[sh] * F[sh] for sh in shifts) - div[I]) / kk, 0)
                p = pn
        self.p = p
        if sol is None:
            gx, gy, gz = _grad(p)
        else:
            def mirrored(sl, c):
                return np.where(sol[sl], p[c], p[sl])
            gx = np.zeros_like(p)
            gy = np.zeros_like(p)
            gz = np.zeros_like(p)
            c = (slice(None), slice(None), slice(1, -1))
            gx[c] = 0.5 * (mirrored((slice(None), slice(None), slice(2, None)), c)
                           - mirrored((slice(None), slice(None), slice(None, -2)), c))
            c = (slice(None), slice(1, -1), slice(None))
            gy[c] = 0.5 * (mirrored((slice(None), slice(2, None), slice(None)), c)
                           - mirrored((slice(None), slice(None, -2), slice(None)), c))
            c = (slice(1, -1), slice(None), slice(None))
            gz[c] = 0.5 * (mirrored((slice(2, None), slice(None), slice(None)), c)
                           - mirrored((slice(None, -2), slice(None), slice(None)), c))
            gx[sol] = gy[sol] = gz[sol] = 0
        self.u -= gx
        self.v -= gy
        self.w -= gz
        if self.floor:
            self.v[:, 0, :] = np.maximum(self.v[:, 0, :], 0)

    def _collider_mesh_local(self, cname, frame, inv):
        m = self.scene.world_mesh(cname, frame)
        if m is None or not m.num_verts:
            return None, None
        P = mu.transform_points(inv, m.verts)
        tris = np.asarray(m.triangles()[0], np.int32).reshape(-1, 3)
        return P, tris

    def _update_colliders(self, frame):
        """Voxelise the colliders at ``frame``: self.solid (uint8 grid) + self.svel (3, grid) in
        local units / second."""
        if not self.colliders:
            self.solid = None
            self.svel = None
            return
        key = round(float(frame), 6)
        if self._col_key == key:
            return
        self._col_key = key
        inv = self._world_to_local(frame)
        shp = self.d.shape
        solid = np.zeros(shp, bool)
        svel = np.zeros((3,) + shp, np.float32)
        fps = float(self.scene.fps)
        eps = 0.25  # frames, for the obstacle velocity finite difference
        lo_grid = -self.size / 2
        for col in self.colliders:
            cname = col.get("node") or col.get("name")
            if cname not in self.scene.nodes:
                continue
            shape = col.get("shape", "mesh")
            P0, tris = self._collider_mesh_local(cname, frame, inv)
            if P0 is None:
                # empty / non-mesh node: fall back to a sphere at its pivot
                Wc = self.scene.world_matrix(cname, frame)
                P0 = mu.transform_points(inv, Wc[:3, 3])[None]
                shape = "sphere"
                tris = None
            inv1 = self._world_to_local(frame + eps)
            P1, _ = self._collider_mesh_local(cname, frame + eps, inv1)
            if P1 is None or P1.shape != P0.shape:
                Wa = self.scene.world_matrix(cname, frame)
                Wb = self.scene.world_matrix(cname, frame + eps)
                P1 = mu.transform_points(inv1 @ Wb @ np.linalg.inv(Wa) @ np.linalg.inv(inv), P0)
            VV = (P1 - P0) * (fps / eps)
            thick = float(col.get("thickness", 0.5 * self.h))
            if shape == "sphere" or tris is None or len(tris) == 0:
                c = P0.mean(0)
                r = float(np.linalg.norm(P0 - c, axis=1).max()) if len(P0) > 1 else float(col.get("radius", 0.5))
                inside = (self.lx - c[0]) ** 2 + (self.ly - c[1]) ** 2 + (self.lz - c[2]) ** 2 < r * r
                vmean = VV.mean(0)
                solid |= inside
                for k in range(3):
                    svel[k][inside] = vmean[k]
                continue
            bmin = P0.min(0) - thick - self.h
            bmax = P0.max(0) + thick + self.h
            i0 = np.clip(np.floor((bmin - lo_grid) / self.h - 0.5).astype(int), 0, [self.nx, self.ny, self.nz])
            i1 = np.clip(np.ceil((bmax - lo_grid) / self.h - 0.5).astype(int) + 1, 0, [self.nx, self.ny, self.nz])
            if np.any(i1 <= i0):
                continue
            sl = (slice(i0[2], i1[2]), slice(i0[1], i1[1]), slice(i0[0], i1[0]))
            if shape == "box":
                bl, bh = P0.min(0), P0.max(0)
                inside = ((self.lx[sl] >= bl[0]) & (self.lx[sl] <= bh[0]) & (self.ly[sl] >= bl[1])
                          & (self.ly[sl] <= bh[1]) & (self.lz[sl] >= bl[2]) & (self.lz[sl] <= bh[2]))
                vmean = VV.mean(0)
                solid[sl] |= inside
                for k in range(3):
                    svel[k][sl][inside] = vmean[k]
                continue
            pts = np.stack([self.lx[sl].ravel(), self.ly[sl].ravel(), self.lz[sl].ravel()], 1)
            if _native_ok():
                dist, vel = _nat.mesh_sdf_vel(P0, tris, VV, pts)
            else:  # coarse fallback: inside the bounding sphere, rigid mean velocity
                c = P0.mean(0)
                r = float(np.linalg.norm(P0 - c, axis=1).max())
                dist = np.linalg.norm(pts - c, axis=1) - r
                vel = np.broadcast_to(VV.mean(0), pts.shape)
            inside = (dist < thick).reshape(solid[sl].shape)
            solid[sl] |= inside
            for k in range(3):
                sub = svel[k][sl]
                sub[inside] = vel[:, k].reshape(inside.shape)[inside]
        self.solid = solid.astype(np.uint8) if solid.any() else None
        self.svel = svel if self.solid is not None else None

    def _apply_colliders(self, frame):
        self._update_colliders(frame)
        if self.solid is None:
            return
        sol = self.solid.astype(bool)
        for k, comp in enumerate((self.u, self.v, self.w)):
            comp[sol] = self.svel_g[k][sol]
        self.d[sol] = 0
        self.T[sol] = 0

    def _advect_vel(self, dt_g):
        if self.native:
            self.u, self.v, self.w = _nat.fluid_advect(self.u, self.v, self.w, [self.u, self.v, self.w], dt_g,
                                                       maccormack=False)
            return
        x = self.cx - dt_g * self.u
        y = self.cy - dt_g * self.v
        z = self.cz - dt_g * self.w
        self.u, self.v, self.w = _sample(self.u, x, y, z), _sample(self.v, x, y, z), _sample(self.w, x, y, z)

    def _advect_scalar(self, f, dt_g):
        # MacCormack with min/max limiter (sharp, stable)
        x = self.cx - dt_g * self.u
        y = self.cy - dt_g * self.v
        z = self.cz - dt_g * self.w
        fwd = _sample(f, x, y, z)
        xb = self.cx + dt_g * self.u
        yb = self.cy + dt_g * self.v
        zb = self.cz + dt_g * self.w
        back = _sample(fwd, xb, yb, zb)
        corr = fwd + 0.5 * (f - back)
        lo, hi = _sample_minmax(f, x, y, z)
        return np.clip(corr, lo, hi).astype(np.float32)

    def _step(self, dt, frame):
        self._emit(dt, frame)
        dt_g = dt / self.h  # world velocity (units/s) -> cells per step factor
        buoy = float(self._pv("buoyancy", 3.0, frame))
        weight = float(self._pv("weight", 0.05, frame))
        self.v += dt * (buoy * self.T - weight * self.d)
        wind = np.asarray(self._pv("wind", [0.0, 0.0, 0.0], frame), float)
        if np.any(wind):
            self.u += dt * wind[0] * 0.5 * (self.d > 1e-3)
            self.v += dt * wind[1] * 0.5 * (self.d > 1e-3)
            self.w += dt * wind[2] * 0.5 * (self.d > 1e-3)
        turb = float(self._pv("turbulence", 0.5, frame))
        if turb > 0:
            tu, tv, tw = self._turbulence(frame, turb)
            act = np.clip(self.T + self.d, 0, 1)
            self.u += dt * tu * act
            self.v += dt * tv * act
            self.w += dt * tw * act
        # velocities are stored in world units/s; vorticity & projection in grid units
        self.u *= dt_g
        self.v *= dt_g
        self.w *= dt_g
        vort = float(self._pv("vorticity", 1.0, frame))
        if vort > 0:
            self._vorticity(vort * dt, 1.0)
        self._advect_vel(1.0)
        self.svel_g = None
        self._update_colliders(frame)
        if self.svel is not None:
            self.svel_g = (self.svel * np.float32(dt_g)).astype(np.float32)
        self._apply_colliders(frame)
        self._project(int(self.spec.get("pressure_iterations", 60 if self.native else 40)))
        if self.native:
            self.d, self.T = _nat.fluid_advect(self.u, self.v, self.w, [self.d, self.T], 1.0, maccormack=True,
                                               solid=self.solid)
        else:
            self.d = self._advect_scalar(self.d, 1.0)
            self.T = self._advect_scalar(self.T, 1.0)
            if self.solid is not None:
                self.d[self.solid.astype(bool)] = 0
                self.T[self.solid.astype(bool)] = 0
        self.u /= dt_g
        self.v /= dt_g
        self.w /= dt_g
        cool = float(self._pv("cooling", 1.0, frame))
        diss = float(self._pv("dissipation", 0.1, frame))
        self.T *= np.float32(np.exp(-cool * dt))
        self.d *= np.float32(np.exp(-diss * dt))
        mh = float(self._pv("max_heat", 1.0, frame))
        np.clip(self.T, 0, mh, out=self.T)

    # ------------------------------------------------------------------ public
    def advance_to(self, f):
        f = int(np.floor(f))
        fps = float(self.scene.fps)
        sub = max(1, self.substeps)
        # CFL-driven extra substeps
        while self.frame < f:
            vmax = float(max(np.abs(self.u).max(), np.abs(self.v).max(), np.abs(self.w).max(), 1e-6))
            n = max(sub, int(np.ceil(vmax / fps / self.h / 2.0)))
            n = min(n, 8)
            for s in range(n):
                self._step(1.0 / fps / n, self.frame + (s + 1) / n)
            self.frame += 1
            self.frames[self.frame] = (self.d.astype(np.float16), self.T.astype(np.float16))

    def at(self, frame):
        fi = int(np.floor(frame))
        if fi < self.start:
            z = np.zeros((self.nz, self.ny, self.nx), np.float32)
            return z, z
        if fi not in self.frames:
            if fi < self.frame:  # rewind (not cached)
                self.__init__(self.scene, self.name)
            self.advance_to(fi)
        d, T = self.frames[fi]
        return d.astype(np.float32), T.astype(np.float32)

    def bake(self, f_end):
        self.advance_to(int(f_end))
        if self.cache:
            self.save(self.cache)

    def save(self, path):
        ks = sorted(self.frames)
        np.savez_compressed(path, frames=np.asarray(ks, np.int32),
                            density=np.stack([self.frames[k][0] for k in ks]),
                            heat=np.stack([self.frames[k][1] for k in ks]),
                            vel=np.stack([self.u, self.v, self.w]).astype(np.float32), last=np.int32(self.frame))

    def _load_cache(self):
        z = np.load(self.cache)
        for i, k in enumerate(z["frames"]):
            self.frames[int(k)] = (z["density"][i], z["heat"][i])
        if len(z["frames"]):
            self.frame = int(z["frames"][-1])
            # resume from the last cached frame if asked for more
            self.d = z["density"][-1].astype(np.float32)
            self.T = z["heat"][-1].astype(np.float32)
            if "vel" in z and int(z["last"]) == self.frame and z["vel"].shape[1:] == self.d.shape:
                self.u, self.v, self.w = [a.astype(np.float32) for a in z["vel"]]


def get_sim(scene, name):
    cache = scene.__dict__.setdefault("_smoke_sims", {})
    if name not in cache:
        cache[name] = SmokeSim(scene, name)
    return cache[name]


def smoke_frame(scene, name, frame):
    """(density, heat) float32 grids [z,y,x] of the smoke node ``name`` at ``frame``."""
    return get_sim(scene, name).at(frame)
