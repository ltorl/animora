"""Polygon meshes: primitives, normals, Catmull-Clark subdivision, triangulation.

A :class:`Mesh` stores general polygons (n-gons), which is what makes Catmull-Clark subdivision
surfaces (Maya *Smooth Mesh*, Pixar OpenSubdiv style) possible:

``verts``        (V,3) float64 vertex positions
``counts``       (F,)  int  number of corners of each face
``indices``      (C,)  int  vertex index of every face corner (faces concatenated, CCW = front)
``uvs``          (C,2) float per-corner texture coordinates (or ``None``)
``normals``      (C,3) float explicit per-corner normals (or ``None`` -> auto smoothing)
``smooth_angle`` degrees; edges whose dihedral angle is above this are shaded sharp
                 (180 = fully smooth, 0 = faceted). Like Maya's *Soften/Harden Edge* angle.
``face_mat``     (F,)  int  per-face material slot (index into the node's material list)
"""
from __future__ import annotations

import math

import numpy as np

from . import noise as _noise


class Mesh:
    def __init__(self, verts, counts, indices, uvs=None, normals=None, smooth_angle=180.0, face_mat=None,
                 name="mesh"):
        self.verts = np.ascontiguousarray(verts, dtype=np.float64).reshape(-1, 3)
        self.counts = np.ascontiguousarray(counts, dtype=np.int64).ravel()
        self.indices = np.ascontiguousarray(indices, dtype=np.int64).ravel()
        if self.counts.sum() != self.indices.size:
            raise ValueError("Mesh: sum(counts) != len(indices)")
        self.uvs = None if uvs is None else np.ascontiguousarray(uvs, np.float64).reshape(-1, 2)
        if self.uvs is not None and len(self.uvs) != len(self.indices):
            raise ValueError("Mesh: uvs must be per-corner (len == len(indices))")
        self.normals = None if normals is None else np.ascontiguousarray(normals, np.float64).reshape(-1, 3)
        self.smooth_angle = float(smooth_angle)
        self.face_mat = (np.zeros(len(self.counts), np.int64) if face_mat is None
                         else np.ascontiguousarray(face_mat, np.int64).ravel())
        self.name = name

    # ------------------------------------------------------------------ basics
    @property
    def num_verts(self):
        return len(self.verts)

    @property
    def num_faces(self):
        return len(self.counts)

    @property
    def offsets(self):
        o = np.zeros(len(self.counts) + 1, np.int64)
        np.cumsum(self.counts, out=o[1:])
        return o

    def copy(self):
        return Mesh(self.verts.copy(), self.counts.copy(), self.indices.copy(),
                    None if self.uvs is None else self.uvs.copy(),
                    None if self.normals is None else self.normals.copy(),
                    self.smooth_angle, self.face_mat.copy(), self.name)

    def with_verts(self, verts):
        """Shallow copy sharing topology with new vertex positions."""
        m = Mesh.__new__(Mesh)
        m.verts = np.asarray(verts, np.float64).reshape(-1, 3)
        m.counts, m.indices, m.uvs = self.counts, self.indices, self.uvs
        m.normals, m.smooth_angle, m.face_mat, m.name = None, self.smooth_angle, self.face_mat, self.name
        return m

    def bounds(self):
        if len(self.verts) == 0:
            return np.zeros(3), np.zeros(3)
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def corner_face(self):
        return np.repeat(np.arange(len(self.counts)), self.counts)

    def corner_next(self):
        """Index of the next corner within the same face for every corner."""
        o = self.offsets
        c = np.arange(len(self.indices))
        f = self.corner_face()
        local = c - o[f]
        return o[f] + (local + 1) % self.counts[f]

    def corner_prev(self):
        o = self.offsets
        c = np.arange(len(self.indices))
        f = self.corner_face()
        local = c - o[f]
        return o[f] + (local - 1) % self.counts[f]

    # ------------------------------------------------------------------ normals
    def face_normals(self, normalized=True):
        """Newell normals (robust for n-gons). Unnormalised length = 2*area."""
        P = self.verts[self.indices]
        Q = self.verts[self.indices[self.corner_next()]]
        cr = np.cross(P, Q)
        f = self.corner_face()
        n = np.zeros((len(self.counts), 3))
        for k in range(3):
            n[:, k] = np.bincount(f, weights=cr[:, k], minlength=len(self.counts))
        if normalized:
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(ln, 1e-20)
        return n

    def vertex_normals(self):
        """Area-weighted smooth per-vertex normals (V,3)."""
        fn = self.face_normals(normalized=False)
        f = self.corner_face()
        vn = np.zeros_like(self.verts)
        for k in range(3):
            vn[:, k] = np.bincount(self.indices, weights=fn[f, k], minlength=len(self.verts))
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.maximum(ln, 1e-20)

    def corner_normals(self):
        """Per-corner shading normals honouring ``smooth_angle`` (or explicit ``normals``)."""
        if self.normals is not None:
            return self.normals
        C = len(self.indices)
        if C == 0:
            return np.zeros((0, 3))
        fn_raw = self.face_normals(normalized=False)
        ln = np.linalg.norm(fn_raw, axis=1, keepdims=True)
        fn = fn_raw / np.maximum(ln, 1e-20)
        f = self.corner_face()
        if self.smooth_angle >= 179.99:
            return self.vertex_normals()[self.indices]
        if self.smooth_angle <= 0.01:
            return fn[f]
        cos_t = math.cos(math.radians(self.smooth_angle))
        # group corners by vertex, compare every pair in the group
        order = np.argsort(self.indices, kind="stable")
        sv = self.indices[order]
        starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
        sizes = np.diff(np.r_[starts, C])
        gsize = np.repeat(sizes, sizes)           # group size for each sorted corner
        gstart = np.repeat(starts, sizes)
        total = int(gsize.sum())
        if total > 60_000_000:  # pathological valence; fall back to vertex normals
            return self.vertex_normals()[self.indices]
        ii = np.repeat(np.arange(C), gsize)       # sorted position of corner i
        run_start = np.repeat(np.cumsum(gsize) - gsize, gsize)
        jj = gstart[ii] + (np.arange(total) - run_start)
        ci, cj = order[ii], order[jj]
        fi, fj = f[ci], f[cj]
        ok = np.einsum("ij,ij->i", fn[fi], fn[fj]) >= cos_t - 1e-6
        w = fn_raw[fj] * ok[:, None]
        out = np.zeros((C, 3))
        for k in range(3):
            out[:, k] = np.bincount(ci, weights=w[:, k], minlength=C)
        ln = np.linalg.norm(out, axis=1, keepdims=True)
        bad = ln[:, 0] < 1e-20
        out = out / np.maximum(ln, 1e-20)
        out[bad] = fn[f[bad]]
        return out

    # ------------------------------------------------------------------ triangulation
    def triangulate_corners(self):
        """Fan triangulation. Returns (T,3) corner indices and (T,) face index."""
        counts = self.counts
        o = self.offsets
        ntri = np.maximum(counts - 2, 0)
        T = int(ntri.sum())
        face = np.repeat(np.arange(len(counts)), ntri)
        tstart = np.repeat(np.cumsum(ntri) - ntri, ntri)
        k = np.arange(T) - tstart
        base = o[face]
        tris = np.stack([base, base + k + 1, base + k + 2], axis=1)
        return tris, face

    def triangles(self):
        tc, face = self.triangulate_corners()
        return self.indices[tc], face

    def to_triangles(self):
        """Return dict with per-triangle-corner arrays P (T,3,3), N, UV (T,3,2), mat (T,)."""
        tc, face = self.triangulate_corners()
        N = self.corner_normals()
        out = {
            "P": self.verts[self.indices[tc]],
            "N": N[tc],
            "UV": (self.uvs[tc] if self.uvs is not None else np.zeros(tc.shape + (2,))),
            "mat": self.face_mat[face],
            "vidx": self.indices[tc],
        }
        return out

    # ------------------------------------------------------------------ edges
    def edges(self):
        """Unique undirected edges. Returns (E,2) vertex pairs, and for every corner the edge id
        of the edge leaving that corner."""
        a = self.indices
        b = self.indices[self.corner_next()]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo * (len(self.verts) + 1) + hi
        uk, inv = np.unique(key, return_inverse=True)
        E = np.stack([uk // (len(self.verts) + 1), uk % (len(self.verts) + 1)], axis=1)
        return E, inv

    # ------------------------------------------------------------------ operations
    def transformed(self, M):
        M = np.asarray(M, np.float64)
        v = self.verts @ M[:3, :3].T + M[:3, 3]
        m = self.with_verts(v)
        if self.normals is not None:
            nm = np.linalg.inv(M[:3, :3]).T
            nn = self.normals @ nm.T
            m.normals = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-20)
        return m

    def flipped(self):
        """Reverse winding of every face."""
        o = self.offsets
        new_idx = np.empty_like(self.indices)
        perm = np.empty(len(self.indices), np.int64)
        for f in range(len(self.counts)):
            s, e = o[f], o[f + 1]
            perm[s:e] = np.r_[s, np.arange(e - 1, s, -1)]
        new_idx = self.indices[perm]
        m = Mesh(self.verts.copy(), self.counts.copy(), new_idx,
                 None if self.uvs is None else self.uvs[perm],
                 None if self.normals is None else -self.normals[perm], self.smooth_angle, self.face_mat.copy(),
                 self.name)
        return m

    def subdivided(self, levels=1):
        m = self
        for _ in range(int(levels)):
            m = catmull_clark(m)
        return m

    def welded(self, tol=1e-6):
        """Merge vertices closer than ``tol``."""
        q = np.round(self.verts / max(tol, 1e-12)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        m = Mesh(self.verts[first], self.counts.copy(), inv[self.indices], self.uvs, self.normals,
                 self.smooth_angle, self.face_mat.copy(), self.name)
        return m

    def __repr__(self):
        return f"<Mesh {self.name!r} V={self.num_verts} F={self.num_faces}>"


# ======================================================================================
# merging
# ======================================================================================
def merge(meshes, name="merged"):
    """Concatenate meshes into one. UVs are filled with zeros for meshes lacking them."""
    meshes = [m for m in meshes if m is not None and m.num_faces > 0]
    if not meshes:
        return Mesh(np.zeros((0, 3)), [], [], name=name)
    verts, counts, idx, uvs, fm = [], [], [], [], []
    off = 0
    any_uv = any(m.uvs is not None for m in meshes)
    for m in meshes:
        verts.append(m.verts)
        counts.append(m.counts)
        idx.append(m.indices + off)
        fm.append(m.face_mat)
        if any_uv:
            uvs.append(m.uvs if m.uvs is not None else np.zeros((len(m.indices), 2)))
        off += len(m.verts)
    return Mesh(np.concatenate(verts), np.concatenate(counts), np.concatenate(idx),
                np.concatenate(uvs) if any_uv else None, None, min(m.smooth_angle for m in meshes),
                np.concatenate(fm), name)


# ======================================================================================
# Catmull-Clark
# ======================================================================================
def catmull_clark(mesh: Mesh) -> Mesh:
    """One level of Catmull-Clark subdivision (with boundary rules). Every n-gon becomes n quads.
    UVs are subdivided linearly (face-varying)."""
    V = mesh.verts
    nv = len(V)
    F = len(mesh.counts)
    if F == 0:
        return mesh.copy()
    idx = mesh.indices
    cf = mesh.corner_face()
    cn = mesh.corner_next()
    cp = mesh.corner_prev()
    # face points
    fp = np.zeros((F, 3))
    for k in range(3):
        fp[:, k] = np.bincount(cf, weights=V[idx, k], minlength=F)
    fp /= mesh.counts[:, None]
    # edges
    E, corner_edge = mesh.edges()
    ne = len(E)
    edge_nf = np.bincount(corner_edge, minlength=ne)
    edge_fsum = np.zeros((ne, 3))
    for k in range(3):
        edge_fsum[:, k] = np.bincount(corner_edge, weights=fp[cf, k], minlength=ne)
    mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    boundary_e = edge_nf < 2
    ep = np.where(boundary_e[:, None], mid,
                  (V[E[:, 0]] + V[E[:, 1]] + edge_fsum) / np.maximum(edge_nf + 2, 1)[:, None])
    # vertex points
    vf_count = np.bincount(idx, minlength=nv).astype(np.float64)  # faces per vertex
    vf_sum = np.zeros((nv, 3))
    for k in range(3):
        vf_sum[:, k] = np.bincount(idx, weights=fp[cf, k], minlength=nv)
    ve_count = np.bincount(E.ravel(), minlength=nv).astype(np.float64)
    ve_sum = np.zeros((nv, 3))
    for k in range(3):
        ve_sum[:, k] = (np.bincount(E[:, 0], weights=mid[:, k], minlength=nv) +
                        np.bincount(E[:, 1], weights=mid[:, k], minlength=nv))
    n = np.maximum(ve_count, 1)
    Favg = vf_sum / np.maximum(vf_count, 1)[:, None]
    Ravg = ve_sum / n[:, None]
    vp = (Favg + 2 * Ravg + (n[:, None] - 3) * V) / n[:, None]
    # boundary vertices
    be = E[boundary_e]
    if len(be):
        bcount = np.bincount(be.ravel(), minlength=nv)
        bsum = np.zeros((nv, 3))
        for k in range(3):
            bsum[:, k] = (np.bincount(be[:, 0], weights=V[be[:, 1], k], minlength=nv) +
                          np.bincount(be[:, 1], weights=V[be[:, 0], k], minlength=nv))
        isb = bcount >= 2
        vp[isb] = 0.75 * V[isb] + 0.125 * bsum[isb] * (2.0 / bcount[isb, None])
        corner_v = (bcount >= 2) & (vf_count <= 1)
        vp[corner_v] = V[corner_v]  # keep corners of open sheets sharp
    isolated = vf_count == 0
    vp[isolated] = V[isolated]
    new_verts = np.concatenate([vp, ep, fp])
    # faces: each corner c -> quad (v_c, e(c->next), face, e(prev->c))
    C = len(idx)
    e_out = corner_edge
    e_in = corner_edge[cp]
    quads = np.stack([idx, nv + e_out, np.full(C, 0) + nv + ne + cf, nv + e_in], axis=1)
    new_idx = quads.ravel()
    new_counts = np.full(C, 4, np.int64)
    new_uvs = None
    if mesh.uvs is not None:
        uv = mesh.uvs
        fuv = np.zeros((F, 2))
        for k in range(2):
            fuv[:, k] = np.bincount(cf, weights=uv[:, k], minlength=F)
        fuv /= mesh.counts[:, None]
        new_uvs = np.stack([uv, 0.5 * (uv + uv[cn]), fuv[cf], 0.5 * (uv + uv[cp])], axis=1).reshape(-1, 2)
    return Mesh(new_verts, new_counts, new_idx, new_uvs, None,
                180.0 if mesh.smooth_angle < 180 else mesh.smooth_angle, mesh.face_mat[cf], mesh.name)


# ======================================================================================
# primitives
# ======================================================================================
def _grid_faces(nu, nv, wrap_u=False):
    """Quad indices for an (nv+1) x (nu+1) vertex grid (row-major, v rows)."""
    cols = nu if wrap_u else nu + 1
    faces = []
    for j in range(nv):
        for i in range(nu):
            i1 = (i + 1) % cols if wrap_u else i + 1
            a = j * cols + i
            b = j * cols + i1
            c = (j + 1) * cols + i1
            d = (j + 1) * cols + i
            faces.append((a, d, c, b))
    return np.array(faces, np.int64)


def plane(size=1.0, subdivisions=1, width=None, depth=None, uv_scale=1.0):
    """XZ plane centred at the origin facing +Y."""
    w = float(width if width is not None else size)
    d = float(depth if depth is not None else size)
    n = max(1, int(subdivisions))
    xs = np.linspace(-w / 2, w / 2, n + 1)
    zs = np.linspace(d / 2, -d / 2, n + 1)
    X, Z = np.meshgrid(xs, zs)
    verts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    us = np.linspace(0, 1, n + 1) * uv_scale
    vs = np.linspace(0, 1, n + 1) * uv_scale
    U, Vv = np.meshgrid(us, vs)
    vuv = np.stack([U.ravel(), Vv.ravel()], axis=1)
    faces = _grid_faces(n, n)
    # winding: a(j,i) d(j+1,i) c(j+1,i+1) b(j,i+1); rows go toward -Z so normal = +Y requires
    # order a,b,c,d
    faces = faces[:, [0, 3, 2, 1]]
    idx = faces.ravel()
    return Mesh(verts, np.full(len(faces), 4), idx, vuv[idx], smooth_angle=180.0, name="plane")


def grid(size=10.0, subdivisions=10, **kw):
    return plane(size=size, subdivisions=subdivisions, **kw)


def cube(size=1.0, width=None, height=None, depth=None, smooth_angle=30.0):
    """Axis-aligned box centred at the origin; 8 shared vertices (subdivides into a sphere-ish
    shape), per-face UVs. ``size`` may be a scalar or ``[x, y, z]``."""
    sz3 = np.asarray(size, float).reshape(-1)
    if sz3.size == 1:
        sz3 = np.repeat(sz3, 3)
    sx = float(width if width is not None else sz3[0]) / 2
    sy = float(height if height is not None else sz3[1]) / 2
    sz = float(depth if depth is not None else sz3[2]) / 2
    v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
    faces = np.array([
        [4, 5, 6, 7],  # +z
        [1, 0, 3, 2],  # -z
        [5, 1, 2, 6],  # +x
        [0, 4, 7, 3],  # -x
        [7, 6, 2, 3],  # +y
        [0, 1, 5, 4],  # -y
    ])
    uv = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), (6, 1))
    return Mesh(v, np.full(6, 4), faces.ravel(), uv, smooth_angle=smooth_angle, name="cube")


box = cube


def uv_sphere(radius=1.0, segments=32, rings=16):
    """Latitude/longitude sphere with single pole vertices (triangle fans at the poles)."""
    segments, rings = max(3, int(segments)), max(2, int(rings))
    verts = [[0, radius, 0]]
    for j in range(1, rings):
        th = math.pi * j / rings
        y = math.cos(th) * radius
        r = math.sin(th) * radius
        for i in range(segments):
            ph = 2 * math.pi * i / segments
            verts.append([r * math.sin(ph), y, r * math.cos(ph)])
    verts.append([0, -radius, 0])
    verts = np.array(verts)
    bottom = len(verts) - 1
    counts, idx, uvs = [], [], []

    def vi(j, i):
        return 1 + (j - 1) * segments + (i % segments)

    for i in range(segments):  # top fan
        u0, u1 = i / segments, (i + 1) / segments
        counts.append(3)
        idx += [0, vi(1, i), vi(1, i + 1)]
        uvs += [[(u0 + u1) / 2, 1.0], [u0, 1 - 1 / rings], [u1, 1 - 1 / rings]]
    for j in range(1, rings - 1):
        for i in range(segments):
            u0, u1 = i / segments, (i + 1) / segments
            v0, v1 = 1 - j / rings, 1 - (j + 1) / rings
            counts.append(4)
            idx += [vi(j, i), vi(j + 1, i), vi(j + 1, i + 1), vi(j, i + 1)]
            uvs += [[u0, v0], [u0, v1], [u1, v1], [u1, v0]]
    for i in range(segments):  # bottom fan
        u0, u1 = i / segments, (i + 1) / segments
        counts.append(3)
        idx += [vi(rings - 1, i), bottom, vi(rings - 1, i + 1)]
        uvs += [[u0, 1 / rings], [(u0 + u1) / 2, 0.0], [u1, 1 / rings]]
    return Mesh(verts, counts, idx, uvs, smooth_angle=180.0, name="sphere")


sphere = uv_sphere


def ico_sphere(radius=1.0, subdivisions=3):
    """Geodesic sphere (uniform triangles). UVs are spherical with seam fix-up."""
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                  [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(int(subdivisions)):
        edges = {}
        verts = list(v)
        nf = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in edges:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                edges[key] = len(verts) - 1
            return edges[key]

        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(verts)
        f = np.array(nf)
    idx = f.ravel()
    p = v[idx]
    u = 0.5 + np.arctan2(p[:, 0], p[:, 2]) / (2 * math.pi)
    vv = 0.5 + np.arcsin(np.clip(p[:, 1], -1, 1)) / math.pi
    uv = np.stack([u, vv], axis=1).reshape(-1, 3, 2)
    # seam fix: if a triangle spans the seam, shift low u's up by 1
    us = uv[:, :, 0]
    span = us.max(axis=1) - us.min(axis=1)
    fix = span > 0.5
    us[fix] = np.where(us[fix] < 0.5, us[fix] + 1, us[fix])
    uv[:, :, 0] = us
    return Mesh(v * radius, np.full(len(f), 3), idx, uv.reshape(-1, 2), smooth_angle=180.0, name="icosphere")


def cylinder(radius=0.5, height=1.0, segments=32, caps=True, height_segments=1, radius_top=None,
             smooth_angle=40.0):
    """Y-up cylinder (or truncated cone with ``radius_top``) centred at the origin."""
    segments = max(3, int(segments))
    hs = max(1, int(height_segments))
    rt = radius if radius_top is None else float(radius_top)
    verts, counts, idx, uvs = [], [], [], []
    for j in range(hs + 1):
        a = j / hs
        y = -height / 2 + a * height
        r = radius + (rt - radius) * a
        for i in range(segments):
            ph = 2 * math.pi * i / segments
            verts.append([r * math.sin(ph), y, r * math.cos(ph)])
    for j in range(hs):
        for i in range(segments):
            i1 = (i + 1) % segments
            a, b = j * segments + i, j * segments + i1
            c, d = (j + 1) * segments + i1, (j + 1) * segments + i
            counts.append(4)
            idx += [a, b, c, d]
            u0, u1 = i / segments, (i + 1) / segments
            v0, v1 = j / hs, (j + 1) / hs
            uvs += [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
    if caps:
        top0 = hs * segments
        ring = list(range(top0, top0 + segments))
        if rt > 1e-9:
            counts.append(segments)
            idx += ring
            uvs += [[0.5 + 0.5 * math.sin(2 * math.pi * i / segments), 0.5 + 0.5 * math.cos(2 * math.pi * i / segments)]
                    for i in range(segments)]
        if radius > 1e-9:
            counts.append(segments)
            ring = list(range(segments))[::-1]
            idx += ring
            uvs += [[0.5 + 0.5 * math.sin(2 * math.pi * i / segments), 0.5 - 0.5 * math.cos(2 * math.pi * i / segments)]
                    for i in ring]
    m = Mesh(np.array(verts), counts, idx, uvs, smooth_angle=smooth_angle, name="cylinder")
    if rt <= 1e-9 or radius <= 1e-9:
        m = m.welded(1e-9)
    return m


def cone(radius=0.5, height=1.0, segments=32, caps=True):
    m = cylinder(radius, height, segments, caps, radius_top=0.0)
    m.name = "cone"
    return m


def disc(radius=1.0, segments=32, rings=1):
    """Flat disc in XZ facing +Y, built from quad rings around a centre fan."""
    segments, rings = max(3, int(segments)), max(1, int(rings))
    verts = [[0, 0, 0]]
    for j in range(1, rings + 1):
        r = radius * j / rings
        for i in range(segments):
            ph = 2 * math.pi * i / segments
            verts.append([r * math.sin(ph), 0, r * math.cos(ph)])
    verts = np.array(verts)
    counts, idx = [], []
    for i in range(segments):
        counts.append(3)
        idx += [0, 1 + i, 1 + (i + 1) % segments]
    for j in range(1, rings):
        for i in range(segments):
            i1 = (i + 1) % segments
            a, b = 1 + (j - 1) * segments + i, 1 + (j - 1) * segments + i1
            c, d = 1 + j * segments + i1, 1 + j * segments + i
            counts.append(4)
            idx += [a, d, c, b]
    idx = np.array(idx)
    uv = 0.5 + 0.5 * verts[idx][:, [0, 2]] / radius
    uv[:, 1] = 1 - uv[:, 1]
    return Mesh(verts, counts, idx, uv, smooth_angle=180.0, name="disc")


def torus(major_radius=1.0, minor_radius=0.25, major_segments=48, minor_segments=16):
    """Torus lying in the XZ plane."""
    M, N = max(3, int(major_segments)), max(3, int(minor_segments))
    verts = []
    for i in range(M):
        u = 2 * math.pi * i / M
        for j in range(N):
            v = 2 * math.pi * j / N
            r = major_radius + minor_radius * math.cos(v)
            verts.append([r * math.sin(u), minor_radius * math.sin(v), r * math.cos(u)])
    counts, idx, uvs = [], [], []
    for i in range(M):
        for j in range(N):
            i1, j1 = (i + 1) % M, (j + 1) % N
            a, b = i * N + j, i1 * N + j
            c, d = i1 * N + j1, i * N + j1
            counts.append(4)
            idx += [a, b, c, d]
            u0, u1 = i / M, (i + 1) / M
            v0, v1 = j / N, (j + 1) / N
            uvs += [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
    return Mesh(np.array(verts), counts, idx, uvs, smooth_angle=180.0, name="torus")


def lathe(profile, segments=32, angle=360.0, smooth_angle=60.0, name="lathe"):
    """Revolve a 2D profile ``[[radius, y], ...]`` (bottom to top) around the Y axis."""
    prof = np.asarray(profile, float).reshape(-1, 2)
    P = len(prof)
    segments = max(3, int(segments))
    full = abs(angle) >= 359.999
    cols = segments if full else segments + 1
    verts = []
    for i in range(cols):
        ph = math.radians(angle) * i / segments
        for r, y in prof:
            verts.append([r * math.sin(ph), y, r * math.cos(ph)])
    # arc-length v coordinate
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    vcoord = seg / max(seg[-1], 1e-9)
    counts, idx, uvs = [], [], []
    for i in range(segments):
        i1 = (i + 1) % cols
        for j in range(P - 1):
            a, b = i * P + j, i1 * P + j
            c, d = i1 * P + j + 1, i * P + j + 1
            counts.append(4)
            idx += [a, b, c, d]
            u0, u1 = i / segments, (i + 1) / segments
            uvs += [[u0, vcoord[j]], [u1, vcoord[j]], [u1, vcoord[j + 1]], [u0, vcoord[j + 1]]]
    m = Mesh(np.array(verts), counts, idx, uvs, smooth_angle=smooth_angle, name=name)
    # collapse vertices on the axis
    m = _collapse_degenerate(m.welded(1e-9))
    return m


def _collapse_degenerate(m: Mesh) -> Mesh:
    """Remove repeated consecutive vertices inside faces (quads -> tris at poles), drop faces
    with < 3 unique corners."""
    o = m.offsets
    keep_corner = np.ones(len(m.indices), bool)
    nxt = m.corner_next()
    dup = m.indices == m.indices[nxt]
    keep_corner[dup] = False
    cf = m.corner_face()
    newcounts = np.bincount(cf[keep_corner], minlength=len(m.counts))
    face_ok = newcounts >= 3
    keep_corner &= face_ok[cf]
    return Mesh(m.verts, newcounts[face_ok], m.indices[keep_corner],
                None if m.uvs is None else m.uvs[keep_corner], None, m.smooth_angle, m.face_mat[face_ok], m.name)


def capsule(radius=0.5, height=2.0, segments=32, rings=8):
    """Y-up capsule: total height ``height`` including the hemispherical caps."""
    cyl = max(0.0, height - 2 * radius)
    prof = []
    for k in range(rings + 1):
        a = -math.pi / 2 + (math.pi / 2) * k / rings
        prof.append([radius * math.cos(a), -cyl / 2 + radius * math.sin(a)])
    for k in range(rings + 1):
        a = (math.pi / 2) * k / rings
        prof.append([radius * math.cos(a), cyl / 2 + radius * math.sin(a)])
    prof[0][0] = 0.0
    prof[-1][0] = 0.0
    return lathe(prof, segments, smooth_angle=180.0, name="capsule")


def tube(path, radius=0.1, segments=12, closed=False, radius_fn=None, caps=True):
    """Sweep a circle along a 3D polyline ``path`` (rotation-minimising frames)."""
    path = np.asarray(path, float).reshape(-1, 3)
    n = len(path)
    if n < 2:
        raise ValueError("tube path needs >= 2 points")
    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    ref = np.array([0, 1, 0]) if abs(tang[0][1]) < 0.9 else np.array([1, 0, 0])
    nrm = np.cross(tang[0], ref)
    nrm /= np.linalg.norm(nrm)
    frames = []
    for i in range(n):
        if i > 0:  # parallel transport
            nrm = nrm - tang[i] * np.dot(nrm, tang[i])
            nrm /= max(np.linalg.norm(nrm), 1e-12)
        b = np.cross(tang[i], nrm)
        frames.append((nrm.copy(), b))
    verts = []
    for i in range(n):
        r = radius_fn(i / (n - 1)) if radius_fn else radius
        nn, bb = frames[i]
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append(path[i] + r * (math.cos(a) * nn + math.sin(a) * bb))
    counts, idx, uvs = [], [], []
    rows = n if not closed else n + 1
    for i in range(rows - 1):
        i0, i1 = i % n, (i + 1) % n
        for j in range(segments):
            j1 = (j + 1) % segments
            a, b, c, d = i0 * segments + j, i0 * segments + j1, i1 * segments + j1, i1 * segments + j
            counts.append(4)
            idx += [a, b, c, d]
            uvs += [[j / segments, i / (rows - 1)], [(j + 1) / segments, i / (rows - 1)],
                    [(j + 1) / segments, (i + 1) / (rows - 1)], [j / segments, (i + 1) / (rows - 1)]]
    if caps and not closed:
        counts.append(segments)
        idx += list(range(segments - 1, -1, -1))
        uvs += [[0.5, 0.5]] * segments
        counts.append(segments)
        idx += list(range((n - 1) * segments, n * segments))
        uvs += [[0.5, 0.5]] * segments
    return Mesh(np.array(verts), counts, idx, uvs, smooth_angle=60.0, name="tube")


def terrain(size=10.0, resolution=64, height=1.5, frequency=0.25, octaves=5, seed=0, ridged=False):
    """Noise heightfield in XZ."""
    m = plane(size, resolution)
    x, z = m.verts[:, 0], m.verts[:, 2]
    if ridged:
        h = 1 - np.abs(_noise.fbm(x * frequency, 0.0, z * frequency, octaves, seed=seed))
        h = h * h
    else:
        h = _noise.fbm(x * frequency, 0.0, z * frequency, octaves, seed=seed)
    m.verts[:, 1] = h * height
    m.name = "terrain"
    return m


def from_triangles(verts, faces, uvs=None, smooth_angle=180.0, name="mesh"):
    faces = np.asarray(faces, np.int64).reshape(-1, 3)
    uv = None
    if uvs is not None:
        uvs = np.asarray(uvs, float)
        uv = uvs[faces.ravel()] if len(uvs) == len(verts) else uvs
    return Mesh(verts, np.full(len(faces), 3), faces.ravel(), uv, smooth_angle=smooth_angle, name=name)


def heart(size=1.0):
    """Heart-shaped lathe-free extrusion example (used by templates)."""
    t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], 1) / 17.0 * size
    return extrude_polygon(pts, depth=0.3 * size)


def extrude_polygon(points2d, depth=0.2, name="extrusion"):
    """Extrude a simple closed 2D polygon (XY) along Z, centred at z=0."""
    pts = np.asarray(points2d, float)
    n = len(pts)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area < 0:
        pts = pts[::-1]
    front = np.c_[pts, np.full(n, depth / 2)]
    back = np.c_[pts, np.full(n, -depth / 2)]
    verts = np.concatenate([front, back])
    counts, idx = [n, n], list(range(n)) + list(range(2 * n - 1, n - 1, -1))
    for i in range(n):
        i1 = (i + 1) % n
        counts.append(4)
        idx += [i, n + i, n + i1, i1]
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    vi = np.array(idx)
    uv = (verts[vi][:, :2] - lo) / span
    return Mesh(verts, counts, idx, uv, smooth_angle=35.0, name=name)


PRIMITIVES = {
    "plane": plane, "grid": grid, "cube": cube, "box": cube, "sphere": uv_sphere, "uv_sphere": uv_sphere,
    "icosphere": ico_sphere, "ico_sphere": ico_sphere, "cylinder": cylinder, "cone": cone, "disc": disc,
    "torus": torus, "capsule": capsule, "lathe": lathe, "tube": tube, "terrain": terrain, "heart": heart,
    "extrude": extrude_polygon,
}


def _register_modeling():
    from . import modeling as _md
    PRIMITIVES.update({"sdf": _md.sdf, "csg": _md.sdf, "metaballs": _md.metaballs, "text": _md.text,
                       "sweep": _md.sweep, "loft": _md.loft})


_register_modeling()


def make_primitive(kind, **params):
    fn = PRIMITIVES.get(kind)
    if fn is None:
        raise ValueError(f"unknown primitive {kind!r}; available: {sorted(PRIMITIVES)}")
    import inspect
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in params.items() if k in sig.parameters}
    unknown = set(params) - set(accepted)
    if unknown:
        raise ValueError(f"primitive {kind!r} does not take {sorted(unknown)}; parameters: {list(sig.parameters)}")
    return fn(**accepted)
