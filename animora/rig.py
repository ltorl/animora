"""Rigging: inverse kinematics, skinning (linear / dual-quaternion) and blend shapes."""
from __future__ import annotations

import math

import numpy as np

from . import mathutil as mu
from .params import Param

IK_SOLVERS = ("auto", "two_bone", "fabrik", "ccd")


# ============================================================================================
# IK
# ============================================================================================
def ik_chain(scene, h):
    """Resolve the joint chain of an IK handle (root first)."""
    if "chain" in h:
        chain = list(h["chain"])
    else:
        if "start" not in h or "end" not in h:
            raise ValueError(f"IK handle {h.get('name', '?')!r} needs 'chain' or 'start'+'end'")
        chain = [h["end"]]
        n = h["end"]
        while n != h["start"]:
            if n not in scene.nodes:
                raise ValueError(f"IK: joint {n!r} does not exist")
            n = scene.nodes[n].parent
            if n is None:
                raise ValueError(f"IK: {h['start']!r} is not an ancestor of {h['end']!r}")
            chain.append(n)
        chain.reverse()
    for j in chain:
        if j not in scene.nodes:
            raise ValueError(f"IK: joint {j!r} does not exist")
    for a, b in zip(chain, chain[1:]):
        if scene.nodes[b].parent != a:
            raise ValueError(f"IK chain: {b!r} is not a child of {a!r}")
    if len(chain) < 2:
        raise ValueError("IK chain needs at least two joints")
    if "target" not in h:
        raise ValueError(f"IK handle on {chain} needs a 'target' node or point")
    return chain


def _frame_from(u, n):
    u = mu.normalize(u)
    n = n - np.dot(n, u) * u
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        return None
    n = n / ln
    return np.stack([u, n, np.cross(u, n)], axis=1)


def _align(u0, n0, u1, n1):
    """Rotation mapping direction u0 -> u1 and plane normal n0 -> n1 (falls back to shortest arc)."""
    A = _frame_from(u0, n0) if n0 is not None else None
    B = _frame_from(u1, n1) if n1 is not None else None
    if A is None or B is None:
        return mu.rotation_between_matrix(u0, u1)
    return B @ A.T


def _solve_two_bone(P, target, pole_dir, stretch=False, soft=0.0):
    a, b, c = P
    l1 = np.linalg.norm(b - a)
    l2 = np.linalg.norm(c - b)
    d_vec = target - a
    d = np.linalg.norm(d_vec)
    if d < 1e-9:
        return P.copy()
    dirn = d_vec / d
    reach = l1 + l2
    if stretch and d > reach:
        k = d / reach
        l1, l2 = l1 * k, l2 * k
    dd = float(np.clip(d, abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6))
    if soft > 0 and not stretch:  # soft IK: ease into full extension to avoid knee pop
        ds = reach - soft
        if d > ds:
            dd = min(dd, ds + soft * (1 - math.exp(-(d - ds) / soft)))
    ortho = pole_dir - np.dot(pole_dir, dirn) * dirn
    if np.linalg.norm(ortho) < 1e-9:
        ortho = np.cross(dirn, [0, 0, 1.0]) if abs(dirn[2]) < 0.9 else np.cross(dirn, [1.0, 0, 0])
    ortho = mu.normalize(ortho)
    cos_a = np.clip((l1 * l1 + dd * dd - l2 * l2) / (2 * l1 * dd), -1, 1)
    sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
    nb = a + l1 * (cos_a * dirn + sin_a * ortho)
    nc = a + dirn * dd
    if stretch and d > reach:
        nc = target.copy()
    return np.stack([a, nb, nc])


def _solve_fabrik(P, target, iterations=20, tol=1e-4):
    P = P.copy()
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    root = P[0].copy()
    if np.linalg.norm(target - root) >= L.sum():
        d = mu.normalize(target - root)
        for i in range(1, len(P)):
            P[i] = P[i - 1] + d * L[i - 1]
        return P
    for _ in range(iterations):
        P[-1] = target
        for i in range(len(P) - 2, -1, -1):
            P[i] = P[i + 1] + mu.normalize(P[i] - P[i + 1]) * L[i]
        P[0] = root
        for i in range(1, len(P)):
            P[i] = P[i - 1] + mu.normalize(P[i] - P[i - 1]) * L[i - 1]
        if np.linalg.norm(P[-1] - target) < tol:
            break
    return P


def _solve_ccd(P, target, iterations=20, tol=1e-4):
    P = P.copy()
    for _ in range(iterations):
        for i in range(len(P) - 2, -1, -1):
            R = mu.rotation_between_matrix(P[-1] - P[i], target - P[i])
            P[i + 1:] = (P[i + 1:] - P[i]) @ R.T + P[i]
        if np.linalg.norm(P[-1] - target) < tol:
            break
    return P


def _apply_pole(P, pole):
    """Rotate interior joints of a FABRIK/CCD solution toward a pole point."""
    P = P.copy()
    for i in range(1, len(P) - 1):
        a, c = P[i - 1], P[i + 1]
        axis = c - a
        if np.linalg.norm(axis) < 1e-9:
            continue
        axis = mu.normalize(axis)
        proj = lambda x: x - a - np.dot(x - a, axis) * axis
        vb, vp = proj(P[i]), proj(pole)
        if np.linalg.norm(vb) < 1e-9 or np.linalg.norm(vp) < 1e-9:
            continue
        ang = math.atan2(np.dot(np.cross(vb, vp), axis), np.dot(vb, vp))
        R = mu.quat_to_matrix3(mu.quat_axis_angle(axis, math.degrees(ang)))
        P[i:] = (P[i:] - a) @ R.T + a  # rotate the rest of the chain about the a-c axis
        # restore end point (rotation about a-c keeps c; joints beyond c are moved consistently)
    return P


def solve_ik(st, h):
    """Solve an IK handle for one frame. Returns {joint: new local matrix}."""
    scene = st.scene
    chain = h["_chain"]
    frame, ctx = st.frame, st.ctx
    weight = float(Param(h.get("weight", 1.0)).evaluate(frame, ctx))
    if weight <= 0 or h.get("enabled", True) is False:
        return {}
    pw0 = st.parent_world(chain[0])
    fkL = [st.fk_local(j) for j in chain]
    W = []
    m = pw0
    for L in fkL:
        m = m @ L
        W.append(m)
    P = np.array([w[:3, 3] for w in W])
    tgt = h["target"]
    if isinstance(tgt, str):
        TW = st.world(tgt)
        target = TW[:3, 3].copy()
    else:
        TW = None
        target = np.asarray(Param(tgt).evaluate(frame, ctx), float)
    # pole direction
    pole_pt = None
    if h.get("pole") is not None:
        pole = h["pole"]
        pole_pt = st.world(pole)[:3, 3] if isinstance(pole, str) else np.asarray(
            Param(pole).evaluate(frame, ctx), float)
    twist = float(Param(h.get("twist", 0.0)).evaluate(frame, ctx))
    solver = h.get("solver", "auto")
    if solver == "auto":
        solver = "two_bone" if len(chain) == 3 else "fabrik"
    if solver not in IK_SOLVERS:
        raise ValueError(f"unknown IK solver {solver!r}; use {IK_SOLVERS}")
    if solver == "two_bone":
        if len(chain) != 3:
            raise ValueError("two_bone IK needs exactly 3 joints (e.g. shoulder, elbow, wrist)")
        if pole_pt is not None:
            pole_dir = pole_pt - P[0]
        else:
            mid = P[0] + np.dot(P[1] - P[0], mu.normalize(P[2] - P[0])) * mu.normalize(P[2] - P[0])
            pole_dir = P[1] - mid
            if np.linalg.norm(pole_dir) < 1e-6:  # straight chain: use the joint's local -z? pick +z of root
                pole_dir = W[1][:3, :3] @ np.array([0, 0, 1.0]) if h.get("bend_axis") is None else \
                    W[1][:3, :3] @ np.asarray(h["bend_axis"], float)
        if twist:
            ax = mu.normalize(target - P[0])
            pole_dir = mu.quat_to_matrix3(mu.quat_axis_angle(ax, twist)) @ pole_dir
        NP = _solve_two_bone(P, target, pole_dir, bool(h.get("stretch", False)), float(h.get("soft", 0.0)))
    else:
        it = int(h.get("iterations", 20))
        NP = _solve_fabrik(P, target, it) if solver == "fabrik" else _solve_ccd(P, target, it)
        if pole_pt is not None:
            NP = _apply_pole(NP, pole_pt)
        if twist:
            ax = mu.normalize(NP[-1] - NP[0])
            R = mu.quat_to_matrix3(mu.quat_axis_angle(ax, twist))
            NP = (NP - NP[0]) @ R.T + NP[0]
    # convert positions to local rotations
    out = {}
    parent_new = pw0
    n = len(chain)
    old_normals = []
    new_normals = []
    for i in range(n - 1):
        if i + 2 < n:
            old_normals.append(np.cross(P[i + 1] - P[i], P[i + 2] - P[i + 1]))
            new_normals.append(np.cross(NP[i + 1] - NP[i], NP[i + 2] - NP[i + 1]))
        else:
            old_normals.append(old_normals[-1] if old_normals else None)
            new_normals.append(new_normals[-1] if new_normals else None)
    for i in range(n - 1):
        Wi = parent_new @ fkL[i]
        child = (Wi @ fkL[i + 1])[:3, 3]
        cur_dir = child - Wi[:3, 3]
        new_dir = NP[i + 1] - Wi[:3, 3]
        # bring the FK bend normal along with the chain's parent rotation
        prot = (parent_new[:3, :3] @ np.linalg.inv(W[i - 1][:3, :3] if i > 0 else pw0[:3, :3]))
        n0 = old_normals[i]
        n0 = None if n0 is None or np.linalg.norm(n0) < 1e-8 else prot @ n0
        n1 = new_normals[i]
        n1 = None if n1 is None or np.linalg.norm(n1) < 1e-8 else n1
        R = _align(cur_dir, n0, new_dir, n1)
        Wn = Wi.copy()
        Wn[:3, :3] = R @ Wi[:3, :3]
        Ln = np.linalg.inv(parent_new) @ Wn
        out[chain[i]] = Ln
        parent_new = Wn
    # end joint orientation
    end = chain[-1]
    if h.get("orient_end") and TW is not None:
        We = parent_new @ fkL[-1]
        s = np.linalg.norm(We[:3, :3], axis=0)
        _, Rt, _ = mu.decompose(TW)
        We[:3, :3] = Rt * s[None, :]
        out[end] = np.linalg.inv(parent_new) @ We
    if weight < 1:
        for j, L in list(out.items()):
            out[j] = mu.blend_matrices([st.fk_local(j), L], [1 - weight, weight])
    return out


# ============================================================================================
# skinning
# ============================================================================================
def skin_joints(scene, node):
    sk = node.spec["skin"]
    if isinstance(sk, (list, tuple)):
        sk = {"joints": sk}
    if "joints" in sk:
        return list(sk["joints"])
    root = sk.get("root")
    if root is None:
        raise ValueError(f"skin on {node.name!r} needs 'joints' or 'root'")
    out = []

    def rec(n):
        if scene.nodes[n].type == "joint":
            out.append(n)
        for c in scene.children(n):
            rec(c)

    rec(root)
    return out


def _skin_spec(node):
    sk = node.spec["skin"]
    return {"joints": sk} if isinstance(sk, (list, tuple)) else dict(sk)


def _point_segment_dist(P, a, b):
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-12:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _vertex_neighbors(mesh):
    E, _ = mesh.edges()
    return E


def bind_world(scene, st, name):
    """World matrix of a joint in its *bind* pose: forward kinematics only (IK handles are
    ignored), so binding at a frame where IK is active still uses the authored rest skeleton."""
    node = scene.nodes[name]
    if node.type != "joint":
        return st.world(name)
    par = node.parent
    pw = bind_world(scene, st, par) if par else np.eye(4)
    return pw @ st.fk_local(name)


def auto_weights(scene, node, rest_world_verts, joints, bind_frame, max_influences=4, smooth=2,
                 falloff=4.0, mesh=None):
    """Distance-based automatic skin weights (bone segments, inverse power falloff, smoothed)."""
    P = rest_world_verts
    nj = len(joints)
    D = np.zeros((len(P), nj))
    st = scene.state(bind_frame)
    jset = set(joints)
    for k, j in enumerate(joints):
        a = bind_world(scene, st, j)[:3, 3]
        kids = [c for c in scene.children(j) if c in jset or scene.nodes[c].type == "joint"]
        if kids:
            d = np.min([_point_segment_dist(P, a, bind_world(scene, st, c)[:3, 3]) for c in kids], axis=0)
        else:
            # leaf joint: extend along the parent->joint direction by half the parent bone
            par = scene.nodes[j].parent
            if par and par in scene.nodes:
                pa = bind_world(scene, st, par)[:3, 3]
                b = a + (a - pa) * 0.5
                d = _point_segment_dist(P, a, b)
            else:
                d = np.linalg.norm(P - a, axis=1)
        D[:, k] = d
    scale = max(1e-6, float(np.median(D.min(axis=1))) + 1e-6)
    Wt = 1.0 / (D / scale + 1e-4) ** falloff
    if max_influences and max_influences < nj:
        idx = np.argsort(-Wt, axis=1)[:, max_influences:]
        np.put_along_axis(Wt, idx, 0.0, axis=1)
    Wt /= Wt.sum(axis=1, keepdims=True)
    if smooth and mesh is not None:
        E = _vertex_neighbors(mesh)
        for _ in range(int(smooth)):
            acc = Wt.copy()
            cnt = np.ones(len(P))
            np.add.at(acc, E[:, 0], Wt[E[:, 1]])
            np.add.at(acc, E[:, 1], Wt[E[:, 0]])
            np.add.at(cnt, E[:, 0], 1)
            np.add.at(cnt, E[:, 1], 1)
            Wt = acc / cnt[:, None]
        if max_influences and max_influences < nj:
            idx = np.argsort(-Wt, axis=1)[:, max_influences:]
            np.put_along_axis(Wt, idx, 0.0, axis=1)
        Wt /= np.maximum(Wt.sum(axis=1, keepdims=True), 1e-12)
    return Wt


def _explicit_weights(spec_w, joints, nverts):
    Wt = np.zeros((nverts, len(joints)))
    for jn, vals in spec_w.items():
        if jn not in joints:
            raise ValueError(f"skin weights: {jn!r} is not one of the skin joints")
        k = joints.index(jn)
        vals = np.asarray(vals, float)
        if vals.ndim == 2:  # [[vertex_index, weight], ...]
            Wt[vals[:, 0].astype(int), k] = vals[:, 1]
        elif len(vals) == nverts:
            Wt[:, k] = vals
        else:
            raise ValueError(f"skin weights for {jn!r}: need {nverts} values or [index, weight] pairs")
    s = Wt.sum(axis=1, keepdims=True)
    zero = s[:, 0] <= 0
    if np.any(zero):
        Wt[zero, 0] = 1.0
        s = Wt.sum(axis=1, keepdims=True)
    return Wt / s


def skin_data(scene, node, mesh):
    """Bind data (cached): joints, weights, bind inverse matrices, bind-space rest verts."""
    key = ("skin", node.name, mesh.num_verts)
    cache = scene._rest_cache
    if key in cache:
        return cache[key]
    sk = _skin_spec(node)
    joints = skin_joints(scene, node)
    if not joints:
        raise ValueError(f"skin on {node.name!r}: no joints found")
    bf = float(sk.get("bind_frame", scene.frame_start))
    st = scene.state(bf)
    Mbind = st.world(node.name)
    rest_obj = scene.object_mesh(node.name, bf)
    rest_w = mu.transform_points(Mbind, rest_obj.verts)
    wspec = sk.get("weights", "auto")
    if sk.get("copy_weights_from"):
        from .weightpaint import copy_weights
        src = scene.nodes[sk["copy_weights_from"]]
        src_rest = scene.object_mesh(src.name, bf)
        sd = skin_data(scene, src, src_rest)
        Wt = copy_weights(sd["weights"], mu.transform_points(sd["mesh_bind"], src_rest.verts),
                          sd["joints"], rest_w, joints)
    elif wspec == "auto" or wspec is None:
        Wt = auto_weights(scene, node, rest_w, joints, bf, int(sk.get("max_influences", 4)),
                          int(sk.get("smooth", 2)), float(sk.get("falloff", 4.0)), rest_obj)
    elif isinstance(wspec, str):
        from .weightpaint import load_weights
        Wt = load_weights(wspec, joints, rest_w, scene.base_dir, sk.get("weights_match", "auto"))
    else:
        Wt = _explicit_weights(wspec, joints, rest_obj.num_verts)
    if sk.get("weight_paint"):
        from .weightpaint import apply_ops, PaintContext
        ctx = PaintContext.from_scene(scene, node, rest_obj, rest_w, joints, bf)
        Wt = apply_ops(Wt, sk["weight_paint"], ctx, sk.get("locked_joints", ()))
    Binv = np.stack([np.linalg.inv(bind_world(scene, st, j)) for j in joints])
    data = {"joints": joints, "weights": Wt, "bind_inv": Binv, "mesh_bind": Mbind,
            "method": sk.get("method", "linear"), "rest_world": rest_w}
    cache[key] = data
    return data


def _mat_to_dq(M):
    q = mu.matrix_to_quat(M[:3, :3])
    t = M[:3, 3]
    w, x, y, z = q
    tq = np.array([0.0, *t])
    d = 0.5 * mu.quat_mul(tq, q)
    return q, d


def skin_mesh(scene, node, mesh, frame):
    """Deform an object-space ``mesh`` by its skin joints at ``frame``; returns world-space Mesh."""
    data = skin_data(scene, node, mesh)
    st = scene.state(frame)
    joints = data["joints"]
    S = np.stack([st.world(j) @ data["bind_inv"][k] for k, j in enumerate(joints)])  # (J,4,4)
    Vw = mu.transform_points(data["mesh_bind"], mesh.verts)  # rest verts (with current blend shapes)
    Wt = data["weights"]
    if data["method"] in ("dual_quaternion", "dq", "dqs"):
        Q = np.zeros((len(joints), 4))
        Dd = np.zeros((len(joints), 4))
        for k in range(len(joints)):
            # remove scale for DQ (rigid part), apply scale linearly afterwards
            q, d = _mat_to_dq(S[k])
            Q[k], Dd[k] = q, d
        # hemisphere alignment relative to the most influential joint per vertex
        piv = Q[np.argmax(Wt, axis=1)]
        sign = np.sign(np.einsum("vj,kj->vk", piv, Q))
        sign[sign == 0] = 1
        Wq = Wt * sign
        b0 = Wq @ Q
        be = Wq @ Dd
        nrm = np.linalg.norm(b0, axis=1, keepdims=True)
        b0, be = b0 / nrm, be / nrm
        w, x, y, z = b0.T
        R = np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
        # translation t = 2 * be * conj(b0)
        conj = b0 * np.array([1, -1, -1, -1])
        tq = 2 * _qmul_batch(be, conj)
        out = np.einsum("vij,vj->vi", R, Vw) + tq[:, 1:]
    else:
        Vh = np.concatenate([Vw, np.ones((len(Vw), 1))], axis=1)
        M = np.einsum("vj,jab->vab", Wt, S[:, :3, :])
        out = np.einsum("vab,vb->va", M, Vh)
    m = mesh.with_verts(out)
    return m


def _qmul_batch(a, b):
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


# ============================================================================================
# blend shapes
# ============================================================================================
def blend_shape_list(node):
    bs = node.spec.get("blend_shapes") or []
    if isinstance(bs, dict):
        bs = [dict(v, name=k) for k, v in bs.items()]
    return bs


def apply_blend_shapes(scene, node, mesh, frame, ctx):
    """Add weighted shape deltas. Weights are node attributes so ``get('node.shape')`` works."""
    from .modifiers import apply_stack
    shapes = blend_shape_list(node)
    out = mesh.verts.copy()
    cache = scene._rest_cache
    for i, s in enumerate(shapes):
        name = s.get("name", f"shape{i}")
        prm = node.attributes.get(name)
        w = float(prm.evaluate(frame, ctx)) if prm is not None else float(Param(s.get("weight", 0)).evaluate(frame, ctx))
        if abs(w) < 1e-6:
            continue
        key = ("blend", node.name, i, mesh.num_verts)
        delta = cache.get(key)
        if delta is None:
            if "modifiers" in s:
                mods = [{k: (Param(v) if k not in ("type", "name", "axis", "direction", "mode", "texture",
                                                    "limits", "offsets", "bounds", "curve") else v)
                         for k, v in m.items()} for m in s["modifiers"]]
                tgt = apply_stack(mesh, mods, frame, ctx, scene.fps)
                if tgt.num_verts != mesh.num_verts:
                    raise ValueError(f"blend shape {name!r}: modifiers changed the vertex count")
                delta = tgt.verts - mesh.verts
                animated = any(isinstance(v, Param) and v.is_animated() for m in mods for v in m.values())
            elif "file" in s:
                from .io.obj import read_mesh
                tm, _ = read_mesh(scene.resolve_path(s["file"]))
                if tm.num_verts != mesh.num_verts:
                    raise ValueError(f"blend shape {name!r}: {s['file']} has {tm.num_verts} verts, "
                                     f"base has {mesh.num_verts}")
                delta = tm.verts - mesh.verts
                animated = False
            elif "verts" in s:
                delta = np.asarray(s["verts"], float).reshape(-1, 3) - mesh.verts
                animated = False
            elif "deltas" in s:
                d = s["deltas"]
                delta = np.zeros_like(mesh.verts)
                if isinstance(d, dict):
                    for k, v in d.items():
                        delta[int(k)] = v
                else:
                    arr = np.asarray(d, float)
                    if arr.ndim == 2 and arr.shape[1] == 4:
                        delta[arr[:, 0].astype(int)] = arr[:, 1:]
                    else:
                        delta = arr.reshape(-1, 3)
                animated = False
            else:
                raise ValueError(f"blend shape {name!r} needs 'modifiers', 'file', 'verts' or 'deltas'")
            if not animated:
                cache[key] = delta
        out += w * delta
    return mesh.with_verts(out)
