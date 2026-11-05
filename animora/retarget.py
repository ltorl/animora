"""HumanIK-style motion retargeting.

Transfers animation from a *source* skeleton (a BVH file, or a joint hierarchy inside a scene)
onto a *target* joint hierarchy with different proportions, rest pose, joint names and
facing direction - the job Maya's HumanIK "characterize + retarget" does.

Pipeline
--------
1. **Characterization** - every joint name of both skeletons is mapped onto canonical humanoid
   slots (``Hips``, ``Spine``..., ``Neck``, ``Head``, ``LeftShoulder``, ``LeftArm``,
   ``LeftForeArm``, ``LeftHand``, ``LeftUpLeg``, ``LeftLeg``, ``LeftFoot``, ``LeftToeBase`` and
   the right side). The matcher understands HumanIK / Mixamo (``mixamorig:LeftForeArm``),
   Unreal (``thigh_l``, ``calf_l``, ``upperarm_l``, ``spine_02``), Blender/Rigify
   (``upper_arm.L``, ``shin.R``), CMU / ASF BVH (``lhumerus``, ``rfemur``, ``LowerBack``),
   SMPL (``left_hip``, ``left_knee``, ``left_collar``) and generic ``L_``/``_R``/``Left``
   spellings. Spine and neck chains with different joint counts are distributed
   proportionally. Joints that share the same name in both skeletons (e.g. fingers) map
   directly. Any slot can be overridden with an explicit ``map``.
2. **Facing alignment** - the yaw between the two characters is detected from their hips /
   shoulders and removed (or set with ``source_yaw``).
3. **Rest-pose matching** - each mapped bone of the source rest pose is rotated onto the
   target rest-pose bone direction, so a T-pose capture drives an A-pose character correctly.
4. **Rotation transfer** - world-space rotation deltas from the source rest pose are applied to
   the target rest pose and converted back to local Euler rotations in each target joint's own
   ``rotate_order``. Unmapped target joints keep their rest rotation.
5. **Root motion** - hips translation is scaled by the ratio of hip heights (leg length),
   relative to each skeleton's ground level. ``in_place`` removes horizontal travel.
6. **Foot locking** - frames where a source foot is planted (low and slow) are detected, the
   target ankle is pinned to one world position for the whole contact, and the leg is solved
   with two-bone IK - removing the foot sliding that differing proportions cause.

Public API: :func:`canonical_slots`, :func:`auto_map`, :func:`retarget`,
:func:`apply_retarget_to_data` (used for the scene-level ``"retarget"`` block) and the
``animora retarget`` command.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np

from . import mathutil as mu

# ----------------------------------------------------------------------------- name matching
SIDED = {
    "Shoulder": ["shoulder", "clavicle", "collar", "clav", "collarbone"],
    "Arm": ["arm", "upperarm", "humerus", "uparm", "bicep"],
    "ForeArm": ["forearm", "lowerarm", "elbow", "radius", "loarm", "ulna"],
    "Hand": ["hand", "wrist"],
    "UpLeg": ["upleg", "thigh", "femur", "upperleg", "hip"],
    "Leg": ["leg", "shin", "calf", "knee", "lowerleg", "tibia", "loleg", "shank"],
    "Foot": ["foot", "ankle"],
    "ToeBase": ["toebase", "toe", "toes", "ball", "toe0"],
}
HIPS = ["hips", "pelvis", "hip", "cog", "root", "centerofmass", "com"]
SPINE = ["spine", "chest", "back", "lowerback", "upperback", "torso", "abdomen", "upperchest", "waist"]
NECK = ["neck"]
HEAD = ["head"]
CANON_ORDER = (["Hips"] + [f"Spine{i or ''}" for i in range(4)] + ["Neck", "Neck1", "Head"] +
               [f"{s}{k}" for s in ("Left", "Right") for k in SIDED])
NEXT = {"Shoulder": "Arm", "Arm": "ForeArm", "ForeArm": "Hand", "UpLeg": "Leg", "Leg": "Foot",
        "Foot": "ToeBase"}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _strip_ns(name):
    return re.split(r"[:|]", name)[-1]


def _side(name):
    """Return ('Left'|'Right'|None, remaining base name)."""
    raw = _strip_ns(name)
    low = raw.lower()
    for word, side in (("left", "Left"), ("right", "Right")):
        if word in low:
            return side, _norm(low.replace(word, "", 1))
    m = re.match(r"^([lr])[_.\- ]+(.+)$", low)
    if m:
        return ("Left" if m.group(1) == "l" else "Right"), _norm(m.group(2))
    m = re.match(r"^(.+?)[_.\- ]+([lr])$", low)
    if m:
        return ("Left" if m.group(2) == "l" else "Right"), _norm(m.group(1))
    m = re.match(r"^(.+?)[_.\- ]+([lr])[_.\- ]+(.+)$", low)
    if m:
        return ("Left" if m.group(2) == "l" else "Right"), _norm(m.group(1) + m.group(3))
    if len(raw) > 2 and raw[0] in "LR" and raw[1].isupper():     # LHipJoint, RFoot
        return ("Left" if raw[0] == "L" else "Right"), _norm(raw[1:])
    if len(low) > 2 and low[0] in "lr":                          # lhumerus, rfemur (CMU)
        rest = _norm(low[1:])
        if any(rest == a for al in SIDED.values() for a in al):
            return ("Left" if low[0] == "l" else "Right"), rest
    return None, _norm(raw)


def canonical_slots(joint_names, parents=None):
    """Map a skeleton's joint names onto canonical HumanIK slots.

    ``joint_names`` must be in hierarchy order (parents before children). ``parents`` (optional
    dict name -> parent) is used to pick the root as ``Hips`` when no name matches.
    Returns ``{slot: joint_name}``.
    """
    info = []
    for n in joint_names:
        side, base = _side(n)
        info.append((n, side, base, re.sub(r"\d+$", "", base)))
    out = {}
    # side-specific conventions decided per skeleton
    for side in ("Left", "Right"):
        bases = {b for (_, s, b, _) in info if s == side}
        bases_nd = {b for (_, s, _, b) in info if s == side}
        has_arm = bases_nd & {"arm", "upperarm", "humerus", "uparm"}
        has_ankle = "ankle" in bases_nd
        has_thigh = bases_nd & {"upleg", "thigh", "femur", "upperleg"}
        for n, s, base, bnd in info:
            if s != side:
                continue
            slot = None
            for k, al in SIDED.items():
                if base in al or bnd in al:
                    slot = k
                    break
            if slot == "Shoulder" and base in ("shoulder",) and not has_arm and \
                    bases_nd & {"collar", "clavicle", "clav", "collarbone"}:
                slot = "Arm"                     # SMPL: shoulder = upper arm when a collar exists
            if slot == "Foot" and base == "foot" and has_ankle:
                slot = "ToeBase"                 # SMPL: ankle = foot joint, foot = toe joint
            if slot == "UpLeg" and base == "hip" and has_thigh:
                slot = None
            if slot and f"{side}{slot}" not in out:
                out[f"{side}{slot}"] = n
    # centre line
    spine, neck = [], []
    for n, s, base, bnd in info:
        if s is not None:
            continue
        if "Hips" not in out and (base in HIPS or bnd in HIPS):
            out["Hips"] = n
        elif bnd in SPINE or base in SPINE or base.startswith("spine"):
            spine.append(n)
        elif bnd in NECK or base.startswith("neck"):
            neck.append(n)
        elif (base in HEAD or bnd in HEAD) and "Head" not in out:
            out["Head"] = n
    if "Hips" not in out and joint_names:
        roots = [n for n in joint_names if not parents or parents.get(n) not in joint_names]
        out["Hips"] = roots[0] if roots else joint_names[0]
        spine = [n for n in spine if n != out["Hips"]]
    for i, n in enumerate(spine[:8]):
        out[f"Spine{i or ''}"] = n
    for i, n in enumerate(neck[:2]):
        out[f"Neck{i or ''}"] = n
    return out


def _chain(slots, prefix):
    keys = sorted([k for k in slots if re.fullmatch(prefix + r"\d*", k)],
                  key=lambda k: int(k[len(prefix):] or 0))
    return [slots[k] for k in keys]


def auto_map(src_names, tgt_names, src_parents=None, tgt_parents=None, overrides=None):
    """Return ``{target_joint: source_joint}``.

    Canonical slots are matched; spine and neck chains of different lengths are distributed
    proportionally; remaining joints with identical (namespace-stripped) names map directly;
    ``overrides`` ``{target: source}`` (or ``{target: null}`` to unmap) win.
    """
    S = canonical_slots(src_names, src_parents)
    T = canonical_slots(tgt_names, tgt_parents)
    m = {}
    for slot, t in T.items():
        if re.fullmatch(r"(Spine|Neck)\d*", slot):
            continue
        if slot in S:
            m[t] = S[slot]
    for prefix in ("Spine", "Neck"):
        sc, tc = _chain(S, prefix), _chain(T, prefix)
        if sc and tc:
            for i, t in enumerate(tc):
                k = 0 if len(tc) == 1 else int(round(i * (len(sc) - 1) / (len(tc) - 1)))
                m[t] = sc[k]
    used_t = set(m)
    src_by = {_norm(_strip_ns(n)): n for n in src_names}
    for t in tgt_names:
        if t not in used_t:
            s = src_by.get(_norm(_strip_ns(t)))
            if s is not None:
                m[t] = s
    for t, s in (overrides or {}).items():
        if s is None:
            m.pop(t, None)
        else:
            m[t] = s
    return m


# ----------------------------------------------------------------------------- skeletons
def _rot(M):
    R = np.asarray(M, float)[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q


def _joint_order(scene, root=None):
    """Joint nodes (hierarchy order) under ``root`` (or all joints)."""
    out = []

    def rec(n):
        if scene.nodes[n].type == "joint":
            out.append(n)
        for c in scene.children(n):
            rec(c)

    if root is not None:
        if root not in scene.nodes:
            raise ValueError(f"retarget: joint {root!r} not found")
        rec(root)
    else:
        for n in scene.order:
            if scene.nodes[n].parent is None:
                rec(n)
    return out


class _Skel:
    """Sampled skeleton: rest world matrices + per-frame world matrices."""

    def __init__(self, scene, joints, rest_frame, rest_override=None):
        from .rig import bind_world
        self.scene, self.joints = scene, joints
        self.parent = {j: scene.nodes[j].parent for j in joints}
        st = scene.state(rest_frame)
        if rest_override is not None:
            self.rest = rest_override
        else:
            self.rest = {j: bind_world(scene, st, j) for j in joints}

    def world(self, frame):
        st = self.scene.state(frame)
        return {j: st.world(j) for j in self.joints}


def _bvh_skeleton(path, fps, start_frame, scale):
    from .io.bvh import bvh_to_scene, read_bvh
    from .scene import Scene
    bj, _, _ = read_bvh(path)
    data = bvh_to_scene(path, fps=fps, scale=scale, display=False, start_frame=start_frame)
    sc = Scene(data)
    joints = [j["name"] for j in bj]
    # rest pose = the hierarchy offsets with zero rotations (the BVH reference pose)
    rest, off = {}, {j["name"]: np.asarray(j["offset"], float) * scale for j in bj}
    for j in bj:
        L = np.eye(4)
        L[:3, 3] = off[j["name"]]
        rest[j["name"]] = (rest[j["parent"]] @ L) if j["parent"] else L
    fr = sc.settings["frame_range"]
    return _Skel(sc, joints, fr[0], rest), (int(fr[0]), int(fr[1])), float(sc.settings["fps"])


def _yaw_of(skel, slots, mats):
    """Facing direction (horizontal unit vector) from left/right hips or shoulders."""
    for a, b in (("LeftUpLeg", "RightUpLeg"), ("LeftArm", "RightArm"), ("LeftShoulder", "RightShoulder")):
        if a in slots and b in slots:
            lr = mats[slots[a]][:3, 3] - mats[slots[b]][:3, 3]
            f = np.cross(lr, [0.0, 1.0, 0.0])
            f[1] = 0.0
            if np.linalg.norm(f) > 1e-9:
                return f / np.linalg.norm(f)
    return np.array([0.0, 0.0, 1.0])


def _ry(deg):
    return mu.rot_y(deg)[:3, :3]


def _lowest(slots, mats, keys=("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")):
    ys = [mats[slots[k]][1, 3] for k in keys if k in slots]
    return min(ys) if ys else min(m[1, 3] for m in mats.values())


def _hip_height(slots, mats):
    h = mats[slots["Hips"]][1, 3] - _lowest(slots, mats)
    if h > 1e-6:
        return h
    ys = [m[1, 3] for m in mats.values()]
    return max(max(ys) - min(ys), 1e-6) * 0.5


def _two_bone(a, b, c, target):
    from .rig import _solve_two_bone
    P = np.stack([a, b, c])
    pole = b - (a + c) * 0.5
    return _solve_two_bone(P, target, pole, soft=0.0)


# ----------------------------------------------------------------------------- retarget
def retarget(source, target_scene, target_root=None, source_root=None, mapping=None,
             frames=None, fps=None, start_frame=None, bvh_scale=0.01, source_yaw="auto",
             match_rest=True, in_place=False, root_motion=True, height_scale="auto",
             foot_lock=False, lock_height=0.06, lock_speed=0.5, rest_frame=None, report=None):
    """Retarget ``source`` onto the joints of ``target_scene``.

    ``source`` is a BVH path or a :class:`~animora.scene.Scene` holding the source skeleton
    (``source_root`` picks its hierarchy). Returns ``(keys, info)`` where ``keys`` is
    ``{target_joint: {"rotate": [...], "translate": [...]?}}`` ready to drop into node specs and
    ``info`` describes the mapping, frame range and scale.
    """
    from .io.bvh import _vec_keys, _unwrap_deg
    tsc = target_scene
    tfps = float(fps or tsc.settings.get("fps", 24))
    if isinstance(source, str):
        sf = int(start_frame if start_frame is not None else tsc.settings.get("frame_range", [1, 1])[0])
        sskel, frng, _ = _bvh_skeleton(source, tfps, sf, bvh_scale)
    else:
        sj = _joint_order(source, source_root)
        fr = source.settings.get("frame_range", [1, 1])
        sskel = _Skel(source, sj, rest_frame if rest_frame is not None else fr[0])
        frng = (int(fr[0]), int(fr[1]))
    tj = _joint_order(tsc, target_root)
    if not tj:
        raise ValueError("retarget: target has no joint nodes")
    trf = rest_frame if rest_frame is not None else tsc.settings.get("frame_range", [1, 1])[0]
    tskel = _Skel(tsc, tj, trf)
    m = auto_map(sskel.joints, tj, sskel.parent, tskel.parent, mapping)
    if not m:
        raise ValueError("retarget: no joints could be matched; pass an explicit 'map'")
    S_slots = canonical_slots(sskel.joints, sskel.parent)
    T_slots = canonical_slots(tj, tskel.parent)
    if frames is None:
        frames = list(range(frng[0], frng[1] + 1))
    frames = [float(f) for f in frames]

    # ---- facing alignment (rotate the source about +Y)
    if source_yaw == "auto" or source_yaw is None:
        fs, ft = _yaw_of(sskel, S_slots, sskel.rest), _yaw_of(tskel, T_slots, tskel.rest)
        yaw = math.degrees(math.atan2(fs[0] * ft[2] - fs[2] * ft[0], fs[0] * ft[0] + fs[2] * ft[2]))
        yaw = -yaw + 0.0
    else:
        yaw = float(source_yaw)
    Y = _ry(yaw)
    Yh = np.eye(4)
    Yh[:3, :3] = Y
    srest = {j: Yh @ M for j, M in sskel.rest.items()}
    trest = tskel.rest

    # ---- rest-pose matching (per mapped bone: source rest direction -> target rest direction)
    t_child = {}
    inv = {t: slot for slot, t in T_slots.items()}
    for t in tj:
        slot = inv.get(t)
        nxt = None
        if slot:
            for side in ("Left", "Right"):
                if slot.startswith(side) and slot[len(side):] in NEXT:
                    nxt = T_slots.get(side + NEXT[slot[len(side):]])
        if nxt is None:
            kids = [c for c in tsc.children(t) if c in m and c in trest]
            nxt = kids[0] if len(kids) == 1 else None
        t_child[t] = nxt
    A = {}
    for t, s in m.items():
        A[t] = np.eye(3)
        c = t_child.get(t)
        if not match_rest or c is None or c not in m or m[c] not in srest or inv.get(t) == "Hips":
            continue
        ds = srest[m[c]][:3, 3] - srest[s][:3, 3]
        dt = trest[c][:3, 3] - trest[t][:3, 3]
        if np.linalg.norm(ds) > 1e-9 and np.linalg.norm(dt) > 1e-9:
            A[t] = mu.rotation_between_matrix(ds, dt)
    Rs0 = {j: _rot(M) for j, M in srest.items()}
    Rt0 = {j: _rot(M) for j, M in trest.items()}

    # ---- root motion scaling
    s_hips, t_hips = S_slots.get("Hips"), T_slots.get("Hips")
    if height_scale == "auto":
        k = _hip_height(T_slots, trest) / _hip_height(S_slots, srest) if (s_hips and t_hips) else 1.0
    else:
        k = float(height_scale)
    samples = [sskel.world(f) for f in frames]
    samples = [{j: Yh @ M for j, M in w.items()} for w in samples]
    s_ground = min(_lowest(S_slots, w) for w in samples)
    t_ground = _lowest(T_slots, trest)

    # ---- target local rest data
    nodes = tsc.nodes
    st_rest = tsc.state(trf)
    Lrest = {j: st_rest.fk_local(j) for j in tj}
    Lscale = {j: np.linalg.norm(Lrest[j][:3, :3], axis=0) for j in tj}
    top = tj[0]
    for j in tj:
        if nodes[j].parent not in tj:
            top = j
            break
    ext_parent = {}
    for j in tj:
        p = nodes[j].parent
        if p not in tj:
            ext_parent[j] = st_rest.world(p) if p else np.eye(4)
    t_hips_node = t_hips if t_hips in tj else top

    def solve_frame(w, h_world):
        """Global target rotations + world positions for one frame."""
        G, W = {}, {}
        for j in tj:
            p = nodes[j].parent
            Pw = W[p] if p in W else ext_parent.get(j, np.eye(4))
            Pr = _rot(Pw)
            if j in m and m[j] in w:
                s = m[j]
                Gj = _rot(w[s]) @ Rs0[s].T @ A[j].T @ Rt0[j]
            else:
                Gj = Pr @ _rot(Lrest[j])
            G[j] = Gj
            L = Lrest[j].copy()
            L[:3, :3] = (Pr.T @ Gj) * Lscale[j]
            if j == t_hips_node and h_world is not None:
                L[:3, 3] = (np.linalg.inv(Pw) @ np.append(h_world, 1.0))[:3]
            W[j] = Pw @ L
        return G, W

    first_h = samples[0][s_hips][:3, 3] if s_hips in samples[0] else np.zeros(3)
    hip_pos = []
    for w in samples:
        if not (root_motion and s_hips and s_hips in w):
            hip_pos.append(None)
            continue
        ps = w[s_hips][:3, 3]
        ht = trest[t_hips_node][:3, 3].copy()
        ht[1] = t_ground + (ps[1] - s_ground) * k
        if not in_place:
            ht[0] += (ps[0] - first_h[0]) * k
            ht[2] += (ps[2] - first_h[2]) * k
        hip_pos.append(ht)
    results = [solve_frame(w, h) for w, h in zip(samples, hip_pos)]

    # ---- foot locking
    locked = {}
    if foot_lock:
        legs = []
        for side in ("Left", "Right"):
            ks = [side + x for x in ("UpLeg", "Leg", "Foot")]
            if all(x in T_slots and x in S_slots for x in ks):
                legs.append(ks)
        dtf = 1.0 / tfps
        leg_len_s = _hip_height(S_slots, srest)
        for ks in legs:
            sfoot = S_slots[ks[2]]
            sy = np.array([w[sfoot][1, 3] for w in samples])
            sp = np.array([w[sfoot][:3, 3] for w in samples])
            rest_ank = srest[sfoot][1, 3] - _lowest(S_slots, srest)
            vel = np.zeros(len(frames))
            if len(frames) > 1:
                d = np.linalg.norm((sp[1:] - sp[:-1])[:, [0, 2]], axis=1) / dtf
                vel[1:] = d
                vel[0] = d[0]
            contact = ((sy - s_ground) < rest_ank + lock_height * leg_len_s) & (vel < lock_speed * leg_len_s)
            up, kn, ft = (T_slots[x] for x in ks)
            i = 0
            n_lock = 0
            while i < len(frames):
                if not contact[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(frames) and contact[j + 1]:
                    j += 1
                if j - i >= 1:
                    pin = results[i][1][ft][:3, 3].copy()
                    pin[1] = max(pin[1], t_ground + (trest[ft][1, 3] - t_ground))  # no ground sink
                    for f in range(i, j + 1):
                        _lock_leg(results[f], tj, nodes, up, kn, ft, pin, Lrest, Lscale, ext_parent,
                                  t_hips_node)
                    n_lock += 1
                i = j + 1
            locked[ks[0][:-5]] = n_lock

    # ---- local Euler keys
    keys = {}
    for j in tj:
        order = nodes[j].rotate_order
        eul = []
        for G, W in results:
            p = nodes[j].parent
            Pr = _rot(W[p]) if p in W else _rot(ext_parent.get(j, np.eye(4)))
            Rl = Pr.T @ G[j]
            M4 = np.eye(4)
            M4[:3, :3] = Rl
            eul.append(mu.matrix_to_euler(M4, order))
        eul = _unwrap_deg(np.asarray(eul))
        if j not in m and np.ptp(eul, axis=0).max() < 1e-4:
            continue
        ent = {"rotate": [_vec_keys(frames, eul[:, a]) for a in range(3)]}
        if j == t_hips_node and hip_pos[0] is not None:
            p = nodes[j].parent
            T = []
            for G, W in results:
                Pw = W[p] if p in W else ext_parent.get(j, np.eye(4))
                T.append((np.linalg.inv(Pw) @ W[j][:, 3])[:3])
            T = np.asarray(T)
            ent["translate"] = [_vec_keys(frames, T[:, a]) for a in range(3)]
        keys[j] = ent
    info = {"map": m, "source_slots": S_slots, "target_slots": T_slots, "frames": [frames[0], frames[-1]],
            "height_scale": k, "source_yaw": yaw, "foot_locks": locked,
            "unmapped_target": [j for j in tj if j not in m]}
    if report:
        report(info)
    return keys, info


def _lock_leg(res, tj, nodes, up, kn, ft, pin, Lrest, Lscale, ext_parent, hips):
    """Two-bone IK on one solved frame so the ankle reaches ``pin``; keeps the foot's world
    orientation; recomputes world matrices of the leg's descendants."""
    G, W = res
    a, b, c = W[up][:3, 3], W[kn][:3, 3], W[ft][:3, 3]
    P = _two_bone(a, b, c, pin)
    R1 = mu.rotation_between_matrix(b - a, P[1] - a)
    G[up] = R1 @ G[up]
    b1 = a + R1 @ (b - a)
    c1 = a + R1 @ (c - a)
    R2 = mu.rotation_between_matrix(c1 - b1, P[2] - P[1])
    G[kn] = R2 @ R1 @ G[kn]
    # foot keeps its global rotation (G[ft] unchanged); rebuild worlds below the thigh
    sub = set()

    def rec(n):
        sub.add(n)
        for x in tj:
            if nodes[x].parent == n:
                rec(x)

    rec(up)
    for j in tj:
        if j not in sub:
            continue
        p = nodes[j].parent
        Pw = W[p] if p in W else ext_parent.get(j, np.eye(4))
        L = Lrest[j].copy()
        L[:3, :3] = (_rot(Pw).T @ G[j]) * Lscale[j]
        W[j] = Pw @ L


# ----------------------------------------------------------------------------- scene hook
def _walk_specs(nodes):
    if isinstance(nodes, dict):
        nodes = [dict(v, name=k) for k, v in nodes.items()]
    for s in nodes:
        yield s
        yield from _walk_specs(s.get("children", []) or [])


def apply_retarget_to_data(data, base_dir=".", report=None):
    """Resolve a scene's top-level ``"retarget"`` block(s): retarget and write the keys into the
    target joint specs. Returns the new data dict (the input is not modified)."""
    import os
    from .scene import Scene
    blocks = data.get("retarget")
    if not blocks:
        return data
    blocks = blocks if isinstance(blocks, list) else [blocks]
    out = copy.deepcopy(data)
    out.pop("retarget", None)
    if isinstance(out.get("nodes"), dict):
        out["nodes"] = [dict(v, name=k) for k, v in out["nodes"].items()]
    infos = []
    for b in blocks:
        b = dict(b)
        tsc = Scene(out, base_dir=base_dir)
        src = b.get("source")
        if src is None and b.get("source_root"):
            source = tsc
        elif isinstance(src, str):
            source = src if os.path.isabs(src) else os.path.join(base_dir, src)
        else:
            raise ValueError("retarget block needs 'source' (a .bvh path) or 'source_root'")
        keys, info = retarget(
            source, tsc, target_root=b.get("target_root"), source_root=b.get("source_root"),
            mapping=b.get("map"), start_frame=b.get("start_frame"), bvh_scale=float(b.get("scale", 0.01)),
            source_yaw=b.get("source_yaw", "auto"), match_rest=bool(b.get("match_rest", True)),
            in_place=bool(b.get("in_place", False)), root_motion=bool(b.get("root_motion", True)),
            height_scale=b.get("height_scale", "auto"), foot_lock=bool(b.get("foot_lock", False)),
            lock_height=float(b.get("lock_height", 0.06)), lock_speed=float(b.get("lock_speed", 0.5)),
            rest_frame=b.get("rest_frame"))
        for spec in _walk_specs(out["nodes"]):
            if spec.get("name") in keys:
                spec.update(keys[spec["name"]])
        if b.get("set_frame_range", True):
            fr = out.setdefault("settings", {}).get("frame_range")
            lo, hi = int(info["frames"][0]), int(math.ceil(info["frames"][1]))
            if fr is None or b.get("set_frame_range") is True:
                out["settings"]["frame_range"] = [lo, hi]
        infos.append(info)
    if report:
        for i in infos:
            report(i)
    return out
