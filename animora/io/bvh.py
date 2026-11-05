"""BVH motion-capture import.

:func:`read_bvh` parses a Biovision Hierarchy file (``HIERARCHY`` + ``MOTION``) into joints and
per-frame channel values. :func:`bvh_to_scene` turns it into an Animora scene: one ``joint``
node per BVH joint (parented like the file, offset = BVH ``OFFSET`` x ``scale``), keyed
``rotate`` in the joint's own channel order (mapped to ``rotate_order``), keyed root
``translate``, and optional display geometry (a sphere per joint and a cylinder per bone,
parented to the joints so it follows the motion and renders).

Channel order note: BVH lists rotation channels outermost-first, e.g. ``Zrotation Xrotation
Yrotation`` means ``R = Rz * Rx * Ry`` - Y is applied first. Animora's ``rotate_order`` names
the application order, so that example becomes ``"yxz"`` (the channel letters reversed).
"""
from __future__ import annotations

import math
import os

import numpy as np


class BVHError(ValueError):
    pass


def read_bvh(path):
    """Parse a BVH file.

    Returns ``(joints, frames, frame_time)`` where ``joints`` is a list of dicts
    ``{"name", "parent", "offset", "channels", "end_site"}`` (``end_site`` is the end offset or
    None) in file order and ``frames`` is an ``(n_frames, n_channels)`` array.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        toks = fh.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(toks):
            raise BVHError(f"{path}: unexpected end of file")
        t = toks[pos]
        pos += 1
        return t

    def expect(word):
        t = nxt()
        if t.upper() != word:
            raise BVHError(f"{path}: expected {word!r}, got {t!r}")

    expect("HIERARCHY")
    joints = []
    names = set()

    def parse_joint(parent):
        name = nxt()
        base, k = name, 1
        while name in names:
            k += 1
            name = f"{base}_{k}"
        names.add(name)
        expect("{")
        j = {"name": name, "parent": parent, "offset": [0.0, 0.0, 0.0], "channels": [], "end_site": None}
        joints.append(j)
        while True:
            t = nxt()
            tu = t.upper()
            if tu == "OFFSET":
                j["offset"] = [float(nxt()) for _ in range(3)]
            elif tu == "CHANNELS":
                n = int(nxt())
                j["channels"] = [nxt().lower() for _ in range(n)]
            elif tu == "JOINT":
                parse_joint(name)
            elif tu == "END":
                nxt()  # "Site"
                expect("{")
                while True:
                    t2 = nxt()
                    if t2.upper() == "OFFSET":
                        j["end_site"] = [float(nxt()) for _ in range(3)]
                    elif t2 == "}":
                        break
            elif t == "}":
                return
            else:
                raise BVHError(f"{path}: unexpected token {t!r} in joint {name!r}")

    while True:
        t = nxt()
        if t.upper() in ("ROOT", "JOINT"):
            parse_joint(None)
        elif t.upper() == "MOTION":
            break
        else:
            raise BVHError(f"{path}: unexpected token {t!r} in HIERARCHY")
    if not joints:
        raise BVHError(f"{path}: no joints")
    t = nxt()
    if t.rstrip(":").upper() != "FRAMES":
        raise BVHError(f"{path}: expected 'Frames:'")
    n_frames = int(nxt())
    t = nxt()
    if t.upper() == "FRAME":
        nxt()  # "Time:"
    frame_time = float(nxt())
    n_ch = sum(len(j["channels"]) for j in joints)
    vals = np.asarray([float(v) for v in toks[pos:pos + n_frames * n_ch]], float)
    n_frames = min(n_frames, len(vals) // max(n_ch, 1))
    frames = vals[:n_frames * n_ch].reshape(n_frames, n_ch) if n_ch else np.zeros((n_frames, 0))
    return joints, frames, frame_time


def _rotation_order(channels):
    rot = [c[0] for c in channels if c.endswith("rotation")]
    if len(rot) != 3 or len(set(rot)) != 3:
        return "xyz", rot
    return "".join(reversed(rot)), rot


def _unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a), axis=0))


def _align_y(d):
    """Euler (xyz, degrees) rotating +Y onto direction d."""
    from .. import mathutil as mu
    d = np.asarray(d, float)
    d = d / max(np.linalg.norm(d), 1e-12)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(np.dot(y, d))
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    M = np.eye(4)
    M[:3, :3] = R
    return [float(a) for a in mu.matrix_to_euler(M, "xyz")]


def bvh_to_scene(path, fps=None, scale=0.01, joint_radius=0.03, scene_path=None, start_frame=1,
                 display=True, step=1):
    """Build an Animora scene dict from a BVH file.

    ``scale`` converts BVH units (commonly centimetres) to scene units (metres by default).
    ``fps`` defaults to ``1 / Frame Time``. ``joint_radius`` sizes the display geometry
    (0 disables it, as does ``display=False``). ``step`` keeps every Nth BVH frame.
    """
    joints, frames, frame_time = read_bvh(path)
    file_fps = 1.0 / frame_time if frame_time > 0 else 30.0
    if abs(file_fps - round(file_fps)) < 1e-2:
        file_fps = float(round(file_fps))
    fps = float(fps) if fps else file_fps
    step = max(1, int(step))
    idx = np.arange(0, len(frames), step)
    data = frames[idx] if len(frames) else frames
    # frame number of each kept sample, retimed into the scene fps
    fnums = start_frame + idx * frame_time * fps if len(idx) else np.zeros(0)
    fnums = np.round(fnums, 4)

    nodes, col = [], 0
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    for j in joints:
        ch = j["channels"]
        nch = len(ch)
        cols = data[:, col:col + nch] if len(data) else np.zeros((0, nch))
        col += nch
        order, rot_axes = _rotation_order(ch)
        spec = {"name": j["name"], "type": "joint", "translate": [o * scale for o in j["offset"]],
                "rotate_order": order}
        if j["parent"]:
            spec["parent"] = j["parent"]
        if len(cols):
            # translation channels (root, sometimes every joint): absolute local position
            tpos = {c[0]: k for k, c in enumerate(ch) if c.endswith("position")}
            if tpos:
                T = np.tile(np.asarray(j["offset"], float), (len(cols), 1)) if len(tpos) < 3 else np.zeros((len(cols), 3))
                for a, k in tpos.items():
                    T[:, "xyz".index(a)] = cols[:, k]
                T = T * scale
                if np.ptp(T, axis=0).max() > 1e-9:
                    spec["translate"] = [_vec_keys(fnums, T[:, a]) for a in range(3)]
                else:
                    spec["translate"] = [float(v) for v in T[0]]
            rpos = {c[0]: k for k, c in enumerate(ch) if c.endswith("rotation")}
            if rpos:
                Rd = np.zeros((len(cols), 3))
                for a, k in rpos.items():
                    Rd[:, "xyz".index(a)] = cols[:, k]
                Rd = _unwrap_deg(Rd)
                if np.ptp(Rd, axis=0).max() > 1e-6:
                    spec["rotate"] = [_vec_keys(fnums, Rd[:, a]) for a in range(3)]
                elif np.abs(Rd[0]).max() > 1e-9:
                    spec["rotate"] = [float(v) for v in Rd[0]]
        spec["comment"] = "BVH channels: " + " ".join(ch)
        nodes.append(spec)

    if display and joint_radius > 0:
        r = float(joint_radius)
        for j in joints:
            nodes.append({"name": f"{j['name']}_geo", "type": "mesh", "parent": j["name"],
                          "primitive": "uv_sphere", "radius": r, "segments": 16, "rings": 8, "material": "bone"})
            tips = [(c["name"], np.asarray(c["offset"], float) * scale) for c in children.get(j["name"], [])]
            if j["end_site"] is not None:
                tips.append((f"{j['name']}_end", np.asarray(j["end_site"], float) * scale))
            for tn, off in tips:
                L = float(np.linalg.norm(off))
                if L < 1e-6:
                    continue
                nodes.append({"name": f"{j['name']}_to_{tn}_bone", "type": "mesh", "parent": j["name"],
                              "primitive": "cylinder", "radius": r * 0.55, "height": L, "segments": 12,
                              "translate": [float(v) for v in off * 0.5], "rotate": _align_y(off),
                              "material": "bone"})

    n = len(data)
    end = int(math.ceil(fnums[-1])) if n else start_frame
    scene = {
        "comment": f"Imported from BVH {os.path.basename(str(path))}: {len(joints)} joints, "
                   f"{len(frames)} frames at {file_fps:.3g} fps.",
        "settings": {"fps": fps, "frame_range": [int(start_frame), max(int(start_frame), end)],
                     "resolution": [1280, 720], "samples": 32},
        "environment": {"type": "sky", "sun_elevation": 40, "sun_azimuth": 140, "strength": 1.0},
        "materials": {"bone": {"preset": "plastic", "base_color": "#d9c9a8", "roughness": 0.45}},
        "nodes": nodes,
    }
    return scene


def _vec_keys(fnums, vals):
    """Linear keys for one channel, dropping samples that are collinear with their neighbours."""
    f = np.asarray(fnums, float)
    v = np.asarray(vals, float)
    if len(f) == 1:
        return {"keys": [[float(f[0]), float(v[0]), "linear"]]}
    keep = np.ones(len(f), bool)
    if len(f) > 2:
        # interior sample is redundant when linear interpolation of its neighbours reproduces it
        t = (f[1:-1] - f[:-2]) / np.maximum(f[2:] - f[:-2], 1e-12)
        pred = v[:-2] + (v[2:] - v[:-2]) * t
        keep[1:-1] = np.abs(pred - v[1:-1]) > 1e-6
    return {"keys": [[float(a), round(float(b), 6), "linear"] for a, b in zip(f[keep], v[keep])]}
