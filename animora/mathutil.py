"""Vector, matrix and quaternion helpers (numpy, column-vector convention: p' = M @ p).

Conventions
-----------
* Right-handed, **Y up**, cameras and lights look down their local **-Z** axis (like Maya).
* Angles in the public API are **degrees**; internal helpers marked ``_rad`` use radians.
* Quaternions are ``(w, x, y, z)`` numpy arrays.
* Euler rotation orders follow Maya naming: ``"xyz"`` means rotate X first, then Y, then Z
  (matrix ``Rz @ Ry @ Rx``).
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def vec3(v):
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return np.repeat(a, 3)
    return a[:3].astype(np.float64)


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def length(v):
    return float(np.linalg.norm(v))


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------
# 4x4 matrices
# ------------------------------------------------------------------------------------------
def identity():
    return np.eye(4)


def translate(t):
    m = np.eye(4)
    m[:3, 3] = vec3(t)
    return m


def scale(s):
    s = vec3(s)
    return np.diag([s[0], s[1], s[2], 1.0])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


ROTATE_ORDERS = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")


def euler_matrix(deg, order="xyz"):
    """Rotation matrix from Euler angles in degrees. ``order`` is the application order."""
    deg = vec3(deg)
    order = order.lower()
    if order not in ROTATE_ORDERS:
        raise ValueError(f"bad rotate order {order!r}; use one of {ROTATE_ORDERS}")
    fns = {"x": (rot_x, deg[0]), "y": (rot_y, deg[1]), "z": (rot_z, deg[2])}
    m = np.eye(4)
    for ax in order:
        f, a = fns[ax]
        m = f(a) @ m
    return m


def matrix_to_euler(m, order="xyz"):
    """Inverse of :func:`euler_matrix` (degrees). Works for any rotate order."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    r = r / np.maximum(np.linalg.norm(r, axis=0, keepdims=True), EPS)
    order = order.lower()
    # Map the general case onto the xyz extraction via axis permutation.
    i, j, k = ("xyz".index(order[0]), "xyz".index(order[1]), "xyz".index(order[2]))
    parity = (j - i) % 3 == 1  # even permutation
    # R = Rk @ Rj @ Ri ;  for even permutation:
    #   r[k,i] = -sin(bj) (even)   etc. -> use the standard Shoemake formulation
    sy = math.sqrt(r[i, i] ** 2 + r[j, i] ** 2)
    if sy > 1e-9:
        ax = math.atan2(r[k, j], r[k, k])
        ay = math.atan2(-r[k, i], sy)
        az = math.atan2(r[j, i], r[i, i])
    else:
        ax = math.atan2(-r[j, k], r[j, j])
        ay = math.atan2(-r[k, i], sy)
        az = 0.0
    if not parity:
        ax, ay, az = -ax, -ay, -az
    out = np.zeros(3)
    out[i], out[j], out[k] = math.degrees(ax), math.degrees(ay), math.degrees(az)
    return out


def compose(t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1), order="xyz", pivot=None, shear=None):
    """Maya-style TRS matrix: ``T * Rp * R * S * Rp^-1`` (pivot defaults to origin)."""
    m = euler_matrix(r, order) @ scale(s)
    if pivot is not None:
        p = vec3(pivot)
        m = translate(p) @ m @ translate(-p)
    return translate(t) @ m


def decompose(m):
    """Split an affine matrix into (translate, rotation 3x3, scale)."""
    m = np.asarray(m, dtype=np.float64)
    t = m[:3, 3].copy()
    a = m[:3, :3]
    s = np.linalg.norm(a, axis=0)
    if np.linalg.det(a) < 0:
        s[0] = -s[0]
    r = a / np.where(np.abs(s) < EPS, 1.0, s)
    return t, r, s


def look_at_matrix(eye, target, up=(0, 1, 0)):
    """World matrix of an object at ``eye`` whose -Z axis points at ``target``."""
    eye, target = vec3(eye), vec3(target)
    f = normalize(target - eye)
    u = vec3(up)
    if abs(np.dot(f, normalize(u))) > 0.9999:
        u = np.array([0.0, 0.0, 1.0]) if abs(f[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    x = normalize(np.cross(f, u))
    y = np.cross(x, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, -f, eye
    return m


def aim_rotation(direction, aim_axis=(0, 0, -1), up=(0, 1, 0), up_axis=(0, 1, 0)):
    """3x3 rotation that maps ``aim_axis`` onto ``direction`` keeping ``up_axis`` near ``up``."""
    d = normalize(vec3(direction))
    up = normalize(vec3(up))
    aim_axis, up_axis = normalize(vec3(aim_axis)), normalize(vec3(up_axis))
    side = np.cross(d, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(d, np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 0, 1.0]))
    side = normalize(side)
    up2 = np.cross(side, d)
    target = np.stack([d, up2, np.cross(d, up2)], axis=1)
    lside = np.cross(aim_axis, up_axis)
    lside = normalize(lside)
    lup = np.cross(lside, aim_axis)
    local = np.stack([aim_axis, lup, np.cross(aim_axis, lup)], axis=1)
    return target @ local.T


def transform_points(m, p):
    p = np.asarray(p, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def transform_vectors(m, v):
    return np.asarray(v, dtype=np.float64) @ m[:3, :3].T


def transform_normals(m, n):
    """Transform normals with the inverse-transpose and renormalise."""
    a = m[:3, :3]
    try:
        it = np.linalg.inv(a).T
    except np.linalg.LinAlgError:
        it = a
    return normalize(np.asarray(n, dtype=np.float64) @ it.T)


def inverse(m):
    return np.linalg.inv(m)


# ------------------------------------------------------------------------------------------
# quaternions (w, x, y, z)
# ------------------------------------------------------------------------------------------
def quat_identity():
    return np.array([1.0, 0, 0, 0])


def quat_axis_angle(axis, deg):
    a = normalize(vec3(axis))
    h = math.radians(deg) * 0.5
    return np.concatenate([[math.cos(h)], a * math.sin(h)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    return np.asarray(q, dtype=np.float64) / max(np.linalg.norm(q), EPS)


def quat_to_matrix3(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_matrix(q):
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix3(q)
    return m


def matrix_to_quat(m):
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    r = r / np.maximum(np.linalg.norm(r, axis=0, keepdims=True), EPS)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    return quat_normalize(np.array(q))


def quat_slerp(a, b, t):
    a, b = quat_normalize(a), quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return quat_normalize(a + (b - a) * t)
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / s


def quat_from_to(u, v):
    """Shortest-arc rotation taking direction ``u`` to direction ``v``."""
    u, v = normalize(vec3(u)), normalize(vec3(v))
    d = float(np.dot(u, v))
    if d < -0.999999:
        axis = np.cross([1.0, 0, 0], u)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0, 1.0, 0], u)
        return quat_axis_angle(axis, 180.0)
    c = np.cross(u, v)
    return quat_normalize(np.array([1.0 + d, c[0], c[1], c[2]]))


def rotate_vector(q, v):
    return quat_to_matrix3(q) @ vec3(v)


def rotation_between_matrix(u, v):
    return quat_to_matrix3(quat_from_to(u, v))


def blend_matrices(mats, weights):
    """Weighted blend of rigid transforms (translations linear, rotations via quaternion nlerp)."""
    weights = np.asarray(weights, dtype=np.float64)
    wsum = weights.sum()
    if wsum <= 0:
        return np.eye(4)
    weights = weights / wsum
    t = np.zeros(3)
    s = np.zeros(3)
    q = np.zeros(4)
    q0 = None
    for m, w in zip(mats, weights):
        tt, r, ss = decompose(m)
        qq = matrix_to_quat(r)
        if q0 is None:
            q0 = qq
        elif np.dot(q0, qq) < 0:
            qq = -qq
        t += w * tt
        s += w * ss
        q += w * qq
    out = quat_to_matrix(quat_normalize(q)) @ scale(s)
    out[:3, 3] = t
    return out
