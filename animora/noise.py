"""Deterministic procedural noise (Perlin gradient noise, fBm, turbulence, curl noise).

All functions accept scalars or numpy arrays (broadcast) and are deterministic for a given
``seed`` so that renders are reproducible across machines and frames.
"""
from __future__ import annotations

import numpy as np

_PERM_CACHE: dict[int, np.ndarray] = {}


def _perm(seed: int) -> np.ndarray:
    seed = int(seed) & 0xFFFFFFFF
    p = _PERM_CACHE.get(seed)
    if p is None:
        rng = np.random.default_rng(seed)
        base = rng.permutation(256).astype(np.int64)
        p = np.concatenate([base, base])
        _PERM_CACHE[seed] = p
    return p


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


_G3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], dtype=np.float64)


def perlin3(x, y=0.0, z=0.0, seed=0):
    """Classic improved Perlin noise in 3D. Returns values in about [-1, 1]."""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64))
    scalar = x.ndim == 0
    x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
    p = _perm(seed)
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    xf, yf, zf = x - xi, y - yi, z - zi
    X, Y, Z = xi.astype(np.int64) & 255, yi.astype(np.int64) & 255, zi.astype(np.int64) & 255
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    def grad(h, dx, dy, dz):
        g = _G3[h & 15]
        return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

    A = p[X] + Y
    AA = p[A & 511] + Z
    AB = p[(A + 1) & 511] + Z
    B = p[(X + 1) & 511] + Y
    BA = p[B & 511] + Z
    BB = p[(B + 1) & 511] + Z
    n000 = grad(p[AA & 511], xf, yf, zf)
    n100 = grad(p[BA & 511], xf - 1, yf, zf)
    n010 = grad(p[AB & 511], xf, yf - 1, zf)
    n110 = grad(p[BB & 511], xf - 1, yf - 1, zf)
    n001 = grad(p[(AA + 1) & 511], xf, yf, zf - 1)
    n101 = grad(p[(BA + 1) & 511], xf - 1, yf, zf - 1)
    n011 = grad(p[(AB + 1) & 511], xf, yf - 1, zf - 1)
    n111 = grad(p[(BB + 1) & 511], xf - 1, yf - 1, zf - 1)
    x1 = n000 + u * (n100 - n000)
    x2 = n010 + u * (n110 - n010)
    x3 = n001 + u * (n101 - n001)
    x4 = n011 + u * (n111 - n011)
    y1 = x1 + v * (x2 - x1)
    y2 = x3 + v * (x4 - x3)
    r = y1 + w * (y2 - y1)
    return float(r[0]) if scalar else r


def noise(x, y=0.0, z=0.0, seed=0):
    """Alias of :func:`perlin3`."""
    return perlin3(x, y, z, seed)


def fbm(x, y=0.0, z=0.0, octaves=4, lacunarity=2.0, gain=0.5, seed=0):
    """Fractal Brownian motion: sum of ``octaves`` noise layers, normalised to about [-1,1]."""
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    x, y, z = (np.asarray(a, np.float64) for a in (x, y, z))
    for o in range(int(octaves)):
        total = total + amp * perlin3(x * freq, y * freq, z * freq, seed + o * 131)
        norm += amp
        amp *= gain
        freq *= lacunarity
    r = total / max(norm, 1e-9)
    return float(r) if np.ndim(r) == 0 else r


def turbulence(x, y=0.0, z=0.0, octaves=4, lacunarity=2.0, gain=0.5, seed=0):
    """Sum of absolute noise values (billowy), in about [0,1]."""
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    x, y, z = (np.asarray(a, np.float64) for a in (x, y, z))
    for o in range(int(octaves)):
        total = total + amp * np.abs(perlin3(x * freq, y * freq, z * freq, seed + o * 131))
        norm += amp
        amp *= gain
        freq *= lacunarity
    r = total / max(norm, 1e-9)
    return float(r) if np.ndim(r) == 0 else r


def noise_vec3(p, seed=0, octaves=1):
    """Vector-valued noise: three decorrelated fBm channels for points ``p`` (...,3)."""
    p = np.asarray(p, np.float64)
    out = np.empty(p.shape, np.float64)
    for c, off in enumerate((0.0, 31.416, 77.77)):
        out[..., c] = fbm(p[..., 0] + off, p[..., 1] - off * 0.5, p[..., 2] + off * 0.25,
                          octaves=octaves, seed=seed + c * 1013)
    return out


def curl_noise(p, seed=0, eps=1e-3, octaves=1):
    """Divergence-free 3D curl noise at points ``p`` (...,3) - great for smoke/particle flow."""
    p = np.asarray(p, np.float64)
    dx = np.array([eps, 0, 0])
    dy = np.array([0, eps, 0])
    dz = np.array([0, 0, eps])
    f = lambda q: noise_vec3(q, seed, octaves)
    dfdx = (f(p + dx) - f(p - dx)) / (2 * eps)
    dfdy = (f(p + dy) - f(p - dy)) / (2 * eps)
    dfdz = (f(p + dz) - f(p - dz)) / (2 * eps)
    cx = dfdy[..., 2] - dfdz[..., 1]
    cy = dfdz[..., 0] - dfdx[..., 2]
    cz = dfdx[..., 1] - dfdy[..., 0]
    return np.stack([cx, cy, cz], axis=-1)


def hash01(*args, seed=0):
    """Deterministic hash of integers/floats to [0,1)."""
    h = (int(seed) * 0x9E3779B1) & 0xFFFFFFFF
    for a in args:
        v = int(np.floor(float(a) * 1000.0)) & 0xFFFFFFFF
        h ^= (v + 0x9E3779B9 + ((h << 6) & 0xFFFFFFFF) + (h >> 2)) & 0xFFFFFFFF
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
    return (h & 0xFFFFFF) / float(0x1000000)


def wiggle(t, freq=1.0, amp=1.0, octaves=2, seed=0):
    """After-Effects style 1D wiggle: smooth random value in [-amp, amp] over time ``t``."""
    return amp * fbm(np.asarray(t, np.float64) * freq, 0.37 * seed + 0.5, 0.0, octaves=octaves, seed=seed)
