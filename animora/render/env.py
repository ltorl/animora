"""World / environment lighting.

The ``environment`` scene block selects how the world surrounding the scene emits light:

``{"type": "sky", ...}`` - physically based daylight (Nishita single-scattering atmosphere)
    ``sun_elevation`` (deg, 35), ``sun_azimuth`` (deg, 0 = sun in -Z / "in front", 90 = +X),
    ``turbidity`` (1 clear ... 10 hazy, 2.5), ``sun_strength`` (irradiance, 4), ``sun_size`` (deg, 0.53),
    ``sun`` (true - add the sun as a sampled distant light), ``ground_color`` ([0.3,0.28,0.25]),
    ``ozone`` (1.0), ``strength`` (sky multiplier, 1), ``resolution`` (512 width).
``{"type": "hdri", "file": "studio.hdr", "strength": 1, "rotation": 0}`` - image based lighting from an
    equirectangular ``.hdr`` / ``.exr`` / ``.pfm`` / LDR image (LDR assumed sRGB).
``{"type": "gradient", "zenith": c, "horizon": c, "ground": c, "sharpness": 1}`` - three-colour dome.
``{"type": "studio", "color": c, "key": 6, "fill": 2, "rim": 4, "floor": c}`` - procedural photo-studio
    dome with soft-box hot spots (great for product turntables).
``{"type": "color", "color": c, "strength": 1}`` - uniform colour (``"color": "#000"`` for black).

Common keys: ``strength`` (multiplier), ``rotation`` (deg around +Y, animatable), ``visible``
(show the environment to camera rays - false renders a ``background`` colour instead while still
lighting the scene), ``background`` (colour used when ``visible`` is false).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..materials import parse_color
from ..params import Param

# Earth atmosphere constants (metres)
R_EARTH = 6360e3
R_ATMOS = 6420e3
H_RAYLEIGH = 8000.0
H_MIE = 1200.0
BETA_R = np.array([5.802e-6, 13.558e-6, 33.1e-6])
BETA_M = 3.996e-6
BETA_OZONE = np.array([0.650e-6, 1.881e-6, 0.085e-6])
MIE_G = 0.76

_CACHE: dict = {}


def sun_direction(elevation_deg, azimuth_deg):
    """Unit vector pointing *towards* the sun. Azimuth 0 is -Z, 90 is +X."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    return np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])


def _ray_sphere_exit(o, d, r):
    """Distance along ``d`` from points ``o`` to the far intersection with a sphere of radius r."""
    b = np.sum(o * d, -1)
    c = np.sum(o * o, -1) - r * r
    disc = np.maximum(b * b - c, 0.0)
    return -b + np.sqrt(disc)


def _optical_depth_to_sun(p, sdir, steps=8):
    t1 = _ray_sphere_exit(p, sdir[None, :], R_ATMOS)
    ds = t1 / steps
    od_r = np.zeros(p.shape[0])
    od_m = np.zeros(p.shape[0])
    blocked = np.zeros(p.shape[0], bool)
    for i in range(steps):
        q = p + sdir[None, :] * ((i + 0.5) * ds)[:, None]
        h = np.linalg.norm(q, axis=1) - R_EARTH
        blocked |= h < 0
        od_r += np.exp(-np.maximum(h, 0) / H_RAYLEIGH) * ds
        od_m += np.exp(-np.maximum(h, 0) / H_MIE) * ds
    return od_r, od_m, blocked


def sun_transmittance(elevation_deg, turbidity=2.5, ozone=1.0):
    """RGB transmittance of sunlight through the atmosphere at the given elevation."""
    sdir = sun_direction(elevation_deg, 0.0)
    p = np.array([[0.0, R_EARTH + 1.0, 0.0]])
    od_r, od_m, blocked = _optical_depth_to_sun(p, sdir, steps=32)
    if blocked[0] and elevation_deg < -1.0:
        return np.zeros(3)
    bm = BETA_M * turbidity
    tau = BETA_R * od_r[0] + bm * 1.1 * od_m[0] + BETA_OZONE * ozone * od_r[0]
    return np.exp(-tau)


def nishita_sky(width=512, sun_elevation=35.0, sun_azimuth=0.0, turbidity=2.5, sun_strength=4.0,
                ground_color=(0.3, 0.28, 0.25), ozone=1.0, view_steps=16, light_steps=8):
    """Render an equirectangular (H = W/2) sky radiance map (no sun disc) in linear RGB.

    Radiance is scaled so that a sun of irradiance ``sun_strength`` at the top of the atmosphere
    produces the matching sky brightness.
    """
    key = (width, round(sun_elevation, 4), round(sun_azimuth, 4), round(turbidity, 4), round(sun_strength, 5),
           tuple(np.round(parse_color(ground_color, working=False), 4)), round(ozone, 3))
    if key in _CACHE:
        return _CACHE[key]
    W, H = int(width), int(width) // 2
    u = (np.arange(W) + 0.5) / W
    v = 1.0 - (np.arange(H) + 0.5) / H
    phi = (u - 0.5) * 2 * np.pi
    theta = (1.0 - v) * np.pi
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    d = np.stack([st * np.sin(phi)[None, :], np.broadcast_to(ct, (H, W)), -st * np.cos(phi)[None, :]], -1)
    dirs = d.reshape(-1, 3)
    sdir = sun_direction(sun_elevation, sun_azimuth)
    bm = BETA_M * turbidity
    img = np.zeros((dirs.shape[0], 3))
    up = dirs[:, 1] > -0.02
    dv = dirs[up].copy()
    dv[:, 1] = np.maximum(dv[:, 1], 0.0)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    o = np.array([0.0, R_EARTH + 1.0, 0.0])
    tmax = _ray_sphere_exit(np.broadcast_to(o, dv.shape), dv, R_ATMOS)
    ds = tmax / view_steps
    mu = dv @ sdir
    phase_r = 3.0 / (16 * np.pi) * (1 + mu * mu)
    g = MIE_G
    phase_m = 3.0 / (8 * np.pi) * ((1 - g * g) * (1 + mu * mu)) / ((2 + g * g) * np.power(1 + g * g - 2 * g * mu, 1.5))
    sum_r = np.zeros((dv.shape[0], 3))
    sum_m = np.zeros((dv.shape[0], 3))
    od_vr = np.zeros(dv.shape[0])
    od_vm = np.zeros(dv.shape[0])
    for i in range(view_steps):
        p = o[None, :] + dv * ((i + 0.5) * ds)[:, None]
        h = np.linalg.norm(p, axis=1) - R_EARTH
        hr = np.exp(-h / H_RAYLEIGH) * ds
        hm = np.exp(-h / H_MIE) * ds
        od_vr += hr
        od_vm += hm
        lr, lm, blocked = _optical_depth_to_sun(p, sdir, light_steps)
        tau = (BETA_R[None] * (od_vr + lr)[:, None] + bm * 1.1 * (od_vm + lm)[:, None]
               + BETA_OZONE[None] * ozone * (od_vr + lr)[:, None])
        att = np.exp(-tau) * (~blocked)[:, None]
        sum_r += att * hr[:, None]
        sum_m += att * hm[:, None]
    L = sun_strength * (sum_r * BETA_R[None] * phase_r[:, None] + sum_m * bm * phase_m[:, None])
    img[up] = L
    # ground: lambertian under sun + sky, faded in below the horizon
    gc = parse_color(ground_color, working=False)
    t_sun = sun_transmittance(sun_elevation, turbidity, ozone)
    horizon = img[up][np.abs(dv[:, 1]) < 0.05].mean(0) if np.any(np.abs(dv[:, 1]) < 0.05) else np.zeros(3)
    ground = gc * (sun_strength * t_sun * max(0.0, sdir[1]) / np.pi + 0.5 * horizon)
    down = ~up
    if np.any(down):
        blend = np.clip(-dirs[down, 1] / 0.08, 0, 1)[:, None]
        img[down] = horizon[None] * (1 - blend) + ground[None] * blend
    out = img.reshape(H, W, 3).astype(np.float32)
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def gradient_dome(width=256, zenith=(0.25, 0.45, 0.9), horizon=(0.8, 0.85, 0.9), ground=(0.25, 0.22, 0.2),
                  sharpness=1.0):
    W, H = int(width), int(width) // 2
    v = 1.0 - (np.arange(H) + 0.5) / H
    y = np.cos((1.0 - v) * np.pi)
    z, hz, gr = parse_color(zenith), parse_color(horizon), parse_color(ground)
    a = np.clip(y, 0, 1) ** (1.0 / max(sharpness, 1e-3))
    b = np.clip(-y * 8, 0, 1)
    col = np.where(y[:, None] >= 0, hz[None] * (1 - a[:, None]) + z[None] * a[:, None],
                   hz[None] * (1 - b[:, None]) + gr[None] * b[:, None])
    return np.ascontiguousarray(np.broadcast_to(col[:, None, :], (H, W, 3)), np.float32)


def studio_dome(width=512, color=(0.6, 0.6, 0.62), key=6.0, fill=2.0, rim=4.0, floor=(0.25, 0.25, 0.25)):
    """Procedural studio: soft grey dome with three soft boxes (key front-left, fill right, rim back)."""
    W, H = int(width), int(width) // 2
    u = (np.arange(W) + 0.5) / W
    v = 1.0 - (np.arange(H) + 0.5) / H
    phi = (u - 0.5) * 2 * np.pi
    theta = (1.0 - v) * np.pi
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    d = np.stack([st * np.sin(phi)[None, :], np.broadcast_to(ct, (H, W)), -st * np.cos(phi)[None, :]], -1)
    base = parse_color(color)
    fl = parse_color(floor)
    y = d[..., 1:2]
    img = np.where(y > 0, base * (0.35 + 0.65 * np.clip(y, 0, 1) ** 0.5), fl * (0.6 + 0.4 * np.clip(-y * 3, 0, 1)))

    def box(direction, size, power):
        dirn = np.asarray(direction, float)
        dirn /= np.linalg.norm(dirn)
        c = np.clip(d @ dirn, -1, 1)
        ang = np.arccos(c)
        return power * np.clip(1.0 - (ang / size) ** 6, 0, 1)[..., None]

    # cameras usually sit on +Z looking at -Z, so "front" lights live on the +Z side
    img = img + box([-0.6, 0.55, 0.6], 0.35, key) + box([0.8, 0.3, 0.45], 0.45, fill) + box([0.1, 0.6, -0.9], 0.25, rim)
    return img.astype(np.float32)


def load_hdri(path, base_dir=".", space="lin_srgb"):
    """Load an HDRI and convert it from ``space`` into the current working space."""
    from ..io.images import read_image
    from .. import color as cm
    p = Path(path)
    if not p.is_absolute():
        p = Path(base_dir) / p
    if not p.exists():
        raise FileNotFoundError(f"HDRI not found: {p}")
    space = cm.canonical(space or "lin_srgb")
    key = ("hdri", str(p.resolve()), p.stat().st_mtime, space, cm.working_space())
    if key not in _CACHE:
        if space == "srgb":
            img, space = read_image(p, linear=False), "lin_srgb"
        else:
            img = read_image(p, linear=True)
        img = cm.convert(img[..., :3], space, cm.working_space()) if space != "raw" else img[..., :3]
        _CACHE[key] = np.ascontiguousarray(img, np.float32)
    return _CACHE[key]


class Environment:
    """Evaluates the ``environment`` block for a given frame and fills a :class:`PackedScene`."""

    def __init__(self, spec=None, base_dir="."):
        if spec is None:
            spec = {"type": "sky"}
        if isinstance(spec, str):
            spec = {"type": "hdri", "file": spec} if Path(spec).suffix else {"type": "color", "color": spec}
        self.spec = dict(spec)
        self.base_dir = base_dir
        self.type = self.spec.get("type", "sky")
        self.params = {}
        for k, v in self.spec.items():
            if k in ("type", "file", "colorspace"):
                continue
            self.params[k] = Param(v)

    def get(self, k, default, frame, ctx):
        p = self.params.get(k)
        if p is None:
            return default
        return p.evaluate(frame, ctx)

    def apply(self, packed, textable, frame=0.0, ctx=None, preview=False):
        g = lambda k, d: self.get(k, d, frame, ctx)
        strength = float(g("strength", 1.0))
        packed.env_strength = strength
        packed.env_rotation = math.radians(float(g("rotation", 0.0)))
        packed.env_visible = 1 if bool(g("visible", True)) else 0
        packed.env_bg = tuple(parse_color(g("background", [0.0, 0.0, 0.0])))
        res = int(g("resolution", 256 if preview else 512))
        t = self.type
        extra_lights = []
        if t in ("none", "black"):
            packed.env_tex = -1
            return extra_lights
        if t == "color":
            packed.env_tex = textable.add_constant(parse_color(g("color", [0.05, 0.05, 0.05])))
            return extra_lights
        if t == "hdri":
            img = load_hdri(self.spec["file"], self.base_dir, self.spec.get("colorspace", "lin_srgb"))
            packed.env_tex = textable.add_image_array(img, wrap="repeat")
            return extra_lights
        if t == "gradient":
            img = gradient_dome(res // 2, g("zenith", [0.25, 0.45, 0.9]), g("horizon", [0.8, 0.85, 0.9]),
                                g("ground", [0.25, 0.22, 0.2]), float(g("sharpness", 1.0)))
            packed.env_tex = textable.add_image_array(img)
            return extra_lights
        if t == "studio":
            img = studio_dome(res, g("color", [0.6, 0.6, 0.62]), float(g("key", 6.0)), float(g("fill", 2.0)),
                              float(g("rim", 4.0)), g("floor", [0.25, 0.25, 0.25]))
            packed.env_tex = textable.add_image_array(img)
            return extra_lights
        if t == "sky":
            el = float(g("sun_elevation", 35.0))
            az = float(g("sun_azimuth", 0.0))
            turb = float(g("turbidity", 2.5))
            ss = float(g("sun_strength", 4.0))
            oz = float(g("ozone", 1.0))
            img = nishita_sky(res, el, az, turb, ss, g("ground_color", [0.3, 0.28, 0.25]), oz,
                              view_steps=10 if preview else 16, light_steps=6 if preview else 8)
            from .. import color as cm
            if cm.working_space() != "lin_srgb":  # the sky model is computed in Rec.709 primaries
                img = np.ascontiguousarray(cm.to_working(img, "lin_srgb"), np.float32)
            packed.env_tex = textable.add_image_array(img)
            if bool(g("sun", True)) and el > -2:
                from . import native as nat
                L = nat.AnLight()
                L.type = 2
                L.visible = 1
                L.cast_shadows = 1
                # the sky texture is rotated by env_rotation; rotate the sun with it
                sd = sun_direction(el, az)
                rot = packed.env_rotation
                c, s = math.cos(rot), math.sin(rot)
                sd = np.array([c * sd[0] + s * sd[2], sd[1], -s * sd[0] + c * sd[2]])
                nat.set_vec(L.direction, -sd)
                L.angle = math.radians(float(g("sun_size", 0.53)) * 0.5)
                col = np.maximum(cm.to_working(sun_transmittance(el, turb, oz), "lin_srgb"), 0.0)
                lum = max(1e-6, float(col.max()))
                nat.set_vec(L.color, col / lum)
                # sky texture is scaled by env strength; keep the sun consistent with it
                L.intensity = ss * lum * strength * float(g("sun_multiplier", 1.0))
                L.blend = 0.0
                extra_lights.append(L)
            return extra_lights
        raise ValueError(f"unknown environment type {t!r} (sky, hdri, gradient, studio, color, none)")
